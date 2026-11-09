"""Training dataset over the v1 per-trial candidate records (multi-frame arm).

Consumes the records written by preprocessing/repack_v1_for_training.py
--train: one example per trial holding a stack of candidate frames --
candidates[0] is the legacy first-face-found frame, candidates[1:] are up to 7
real frames from the trial's settling window (frame_index 3 .. n-4).

``frame_mode`` selects the experiment arm:
    "first"   always candidates[0] -- the single-frame BASELINE, equivalent to
              the legacy pipeline but from the v1 (RGB-correct) extraction
    "random"  a fresh uniform draw from candidates[1:] every epoch (falls back
              to [0] when a short trial has no window candidates) -- the
              MULTI-FRAME arm: real nuisance variation under a fixed label

Pipeline cost discipline (the first version starved the TPU at ~25 s/step):
decode + subject-grouping happen ONCE on padded fixed-shape candidate stacks
and are CACHED, exactly like the legacy pipeline caches its grouped images.
The per-epoch frame draw is a single tf.gather over the cached (144, 8, ...)
stack -- cheap -- followed by the usual fresh masking/augmentation. The two
arms differ ONLY in the gather index.

Usage in a training run (instead of data.build_train_dataset):
    train_ds, n_subjects = frames_data.build_train_dataset_v1(
        cfg, "v1_train_frames/", frame_mode="random")
"""

import glob
import os

import tensorflow as tf

from et_util.config import TrainConfig
from et_util.data import _add_masks, prepare_model_inputs
from et_util.augment import make_augment_fn

AUTOTUNE = tf.data.AUTOTUNE
K_MAX = 8  # repack writes at most 1 legacy + 7 window candidates per trial


def _parse_trial(element):
    """-> (candidates (K_MAX, 36, 144, 1) uint8 zero-padded, n_cand, coords, sid)."""
    content = tf.io.parse_single_example(element, {
        "subject_id": tf.io.FixedLenFeature([], tf.int64),
        "x": tf.io.FixedLenFeature([], tf.float32),
        "y": tf.io.FixedLenFeature([], tf.float32),
        "n_cand": tf.io.FixedLenFeature([], tf.int64),
        "candidates": tf.io.FixedLenFeature([], tf.string),
    })
    cands = tf.io.parse_tensor(content["candidates"], out_type=tf.uint8)
    cands = tf.ensure_shape(cands, [None, 36, 144])[..., tf.newaxis]
    n_cand = tf.minimum(tf.cast(content["n_cand"], tf.int32), K_MAX)
    cands = tf.pad(cands[:K_MAX], [[0, K_MAX - n_cand], [0, 0], [0, 0], [0, 0]])
    cands = tf.ensure_shape(cands, [K_MAX, 36, 144, 1])
    coords = tf.stack([content["x"], content["y"]]) / 100.0
    return cands, n_cand, coords, content["subject_id"]


def _group_subjects_cands(ds, max_targets):
    """Group per-trial candidate tuples into one fixed batch per subject."""
    grouped = ds.group_by_window(
        key_func=lambda cands, n_cand, coords, sid: sid,
        reduce_func=lambda sid, d: d.batch(batch_size=max_targets),
        window_size=max_targets,
    )
    return grouped.filter(
        lambda cands, n_cand, coords, sids: tf.shape(cands)[0] >= max_targets)


def _make_pick_fn(frame_mode):
    """Per-SUBJECT map: gather one frame per trial from the cached stacks."""
    if frame_mode not in ("first", "random"):
        raise ValueError(f"frame_mode must be 'first' or 'random', got {frame_mode!r}")

    def pick(cands, n_cand, coords, sids):
        n_trials = tf.shape(cands)[0]
        if frame_mode == "first":
            idx = tf.zeros([n_trials], tf.int32)
        else:
            # uniform over window candidates [1, n_cand); trials with no
            # window candidates (n_cand == 1) fall back to the legacy frame
            u = tf.random.uniform([n_trials], dtype=tf.float32)
            span = tf.cast(tf.maximum(n_cand - 1, 1), tf.float32)
            idx = 1 + tf.cast(u * span, tf.int32)          # in [1, n_cand)
            idx = tf.where(n_cand > 1, idx, tf.zeros_like(idx))
        images = tf.gather(cands, idx, batch_dims=1)        # (n, 36, 144, 1)
        return images, coords, sids

    return pick


def build_train_dataset_v1(cfg: TrainConfig, tfrecord_dir, frame_mode="random",
                           cache_file=""):
    """(dataset, n_subjects) -- drop-in for data.build_train_dataset.

    cache_file: "" caches the grouped candidate stacks in memory (~10 GB for
    the full eyedata23 set); a path caches on disk instead (Colab disk is
    plenty); None disables caching (slow -- re-decodes every epoch).
    """
    files = sorted(glob.glob(os.path.join(tfrecord_dir, "**", "*.tfrecords"),
                             recursive=True))
    if not files:
        raise FileNotFoundError(f"no .tfrecords under {tfrecord_dir!r}")
    raw = tf.data.TFRecordDataset(files, num_parallel_reads=AUTOTUNE)
    trials = raw.map(_parse_trial, num_parallel_calls=AUTOTUNE)

    grouped = _group_subjects_cands(trials, cfg.max_targets)
    if cache_file is not None:
        grouped = grouped.cache(cache_file)      # decode + group ONCE

    ds = grouped.map(_make_pick_fn(frame_mode), num_parallel_calls=AUTOTUNE)
    masked = _add_masks(ds, cfg)                 # fresh masks per epoch
    out = masked.map(prepare_model_inputs, num_parallel_calls=AUTOTUNE).shuffle(200)
    if cfg.augmentation:
        out = out.map(make_augment_fn(cfg), num_parallel_calls=AUTOTUNE)
    return out.repeat().prefetch(AUTOTUNE), len(files)


# ---------------------------------------------------------------------------
# Pose-token variants (2026-06-11 maximalist bet): consume the *_pose repack
# records, which add per-candidate pose vectors (train) / per-trial pose
# (eval). Pose is standardized (et_util/pose.py constants) and reduced to the
# 7 verified features at parse time. Affine augmentation must be OFF for
# these (warps would desynchronize pose from pixels).
# ---------------------------------------------------------------------------
from et_util.pose import POSE_MEAN, POSE_STD, GOOD_POSE_IDX
from et_util.config import IMAGE_SHAPE

_PMEAN = tf.constant(POSE_MEAN[GOOD_POSE_IDX])
_PSTD = tf.constant(POSE_STD[GOOD_POSE_IDX])
_PIDX = tf.constant(GOOD_POSE_IDX, tf.int32)


def _std_pose(p):
    return (tf.gather(p, _PIDX, axis=-1) - _PMEAN) / _PSTD


def _parse_trial_pose(element):
    content = tf.io.parse_single_example(element, {
        "subject_id": tf.io.FixedLenFeature([], tf.int64),
        "x": tf.io.FixedLenFeature([], tf.float32),
        "y": tf.io.FixedLenFeature([], tf.float32),
        "n_cand": tf.io.FixedLenFeature([], tf.int64),
        "candidates": tf.io.FixedLenFeature([], tf.string),
        "cand_pose": tf.io.FixedLenFeature([], tf.string),
    })
    cands = tf.io.parse_tensor(content["candidates"], out_type=tf.uint8)
    cands = tf.ensure_shape(cands, [None, 36, 144])[..., tf.newaxis]
    pose = tf.io.parse_tensor(content["cand_pose"], out_type=tf.float32)
    pose = _std_pose(tf.ensure_shape(pose, [None, 9]))
    n_cand = tf.minimum(tf.cast(content["n_cand"], tf.int32), K_MAX)
    pad = [[0, K_MAX - n_cand]]
    cands = tf.ensure_shape(
        tf.pad(cands[:K_MAX], pad + [[0, 0], [0, 0], [0, 0]]),
        [K_MAX, 36, 144, 1])
    pose = tf.ensure_shape(tf.pad(pose[:K_MAX], pad + [[0, 0]]),
                           [K_MAX, len(GOOD_POSE_IDX)])
    coords = tf.stack([content["x"], content["y"]]) / 100.0
    return cands, pose, n_cand, coords, content["subject_id"]


def _group_subjects_cands_pose(ds, max_targets):
    grouped = ds.group_by_window(
        key_func=lambda c, p, n, co, sid: sid,
        reduce_func=lambda sid, d: d.batch(batch_size=max_targets),
        window_size=max_targets,
    )
    return grouped.filter(
        lambda c, p, n, co, sids: tf.shape(c)[0] >= max_targets)


def _make_pick_fn_pose(frame_mode):
    if frame_mode not in ("first", "random"):
        raise ValueError(frame_mode)

    def pick(cands, pose, n_cand, coords, sids):
        n_trials = tf.shape(cands)[0]
        if frame_mode == "first":
            idx = tf.zeros([n_trials], tf.int32)
        else:
            u = tf.random.uniform([n_trials], dtype=tf.float32)
            span = tf.cast(tf.maximum(n_cand - 1, 1), tf.float32)
            idx = 1 + tf.cast(u * span, tf.int32)
            idx = tf.where(n_cand > 1, idx, tf.zeros_like(idx))
        images = tf.gather(cands, idx, batch_dims=1)
        psel = tf.gather(pose, idx, batch_dims=1)
        return images, psel, coords, sids

    return pick


def _add_masks_pose(grouped, cfg):
    """data._add_masks with the pose sequence carried through (same padding)."""
    max_targets = cfg.max_targets
    pdim = len(GOOD_POSE_IDX)

    def add_masks(images, pose, coords, subject_ids):
        actual = tf.shape(images)[0]
        n_cal = tf.random.uniform([], cfg.min_cal_points, cfg.max_cal_points, "int32")
        cal_idx = tf.random.shuffle(tf.range(actual))[:n_cal]
        cal_mask = tf.scatter_nd(tf.expand_dims(cal_idx, 1),
                                 tf.ones(n_cal, "int8"), [max_targets])
        target_mask = 1 - cal_mask
        pad_n = max_targets - actual
        images = tf.ensure_shape(
            tf.pad(tf.reshape(images, (-1, *IMAGE_SHAPE)),
                   [[0, pad_n], [0, 0], [0, 0], [0, 0]]),
            [max_targets, *IMAGE_SHAPE])
        pose = tf.ensure_shape(tf.pad(pose, [[0, pad_n], [0, 0]]),
                               [max_targets, pdim])
        coords = tf.ensure_shape(tf.pad(coords, [[0, pad_n], [0, 0]]),
                                 [max_targets, 2])
        inputs = {"Input_All_Images": images, "Input_All_Coords": coords,
                  "Input_Calibration_Mask": tf.ensure_shape(cal_mask, [max_targets]),
                  "Input_All_Pose": pose}
        return inputs, coords, tf.ensure_shape(target_mask, [max_targets])

    return grouped.map(add_masks, num_parallel_calls=AUTOTUNE)


def build_train_dataset_v1_pose(cfg, tfrecord_dir, frame_mode="random",
                                cache_file=""):
    """(dataset, n_subjects) over the pose-bearing candidate records."""
    files = sorted(glob.glob(os.path.join(tfrecord_dir, "**", "*.tfrecords"),
                             recursive=True))
    if not files:
        raise FileNotFoundError(f"no .tfrecords under {tfrecord_dir!r}")
    raw = tf.data.TFRecordDataset(files, num_parallel_reads=AUTOTUNE)
    trials = raw.map(_parse_trial_pose, num_parallel_calls=AUTOTUNE)
    grouped = _group_subjects_cands_pose(trials, cfg.max_targets)
    if cache_file is not None:
        grouped = grouped.cache(cache_file)
    ds = grouped.map(_make_pick_fn_pose(frame_mode), num_parallel_calls=AUTOTUNE)
    out = _add_masks_pose(ds, cfg).shuffle(200)
    if cfg.augmentation:
        out = out.map(make_augment_fn(cfg), num_parallel_calls=AUTOTUNE)
    return out.repeat().prefetch(AUTOTUNE), len(files)


# --- two-phase eval over the pose-bearing frame-0 records -------------------
def _parse_eval_pose(element):
    content = tf.io.parse_single_example(element, {
        "x": tf.io.FixedLenFeature([], tf.float32),
        "y": tf.io.FixedLenFeature([], tf.float32),
        "eye_img": tf.io.FixedLenFeature([], tf.string),
        "subject_id": tf.io.FixedLenFeature([], tf.int64),
        "phase": tf.io.FixedLenFeature([], tf.int64),
        "pose": tf.io.FixedLenFeature([], tf.string),
    })
    img = tf.io.parse_tensor(content["eye_img"], out_type=tf.uint8)
    img = tf.ensure_shape(img, (36, 144))
    pose = _std_pose(tf.ensure_shape(
        tf.io.parse_tensor(content["pose"], out_type=tf.float32), [9]))
    coords = tf.stack([content["x"], content["y"]]) / 100.0
    return img, pose, content["phase"], coords, content["subject_id"]


def build_two_phase_eval_dataset_pose(cfg, cal_phase, tfrecord_dir=None,
                                      calibration_points=None):
    """Pose-aware twin of data.build_two_phase_eval_dataset (same protocol)."""
    from et_util.data import default_calibration_grid
    if calibration_points is None:
        calibration_points = default_calibration_grid()
    cal_points = tf.constant(calibration_points, tf.float32)
    d = tfrecord_dir or cfg.eval_tfrecord_dir
    files = sorted(glob.glob(os.path.join(d, "**", "*.tfrecords"), recursive=True))
    raw = tf.data.TFRecordDataset(files)
    ds = raw.map(_parse_eval_pose, num_parallel_calls=AUTOTUNE)
    max_targets = cfg.eval_max_targets
    pdim = len(GOOD_POSE_IDX)

    grouped = ds.group_by_window(
        key_func=lambda img, pose, phase, coords, sid: sid,
        reduce_func=lambda sid, dd: dd.batch(batch_size=max_targets),
        window_size=max_targets,
    )

    def add_masks(images, pose, phase, coords, subject_ids):
        actual = tf.shape(images)[0]
        phase1 = tf.cast(tf.equal(phase, 1), "int8")
        phase2 = tf.cast(tf.equal(phase, 2), "int8")
        matches = tf.reduce_all(tf.equal(
            tf.expand_dims(coords, 1), tf.expand_dims(cal_points, 0)), axis=-1)
        cal_mask = tf.cast(tf.reduce_any(matches, axis=1), "int8")
        target_mask = (1 - cal_mask) * phase2
        cal_mask = cal_mask * (phase1 if cal_phase == 1 else phase2)
        pad_n = max_targets - actual
        images = tf.ensure_shape(
            tf.pad(tf.reshape(images, (-1, *IMAGE_SHAPE)),
                   [[0, pad_n], [0, 0], [0, 0], [0, 0]]),
            [max_targets, *IMAGE_SHAPE])
        pose = tf.ensure_shape(tf.pad(pose, [[0, pad_n], [0, 0]]),
                               [max_targets, pdim])
        coords = tf.ensure_shape(tf.pad(coords, [[0, pad_n], [0, 0]]),
                                 [max_targets, 2])
        inputs = {"Input_All_Images": images, "Input_All_Coords": coords,
                  "Input_Calibration_Mask": tf.ensure_shape(
                      tf.pad(cal_mask, [[0, pad_n]]), [max_targets]),
                  "Input_All_Pose": pose}
        return inputs, coords, tf.ensure_shape(
            tf.pad(target_mask, [[0, pad_n]]), [max_targets])

    return grouped.map(add_masks, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
