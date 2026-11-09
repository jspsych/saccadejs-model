"""Training/eval dataset construction: subject grouping, calibration masking,
and the model-input pipeline.

Ported from the training notebook, parameterized by ``TrainConfig``.
"""

import glob
import os

import numpy as np
import tensorflow as tf

import et_util.dataset_utils as dataset_utils
from et_util.dataset_utils import (
    parse_single_eye_tfrecord,
    rescale_coords_map,
    parse_single_eye_tfrecord_with_phase,
    rescale_coords_map_with_phase,
)
from et_util.config import TrainConfig, IMAGE_SHAPE
from et_util.augment import make_augment_fn

AUTOTUNE = tf.data.AUTOTUNE


def default_calibration_grid() -> np.ndarray:
    """The fixed 20-point (5x4) calibration grid, scaled to [0, 1]."""
    cal_points = np.array([
        [5, 5], [5, 27.5], [5, 50], [5, 72.5], [5, 95],
        [35, 5], [35, 27.5], [35, 50], [35, 72.5], [35, 95],
        [65, 5], [65, 27.5], [65, 50], [65, 72.5], [65, 95],
        [95, 5], [95, 27.5], [95, 50], [95, 72.5], [95, 95],
    ], dtype=np.float32)
    return cal_points / 100.0


def _group_subjects(dataset, cfg: TrainConfig):
    """Group the per-image dataset into one fixed-size batch of images per subject.

    Deterministic (no randomness), so the result can be ``.cache()``d -- TFRecords
    get decoded and grouped once, then replayed every epoch.
    """
    max_targets = cfg.max_targets

    grouped = dataset.group_by_window(
        key_func=lambda img, coords, subject_id: subject_id,
        reduce_func=lambda subject_id, ds: ds.batch(batch_size=max_targets),
        window_size=max_targets,
    )
    return grouped.filter(lambda images, coords, subject_ids: tf.shape(images)[0] >= max_targets)


def _add_masks(grouped, cfg: TrainConfig, calibration_points=None):
    """Attach calibration/target masks to each grouped subject batch.

    Kept separate from (and applied *after*) the cached grouping so the random
    calibration selection is freshly sampled every epoch.
    """
    max_targets = cfg.max_targets

    def add_masks_to_batch(images, coords, subject_ids):
        actual_batch_size = tf.shape(images)[0]
        cal_mask = tf.zeros(max_targets, dtype="int8")

        if calibration_points is None:
            n_cal_images = tf.random.uniform(
                shape=[], minval=cfg.min_cal_points, maxval=cfg.max_cal_points, dtype="int32"
            )
            random_indices = tf.random.shuffle(tf.range(actual_batch_size))
            cal_indices = random_indices[:n_cal_images]
            cal_mask = tf.scatter_nd(
                tf.expand_dims(cal_indices, 1),
                tf.ones(n_cal_images, dtype="int8"),
                [max_targets],
            )
        else:
            coords_xpand = tf.expand_dims(coords, axis=1)
            cal_xpand = tf.expand_dims(calibration_points, axis=0)
            matches = tf.reduce_all(tf.equal(coords_xpand, cal_xpand), axis=-1)
            point_matches = tf.reduce_any(matches, axis=1)
            cal_mask = tf.cast(point_matches, dtype="int8")

        target_mask = 1 - cal_mask

        padded_images = tf.pad(
            tf.reshape(images, (-1, *IMAGE_SHAPE)),
            [[0, max_targets - actual_batch_size], [0, 0], [0, 0], [0, 0]],
        )
        padded_coords = tf.pad(coords, [[0, max_targets - actual_batch_size], [0, 0]])

        padded_images = tf.ensure_shape(padded_images, [max_targets, *IMAGE_SHAPE])
        padded_coords = tf.ensure_shape(padded_coords, [max_targets, 2])
        padded_cal_mask = tf.ensure_shape(cal_mask, [max_targets])
        padded_target_mask = tf.ensure_shape(target_mask, [max_targets])
        return (padded_images, padded_coords, padded_cal_mask, padded_target_mask), padded_coords, subject_ids

    return grouped.map(add_masks_to_batch, num_parallel_calls=AUTOTUNE)


def prepare_masked_dataset(dataset, cfg: TrainConfig, calibration_points=None):
    """Group images by subject and attach calibration/target masks.

    If ``calibration_points`` is None, a random number (min..max) of calibration
    images is chosen per subject. Otherwise images whose coords match the given
    points are used as calibration (fixed-grid evaluation).
    """
    return _add_masks(_group_subjects(dataset, cfg), cfg, calibration_points)


def prepare_model_inputs(features, labels, subject_ids):
    """Restructure (images, coords, cal_mask, target_mask) into the model's input dict.

    The target mask is returned as the third element -- used as per-sample loss weights.
    """
    images, coords, cal_mask, target_mask = features
    inputs = {
        "Input_All_Images": images,
        "Input_All_Coords": coords,
        "Input_Calibration_Mask": cal_mask,
    }
    return inputs, labels, target_mask


def _load_rescaled(cfg: TrainConfig, tfrecord_dir: str):
    train_data, _, _ = dataset_utils.process_tfr_to_tfds(
        tfrecord_dir,
        parse_single_eye_tfrecord,
        train_split=1.0,
        val_split=0.0,
        test_split=0.0,
        random_seed=cfg.random_seed,
        group_function=lambda img, coords, subject_id: subject_id,
    )
    return train_data.map(rescale_coords_map)


def _resolve_dir(cfg: TrainConfig, tfrecord_dir):
    d = tfrecord_dir or cfg.tfrecord_dir
    return d if d.endswith("/") else d + "/"


def _count_subjects(tfrecord_dir: str) -> int:
    """Subject count = number of .tfrecords files (one file per subject).

    Cheap (no decoding); used only to size steps_per_epoch / the LR schedule.
    Slightly approximate (ignores subjects later dropped by the <144-image filter),
    which is harmless for the schedule.
    """
    return len(glob.glob(os.path.join(tfrecord_dir, "**", "*.tfrecords"), recursive=True))


def build_train_dataset(cfg: TrainConfig, tfrecord_dir=None):
    """Return (dataset, n_subjects) for ``model.fit(ds, steps_per_epoch=n//batch)``.

    Pipeline speedups (training is input-bound, GPU otherwise idle):
      - ``.cache()`` the decoded, grouped images -> TFRecords parsed once, not every
        epoch (masking + augmentation still run fresh each epoch, after the cache);
      - ``.repeat()`` so a fixed ``steps_per_epoch`` bounds the epoch -- removes the
        "ran out of data" warning and the old full-iteration counting pass;
      - subject count from the file count instead of iterating the dataset.
    Augmentation runs on CPU (train.py hides the GPU from TF), overlapping with
    JAX training on the GPU.
    """
    tfrecord_dir = _resolve_dir(cfg, tfrecord_dir)
    rescaled = _load_rescaled(cfg, tfrecord_dir)

    grouped = _group_subjects(rescaled, cfg).cache()          # decode + group once
    ds = _add_masks(grouped, cfg)                             # fresh random masks per epoch
    ds = ds.map(prepare_model_inputs, num_parallel_calls=AUTOTUNE).shuffle(200)
    if cfg.augmentation:
        ds = ds.map(make_augment_fn(cfg), num_parallel_calls=AUTOTUNE)
    ds = ds.repeat().prefetch(AUTOTUNE)

    return ds, _count_subjects(tfrecord_dir)


def build_eval_dataset(cfg: TrainConfig, tfrecord_dir=None, calibration_points=None):
    """Eval dataset using the fixed calibration grid (no augmentation).

    Within-session eval on the *training* dataset (no phase field): both the
    calibration and target points come from the same session. This is the
    optimistic case -- prefer ``build_two_phase_eval_dataset`` for the real number.
    """
    if calibration_points is None:
        calibration_points = default_calibration_grid()
    rescaled = _load_rescaled(cfg, _resolve_dir(cfg, tfrecord_dir))
    masked = prepare_masked_dataset(rescaled, cfg, calibration_points=calibration_points)
    return masked.map(prepare_model_inputs, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)


# --- Two-phase evaluation -------------------------------------------------
# The scientifically meaningful eval: a held-out set (OSF ``uf2sh``) whose
# records carry a ``phase`` field (288 points/subject, 144 per phase, sharing
# the same 20-point grid). Calibrate on one phase's grid points; score the
# *other* phase's non-grid points -> measures temporal generalization.

def _load_rescaled_with_phase(cfg: TrainConfig, tfrecord_dir: str):
    """Load the phase-tagged eval records as a flat per-image dataset (coords in [0,1])."""
    test_data, _, _ = dataset_utils.process_tfr_to_tfds(
        tfrecord_dir,
        parse_single_eye_tfrecord_with_phase,
        train_split=1.0,
        val_split=0.0,
        test_split=0.0,
        random_seed=cfg.random_seed,
        group_function=lambda img, phase, coords, subject_id: subject_id,
    )
    return test_data.map(rescale_coords_map_with_phase)


def _group_subjects_phase(dataset, max_targets: int):
    """Group the phase-tagged per-image dataset into one batch of <=max_targets per subject."""
    grouped = dataset.group_by_window(
        key_func=lambda img, phase, coords, subject_id: subject_id,
        reduce_func=lambda subject_id, ds: ds.batch(batch_size=max_targets),
        window_size=max_targets,
    )
    return grouped


def _add_phase_masks(grouped, cfg: TrainConfig, calibration_points, cal_phase: int):
    """Attach phase-gated calibration/target masks.

    ``target_mask`` = non-grid points in **phase 2** (the held-out targets).
    ``cal_mask``    = grid points in the chosen calibration phase. cal_phase=1
    calibrates from phase 1 (cross-phase test); cal_phase=2 from phase 2
    (within-phase sanity check). Matches the original analysis notebook exactly.
    """
    max_targets = cfg.eval_max_targets
    calibration_points = tf.constant(calibration_points, dtype=tf.float32)

    def add_masks_to_batch(images, phase, coords, subject_ids):
        actual_batch_size = tf.shape(images)[0]

        phase1_mask = tf.cast(tf.equal(phase, 1), dtype="int8")
        phase2_mask = tf.cast(tf.equal(phase, 2), dtype="int8")

        # grid match (across both phases), then gate by phase below
        coords_xpand = tf.expand_dims(coords, axis=1)
        cal_xpand = tf.expand_dims(calibration_points, axis=0)
        matches = tf.reduce_all(tf.equal(coords_xpand, cal_xpand), axis=-1)
        point_matches = tf.reduce_any(matches, axis=1)
        cal_mask = tf.cast(point_matches, dtype="int8")

        # targets are the non-grid points of phase 2 (computed before gating cal_mask)
        target_mask = (1 - cal_mask) * phase2_mask
        if cal_phase == 1:
            cal_mask = cal_mask * phase1_mask
        elif cal_phase == 2:
            cal_mask = cal_mask * phase2_mask

        padded_images = tf.pad(
            tf.reshape(images, (-1, *IMAGE_SHAPE)),
            [[0, max_targets - actual_batch_size], [0, 0], [0, 0], [0, 0]],
        )
        padded_coords = tf.pad(coords, [[0, max_targets - actual_batch_size], [0, 0]])
        padded_cal_mask = tf.pad(cal_mask, [[0, max_targets - actual_batch_size]])
        padded_target_mask = tf.pad(target_mask, [[0, max_targets - actual_batch_size]])

        padded_images = tf.ensure_shape(padded_images, [max_targets, *IMAGE_SHAPE])
        padded_coords = tf.ensure_shape(padded_coords, [max_targets, 2])
        padded_cal_mask = tf.ensure_shape(padded_cal_mask, [max_targets])
        padded_target_mask = tf.ensure_shape(padded_target_mask, [max_targets])
        return (padded_images, padded_coords, padded_cal_mask, padded_target_mask), padded_coords, subject_ids

    return grouped.map(add_masks_to_batch, num_parallel_calls=AUTOTUNE)


def build_two_phase_eval_dataset(cfg: TrainConfig, cal_phase: int,
                                 tfrecord_dir=None, calibration_points=None):
    """Two-phase eval dataset: calibrate on ``cal_phase``'s grid, score phase-2 targets.

    Returns elements ``(inputs, labels, target_mask)`` where ``target_mask`` marks
    the held-out phase-2 non-grid points to score (calibration points are masked out).
    No augmentation. ``cal_phase`` in {1, 2}.
    """
    if calibration_points is None:
        calibration_points = default_calibration_grid()
    d = tfrecord_dir or cfg.eval_tfrecord_dir
    d = d if d.endswith("/") else d + "/"
    rescaled = _load_rescaled_with_phase(cfg, d)
    grouped = _group_subjects_phase(rescaled, cfg.eval_max_targets)
    masked = _add_phase_masks(grouped, cfg, calibration_points, cal_phase)
    return masked.map(prepare_model_inputs, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
