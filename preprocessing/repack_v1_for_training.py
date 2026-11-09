"""Repack the v1 all-frames records into model-ready TRAINING/EVAL sets.

The multi-frame TRAINING experiment (2026-06-11) needs the v1 frames in a
compact, trainable form -- the raw v1 records carry jpeg payloads/landmarks
that make per-epoch reads impractical (~70 GB). This writes:

TRAIN (--train, from eyedata23): one example per TRIAL with a small stack of
candidate frames the input pipeline samples from per epoch:
    subject_id  int64   int(subject, 36) -- legacy convention
    x, y        float   gaze label 0-100 (v1 labels, post swap-fix)
    n_cand      int64   1..8
    candidates  bytes   serialized uint8 tensor (n_cand, 36, 144):
                        [0] = the FIRST face-found frame (the legacy frame /
                              baseline arm), [1:] = up to 7 face-found frames
                        evenly spaced in the settling window
                        3 <= frame_index <= n_frames-4 (the prediction-cluster
                        analysis showed frames 0-2 are gaze-in-transit and the
                        trial tail drifts -- those would inject label error)
    -> ~10 GB total, et_util/frames_data.py consumes it

EVAL (--eval, from eyedata24): one example per trial, FIRST face-found frame,
in the LEGACY single-eye-with-phase schema (landmarks/img dims/x/y/eye_img/
subject_id/phase) so the existing two-phase eval pipeline reads it unchanged
-- just point cfg.eval_tfrecord_dir at it. This keeps the 20-ep screen's eval
inside the v1 extraction (RGB-correct; old-record numbers don't compare).

Trials with no face-found frame are dropped (legacy behavior). Subjects then
falling below 144 usable trials get filtered by the standard pipeline.

Run on the HPC (sharded, like the other v1 jobs):
    PYTHONPATH=. .venv/bin/python preprocessing/repack_v1_for_training.py \
        --train --src /work/jdeleeuw/frames/v1 \
        --out /work/jdeleeuw/train_repack --shard $I/$N
"""

import argparse
import os
import sys
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

from et_util.pose import pose_features

WINDOW_START = 3          # drop gaze-in-transit head of the trial
WINDOW_END_TRIM = 3       # drop anticipatory-drift tail
MAX_EXTRA_CANDIDATES = 7  # candidates[1:]; [0] is always the legacy frame

_FEATURES = None  # built lazily (tf import)


def _features():
    global _FEATURES
    import tensorflow as tf
    if _FEATURES is None:
        _FEATURES = {
            "trial": tf.io.FixedLenFeature([], tf.string),
            "phase": tf.io.FixedLenFeature([], tf.int64),
            "x": tf.io.FixedLenFeature([], tf.float32),
            "y": tf.io.FixedLenFeature([], tf.float32),
            "frame_index": tf.io.FixedLenFeature([], tf.int64),
            "n_frames": tf.io.FixedLenFeature([], tf.int64),
            "face_found": tf.io.FixedLenFeature([], tf.int64),
            "eye_img": tf.io.FixedLenFeature([], tf.string),
            "landmarks": tf.io.FixedLenFeature([], tf.string),
        }
    return _FEATURES


def _bytes(v):
    import tensorflow as tf
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))


def _int64(v):
    import tensorflow as tf
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(v)]))


def _float(v):
    import tensorflow as tf
    return tf.train.Feature(float_list=tf.train.FloatList(value=[float(v)]))


def load_trials(tfrecord_path):
    """Parse one v1 subject record -> {trial: [frame dicts sorted by index]}."""
    import tensorflow as tf
    trials = defaultdict(list)
    ds = tf.data.TFRecordDataset(str(tfrecord_path)).map(
        lambda el: tf.io.parse_single_example(el, _features()),
        num_parallel_calls=tf.data.AUTOTUNE)
    for c in ds.as_numpy_iterator():
        if c["face_found"] != 1:
            continue
        img = tf.io.parse_tensor(c["eye_img"], out_type=tf.uint8).numpy()
        if img.shape != (36, 144):
            continue
        lm = tf.io.parse_tensor(c["landmarks"], out_type=tf.float32).numpy()
        trials[c["trial"].decode()].append({
            "img": img, "x": float(c["x"]), "y": float(c["y"]),
            "phase": int(c["phase"]), "idx": int(c["frame_index"]),
            "n_frames": int(c["n_frames"]), "landmarks": c["landmarks"],
            "pose": pose_features(lm).astype(np.float32),
        })
    for t in trials.values():
        t.sort(key=lambda f: f["idx"])
    return trials


def pick_candidates(frames):
    """[legacy frame] + up to 7 evenly spaced face-found frames in the window."""
    first = frames[0]
    lo, hi = WINDOW_START, frames[0]["n_frames"] - 1 - WINDOW_END_TRIM
    pool = [f for f in frames if lo <= f["idx"] <= hi and f is not first]
    if not pool:  # short trial: fall back to whatever else exists
        pool = [f for f in frames if f is not first]
    if len(pool) > MAX_EXTRA_CANDIDATES:
        sel = np.round(np.linspace(0, len(pool) - 1,
                                   MAX_EXTRA_CANDIDATES)).astype(int)
        pool = [pool[i] for i in sel]
    return [first] + pool


def repack_train(rec_path, out_dir):
    import tensorflow as tf
    trials = load_trials(rec_path)
    sid = int(rec_path.stem, 36)
    out = out_dir / f"{rec_path.stem}.tfrecords"
    tmp = str(out) + ".tmp"
    n = 0
    with tf.io.TFRecordWriter(tmp) as w:
        for frames in trials.values():
            cands = pick_candidates(frames)
            stack = np.stack([f["img"] for f in cands])  # (n_cand, 36, 144)
            pstack = np.stack([f["pose"] for f in cands])  # (n_cand, 9)
            ex = tf.train.Example(features=tf.train.Features(feature={
                "subject_id": _int64(sid),
                "x": _float(frames[0]["x"]),
                "y": _float(frames[0]["y"]),
                "n_cand": _int64(len(cands)),
                "candidates": _bytes(tf.io.serialize_tensor(stack).numpy()),
                "cand_pose": _bytes(tf.io.serialize_tensor(pstack).numpy()),
            }))
            w.write(ex.SerializeToString())
            n += 1
    os.replace(tmp, out)
    return n


def repack_eval(rec_path, out_dir):
    import tensorflow as tf
    trials = load_trials(rec_path)
    sid = int(rec_path.stem, 36)
    out = out_dir / f"{rec_path.stem}.tfrecords"
    tmp = str(out) + ".tmp"
    n = 0
    with tf.io.TFRecordWriter(tmp) as w:
        for frames in trials.values():
            f0 = frames[0]  # first face-found frame = the legacy frame
            ex = tf.train.Example(features=tf.train.Features(feature={
                "landmarks": _bytes(f0["landmarks"]),
                "img_width": _int64(144),
                "img_height": _int64(36),
                "x": _float(f0["x"]),
                "y": _float(f0["y"]),
                "eye_img": _bytes(tf.io.serialize_tensor(f0["img"]).numpy()),
                "subject_id": _int64(sid),
                "phase": _int64(f0["phase"]),
                "pose": _bytes(tf.io.serialize_tensor(f0["pose"]).numpy()),
            }))
            w.write(ex.SerializeToString())
            n += 1
    os.replace(tmp, out)
    return n


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", action="store_true",
                      help="eyedata23 -> per-trial candidate records")
    mode.add_argument("--eval", action="store_true",
                      help="eyedata24 -> legacy-schema frame-0 records w/ phase")
    ap.add_argument("--src", default="/work/jdeleeuw/frames/v1")
    ap.add_argument("--out", default="/work/jdeleeuw/train_repack")
    ap.add_argument("--shard", metavar="I/N")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    dataset = "eyedata23" if args.train else "eyedata24"
    # pose-bearing records (added 06-11 for the pose-token model) go to new
    # dirs; the original pose-less sets stay untouched
    sub = "v1_train_frames_pose" if args.train else "v1_eval_frame0_pose"
    out_dir = Path(args.out) / sub
    out_dir.mkdir(parents=True, exist_ok=True)

    records = sorted(Path(args.src, dataset).glob("*.tfrecord"))
    if args.shard:
        i, nshard = (int(v) for v in args.shard.split("/"))
        records = [r for r in records if zlib.crc32(r.stem.encode()) % nshard == i]
    if args.limit:
        records = records[:args.limit]
    records = [r for r in records
               if not (out_dir / f"{r.stem}.tfrecords").exists()]
    print(f"{dataset}: {len(records)} subjects to repack -> {out_dir}")

    fn = repack_train if args.train else repack_eval
    low = 0
    for i, rec in enumerate(records, 1):
        try:
            n = fn(rec, out_dir)
        except Exception as e:  # one bad subject must not kill the shard
            print(f"[{i}/{len(records)}] {rec.stem}: ERROR {e}", file=sys.stderr)
            continue
        low += args.train and n < 144
        print(f"[{i}/{len(records)}] {rec.stem}: {n} trials", flush=True)
    if args.train:
        print(f"done ({low} subjects below 144 usable trials -- the standard "
              f"pipeline will drop them)")


if __name__ == "__main__":
    main()
