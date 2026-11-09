"""Per-frame image and head-pose statistics from the v1 all-frames TFRecords, for the paper.

The paper's validation study (jspsych/eyetracking-cookoff-exp) relates accuracy to the eye
image's contrast, the difference in brightness between the eyes, the camera's sensor noise
and the head's yaw and pitch. extract_v1_embeddings.py keeps only sharpness, brightness, the
eye aspect ratios and the landmark pose proxies, so this computes the rest from the same
records (schema in preprocessing/extract_frames_to_gcs.py), every face-found frame, and
writes one npz per subject:

    trial            (N,) unicode  source webm filename, as in the embeddings' npz
    frame_index      (N,) int16    0-based index within the trial webm
    contrast         (N,) float32  SD of the native-resolution gray eye crop (the frame
                                   extraction's brightness_std)
    brightness_left  (N,) float32  mean gray of the participant's left eye: the crop's
    brightness_right (N,) float32  right half in the unmirrored image, and their right eye
    noise            (N,) float32  mean absolute change of the gray crop, resized to 144 x 36,
                                   from the trial's previous frame; NaN when that frame is not
                                   the one just before (the first, or after a no-face frame)
    yaw, pitch       (N,) float32  degrees, et_util.head_pose; pitch positive tipped up
    crop_width       (N,) float32  the eye crop's size in camera pixels (eye_hires_width
    crop_height      (N,) float32  and _height)
    camera_width     ()   float32  the camera image's size, from the crops (see
    camera_height    ()   float32  et_util.head_pose.image_size_from_crop): the median over
                                   frames

Then analysis/build_paper_eval_dataset.py --frame-stats joins them to the embeddings.

Run on an emc node of the HPC, like extract_v1_embeddings.py (CPU only, no model):

    PYTHONPATH=. KERAS_BACKEND=tensorflow .venv/bin/python \\
        analysis/extract_v1_frame_stats.py \\
        --frames-dir /work/jdeleeuw/frames/v1 --dataset eyedata24 \\
        --out /work/jdeleeuw/frame_stats/v1_eyedata24

Idempotent: subjects whose npz already exists are skipped (--overwrite to force). --shard
I/N partitions subjects as the other v1 scripts do, for a SLURM array.
"""

import argparse
import os
import zlib
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import numpy as np
import tensorflow as tf

from et_util.head_pose import head_angles, image_size_from_crop

_FEATURES = {
    "trial": tf.io.FixedLenFeature([], tf.string),
    "frame_index": tf.io.FixedLenFeature([], tf.int64),
    "face_found": tf.io.FixedLenFeature([], tf.int64),
    "landmarks": tf.io.FixedLenFeature([], tf.string),
    "eye_hires_jpeg": tf.io.FixedLenFeature([], tf.string),
    "eye_hires_width": tf.io.FixedLenFeature([], tf.int64),
    "eye_hires_height": tf.io.FixedLenFeature([], tf.int64),
    "brightness_std": tf.io.FixedLenFeature([], tf.float32),
}

# The size the crop is compared at from frame to frame, as the model's input.
NOISE_SIZE = (144, 36)


def frame_rows(tfrecord_path):
    """Every face-found frame's statistics, and its landmarks for the pose fit."""
    ds = (tf.data.TFRecordDataset(str(tfrecord_path))
          .map(lambda el: tf.io.parse_single_example(el, _FEATURES),
               num_parallel_calls=tf.data.AUTOTUNE)
          .filter(lambda c: tf.equal(c["face_found"], 1))
          .prefetch(tf.data.AUTOTUNE))
    for c in ds.as_numpy_iterator():
        crop = cv2.imdecode(np.frombuffer(c["eye_hires_jpeg"], np.uint8), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        half = gray.shape[1] // 2
        yield {
            "trial": c["trial"].decode(),
            "frame_index": int(c["frame_index"]),
            "contrast": float(c["brightness_std"]),
            # Unmirrored: the participant's left eye is on the image's right.
            "brightness_left": float(gray[:, half:].mean()),
            "brightness_right": float(gray[:, :half].mean()),
            "patch": cv2.resize(gray, NOISE_SIZE, interpolation=cv2.INTER_AREA),
            "crop_width": float(c["eye_hires_width"]),
            "crop_height": float(c["eye_hires_height"]),
            "landmarks": tf.io.parse_tensor(c["landmarks"], out_type=tf.float32).numpy(),
        }


def process_subject(tfrecord_path):
    rows = list(frame_rows(tfrecord_path))
    if not rows:
        return None
    rows.sort(key=lambda r: (r["trial"], r["frame_index"]))

    # Frame-to-frame noise, against the trial's previous frame only if it is the one just before.
    noise = np.full(len(rows), np.nan, np.float32)
    for i in range(1, len(rows)):
        a, b = rows[i - 1], rows[i]
        if a["trial"] == b["trial"] and b["frame_index"] == a["frame_index"] + 1:
            noise[i] = np.abs(b["patch"] - a["patch"]).mean()

    # The camera image's size, the median over frames, then each frame's pose in it.
    sizes = np.array([image_size_from_crop(r["landmarks"], r["crop_width"], r["crop_height"])
                      for r in rows])
    width, height = np.median(sizes, axis=0)
    angles = np.array([head_angles(r["landmarks"], width, height) for r in rows])

    column = lambda k: np.array([r[k] for r in rows], np.float32)
    return {
        "trial": np.array([r["trial"] for r in rows]),
        "frame_index": np.array([r["frame_index"] for r in rows], np.int16),
        "contrast": column("contrast"),
        "brightness_left": column("brightness_left"),
        "brightness_right": column("brightness_right"),
        "noise": noise,
        "yaw": angles[:, 0].astype(np.float32),
        "pitch": angles[:, 1].astype(np.float32),
        "crop_width": column("crop_width"),
        "crop_height": column("crop_height"),
        "camera_width": np.float32(width),
        "camera_height": np.float32(height),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True, help="v1 root (contains eyedata{23,24}/)")
    ap.add_argument("--dataset", default="eyedata24")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, help="first N subjects (smoke test)")
    ap.add_argument("--shard", metavar="I/N",
                    help="process only subjects where crc32(subject_id) %% N == I")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = sorted(Path(args.frames_dir, args.dataset).glob("*.tfrecord"))
    if args.shard:
        shard_i, shard_n = (int(x) for x in args.shard.split("/"))
        records = [r for r in records if zlib.crc32(r.stem.encode()) % shard_n == shard_i]
    if args.limit:
        records = records[:args.limit]
    print(f"{len(records)} subjects")

    for i, rec in enumerate(records, 1):
        dst = out_dir / f"{rec.stem}.npz"
        if dst.exists() and not args.overwrite:
            continue
        stats = process_subject(rec)
        if stats is None:
            print(f"[{i}/{len(records)}] {rec.stem}: NO face-found frames, skipped")
            continue
        np.savez_compressed(dst, **stats)
        print(f"[{i}/{len(records)}] {rec.stem}: {len(stats['trial'])} frames, camera "
              f"{stats['camera_width']:.0f} x {stats['camera_height']:.0f}", flush=True)

    print("done")


if __name__ == "__main__":
    main()
