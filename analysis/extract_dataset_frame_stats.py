"""Per-frame recording conditions for every released participant, for the paper's dataset figure.

The paper's section 2 describes the dataset with the same measures that sections 4 and 6 relate
to accuracy. extract_v1_frame_stats.py computes some of them for eyedata24 only; this computes
all of them, the same way, for both collections, from the v1 all-frames TFRecords (schema in
preprocessing/extract_frames_to_gcs.py), every face-found frame. One npz per subject:

    trial              (N,) unicode  source webm filename
    frame_index        (N,) int16    0-based index within the trial webm
    pos_msec           (N,) float32  the decoder's timestamp of the frame
    sharpness          (N,) float32  variance of the Laplacian of the native-resolution gray crop
    brightness         (N,) float32  mean of the native-resolution gray crop
    contrast           (N,) float32  its SD
    brightness_left    (N,) float32  mean gray of the participant's left eye: the crop's right
    brightness_right   (N,) float32  half in the unmirrored image, and their right eye
    noise              (N,) float32  mean absolute change of the gray crop, resized to 144 x 36,
                                     from the trial's previous frame; NaN when that frame is not
                                     the one just before
    ear_left, ear_right (N,) float32 each eye's aspect ratio
    pose9              (N, 9) float32 et_util.pose.pose_features (POSE_NAMES)
    yaw, pitch         (N,) float32  degrees, et_util.head_pose; pitch positive tipped up
    crop_width         (N,) float32  the eye crop's size in camera pixels
    crop_height        (N,) float32
    camera_width       ()   float32  the camera image's size, from the crops: the median over
    camera_height      ()   float32  frames

Run on an emc node of the HPC (CPU only), as a SLURM array over --shard:

    PYTHONPATH=. .venv/bin/python analysis/extract_dataset_frame_stats.py \\
        --frames-dir /work/jdeleeuw/frames/v1 --dataset eyedata23 \\
        --include participants.csv --out /work/jdeleeuw/frame_stats/dataset/eyedata23

--include is the dataset summary's participants.csv: only released participants are processed.
Idempotent: subjects whose npz already exists are skipped (--overwrite to force).
"""

import argparse
import csv
import os
import zlib
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import numpy as np
import tensorflow as tf

from et_util.head_pose import head_angles, image_size_from_crop
from et_util.pose import pose_features

_FEATURES = {
    "trial": tf.io.FixedLenFeature([], tf.string),
    "frame_index": tf.io.FixedLenFeature([], tf.int64),
    "pos_msec": tf.io.FixedLenFeature([], tf.float32),
    "face_found": tf.io.FixedLenFeature([], tf.int64),
    "landmarks": tf.io.FixedLenFeature([], tf.string),
    "eye_hires_jpeg": tf.io.FixedLenFeature([], tf.string),
    "eye_hires_width": tf.io.FixedLenFeature([], tf.int64),
    "eye_hires_height": tf.io.FixedLenFeature([], tf.int64),
    "sharpness": tf.io.FixedLenFeature([], tf.float32),
    "brightness_mean": tf.io.FixedLenFeature([], tf.float32),
    "brightness_std": tf.io.FixedLenFeature([], tf.float32),
    "ear_left": tf.io.FixedLenFeature([], tf.float32),
    "ear_right": tf.io.FixedLenFeature([], tf.float32),
}

# The size the crop is compared at from frame to frame, as the model's input.
NOISE_SIZE = (144, 36)

_SCALARS = ["pos_msec", "sharpness", "brightness", "contrast", "brightness_left",
            "brightness_right", "ear_left", "ear_right", "crop_width", "crop_height"]


def frame_rows(tfrecord_path):
    """Every face-found frame's statistics, and its landmarks for the pose."""
    ds = (tf.data.TFRecordDataset(str(tfrecord_path))
          .map(lambda el: tf.io.parse_single_example(el, _FEATURES))
          .filter(lambda c: tf.equal(c["face_found"], 1)))
    for c in ds.as_numpy_iterator():
        crop = cv2.imdecode(np.frombuffer(c["eye_hires_jpeg"], np.uint8), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        half = gray.shape[1] // 2
        yield {
            "trial": c["trial"].decode(),
            "frame_index": int(c["frame_index"]),
            "pos_msec": float(c["pos_msec"]),
            "sharpness": float(c["sharpness"]),
            "brightness": float(c["brightness_mean"]),
            "contrast": float(c["brightness_std"]),
            # Unmirrored: the participant's left eye is on the image's right.
            "brightness_left": float(gray[:, half:].mean()),
            "brightness_right": float(gray[:, :half].mean()),
            "ear_left": float(c["ear_left"]),
            "ear_right": float(c["ear_right"]),
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

    out = {k: np.array([r[k] for r in rows], np.float32) for k in _SCALARS}
    out.update(
        trial=np.array([r["trial"] for r in rows]),
        frame_index=np.array([r["frame_index"] for r in rows], np.int16),
        noise=noise,
        pose9=np.stack([pose_features(r["landmarks"]) for r in rows]).astype(np.float32),
        yaw=angles[:, 0].astype(np.float32),
        pitch=angles[:, 1].astype(np.float32),
        camera_width=np.float32(width),
        camera_height=np.float32(height),
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True, help="v1 root (contains eyedata{23,24}/)")
    ap.add_argument("--dataset", required=True, choices=["eyedata23", "eyedata24"])
    ap.add_argument("--include", required=True, help="the dataset summary's participants.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, help="first N subjects (smoke test)")
    ap.add_argument("--shard", metavar="I/N",
                    help="process only subjects where crc32(subject_id) %% N == I")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    with open(args.include) as f:
        released = {r["participant"] for r in csv.DictReader(f) if r["collection"] == args.dataset}
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = sorted(r for r in Path(args.frames_dir, args.dataset).glob("*.tfrecord")
                     if r.stem in released)
    missing = len(released) - len(records)
    if args.shard:
        shard_i, shard_n = (int(x) for x in args.shard.split("/"))
        records = [r for r in records if zlib.crc32(r.stem.encode()) % shard_n == shard_i]
    if args.limit:
        records = records[:args.limit]
    print(f"{len(records)} subjects ({missing} released subjects have no record)", flush=True)

    for i, rec in enumerate(records, 1):
        dst = out_dir / f"{rec.stem}.npz"
        if dst.exists() and not args.overwrite:
            continue
        stats = process_subject(rec)
        if stats is None:
            print(f"[{i}/{len(records)}] {rec.stem}: NO face-found frames, skipped")
            continue
        np.savez_compressed(dst, **stats)
        print(f"[{i}/{len(records)}] {rec.stem}: {len(stats['trial'])} frames", flush=True)

    print("done")


if __name__ == "__main__":
    main()
