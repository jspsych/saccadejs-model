"""Extract frozen per-frame embeddings from the v1 all-frames TFRecords.

Feeds every face-found frame of each subject's v1 record (the all-frames
extraction in gs://webcam-jspsych-frames / /work/jdeleeuw/frames/v1, schema in
preprocessing/extract_frames_to_gcs.py) through a trained checkpoint's
Eye_Image_Embedding submodel and writes one npz per subject:

    E            (N, D) float32   frozen embeddings (tanh output)
    w            (N,)   float32   calibration weight (the Calibration_Weights
                                  head on E; saved only when it is a pointwise
                                  Dense, as in every exported model)
    x, y         (N,)   float32   gaze label, 0-100 screen pct (as stored)
    phase        (N,)   int16     1|2 (eyedata24), 0 (eyedata23)
    frame_index  (N,)   int16     0-based index within the trial webm
    n_frames     (N,)   int16     total decoded frames in the trial
    trial        (N,)   unicode   source webm filename (unique trial key)
    pos_msec, sharpness, brightness_mean, ear_left, ear_right  (N,) float32
    pose9        (N, 9) float32   head pose from the stored landmarks
                                  (et_util.pose.POSE_NAMES); with --pose-token
                                  or --save-pose

This is the shared artifact for the Tier-1 frame-level experiments
(multi-frame calibration, frame-index sweep, target TTA, QA gating): extract
once per checkpoint, then every oracle is pure numpy. Only face_found frames
are kept -- no-face frames have no crop to embed (recover gaps from
frame_index/n_frames if needed).

Run on an emc node of the HPC (data is on /work; CPU-only TF is fine):

    PYTHONPATH=. KERAS_BACKEND=tensorflow .venv/bin/python \
        analysis/extract_v1_embeddings.py \
        --frames-dir /work/jdeleeuw/frames/v1 --dataset eyedata24 \
        --weights hybrid_8lsofasl.weights.h5 \
        --backbone hybrid --embedding-dim 128 --token-pool-extra 1 \
        --out /work/jdeleeuw/embeddings/v1_eyedata24_hybrid_8lsofasl

The deployed model (eye-embedding 1.0.0, W&B kwppnkuj) for the paper's model
performance section:

    ... --weights multiframe_random_s42_ext70.weights.h5 --save-pose \
        --out /work/jdeleeuw/embeddings/v1_eyedata24_kwppnkuj

Idempotent: subjects whose npz already exists are skipped (--overwrite to
force). NOTE: comparisons must stay WITHIN the v1 extraction -- the RGB fix
(2026-06-10) means v1 records are not comparable to the published BGR-era
tfrecords.
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf

from et_util.config import TrainConfig
from et_util.models import create_masked_model
from et_util.pose import GOOD_POSE_IDX, POSE_MEAN, POSE_STD, pose_features

# Per-frame schema written by preprocessing/extract_frames_to_gcs.py
_FEATURES = {
    "trial": tf.io.FixedLenFeature([], tf.string),
    "phase": tf.io.FixedLenFeature([], tf.int64),
    "x": tf.io.FixedLenFeature([], tf.float32),
    "y": tf.io.FixedLenFeature([], tf.float32),
    "frame_index": tf.io.FixedLenFeature([], tf.int64),
    "n_frames": tf.io.FixedLenFeature([], tf.int64),
    "pos_msec": tf.io.FixedLenFeature([], tf.float32),
    "face_found": tf.io.FixedLenFeature([], tf.int64),
    "eye_img": tf.io.FixedLenFeature([], tf.string),
    "sharpness": tf.io.FixedLenFeature([], tf.float32),
    "brightness_mean": tf.io.FixedLenFeature([], tf.float32),
    "ear_left": tf.io.FixedLenFeature([], tf.float32),
    "ear_right": tf.io.FixedLenFeature([], tf.float32),
}

_META_KEYS = ["x", "y", "phase", "frame_index", "n_frames", "pos_msec",
              "sharpness", "brightness_mean", "ear_left", "ear_right"]


def build_embedding_model(args):
    """Load the full masked model, return its inner per-image embedding model
    and its calibration-weight head (None unless a pointwise Dense)."""
    cfg = TrainConfig(backbone=args.backbone, embedding_dim=args.embedding_dim,
                      token_pool_extra=args.token_pool_extra,
                      pose_token=args.pose_token)
    model = create_masked_model(cfg)
    model.load_weights(args.weights)
    # SimpleTimeDistributed is a keras Wrapper (.layer = Eye_Image_Embedding);
    # the pose-token variant uses PairTimeDistributed (.inner, takes
    # [image, pose7]). Images are raw 0-255 (Rescaling(1/255) is inside).
    wrapper = model.get_layer("Image_Embeddings")
    # The weight head scores one embedding at a time only when it is a Dense;
    # a set-attention head needs the whole calibration set (see
    # web/export/export_onnx.build_export_model).
    import keras
    cal = model.get_layer("Calibration_Weights")
    cal = cal if isinstance(cal, keras.layers.Dense) else None
    return (wrapper.inner if args.pose_token else wrapper.layer), cal


def _decode_frame(c):
    img = tf.io.parse_tensor(c["eye_img"], out_type=tf.uint8)
    img = tf.ensure_shape(img, (36, 144))
    return {**c, "eye_img": img}


def _decode_frame_pose(c):
    c = _decode_frame(c)
    lm = tf.io.parse_tensor(c["landmarks"], out_type=tf.float32)
    return {**c, "landmarks": tf.ensure_shape(lm, (478, 3))}


def process_subject(emb_model, tfrecord_path, batch_size, pose=False,
                    cal=None, save_pose=False):
    """Stream one subject's record through the embedding model.

    Decode/parse runs in parallel tf.data threads, overlapped with inference
    (the serial per-frame Python loop this replaces was ~10x slower).
    With pose=True, per-frame head-pose features are computed from the stored
    landmarks and fed alongside the image (standardized exactly as in
    frames_data._std_pose); the raw 9-D features are kept in the npz as pose9.
    With save_pose=True they are computed and kept but not fed to the model.
    With cal, each frame's calibration weight is kept as w.
    Returns (E (N,D) float32, meta dict) or (None, None) if no usable frames.
    """
    features = dict(_FEATURES)
    need_landmarks = pose or save_pose
    if need_landmarks:
        features["landmarks"] = tf.io.FixedLenFeature([], tf.string)
    ds = (tf.data.TFRecordDataset(str(tfrecord_path))
          .map(lambda el: tf.io.parse_single_example(el, features),
               num_parallel_calls=tf.data.AUTOTUNE)
          .filter(lambda c: tf.equal(c["face_found"], 1))
          .map(_decode_frame_pose if need_landmarks else _decode_frame,
               num_parallel_calls=tf.data.AUTOTUNE)
          .batch(batch_size)
          .prefetch(tf.data.AUTOTUNE))

    E_chunks, w_chunks, trials, pose9 = [], [], [], []
    meta = {k: [] for k in _META_KEYS}
    for batch in ds:
        x = tf.cast(batch["eye_img"][..., None], tf.float32)
        if need_landmarks:
            p9 = np.stack([pose_features(lm)
                           for lm in batch["landmarks"].numpy()])
            pose9.append(p9)
        if pose:
            p7 = ((p9[:, GOOD_POSE_IDX] - POSE_MEAN[GOOD_POSE_IDX])
                  / POSE_STD[GOOD_POSE_IDX]).astype(np.float32)
            E = emb_model([x, tf.convert_to_tensor(p7)], training=False)
        else:
            E = emb_model(x, training=False)
        E_chunks.append(np.asarray(E))
        if cal is not None:
            w_chunks.append(np.asarray(cal(E))[:, 0])
        trials.append(batch["trial"].numpy())
        for k in _META_KEYS:
            meta[k].append(batch[k].numpy())
    if not E_chunks:
        return None, None
    out = {k: np.concatenate(v).astype(np.float32) for k, v in meta.items()}
    for k in ("phase", "frame_index", "n_frames"):
        out[k] = out[k].astype(np.int16)
    out["trial"] = np.array([t.decode() for t in np.concatenate(trials)])
    if need_landmarks:
        out["pose9"] = np.concatenate(pose9).astype(np.float32)
    if w_chunks:
        out["w"] = np.concatenate(w_chunks).astype(np.float32)
    return np.concatenate(E_chunks).astype(np.float32), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True, help="v1 root (contains eyedata{23,24}/)")
    ap.add_argument("--dataset", default="eyedata24")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--backbone", default="hybrid")
    ap.add_argument("--embedding-dim", type=int, default=128)
    ap.add_argument("--token-pool-extra", type=int, default=1)
    ap.add_argument("--pose-token", action="store_true",
                    help="checkpoint is a pose-token model: compute per-frame "
                         "pose features from stored landmarks and feed "
                         "[image, pose7]")
    ap.add_argument("--save-pose", action="store_true",
                    help="keep per-frame head pose (pose9) from the stored "
                         "landmarks without feeding it to the model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--limit", type=int, help="first N subjects (smoke test)")
    ap.add_argument("--shard", metavar="I/N",
                    help="process only subjects where crc32(subject_id) %% N == I "
                         "(stable partition, SLURM-array-safe; same convention "
                         "as preprocessing/extract_frames_to_gcs.py)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    # Under SLURM, cap TF threads to the allocation (several shards share a node)
    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 0))
    if n_cpus:
        tf.config.threading.set_intra_op_parallelism_threads(n_cpus)
        tf.config.threading.set_inter_op_parallelism_threads(2)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "extraction_meta.json").write_text(json.dumps({
        "weights": os.path.basename(args.weights), "backbone": args.backbone,
        "embedding_dim": args.embedding_dim, "token_pool_extra": args.token_pool_extra,
        "pose_token": args.pose_token, "save_pose": args.save_pose,
        "frames_dir": args.frames_dir, "dataset": args.dataset,
    }, indent=2))

    records = sorted(Path(args.frames_dir, args.dataset).glob("*.tfrecord"))
    if args.shard:
        import zlib
        shard_i, shard_n = (int(x) for x in args.shard.split("/"))
        records = [r for r in records
                   if zlib.crc32(r.stem.encode()) % shard_n == shard_i]
    if args.limit:
        records = records[:args.limit]
    emb_model, cal = build_embedding_model(args)
    print(f"{len(records)} subjects, embedding dim {args.embedding_dim}, "
          f"calibration weight {'saved' if cal is not None else 'not saved'}")

    for i, rec in enumerate(records, 1):
        dst = out_dir / f"{rec.stem}.npz"
        if dst.exists() and not args.overwrite:
            continue
        E, meta = process_subject(emb_model, rec, args.batch_size,
                                  pose=args.pose_token, cal=cal,
                                  save_pose=args.save_pose)
        if E is None:
            print(f"[{i}/{len(records)}] {rec.stem}: NO face-found frames, skipped")
            continue
        np.savez_compressed(dst, E=E, **meta)
        print(f"[{i}/{len(records)}] {rec.stem}: {len(E)} frames "
              f"({len(np.unique(meta['trial']))} trials)", flush=True)

    print("done")


if __name__ == "__main__":
    main()
