#!/usr/bin/env python3
"""
Extract ALL frames from the raw trial webms on GCS into per-frame TFRecords.

Reads per-subject tars from the raw bucket (gs://webcam-jspsych-webm), decodes
every frame of every trial webm, runs MediaPipe face mesh per frame, and writes
one TFRecord file per subject to the processed bucket. This is the build-once
job that turns frame-index sweeps, multi-frame calibration/TTA, QA gating, and
crop-format experiments into dataset-side filters instead of new pipelines.

Output layout (DST_BUCKET, default gs://webcam-jspsych-frames):
    {version}/{dataset}/{subject_id}.tfrecord     one example per FRAME
    {version}/{dataset}/{subject_id}.stats.json   per-subject QA summary
    {version}/manifest.csv                        built by --finalize-manifest

Per-frame example schema (one tf.train.Example per decoded frame):
    subject_id        bytes   raw subject id string (NOT the int id of the old
                              tfrecords — map dataset-side if needed)
    dataset           bytes   "eyedata23" | "eyedata24"
    trial             bytes   source webm filename (unique trial key)
    phase             int64   1|2 for eyedata24, 0 for eyedata23 (single-phase)
    x, y              float   gaze label, 0-100 screen pct (matches old records)
    frame_index       int64   0-based index within the trial webm
    n_frames          int64   total decoded frames in this trial
    pos_msec          float   decoder timestamp of the frame
    face_found        int64   1 if MediaPipe found a face (else image/landmark
                              features are empty/zero and QA floats are -1)
    landmarks         bytes   serialized float32 tensor (478, 3), MediaPipe mesh
    eye_img           bytes   serialized uint8 tensor (36, 144) — same recipe
                              as the legacy single-eye pipeline (both-eyes
                              crop pad=0.01 -> gray -> resize 144x36 -> CLAHE
                              clipLimit=2.0 tile 8x8), directly consumable by
                              existing checkpoints. NOT bit-identical to the
                              legacy records: this pipeline feeds MediaPipe
                              RGB (correct), while the legacy pipeline fed BGR
                              (a bug, fixed 2026-06-10), so landmark-derived
                              crop boxes shift slightly. Run comparisons
                              within ONE extraction; don't mix old and new
                              records as arms of the same experiment
    img_width         int64   144 (legacy parser compat)
    img_height        int64   36  (legacy parser compat)
    eye_hires_jpeg    bytes   JPEG (color, q90) of the SAME both-eyes crop at
                              native resolution, pre-CLAHE — feeds future
                              higher-res / RGB / normalization experiments
    eye_hires_width   int64
    eye_hires_height  int64
    frame_small_jpeg  bytes   JPEG (color, q80) of the full frame at 160x120 —
                              head position/context for iTracker-style inputs
    sharpness         float   variance of Laplacian on the native-res gray crop
    brightness_mean   float   mean of the native-res gray crop
    brightness_std    float   std of the native-res gray crop
    ear_left          float   eye aspect ratio, subject's left eye
    ear_right         float   eye aspect ratio, subject's right eye
    eye_width_px      float   crop width in source pixels (framing/distance QA)

Usage (on a GCE VM in us-central1 or Colab; needs the repo installed for
et_util, plus mediapipe/opencv/tensorflow, and gcloud auth with read on the
raw bucket and write on the processed bucket):

    # smoke test: 2 subjects from eyedata24
    python preprocessing/extract_frames_to_gcs.py \
        --dataset 24 --include-list consented_subjects.csv --limit 2

    # full run, both datasets
    python preprocessing/extract_frames_to_gcs.py \
        --all --include-list consented_subjects.csv --workers 30

    # after all subjects are done: aggregate stats into manifest.csv
    python preprocessing/extract_frames_to_gcs.py --all \
        --include-list consented_subjects.csv --finalize-manifest

--src and --dst accept gs:// URLs or local directories, so the job also runs
STAGED on an HPC cluster whose compute nodes lack internet access:

    1. stage the per-subject tars onto scratch, mirroring the bucket layout
       ({src}/eyedata{N}/{subject}.tar + {src}/eyedata{N}/manifest.csv) —
       e.g. rsync from the EC2 box (eyedata{N}/data/webmtar/, manifest one
       level up at eyedata{N}/data/manifest.csv) or gcloud cp from the bucket
    2. run shards as a job array, writing to local scratch:
           python preprocessing/extract_frames_to_gcs.py --all \
               --include-list include_list_v1.csv \
               --src /scratch/webm --dst /scratch/frames \
               --shard ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}
    3. bulk-upload from a node with internet (login/DTN), then finalize:
           gcloud storage rsync -r /scratch/frames/v1 \
               gs://webcam-jspsych-frames/v1
           python preprocessing/extract_frames_to_gcs.py --all \
               --include-list include_list_v1.csv --finalize-manifest

--shard I/N partitions subjects by a stable hash, so concurrent instances
never process the same subject; it works for the streaming mode too.

--include-list is REQUIRED (csv with a subject_id column): the raw bucket
includes non-consented subjects; only listed subjects are processed. Derive it
from the filtered subject set of notebook 01 / the existing public tfrecords.

Idempotent: subjects whose .tfrecord already exists in the destination are
skipped (one bucket listing up front, not per-subject stat calls). Safe to
kill and re-run; use --overwrite to force.

MediaPipe runs static_image_mode=True per frame by default (each frame stands
alone, like the legacy JPG pipeline). --tracking switches to video mode
(~2-3x faster, temporally smoother landmarks, slightly different crop boxes).
"""

import argparse
import csv
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

# Heavy imports (cv2, mediapipe, tensorflow) happen inside workers so the
# parent process stays light and fork-safe.

SRC_BUCKET = "gs://webcam-jspsych-webm"
DST_BUCKET = "gs://webcam-jspsych-frames"
VERSION = "v1"

# Standard 6-point EAR landmark sets (MediaPipe FaceMesh indices).
# EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
EAR_RIGHT = [33, 160, 158, 133, 153, 144]   # subject's right eye
EAR_LEFT = [362, 385, 387, 263, 373, 380]   # subject's left eye

# ---------------------------------------------------------------------------
# Worker-process globals, built once per worker by _init_worker
# ---------------------------------------------------------------------------
_G = {}


def _init_worker(tracking_mode):
    import cv2
    import mediapipe as mp
    import tensorflow as tf  # noqa: F401  (verify import before work starts)

    tf_quiet()
    _G["cv2"] = cv2
    _G["face_mesh"] = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=not tracking_mode,
        refine_landmarks=True,
        max_num_faces=1,
    )
    _G["clahe"] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def tf_quiet():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


# ---------------------------------------------------------------------------
# Storage helpers: --src/--dst accept gs:// URLs or local directories, so the
# job runs streaming (cloud VM) or staged (HPC scratch + bulk upload later)
# ---------------------------------------------------------------------------
def _is_gcs(path):
    return str(path).startswith("gs://")


def _cp(src, dst):
    """Copy a file (local and/or gs://). Returns an error string or None."""
    if _is_gcs(src) or _is_gcs(dst):
        r = subprocess.run(["gcloud", "storage", "cp", str(src), str(dst)],
                           capture_output=True, text=True)
        return None if r.returncode == 0 else r.stderr.strip()[:200]
    try:
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return None
    except OSError as e:
        return str(e)[:200]


def _read_text(path):
    if _is_gcs(path):
        return subprocess.run(["gcloud", "storage", "cat", str(path)],
                              capture_output=True, text=True, check=True).stdout
    return Path(path).read_text()


# ---------------------------------------------------------------------------
# tf.train feature helpers
# ---------------------------------------------------------------------------
def _bytes(v):
    import tensorflow as tf
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))


def _int64(v):
    import tensorflow as tf
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(v)]))


def _float(v):
    import tensorflow as tf
    return tf.train.Feature(float_list=tf.train.FloatList(value=[float(v)]))


# ---------------------------------------------------------------------------
# Per-frame processing
# ---------------------------------------------------------------------------
def _ear(lm, idx):
    p = [np.array(lm[i][:2]) for i in idx]
    v1 = np.linalg.norm(p[1] - p[5])
    v2 = np.linalg.norm(p[2] - p[4])
    h = np.linalg.norm(p[0] - p[3])
    return float((v1 + v2) / (2.0 * h)) if h > 0 else -1.0


def process_frame(frame_bgr, meta, frame_index, n_frames, pos_msec):
    """Build the tf.train.Example for one decoded frame. Returns (example, ok)."""
    import tensorflow as tf
    from et_util.process_functions import getBothEyesAsSingleImage

    cv2 = _G["cv2"]
    results = _G["face_mesh"].process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

    feat = {
        "subject_id": _bytes(meta["subject_id"].encode()),
        "dataset": _bytes(meta["dataset"].encode()),
        "trial": _bytes(meta["filename"].encode()),
        "phase": _int64(int(meta["phase"]) if meta["phase"] else 0),
        "x": _float(float(meta["x_pct"])),
        "y": _float(float(meta["y_pct"])),
        "frame_index": _int64(frame_index),
        "n_frames": _int64(n_frames),
        "pos_msec": _float(pos_msec),
    }

    ok = False
    if results.multi_face_landmarks:
        lm_arr = [[l.x, l.y, l.z] for l in results.multi_face_landmarks[0].landmark]
        crop = getBothEyesAsSingleImage(frame_bgr, lm_arr)
        if crop is not None and 0 not in crop.shape:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            eye_img = _G["clahe"].apply(cv2.resize(gray, (144, 36)))

            _, hires_jpeg = cv2.imencode(
                ".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            _, small_jpeg = cv2.imencode(
                ".jpg", cv2.resize(frame_bgr, (160, 120)),
                [cv2.IMWRITE_JPEG_QUALITY, 80])

            feat.update({
                "face_found": _int64(1),
                "landmarks": _bytes(tf.io.serialize_tensor(
                    tf.constant(lm_arr, dtype=tf.float32)).numpy()),
                "eye_img": _bytes(tf.io.serialize_tensor(eye_img).numpy()),
                "img_width": _int64(144),
                "img_height": _int64(36),
                "eye_hires_jpeg": _bytes(hires_jpeg.tobytes()),
                "eye_hires_width": _int64(crop.shape[1]),
                "eye_hires_height": _int64(crop.shape[0]),
                "frame_small_jpeg": _bytes(small_jpeg.tobytes()),
                "sharpness": _float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                "brightness_mean": _float(gray.mean()),
                "brightness_std": _float(gray.std()),
                "ear_left": _float(_ear(lm_arr, EAR_LEFT)),
                "ear_right": _float(_ear(lm_arr, EAR_RIGHT)),
                "eye_width_px": _float(crop.shape[1]),
            })
            ok = True

    if not ok:
        feat.update({
            "face_found": _int64(0),
            "landmarks": _bytes(b""),
            "eye_img": _bytes(b""),
            "img_width": _int64(0),
            "img_height": _int64(0),
            "eye_hires_jpeg": _bytes(b""),
            "eye_hires_width": _int64(0),
            "eye_hires_height": _int64(0),
            "frame_small_jpeg": _bytes(b""),
            "sharpness": _float(-1),
            "brightness_mean": _float(-1),
            "brightness_std": _float(-1),
            "ear_left": _float(-1),
            "ear_right": _float(-1),
            "eye_width_px": _float(-1),
        })

    return tf.train.Example(features=tf.train.Features(feature=feat)), ok


def process_trial(webm_path, meta, max_frames):
    """Decode a trial webm and yield examples. Returns (examples, trial_stats)."""
    cv2 = _G["cv2"]
    cap = cv2.VideoCapture(str(webm_path))
    frames = []
    while True:
        pos = cap.get(cv2.CAP_PROP_POS_MSEC)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append((pos, frame))
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()

    examples, n_ok = [], 0
    for i, (pos, frame) in enumerate(frames):
        ex, ok = process_frame(frame, meta, i, len(frames), pos)
        examples.append(ex)
        n_ok += ok
    return examples, {"n_frames": len(frames), "n_face_found": n_ok}


# ---------------------------------------------------------------------------
# Per-subject job (runs in a worker process)
# ---------------------------------------------------------------------------
def process_subject(job):
    """job = (dataset, subject_id, [manifest rows], cfg dict)"""
    import tensorflow as tf

    dataset, subject_id, rows, cfg = job
    src_tar = f"{cfg['src']}/{dataset}/{subject_id}.tar"
    dst_prefix = f"{cfg['dst']}/{cfg['version']}/{dataset}"

    stats = {
        "dataset": dataset, "subject_id": subject_id,
        "n_trials": 0, "n_trials_failed": 0,
        "n_frames": 0, "n_face_found": 0, "error": "",
    }

    with tempfile.TemporaryDirectory(prefix="etframes_") as tmp:
        tmp = Path(tmp)
        tar_path = tmp / f"{subject_id}.tar"
        err = _cp(src_tar, tar_path)
        if err:
            stats["error"] = f"download failed: {err}"
            return stats

        webm_dir = tmp / "webm"
        webm_dir.mkdir()
        try:
            with tarfile.open(tar_path) as tf_archive:
                tf_archive.extractall(webm_dir)
        except tarfile.TarError as e:
            stats["error"] = f"untar failed: {e}"
            return stats
        tar_path.unlink()

        out_path = tmp / f"{subject_id}.tfrecord"
        with tf.io.TFRecordWriter(str(out_path)) as writer:
            for row in sorted(rows, key=lambda r: r["filename"]):
                webm = webm_dir / row["filename"]
                if not webm.exists():
                    stats["n_trials_failed"] += 1
                    continue
                try:
                    examples, tstats = process_trial(webm, row, cfg["max_frames"])
                except Exception as e:  # one bad webm must not kill the subject
                    print(f"  trial error {row['filename']}: {e}", file=sys.stderr)
                    stats["n_trials_failed"] += 1
                    continue
                if not examples:
                    stats["n_trials_failed"] += 1
                    continue
                for ex in examples:
                    writer.write(ex.SerializeToString())
                stats["n_trials"] += 1
                stats["n_frames"] += tstats["n_frames"]
                stats["n_face_found"] += tstats["n_face_found"]

        stats_path = tmp / f"{subject_id}.stats.json"
        stats_path.write_text(json.dumps(stats))

        # upload tfrecord first, stats last: a stats.json at the destination
        # means the tfrecord copy definitely completed
        for local in (out_path, stats_path):
            err = _cp(local, f"{dst_prefix}/{local.name}")
            if err:
                stats["error"] = f"upload failed: {err}"
                return stats

    return stats


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def load_manifest(src, dataset):
    out = _read_text(f"{src}/eyedata{dataset}/manifest.csv")
    rows = list(csv.DictReader(io.StringIO(out)))
    by_subject = defaultdict(list)
    for r in rows:
        by_subject[r["subject_id"]].append(r)
    return by_subject


def list_done(dst, version, dataset):
    """Subjects whose stats.json exists (written last => fully done)."""
    prefix = f"{dst}/{version}/eyedata{dataset}"
    if _is_gcs(dst):
        r = subprocess.run(
            ["gcloud", "storage", "ls", f"{prefix}/*.stats.json"],
            capture_output=True, text=True)
        if r.returncode != 0:  # empty prefix
            return set()
        names = [Path(line).name for line in r.stdout.splitlines() if line.strip()]
    else:
        names = [p.name for p in Path(prefix).glob("*.stats.json")]
    return {n.replace(".stats.json", "") for n in names}


def load_include_list(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if rows and "subject_id" in rows[0]:
        return {r["subject_id"].strip() for r in rows}
    # fall back to a headerless one-column file
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def finalize_manifest(cfg, datasets):
    """Aggregate all per-subject stats.json into {version}/manifest.csv."""
    all_stats = []
    with tempfile.TemporaryDirectory() as tmp:
        for ds in datasets:
            prefix = f"{cfg['dst']}/{cfg['version']}/eyedata{ds}"
            if _is_gcs(cfg["dst"]):
                subprocess.run(
                    ["gcloud", "storage", "cp", f"{prefix}/*.stats.json", tmp],
                    capture_output=True, text=True)
                stat_files = list(Path(tmp).glob("*.stats.json"))
            else:
                stat_files = list(Path(prefix).glob("*.stats.json"))
            for p in stat_files:
                all_stats.append(json.loads(p.read_text()))
                if p.parent == Path(tmp):
                    p.unlink()
        out = Path(tmp) / "manifest.csv"
        fields = ["dataset", "subject_id", "n_trials", "n_trials_failed",
                  "n_frames", "n_face_found", "error"]
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(sorted(all_stats, key=lambda s: (s["dataset"], s["subject_id"])))
        err = _cp(out, f"{cfg['dst']}/{cfg['version']}/manifest.csv")
        if err:
            sys.exit(f"manifest upload failed: {err}")
    print(f"manifest: {len(all_stats)} subjects -> {cfg['dst']}/{cfg['version']}/manifest.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", choices=["23", "24"])
    group.add_argument("--all", action="store_true")
    ap.add_argument("--include-list", required=True,
                    help="csv of consented subject_ids (REQUIRED — raw bucket "
                         "contains non-consented subjects)")
    ap.add_argument("--src", default=SRC_BUCKET)
    ap.add_argument("--dst", default=DST_BUCKET)
    ap.add_argument("--version", default=VERSION)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--limit", type=int, help="process only the first N subjects (smoke test)")
    ap.add_argument("--subject-id", help="process a single subject")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="cap decoded frames per trial (0 = all)")
    ap.add_argument("--shard", metavar="I/N",
                    help="process only subjects where crc32(subject_id) %% N == I "
                         "(I in 0..N-1). Stable partition so concurrent instances "
                         "(e.g. a SLURM array) never collide")
    ap.add_argument("--tracking", action="store_true",
                    help="MediaPipe video mode (faster, temporally smoother "
                         "landmarks; default is static mode per frame)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--finalize-manifest", action="store_true",
                    help="aggregate stats.json files into manifest.csv and exit")
    args = ap.parse_args()

    datasets = ["23", "24"] if args.all else [args.dataset]
    cfg = {"src": args.src, "dst": args.dst, "version": args.version,
           "max_frames": args.max_frames}

    if args.finalize_manifest:
        finalize_manifest(cfg, datasets)
        return

    include = load_include_list(args.include_list)
    print(f"include list: {len(include)} subjects")

    shard_i, shard_n = 0, 1
    if args.shard:
        shard_i, shard_n = (int(x) for x in args.shard.split("/"))
        if not 0 <= shard_i < shard_n:
            sys.exit(f"--shard {args.shard}: I must be in 0..N-1")

    jobs = []
    for ds in datasets:
        by_subject = load_manifest(args.src, ds)
        done = set() if args.overwrite else list_done(args.dst, args.version, ds)
        subjects = sorted(by_subject)
        n_total = len(subjects)
        subjects = [s for s in subjects if s in include]
        n_consented = len(subjects)
        if args.subject_id:
            subjects = [s for s in subjects if s == args.subject_id]
        if shard_n > 1:
            subjects = [s for s in subjects
                        if zlib.crc32(s.encode()) % shard_n == shard_i]
        subjects = [s for s in subjects if s not in done]
        shard_note = f" (shard {shard_i}/{shard_n})" if shard_n > 1 else ""
        print(f"eyedata{ds}: {n_total} in manifest, {n_consented} consented, "
              f"{len(done)} already done, {len(subjects)} to process{shard_note}")
        jobs += [(f"eyedata{ds}", s, by_subject[s], cfg) for s in subjects]

    if args.limit:
        jobs = jobs[:args.limit]
    if not jobs:
        print("nothing to do")
        return

    print(f"processing {len(jobs)} subjects with {args.workers} workers")
    import multiprocessing as mp
    ctx = mp.get_context("spawn")  # mediapipe/tensorflow are not fork-safe
    n_err = 0
    with ctx.Pool(args.workers, initializer=_init_worker,
                  initargs=(args.tracking,)) as pool:
        for i, stats in enumerate(pool.imap_unordered(process_subject, jobs), 1):
            tag = "ERROR " + stats["error"] if stats["error"] else (
                f"{stats['n_trials']} trials, {stats['n_frames']} frames, "
                f"{stats['n_face_found']} with face")
            n_err += bool(stats["error"])
            print(f"[{i}/{len(jobs)}] {stats['dataset']}/{stats['subject_id']}: {tag}")

    print(f"done: {len(jobs) - n_err} ok, {n_err} errors")
    if not n_err:
        print("run again with --finalize-manifest to build manifest.csv")


if __name__ == "__main__":
    main()
