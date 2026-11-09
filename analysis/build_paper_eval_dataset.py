"""Package per-frame embeddings of the held-out eval set (eyedata24) for the paper.

Reads the per-subject npz files that analysis/extract_v1_embeddings.py writes
(run with the deployed checkpoint and --save-pose) and the extraction's
per-subject stats, and writes a self-contained dataset in the layout of the
validation study's (jspsych/eyetracking-cookoff-exp), so the paper's R code
reads both the same way:

    participants.csv            one row per eyedata24 subject, including those
                                with no usable frames
    trials.csv                  each trial's position in the session
    participants/<id>.json.gz   one object per trial, per-frame streams packed
                                as base64 over little-endian typed arrays
    README.md

Every face-found frame is kept: the recording starts 1000 ms into each trial,
after the saccade to the target, so every frame is fixation.

With --frame-stats (analysis/extract_v1_frame_stats.py's output) the dataset is
format 2: each frame also gets the eye crop's contrast, each eye's brightness
and the frame-to-frame noise (in eye_image_stats), the head's yaw and pitch in
degrees (head_angles) and the crop's size in camera pixels (eye_crop_px), and
participants.csv the camera image's size. Without it, format 1.

Numpy only, so it runs on the HPC login node:

    ~/eye-tracking/.venv/bin/python analysis/build_paper_eval_dataset.py \
        --embeddings /work/jdeleeuw/embeddings/v1_eyedata24_kwppnkuj \
        --stats /work/jdeleeuw/frames/v1/eyedata24 \
        --session-info <dir from analysis/eyedata24_session_info.py> \
        --frame-stats /work/jdeleeuw/frame_stats/v1_eyedata24 \
        --out /work/jdeleeuw/paper/saccadejs-eval

Participant files already written are kept unless --overwrite.
"""

import argparse
import base64
import csv
import gzip
import json
from pathlib import Path

import numpy as np

FORMAT_1 = "saccadejs-eval/1"
FORMAT_2 = "saccadejs-eval/2"

POSE_NAMES = ["center_x", "center_y", "log_iod", "roll", "depth",
              "width_asym", "z_asym", "nose_dx", "nose_dy"]


def pack(a, dtype):
    """A 2D array as base64 over its little-endian bytes, row after row."""
    return base64.b64encode(
        np.ascontiguousarray(a, dtype=np.dtype(dtype).newbyteorder("<")).tobytes()
    ).decode("ascii")


def aligned_stats(d, stats):
    """The frame statistics in the order of the embeddings' frames, matched on trial and
    frame index. Every embedded frame is face-found, as is every frame with statistics."""
    where = {(t, int(f)): i for i, (t, f) in enumerate(zip(stats["trial"], stats["frame_index"]))}
    try:
        order = np.array([where[(t, int(f))] for t, f in zip(d["trial"], d["frame_index"])])
    except KeyError as e:
        raise SystemExit(f"no frame statistics for frame {e.args[0]}")
    return {k: stats[k][order] for k in
            ("contrast", "brightness_left", "brightness_right", "noise", "yaw", "pitch",
             "crop_width", "crop_height")}


def trial_rows(d, stats=None):
    """One dict per trial, frames in recording order. With `stats` (from aligned_stats),
    the format-2 streams too."""
    trial = d["trial"]
    rows = []
    for name in np.unique(trial):
        idx = np.flatnonzero(trial == name)
        idx = idx[np.argsort(d["frame_index"][idx], kind="stable")]
        first = idx[0]
        rows.append({
            "trial": name.removesuffix(".webm"),
            "phase": int(d["phase"][first]),
            # Target position in percent of the screen's width and height.
            "x": float(d["x"][first]),
            "y": float(d["y"][first]),
            # Decoded frames in the trial's video, face found or not.
            "n_frames": int(d["n_frames"][first]),
            "frame_index": pack(d["frame_index"][idx], "i2"),
            "time_ms": pack(d["pos_msec"][idx], "f2"),
            "embedding": pack(d["E"][idx], "f2"),
            "weight": pack(d["w"][idx], "f2"),
            "eye_aspect_ratio": pack(np.c_[d["ear_left"][idx], d["ear_right"][idx]], "f2"),
            "eye_image_stats": pack(np.c_[d["sharpness"][idx], d["brightness_mean"][idx]], "f2"),
            "head_pose": pack(d["pose9"][idx], "f2"),
        })
        if stats is not None:
            rows[-1]["eye_image_stats"] = pack(np.c_[
                d["sharpness"][idx], d["brightness_mean"][idx], stats["contrast"][idx],
                stats["brightness_left"][idx], stats["brightness_right"][idx],
                stats["noise"][idx]], "f2")
            rows[-1]["head_angles"] = pack(np.c_[stats["yaw"][idx], stats["pitch"][idx]], "f2")
            rows[-1]["eye_crop_px"] = pack(
                np.c_[stats["crop_width"][idx], stats["crop_height"][idx]], "f2")
    # Phase, then position: the files do not record the order trials were shown in.
    rows.sort(key=lambda r: (r["phase"], r["x"], r["y"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", required=True, help="extract_v1_embeddings.py output")
    ap.add_argument("--stats", required=True, help="dir of <id>.stats.json from the frame extraction")
    ap.add_argument("--session-info", required=True,
                    help="participants.csv and trials.csv from eyedata24_session_info.py")
    ap.add_argument("--frame-stats",
                    help="extract_v1_frame_stats.py output; makes the dataset format 2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    fmt = FORMAT_2 if args.frame_stats else FORMAT_1

    emb_dir, out = Path(args.embeddings), Path(args.out)
    meta = json.loads((emb_dir / "extraction_meta.json").read_text())
    (out / "participants").mkdir(parents=True, exist_ok=True)
    info_dir = Path(args.session_info)
    with open(info_dir / "participants.csv") as f:
        info = {r["participant"]: r for r in csv.DictReader(f)}

    participants = []
    for stats_path in sorted(Path(args.stats).glob("*.stats.json")):
        stats = json.loads(stats_path.read_text())
        pid = stats["subject_id"]
        npz = emb_dir / f"{pid}.npz"
        row = {"participant": pid, "n_trials": stats["n_trials"],
               "n_frames": stats["n_frames"], "n_face_found": stats["n_face_found"],
               "n_trials_embedded": 0, "n_frames_embedded": 0,
               **{k: v for k, v in info[pid].items() if k != "participant"}}
        if args.frame_stats:
            row["camera_width"] = row["camera_height"] = ""
        if npz.exists():
            d = np.load(npz)
            missing = {"w", "pose9"} - set(d.files)
            if missing:
                raise SystemExit(f"{npz} lacks {missing}: extract with the deployed "
                                 "checkpoint and --save-pose")
            stats = None
            if args.frame_stats:
                s = np.load(Path(args.frame_stats) / f"{pid}.npz")
                stats = aligned_stats(d, s)
                row["camera_width"] = round(float(s["camera_width"]))
                row["camera_height"] = round(float(s["camera_height"]))
            dst = out / "participants" / f"{pid}.json.gz"
            if args.overwrite or not dst.exists():
                with gzip.open(dst, "wt") as f:
                    json.dump({"format": fmt, "participant": pid,
                               "trials": trial_rows(d, stats)}, f)
            row["n_trials_embedded"] = len(np.unique(d["trial"]))
            row["n_frames_embedded"] = len(d["E"])
        participants.append(row)

    with open(out / "participants.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(participants[0]))
        w.writeheader()
        w.writerows(participants)

    (out / "trials.csv").write_text((info_dir / "trials.csv").read_text())
    (out / "extraction_meta.json").write_text(json.dumps(meta, indent=2))
    readme = Path(__file__).with_name("paper_eval_dataset_README.md")
    (out / "README.md").write_text(readme.read_text())
    n = sum(r["n_trials_embedded"] > 0 for r in participants)
    print(f"{len(participants)} participants, {n} with embedded frames -> {out}")


if __name__ == "__main__":
    main()
