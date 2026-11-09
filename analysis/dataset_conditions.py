"""One row per released participant: their video format and the median recording conditions, for
the dataset summary (analysis/build_dataset_summary.py --conditions).

Reduces the per-clip metadata from analysis/probe_clips.py and the per-frame statistics from
analysis/extract_dataset_frame_stats.py. The frame measures are defined as in the paper's
section 4 (its R/model_eval.R), over each participant's face-found frames:

    video_codec, video_width, video_height   the most common over their clips (no participant
                                              has more than one)
    clips_timed        clips whose frame timestamps are all readable
    frame_rate         frames per second: 1000 over the median gap between consecutive readable
                       timestamps, over all clips, leaving out each clip's first gap, which is
                       often long
    clip_ms            the median clip's length: its last frame's timestamp plus one gap
    eye_crop_width     the eye crop's width in camera pixels
    brightness         mean gray of the eye crop (0-255)
    contrast           its SD
    brightness_difference  |mean left-eye brightness - mean right-eye brightness|
    noise              mean absolute change of the 144 x 36 crop from the frame just before
    sharpness          variance of the crop's Laplacian
    eye_aspect_ratio   the less open eye's aspect ratio
    blink_rate         share of frames in which that is below 0.7 of the participant's median
    interocular_distance  between the outer eye corners, as a share of the image's width
    head_x_sd, head_y_sd  SD of the eyes' midpoint, as a share of the image's width and height
    distance_sd        SD of the log interocular distance: the relative change toward and away
                       from the camera
    yaw, pitch         degrees (et_util.head_pose); pitch positive tipped up
    roll, roll_sd      degrees, the line between the outer eye corners; its median and SD

All medians unless noted. A participant without readable timestamps has no frame rate or clip
length; one with no face-found frame has only the video columns.

    python analysis/dataset_conditions.py --include preprocessing/include_list_v1.csv \\
        --clips '/work/jdeleeuw/paper/clip_meta/clips.*.csv' \\
        --frame-stats /work/jdeleeuw/frame_stats/dataset --out dataset_conditions.csv
"""

import argparse
import csv
import glob
import os
from collections import Counter, defaultdict

import numpy as np

BLINK_SHARE = 0.7  # as the paper's R/model_eval.R

COLUMNS = ["collection", "participant", "video_codec", "video_width", "video_height",
           "clips_timed", "frame_rate", "clip_ms", "eye_crop_width", "brightness", "contrast",
           "brightness_difference", "noise", "sharpness", "eye_aspect_ratio", "blink_rate",
           "interocular_distance", "head_x_sd", "head_y_sd", "distance_sd", "yaw", "pitch",
           "roll", "roll_sd"]


def clip_conditions(paths):
    """Per participant: the video's format, and frame rate and clip length from the timestamps."""
    csv.field_size_limit(10 ** 8)
    formats, gaps, lengths = defaultdict(Counter), defaultdict(list), defaultdict(list)
    for path in paths:
        with open(path) as f:
            for r in csv.DictReader(f):
                key = (r["collection"], r["participant"])
                formats[key][(r["codec"], r["width"], r["height"])] += 1
                times = np.array([float(t) for t in r["timestamps_ms"].split(";") if t])
                if len(times) < 3:
                    continue
                # Gaps between readable timestamps, after the first; a clip's length only when
                # every timestamp is readable.
                steps = np.diff(times[1:])
                ok = (times[1:-1] >= 0) & (times[2:] >= 0)
                gaps[key].extend(steps[ok])
                if ok.all() and times[0] >= 0 and (np.diff(times) >= 0).all():
                    lengths[key].append(times[-1] + np.median(steps))
    out = {}
    for key, fmt in formats.items():
        (codec, width, height), _ = fmt.most_common(1)[0]
        g = np.array(gaps[key])
        g = g[g > 0]
        out[key] = {"video_codec": codec, "video_width": width, "video_height": height,
                    "clips_timed": len(lengths[key]),
                    "frame_rate": 1000 / np.median(g) if len(g) else np.nan,
                    "clip_ms": np.median(lengths[key]) if lengths[key] else np.nan}
    return out


def frame_conditions(path):
    z = np.load(path)
    pose = z["pose9"]  # et_util.pose.POSE_NAMES
    openness = np.minimum(z["ear_left"], z["ear_right"])
    roll = np.degrees(pose[:, 3])
    return {
        "eye_crop_width": np.median(z["crop_width"]),
        "brightness": np.median(z["brightness"]),
        "contrast": np.median(z["contrast"]),
        "brightness_difference": abs(z["brightness_left"].mean() - z["brightness_right"].mean()),
        "noise": np.nanmedian(z["noise"]) if np.isfinite(z["noise"]).any() else np.nan,
        "sharpness": np.median(z["sharpness"]),
        "eye_aspect_ratio": np.median(openness),
        "blink_rate": np.mean(openness < BLINK_SHARE * np.median(openness)),
        "interocular_distance": np.exp(np.median(pose[:, 2])),
        "head_x_sd": np.std(pose[:, 0], ddof=1) if len(pose) > 1 else np.nan,
        "head_y_sd": np.std(pose[:, 1], ddof=1) if len(pose) > 1 else np.nan,
        "distance_sd": np.std(pose[:, 2], ddof=1) if len(pose) > 1 else np.nan,
        "yaw": np.nanmedian(z["yaw"]),
        "pitch": np.nanmedian(z["pitch"]),
        "roll": np.median(roll),
        "roll_sd": np.std(roll, ddof=1) if len(roll) > 1 else np.nan,
    }


def fmt(v):
    if isinstance(v, str):
        return v
    return "" if v is None or not np.isfinite(v) else f"{v:.5g}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--include", required=True, help="CSV with dataset and subject_id columns")
    ap.add_argument("--clips", required=True, help="glob of analysis/probe_clips.py's CSVs")
    ap.add_argument("--frame-stats", required=True,
                    help="analysis/extract_dataset_frame_stats.py's output: eyedata{23,24}/*.npz")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.include) as f:
        released = [(r["dataset"], r["subject_id"]) for r in csv.DictReader(f)]
    clips = clip_conditions(sorted(glob.glob(args.clips)))
    missing = [k for k in released if k not in clips]
    if missing:
        raise SystemExit(f"{len(missing)} released participants have no probed clips")

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for collection, code in sorted(released):
            row = {"collection": collection, "participant": code, **clips[(collection, code)]}
            npz = os.path.join(args.frame_stats, collection, code + ".npz")
            if os.path.exists(npz):
                row.update(frame_conditions(npz))
            w.writerow([fmt(row.get(c, np.nan)) for c in COLUMNS])
    print(len(released), "participants written")


if __name__ == "__main__":
    main()
