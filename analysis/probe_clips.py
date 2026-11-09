"""Each released trial video's codec, frame size and frame timestamps, for the dataset summary.

Reads every webm in each released participant's tar (the raw bucket's layout, staged on the HPC
as {webm_dir}/eyedata{N}/{subject}.tar), decodes it in a temporary directory, and writes one CSV
row per clip, keyed by the participant's code:

    collection, participant, phase   phase 1 or 2 for eyedata24, blank for eyedata23
    codec                            OpenCV's FourCC for the stream, e.g. h264 or VP80
    width, height                    the video's frame size in pixels
    n_frames                         frames decoded
    timestamps_ms                    each frame's decoder timestamp, ";"-separated; some clips
                                     carry an unreadable one (a huge negative number)

Then analysis/dataset_conditions.py reduces these to one row per participant. Run on an emc node
of the HPC (CPU only), sharded like the other v1 scripts:

    .venv/bin/python analysis/probe_clips.py --include preprocessing/include_list_v1.csv \\
        --webm-dir /work/jdeleeuw/webm --out clips.0.csv --shard 0/2 --workers 120
"""

import argparse
import csv
import os
import tarfile
import tempfile
import zlib
from concurrent.futures import ProcessPoolExecutor

import cv2


def fourcc(v):
    v = int(v)
    return "".join(chr((v >> 8 * i) & 0xFF) for i in range(4)).strip("\x00 ")


def probe_participant(job):
    collection, code, webm_dir = job
    rows = []
    with tempfile.TemporaryDirectory() as tmp, \
            tarfile.open(os.path.join(webm_dir, collection, code + ".tar")) as tf:
        for m in tf:
            if not m.name.endswith(".webm"):
                continue
            path = os.path.join(tmp, "clip.webm")
            with open(path, "wb") as out:
                out.write(tf.extractfile(m).read())
            cap = cv2.VideoCapture(path)
            codec = fourcc(cap.get(cv2.CAP_PROP_FOURCC))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            times = []
            while cap.grab():
                times.append(cap.get(cv2.CAP_PROP_POS_MSEC))
            cap.release()
            # eyedata24's files are <code>_<phase>_<x>_<y>.webm; eyedata23's have no phase.
            parts = os.path.basename(m.name)[:-len(".webm")].split("_")
            phase = parts[1] if collection == "eyedata24" and len(parts) == 4 else ""
            rows.append([collection, code, phase, codec, width, height, len(times),
                         ";".join(f"{t:.1f}" for t in times)])
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--include", required=True, help="CSV with dataset and subject_id columns")
    ap.add_argument("--webm-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1", metavar="I/N",
                    help="only subjects where crc32(subject_id) %% N == I")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    with open(args.include) as f:
        jobs = [(r["dataset"], r["subject_id"], args.webm_dir) for r in csv.DictReader(f)
                if zlib.crc32(r["subject_id"].encode()) % shard_n == shard_i]
    with open(args.out, "w", newline="") as f, ProcessPoolExecutor(args.workers) as ex:
        w = csv.writer(f)
        w.writerow(["collection", "participant", "phase", "codec", "width", "height",
                    "n_frames", "timestamps_ms"])
        for rows in ex.map(probe_participant, jobs, chunksize=4):
            w.writerows(rows)


if __name__ == "__main__":
    main()
