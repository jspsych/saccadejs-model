"""Fix the swapped x/y labels in the v1 all-frames records (bug, 2026-06-11).

The webm filenames encode {subject}_[{phase}_]{x}_{y}; server/repack_webm.py
named those tokens y_pct/x_pct in the bucket manifests, so every v1 tfrecord
(and every npz derived from them) carries x and y swapped relative to the
published records / model convention. Pixels, landmarks and crops are
unaffected -- this rewrites ONLY the two label floats.

Idempotent by inspection, not by marker: the trial lattice is 16 wide x 9 tall
(true x = 16 integer columns 5,11..95; true y = 9 rows 5,16.25..95 -- the .25
quarter values only ever appear in y). Each file is checked and only swapped
if it is detectably wrong, so re-running is safe.

Modes (combinable):
  --records DIR [--shard I/N]   rewrite {DIR}/eyedata{23,24}/*.tfrecord
  --npz DIR                     rewrite per-subject embedding npz files
  --manifests PATH [PATH...]    swap the y_pct/x_pct column NAMES in a repack
                                manifest csv (data stays in place; the names
                                were wrong, not the values)

Run on an emc node (tfrecord mode imports tensorflow):
  PYTHONPATH=. .venv/bin/python preprocessing/fix_v1_labels.py \
      --records /work/jdeleeuw/frames/v1 --shard $I/$N
"""

import argparse
import csv
import os
import sys
import zlib
from pathlib import Path

import numpy as np

QUARTERS = {16.25, 38.75, 61.25, 83.75}  # values unique to the 9-row y lattice


def diagnose(xs, ys):
    """'swapped' | 'correct' | 'ambiguous' from the label values of one subject."""
    ux, uy = set(np.unique(xs).tolist()), set(np.unique(ys).tolist())
    if ux & QUARTERS:
        return "swapped"       # stored x holds the 9-row (true y) lattice
    if uy & QUARTERS:
        return "correct"
    if len(ux) != len(uy):     # 16 columns vs 9 rows
        return "swapped" if len(ux) < len(uy) else "correct"
    return "ambiguous"


def fix_tfrecord(path):
    """Swap x/y in every example of one tfrecord (atomic replace). Returns status."""
    import tensorflow as tf

    raw = [r.numpy() for r in tf.data.TFRecordDataset(str(path))]
    if not raw:
        return "empty"
    exs = [tf.train.Example.FromString(r) for r in raw]
    xs = np.array([e.features.feature["x"].float_list.value[0] for e in exs])
    ys = np.array([e.features.feature["y"].float_list.value[0] for e in exs])
    status = diagnose(xs, ys)
    if status != "swapped":
        return status
    tmp = str(path) + ".tmp"
    with tf.io.TFRecordWriter(tmp) as w:
        for e in exs:
            f = e.features.feature
            x = f["x"].float_list.value[0]
            f["x"].float_list.value[0] = f["y"].float_list.value[0]
            f["y"].float_list.value[0] = x
            w.write(e.SerializeToString())
    os.replace(tmp, path)
    return "fixed"


def fix_npz(path):
    z = dict(np.load(path, allow_pickle=False))
    status = diagnose(z["x"], z["y"])
    if status != "swapped":
        return status
    z["x"], z["y"] = z["y"], z["x"]
    tmp = str(path) + ".tmp.npz"
    np.savez_compressed(tmp, **z)
    os.replace(tmp, path)
    return "fixed"


def fix_manifest(path):
    """Swap the y_pct/x_pct COLUMN NAMES (values were always in filename order)."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    if "y_pct" not in header or "x_pct" not in header:
        return "no x/y columns"
    iy, ix = header.index("y_pct"), header.index("x_pct")
    if ix < iy:
        return "correct"      # already x-before-y: fixed previously
    header[iy], header[ix] = "x_pct", "y_pct"
    tmp = str(path) + ".tmp"
    with open(tmp, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    os.replace(tmp, path)
    return "fixed"


def _self_test():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        # npz round-trip: swapped -> fixed -> correct (second run = no-op)
        p = Path(tmp, "s.npz")
        true_x = np.array([5.0, 11.0, 17.0, 95.0], np.float32)
        true_y = np.array([5.0, 16.25, 38.75, 95.0], np.float32)
        np.savez_compressed(p, x=true_y, y=true_x, E=np.zeros((4, 2)))  # swapped
        assert fix_npz(p) == "fixed"
        z = np.load(p)
        assert z["x"].tolist() == true_x.tolist() and z["y"].tolist() == true_y.tolist()
        assert fix_npz(p) == "correct"
        # manifest: names swap once, then stable
        m = Path(tmp, "manifest.csv")
        m.write_text("subject_id,phase,y_pct,x_pct,filename\na,1,5,16.25,f.webm\n")
        assert fix_manifest(m) == "fixed"
        assert "phase,x_pct,y_pct," in m.read_text().splitlines()[0]
        assert fix_manifest(m) == "correct"
        # diagnose corner: equal-cardinality fallback
        assert diagnose(np.array([16.25, 5.0]), np.array([5.0, 11.0])) == "swapped"
        assert diagnose(np.array([5.0, 11.0]), np.array([16.25, 5.0])) == "correct"
        assert diagnose(np.array([5.0]), np.array([5.0])) == "ambiguous"
    print("[self-test] OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", help="v1 root containing eyedata{23,24}/")
    ap.add_argument("--npz", help="directory of per-subject embedding npz")
    ap.add_argument("--manifests", nargs="+", help="repack manifest csv paths")
    ap.add_argument("--shard", metavar="I/N", help="crc32 partition for --records")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        _self_test()
        return
    if not (args.records or args.npz or args.manifests):
        sys.exit("nothing to do: pass --records/--npz/--manifests (or --self-test)")

    counts = {}
    def tally(kind, name, status):
        counts[status] = counts.get(status, 0) + 1
        if status not in ("correct",):
            print(f"{kind} {name}: {status}", flush=True)

    if args.records:
        shard_i, shard_n = 0, 1
        if args.shard:
            shard_i, shard_n = (int(v) for v in args.shard.split("/"))
        files = sorted(Path(args.records).glob("eyedata*/*.tfrecord"))
        files = [p for p in files
                 if zlib.crc32(p.stem.encode()) % shard_n == shard_i]
        print(f"records: {len(files)} files in shard {shard_i}/{shard_n}")
        for p in files:
            tally("record", p.name, fix_tfrecord(p))

    if args.npz:
        for p in sorted(Path(args.npz).glob("*.npz")):
            tally("npz", p.name, fix_npz(p))

    for m in args.manifests or []:
        tally("manifest", m, fix_manifest(m))

    print(f"done: {counts}")


if __name__ == "__main__":
    main()
