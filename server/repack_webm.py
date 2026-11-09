#!/usr/bin/env python3
"""
Repack flat webm/ dirs into per-subject tar archives and build a manifest CSV.

Usage (run on the EC2 server):
  python3 repack_webm.py --dataset 23          # full eyedata23
  python3 repack_webm.py --dataset 24          # full eyedata24
  python3 repack_webm.py --dataset 23 --test 5 # smoke test: 5 subjects only
  python3 repack_webm.py --all                 # both datasets

Output:
  eyedata{N}/data/webmtar/{subject}.tar  — one tar per subject
  eyedata{N}/data/manifest.csv           — one row per webm file

Filename schemas (x THEN y, matching the legacy jpg parser in
et_util/tfrecord_processing.py -- an earlier version of this script had the
two names swapped, which poisoned the v1 frame extraction's labels; fixed
2026-06-11, see preprocessing/fix_v1_labels.py):
  eyedata23: {subject}_{x_pct}_{y_pct}.webm
  eyedata24: {subject}_{phase}_{x_pct}_{y_pct}.webm

Tars are written to a .tmp file then atomically renamed, so a crash leaves no
partial archives. Already-completed tars are skipped on re-run.
"""

import argparse
import csv
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

BASE = Path("/var/www/html/jdeleeuw")
MANIFEST_FIELDS = ["dataset", "subject_id", "phase", "x_pct", "y_pct", "filename"]


def parse_filenames(webm_dir: Path, dataset: str) -> dict[str, list[dict]]:
    subjects: dict[str, list[dict]] = defaultdict(list)
    for fname in os.listdir(webm_dir):
        if not fname.endswith(".webm"):
            continue
        stem = fname[:-5]
        parts = stem.split("_")
        if dataset == "23":
            if len(parts) != 3:
                print(f"  WARNING: unexpected filename {fname!r}, skipping", file=sys.stderr)
                continue
            subject, x_pct, y_pct = parts
            phase = ""
        else:
            if len(parts) != 4:
                print(f"  WARNING: unexpected filename {fname!r}, skipping", file=sys.stderr)
                continue
            subject, phase, x_pct, y_pct = parts
        if not subject:
            print(f"  WARNING: empty subject in {fname!r}, skipping", file=sys.stderr)
            continue
        subjects[subject].append(
            {"dataset": f"eyedata{dataset}", "subject_id": subject,
             "phase": phase, "y_pct": y_pct, "x_pct": x_pct, "filename": fname}
        )
    return subjects


def repack_dataset(dataset: str, test_n: Optional[int] = None) -> list[dict]:
    data_dir = BASE / f"eyedata{dataset}" / "data"
    webm_dir = data_dir / "webm"
    tar_dir = data_dir / "webmtar"
    tar_dir.mkdir(exist_ok=True)

    print(f"\n=== eyedata{dataset}: scanning filenames ===")
    subjects = parse_filenames(webm_dir, dataset)
    subject_list = sorted(subjects.keys())
    total = len(subject_list)
    print(f"  {total} subjects, {sum(len(v) for v in subjects.values())} files")

    if test_n:
        subject_list = subject_list[:test_n]
        print(f"  TEST MODE: processing {test_n} subjects only")

    all_rows: list[dict] = []
    skipped = 0

    for i, subject in enumerate(subject_list, 1):
        tar_path = tar_dir / f"{subject}.tar"
        tmp_path = tar_dir / f"{subject}.tar.tmp"

        if tar_path.exists():
            skipped += 1
            all_rows.extend(subjects[subject])
            continue

        file_list = [r["filename"] for r in subjects[subject]]
        # build tar from within webm_dir so paths inside the archive are bare filenames
        result = subprocess.run(
            ["tar", "cf", str(tmp_path)] + file_list,
            cwd=webm_dir,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"  ERROR tarring {subject}: {result.stderr}", file=sys.stderr)
            tmp_path.unlink(missing_ok=True)
            continue

        tmp_path.rename(tar_path)
        all_rows.extend(subjects[subject])

        if i % 100 == 0 or i == len(subject_list):
            done = i - skipped
            print(f"  [{i}/{len(subject_list)}] {done} packed, {skipped} skipped")

    return all_rows


def write_manifest(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  manifest: {len(rows)} rows → {path}")


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", choices=["23", "24"])
    group.add_argument("--all", action="store_true")
    parser.add_argument("--test", type=int, metavar="N",
                        help="process only the first N subjects (smoke test)")
    args = parser.parse_args()

    datasets = ["23", "24"] if args.all else [args.dataset]
    all_rows: list[dict] = []

    for ds in datasets:
        rows = repack_dataset(ds, test_n=args.test)
        all_rows.extend(rows)
        manifest_path = BASE / f"eyedata{ds}" / "data" / "manifest.csv"
        write_manifest(rows, manifest_path)

    if args.all:
        combined_path = BASE / "manifest_all.csv"
        write_manifest(all_rows, combined_path)
        print(f"\nCombined manifest: {combined_path}")


if __name__ == "__main__":
    main()
