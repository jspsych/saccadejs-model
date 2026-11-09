"""Summarize the SaccadeJS dataset for the paper: who is in it, and how they got there.

Reads the raw jsPsych session JSON from the private OSF project, the raw video manifests and the
frame extraction's manifest on GCS, and writes two files with no identifiers in them:

    participants.csv   one row per released participant, keyed by the experiment's random
                       8-character code: collection, split, trials, frames, frames with a face,
                       webcam type, glasses, the browser window's size, and, with
                       --conditions, the video's format and the median recording conditions
    consent.csv        counts only, per collection, of each step from a saved session to a
                       released participant: who opted out of public release, who was removed
                       on request, who was dropped for incomplete data
    demographics.csv   with --prolific-demographics, counts only, per collection, of the
                       released participants' Prolific demographics: age, sex, ethnicity,
                       country of residence, first language, student and employment status
    README.md

A participant is released if they are in the consent include list
(preprocessing/include_list_v1.csv); their trial and frame counts come from the frame
extraction's manifest. The build stops if anyone removed on request is still released. Anyone who
withdrew appears only in the counts, never as a row.

The session JSON and the Prolific exports are read in memory and never written to disk. Needs an
OSF token with access to the private project, gcloud credentials for the two buckets, and, for the
demographics, a Prolific API token for the account that ran the studies:

    OSF_TOKEN=... PROLIFIC_TOKEN=... python analysis/build_dataset_summary.py \
        --exclude-prolific <private file of Prolific IDs removed on request> \
        --held-out-consent <private CSV: prolific_id, permission (1, 0 or blank)> \
        --conditions <analysis/dataset_conditions.py's CSV> \
        --prolific-demographics \
        --out data/saccadejs-dataset-summary

Standard library only, plus the gcloud CLI.
"""

import argparse
import csv
import io
import json
import os
import re
import subprocess
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path

FORMAT = "saccadejs-dataset-summary/3"

# The two collections: the training set, recorded in 2023, and the held-out evaluation set,
# recorded in December 2024 and March 2025 with each participant doing the task twice. Each was
# run as one or more Prolific studies.
COLLECTIONS = {
    "eyedata23": {"split": "training", "osf_json": "vmdu6", "trials": 144,
                  "prolific_studies": ["647f73837293678ea4bc228c"]},
    "eyedata24": {"split": "held-out", "osf_json": "c5ux8", "trials": 288,
                  "prolific_studies": ["676187dc9264a41bd3d2672e", "67e5b460adc5e865e4ba929c"]},
}
FRAMES_MANIFEST = "gs://webcam-jspsych-frames/v1/manifest.csv"
INCLUDE_LIST = Path(__file__).resolve().parent.parent / "preprocessing" / "include_list_v1.csv"
VIDEO_MANIFEST = "gs://webcam-jspsych-webm/{collection}/manifest.csv"


# consent.csv's rows, in order from a recorded session to a released participant.
COUNTS = ["recorded videos", "saved sessions", "answered the consent question",
          "consent: public", "consent: opted out", "opted out: research only", "opted out: delete",
          "removed on request", "public, not removed: complete videos",
          "public, not removed: incomplete videos", "public, not removed, not released",
          "public, not removed, complete videos, not released",
          "released, with a saved session", "released"]
# eyedata24 only: the Prolific messages, and on what consent each released participant is in.
MESSAGE_COUNTS = ["asked by Prolific message", "by message: agreed", "by message: declined",
                  "by message: no response", "released, consent in the experiment",
                  "released, consent by Prolific message"]

# Questions, found by their wording, since only some versions of the experiment labeled them.
WEBCAM = ("integrated or external webcam", ["integrated", "external"])
GLASSES = ("will you be wearing glasses", ["yes", "no"])
PUBLIC = ("If you do not want your videos to be part", ["public", "opted out"])
RESEARCH = ("Is it OK for us to use these videos", ["research only", "delete"])


def read_text(path):
    """A local file, or a gs:// object through gcloud."""
    if path.startswith("gs://"):
        return subprocess.run(["gcloud", "storage", "cat", path], check=True,
                              capture_output=True, text=True).stdout
    return Path(path).read_text()


def osf_sessions(node, token):
    """Each saved session's rows, from the node's JSON archives, read in memory."""
    def get(url):
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(req) as r:
            return r.read()

    files = json.loads(get(f"https://api.osf.io/v2/nodes/{node}/files/osfstorage/"))["data"]
    for f in files:
        if not f["attributes"]["name"].endswith(".tar.gz"):
            continue
        with tarfile.open(fileobj=io.BytesIO(get(f["links"]["download"]))) as t:
            for m in t.getmembers():
                # A session is <code>.json; <code>_interactions.json is its browser events.
                if m.isfile() and re.fullmatch(r"\w{8}\.json", Path(m.name).name):
                    rows = json.loads(t.extractfile(m).read())
                    if isinstance(rows, str):
                        rows = json.loads(rows)
                    yield Path(m.name).stem, rows


# demographics.csv's variables: the Prolific export's column for each. Countries and languages
# with fewer than SMALL released participants in a collection are counted together, apart from
# Prolific's own "Other".
DEMOGRAPHICS = {"sex": "Sex", "ethnicity": "Ethnicity simplified",
                "country of residence": "Country of residence", "first language": "Language",
                "student status": "Student status", "employment status": "Employment status"}
GROUPED = {"country of residence", "first language"}
SMALL = 5
AGE_BANDS = [(18, 24), (25, 34), (35, 44), (45, 54), (55, 64), (65, 120)]
# What the export shows for a value it no longer has.
UNAVAILABLE = {"", "DATA_EXPIRED", "CONSENT_REVOKED"}


def prolific_export(study, token):
    """Each participant's row of a Prolific study's export, read in memory: their approved
    submission if they have one."""
    req = urllib.request.Request(f"https://api.prolific.com/api/v1/studies/{study}/export/",
                                 headers={"Authorization": f"Token {token}"})
    with urllib.request.urlopen(req) as r:
        rows = list(csv.DictReader(io.StringIO(r.read().decode("utf-8"))))
    out = {}
    for r in sorted(rows, key=lambda r: r["Status"] == "APPROVED"):
        out[r["Participant id"]] = r
    return out


def demographic_counts(collection, rows):
    """demographics.csv's rows for one collection, from its released participants' export rows
    (None for a participant not found in the export)."""
    out = []

    def add(variable, counts):
        out.extend({"collection": collection, "variable": variable, "value": v, "n": n}
                   for v, n in counts)

    ages = []
    for r in rows:
        v = r and r["Age"]
        if v and v not in UNAVAILABLE:
            ages.append(int(float(v)))
    labels = [f"{lo}–{hi}" if hi < 120 else f"{lo}+" for lo, hi in AGE_BANDS]
    bands = Counter(label for a in ages
                    for label, (lo, hi) in zip(labels, AGE_BANDS) if lo <= a <= hi)
    add("age", [(label, bands[label]) for label in labels]
        + [("not available", len(rows) - len(ages))])
    ages.sort()
    mean = sum(ages) / len(ages)
    sd = (sum((a - mean) ** 2 for a in ages) / (len(ages) - 1)) ** 0.5
    median = (ages[(len(ages) - 1) // 2] + ages[len(ages) // 2]) / 2
    add("age (years)", [("mean", round(mean, 2)), ("sd", round(sd, 2)), ("median", median),
                        ("min", ages[0]), ("max", ages[-1])])
    for variable, column in DEMOGRAPHICS.items():
        # The export writes a few multi-part answers as a list's repr, e.g. "homemaker', 'retired".
        counts = Counter(r[column].replace("', '", ", ") if r and r[column] not in UNAVAILABLE
                         else "not available" for r in rows)
        missing = counts.pop("not available", 0)
        if variable in GROUPED:
            small = [v for v, n in counts.items() if n < SMALL]
            other = sum(counts.pop(v) for v in small)
        listed = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        if variable in GROUPED and other:
            listed.append((f"other (fewer than {SMALL} each)", other))
        add(variable, listed + [("not available", missing)])
    return out


def answer(rows, question):
    """The label of a button response, found by the question's wording, or ""."""
    text, labels = question
    hits = [r for r in rows if r.get("trial_type") == "html-button-response"
            and text in str(r.get("stimulus", ""))]
    if not hits or hits[-1].get("response") in (None, ""):
        return ""
    return labels[int(hits[-1]["response"])]


def window(rows):
    check = [r for r in rows if r.get("trial_type") == "browser-check"]
    return (check[0].get("width", ""), check[0].get("height", "")) if check else ("", "")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--exclude-prolific", type=Path,
                    help="file of Prolific IDs removed from the dataset on request, one per line")
    ap.add_argument("--held-out-consent", type=Path,
                    help="CSV of the held-out (eyedata24) participants asked through Prolific's "
                         "messages whether their videos may be public, since that collection's "
                         "final consent answer was not saved for every session: prolific_id, and "
                         "permission 1 (yes), 0 (no) or blank (no response).")
    ap.add_argument("--frames-manifest", default=FRAMES_MANIFEST)
    ap.add_argument("--conditions", type=Path,
                    help="analysis/dataset_conditions.py's CSV: each released participant's video "
                         "format and median recording conditions, added as columns")
    ap.add_argument("--prolific-demographics", action="store_true",
                    help="write demographics.csv from the Prolific studies' exports; needs "
                         "PROLIFIC_TOKEN")
    args = ap.parse_args()
    token = os.environ["OSF_TOKEN"]
    removed_ids = set(args.exclude_prolific.read_text().split()) if args.exclude_prolific else set()
    by_message = ({r["prolific_id"]: r["permission"].strip()
                   for r in csv.DictReader(open(args.held_out_consent))}
                  if args.held_out_consent else {})

    included = {(r["dataset"], r["subject_id"]) for r in csv.DictReader(open(INCLUDE_LIST))}
    released = {}
    for r in csv.DictReader(io.StringIO(read_text(args.frames_manifest))):
        if (r["dataset"], r["subject_id"]) in included:
            released[(r["dataset"], r["subject_id"])] = r
    missing = included - set(released)
    if missing:
        raise SystemExit(f"{len(missing)} included participants are not in the frame manifest")

    people, consent, demographics = [], [], []
    for collection, c in COLLECTIONS.items():
        export = {}
        if args.prolific_demographics:
            for study in c["prolific_studies"]:
                export.update(prolific_export(study, os.environ["PROLIFIC_TOKEN"]))
        released_demographics = []
        videos = Counter(r["subject_id"] for r in
                         csv.DictReader(io.StringIO(read_text(VIDEO_MANIFEST.format(collection=collection)))))
        n = Counter()
        n["recorded videos"] = len(videos)
        for code, rows in osf_sessions(c["osf_json"], token):
            n["saved sessions"] += 1
            public = answer(rows, PUBLIC)
            removed = any(r.get("prolific_id") in removed_ids for r in rows)
            complete = videos.get(code, 0) == c["trials"]
            is_released = (collection, code) in released
            if public:
                n["answered the consent question"] += 1
                n[f"consent: {public}"] += 1
                if public == "opted out":
                    n[f"opted out: {answer(rows, RESEARCH) or 'no answer'}"] += 1
            if removed:
                n["removed on request"] += 1
                if is_released:
                    raise SystemExit("a participant removed on request is still in the include list")
            if public == "public" and not removed:
                n["public, not removed: complete videos" if complete
                  else "public, not removed: incomplete videos"] += 1
                if not is_released:
                    n["public, not removed, not released"] += 1
                    if complete:
                        n["public, not removed, complete videos, not released"] += 1
            message = next((by_message[r["prolific_id"]] for r in rows[:1]
                            if r.get("prolific_id") in by_message), None)
            # Released only with a yes, in the experiment or by message, and no later no.
            if is_released:
                if message == "0" or (public != "public" and message != "1"):
                    raise SystemExit("a released participant has no recorded consent")
                if collection == "eyedata24":
                    n["released, consent in the experiment" if public == "public"
                      else "released, consent by Prolific message"] += 1
            if is_released:
                n["released, with a saved session"] += 1
                prolific_id = next((r["prolific_id"] for r in rows if r.get("prolific_id")), None)
                released_demographics.append(export.get(prolific_id))
                m = released[(collection, code)]
                width, height = window(rows)
                people.append({
                    "participant": code, "collection": collection, "split": c["split"],
                    "n_trials": m["n_trials"], "n_frames": m["n_frames"],
                    "n_face_found": m["n_face_found"],
                    "webcam": answer(rows, WEBCAM), "glasses": answer(rows, GLASSES),
                    "screen_width": width, "screen_height": height,
                })
        n["released"] = sum(1 for (d, _) in released if d == collection)
        counts = COUNTS
        if collection == "eyedata24":
            answers = Counter(by_message.values())
            n["asked by Prolific message"] = len(by_message)
            n["by message: agreed"] = answers["1"]
            n["by message: declined"] = answers["0"]
            n["by message: no response"] = answers[""]
            counts = COUNTS + MESSAGE_COUNTS
        consent += [{"collection": collection, "count": k, "n": n.get(k, 0)} for k in counts]
        if args.prolific_demographics:
            # Released participants without a saved session have no Prolific ID to look up.
            released_demographics += [None] * (n["released"] - n["released, with a saved session"])
            demographics += demographic_counts(collection, released_demographics)

    # Released participants without a saved session still get a row, from the manifest alone.
    have = {(p["collection"], p["participant"]) for p in people}
    for (collection, code), m in released.items():
        if (collection, code) not in have:
            people.append({"participant": code, "collection": collection,
                           "split": COLLECTIONS[collection]["split"],
                           "n_trials": m["n_trials"], "n_frames": m["n_frames"],
                           "n_face_found": m["n_face_found"], "webcam": "", "glasses": "",
                           "screen_width": "", "screen_height": ""})
    people.sort(key=lambda p: (p["collection"], p["participant"]))

    if args.conditions:
        with open(args.conditions) as f:
            conditions = {(r.pop("collection"), r.pop("participant")): r for r in csv.DictReader(f)}
        missing = [p for p in people if (p["collection"], p["participant"]) not in conditions]
        if missing:
            raise SystemExit(f"{len(missing)} released participants have no recording conditions")
        for p in people:
            p.update(conditions[(p["collection"], p["participant"])])

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "participants.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(people[0]))
        w.writeheader()
        w.writerows(people)
    with open(args.out / "consent.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["collection", "count", "n"])
        w.writeheader()
        w.writerows(consent)
    if demographics:
        with open(args.out / "demographics.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["collection", "variable", "value", "n"])
            w.writeheader()
            w.writerows(demographics)
    readme = Path(__file__).with_name("dataset_summary_README.md").read_text()
    (args.out / "README.md").write_text(readme.replace("{FORMAT}", FORMAT))
    for row in consent:
        print(f"{row['collection']:10} {row['count']:45} {row['n']}")
    print(len(people), "participants written")


if __name__ == "__main__":
    main()
