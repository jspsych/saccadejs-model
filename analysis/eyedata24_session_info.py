"""Non-identifying session facts for eyedata24, from the raw jsPsych JSON.

Run where the raw data lives (the lab server), with a file of the participant
codes to include; writes two CSVs to the output directory and nothing else.
No Prolific, study or session ID leaves this script.

    participants.csv  participant, webcam, glasses, screen_width, screen_height
    trials.csv        participant, trial, order, block, time_s

`order` is the trial's 1-based position among the participant's 288 recordings;
`time_s` is when it was shown, in seconds from the start of the session.
screen_width and screen_height are the browser window's size in CSS pixels
after the experiment went fullscreen (jsPsych's browser-check).

    python3 analysis/eyedata24_session_info.py JSON_DIR OUT_DIR codes.txt
"""

import csv
import json
import sys
from pathlib import Path

WEBCAM_QUESTION = "integrated or external webcam"
WEBCAM = ["integrated", "external"]
GLASSES_QUESTION = "will you be wearing glasses"
GLASSES = ["yes", "no"]


def answer(rows, question, labels):
    hits = [r for r in rows if r.get("trial_type") == "html-button-response"
            and question in str(r.get("stimulus", ""))]
    if len(hits) != 1 or hits[0].get("response") in (None, ""):
        return ""
    return labels[int(hits[0]["response"])]


def main():
    json_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    codes = Path(sys.argv[3]).read_text().split()
    out.mkdir(parents=True, exist_ok=True)

    people, trials = [], []
    for code in codes:
        path = json_dir / f"{code}.json"
        if not path.exists():
            people.append({"participant": code, "webcam": "", "glasses": "",
                           "screen_width": "", "screen_height": ""})
            continue
        rows = json.loads(path.read_text())
        check = [r for r in rows if r.get("trial_type") == "browser-check"]
        people.append({
            "participant": code,
            "webcam": answer(rows, WEBCAM_QUESTION, WEBCAM),
            "glasses": answer(rows, GLASSES_QUESTION, GLASSES),
            "screen_width": check[0].get("width", "") if check else "",
            "screen_height": check[0].get("height", "") if check else "",
        })
        videos = sorted((r for r in rows if r.get("trial_type") == "html-video-response"),
                        key=lambda r: int(r["trial_index"]))
        for i, r in enumerate(videos, 1):
            trials.append({
                "participant": code,
                "trial": str(r["response"]).removesuffix(".webm"),
                "order": i,
                "block": r.get("block", ""),
                "time_s": round(float(r["time_elapsed"]) / 1000, 3),
            })

    for name, table in (("participants.csv", people), ("trials.csv", trials)):
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(table[0]))
            w.writeheader()
            w.writerows(table)
    print(f"{len(people)} participants, {len(trials)} trials -> {out}")


if __name__ == "__main__":
    main()
