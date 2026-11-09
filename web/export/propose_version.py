"""Propose the next model version by diffing a fresh export against the published registry.

The rule lives in web/ARCHITECTURE.md: **minor** = new weights behind an identical
interface (drop-in), **major** = preprocessing or I/O changed, and a major carries the
`contract` int up with it. Both halves of that are already written down -- the new side in
`out/generated/export_manifest.json`, the published side in saccade.js's
`packages/core/models/releases.json` -- so the bump is a diff, not a judgement call.

What the runtime must implement is hashed into `contractHash`, which the proposed registry
entry carries so the next release can compare against it mechanically. Releases published
before this script existed have no such hash; against one of those the preprocessing half
cannot be compared and the verdict is `flagged` (exit 1) rather than a guess. Backfill the
printed hash into that entry once, and the wart is gone.

Opset changes are flagged, never classified: an opset bump is not a contract break by the
definition above (the browser implements nothing new) but it does change which
onnxruntime-web versions can load the file, which is a human's call.

    PYTHONPATH=. web/export/.venv/bin/python web/export/propose_version.py
"""

import argparse
import datetime
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "out" / "generated" / "export_manifest.json"
ONNX = HERE / "out" / "models" / "eye_embedding.onnx"
OUT_DIR = HERE / "out" / "generated"
RELEASES = ("https://raw.githubusercontent.com/jspsych/saccadejs/main/"
            "packages/core/models/releases.json")


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def load_releases(src):
    if str(src).startswith(("http://", "https://")):
        with urllib.request.urlopen(str(src), timeout=30) as r:
            return json.loads(r.read())
    return json.loads(Path(src).read_text())


def semver(v):
    try:
        parts = [int(p) for p in str(v).split(".")]
    except ValueError:
        raise SystemExit(f"unparseable version in the registry: {v!r}")
    if len(parts) != 3:
        raise SystemExit(f"unparseable version in the registry: {v!r}")
    return tuple(parts)


def io_of(manifest):
    """The registry's `io` block, built from the export manifest.

    `weightOutput` is present only for a model that carries a calibration weight as a second
    graph output. Gaining or losing it changes what the runtime reads, so it belongs in the
    contract hash: a model whose weighting disappeared would otherwise look like a minor
    release while silently changing every fit from weighted to unweighted.
    """
    m = manifest["model"]
    io = {
        "input": {"name": m["input"]["name"], "shape": m["input"]["shape"],
                  "dtype": m["input"]["dtype"]},
        "output": {"name": m["output"]["name"], "shape": m["output"]["shape"],
                   "dtype": m["output"]["dtype"]},
        "inputRange": m["input_range"],
        "opset": m["opset"],
    }
    w = m.get("weight_output")
    if w:
        io["weightOutput"] = {"name": w["name"], "shape": w["shape"], "dtype": w["dtype"],
                              "range": w["range"], "granularity": w["granularity"]}
    return io


def contract_hash(manifest):
    """Hash of everything a browser must implement to run this model correctly.

    Deliberately excludes the opset (a loader concern, not an implementation one) and the
    `note` in the preprocess block (prose, and it has already changed once without any
    behaviour changing with it).
    """
    io = io_of(manifest)
    io.pop("opset")
    payload = {
        "preprocess": {k: v for k, v in manifest["preprocess"].items() if k != "note"},
        "io": io,
        "embeddingDim": manifest["model"]["embedding_dim"],
    }
    return "sha256:" + hashlib.sha256(canonical(payload).encode()).hexdigest()


def checkpoint_id(manifest):
    ck = manifest["checkpoint"]
    return f"{ck['entity']}/{ck['project']}/{ck['run']}"


def same_checkpoint(manifest, release):
    """Did this export come from the weights the baseline release was built from?"""
    return release.get("provenance", {}).get("wandb") == checkpoint_id(manifest)


def diff(a, b, path=""):
    """Every leaf where two JSON structures disagree, as 'path: old -> new' lines."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            out += diff(a.get(k), b.get(k), f"{path}.{k}" if path else k)
        return out
    return [] if a == b else [f"{path}: {json.dumps(a)} -> {json.dumps(b)}"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=MANIFEST, type=Path)
    ap.add_argument("--onnx", default=ONNX, type=Path)
    ap.add_argument("--releases", default=RELEASES,
                    help="registry to diff against: a URL or a local releases.json")
    ap.add_argument("--out", default=OUT_DIR, type=Path,
                    help="where version_proposal.json / release_entry.json are written")
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    blob = args.onnx.read_bytes()
    sha = hashlib.sha256(blob).hexdigest()
    chash = contract_hash(manifest)
    io = io_of(manifest)

    registry = load_releases(args.releases)
    releases = registry.get("releases", [])
    if not releases:
        raise SystemExit(f"no releases in {args.releases}")
    base = max(releases, key=lambda r: semver(r["version"]))
    if base is not releases[-1]:
        print(f"warning: newest release {base['version']} is not the last entry "
              "(the registry documents itself as oldest-first)")

    base_io = dict(base.get("io", {}))
    base_opset = base_io.pop("opset", None)
    new_io = dict(io)
    new_opset = new_io.pop("opset")
    io_changes = diff(base_io, new_io)
    opset_changed = base_opset != new_opset
    base_hash = base.get("contractHash")

    print(f"baseline   {base['version']}  sha256:{base['sha256'][:12]}  "
          f"contract {base['contract']}")
    print(f"candidate  sha256:{sha[:12]}  {len(blob)} bytes  {chash}")
    print(f"checkpoint {manifest['checkpoint']['entity']}/"
          f"{manifest['checkpoint']['project']}/{manifest['checkpoint']['run']}")

    reasons = []
    if sha == base["sha256"]:
        decision = "none"
        reasons.append("byte-identical to the published model; nothing to release")
        if base_hash is None:
            reasons.append(
                f"baseline {base['version']} has no contractHash. These are its own bytes, "
                f"so once you have checked this export's preprocess block against what that "
                f"release shipped, backfill \"contractHash\": \"{chash}\" into its entry -- "
                "the next release can then be classified mechanically.")
    elif io_changes:
        decision = "major"
        reasons += [f"I/O changed -- {c}" for c in io_changes]
    elif base_hash is None:
        decision = "flagged"
        reasons.append(
            f"baseline {base['version']} predates contractHash, so the preprocessing half "
            "cannot be compared. Verify by hand that crop/gray/resize/CLAHE are unchanged, "
            f"then backfill \"contractHash\": \"{chash}\" into that entry and re-run.")
    elif base_hash != chash:
        decision = "major"
        reasons.append(f"contract changed -- {base_hash} -> {chash}")
    elif same_checkpoint(manifest, base):
        # Same weights, same contract, different bytes. "minor" would be a lie: the
        # weights are not new, and publishing would change the identity recorded in
        # participants' data without changing the model they saw.
        decision = "flagged"
        reasons.append(
            f"these are {base['version']}'s own weights ({checkpoint_id(manifest)}) behind an "
            "unchanged contract, but the bytes differ -- a re-export, not a release. Either "
            "the export is not reproducible (see canonicalize() in export_onnx.py) or the "
            "toolchain moved deliberately; decide which, rather than minting a version that "
            "changes every participant's recorded model for no change in the model.")
    else:
        decision = "minor"
        reasons.append("new weights behind an identical interface")

    if opset_changed and decision != "none":
        reasons.append(
            f"opset {base_opset} -> {new_opset}: not a contract break, but it changes which "
            "onnxruntime-web versions can load the file -- decide deliberately")
        decision = "flagged" if decision == "minor" else decision

    M, m, p = semver(base["version"])
    version = {"major": f"{M + 1}.0.0", "minor": f"{M}.{m + 1}.0"}.get(decision)
    contract = base["contract"] + 1 if decision == "major" else base["contract"]

    print(f"\nverdict: {decision}" + (f"  ->  {version}" if version else ""))
    for r in reasons:
        print(f"  - {r}")

    args.out.mkdir(parents=True, exist_ok=True)
    proposal = {"decision": decision, "version": version, "contract": contract,
                "sha256": sha, "bytes": len(blob), "contractHash": chash,
                "baseline": base["version"], "reasons": reasons}
    (args.out / "version_proposal.json").write_text(json.dumps(proposal, indent=2) + "\n")

    if version:
        ck = manifest["checkpoint"]
        # Match the registry's existing shape: UTC, seconds, trailing Z.
        exported = (datetime.datetime.fromisoformat(manifest["exported_at"])
                    .astimezone(datetime.timezone.utc)
                    .strftime("%Y-%m-%dT%H:%M:%SZ"))
        entry = {
            "version": version,
            "date": exported[:10],
            "served": True,
            "file": "eye_embedding.onnx",
            "sha256": sha,
            "bytes": len(blob),
            "contract": contract,
            "contractHash": chash,
            "io": io,
            "provenance": {
                "wandb": f"{ck['entity']}/{ck['project']}/{ck['run']}",
                "checkpoint": ck["file"],
                "exportedAt": exported,
                "toolchain": manifest["versions"],
            },
            "notes": "",
        }
        path = args.out / "release_entry.json"
        path.write_text(json.dumps(entry, indent=2) + "\n")
        print(f"\nwrote {path} -- paste into releases.json, fill in `notes`, and remember "
              "to move the outgoing model into its version directory")

    return 1 if decision == "flagged" else 0


if __name__ == "__main__":
    sys.exit(main())
