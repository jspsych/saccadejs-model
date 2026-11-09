"""The model-version rule, exercised against a synthetic export.

`web/export/propose_version.py` decides a release's semver by diffing a fresh export
manifest against the published registry (see web/ARCHITECTURE.md). The rule is meant to be
mechanical -- minor for new weights, major for a changed contract, and a loud non-zero exit
rather than a guess when it cannot tell -- which is exactly the kind of thing that rots
quietly. These tests pin the decision table.

The module lives in web/export, outside the installed package, and imports nothing beyond
the standard library, so it is loaded by path.
"""

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "web" / "export" / "propose_version.py"

spec = importlib.util.spec_from_file_location("propose_version", SRC)
pv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pv)


MANIFEST = {
    "checkpoint": {"entity": "e", "project": "p", "run": "r", "file": "w.weights.h5"},
    "model": {
        "input": {"name": "eye_image", "shape": [1, 36, 144, 1], "dtype": "float32"},
        "output": {"name": "embedding", "shape": [1, 128], "dtype": "float32"},
        "input_range": "uint8_0_255",
        "embedding_dim": 128,
        "opset": 17,
    },
    "preprocess": {
        "pad": 0.01,
        "landmarks": {"left": 226, "right": 446, "top": [27, 257], "bottom": [23, 253]},
        "gray": "bt601",
        "resize": {"method": "cv2_inter_linear", "width": 144, "height": 36},
        "clahe": {"clip": 2.0, "tiles": [8, 8]},
        "order": ["crop", "gray", "resize", "clahe"],
        "note": "prose that must not affect the hash",
    },
    "versions": {"keras": "3.14.1"},
    "exported_at": "2026-09-05T12:54:51.291356+00:00",
}

ONNX_BYTES = b"pretend onnx"
BASE_SHA = "0" * 64


def baseline(**over):
    r = {"version": "1.0.0", "served": True, "file": "eye_embedding.onnx",
         "sha256": BASE_SHA, "bytes": 12, "contract": 1,
         "contractHash": pv.contract_hash(MANIFEST),
         "io": pv.io_of(MANIFEST),
         "provenance": {"wandb": "e/p/an-older-run"}}
    r.update(over)
    return {"id": "eye-embedding", "releases": [copy.deepcopy(r)]}


def run(tmp_path, manifest=None, registry=None, onnx=ONNX_BYTES):
    """Run main() over synthetic inputs; returns (exit code, proposal, entry|None)."""
    (tmp_path / "export_manifest.json").write_text(
        json.dumps(manifest if manifest is not None else MANIFEST))
    (tmp_path / "m.onnx").write_bytes(onnx)
    (tmp_path / "releases.json").write_text(
        json.dumps(registry if registry is not None else baseline()))
    argv = ["propose_version.py",
            "--manifest", str(tmp_path / "export_manifest.json"),
            "--onnx", str(tmp_path / "m.onnx"),
            "--releases", str(tmp_path / "releases.json"),
            "--out", str(tmp_path / "out")]
    old, sys.argv = sys.argv, argv
    try:
        code = pv.main()
    finally:
        sys.argv = old
    proposal = json.loads((tmp_path / "out" / "version_proposal.json").read_text())
    entry_path = tmp_path / "out" / "release_entry.json"
    entry = json.loads(entry_path.read_text()) if entry_path.exists() else None
    return code, proposal, entry


def test_new_weights_same_contract_is_a_minor_bump(tmp_path):
    code, proposal, entry = run(tmp_path)
    assert (code, proposal["decision"], proposal["version"]) == (0, "minor", "1.1.0")
    assert entry["contract"] == 1, "a minor must not move the contract"
    assert entry["sha256"] == proposal["sha256"]
    assert entry["provenance"]["wandb"] == "e/p/r"
    assert entry["provenance"]["exportedAt"] == "2026-09-05T12:54:51Z", "registry format"


@pytest.mark.parametrize("mutate", [
    pytest.param(lambda m: m["preprocess"]["clahe"].__setitem__("clip", 3.0), id="clahe"),
    pytest.param(lambda m: m["preprocess"]["landmarks"].__setitem__("left", 33), id="landmark"),
    pytest.param(lambda m: m["preprocess"]["resize"].__setitem__("width", 128), id="resize"),
    pytest.param(lambda m: m["preprocess"]["order"].reverse(), id="order"),
    pytest.param(lambda m: m["model"].__setitem__("input_range", "float_0_1"), id="range"),
])
def test_contract_changes_are_major_and_carry_the_contract_int(tmp_path, mutate):
    m = copy.deepcopy(MANIFEST)
    mutate(m)
    code, proposal, entry = run(tmp_path, manifest=m)
    assert (code, proposal["decision"], proposal["version"]) == (0, "major", "2.0.0")
    assert entry["contract"] == 2


def test_io_change_is_major_even_without_a_baseline_hash(tmp_path):
    m = copy.deepcopy(MANIFEST)
    m["model"]["output"]["shape"] = [1, 200]
    m["model"]["embedding_dim"] = 200
    reg = baseline()
    del reg["releases"][0]["contractHash"]
    code, proposal, _ = run(tmp_path, manifest=m, registry=reg)
    assert (code, proposal["decision"], proposal["version"]) == (0, "major", "2.0.0")


def test_prose_note_does_not_move_the_contract(tmp_path):
    m = copy.deepcopy(MANIFEST)
    m["preprocess"]["note"] = "reworded entirely"
    code, proposal, _ = run(tmp_path, manifest=m)
    assert proposal["decision"] == "minor"


def test_identical_bytes_are_not_a_release(tmp_path):
    import hashlib
    reg = baseline(sha256=hashlib.sha256(ONNX_BYTES).hexdigest())
    code, proposal, entry = run(tmp_path, registry=reg)
    assert (code, proposal["decision"], proposal["version"], entry) == (0, "none", None, None)


def test_baseline_without_a_contract_hash_is_flagged_not_guessed(tmp_path):
    reg = baseline()
    del reg["releases"][0]["contractHash"]
    code, proposal, entry = run(tmp_path, registry=reg)
    assert (code, proposal["decision"], entry) == (1, "flagged", None)
    assert proposal["contractHash"] in " ".join(proposal["reasons"]), "prints the backfill"


def test_opset_change_is_flagged_rather_than_classified(tmp_path):
    reg = baseline()
    reg["releases"][0]["io"]["opset"] = 15
    code, proposal, entry = run(tmp_path, registry=reg)
    assert (code, proposal["decision"], entry) == (1, "flagged", None)


def test_reexporting_the_published_checkpoint_is_flagged_not_a_minor(tmp_path):
    """Same weights, unchanged contract, different bytes. "minor" would be a lie: it would
    change the model recorded in participants' data without changing the model."""
    reg = baseline()
    reg["releases"][0]["provenance"]["wandb"] = "e/p/r"      # the manifest's own checkpoint
    code, proposal, entry = run(tmp_path, registry=reg)
    assert (code, proposal["decision"], proposal["version"], entry) == (1, "flagged", None, None)
    assert "re-export, not a release" in " ".join(proposal["reasons"])


def test_a_contract_change_outranks_the_same_checkpoint(tmp_path):
    """Same weights but a changed contract is still a major -- the guard must not swallow it."""
    m = copy.deepcopy(MANIFEST)
    m["preprocess"]["clahe"]["clip"] = 3.0
    reg = baseline()
    reg["releases"][0]["provenance"]["wandb"] = "e/p/r"
    code, proposal, _ = run(tmp_path, manifest=m, registry=reg)
    assert (code, proposal["decision"], proposal["version"]) == (0, "major", "2.0.0")


def test_newest_release_wins_over_registry_order(tmp_path):
    reg = baseline()
    old = copy.deepcopy(reg["releases"][0])
    old["version"] = "0.9.0"
    reg["releases"].append(old)          # out of order on purpose
    code, proposal, _ = run(tmp_path, registry=reg)
    assert proposal["baseline"] == "1.0.0"
