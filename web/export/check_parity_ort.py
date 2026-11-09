import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
FIX = HERE / "fixtures"
ONNX = HERE / "out" / "models" / "eye_embedding.onnx"
GATE = 1e-4


EMB_OUT = "embedding"
WEIGHT_OUT = "cal_weight"


def report(label, got, want):
    d = np.abs(got - want)
    print(f"{label}: max-abs {d.max():.3e}  mean-abs {d.mean():.3e}  "
          f"worst sample {int(d.max(1).argmax())} ({d.max(1).max():.3e})")
    return d.max()


def main():
    X = np.load(FIX / "model_inputs.npy")
    E = np.load(FIX / "model_embeddings.npy")
    sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    names = [o.name for o in sess.get_outputs()]
    if EMB_OUT not in names:
        print(f"FAIL: no {EMB_OUT!r} output; found {names}")
        return 1

    # Outputs are fetched by name, never by position: a graph with two outputs gives no
    # guarantee about their order, and comparing the weight against the embedding fixture
    # would fail in a way that looks like a numerical regression.
    runs = [dict(zip(names, sess.run(None, {name: X[i:i + 1]}))) for i in range(len(X))]
    print(f"N={len(X)} input={name} outputs={names}")

    worst = report(EMB_OUT, np.concatenate([r[EMB_OUT] for r in runs]), E)

    wpath = FIX / "model_cal_weights.npy"
    if WEIGHT_OUT in names and wpath.exists():
        W = np.load(wpath)
        got = np.concatenate([r[WEIGHT_OUT] for r in runs])
        worst = max(worst, report(WEIGHT_OUT, got, W))
        if got.min() < 0.0 or got.max() > 1.0:
            print(f"FAIL: {WEIGHT_OUT} outside [0, 1]: [{got.min():.3f}, {got.max():.3f}]")
            return 1
    elif WEIGHT_OUT in names:
        print(f"note: model has {WEIGHT_OUT} but {wpath.name} is missing; "
              "re-run make_fixtures.py to check it")
    elif wpath.exists():
        print(f"note: {wpath.name} exists but the model has no {WEIGHT_OUT} output")

    if worst >= GATE:
        print(f"FAIL: max-abs >= {GATE}")
        return 1
    print(f"PASS (< {GATE})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
