import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# The run to export is an input, not a constant: the publisher workflow sets these so a
# release can be cut from any candidate run. The defaults are the model 1.0.0 checkpoint.
# export_onnx.py imports these, so the export manifest records whichever run was used.
ENTITY = os.environ.get("WANDB_ENTITY", "vassar-cogsci-lab")
PROJECT = os.environ.get("WANDB_PROJECT", "eye-tracking-multiframe")
RUN = os.environ.get("WANDB_RUN", "kwppnkuj")
FILE = os.environ.get("WANDB_WEIGHTS_FILE", "full_model.weights.h5")
EXPECTED_PARAMS = 5_198_273

OUT_DIR = Path(__file__).resolve().parent / "checkpoints" / RUN


def _pick(run):
    files = list(run.files())
    for f in files:
        if f.name == FILE:
            return f
    h5 = [f for f in files if f.name.endswith(".weights.h5")]
    if len(h5) != 1:
        raise RuntimeError(f"cannot pick a weights file from {[f.name for f in files]}")
    return h5[0]


def download() -> Path:
    import requests
    import wandb

    api = wandb.Api()
    run = api.run(f"{ENTITY}/{PROJECT}/{RUN}")
    wf = _pick(run)
    out = OUT_DIR / wf.name
    if out.exists() and out.stat().st_size == wf.size:
        print(f"present: {out} ({out.stat().st_size / 1e6:.1f} MB)")
        return out
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    with requests.get(wf.url, auth=("api", api.api_key), stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
    assert tmp.stat().st_size == wf.size, (tmp.stat().st_size, wf.size)
    tmp.rename(out)
    print(f"downloaded: {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out


def verify(path: Path) -> None:
    import numpy as np

    from et_util.config import TrainConfig
    from et_util.models import create_masked_model

    model = create_masked_model(TrainConfig())
    model.load_weights(str(path))
    n = int(sum(int(np.prod(v.shape)) for v in model.weights))
    print(f"params: {n}")
    assert n == EXPECTED_PARAMS, f"expected {EXPECTED_PARAMS}, got {n}"
    print("param count OK")


if __name__ == "__main__":
    verify(download())
