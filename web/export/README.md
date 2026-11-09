# web/export — ONNX export + fixtures, the publisher for saccade.js

Isolated uv project (python 3.12). `et_util` is imported from the repo root, so every
script runs **from the repo root** with `PYTHONPATH=.` and `KERAS_BACKEND=tensorflow`
(jax is not installed here; keras locks its backend at `et_util` import time).

**3.12, not 3.13**: onnx 1.17.0 publishes no cp313 wheels, so a 3.13 environment builds it
from source — slow on a Mac and a cmake failure on a CI runner. The `<1.18` pin is the one
worth keeping: model 1.0.0 was exported with onnx 1.17.0, and the toolchain is part of what
makes the bytes reproducible.

## Setup (once)

```
cd web/export && uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
  --override <(echo "protobuf>=5.28,<7") \
  "tensorflow==2.20.0" "keras==3.14.1" "tf2onnx==1.16.1" "onnx>=1.16,<1.18" \
  "onnxruntime>=1.20" numpy opencv-python "mediapipe==0.10.35" wandb h5py
```

(The protobuf override is required: tf2onnx 1.16.1 pins `protobuf~=3.20`, TF 2.20 needs 5.x.)

## In CI (the way a release should be built)

`.github/workflows/export-model.yml` — **Actions → Export model → Run workflow** — does
everything below on `ubuntu-latest`, which is the only reason to prefer it: linux/x86 is a
hard requirement for the fixtures (see below), and a runner is x86 by construction. Inputs
are the W&B entity/project/run; it reads `WANDB_API_KEY` from the org secret, runs the parity
gate, proposes a version, and uploads the whole bundle as a job artifact.

It installs with `uv sync --project web/export --frozen`, so `uv.lock` is what CI actually
builds against — regenerate it (`uv lock --project web/export`) whenever `pyproject.toml`
changes, or CI fails rather than silently re-resolving.

## Run, in order (from the repo root)

```
KERAS_BACKEND=tensorflow PYTHONPATH=. web/export/.venv/bin/python web/export/download_checkpoint.py
KERAS_BACKEND=tensorflow PYTHONPATH=. web/export/.venv/bin/python web/export/export_onnx.py
KERAS_BACKEND=tensorflow PYTHONPATH=. web/export/.venv/bin/python web/export/make_fixtures.py
PYTHONPATH=. web/export/.venv/bin/python web/export/check_parity_ort.py
PYTHONPATH=. web/export/.venv/bin/python web/export/propose_version.py
```

`download_checkpoint.py` takes the run from `WANDB_ENTITY` / `WANDB_PROJECT` / `WANDB_RUN`
(defaults = the model 1.0.0 checkpoint), and `export_onnx.py` imports those, so the manifest
records whichever run was exported.

`propose_version.py` compares the export against the published `releases.json` and prints the
next version — **minor** for new weights, **major** for a changed contract — writing
`out/generated/version_proposal.json` and, when there is something to release,
`out/generated/release_entry.json` to paste into the registry. It exits non-zero rather than
guess (opset changes; a baseline with no `contractHash`). `--releases <path>` points it at a
local checkout instead of the published registry. See `web/ARCHITECTURE.md`.

Outputs (everything under `out/` is gitignored):

- `web/export/checkpoints/kwppnkuj/*.weights.h5` (gitignored, 62.8 MB)
- `web/export/out/models/eye_embedding.onnx` (~20.7 MB) — two outputs, see below
- `web/export/out/generated/cal_weights.json`, `export_manifest.json`
- `web/export/fixtures/model_inputs.npy`, `model_embeddings.npy`, `model_cal_weights.npy`
  (**tracked** — the parity ground truth), plus `ridge_fixture.json` and, when real frames
  are captured,
  `frame_k.png`, `landmarks_k.json`, `crop_gray_k.png`, `resized_k.png`, `clahe_k.png`
- `web/export/fixtures/preproc/` from `make_preproc_fixtures.py`

## The two graph outputs

The exported graph is `eye_image [1,36,144,1]` in, and out:

| output | shape | what |
|---|---|---|
| `embedding` | `[1,128]` | the eye embedding a per-participant ridge is fit on |
| `cal_weight` | `[1,1]` | that frame's calibration weight, sigmoid, in [0, 1] |

`cal_weight` is the trained `Calibration_Weights` head. It already belongs to
`create_masked_model`, and it already scores one image's embedding — `Dense(1, sigmoid)`
applied through `SimpleTimeDistributed`. `build_export_model` re-uses that trained layer on a
single image, so the exported weight is the one the model trained with, not a re-fit.

It belongs in the graph because the model is fetched by URL at runtime while
`cal_weights.json` is compiled into whatever `@saccadejs/core` a study installed. Those two
version numbers move independently by design, so a head kept outside the graph can end up
scoring embeddings from a model it was never trained on, with nothing to signal it. Inside the
graph, the sha256 that identifies the model covers the head as well. `cal_weights.json` is
still written, since saccade.js ships it as the opt-in `CAL_HEAD` for reproducing 1.0.0's
weighting.

**A set-level head cannot be exported this way.** With
`cfg.calibration_weighting == "set_attention"`, `CalibrationWeightingHead` attends across the
whole calibration set and optionally reads each point's coordinates. The browser runs this
graph on one crop at a time with neither, so `build_export_model` refuses rather than emit a
graph that runs and scores each frame out of context. Weighting like that must run once over
the collected set at fit time, as a separate artefact rather than a second output.

Adding the output is a **major** contract change: `weightOutput` is part of `io_of`, so
`propose_version.py` classifies it that way on its own.

`export_onnx.py` finishes by canonicalizing the graph — structural node order, positional
names for everything tf2onnx generated, sorted attributes — so the same checkpoint exports to
the same bytes every time. Without it every run produced a different sha256, and since the
sha256 *is* the model's identity in the recorded data, that made a re-export look like a new
release. The workflow exports twice and compares. (Byte-identity holds per platform: one
folded constant differs by ~6e-08 between arm64 and x86, which is one more reason releases
come from CI.)

`check_parity_ort.py` exits non-zero if onnxruntime-CPU differs from Keras by >= 1e-4.
`parity/` is the browser counterpart: it runs the same fixtures through onnxruntime-web on
**WebGPU**, which the CPU gate does not cover. `npm install && npm run dev` in `parity/`,
after an export has populated `out/`.

## Handing artifacts to saccade.js

This directory is the publisher; `jspsych/saccadejs` is the consumer. Nothing crosses
automatically — copy deliberately, and only from a linux/x86 build (see below):

| from `web/export` | to `jspsych/saccadejs` |
|---|---|
| `out/models/eye_embedding.onnx` | `packages/core/models/eye_embedding.onnx` |
| `out/generated/cal_weights.json`, `export_manifest.json` | `packages/core/src/generated/` |
| `fixtures/preproc/*`, `fixtures/ridge_fixture.json` | `packages/core/test/fixtures/` |

Both fixture generators take `--out`, so the last row can be written straight across:

```
web/export/.venv/bin/python web/export/make_preproc_fixtures.py \
  --out ../saccadejs/packages/core/test/fixtures/preproc
```

The `.npy` model-parity fixtures stay here — `check_parity_ort.py` and `parity/` are the
only consumers, and both are export-side.

## Real-frame fixtures

`make_fixtures.py` tries a ~5 s webcam capture. macOS denies camera access to a
non-interactive shell, so run it from an interactive terminal (grant the camera
prompt) — or pass a directory of face images instead. On Apple silicon you must
*also* pass `--cv-python`, see below:

```
KERAS_BACKEND=tensorflow PYTHONPATH=. web/export/.venv/bin/python \
  web/export/make_fixtures.py --cv-python /tmp/cvx86/bin/python
```

`--no-camera` skips the capture attempt and emits only the synthetic fixtures.
`face_landmarker.task` is downloaded into `web/export/` on first use.

### Why `--cv-python` (Apple silicon)

The opencv-python wheels for arm64 ship the KleidiCV HAL; its `cv2.resize`
INTER_LINEAR and CLAHE deviate from the portable C++ path — the one the training
pipeline ran on linux/x86 — by up to 3 gray levels. Staged crop fixtures made with
that build would look like a bug in the TS port. `make_fixtures.py` verifies the
local build with a CRC guard (shared with `make_preproc_fixtures.py`) and, if it
fails, either skips the staged fixtures with a warning or — with `--cv-python` —
re-runs *only* the imgproc stages in the interpreter you point it at. Build one:

```
uv python install cpython-3.11-macos-x86_64
uv venv --python cpython-3.11-macos-x86_64 /tmp/cvx86
VIRTUAL_ENV=/tmp/cvx86 uv pip install "opencv-python==4.10.0.84" "numpy<2"
```

Verified: arm64 cv2 5.0.0 fails the guard, x86_64 cv2 4.10.0 passes it. The model
I/O fixtures (`model_inputs.npy` / `model_embeddings.npy`) are unaffected for the
synthetic images; only the real-crop rows depend on cv2 imgproc, and those are
appended only when the staged outputs were produced by a good build.
