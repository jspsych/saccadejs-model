# CLAUDE.md

Guidance for Claude Code working in this repo (webcam gaze-prediction / eye tracking).

## Project shape

Monorepo for the whole pipeline. See `README.md` for the end-to-end flow.

- `et_util/` — the installable package (one source of truth), and the only place the version
  is declared (`__version__`; `pyproject.toml` reads it via `[tool.hatch.version]`).
  - Training: `config.py` (the `TrainConfig` dataclass + OSF component IDs), `models.py`,
    `custom_layers.py`, `custom_loss.py`, `augment.py`, `train.py`.
  - Data: `frames_data.py` is the **current** path (per-trial frames; `config.py` points at
    it); `data.py` + `dataset_utils.py` are the older tfrecord path, still used by the
    two-phase eval and by `frames_data.py`.
  - Preprocessing: `process_functions.py` (the eye-crop geometry that the saccade.js
    `crop.ts` port must stay bit-exact with), `tfrecord_processing.py`, `pose.py`.

  `__init__.py` sets the keras backend and then imports **only** what Keras must have
  registered (`custom_layers`, `custom_loss`); it runs on every `python -m et_util.train`, so
  nothing else belongs there. Import anything else from its submodule directly.
- `preprocessing/` — the two dataset-build Colab notebooks (R + Python).
- `training/` — the headless TPU infra: `colab_bootstrap.sh`, `run_multiframe_screen_colab.sh`.
- `analysis/` — `extract_v1_embeddings.py` (per-frame embeddings from a checkpoint, run on
  HPC). The exploratory oracles, sweeps and legacy evaluation notebooks were removed
  before the repo went public; git history and `EXPERIMENT_LOG.md` have them. Rendered
  figures are gitignored.
- `server/` — lab-server scripts.
- `web/` — everything deploy-adjacent that is still R&D. **The browser library itself lives
  in a separate repo**, `jspsych/saccadejs` (product name **saccade.js**); this repo is the
  *publisher* of the artifacts it consumes, never the consumer. What remains here:
  - `web/export/` — Keras → ONNX export of the per-image embedding model + the fixtures
    (own uv venv, `KERAS_BACKEND=tensorflow`; see its README). Outputs land in
    `web/export/out/` (gitignored) and `web/export/fixtures/`; its README has the table of
    what to copy into saccadejs, and both fixture generators take `--out` to write there
    directly. Fixtures for the preprocessing tests must come from an **x86** OpenCV build
    (arm64 wheels use a different resize HAL). `web/export/parity/` is a small Vite page
    that reruns the parity fixtures through onnxruntime-web on **WebGPU** — the CPU gate
    (`check_parity_ort.py`) does not cover the provider that actually ships.
  - `web/rig/` — the hardware timing rig: RP2040 firmware plus `panel/`, the Web Serial
    browser panel and the screen→webcam loopback analysis it validates (`npm test` there).
  - `web/ARCHITECTURE.md` — how a trained model reaches a browser: the three version numbers,
    where the artifact lives, the retention policy, the provenance recorded in experiment
    data, and the promotion pipeline that is still to build. Read it before touching the
    export or the release flow.

  `web/demo/` is **gone** (removed 2026-09-08) — it was superseded by `packages/core` in
  saccadejs and had begun to drift. Do not resurrect a demo here; changes to the browser
  runtime belong in that repo.

Train from the CLI, not notebooks: `python -m et_util.train` (defaults = the proven best
config: hybrid, embedding_dim=128, token_pool_extra=1 — cross-phase median 0.073 ± 0.001,
promoted 2026-06-10). Every knob and its default lives in `et_util/config.py`.
**The keras backend defaults to jax**, set by `os.environ.setdefault` at the top of
`et_util/__init__.py` — before any import, because `custom_layers` imports keras at
package-init time and keras locks its backend in there. An explicitly exported
`KERAS_BACKEND` still wins (that is how the eval-only analysis scripts select
tensorflow). Any new setdefault must go *above* the imports in `__init__.py`; placing
one lower, or in a submodule like `train.py`, is dead code under `python -m`.

There is no committed model checkpoint here (weights are gitignored). Historical checkpoints
need their original dims passed explicitly — the historical `full_model.weights.h5` is a
**DenseNet** model: load it with `TrainConfig(backbone="densenet", embedding_dim=200)`, not
the defaults, or `load_weights` fails. The best hybrid checkpoints (`8lsofasl` seed-42 etc.,
W&B project `eye-tracking-calib-attn`) match the current defaults.

## Where to run jobs: HPC for eval-only, Colab TPU for training

**Default routing (decided 2026-06-17):**
- **Eval-only / inference jobs (λ sweeps, re-evals, embedding extraction, analysis
  over checkpoints) → Vassar HPC (`hopper.vassar.edu`, ssh alias `hopper`; `emc` partition via SLURM).**
  No accelerator-allocation lottery, no ~60-min session-death risk, no compute-unit
  burn, persistent venv (`~/eye-tracking/.venv`); SLURM owns the job lifecycle so
  results land on `/work` and are read over SSH — none of the W&B-decoupling /
  no-poll gymnastics below are needed. `emc` is CPU-only, which is fine for
  inference (batch job; wall-clock doesn't matter). First-time setup: confirm the
  jax/keras-3 eval stack in the venv, stage `v1_eval_frame0` on `/work`, pull
  checkpoints from W&B or `/work`.
- **Training jobs (the screens) → Colab TPU** (`emc` is CPU-only; the hybrid
  backbone needs the TPU). Use the headless `colab` CLI recipe below.

## Running on Colab (via the `colab-mcp` bridge)

Training/eval run on Google Colab GPUs, driven from here through the official
`googlecolab/colab-mcp` MCP server. **The bridge connects to a Colab notebook the user has
open in their browser** — it is not headless. Key facts, learned the hard way:

### Connecting
- The MCP server is registered (user scope): `uvx git+https://github.com/googlecolab/colab-mcp`
  (needs `uv` installed; `brew install uv`).
- **You** open the connection by calling `mcp__colab__open_colab_browser_connection` — there is
  **nothing the user clicks in the browser** to initiate it. The server is a *proxy*: the cell
  tools (`run_code_cell`, `add_code_cell`, `get_cells`, `update_cell`, `delete_cell`, `move_cell`)
  only appear **after** a Colab session is connected.
- The user must have a Colab notebook tab open and a **GPU runtime** selected
  (Runtime → Change runtime type → T4). Verify: `nvidia-smi` and `import jax; jax.devices()`
  should show a `CudaDevice`.
- **Prefer the T4 *High-RAM* runtime (8 vCPU) over the free T4 (2 vCPU).** Training is
  bottlenecked by CPU augmentation on 2 cores; 8 cores make it fully GPU-bound — measured
  **1.36× faster** (303 → 223 ms/step), same GPU, free. See `EXPERIMENT_LOG.md` (2026-06-07).

### The bridge drops intermittently
- When it's dropped, every colab tool returns **`Unknown tool: <name>`**. Recover by calling
  `mcp__colab__open_colab_browser_connection` again, then retry.
- A reconnect may attach to a **fresh/blank notebook**, and the kernel may have restarted — so
  **do not rely on specific cell IDs or in-kernel variables surviving**. Re-fetch state from the
  filesystem/process list instead (e.g. `pgrep -af et_util.train`, `tail` the log file). The
  training *runtime* (files under `/content`, detached processes) persists across bridge drops as
  long as the Colab VM itself stays alive.

### Auth (private repo + credentials)
- **Colab's built-in GitHub auth does NOT reach the runtime shell** — a plain `git clone` of this
  private repo fails with exit 128 ("could not read Username"). Credentials must live *in the
  runtime*. Either:
  - a `GH_TOKEN` Colab secret (fine-grained PAT, **Resource owner = `jspsych`**, repo =
    `jspsych/eye-tracking`, Contents: Read-only), then clone with
    `https://x-access-token:{token}@github.com/...` via `subprocess` (so the token isn't echoed); or
  - `gh auth login` run interactively by the user in the Colab tab (gh is preinstalled).
- Required **Colab secrets** (🔑 sidebar, "Notebook access" ON), read via
  `from google.colab import userdata; userdata.get("...")`:
  `GH_TOKEN`, `WANDB_API_KEY`, `osftoken`, `osfusername`. Re-set them into `os.environ` after any
  kernel restart.

### Running long jobs without blocking the bridge
- `run_code_cell` blocks until the cell finishes — **never run a multi-minute training cell
  synchronously**. Launch training as a **detached background process** and poll a log file:
  ```python
  import subprocess, os
  env = dict(os.environ); env["KERAS_BACKEND"] = "jax"; env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
  logf = open("/content/eye-tracking/train.log", "w")
  proc = subprocess.Popen(["python", "-m", "et_util.train", "--backbone", "densenet",
                           "--train-epochs", "10"], cwd="/content/eye-tracking", env=env,
                          stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
  ```
  `start_new_session=True` detaches it so it survives bridge drops. Poll with a short cell that
  checks `pgrep -af et_util.train` and tails the log.
- **Output flooding:** progress bars (osf download, keras `fit`) use `\r` and will blow past the
  tool's output limit. Always filter: `tail -c 6000 train.log | tr '\r' '\n' | tail -25`. Redirect
  noisy commands to a log rather than printing inline.
- Polling from the host across turns: a recurring `CronCreate` job that re-runs the poll works, but
  **delete it (`CronDelete`) as soon as the run ends or the user closes Colab**, or it keeps firing
  and failing.

### Robust headless execution via the `colab` CLI (google-colab-cli)

The `colab` CLI (`uv tool install google-colab-cli`) is the **headless** alternative to the
`colab-mcp` browser bridge — it provisions/drives VMs from the terminal (`colab --auth=adc new -s
<name> --tpu v5e1`, `colab exec`, `colab run`). Read `colab skill` before non-trivial use. It is the
right tool for long unattended training. **The hard-won lesson (2026-06-16), after three TPU runs
died mid-training:**

- **Root cause was self-inflicted by polling — NOT a token/session time-limit.** The keep-alive
  daemon (`spawn_keep_alive`) re-checks `state.store.get(<name>)` every tick and **exits with
  `reason=session_not_found` the instant the local store entry disappears**. And `colab exec` /
  `colab sessions` *"clean up local state automatically"* on any transient 404/401. So a single
  flaky poll prunes the local entry → the daemon's next tick finds nothing → keep-alive dies → the
  VM idles out minutes later. **Polling the live session is what kills it.** The only real hard cap
  is **24 h** (the daemon's `max_duration`); a 45-min job fits trivially if left undisturbed.
- Keep-alive itself is healthy: the installed code pings the TFE endpoint
  (`colab.research.google.com/tun/m/<ep>/keep-alive/`, `client.py:295`), *not* the restricted
  `colab.pa.googleapis.com` RPC. The shipped `COLAB_SKILL.md` text still says the RPC — that's stale
  doc-lag; the `colab.pa.googleapis.com` refs are only for *assignment allocation*.

**Four rules for any headless run:**
0. **NEVER import jax / call `jax.devices()` in the exec *kernel* on a TPU box.** The v5e chip can be
   held by **exactly one process**; the moment the kernel touches jax it grabs `libtpu.so`, and the
   detached training child then dies at startup with `Unable to initialize backend 'tpu': ABORTED:
   The TPU is already in use by process with pid <kernel>` (this is the recurring "TPU-lock gotcha"
   — bitten 2026-06-11 and again 2026-06-18). So: do TPU sanity checks (`jax.devices()`) **inside
   the training process**, not the kernel; keep the launch script pure `subprocess.Popen` with **no
   jax import**; and never run a jax-importing health-check exec against a live run. If the kernel
   already grabbed the TPU, `colab restart-kernel -s <name>` frees it (re-set `WANDB_API_KEY` into
   the child env afterward — kernel restart wipes `os.environ`). Log-/ps-reading execs are TPU-safe.
1. **Launch once, then never `colab exec`/`colab sessions` against the live run.** Those prune local
   state and kill keep-alive. `colab log -s <name>` is safe (it reads the local history file).
2. **Detach the job on the VM** (`nohup … & disown`, or `subprocess.Popen(..., start_new_session=
   True)`) so it's an OS process independent of the exec socket — a dropped connection can't touch
   it.
3. **Read the result from W&B, never off the VM.** Run with `--eval-every-epochs N --wandb-project
   eye-tracking-multiframe`: `PeriodicTwoPhaseEval` logs the cross-phase **median** trajectory to
   W&B (tagged `periodic_eval/epoch`) and `wandb.save`s a checkpoint each eval, so the result
   outlives the session and a matched comparison is available at the highest common epoch even if a
   VM dies early. (Before the 2026-06-16 tweak, periodic medians only hit stdout.)

Recipe: provision once → bootstrap (install + data) → set `WANDB_API_KEY` on the VM → `nohup` the
driver with `--eval-every-epochs 5` → poll **W&B locally** (zero `colab exec`) → `colab stop` when
W&B shows the finals. No exec-polling, no race, no self-inflicted session death.

### Gotchas specific to this code on Colab
- **Don't `pip install tensorflow*` on a TPU VM — it already ships TF.** The Colab TPU image has a
  working `tensorflow` (e.g. 2.20). `pip install … tensorflow-cpu` layers a *second*, ABI-mismatched
  build (e.g. 2.21) on top, and `import tensorflow` then dies with `ImportError:
  _pywrap_tensorflow_lite_metrics_wrapper.so: undefined symbol …` (data.py imports TF for tf.data, so
  the whole run crashes at import). Bootstrap should install only what's missing (jax[tpu], keras,
  keras-hub, wandb, osfclient) and **leave TF alone**. If already broken:
  `pip uninstall -y tensorflow tensorflow-cpu && pip install -q tensorflow-cpu` (one coherent build),
  then verify `from et_util import data` imports before launching. (Bitten 2026-06-18.)
- **OSF data layout:** the `6b5cd` training tarball extracts its `*.tfrecords` **flat into the cwd**,
  not into a `single_eye_tfrecords/` subdir — `mv *.tfrecords single_eye_tfrecords/` after untar (or
  point `--tfrecord-dir` at wherever they land). The private eval set (`uf2sh`, needs OSF auth)
  extracts similarly — glob recursively for `*.tfrecords` and pass that dir to `--eval-tfrecord-dir`.
- **JAX backend + W&B:** `wandb.integration.keras.WandbMetricsLogger` reads the LR via `.numpy()`
  (TF-only) and crashes under `KERAS_BACKEND=jax` at the first `on_epoch_end`. `train.py` already
  uses a custom backend-agnostic epoch logger instead — keep it that way.
- **Startup cost:** `train.py` does one full pass over the dataset to count subjects (for the LR
  schedule) — ~1–2 min before epoch 1. `--train-epochs` must be **> `warmup_epochs` (2)** or the
  cosine-decay window goes negative.
- **TFRecords (training data):** `osf -p 6b5cd fetch single_eye_tfrecords.tar.gz` (~2.5 GB) → one
  `.tfrecords` file per subject (~1747). For quick smoke tests, copy ~15 files into a subset dir
  and pass `--tfrecord-dir`.
- **Raw webm video:** `gs://webcam-jspsych-webm` (GCP project `webcam-eyetracking-jspsych`,
  `us-central1`). Per-subject tars — `eyedata23/{subject}.tar` (2,130 subjects, 56 GB) and
  `eyedata24/{subject}.tar` (352 subjects, 19 GB). Manifests at
  `eyedata23/manifest.csv`, `eyedata24/manifest.csv`, and `manifest_all.csv` (columns:
  `dataset, subject_id, phase, y_pct, x_pct, filename`; eyedata24 has phases 1 & 2, eyedata23
  phase is blank). Access via `gcloud storage cp gs://webcam-jspsych-webm/eyedata23/{id}.tar .`
  or stream directly with `tf.data` from `gs://` paths. Colab already uses the same Google
  identity — no extra auth needed if the lab account has `roles/storage.objectViewer` on the
  bucket.
- **Perf:** training is **input-pipeline-bound** — the GPU sits near 0% util while the CPU
  augmentation pipeline feeds it. `data.py`/`train.py` already mitigate this: `.cache()` the
  decoded/grouped data (parse once), `.repeat()` + file-based subject count (no counting pass),
  and TF's GPU is hidden so JAX owns it while augmentation runs on CPU. Per-frame photometric
  augmentation (`per_image_augmentation=True`) is vectorized over the time axis — no `tf.map_fn`,
  and blur is dropped from that path (blur survives only in the per-sequence mode).

## CI

`.github/workflows/ci.yml` runs on pushes to `main`, on PRs, and on demand. Four jobs:

| job | what |
|---|---|
| `guard` | the dependency-free half of `tests/test_keras_backend.py` — reports in seconds |
| `python` | `pip install -e ".[dev]"`, full pytest, then builds the model and asserts 5,198,273 params and that `pyproject`/`__version__` agree |
| `rig-panel` | `web/rig/panel`: typecheck, 54 vitest tests, build |
| `parity` | `web/export/parity`: typecheck, build |

The `dev` extra exists so the two backend-probe tests **run** instead of skipping: they need
jax importable to prove the default is jax. jax is deliberately *not* a hard dependency —
training installs `jax[tpu]` on the Colab VM, and the eval paths export
`KERAS_BACKEND=tensorflow`. Run the suite locally with `pip install -e ".[dev]" && pytest`.

The `python` job installs the package rather than just adding the repo to `PYTHONPATH`, so an
imported-but-undeclared dependency fails there. (That is how `experiment_utils.py`'s `sklearn`
import went unnoticed for months.)

`export-model.yml` is the publisher, added 2026-09-09 and **manual only**
(`workflow_dispatch`): the linux/x86 export of a W&B run, the parity gate, the preprocessing
fixtures, and `propose_version.py`, uploaded as a job artifact. It reads `WANDB_API_KEY` from
the org secret. Still not built: the saccadejs fixture tests, the held-out eval gate, and the
auto-PR that would actually cut a release — see `web/ARCHITECTURE.md`.

## Conventions
- **Track experiment work across two files:**
  - `TODO.md` — the forward-looking backlog of open experiment ideas (what to try next, with
    de-risk status and prior evidence so killed ideas aren't re-run). Consult it to pick the
    next experiment; check items off / delete them as they're done, and add new ideas as they
    come up.
  - `EXPERIMENT_LOG.md` — the backward-looking record. **Always add a brief entry at the top**
    after any experiment or notable change (training runs/results, pipeline or model changes,
    perf findings, bugs fixed) with what changed, what you learned, and the result. Include
    date + time — run `date` for an accurate local timestamp; don't guess it.
  - When a `TODO.md` item is finished, write its result in `EXPERIMENT_LOG.md` and remove it
    from `TODO.md` (keep a one-line note under "Closed / null" if it's a dead end worth not
    re-running).
- Keep the Keras serialization label `package="et_util"` (saved weights depend on it).
- Never commit secrets. The preprocessing notebooks and server scripts read OSF/W&B/GitHub creds
  from the environment; outputs are stripped before committing.
