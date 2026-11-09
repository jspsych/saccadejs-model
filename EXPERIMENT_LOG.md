# Experiment Log

Brief, dated notes on experiments and notable changes — what changed, what we
learned, and the result. **Newest entries on top. Keep each entry short.**
Include date + time (run `date` for an accurate local timestamp).

---

## 2026-09-09 10:19 EDT — The ONNX export was never reproducible; now it is (per platform)

The first real CI export proposed **minor → 1.1.0** for `kwppnkuj` — the checkpoint model
1.0.0 was already built from. Not a bug in the proposal: the bytes genuinely differed
(`a685d3e18619` vs the published `c323131f7660`, 105 bytes larger). Chased it down.

The weights are bit-identical. What varies is tf2onnx bookkeeping: nodes numbered from a
process-global counter (`Unsqueeze__1071` vs `Unsqueeze__684`), Python `id()` values baked
into einsum-derived names (`einsum5213114608_ba_left_set__9`), node *and attribute* emission
order, and which of several identical constants survives deduplication — the survivor carries
whichever TF variable path named it, so even "readable" names are not stable. `PYTHONHASHSEED=0`
does nothing: two seeded runs still differed, so it is not hash randomisation.

`canonicalize()` in `export_onnx.py` now runs last: topological sort with a structural
signature (content-hashed, name-independent) as the tie-break, generated names replaced by
position, identical constants merged by content keeping the first name in sort order, small
constants (≤16 elements) numbered rather than named, attributes sorted. Verified: two
consecutive Mac exports are now byte-identical (`32aa4a9d…` twice), where before every run
differed. Parity still passes at 1.311e-06.

**What is still not reproducible, and why that is acceptable:** across platforms one folded
constant differs in the last ULP — 58 of 256 elements, ~6e-08, TF's own constant folding on
arm64 versus x86. Nothing to do from here, and it does not matter: releases are cut only by
CI on linux/x86, and `export-model.yml` now exports twice and compares sha256 as a gate.

Also: `propose_version.py` refuses to call a re-export a release. Same checkpoint + unchanged
contract + different bytes is now `flagged`, not `minor` — publishing that would change the
model string recorded in participants' data without changing the model they saw.

Consequence to decide later: the canonical build of `kwppnkuj` is not byte-identical to
published 1.0.0, so re-cutting it would need a deliberate version, not an automatic one.

## 2026-09-09 09:40 EDT — Publisher stage one: `export-model.yml`, and semver decided by diff

`WANDB_API_KEY` is now an org-level secret, which unblocked the first half of the promotion
pipeline in `web/ARCHITECTURE.md`. Added `.github/workflows/export-model.yml`: a
`workflow_dispatch` job on `ubuntu-latest` (x86 by construction — the whole reason to build
releases in CI rather than on the Mac) that downloads a W&B checkpoint, exports the ONNX,
generates both fixture sets, runs the ORT-vs-Keras parity gate, proposes a version, and
uploads the bundle. Manual on purpose: which run gets promoted is a judgement call, and no
repo event corresponds to "a new model exists".

Two supporting changes. `download_checkpoint.py` now reads entity/project/run from the
environment instead of hardcoding them, so the run is a workflow input and the export
manifest records whichever run was used. And `propose_version.py` computes the next semver
by diffing the fresh manifest against the published registry.

The one thing that needed a decision: `releases.json` recorded the model's I/O but *not* its
preprocessing, so "major = preprocessing or I/O changed" was only half answerable from the
registry. Added `contractHash` to the release entry — sha256 over the whole preprocess block,
the I/O, and the embedding dim, i.e. everything a browser must implement. Opset is
deliberately outside it (it changes what can *load* the file, not what the runtime
implements) and is flagged for a human instead. Verified against the real export: identical
bytes → "none"; doctored baselines → minor / major / flagged as expected.

Model 1.0.0 predates the field, so the first dispatch will print the hash for its own bytes
(`sha256:3d8321…`) to backfill; until that lands, a new export against it comes out `flagged`
rather than guessed. Also added `web/export/uv.lock` and CI installs with `--frozen`, so the
same bundle builds in six months.

Still not built: the saccadejs fixture tests, the held-out eval gate (`v1_eval_frame0` lives
on HPC `/work`, unreachable from a runner), and the auto-PR (needs a cross-repo token).

## 2026-09-09 08:18 EDT — ARCHITECTURE.md rewritten to match what got built; retention decided (keep every version)

`web/ARCHITECTURE.md` had drifted badly in both directions: it still named Hugging Face as the
source of truth, and still listed as "open" several things that now exist. Rewrote it around
what is actually true — 218 lines down to 159, and no longer a proposal.

What it now records: the two-repo split and that the boundary is the artifact rather than an
import; the three version numbers; the three homes (versioned Pages URL live, npm pending,
`assets.modelUrl` for self-hosting); the provenance column in experiment data and why a version
string there means the bytes were verified; and the promotion pipeline as the one big open
piece, still blocked on `WANDB_API_KEY`. Hugging Face is written up as considered-and-rejected
with the reason, so it does not get re-proposed. Zenodo keeps a narrow role — an archived copy
for a version withdrawn from the site — which the retention decision makes moot for now.

**Retention decided: every published version stays served.** 20 MB per version on GitHub's own
infrastructure is not worth optimising against a URL that has to keep resolving. This turned
out to need a code change, not just a policy sentence: `copy-model.mjs` staged only the newest
release, so the first time a second version existed, 1.0.0's URL would have 404ed silently. It
now stages every served release and fails the build if one has lost its file. Verified by
simulating a two-release registry (both stage; deleting a served file fails with the intended
message), then restoring.

Layout that falls out of it: the current model stays at `models/eye_embedding.onnx`, because
`assets.ts`'s `packageModelUrl()` resolves that exact path for bundler users; superseded
versions move to `models/<version>/`, with each release's `file` field naming where its bytes
live. A release means moving the outgoing model down and updating two `file` fields — and
forgetting fails loudly rather than quietly dropping a version off the site.

Also verified while rewriting: nothing yet enforces the `contract` number at load time. The
registry carries it and identity resolution reads it, but the library will still happily load
a model whose contract it does not implement. Listed as open item 2; cheap, and worth doing
before a second contract exists.

## 2026-09-08 18:41 EDT — CI, at last: four jobs on every push, and the backend tests actually run

The repo had no CI and one test file that had never executed (pytest was in neither venv).
`.github/workflows/ci.yml` now runs on pushes to `main`, PRs, and `workflow_dispatch`:

| job | what |
|---|---|
| `guard` | the dependency-free half of `test_keras_backend.py`; seconds, no heavy install |
| `python` | `pip install -e ".[dev]"` + full pytest + a build asserting 5,198,273 params |
| `rig-panel` | typecheck, 54 vitest tests, build |
| `parity` | typecheck, build |

Two things had to change to make it real:

- **A `dev` extra (`pytest`, `jax`).** Two of the three backend tests skip when jax is not
  importable — which was every environment we actually had, so the test guarding the 360x
  slowdown had never asserted anything. With the extra installed the suite is **3 passed**
  rather than 1 passed / 2 skipped. jax stays out of the hard dependency list on purpose:
  training installs `jax[tpu]` on the VM and eval exports `KERAS_BACKEND=tensorflow`.
- **`web/export/parity/src/model.ts` no longer imports the generated manifest.** It was doing
  a *build-time* `import manifest from "./generated/export_manifest.json"` through a symlink
  into the gitignored `out/`, so the page could not typecheck in a fresh clone — CI would have
  failed at checkout. The manifest only supplied two strings that already had hardcoded
  fallbacks matching `export_onnx.py`'s `IN_NAME`/`OUT_NAME`, so they are constants now. The
  `bindNames()` fallback (silently rebinding to `session.inputNames[0]` on a mismatch) became
  `assertNames()`, which throws: on a *parity* page, quietly accepting whatever the model
  exposes hides the exact discrepancy the page exists to find. First piece of the
  `web/ARCHITECTURE.md` drift fix to actually land.

The `python` job installs the package rather than setting `PYTHONPATH`, so an
imported-but-undeclared dependency fails there — the failure mode that let
`experiment_utils.py`'s `sklearn` import sit dead for months.

All four jobs were run locally before committing: 3 pytest passed, 54 vitest passed, both web
projects typecheck and build, and `npm ci` reproduces from both committed lockfiles.

**Follow-up (18:46): the `parity` job failed on the first real run** — and the local check had
missed it. `public/models` was a committed symlink to `../../out/models`, which is gitignored,
so it dangles in a fresh clone; vite's `prepareOutDir` does a `statSync` across `publicDir` and
dies on it (`ENOENT ... /public/models`). Locally `out/models` exists, so the build passed. The
fix stops tracking symlinks that point into gitignored territory: `scripts/link-assets.mjs`
creates `../out/models` and both symlinks, wired to `predev`/`prebuild`, and `public/` is now
gitignored. A build before any export now succeeds with an empty `models/` — the page 404s at
runtime until you export, which is the honest behaviour. Verified this time in an actual
`git clone` of the repo with the pending changes applied, not in the working tree.

Still not built: the **publisher** workflow from `web/ARCHITECTURE.md` (linux/x86 export,
`check_parity_ort.py`, fixture gates, model release). It needs `WANDB_API_KEY` as a repository
secret first.

## 2026-09-08 18:26 EDT — R&D cleanup: 574 dead lines out of et_util, figures uncommitted, configs/ folded away

Housekeeping pass over the non-web side, after the saccade.js split closed.

- **Deleted three dead modules** (574 lines, zero references anywhere):
  `embedding_preprocessing.py` (an older TF-graph eye-alignment approach),
  `experiment_utils.py` and `model_layers.py` (a siamese/triplet experiment; the latter was
  imported only by the former). `experiment_utils` also imported `sklearn`, which is not in
  `pyproject.toml` — it would have raised on a clean install, which is how long it had been
  dead. Verified after: `create_masked_model(TrainConfig())` still builds at 5,198,273 params
  (the count `download_checkpoint.py` asserts), and every `analysis/*.py` still resolves its
  `et_util` imports.
- **`et_util/__init__.py` trimmed** to only what Keras must have registered
  (`custom_layers`, `custom_loss`). The four `inference_utils` re-exports are gone — nothing
  used the package-level names, the one notebook that wants them imports the submodule
  directly, and they were being paid for on every `python -m et_util.train`.
- **Version drift fixed.** `pyproject.toml` said 0.1.0 while `et_util.__version__` said 0.3.0.
  `pyproject` now declares `dynamic = ["version"]` with `[tool.hatch.version]` reading
  `et_util/__init__.py`, so there is one source of truth. `importlib.metadata.version(
  "eye-tracking")` and `et_util.__version__` now both report 0.3.0.
- **Figures uncommitted.** Dropped the 9 rendered PNGs from `analysis/figures/` (5.04 MB, of
  which `pose_features_overlay.png` was 3.1 MB) and gitignored `analysis/figures/*.png`; each
  traces to a producing script that takes `--fig-dir`, and git history keeps the images. Kept
  `pose_token_architecture.svg` (8 KB, hand-made, no producing script) and
  `ctr_ridge_divergence.json` (numeric results, not a figure).
- **`configs/` deleted.** It held only a README whose three "accuracy experiments discussed"
  had all since shipped (`ridge_center_targets`, `pose_token`) or moved to
  `sweeps/sweep.yaml`, and whose CLI example still quoted the pre-2026-06-10
  `--embedding-dim 200` default. Nothing needed rescuing into `TODO.md`.
- **Colab scripts rehomed.** `colab_bootstrap.sh` (TPU VM bootstrap) and
  `run_multiframe_screen_colab.sh` (a training screen) moved `analysis/` → `training/`; they
  were training infra sitting in the analysis directory.
- `CLAUDE.md` now lists the actual `et_util` module set — the old text omitted
  `frames_data.py`, which `config.py` points at as the current data path.

Still open: there is no CI, and `pytest` is in neither venv, so `tests/test_keras_backend.py`
still cannot run.

## 2026-09-08 16:09 EDT — Repo split finished: `web/demo` deleted, rig panel and parity page rehomed, export repointed

The browser runtime moved to `jspsych/saccadejs` some time ago (`packages/core` is a port of
`web/demo/src` at `a6609ab`); this closes the loop on our side.

**Merged `web-demo` into `main` first.** The branch was 18 commits ahead and *unpushed* — and
only 3 of those were web work. The other 15 were R&D that had never reached `main`, including
the `KERAS_BACKEND` fix in `et_util/__init__.py` (the 109 s/step vs 0.3 s/step bug), LR warmup
in steps rather than epochs, the contrastive aux loss, `--init-weights`, and
`tests/test_keras_backend.py` — the only test file in the repo. `main` is now 23 ahead of
`origin/main` and still unpushed.

**`web/demo` deleted** (~11k lines). Audited what would be lost first; three things had no
counterpart in saccadejs and were salvaged rather than dropped:

- `model_inputs.npy` / `model_embeddings.npy` → `web/export/fixtures/` (tracked). These are
  what `check_parity_ort.py` reads; deleting the demo would have broken the release gate.
- `parity.ts` + `parity.html` → `web/export/parity/`, its own small Vite app. It is the only
  check of onnxruntime-web on **WebGPU**; `check_parity_ort.py` covers CPU only.
- `rig.ts`, `rigAnalysis.ts`, `timing.ts`, `loopback.ts` + their tests → `web/rig/panel/`,
  beside the RP2040 firmware they drive. Without this the firmware had no UI.

`crop.ts` was checked before deleting: it differed from the saccadejs copy by 48 lines of
pure Prettier reflow (printWidth 80 → 100), algorithm byte-identical. The preproc fixtures are
fully mirrored there too, so nothing was lost.

**Export paths repointed.** Outputs now land in `web/export/out/{models,generated}`
(gitignored) instead of writing into the demo tree; `make_fixtures.py` and
`make_preproc_fixtures.py` grew `--out` so fixtures can be written straight into
`saccadejs/packages/core/test/fixtures/`. `web/export/README.md` has the table of what to
copy across. The parity app reaches the outputs through relative symlinks.

**Verified:** rig panel 54/54 tests pass and typechecks; parity app typechecks and builds;
`check_parity_ort.py` passes end-to-end on the new paths (max-abs 1.311e-06 < 1e-4). The
duplicated `loopback.ts` / `model.ts` / `types.ts` carry provenance headers naming the
saccadejs originals.

Docs updated: `CLAUDE.md` project shape, `README.md` layout, `TODO.md` (dropped the
jsPsych-extension item — shipped in saccadejs), `web/ARCHITECTURE.md` (marked partly
implemented: `assets.ts` already does the resolution tiers; hashes, contract check, model
identity in the data, and the CI publisher remain open).

Loose end: `pytest` is in neither venv, so `tests/test_keras_backend.py` cannot actually run.
There is still no CI in this repo.

## 2026-09-08 15:46 EDT — Model versioning/deployment design for saccade.js (`web/ARCHITECTURE.md`)

Wrote `web/ARCHITECTURE.md`: the target architecture for shipping the embedding model to
browsers. Design only, nothing implemented yet. Core decisions:

- **The release unit is an immutable directory, not a file** — `eye-embedding/<semver>/`
  holding `model.json` (the manifest, with a sha256 for every file), `model.onnx`, and
  `cal_weights.json`. `export_manifest.json` already carries the right content (preprocess
  contract, I/O, ridge defaults, W&B provenance); it just needs `id`/`version`/`contract`/
  hashes/`eval` and to live *beside* the weights.
- **Three version numbers**: library semver, model semver (MINOR = new weights same
  contract, MAJOR = contract change), and an integer `contract` the runtime declares support
  for. Weights swaps safe by construction; contract changes fail loudly.
- **Found a real drift bug**: `model.ts` imports the manifest at *build* time while fetching
  the weights at *runtime* from a gitignored, locally-regenerated `.onnx` — the two can
  disagree undetected, and the `?? "eye_image"` fallback plus `bindNames()` reassigning to
  `inputNames[0]` hide the mismatch rather than surfacing it. Fix: fetch `model.json` at
  runtime, keep only a pinned `{id, version, sha256, contract}` in the bundle, assert, verify
  bytes with SubtleCrypto, drop the fallbacks.
- **Distribution**: HF Hub as source of truth (immutable revisions + model card), optional
  npm package, and a `modelUrl` self-host escape hatch (required for IRB/offline labs).
  Never resolve `latest` at runtime — a mid-study model swap silently splits a dataset.
- **CI on linux/x86 is the only thing that may publish**, because the arm64 OpenCV KleidiCV
  HAL diverges on resize/CLAHE by up to 3 gray levels. Gates: `check_parity_ort.py`, the demo
  fixture tests, and a held-out cross-phase median written into `model.json`.
- **Provenance into the jsPsych data** (`saccade_version`, `model@version`, `model_sha256`,
  `contract`, `ep`) — the thing that makes a result reproducible years later.
- fp16/int8 (open `TODO.md` item) drops in as *variants within* a model version, each with
  its own hash and measured accuracy delta. No release-process change.

## 2026-09-05 18:07 EDT — The relay test's zero point was wrong: plateau instead of mean, plus a row-band rolling-shutter check

The 60 s Test 2 run (63 pulses, loopback L = 95) was being judged on the **mean** of the
first-lit-frame stamps, which is not a bias test: a stamp always lands in the frame *after*
the LED came on, so that mean sits about half a frame period high however right the
correction is. Replaced with the same intersection statistic the loopback's own edge
estimator uses, which makes the two directly subtractable:

- **`relayPlateau`**: one pulse seen in frame j says only that the light arrived after frame
  j−1 was stamped and no later than frame j's stamp, so the lags consistent with it are
  `(c_{j-1} − t_flip − D, c_j − t_flip − D]`. Scanning τ over 0–600 ms at 0.5 ms and taking
  the longest maximal-count run gives `{centerMs, lowMs, highMs, widthMs, consistent, n}`.
  The primary metric is now `plateauDeltaMs = plateau.centerMs − loopbackLagMs`; the verdict
  is OK when |Δ| ≤ 5 ms and ≥70 % of pulses are consistent. The mean and direct lag stay in
  the report, labelled as not a bias test.
- **`exposureEstimateMs`** from the jump sizes: full = 90th percentile of per-pulse jumps,
  and the fraction jumping ≥ 0.9·full is (P − E)/P.
- **`midExposureIdealMs` = plateau + E/2 + (mean − min display lag)**: the plateau lands on
  the frame boundary and on the *shortest* display lag, so that is the constant for
  mid-exposure alignment with an average flip.
- **`rowBands`**: four horizontal bands of the stored 160×120 gray plane, each run through
  `estimateLagEdges` against the run's own flips; the slope of band lag against band row,
  times the frame height, is the readout time, and the fit evaluated at the LED's row says
  how much of any screen-vs-LED gap is just rolling shutter. In `rig.rowBands`.

Synthetic checks: a continuous-integration camera (exposure = frame period) gives plateau
40.75 ms against a truth of 40.67 with 79/81 pulses consistent; a 20 ms top-to-bottom rolling
shutter comes back as 21 ms with bands 57 / 53 / 47 / 41.5 ms against a truth of
57.6 / 52.5 / 47.5 / 42.4.

**Caveat on the exposure estimator, measured not guessed.** On the synthetic camera whose
exposure *is* the full frame period, the full-jump fraction comes out at 0.247, giving
E ≈ 25 ms when the truth is 33.4. The reason: the largest-jump rule always picks the bigger
of the two split halves, so the observed jump is `max(u, P−u)`, which is ≥ half a full jump
even when nothing was caught whole — about 20 % of pulses clear the 0.9 threshold on that
alone. So the estimate is biased low by roughly 0.2·P (~7 ms at 30 fps), and the real run's
26 % → 25 ms is equally consistent with a camera integrating the whole frame.

## 2026-09-05 17:15 EDT — `timing.html` restructured into a basic mode with three test presets, plus an Advanced toggle

The page had grown into a wall of knobs that only its author could drive. It now opens in a
**basic** mode: one paragraph of what it measures and that it does not flicker, a rig
connect row (port, photodiode envelope, dropped lines if the firmware ever reports them), an
aim row that appears when a rig is connected (mirrored preview, redness difference view on by
default, Locate LED, the contrast readout), **three preset buttons**, a summary card per test,
and Copy JSON. Each preset writes the advanced controls and then starts an ordinary run, so
advanced mode always shows exactly the settings that produced a result:

- **Test 1 — Loopback** (no rig): sparse, 15 s, full contrast, gaps 500–1000, relay off.
- **Test 2 — Rig relay validation** (rig only): sparse 60 s, relay both edges, D 100–140 ms,
  100 ms pulses, train off.
- **Test 3 — Exposure sweep** (rig only): sparse 60 s, relay off, train on, gaps 600–1200 ms.

Summary cards lead with the verdict, then a handful of headline numbers (loopback lag with
plateau/D/halves; display lag, relay residual ± SE, direct vs loopback lag, LED contrast;
exposure window and width). The last result of each test stays on screen — in memory only,
nothing persisted. **Advanced ▸** (remembered in localStorage, default off) reveals the full
settings fieldset with its Start button, every rig control including relay/delay/pulse/train/
gaps/edge-reporting and the plane selector, the raw brightest-N numbers, the monospace
report, both plots, and the JSON textarea. Nothing is basic-only: advanced only ever adds.

Test 3 needed a verdict of its own — the relay verdict is meaningless when the relay is off —
so it is judged on whether the sweep resolved: ≥20 pulses, both f crossings bracketed, and a
window narrower than one camera frame.

## 2026-09-05 18:15 EDT — Rig validation closes: LED-based and screen-based lag agree to 3 ms; the relay test's zero point, exposure, display-lag bimodality, rolling shutter

Three 60 s Test-2 runs (relay both edges, D ∈ [100, 140] ms, ~63 matched pulses each).

**The relay test's zero point is not zero.** "First frame in which the LED appears" is on
average ~P/2 after the LED came on, so a correct loopback still yields a positive mean
residual (+12.2 ± 1.8 ms observed; expected in the low teens). Replaced by an
apples-to-apples statistic: the same intersection/plateau estimator the loopback uses,
applied to the LED pulses (interval `(c_{j−1} − flip − D, c_j − flip − D]` per pulse).

| run | LED plateau | loopback plateau | Δ | consistent |
|---|---|---|---|---|
| 17:50 | 87.0 | 95.0 | −8.0 | 53/62 |
| 18:12 | 98.0 | 95.0 | +3.0 | 53/63 |

Run-to-run scatter of a min-type statistic is several ms; **the two methods agree within
~5 ms**, which validates the loopback against an external clock *on the convention it
measures* (start of the exposure window, fastest display path).

**Rolling shutter** (loopback estimator on four horizontal bands of the stored frames):
bands 100 / 95 / 94 / 93 ms → readout ≈ 9 ms top-to-bottom, three lower bands within 2 ms;
not a material bias. **Display lag is bimodal**: 42 ms (next vsync) ~70 %, 59 ms (one
refresh late) ~30 %; mean 46–47. **Exposure** from the fraction of pulses caught whole in
one frame ≈ 25–29 ms, but that estimator is biased low by ~0.2 P (the largest-jump rule
never sees less than half a jump), so it is consistent with full-frame integration — Test 3
(phase sweep) is the unbiased measurement.

**Later that evening, with more light on the face (camera back at 30 fps, exposure
shorter):** Test 1 loopback **74.5 ms** (D = 1.00, halves 72/77); Test 2 **LED plateau
77.8 vs loopback 78.0 — Δ −0.3 ms, 51/54 consistent**; display lag 43 mean / 41 min
(single mode this time); rolling shutter ≈ 3 ms. Two Test-3 bugs found and fixed from
its first real runs: (1) the on/off-frame guard shrank each 100 ms pulse window by a
frame interval at *both* ends, leaving 14 ms at 23 fps and no usable lit frames —
captureTime is the *end* of the exposure so only the start needs guarding (now
`litWindow`/`isDark`); (2) the first-frame fraction was normalised by the two-frame total,
which caps it at 0.5 for a pulse longer than a frame (the next frame is fully lit) —
long pulses now normalise by the full LED level. The window is now a line fit through
the 0.15–0.85 part of the ramp extrapolated to f = 1 and f = 0 (the detector's noise
floor truncates the bottom of the ramp, so a 0.1 crossing need not exist).

**Conclusion (decided 2026-09-05 evening).** Test 2 is the result: the loopback, with
nothing but a webcam and a screen, matches an external clock + LED to < 1 ms on the same
statistic. Test 3's refinement (mid-exposure reference, ±E/2 ≤ ~17 ms) is below what
internet participants' hardware can support — exposure is set by each room's light and
cannot be measured without an LED — so it is *not* required. Field method: run the
15 s no-flicker loopback per session; the correction is its lag on the stated
convention (start of exposure, fastest display path), with a bounded residual of half
an exposure plus one refresh of display jitter. Possible later refinement, not needed to
use the method: estimate exposure from the loopback's own screen-edge jump fractions.

**Convention → gaze constant.** For mid-exposure alignment under mean display lag:
L_ideal = plateau + E/2 + (D_mean − D_min) ≈ 98 + 14.5 + 4 ≈ 117 ms on this hardware
(vs the raw loopback 95). The page now reports the plateau delta as the verdict, the
exposure estimate, the row-band readout, and the mid-exposure ideal.

## 2026-09-05 16:45 EDT — Rig-informed LED localisation (gray + redness ROI), jump-based frame pairing, and an aiming preview

The 60 s run's bimodal residuals (half near 0, half near +25…+39 ms) were a **detector**
failure, not a timing one: the brightest-24 detector was being won by face and lamp
highlights, leaving the LED only 7.6 counts above baseline (MAD 2.0), so the partly-exposed
first frame often missed the threshold and the *next* frame was taken — exactly one frame
late. Four changes:

1. **The rig knows when the LED was on, so let it say where the LED is.** Every camera frame
   now keeps its 160×120 gray *and* redness (`clamp(R − max(G,B))`) planes while the rig is
   connected (~70 MB for 60 s, never exported). After the run, frames are split into on/off
   sets from the `L` events and the sync fit (one frame interval of guard at both ends), the
   per-pixel mean(on) − mean(off) contrast map is computed **on both planes**, the top ≤16
   pixels of each become an ROI, and whichever ROI's signal separates better *in units of its
   own baseline MAD* is what `ledDetector` / `relayResiduals` / `exposureWindow` then run on.
   A red LED on a face wins on redness; one that blooms to white wins on gray. Falls back to
   brightest-24 on **redness** (not gray) when localisation fails, and says so in the report.
2. **First-lit frame by largest jump, not by threshold** — the same convention the loopback's
   edge estimator uses, so it does not care what fraction of the exposure was lit; the
   threshold survives only as a noise gate (the rise must clear 3 MADs). Each residual row now
   also carries `f`, the partial-exposure fraction of the chosen frame, and the jump size.
3. **Half-split tolerance scales with edges per half**: ≥15 edges → 8 ms, else the old 20 ms.
   The 60 s run reported halves 74.5 / 92.0 ms with 39 edges each and still said OK.
4. **An aiming preview in the panel** (works with no rig): mirrored 320×240 canvas, live gray
   and redness brightest-24 numbers plus the ROI mean, a difference view against a ~2 s EMA
   background so anything blinking pops, and **Locate LED** — three 400 ms pulses, a 10-ping
   sync, the same contrast code, then the ROI drawn as a box on the preview and reported in
   counts and MADs. The run re-localises from its own data and reports how far its centroid
   is from the panel's.

Two judgement calls. The ROI drops pixels below 25 % of the best pixel's contrast (a fixed
16 would pad a 9-pixel LED with noise pixels, diluting the signal and moving the reported
centroid), and rejects the whole ROI unless the best pixel stands 8 MADs above the middle of
the contrast map — the brightest pixel of a *noise* map still stands ~6 MADs above it purely
because there are 19200 of them, so without that check "no LED in frame" comes back as a
confident ROI on a random pixel. Tests: a 3×3 blob under brighter static highlights is found
with ≥10× the brightest-24 contrast; a red blob on a face with a white highlight selects the
redness plane; a scene with no LED returns null and falls back.

## 2026-09-05 15:32 EDT — Browser side of the timing rig: Web Serial panel, LED detector, and the three analyses in `timing.html`

Wired the rig into the loopback page. New `web/demo/src/rig.ts` (Web Serial client: local
typings + feature detection, line reader, `sync(20)` sequential pings, relay/train/pulse
setters) and `web/demo/src/rigAnalysis.ts` (DOM-free, unit-tested): `fitSync` (RTT-median
outlier rejection, centred least squares, rig µs → `performance.now()`), `displayLag`
(flip → photodiode edge, rising/falling separately, edges consumed once), `relayResiduals`
(the closed-loop test: `(captureTime − L) − flip − D`), `exposureWindow`. The camera
sampler gained a second 160×120 canvas whose `peak` is the mean of the 24 brightest pixels
via a 256-bin histogram — a 5 mm LED is invisible in the 64×48 whole-frame mean the
loopback uses. A run with the rig connected syncs before the countdown and after capture,
so the fit interpolates over the run instead of extrapolating; 24 synthetic tests pass
(a 300 ppm clock, 24/130 ms display lag, zero-mean residuals that shift to −15 ms when the
loopback lag is 15 ms too large, and a 12 ms exposure window recovered at −20 ms).

**First rig run (15 s sparse, relay rising, D ∈ [0, 40] ms, 8 pulses).** Sync: 39/40
pings, RTT median 0.50 ms, residual sd **0.03 ms**, slope −263 ppm (= the audio rig's
+262.5 ppm crystal offset, seen from the other side). **Display lag ≈ 47 ms** (rAF stamp →
50 % photodiode crossing): rising 48.4 ± 8.6, falling 45.5 ± 7.5 ms, range 41–59 — about
three 60 Hz refreshes with one refresh of jitter; rise and fall equal at the 50 % point, so
the LCD's slow 10–90 % fall does *not* bias the sparse estimator (both edge directions
stay). **Relay: residual 8.0 ± 13.2 ms (n 8)**; direct L via LED 100.0 vs loopback L 92.0
this run (up from 82 earlier — room lighting changed with the rig, camera exposure with
it). The 13.2 ms sd is physics, not noise: display jitter 8.6 ⊕ one-frame quantisation
9.6 = 12.9. SE of the mean 4.7 ms → inconclusive at n 8; verdict rewritten to test the mean
against its SE (needs ~45 pulses for ±2 ms). LED detector: only 15 counts above the
brightest-pixel baseline (MAD 0.9) — detectable, but the LED is not the brightest thing in
frame; aim it more directly for the exposure sweep.

Two judgement calls worth knowing about. (1) The exposure `f` is normalised by the
two-frame total **floored at the full LED level**: where consecutive exposures tile, the
two are identical, but a camera whose exposure is shorter than its frame period drops the
light that falls in the dead time and the plain two-frame ratio would be identically 1.
(2) The first-lit-frame threshold is 50 % of the LED level for the relay test but just
above the noise floor for the exposure sweep, which otherwise cannot see f below 0.5.

## 2026-09-05 15:13 EDT — Hardware timing rig: fresh RP2040 firmware (photodiode edge timestamper + LED relay), flashed and verified

Repurposed the `jspsych-audio-latency` rig (RP2040-Zero + TEMT6000 on GP26) with **new
firmware** written for this measurement rather than the audio rig's sample streamer:
`web/rig/firmware/eye_timing_rig/`. Added a red LED on GP15 via 120 Ω (12 mA drive).
The board is a *timestamping relay*: 20 kHz photodiode sampling from a hardware timer,
self-adjusting min/max envelope, 50 % crossings timestamped with `time_us_64()` (hysteresis
armed at 35/65 %), optional relay of each edge to the LED after a fixed or per-edge-random
delay via an **absolute-time** alarm (main-loop latency can't shift it), a free LED pulse
train, and sync pings. ASCII line protocol for Web Serial; see `web/rig/README.md`.

Bring-up: compiled 61 KB, uploaded over USB (`/dev/cu.usbmodem101`). `p` pulse → on/off
gap 100.009 ms; 3 s pulse → 3.000 s; LED visible after fixing polarity. First photodiode
reading pinned at ~4060/4095 = **floating input (ground had come loose)**, the same
symptom the audio-rig notes record; with GND reconnected: ~500 counts in room light,
~110 shaded, envelope tracked and emitted clean R/F edges.

Why the relay design: rig reports D (photons→LED); browser reports (LED-frame
captureTime − L) − flip time. Those must be equal — display and camera lag cancel by
construction, no cross-clock mapping, no physiology. Sync pings additionally give display
lag per edge direction (the audio rig measured this panel at 15 ms rise / ~130 ms fall — if
so, the loopback's falling edges are smeared over 4 camera frames and the sparse estimator
should use rising edges only). Browser panel (Web Serial + LED detector + analyses) in
progress.

## 2026-09-05 10:16 EDT — Temporal synchrony: tracker stamps samples with camera `captureTime`; new `timing.html` screen→webcam loopback measures display+camera lag

**Problem.** For an experiment the gaze series {x, y, t} must be aligned with the
experiment clock. The tracker was stamping frames with `performance.now()` at the moment
JS got scheduled, ignoring the `requestVideoFrameCallback` metadata, and the TTA ring mean
was implicitly stamped as the newest frame (a (N−1)/2-frame group delay, ~67 ms at N=5).

**Tracker changes** (`web/demo/src/tracker.ts`). Every `TrackerFrame` now carries a
`FrameTime`: `capture` (= rVFC `captureTime`, falling back to `receiveTime`, then the
callback `now`; `source` says which), `receive`, `presentedFrames` + `dropped` (gaps in
the counter), `callback`, `emit` (prediction available), and `meanCapture` (mean capture
time of the ring-buffer frames — the time the smoothed gaze actually refers to). All on
the `performance.now()` timeline jsPsych uses. HUD shows capture→emit latency, TTA group
delay, the clock source, and cumulative dropped frames. Rule for the recorder: **t =
capture**; pipeline latency is a QC metric, not part of t; smooth offline, non-causally.

**What `captureTime` cannot see** is display lag (rAF flip → photons) + camera lag
(photons → `captureTime`). Their sum is exactly the correction needed: for stimulus logged
at t_s and eye frame stamped t_g, true interval = (t_g − t_s) − (d_display + d_camera).

**`timing.html` / `src/timing.ts` / `src/loopback.ts`** (Opus subagent; reviewed).
Flashes the full screen in an m-sequence (Galois LFSR, order 7/8; the first Fibonacci
attempt was non-maximal at every order and aliased badly — test now asserts period and
ones-count) with each symbol held for K refreshes (default 5 → 83 ms, deliberately not an
integer number of 30-fps camera frames), logs each flip's rAF timestamp, samples mean
webcam luminance per rVFC frame with all four metadata clocks, and scans τ ∈ [0, 600] ms
at 1 ms for the Pearson-r peak (parabolic refinement). Reports the lag three ways
(captureTime / receiveTime / callback now) so the clock choice is visible, a half-split
stability check, camera and rAF interval stats, dropped frames, a verdict, r(τ) plot, and
a JSON export for cross-machine collection. Synthetic tests (30 fps, ±3 ms jitter, true lag
87 / 250 ms): median refined error 0.26 ms, worst ~11 ms over 40 seeds on 8 s of data →
default duration raised to 10 s, half-split tolerance 20 ms. 24 new tests; 51/52 pass.

**First real run (user's Mac, built-in camera, Chrome, 10 s, symbol 83.5 ms):**
captureTime lag **81.5 ms** (raw 81), peak r 0.825, halves 78.5 / 81.5 (Δ 3 ms), 298
samples, verdict OK. Callback-`now` variant 85.5 ms, i.e. `captureTime` is only ~4 ms
earlier than JS delivery — so on macOS Chrome it is a *delivery* stamp, not an exposure
stamp, and the 81 ms is camera pipeline (exposure + readout + ISP + USB) plus display
lag (1–2 refreshes + panel). `receiveTime` was absent (it is for remote WebRTC streams).
Camera dt 33.35 ± 0.84 ms (min 24.3, max 39.1 — ±8 ms stamp jitter), 0 dropped; rAF
16.69 ms with one 33 ms hiccup. Two page fixes from this run: video dims were read after
the stream was released (`0×0`), and the second-peak window (40 ms) sat inside the 83 ms
main lobe, so 2nd/peak = 0.60 was the lobe shoulder, not ambiguity — window is now
1.5× the symbol duration.

**No-flicker mode (now the default).** The lag information is in the edges, not the rate,
so `timing.html` gained a "sparse steps" mode: alternate levels held for a random
500–1000 ms (seeded, replayable) → ≤ 1 flash/s, a third of the WCAG 2.3.1 / Harding
3/s limit, ~20 edges in 15 s. Estimator: with sparse edges r(τ) is piecewise constant, so
`estimateLag` now also returns the **plateau** (contiguous lags with the peak r = the lags
consistent with every edge); its center is the estimate and its width the uncertainty.
Synthetic (30 fps, ±3 ms jitter, 15 s): center within 4 ms of truth over 5 seeds, plateau
< 34 ms. Verdict in sparse mode checks plateau width ≤ one camera frame instead of the
second-peak ratio (which is meaningless when r decays slowly). Optional 20 %/80 % gray
contrast. m-sequence mode kept (warning shown only when selected). 55/56 tests.

**Sparse mode, first real run: wrong (11 ms, halves 49 / 4, D verdict UNRELIABLE) — and
the failure is diagnostic.** All estimates fell *below* the 81.5 ms m-sequence value. That
is the signature of webcam auto-exposure: with 0.5–1 s holds the exposure loop re-adapts
inside each hold, so each edge produces a jump that decays back to baseline rather than a
sustained step. Against a pulse-shaped response the step-template correlation C(τ) is flat
for every τ below the true lag (the step still covers the whole pulse) and only falls off
above it, so the argmax wanders anywhere below truth. The m-sequence was immune because
its 83 ms holds are shorter than the exposure loop's time constant. Fix: an
**edge-difference estimator** (`estimateLagEdges`): D(τ) = Σ_edges dir_e · Δlum at the
first sample ≥ t_e + τ, normalised so 1.0 = every edge's full jump captured. Uses only the
jump, so nothing slower than one frame (exposure, gain, drift, subject motion) affects it;
piecewise constant → plateau as before. Tests: a simulated exposure loop (τ = 250 ms)
breaks the correlation estimator (asserted) while the edge estimator stays within 4 ms of
truth over 5 seeds and agrees with the correlation estimator on an m-sequence. The page
now uses it in sparse mode (verdict floor D ≥ 0.5) and records the track's `exposureMode`
and `frameRate`. 59/60 tests.

**Sparse mode re-run with the edge estimator: 82.0 ms, plateau ±0 (all 20 edges intersect
in one 1-ms bin), D = 1.000, halves 84 / 82, verdict OK** — vs **81.5 ms** from the
m-sequence on the same Mac/camera. Two methods with different stimuli, different
estimators and different failure modes agree to 0.5 ms, so the no-flicker sparse mode is
the field default and the m-sequence is lab-only. Callback-`now` variant 91.5 ±7.5: the
9.5 ms captureTime→callback gap here (4 ms in the m-seq run) is scheduling jitter, another
reason to stamp with captureTime. Track reports 640×480@30, `exposureMode` undefined
(macOS Chrome doesn't expose it). Note this is agreement *between two browser-side
methods*; neither can see whether `captureTime` sits at exposure or delivery — that still
needs the photodiode/LED lab check.

**Not yet done:** more
machines/webcams (is 81 ms typical? how wide is the spread?), and the lab
validation of the method itself with a photodiode + camera-visible LED (Bridges et al.
2020 style) to confirm macOS Chrome `captureTime` tracks exposure rather than delivery.

## 2026-09-05 09:05 EDT — Browser demo: `kwppnkuj` embedding model exported to ONNX and running live at 30 fps on WebGPU; 13-pt calibration → validation median 0.036 viewport units

New `web/` tree: `web/export/` (Python, own uv venv) and `web/demo/` (Vite + TS, no
framework). The deployment split is the one the analysis scripts already used offline:
the per-image `Eye_Image_Embedding` sub-model runs per frame; the weighted ridge
(`custom_layers.py:99-161`) is solved once at calibration in JS (Cholesky on 128×128)
and cached as a 128×2 kernel, so per-frame cost after embedding is one matmul. TTA is a
ring-buffer mean over the last N embeddings (linear ⇒ same as averaging predictions).

**Export.** `kwppnkuj` (W&B `eye-tracking-multiframe`; its weights file is
`multiframe_random_s42_ext70.weights.h5`, not `full_model.weights.h5`) loads with plain
`TrainConfig()`, 5,198,273 params. Keras 3.14 `model.export(format="onnx")` on the TF
backend (= tf2onnx 1.16.1, protobuf override needed) worked directly — no Einsum nodes
(attention became MatMul), one `Erfc`→`Erf` rewrite needed. Post-export surgery fuses
tf2onnx's LayerNorm chains (ORT's own fusion does not recognize TF's `x·s + (β − μ·s)`
form) into 25 opset-17 `LayerNormalization` + 4 `com.microsoft.Gelu`: 594 → 287 nodes,
20.6 MB, opset 17. Parity vs Keras: ORT-CPU 1.3e-6, browser WebGPU 1.2e-6, wasm 1.5e-6.

**Preprocessing port** (`crop.ts`) is bit-exact vs OpenCV on 21 fixture cases: crop
(landmarks 226/446/27/257/23/253, pad 0.01, `int()` truncation) → BT.601 fixed-point gray
→ cv2 INTER_LINEAR 144×36 → CLAHE(2.0, 8×8; 5×18 padded tiles, clip 1). **Gotcha:**
arm64 `opencv-python` ships the KleidiCV HAL whose `resize`/CLAHE differ from the
x86/Linux path the training data was built on by up to 3 gray levels — fixtures must be
generated from an x86 (Rosetta) OpenCV; `make_preproc_fixtures.py` has a CRC guard.
Browser landmarks come from the Tasks `FaceLandmarker` (training used legacy FaceMesh):
same indices, slightly different coordinates — an unquantified domain shift.

**Live (user's Mac, Chrome, 640×480 webcam, WebGPU EP, MediaPipe GPU delegate):**
- Serial loop: 15 fps (every other camera frame dropped), landmark 9.9 / crop 0.2 /
  embed 26.5 ms. Pipelining CPU capture of frame N+1 under the GPU embed of frame N →
  **30 fps (camera-capped)**, landmark 8.8 / crop 0.2 / embed 14.7 ms, ~44 ms latency.
- Isolated steady-state embed: 10.5 ms (594 nodes) → **7.0 ms** (287 nodes) on WebGPU;
  wasm ~116 ms either way (compute-bound). ORT graph capture needs GPU-resident I/O
  tensors, so it is off (open follow-up).
- 13-point calibration (1000 ms settle + 500 ms capture, λ=1, centered) then 9-point
  validation: **median 0.0359** viewport units (range 0.018–0.073, worst at the
  0.85/0.85 corner) — better than the within-phase offline numbers (~0.05).
  Single subject, n≈9/target; not a benchmark, a smoke test.

Run: `cd web/demo && npm install && npm run dev` (needs `public/models/eye_embedding.onnx`
from `web/export/README.md`; space = calibrate). Real-frame fixtures (`make_fixtures.py`)
still need an interactive terminal for the camera — one crop test remains skipped.

---

## 2026-08-08 09:14 EDT — REVERSAL: a REAL-trained model reads the synthetic images fine (median 0.125, real spread) → the synthetic collapse is an OPTIMISATION/INIT failure, not a data-content failure. Also: v4 collapses too (8/8), so the renderer's iris fix was not the cause; and same-seed TPU runs are NOT deterministic

Branch `fix/warmup-steps-size-invariant`, Colab TPU v6e1, production path, gates passed
(jax backend confirmed, 154 ms/step, `150 steps/epoch x 20 epochs = 3000 steps; warmup 700`).

**1. Synthetic still collapses, now 8/8.** `simulated-eyes` shipped a v4 corpus fixing a
geometry bug that was corrupting ~58% of eye-frames (the iris disc was coplanar with the
n-gon the sclera boolean leaves behind, so Cycles tie-broke per ray). Seeds 42 and 43 —
both *proven-viable inits* — flat at 0.216–0.219 for all 20 epochs, eval median **0.2514
with a zero-width IQR**, p90 == median, n=230, byte-identical to the 7 collapsed runs of
08-06 (`worst=0.2553` cross-phase, `0.2544` within-phase). Both exited 0: clean completion,
not a crash. v4's losses are statistically indistinguishable from v2's on the same seeds.
**So the renderer defect was not the cause.** Real defect, correct fix, zero effect on
trainability.

**2. THE RESULT — train on REAL, evaluate that same model on BOTH eval sets.**

| seed | real eval (control) | synthetic v4 `eval_phase` |
|---|---|---|
| 42 | median 0.0857, IQR [0.0526, 0.1389], p90 0.2204, n=230 | median **0.1250**, IQR [0.0978, 0.1660], p90 0.2148, n=134 |
| 43 | median 0.0813, IQR [0.0554, 0.1362], p90 0.2195, n=230 | median **0.1258**, IQR [0.0964, 0.1643], p90 0.2097, n=134 |

- **The synthetic eval is non-degenerate.** IQR width ~0.07, median ~0.125, nowhere near the
  0.2514 attractor. The run's n=134 agrees with the v4 eval set's own independently measured
  retention (134 of 150 survive the `cal_phase=1` completeness filter) — a consistency check
  between run and set, **not** a prediction that was made and met. Training arm was
  healthy (0.2003 → 0.1430 → 0.1247 → 0.0873 → 0.0654 → 0.0562) and the controls landed at
  0.0857/0.0813 against an expected 0.0832/0.0815 — an ordinary good model, not a lucky one.
- **Therefore the synthetic images DO carry gaze signal this network can read, and the
  collapse is an OPTIMISATION / INITIALISATION problem, not a data-content problem.** The
  criterion was written down **before** the numbers arrived; this is not a post-hoc reading.
- **Synthetic is ~1.5x worse than real — a real but MODERATE domain gap**, not a wall.
- Caveat: the synthetic eval set inherits every corpus-uniform render defect, so 0.125 is
  consistent with both a domain gap and an intermittently degraded signal, and this
  experiment cannot separate them.

**3. METHODOLOGY — same-seed, same-data TPU runs on this harness are NOT deterministic.**
Two runs with identical seed and identical training data, differing **only** in
`--eval-tfrecord-dir`, diverged: epoch-2 loss **0.1430 vs 0.1489**, different final weights.
Anyone building a 2x2 by re-running with a fixed seed is comparing two different models and
will call it an eval-set effect.

Because of that the 2x2 above is **one trained model per seed, evaluated twice** via an
eval-only path (standalone script in scratch, no repo files touched; `create_masked_model(cfg)`
+ `load_weights` + `train.evaluate()` with `TrainConfig` defaults). **Validated, not
asserted:** seed 42's synthetic number was obtained two independent ways — a fresh training
run (0.1249) and eval-only from a different model's weights (0.1250) — agreeing to four
decimals.

**Eval/train sets (GCS):** `gs://webcam-jspsych-frames/synthetic/v4/train_frames/` (750
subjects) and `gs://webcam-jspsych-frames/synthetic/v4/eval_phase/` (150 subjects, **134
retained** at `cal_phase=1`). **WARNING: the train prefix CONTAINS the eval subjects** — any
synthetic *training* run must restrict to seeds **000–599**, or it trains on its own eval set.
Note the v2 build of the same 150 seeds retained **136**, not 134, and both are correct: the
filter depends on which frames survived conversion, and MediaPipe drops different frames on
different renders (v2 converted 41,622 frames from these seeds, v4 41,662). A dozen subjects
sit 1–5 calibration frames from the boundary in any given corpus, so a 2-subject swing is
expected. **Measure the retention of the set you are evaluating on; don't carry it across.**

**Next:** stop hunting image defects for this. The live question is why training from scratch
on synthetic cannot escape the constant-prediction plateau when the images are demonstrably
readable. Directions: warm-start from a real-trained checkpoint (fine-tune on synthetic
000–599, with a real control arm — **IN FLIGHT, unresolved as of this entry**), mixed
real+synthetic corpora, and optimiser-side changes (LR schedule, warmup, init). Note warmup
in steps was already made size-invariant and did not rescue it.

---

## 2026-08-06 19:04 EDT — The harness DOES train today (real baseline reproduced, 1/3 collapse) and the synthetic corpus collapses 6/6 at matched size/schedule/eval → the failure is DATA-LINKED, not seed luck. Also: KERAS_BACKEND was never being set, costing ~360x

Branch `fix/warmup-steps-size-invariant`, Colab TPU v6e1, production path
(`--frames-tfrecord-dir` + plain `v1_eval_frame0`, no `--pose-token`), `frame_mode=first`,
20 epochs. Synthetic is `simulated-eyes` v2 repacked into the candidate schema
(`simeyes/export/repack_frames.py`), 750 subjects, one file per subject.

| train set | subjects | seeds | cross-phase medians |
|---|---|---|---|
| real `v1_train_frames` | 1747 | 42/43/44 | 0.0764 / 0.0752 / **COLLAPSED** |
| real subset (first 750 files by sort) | 750 | 42/43/44 | 0.0832 / 0.0815 / **COLLAPSED** |
| synthetic v2 repacked | 750 | 42–47 | **COLLAPSED ×6** |

- **Phase 0 passes.** The real 1747 baseline reproduces the promoted ~0.0775 reference, and
  its 1/3 collapse rate is the historical base rate. Whatever is wrong is not the harness.
- **Dataset size is not the blocker.** Cutting real to 750 costs ~0.006 median and trains
  fine. The 750-subject scale is viable.
- **6/6 on synthetic is not seed luck.** P(6/6) = (1/3)^6 = 0.0014 under the base rate.
  Stronger than that: seeds 42 and 43 are *proven-viable inits* — each trained successfully
  twice on real data — and both died on synthetic. The failure is data-linked.
- **0.2514 IS NOT A SYNTHETIC PERFORMANCE NUMBER.** It is the constant-prediction attractor.
  All 7 collapsed runs across both corpora emit a **byte-identical** eval block
  (mean/median/IQR/p90 all 0.2514, cross-phase worst 0.2553, within-phase worst 0.2544,
  n=230/232). One degenerate solution, not a family of poor fits. **No synthetic transfer
  number exists yet.** If you find 0.2514 quoted anywhere as a result, it is this.
- **Health-check caveat:** "loss clearly below 0.22 by epoch 2" needs margin — a collapsed
  run sits at **0.2171**, numerically below 0.22. Use the **zero-width eval IQR**; it is the
  unambiguous discriminator.

**BUG FOUND AND FIXED — `KERAS_BACKEND` was dead code (`3bf3819`, doc follow-up `e47cd32`).**
`et_util/__init__.py` imports keras (via `custom_layers`) at package init, *before*
`train.py`'s `os.environ.setdefault("KERAS_BACKEND", "jax")` ever runs. Under
`python -m et_util.train` the setdefault therefore did nothing and keras silently used the
**tensorflow** backend. Measured on a Colab TPU: **109 s/step (TF-CPU) vs 302 ms/step
(jax-TPU), ~360x.** Fix moves the setdefault to the top of `__init__.py`. Three jax analysis
scripts had the same latent bug and are cured by it. Regression tests added under `tests/`
— the repo previously had **no test infrastructure at all**.
Detection tell: a healthy jax run prints `[train] jax backend: flash/splash attention disabled`.

**Colab CLI operational notes (cost real time today):**
- Sessions are **not** time-limited at 60 min (24 h cap), but any `colab exec` / `sessions`
  call that hits 404/401 prunes local state → the keep-alive daemon exits → the VM is
  reclaimed. Sessions died at exactly **T+60:08** because the ADC token minted at allocation
  expires at 59 min and the next poll 401'd. **Pull all results before T+58**, or use W&B.
- `colab download` works while the kernel is busy — use it.
- Bootstrap: fetching only `v1_train_frames` + `v1_eval_frame0` (8.7 GB, skipping the
  `_pose` dirs) and running pip **in parallel** cut setup from ~12 min to ~2 min.

**Next:** the question has moved from "does the corrected corpus transfer?" to "why can the
trainer get no purchase on this corpus at all?" Leading suspect is iris/sclera contrast —
see `simulated-eyes/HANDOFF.md` §3 for the measurement and the in-flight split experiment.

**SUPERSEDED 2026-08-08:** iris/sclera contrast is not the collapse mechanism, and neither is
any other image defect — a real-trained model reads synthetic at median 0.125,
non-degenerate. "Why can the trainer get no purchase on this corpus at all?" is the wrong
question; it can, from a good init. See the 2026-08-08 entry.

---

## 2026-08-06 10:53 EDT — Synthetic-corpus screen INVALIDATED by init-collapse: 5/6 runs dead incl. a REAL-data control; default seed 12604 collapses; short-epoch probes do NOT screen seeds

Ran the corrected synthetic corpus (`simulated-eyes` v2, 750 subjects) → real eval on
Colab TPU v6e1, defaults, `--no-wandb`. **Five of six runs init-collapsed**, so none of
them measured anything about the data.

| train set | seed | epochs | cross-phase | state |
|---|---|---|---|---|
| synthetic v2 (750) | 12604 (default) | 50 | 0.2514, IQR width 0 | collapsed |
| **real 200 → real 57** | 12604 | 10+ | 0.2514, IQR width 0 | **collapsed** |
| synthetic v2 | 42 | 3 | 0.2514, IQR width 0 | collapsed |
| synthetic v2 | 43 | 3 | 0.2494, IQR [0.2449, 0.2522] | alive |
| synthetic v2 | 43 | 50 | 0.2514, IQR width 0 | collapsed |

- **`config.py` ships `random_seed = 12604`, and it collapses on this config.** Anyone
  running defaults gets a dead run. Three of mine did before I noticed they shared a seed.
- **The real-data control collapses too** — real images, real labels, same harness. So a
  flatline here is never evidence about synthetic data. It also retro-invalidates the
  premise of the whole `simulated-eyes` defect investigation, whose stated motivation was
  "loss 0.216 flat, eval degenerate 0.2514" — the collapse signature.
- **A short-epoch probe does NOT screen seeds.** Seed 43 survived 3 epochs and died at 50.
  LR is cosine-decayed over `train_epochs - warmup_epochs`, so a 3-epoch run has a 1-epoch
  decay window vs 48 — a different trajectory from step 0, not a prefix. Screen at the
  epoch count you intend to run.
- **Collapse rate was 5/6, well above the documented ~1/3**, with `clipnorm=1.0` already
  on. Biggest difference from the historically healthy runs is subject count: ~1,747 real
  then vs 750 synthetic / 200 real here → fewer steps/epoch → different effective LR and
  warmup. Untested hypothesis, cheap to test, first thing to try.
- **TPU splash-attention errors are a RED HERRING** — ~60/run of
  `ValueError: q_block_size=128 should divide q_seq_len=72` from
  `keras/src/backend/jax/nn.py::wrap_flash_attention`. Caught, falls back, and appears
  identically under the pinned known-good stack (tf 2.19.1 / keras 3.13.2 / jax 0.7.2 /
  np 2.0.2). Not the cause of anything; don't chase it again.
- **Read the IQR width before the median.** Collapsed = `IQR=[0.2514, 0.2514]`, `p90 ==
  median`. A merely-bad model has spread.
- **Colab image is not stable between allocations**: two VMs shipped working tf 2.20.0, a
  third was the LEAN image with no tensorflow. `requirements-colab.txt` targets the lean
  one — probe `import tensorflow` first and install accordingly, or you layer an
  ABI-mismatched second TF build.
- **Lost one run to exec-polling** exactly as the four rules warn: a transient 404 pruned
  local state → keep-alive died → VM released, and the results file lived only on
  `/content`. Fix used afterwards: run training in the foreground of a single `colab exec`
  and `tee` stdout to a LOCAL file, so a drop costs the tail not the result.

**FOLLOW-UP (same session): a mechanism for the elevated collapse rate.** `train.py`
builds the schedule as `steps_per_epoch = n_individuals // batch_size` and
`warmup_steps = steps_per_epoch * warmup_epochs`, so **warmup length in optimizer steps
scales with dataset size**:

| training set | subjects | steps/epoch | warmup steps |
|---|---|---|---|
| historical healthy runs | ~1,747 | 349 | **698** |
| synthetic v2 | 750 | 150 | 300 |
| real control | 200 | 40 | **80** |

Warmup is what protects against early-training blowup, and init-collapse is an
early-training phenomenon. My runs got 2.3-8.7x less of it than the runs that trained
reliably, and the one with the least (200-subject real control) collapsed hardest. This
predicts collapse propensity is driven by DATASET SIZE, not by synthetic-vs-real.

**It also confounds the headline comparison.** Synthetic (750) vs real (1,747) differ 2.3x
in warmup steps, so "synthetic transfers worse" and "the synthetic arm got half the
warmup" are currently the same measurement.

Proposed test, ~1 h: hold data fixed (synthetic v2), `warmup_epochs` 2 vs 5 (5 restores
~750 warmup steps), 4 paired seeds per arm, classify by IQR width at epoch 10 while
running the true 50-epoch schedule. If longer warmup reduces the collapse rate, specify
warmup in STEPS not epochs, and equalise warmup steps across arms before comparing
corpora. UNTESTED as of this entry.

**Open, and untouched by all of this: does the corrected synthetic corpus transfer?** It
has never been measured on a live init. Next: screen 4-6 seeds at full length, test the
subject-count hypothesis against the ~1,747-subject real set, then run the comparison.

---

## 2026-06-20 17:46 EDT — emb_dim {64,32,16} sweep: ALL three init-COLLAPSED at seed 42 → narrowing the embedding is init-fragile at the default LR (inconclusive on conditioning)

Followed the 06-19 diagnostic lead (embedding eff-dim ~4, so 128 looks oversized). Sweep
emb_dim ∈ {64,32,16}, seed 42, 20ep, uf2sh, aux=0, default LR 3e-4 + clipnorm 1.0
(W&B `eye-tracking-embdim-sweep`, Colab TPU). **All three init-collapsed** — train loss
flat ~0.217 from epoch 0, cross-phase median 0.2514 (the dead-seed plateau; p25=p75=p90).
The matched emb128 baseline at the SAME seed 42 trains fine (0.0743, `graceful-rain-3`).

- **Finding: narrowing the embedding destabilizes init at the default LR** — even with
  clipnorm=1.0. Not the ~⅓ random rate; it's 3/3 at the width that emb128 survives, so it's
  width-driven. The narrow `Dense(emb,tanh)→ridge` path falls into predict-the-prior.
- **Inconclusive on the actual conditioning question** — no narrow arm trained, so we have no
  read on whether smaller D conditions the ridge better. Disambiguation (emb32 @ seed 123)
  did NOT complete: the Colab exec channel went flaky and the launch never reached wandb.init
  (session later orphaned).
- **If revisited:** narrow widths need an init fix — lower LR (~1e-4) and/or warmup 2→5ep
  (NB: `warmup_epochs` isn't CLI-exposed yet), or a check-and-reroll seed loop — AND the
  emb128 baseline must be re-run at the same setting for a fair comparison. Given the
  contrastive thread was already a small/benign effect and this needs careful re-baselining,
  it's lower priority than the data-side levers.

## 2026-06-19 17:53 EDT — Why the contrastive win is small: the embedding is ALREADY ~3-4D & 0.87 gaze-correlated. Per-subject structure is affine realignment in a SHARED manifold, not high-D subspace selection; aux doesn't harm it (uniformity worry NOT confirmed)

HPC ridge-map-divergence diagnostic (`analysis/ridge_map_divergence.py`, SLURM emc
jobs 5551/5552 extract + 5553 compare, frozen embeddings of both 06-19 checkpoints
on eyedata24 two-phase, n=246). Tests the design hypothesis that the high-D
embedding is carved into per-subject subspaces and the aux loss enforces a harmful
uniform space.

| metric | baseline (aux=0) | aux=0.3 | Δ |
|--------|------|------|------|
| dist_corr_median | 0.872 | 0.911 | +0.039 |
| eff_dim_median (per subj) | 2.28 | 1.85 | −0.43 |
| eff_dim_pooled | 4.09 | 3.05 | −1.04 |
| map_specificity_median | 0.152 | 0.141 | −0.011 |
| per_subject_gain_median | 0.0552 | 0.0547 | −0.0005 |
| cross_persubj_median | 0.0692 | 0.0682 | −0.001 |

- **Premise reframed:** the 128-D embedding already lives on a **~3-4D manifold**
  (eff-dim 2.3/subj, 4.1 pooled) and is **already 0.87 gaze-correlated** in the
  baseline. It is NOT a high-D space partitioned per subject.
- **Personalization = per-subject AFFINE realignment in a SHARED manifold:** using
  another subject's map ~triples error (own 0.069 → other ~0.22; map_specificity
  0.15), and per-subject beats one global map by 0.055 — yet everyone shares the
  ~4D manifold. Matches the prior "cross-phase penalty is mapping-level" result.
- **Aux is benign, not harmful:** it raises dist_corr (+0.04, did its job) and
  tightens the manifold (eff-dim ↓), but per_subject_gain is unchanged (−0.0005)
  and map_specificity barely moves (−0.01); cross-phase slightly better (−0.001).
  So the **uniformity worry is not confirmed** — the loss isn't fighting per-subject
  structure, and the per-subject-friendly relaxation (triplet/rank) is solving a
  non-problem → DROPPED.
- **Why the win is small = little headroom** (baseline already ~3D + 0.87 corr), not
  a flexibility conflict. Implications: (a) the per-subject-friendly loss redesign is
  off the table; (b) emb_dim could likely drop well below 128 (eff-dim ~4) — a cheap
  conditioning win to screen; (c) the real lever stays the per-subject MAP / the
  cross-phase affine drift (mapping-level, head-pose-invisible), which the contrastive
  loss can't touch. (json: `analysis/figures/ctr_ridge_divergence.json`.)

## 2026-06-19 10:57 EDT — Contrastive embedding-distance aux loss (WebEyeTrack Stage-1): implemented + matched A/B = MILD POSITIVE (median −0.0016 @20ep, consistent but near noise; needs 2nd seed + weight sweep)

New `aux_contrastive_weight` knob + `ContrastiveEmbeddingDistanceLoss` layer
(`custom_layers.py`): identity passthrough on the per-image embedding tensor that
`add_loss`es the MSE between unit-mean-normalized pairwise EMBEDDING distances and
pairwise (16:9-weighted) GAZE-target distances. weight=0 ⇒ layer not inserted
(default model byte-identical, checkpoints load). Branch `contrastive-embedding-loss`.
**The `add_loss` path compiles + trains under JAX-jit on TPU** — validated end-to-end.

- **Matched A/B** (W&B `eye-tracking-contrastive-aux`, TPU v5e, seed 42, 20ep, current
  defaults train λ=2 + center + clipnorm 1.0, eval **uf2sh** n=211; only diff = aux loss):

  | epoch | baseline `graceful-rain-3` | aux=0.3 `easy-dragon-2` | Δ median |
  |-------|------|------|------|
  | 5  | 0.0896 | 0.0861 | −0.0035 |
  | 10 | 0.0825 | 0.0787 | −0.0038 |
  | 15 | 0.0759 | 0.0754 | −0.0005 |
  | 20 | **0.0743** | **0.0727** | **−0.0016** |

- **Verdict: mild positive, not yet conclusive.** Aux ≤ baseline at every checkpoint AND
  every percentile @20ep (mean 0.0945 vs 0.0981; p90 0.1899 vs 0.1992; worst 0.2905 vs
  0.3029) — a consistent whole-distribution shift, including the tail. BUT the final median
  gap (−0.0016, −2.2%) is only ~1.5× the ~0.001 seed noise. The gap is **largest mid-training
  (−0.004 @ ep10) and shrinks by ep20** → reads more like *faster convergence* than a higher
  ceiling; unclear if it survives to full convergence or the baseline fully catches up.
- **NEXT:** (1) 2nd seed (123/789) at weight 0.3 to clear the noise floor; (2) weight sweep
  {0.1, 1.0} — the mid-training lead hints a different weight may help more; (3) one 50ep
  arm to see if the gap persists or closes at convergence. Keep `aux_contrastive_weight=0`
  default until confirmed.
- **Infra:** bootstrap is now turnkey but 3 self-inflicted false starts cost time first —
  OSF flat-extract layout, a TF double-install (image TF 2.20 + pip tensorflow_cpu 2.21 →
  undefined-symbol crash), and the recurring TPU-lock (kernel `jax.devices()` grabs the chip,
  locks out the detached trainer). All fixed + documented in CLAUDE.md (headless rule #0 +
  Colab gotchas). NOTE: the local ADC token expires ~hourly and dies during long unattended
  polls — it killed the first VM's keep-alive and blocked the `colab stop` of the second; re-mint
  with `gcloud auth application-default login --scopes=…,colaboratory` (all 4 scopes, one line).

## 2026-06-17 13:30 EDT — Train-λ=2 win confirmed across 2 clean seeds; init-collapse caught → `clipnorm` default 0→1.0

Confirmation seeds for the 06-16 result (center ON, eval λ=2, train λ=0.1 vs 2.0):

| seed | train λ=0.1 | train λ=2.0 | Δ median |
|------|-------------|-------------|----------|
| 42   | 0.0828      | 0.0755      | −0.0073  |
| 43   | 0.0830      | 0.0757      | −0.0073  |
| 44   | 0.2514 ✗    | 0.2514 ✗    | dead seed |

- **Confirmed:** identical −0.0073 (−8.8%) median gap at both clean seeds, abs
  reproducing ±0.0002; mean agrees (λ=2 ≈0.100 vs λ=0.1 ≈0.104). New default
  (train λ=2 + center) = **0.0756 ± 0.0001**. Not a fluke.
- **Seed 44 init-collapsed** — *both* arms hit the identical 0.25137 (shared
  seed-44 init; median never moved off ep5), so it's init not λ. Discarded.
- **Fix: `clipnorm` default 0.0→1.0** (`config.py`) — the documented guard against
  the ~⅓ init-collapse rate, which was off. Costs ~nothing.
- Pending: eval-λ optimum sweep on the 4 good checkpoints (free, eval-only).

## 2026-06-16 21:15 EDT — Training at the eval-time ridge λ beats train-low by ~10% median (robust, NOT the predicted null); centering additive. Defaults updated: ridge λ 0.1→2.0, centering on

Two shared-seed (42) 2-arm screens on the v1 single-frame eval (`v1_eval_frame0`,
NO TTA), 20ep, eval λ=2. Full 2×2 {train λ} × {centering}, cross-phase **median**
(mean in parens):

|            | train λ=0.1     | train λ=2.0     |
|------------|-----------------|-----------------|
| center OFF | 0.0868 (0.1061) | 0.0769 (0.1002) |
| center ON  | 0.0828 (0.1044) | **0.0755 (0.1002)** |

- **Train-λ matched to the eval operating point is the big win:** B>A by −11.4%
  median (OFF) / −8.9% (ON). Effect (~0.01) ≫ ~0.001 seed noise. This **refutes
  the embedding-norm "absorption" prior** — the unnormalized backbone did NOT
  rescale away a high train-λ. Mechanism for a later look (cal-weight head? norm
  just doesn't grow in 20ep?).
- **Centering is a smaller, additive gain, larger at low λ:** A −4.6% median,
  B −1.9% (the λ=2 cell's ~0.0015 is only ~1.5× noise — marginal). Fits the
  mechanism: at low λ the under-regularized solve drifts to the corner more, so a
  center fallback fires more often.
- **Best cell = train λ=2 + centering = 0.0755 / 0.1002**, best on both metrics.
- **Defaults changed** (`config.py`): `ridge_regularization` 0.1→2.0 (eval inherits
  via `eval_ridge_regularization=None` → train & eval both λ=2); `ridge_center_targets`
  False→True (flipped earlier today, first-principles + this).
- Caveats: all single-seed; single-frame eval (TTA headroom untouched); the
  screen fixed eval λ=2, so this shows "train at your eval λ", not that λ=2 is the
  global optimum. Confirm seed on the best cell still wanted.
- Ops: ran headless via the no-poll / W&B-decoupled colab-CLI recipe (one
  setsid-detached driver, eval-every-5 → W&B, never exec-poll the live session);
  3 full screens completed without a session death (vs prior 100% kill rate).

## 2026-06-15 18:03 EDT — Ridge intercept resolved: a FIXED screen-center prior (predict offsets from 0.5) beats none/unpenalized/penalized-R0; tail-grade, strict win at small n. Implemented as a knob; co-adaptation retrain queued

Follow-on to the λ>>1 corner-collapse (06-15 12:56). The deployed per-subject
ridge has no intercept, so heavy shrinkage / few cal points collapse predictions
to the screen CORNER (0,0). Tested three fixes post-hoc on frozen embeddings
(150ep pose-token, n=244, FPS-spread cal points), median / p90 / worst @λ=1:

|  n | none          | unpenalized   | **center (0.5)** |
|----|---------------|---------------|------------------|
|  5 | .082/.245/.384| .082/.247/.344| **.080/.249/.340** |
| 20 | .062/.202/.328| .063/.206/.318| **.063/.203/.315** |
| 50 | .057/.192/.297| .057/.194/.291| **.057/.193/.292** |

- **Center-offset (solve on `coords-0.5`, add back) is the best variant.** It
  matches/beats on median (best at small n, where estimating an offset from few
  points is noisy), consistently shaves the worst subject, and costs ZERO DOF
  (fixed prior mean, not an estimated intercept).
- **This explains the 06-07 R0 NULL: that was the PENALIZED intercept** (λ on the
  bias too), a dead wash in every cell here -- it also collapses to the corner.
  Only the unpenalized / fixed-center form has the graceful centroid/center
  fallback. So R0 wasn't the wrong idea, it was the wrong parameterization.
- **The real payoff: a fixed-center prior makes λ a SAFE tail knob.** Over-reg now
  lands at center, not the corner, so you can regularize harder for the tail
  without the median falling apart. n=5 center+λ3 = **.078 / .229 / .312** --
  STRICTLY better than the n=5 baseline (.082 / .245 / .384) on all three. n=20
  center+λ3 is median-neutral, worst .328->.291. Small n is the *deployment*
  regime (the 12:56 knee says real users calibrate ~10-13 pts), so this is a free
  win right where it counts. (λ much past ~3-10 still over-shrinks the median.)
- Verdict: tail/robustness-grade, not a headline-median mover (nothing has been
  since tokpool1) -- same bucket as pose/QA-gate/cal-pose. But no-downside and
  free on any checkpoint.

**Implemented (this commit):** `MaskedWeightedRidgeRegressionLayer(center_targets=)`
+ `TrainConfig.ridge_center_targets` (training, lets the backbone co-adapt) and
`eval_ridge_center_targets` (eval/deploy override, free post-hoc -- mirrors the
`eval_ridge_regularization` λ knob, threaded through `_build_eval_model` + CLI).
Layer unit-tested against the numpy ref (matches to 1e-6; λ=1e6 collapses to
[0.5,0.5] not [0,0]; serializes). **Recommend deploy: eval center_targets=ON,
λ≈1 (and consider λ≈3 for the few-point calibration UX).**

**Queued:** 20ep shared-seed (42) screen, `ridge_center_targets` ON vs OFF, to
test whether co-adaptation adds anything beyond the free post-hoc gain (prior:
~NULL, like every other co-adaptation screen). Running on Colab TPU via the new
`google-colab-cli`.

---

## 2026-06-15 12:56 EDT — Calibration-COUNT × TTA curve (150ep pose-token model): diminishing returns, knee at ~12-16 points; the deployed 20-grid sits just past it; extra points mainly de-risk the tail

`analysis/cal_count_tta_sweep.py` + `plot_cal_count_tta.py` (figure
`analysis/figures/cal_count_tta_posetoken.png`). First per-frame embedding
extraction of the 150ep pose-token checkpoint (`xt104syi`, W&B project
`eye-tracking-multiframe`): `extract_v1_embeddings.py` gained `--pose-token`
(per-frame pose from stored landmarks → `[image, pose7]` through the
`PairTimeDistributed` submodel; raw pose9 kept in the npz). Artifact at
`/work/jdeleeuw/embeddings/v1_eyedata24_posetoken_xt104syi`.

Protocol: cal POOL = all phase-1 lattice points (frame 0, deployed recipe),
targets = all phase-2 trials with TTA (mean embedding over the trial's frames,
FIXED across all n), n cal points drawn by farthest-point sampling
(screen-covering, like a real calibration routine), 5 draws/subject averaged,
n=244 subjects (≥50 cal points). Per-subject mean cross-phase NED.

Median TTA error vs n (λ=1, the TTA-optimal solve):
  n=5 .081 → 10 .069 → 14 .065 → 20 .062 → 28 .061 → 40 .059 → 50 .058
- **Diminishing returns with a knee at ~12-16 points.** 5→14 buys −0.016
  median (~20%); 20→50 buys only −0.0045 (~7%) for 2.5× the points. ~10-12
  screen-covering points already capture most of the achievable accuracy.
- **The deployed 20-point grid sits just past the knee** — a well-chosen
  default (median 0.062 @λ1; close to the 06-11 confirmed 0.0601 on the 70ep
  arm-B model, the small gap = different checkpoint + FPS-over-lattice vs the
  exact 20-grid).
- **Extra points past the knee mainly de-risk the TAIL, not the median.** p90
  0.244→0.192 and worst-subject 0.38→0.31 keep falling out to n=50 while the
  median is flat — consistent with the recalibration-gate picture (hard
  subjects need more support; the central user is near-saturated by ~12).
- **λ=1 dominates λ=0.1 everywhere, most at low n** (.081 vs .081 at n=5 but
  the λ0.1 curve is noisier and worse through the mid-range; an under-determined
  small-n solve wants more shrinkage). Deploy TTA with λ≈1 regardless of n.
- Caveat: pose adds nothing to accuracy (NULL, 06-12), so this curve reads as
  representative of the architecture, not a pose effect. Spread (FPS) sampling;
  a random-location contrast is available via `--policy random`.

**Follow-up — solve-time λ beyond 1 (inline check, same artifact):** λ>>1 is
CATASTROPHIC, not a plateau. The per-subject ridge has no intercept (bias=False,
matching the deployed layer), so kernel=(XᵀX+λI)⁻¹Xᵀy → 0 as λ→∞ and every
prediction collapses to the screen CORNER (0,0), not the calibration centroid.
Median TTA NED by λ (n=20): .068(0.1) → **.062(1)** → .064(3) → .071(10) →
.094(30) → .158(100) → .379(1000), climbing toward the predict-(0,0) ceiling
(median 0.53). Optimum is a shallow basin at λ≈1-3; anything past ~3 falls off a
cliff. Two nuances: (a) optimal λ shifts with n — small/under-determined n=5
peaks at λ≈3 (wants more shrinkage), n=20/50 at λ≈1; (b) more cal points BUFFER
over-regularization — at λ=1000 n=50 (.287) degrades far less than n=5 (.479),
because XᵀX grows with row count so a fixed λ is effectively weaker. Confirms
λ≈1 as a safe deploy default (soft optimum), with no upside to raising it.

---

## 2026-06-12 14:25 EDT — QA-gated TTA: NULL. Plain all-frames embedding-mean is already optimal; the tail is not frame-quality-driven. Cal-frame "blink repair" actively hurts the worst subjects

`analysis/qa_gated_tta_sweep.py` (frozen `8lsofasl` embeddings, n=230, per-subject
npz artifact on HPC; per-subject losses at `/work/jdeleeuw/qa_gated_tta_sweep.npz`).
Arms vs the tta-all bar (mean embedding over all face-found target frames): EAR gates
at 0.5/0.65/0.75/0.85 × subject median, rel-sharpness gates, combined, prediction-median,
prediction-trim50, and a cal-side "first EAR-ok frame instead of frame 0" repair.

- **Every gated/robust arm is within noise of plain tta-all** at both λ: median deltas
  ±0.0007, p90 ±0.005, worst-subject essentially untouched (λ0.1: 0.4381 → best 0.4359;
  λ1: 0.2797 → 0.2772). The blink-band gate (EAR≥0.75) drops only 5.4% of frames — the
  mean over ~15-24 frames already dilutes them; there is nothing to rescue.
- **The tail is NOT bad-frame-driven.** Consistent with 06-11 (75% of error is bias;
  worst subject survives the pose gate too): tail subjects fail at the solve/mapping
  level, not the frame level. Tail work belongs to the recalibration-gate committee,
  not to inference-time frame filtering.
- **Cal-repair HURTS the worst subjects** (worst +0.047 @λ0.1, +0.068 @λ1, only 58-61/230
  improved on the tta-all arm): swapping cal frame 0 for a later EAR-ok frame trades a
  rare blink for fixation-drift exposure. Don't ship it.
- pred-median is the only arm never worse (λ0.1: med −0.0007, p90 −0.0009, 122/230) —
  inside noise, but it's free and blink-proof by construction; acceptable deployment
  alternative to the mean, not a required change.
- λ=1 remains the better solve-λ for TTA inference (median 0.0615 vs 0.0646), as on 06-11.

**Deployment implication: wire plain embedding-mean TTA (all face-found frames, λ≈1),
no QA gating needed on the target side. The QA bits stay valuable for the PRE-FLIGHT
gate (blink_frac ρ=+0.32), just not for frame filtering at inference.**

---

## 2026-06-12 13:01 EDT — Maximalist pose-token bet: NULL. 150ep trajectory plateaus at 0.0693-0.0698 from epoch ~80; not undertrained; co-adaptation can't extract correction-grade signal from pose either

The 150-ep pose-token run (W&B `xt104syi` feasible-field-8, TPU v5e, ~4.7 h): hybrid +
multi-frame sampling + the 7 verified pose features projected to a 73rd TOKEN attending
with the image tokens in all 4 transformer layers (+2,048 params; architecture diagram
`analysis/figures/pose_token_architecture.svg`). Single 150-ep cosine, clipnorm=1.0,
photometric-only aug (affine is FUNDAMENTALLY incompatible with pose conditioning — see
TODO note: pose-adjusted warps would train the pose correction toward zero), eval every
10 ep, checkpoints synced to W&B.

Cross-phase median trajectory (single-frame eval, n=230):
  ep10 .0846 → ep20 .0782 → ep30 .0760 → ep50 .0739 → ep60 .0717 → ep70 .0714 →
  ep80 .0698 → ep90-100 .0693 → flat .069-.070 → **ep150 final .0698**
  (within-phase plateaus .057-.058; p90 improves to ~.20; worst ~.30; no init collapse.)

- **Verdict: NULL.** At matched epochs the pose run ties arm-B-no-pose (.0714@70 vs
  .0713@70; .0698-.0693 plateau vs .0706@60 stitched) — all inside seed noise, and the
  run carries the pose+no-affine confound in pose's FAVOR if affine aug helps. The
  oracle prediction held through full backbone co-adaptation: landmark-visible pose
  carries no correction-grade signal, even when the network can reshape its features
  around it. Pose levers now closed at every level (warp / fusion oracle / attention
  oracle / co-adapted training).
- **Undertrained question settled (the run's second purpose):** plateau is unambiguous
  from ep ~80-90; 150 ep buys ~nothing over ~100. The "double dip" of the 06-11
  restart-stitched run was just convergence — the clean single cosine lands in the same
  .069-.071 regime. ~100ep single cosine ≈ the right budget for this architecture/data.
- Incidental: .0693-.0698 is the best single-frame trained number so far (vs .0706),
  most plausibly the longer/cleaner schedule, not pose. Weights + per-10ep checkpoints
  on the W&B run. TTA eval of this model would need per-frame pose (TODO note (c)) —
  given the null, not worth running; the pose-less recipe is the keeper.

**Strategic picture unchanged by one more null: external pretraining is the open lever;
TTA + gating (06-11) remain the banked wins.**

---

## 2026-06-11 21:34 EDT — Recalibration-gate simulation: "hold still during calibration" alone gets the served population to median 0.046-0.051

Simulated the product flow "too much head movement during calibration → recalibrate" on
the REAL model (70-ep arm-B + TTA inference, n=230): reject the top-X% by cal-pose
instability, look at the survivors. (Inline analysis joining the phase_shift_gate2 npz
with the eval_sf_vs_tta_ext70 CSVs.)

  reject  n    median   mean    p90     worst
    0%   230   0.0601  0.0863  0.1863  0.3425
   20%   184   0.0511  0.0717  0.1482  0.3425
   30%   161   0.0464  0.0650  0.1216  0.2525   (saturates past ~30%)

- **Near-optimal on the median**: vs the PERFECT gate (reject by true error) the medians
  almost coincide at every rejection level (e.g. 30%: 0.0464 vs 0.0456) — for the central
  user this single signal captures ~all the achievable compositional gain.
- **Misses the extreme tail**: the worst subject (0.34) survives to 30% — their failure
  isn't head motion. The gate needs to be a committee (pose instability + blink_frac +
  sharp_cv + post-cal cluster spread catch different failure modes).
- Stacked picture, no retraining anywhere: deployed single-frame 0.0717 → TTA 0.0601 →
  TTA + 20% recalibration gate **0.0511** (~29% total reduction for the served population).
- Caveats: simulates EXCLUSION not actual recalibration (re-runs would recover some
  users); percentile threshold is in-sample (one parameter, low overfit risk);
  implementable client-side in real time (landmarks already computed on-device).

An important step for the pre-flight-gate design — not the capstone: the committee
version, threshold validation on held-out data, and the recalibrate-and-retry flow are
still open.

---

## 2026-06-11 21:26 EDT — Pose gate v2 + fusion oracle: pose is RELIABILITY-grade signal, not CORRECTION-grade. Head-context stream now triple-gated null. Cal-pose instability = strongest gate feature yet (ρ=0.47)

Two upgrades over the 20:42 gate, both per user critique: (a) yaw/pitch proxies added
(eye-width asym, corner-z asym, nose-bridge offsets — visually verified feature-by-feature
on overlaid frames, `analysis/pose_feature_overlay.py`; depth + z_asym ride MediaPipe's
weak z and were dropped from fusion); (b) DISTRIBUTIONAL coverage replaces median deltas
(cal-pose variance, KL(tgt||cal), Mahalanobis extrapolation fraction, embedding-space
nearest-cal-neighbor coverage — generalization is about support coverage, not mean shift).

**Gate v2 (`phase_shift_gate.py`, n=224):**
- Yaw/pitch deltas do NOT rescue the penalty story (best: nose_dy ρ=+0.135).
- **Coverage features strongly predict error LEVEL but not the cross-phase penalty:**
  cal_pose_std ρ=+0.47 vs cross error (+0.43 within, +0.05 penalty) — a head that wobbles
  DURING CALIBRATION makes a bad solve everywhere. **Strongest image-side gate feature
  found** (beats blink_frac +0.32, sharp_cv +0.36); actionable feedback ("hold still
  during calibration"). emb-coverage ρ=+0.52 similar (may overlap cluster-spread).
- Mapping-distortion predictability: still ~null except vertical gain (A11−1):
  CV R² = +0.23 — the only distortion component with any pose-explainable variance.

**Fusion oracle (`pose_fusion_oracle.py`, 7 verified features, CAL-RELATIVE pose,
n=230): NULL at every level of expressiveness.**
- A additive [E, α·pose_rel]: α=0.1 wash (−0.0002); α≥1 catastrophic (+0.018) — 20 cal
  points can't support the extra dims.
- B pose-keyed kernel attention over cal rows (degenerate cross-attention): flat at best
  (h=2, −0.0002), worse when localized (h=0.5, +0.006). No leverage: cal is one sitting,
  minimal pose spread to attend over. The learned-attention version has nothing to amplify.
- C population residual ceiling g(pose_rel), leave-subject-out, linear AND RBF: +0.0005 /
  +0.005. **No function of these features predicts the residuals — bounds EVERY
  architecture consuming them.**

**Synthesis: landmark-visible head geometry tells you WHICH subjects fail (ρ≈0.5
reliability signal → pre-flight gate) but NOT how to correct them (correction-grade
information ≈ zero).** Head-context input stream: now gated null three independent ways
(warp, deltas, fusion ceiling) — deprioritized with confidence. Formal caveat stands: an
oracle can't see backbone co-adaptation on full-face appearance (vs landmark summaries),
but the prior is now very low. The strategic lever standing: external pretraining.

---

## 2026-06-11 21:05 EDT — Native-resolution audit: the higher-res-crops lever is DEAD for this corpus (median native crop = 146×38 ≈ current 36×144)

Per-subject median native both-eyes crop size (frame 0, from the stored
`eye_hires_width/height`): **eyedata23 (train, n=194 sample): width p50=146 (p95=203),
height p50=38 (p95=54)** — vs the current 144×36 input. ≥1.5× width: 3%; ≥2×: 0%; at or
below current: ~47%. eyedata24 similar (p50 154×40, 9% ≥1.5×). Re-training at 72×288
would be pure upsampling for ~97% of subjects — the information isn't in the source
video. The TODO assumption ("source JPGs are higher-res than the crops") is false for
this corpus; the lever only returns if future collection uses higher-res cameras/closer
framing. Surviving smaller input-side idea: **RGB color at the SAME resolution** (the
hires crops are color; grayscale is a real discard) — un-gated, cheap to screen.

**Big-lever scoreboard after tonight's two gates: head-context stream — gate failed;
higher resolution — no information to add. Remaining strategic bet: external
pretraining (GazeCapture / MPIIFaceGaze). Next TPU session stays: the v1 baseline+recipe
runs (70ep single cosine + EMA, multi-seed).**

---

## 2026-06-11 20:42 EDT — Phase-shift gate: DOUBLE NULL. No recoverable geometric nuisance, and head-pose deltas do NOT predict the mapping distortion — head-context stream gate FAILED

`analysis/phase_shift_gate.py` (the no-TPU gate for the head-context lever; TODO
near-term #1, sharpened per discussion — the warp is a DIAGNOSTIC decomposition, not a
fix). Frozen hybrid `8lsofasl`, v1 eyedata24, n=224 with complete cal in BOTH phases.

- **ARM 1 (warp decomposition):** re-embedding crops affine-aligned (gaze-invariant
  landmarks only) to each subject's phase-1 template makes cross-phase slightly WORSE
  (median 0.0759 → 0.0768, 88/224 improved) — and the within-phase sanity arm shows the
  warp itself costs ~0.002 (interpolation). **Net affine-nuisance component ≈ ZERO**: the
  landmark-anchored crop box already removes whatever an affine can remove. The
  cross-phase penalty (median 0.0084 = 0.0759 cross vs 0.0605 within) is mapping-level.
- **ARM 2 (pose deltas → penalty):** ρ ≤ 0.14 for every head-geometry delta (center,
  scale, roll, depth). Subjects who visibly moved more barely suffer more.
- **ARM 2b (learnability):** per-subject mapping distortion under the P1 solve
  (pred ≈ A·y + b on P2 targets) is NOT predictable from pose deltas — 5-fold CV R² ≈ 0
  or negative on all six components of (b, A−I).
- **The by-product finding:** the distortion itself is LARGE (diag(A)−1 sd ≈ 0.21-0.26 —
  the prediction-compression visible in the 10:32 cluster figures) but unexplained by
  visible head geometry. Whatever drives per-subject mapping error, the landmarks can't
  see it (candidates: kappa angle / eye anatomy, lighting state, cal quality).

**Implication for the big-lever queue:** the iTracker-style head-context input stream
just failed its mechanism gate — deprioritize it. Caveats kept honest: linear model,
5 crude pose features, distortion estimates noisy (~124 targets/subject), and a
co-adapted stream could in principle help within-phase too — but as a prior-setter the
geometry→gaze-correction story has no support in this data. Next gates: native-resolution
audit (the higher-res lever), then external pretraining as the remaining strategic bet.

---

## 2026-06-11 18:57 EDT — Do EAR/sharpness predict accuracy? Yes, weakly per frame; at the subject level INCONSISTENCY beats LEVEL across every signal

Quantified the QA→accuracy relationship at both levels (frozen-embedding per-frame errors,
deployed solve, n=230 / 200k frames; extends the 06-09 gate screen with the EAR features
it was missing):

- **Frame level: real but weak.** EAR ρ=−0.18, rel-sharpness ρ=−0.10 (−0.20/−0.13 as
  distance-from-sweet-spot). Matches the binned curves (`qa_thresholds.png`): the signal
  lives in the bottom ~5% of frames, not the bulk — fine for gating a TTA average, useless
  as a per-frame confidence score.
- **Subject level: the inconsistency family wins.** sharp_cv ρ=+0.36 (NEW — sharpness
  LEVEL is weak, −0.16, replicating 06-09's −0.12), **blink_frac (EAR<0.75×median)
  ρ=+0.32 (NEW)**, bright_std +0.33 / bright_med −0.26 (replicates 06-09). EAR level
  −0.19, EAR std +0.16. Same pattern 06-09 found for brightness, now shown for sharpness
  and blinks: a subject whose frames FLUCTUATE is worse than one uniformly dim/blurry.
- Gate design implication: `blink_frac` + `sharp_cv` join the PRE-calibration feature
  tier (no model needed); cluster spread (ρ≈0.84, 10:32 entry) remains the post-cal
  tier and dwarfs them all.

---

## 2026-06-11 18:33 EDT — TTA CONFIRMED ON THE REAL MODEL: cross-phase median 0.0717 → 0.0601 (202/230 subjects improve). Best number the project has produced.

Real-model single-frame vs all-frames eval (`analysis/eval_singleframe_vs_tta.py`) on the
70-ep arm-B checkpoint (`multiframe_random_s42_ext70`, W&B `kwppnkuj`): cal slots fixed at
frame 0 (identical solve, learned weighting head included), target slots swept over frame
positions k=0..12, predictions averaged per trial over its real frame count. v1 eyedata24
all-frames records, complete-cal filter, n=230. Ran as a 16-shard SLURM array on the HPC
after the Colab session died (~110 s/subject; single-process TF only used ~31 of 128
cores — shard by subject, not cores).

- **single (frame 0):  median 0.0717  mean 0.0962  p90 0.207  worst 0.366** — matches the
  in-training ep-70 eval (0.0713/0.0962) through a fully independent code path. ✓
- mid-frame:           median 0.0678
- **TTA (all frames):  median 0.0601  mean 0.0863  p90 0.186  worst 0.343**
- **Paired: −0.0075 median, 202/230 subjects improved (88%).** Every distribution stat
  improves. Absolute median gain −0.0116 (16% relative) — same magnitude the frozen-
  checkpoint oracle predicted this morning (0.0778→0.0646), so the learned weighting head
  doesn't eat the gain (as expected: it only touches the solve; predictions are linear).

The TTA chain is now closed end-to-end in ONE day: oracle → cluster characterization →
real-model confirmation. **0.0601 is the best cross-phase median this project has
recorded** (population/extraction differ from the old 0.072-0.073 numbers — within-v1
comparison is the meaningful one: 0.0717 → 0.0601 on the same model). Remaining to bank
it: wire frame-averaged inference into the deployment path (and the QA-gated variant —
drop blink/no-face frames — for the tail), and report single+TTA in all future evals.

---

## 2026-06-11 13:36 EDT — Multi-frame TRAINING screen: NULL (dead tie at 20ep). Plus the TPU-lock gotcha, again.

20-ep shared-seed screen (seed 42, Colab TPU v5e, W&B `eye-tracking-multiframe`), both arms
entirely within the v1 extraction (train: repacked eyedata23 candidate records; eval: v1
frame-0 two-phase set). Arms differ ONLY in which real frame each trial contributes per
epoch (`et_util/frames_data.py`):

- ARM A `frame_mode=first` (single-frame baseline): cross-phase **median 0.0775**,
  mean 0.1032, p90 0.218, worst 0.371 (n=230); within-phase 0.0626. No init collapse.
- ARM B `frame_mode=random` (settling-window draw per epoch): cross-phase **median 0.0775**
  (delta +0.0000), mean 0.1052, p90 0.219, worst 0.388; within-phase 0.0644. Slightly
  LOWER train loss (0.0496 vs 0.0508) — the real-frame variation isn't even harder to fit.

**Verdict: real-frame sampling as training augmentation adds nothing at 20ep** — exact
median tie, tail marginally worse. Most likely the per-image synthetic augmentation
(brightness/contrast/gamma/noise per frame) already covers the photometric nuisance that
within-trial frames carry — the same redundancy argument that retired the session-shift
photometric aug on analysis. Don't extend to 50ep/multi-seed without a new angle (e.g.
temporal-consistency loss across a trial's frames, or aug-off ablation to test redundancy).

Notable numbers: a from-scratch 20-ep model on v1 frame-0 data (0.0775) matches the
50-ep old-data hybrid checkpoint scored on the same v1 eval (0.0778, this morning's
oracle baseline) — but that's a cross-extraction comparison, treat as coincidence.
Checkpoints preserved in the W&B runs (dulcet-field-3, eager-dawn-4).

**Colab TPU gotcha (cost ~1 h, now twice-learned):** running `jax.devices()` (or any JAX)
in the notebook kernel locks the TPU to the kernel process — a `!python` training
subprocess then silently falls back to CPU (~24 s/step vs 307 ms/step, ~80×). Train
IN-KERNEL (`train_mod.run(cfg)`) on Colab TPU, or never touch JAX in the kernel first.
(Also fixed for real along the way: the frames pipeline now caches GROUPED candidate
stacks and per-epoch picks are a vectorized gather — the first version re-decoded
everything per epoch.)

---

## 2026-06-11 10:32 EDT — Prediction-cluster characterization: 75% of error is BIAS, settling takes ~4 frames, and cluster spread is a STRONG no-ground-truth confidence signal (r=0.88)

Deployment can't average target frames (no trial structure in the wild — must predict
per frame), so instead characterized the per-trial cluster of frame-wise predictions
(`analysis/prediction_cluster_analysis.py`, deployed solve cal@frame0 λ=0.1, n=230):

- **Decomposition (exact, weighted-NED space):** per-frame error median 0.0715 =
  bias (centroid offset) median **0.0646** + jitter (RMS spread) median **0.0354**;
  jitter is only **~25% of squared error** (IQR 15-35%). So the TTA gain measured earlier
  IS the jitter term, 0.0646 is the hard floor for ANY temporal smoothing, and ~75% of
  the squared error is systematic — the data-bound diagnosis again.
- **Settling is real and takes ~3-4 frames:** distance-to-centroid falls 0.055 (pos 0)
  → 0.040 (pos 3-4) → plateau ~0.035; error-to-target falls monotonically 0.109 → 0.086
  through pos 12 (late positions partly a longer-trial composition effect). Deploy
  implication: discard/down-weight the first ~3 frames after stimulus onset; light
  temporal smoothing can recover most of the jitter term.
- **Cluster shape:** raw-screen-pct std_y (3.67) ≈ 2× std_x (1.92) — but in PHYSICAL
  screen units (16:9) that's nearly isotropic; the apparent vertical dominance is mostly
  the unit system.
- **The headline: per-subject jitter predicts per-subject error at r=0.88 / ρ=0.84 —
  and even predicts the BIAS component (r=0.84), which spread "can't see".** Cluster
  spread is computable at deployment with NO ground truth (any fixation video, e.g.
  validation points after calibration). Compare: the best feature in the 06-09
  pre-flight-gate screen was ρ≈−0.52 (N frames). This single signal likely dominates
  the entire 7-feature gate — fold it in as the primary reliability feature.

---

## 2026-06-11 10:26 EDT — Target-side TTA is a REAL WIN (median −0.013, every stat improves); frame 0 is the worst single target frame. v1 label swap FIXED everywhere.

**Label fix executed** (SLURM chain 5285-5288): all 2,004 v1 tfrecords swap-corrected
(tally sums exactly to 2,004, zero ambiguous — the fixer detects swapped labels from the
lattice geometry, so it's idempotent by inspection; `preprocessing/fix_v1_labels.py`),
plus the 250 embedding npz, the staged /work webm manifests, the GCS webm-bucket manifests
(column names), and `server/repack_webm.py` at the source. Corrected v1 records re-synced
to `gs://webcam-jspsych-frames/v1`.

**Frame-index sweep + target TTA** (`analysis/frame_index_sweep.py`, same frozen-hybrid
embeddings + protocol as the morning's cal-side oracle, n=230):

- **TARGET TTA (mean embedding over ALL the trial's frames, cal unchanged @frame 0):
  median 0.0778 → 0.0646 (−0.0132) at the deployed λ=0.1.** Mean −0.013, p90 −0.030.
  With λ=1: **0.0615 median** — combined −0.016 vs the deployed recipe (~21% relative),
  and every distribution stat improves. This is 3-4× the seed-noise band and bigger than
  the λ knob or the backbone upgrade. emb-mean ≡ pred-mean exactly (linear readout —
  internal sanity check passed). First-3-frames-only captures ~60% of the gain.
- **Frame-index: frame 0 is the WORST single target frame.** Single-target-frame sweep
  improves ~monotonically to mid-trial (p=mid −0.0078; p=last slightly worse than mid) —
  the autoexposure/fixation-settle hypothesis is REAL on the target side. But no single
  frame comes close to the TTA average, so frame choice is moot at deploy: average them.
- Coherent with the morning's cal-side NULL: the ridge already averages over ~20 cal
  points, so cal-side enrichment adds nothing; per-TARGET frame noise is un-averaged in
  the deployed recipe, and that's exactly where TTA collects.
- Asymmetry note: calibration solves prefer EARLY frames (matched cal@mid+tgt@mid is
  worse than cal@0+tgt@mid by ~0.011 at λ=0.1; mostly vanishes at λ=1) — unexplained,
  doesn't change the conclusion.

**Deploy implication:** at inference, average the per-frame embeddings (or equivalently
predictions) across each trial's frames — frames the system already records. No
retraining. Because predictions are linear in embeddings, the learned weighting head only
affects the SOLVE, so this oracle gain should transfer to the real model ~directly.
NEXT: confirm on the real model eval (multi-frame targets through the actual checkpoint
+ learned weights), then wire into the deployment path. Also worth checking TTA × QA
gating (drop face_found=0 / blink frames from the average) for the tail.

---

## 2026-06-11 09:53 EDT — Multi-frame calibration oracle: NULL beyond solve-λ. Plus: v1 labels have x/y SWAPPED (bug, fix pending)

**Bug found on the way in:** every v1 record's `x`/`y` labels are SWAPPED. Webm filenames are
`{subject}_{phase}_{x}_{y}.webm` (per the legacy parser, `tfrecord_processing.py:59`), but
`server/repack_webm.py` named those tokens `y_pct,x_pct` in the bucket manifests, and the
extraction job copied them through. Pixels/landmarks/crops unaffected — only the two label
floats. Caught because the 20-point grid matched zero phase-1 trials (the eval lattice is
16 wide × 9 tall; the stored "x" had 9 unique values). Interim: `--swap-xy` flag in the
oracle. TODO durable fix: rewrite v1 record labels + bucket manifests + repack_webm.py
field names (cheap label-rewrite job, NO re-extraction needed).

**Experiment** (the calibration-side TTA item, real frames): embedded ALL eyedata24 v1
frames (~950k face-found frames, 250 subjects) with frozen hybrid `8lsofasl`
(tokpool1+emb128, seed 42) — `analysis/extract_v1_embeddings.py`, 16-shard SLURM array,
~50 min on 4 emc nodes. Oracle (`analysis/multiframe_calibration_oracle.py`): cross-phase,
cal = P1 grid trials enriched with k frames each, targets = P2 non-grid FIRST frame (fixed),
complete-calibration filter, n=230. Factorial: k ∈ {1,2,3,5,8,all} × policy {first, spread,
sharp} × agg {stack k rows, mean-embedding} × λ ∈ {0.1, 0.3, 1.0}.

- **v1 first-frame baseline (deployed recipe): median 0.0778**, mean 0.109, p90 0.244,
  worst 0.470 (n=230 — looser filter population than the old n=211 clean eval; RGB-correct,
  so NOT comparable to old-record numbers, as expected).
- **λ alone (k=1, λ=1.0): median −0.0045** (165/230 subjects improve), mean −0.013,
  p90 0.244→0.208, **worst 0.470→0.295**. Replicates the 06-09 λ-oracle on the new
  extraction: λ is the tail knob.
- **Best multi-frame cell (k=2 spread/stack λ=1): median 0.0704** — but paired delta vs
  deployed is −0.0041 vs λ-only's −0.0045, and mean/worst are no better than λ-only.
  **Multi-frame calibration adds ≈ NOTHING once the solve is properly regularized.**
- Stacking k rows at the deployed λ=0.1 actively HURTS (+0.007 @ k=2 → +0.016 @ k=all):
  more rows at fixed λ = under-regularization. Any multi-frame deploy MUST retune λ.
- 'sharp' (top-k Laplacian) is the WORST selection policy at every k — sharpness-based
  cal-frame picking looks unpromising. 'first' ≈ 'spread'.
- Monotonic degradation past k≈2-3 even with mean-agg: late-trial frames dilute (0.5 s
  trials — fixation drift). Early frames are GOOD — evidence against the
  autoexposure-settle/frame-1-is-worst hypothesis, pending the target-side frame-index sweep.
- Data notes: cal trials avg 12.3 frames; 7 subjects have face_found=0 on EVERY v1 frame
  (the BGR-era pipeline found faces for them — borderline detections flip with the channel
  fix); 20/250 subjects fail the complete-cal filter (consistent with the known uf2sh
  quality issues).

**Verdict: the embedding→ridge wall holds.** Another interface manipulation, another null —
calibration-side multi-frame TTA joins centering/per-subject-λ/reweighting in Closed/null.
Artifact kept: per-frame embeddings at `/work/jdeleeuw/embeddings/
v1_eyedata24_hybrid_8lsofasl` (250 npz, E+labels+QA per frame) — directly reusable for the
target-side frame-index sweep and target-TTA, which are NOT answered by this result.

---

## 2026-06-10 20:59 EDT — Full-corpus all-frames extraction COMPLETE on HPC (19 min, 0 errors)

SLURM array 5257 (4 emc nodes × 120 workers): all shards COMPLETED in **~19 min wall-clock**.

- **2,004/2,004 subjects, 325,584 trials, 0 failed trials, 4,303,834 frames extracted.**
- **Face-found rate 97.5%** (no-face frames are kept with `face_found=0` + QA fields — that
  2.5% is exactly the QA-gating signal).
- Avg **13.2 frames/trial** (not the nominal ~24 — webcam fps varies a lot by subject
  hardware; plan multi-frame experiments around variable k, not fixed 24).
- Output: **70 GB** at `/work/jdeleeuw/frames/v1/eyedata{23,24}/{subject}.tfrecord` (+
  per-subject stats.json). gcloud SDK installed at `~/google-cloud-sdk` on the login node.
- **DONE (21:30 EDT): uploaded + finalized.** `gs://webcam-jspsych-frames/v1` = 74.5 GB,
  4,008 objects + `manifest.csv` (2,004 rows) + `include_list.csv`, verified from a second
  machine. Upload from HPC login node at ~111 MiB/s (~11 min). The HPC copy stays at
  `/work/jdeleeuw/frames/v1` (free working copy for cluster-side experiments).
- NEXT: frame-index sweep + multi-frame calibration oracle on the new records (compare WITHIN
  this extraction only — RGB fix means old records aren't comparable).

---

## 2026-06-10 16:29 EDT — All-frames extraction: smoke test PASSED on Vassar HPC; full corpus staged + array job launched

Pivoted execution to the Vassar HPC (`jr.vassar.edu`, SLURM) — webms staged from EC2 (free
from our side; GCS ingress free), so the whole pipeline costs ~$0 vs ~$9 GCS egress.

- **Cluster facts:** `emc` partition = 4× (128 CPU, 512 GB, AVX2) — the workhorses. Login node
  + `general` node1/2 are pre-AVX 2010 Opterons: tensorflow/mediapipe wheels DIE there
  ("Illegal instruction") — always `srun -p emc`. Compute nodes have internet but no libGL.
- **Working env** (`~/eye-tracking/.venv`, py3.10): mediapipe **0.10.14** (0.10.35 REMOVED the
  legacy `mp.solutions` API), tensorflow-cpu **2.17.1** (2.21 needs protobuf≥6, mediapipe needs
  <5), numpy<2, opencv-python-headless **4.9.0.80** force-reinstalled --no-deps LAST (mediapipe's
  opencv-contrib dep needs libGL; both opencvs share the cv2 dir, so uninstall contrib THEN
  reinstall headless; 4.13 headless needs numpy≥2). Pin file worth creating if this recurs.
- **Script additions:** `--src/--dst` accept local dirs (staged HPC mode, no gcloud on compute
  nodes) and `--shard I/N` (stable crc32 partition, SLURM-array-safe). `preprocessing/
  extract_frames.sbatch` = 4-task array, 1 emc node/shard, 120 workers.
- **Smoke test (2 eyedata24 subjects, emc node): PASSED.** 576 trials, 10,107 frames, 0 errors,
  100% face-found. Schema validated (labels match filenames; landmarks normalized; CLAHE crop
  full-range; hires ~40×166 native; EAR p5≈0.10 = blinks visible to QA). Visual check of crops:
  good. Real fact: frames/trial varies by subject hardware (~10 vs ~25, "24" is nominal).
  ~88 MB/subject → ~180 GB total estimate holds.
- **Transfer EC2→HPC: DONE** (15:58–16:25, ~46 MB/s). 2,130+352 tars + manifests = 74 GB at
  `/work/jdeleeuw/webm/eyedata{23,24}/` (mirrors bucket layout). EC2 pem staged at
  `~/.ssh/jpsychaws.pem` on the cluster.
- **16:27 — full extraction launched:** sbatch array **5257**, shards 0–3 on node5–8, dst
  `/work/jdeleeuw/frames`. ETA ~30–60 min. Remaining after: gcloud rsync `/work/jdeleeuw/
  frames/v1` → `gs://webcam-jspsych-frames/v1` from login node (gcloud installing; needs auth)
  + `--finalize-manifest`.

---

## 2026-06-10 14:46 EDT — BGR→RGB MediaPipe bug found + fixed; all-frames GCS extraction job written

**Bug:** every `face_mesh.process()` call in `et_util` (5 in `mp_processing.py`, 4 in
`tfrecord_processing.py`) fed BGR frames (`cv2.imread`/`VideoCapture` output) to MediaPipe,
which expects RGB (per the official `FaceMesh.process` docstring). So ALL existing landmarks —
and the landmark-derived 36×144 crop boxes in every published TFRecord (train `6b5cd`, eval
`uf2sh`) — were computed on channel-swapped images. Grayscale pixel values are unaffected
(`BGR2GRAY` ran on correctly-ordered BGR); the only pathway into the data is landmark
positions → crop geometry. Magnitude unmeasured.

**Decision (user):** fix outright rather than preserve the quirk for comparability. All 9
legacy call sites now convert BGR→RGB. Consequence: regenerated data will NOT be bit-identical
to the published TFRecords — never mix old and new records as arms of one experiment; rerun
baselines within the new extraction.

**New job:** `preprocessing/extract_frames_to_gcs.py` (untested draft) — decodes ALL ~24
frames per trial webm from `gs://webcam-jspsych-webm`, MediaPipe per frame (RGB, static mode;
`--tracking` optional), writes per-frame examples to per-subject TFRecords in a NEW bucket
(default `gs://webcam-jspsych-frames`, `{version}/` prefix, v1). Schema per frame: legacy-
recipe 36×144 CLAHE `eye_img`, native-res color pre-CLAHE `eye_hires_jpeg`, 160×120
`frame_small_jpeg`, 478×3 landmarks, QA scalars (sharpness, brightness, EAR L/R, eye width,
timestamps). Idempotent (stats.json uploaded last = done marker), manifest-driven, requires a
consent `--include-list`. Est. ~170–190 GB (~$4/mo), ~80–100 CPU-h (32-vCPU spot VM: ~3–4 h,
$1–2). Unlocks: frame-index sweep, multi-frame calibration/TTA, QA gating, higher-res/RGB
crops, Zhang–Sugano normalization, iTracker face stream. Not yet run — smoke test next.

**Setup done (14:55 EDT):** bucket `gs://webcam-jspsych-frames` created (us-central1,
STANDARD, uniform access, public-access-prevention enforced). Consent include-list built
conservatively as the PUBLISHED subject set (consent ∩ complete ∩ not-excluded): 1,747
eyedata23 ids from listing the OSF 6b5cd tarball members + 257 eyedata24 ids from the local
uf2sh eval tfrecords. Validated: every id present in its GCS manifest, zero cross-dataset
overlap. → `preprocessing/include_list_v1.csv` (2,004 subjects) + provenance copy at
`gs://webcam-jspsych-frames/v1/include_list.csv`. Note: this excludes ~383 eyedata23 + ~95
eyedata24 raw subjects (non-consented OR incomplete) — if more data is ever wanted, the
research-videos-consent-but-incomplete subset could be recovered via notebook-01's source
JSONs, separate decision.

---

## 2026-06-10 14:24 EDT — Raw video migrated to GCS (EC2 no longer sole copy)

Repacked all raw webm files into per-subject tar archives and uploaded to
`gs://webcam-jspsych-webm` (GCP project `webcam-eyetracking-jspsych`, `us-central1` Standard,
uniform bucket-level access, public access prevention enforced).

- eyedata23: 2,130 tars, 296,579 files → 56 GB
- eyedata24: 352 tars, 95,909 files → 19 GB
- Manifests: `eyedata23/manifest.csv`, `eyedata24/manifest.csv`, `manifest_all.csv` (392,488 rows)
- Upload speed: 107.6 MB/s; total transfer ~21 min

Filename schemas encoded into manifest columns (`subject_id`, `phase`, `y_pct`, `x_pct`).
eyedata24 has a `phase` column (1 or 2); eyedata23 `phase` is blank (single-phase).
Repack script: `server/repack_webm.py` (idempotent, skips existing tars).

Both EC2 and GCS now hold the raw video — EC2 retirement (TODO item 4) is the remaining step.
OSF public releases (jpg/tfrecords) unchanged.

---

## 2026-06-10 11:24 EDT — Hardware cost benchmark (GCP grant app): TPU v5e ≈5× faster / 3.3× cheaper than L4 for the hybrid; v5e already saturated at bs5; hybrid can't run fp16 on GPU

Colab timing probes for the Google Cloud research-credits budget (densenet = 50ep full runs;
hybrid = 3ep settled-step probes, throughput extrapolated):

| arm | hw | config | settled | 50-ep est | est cost/run |
|---|---|---|---|---|---|
| densenet bs5 fp16 | L4 | 50ep measured | 63 ms/step | 20m15s (measured) | $0.29 |
| hybrid bs5 bf16 | L4 | 3ep+kill @ep8 | 562 ms/step, 347 s/ep | ~4.8 h | ~$4.10 |
| hybrid bs5 bf16 | TPU v5e | 3ep probe | **204 ms/step, 71 s/ep** | **~1.05 h** | **~$1.25** |
| hybrid bs16 bf16 | TPU v5e | 3ep probe | 639 ms/step, 70 s/ep | — | no gain |

- **v5e is saturated at bs5**: bs16 throughput identical (+2%); bs20 OOMs (16.58 G needed vs
  15.75 G HBM, ceiling ≈ bs19). Large-batch training buys nothing on v5e — don't bother with
  LR-scaling experiments there. v6e (32 G, ~2× FLOPs) might reopen it.
- **GPU gotchas (hybrid had never run on GPU before):** `mixed_float16` FAILS on L4/JAX — cuDNN
  flash attention rejects the Q=1/KV=72 attention-pool readout, and a separate XLA "Unsupported
  dot precision algorithm: 4" persists even with flash attention off. Working recipe:
  `mixed_bfloat16` + `keras.config.disable_flash_attention()`. (config.py's fp16 guidance is
  densenet-only.) On TPU no workaround needed — splash-attention kernel errors on the same Q=1
  shape but JAX falls back silently.
- L4 per-step gap (2.75×) < per-epoch gap (4.9×): L4 epochs carried ~150 s non-step overhead
  (aug pipeline, 12 vCPU vs TPU VM 24). Sanity: L4 densenet 50ep reproduced cross-phase median
  0.0834 (historical 0.084). W&B: 0hdl4oj6 (densenet), 6gcmouq5 (killed hybrid).
- Driver-notebook gap found: `training/train.ipynb` never fetches the uf2sh eval set, so
  `evaluate()` crashes post-training — needs `osf -p uf2sh fetch single_eye_tfrecords.tar.gz
  eval_single_eye_tfrecords.tar.gz` (fix not yet committed).

## 2026-06-10 08:17 EDT — tokpool1+emb128 promoted to default config; branches merged to main

`personalization-attn` (which contains `hybrid-attn-pool`) fast-forward-merged into `main`.
Defaults changed: `embedding_dim` 200→128, `token_pool_extra` 0→1 — `TrainConfig()` now builds
the proven 5.20M-param tokpool1+emb128 hybrid (multi-seed cross-phase median 0.073 ± 0.001).
Verified: default build = 5,198,273 params (matches checkpoint `8lsofasl` exactly) and the
historical densenet `full_model.weights.h5` still loads with explicit
`TrainConfig(backbone="densenet", embedding_dim=200)`. Callers of the old defaults updated
(CLAUDE.md recipes, `training/train.ipynb` driver, `run_centering_oracle_colab.py`).
GOTCHA now documented in config.py + CLAUDE.md: historical checkpoints need their original
dims passed explicitly; the W&B `eye-tracking-calib-attn` hybrid checkpoints match the new
defaults.

## 2026-06-09 21:08 EDT — Multi-seed COMPLETE (3 healthy seeds): baseline 0.072–0.074 median; 3-seed ensemble = best-member median + mean −0.003

Seed-789 trained healthily (replacement for collapsed 456): cross-phase median 0.0721 (own-run
bfloat16 eval) / 0.0729 (float32 ensemble-script eval — tiny policy-dependent wobble, ~0.001).
**Final multi-seed picture for tokpool1+emb128 linear (40ep), cross-phase clean eval, float32:**

| | seed 42 | seed 123 | seed 789 | **3-seed pred-ensemble** |
|---|---|---|---|---|
| median | 0.0723 | 0.0743 | 0.0729 | **0.0721** |
| mean | 0.0964 | 0.0964 | 0.0955 | **0.0934** |
| p90 | 0.1969 | 0.2024 | 0.1976 | **0.1947** |

- **Multi-seed baseline: median 0.073 ± 0.001** — the definitive number to quote (and to judge
  future ~0.005 "improvements" against). Within-phase ensemble: median 0.0534, mean 0.0749.
- **Ensembling delivers exactly the textbook gain**: ties the best member's median, beats every
  member's mean (−0.002/−0.003) and p90, monotone 2→3 seeds (mean 0.0943→0.0934). It's the
  cheapest mean/tail improvement on the board if deploy-side 3× inference is acceptable.
- TODO item 3 remaining: EMA/SWA variant, R0 multi-seed re-check (needs `--ridge-bias` merged),
  promote tokpool1+emb128 default + merge `hybrid-attn-pool`.

## 2026-06-09 20:30 EDT — Multi-seed: seed-456 COLLAPSED at init (1/3 rate!); 2-seed prediction ensemble gives the expected free mean/p90 gain

Continuing TODO item 3 on tokpool1+emb128 linear (40ep, mixed_bfloat16, v6e):
- **Seed-456 collapsed at initialization**: train loss flat at ~0.222 from epoch 1 through 40
  (never broke symmetry), eval = constant-prediction degenerate (every subject 0.2515). NOT an
  eval bug — the model trained to predict ~screen-center. So the config has a real init-collapse
  rate (1 of 3 seeds observed). METHODOLOGY: collapse is detectable by epoch 2 (loss stuck at
  ~0.22 vs ~0.12 on healthy runs) — future multi-seed/sweep runs should check-and-reroll early
  instead of burning 40ep. Seed-789 launched as replacement (W&B `7vwqd749` = the dead run).
- **2-seed prediction ensemble (seeds 42+123, `analysis/ensemble_eval.py` new):** cross-phase
  median 0.0726 (members 0.0723/0.0743), mean **0.0943** (members 0.0964/0.0964), p90 0.1948,
  IQR tightened. Beats both members on mean/p90/IQR, ties the better member's median — the
  classic free noise-band gain at n=2. Within-phase: 0.0545/0.0756 (small consistent gain).
  3-seed update pending seed-789.

## 2026-06-09 19:45 EDT — Real-eval lambda sweep: the oracle's MEDIAN gain does NOT transfer (learned weights already cover it); lambda is a TAIL knob. Plus: 0.072 confirmed at a 2nd seed.

Confirmed the lambda-oracle pick through the real eval path (`analysis/sweep_eval_lambda.py`,
rebuilds the eval model per `eval_ridge_regularization`, learned calibration weights included).
Hybrid `8lsofasl` (tokpool1+emb128), clean eval n=211, cross-phase:

| solve lambda | 0.1 (trained) | 0.3 | 0.5 | 0.7 | 1.0 | 1.5 | 3.0 |
|---|---|---|---|---|---|---|---|
| median | **0.0723** | 0.0731 | 0.0745 | 0.0745 | 0.0757 | 0.0753 | 0.0837 |
| mean | 0.0964 | 0.0929 | **0.0926** | 0.0928 | 0.0936 | 0.0955 | 0.1032 |
| p90 | 0.1969 | 0.1853 | 0.1832 | **0.1824** | 0.1822 | 0.1824 | 0.1895 |
| worst | 0.3679 | 0.3119 | 0.3036 | 0.2970 | 0.2886 | 0.2773 | 0.2578 |

**Findings:**
- **The oracle's −0.006 median gain did NOT transfer: median is already optimal at lambda=0.1
  in the real model.** Resolution of the discrepancy: the oracle solves with UNIFORM weights
  (median 0.0785 @ 0.1 → 0.0725 @ 0.7); the real solve's learned per-point weights buy the
  same ~0.006 at lambda=0.1 (0.0723). The weighting head and extra shrinkage are REDUNDANT
  regularizers for the bulk of subjects. ORACLE LESSON recorded: a uniform-weight ridge oracle
  overstates what lambda retuning adds on top of the learned weighting — treat its absolute
  optimum as an upper bound on the combined mechanism.
- **What survives is the tail: lambda≈0.5 trades ~0.002 median for mean −0.004, p90 −0.015,
  worst 0.37→0.30** (and monotonically less at higher lambda: worst 0.26 @ 3.0 for +0.011
  median). A deploy-time robustness knob, free on any checkpoint.
- **Densenet `gf1zjs9v` DOES gain median from lambda: 0.0782 @ 0.1 → 0.0749 @ 0.3** (mean
  0.0999 → 0.0962; also reproduces the historical clean-eval numbers exactly, validating the
  re-report). Its (older-trained) weighting head regularizes less well than the hybrid's, so
  shrinkage still has median headroom there.
- **Decision: keep `eval_ridge_regularization` default None (= training lambda).** The headline
  hybrid number stays median 0.0723 @ 0.1; quote lambda=0.5 as the robustness option (tail
  −0.07 worst). For densenet comparisons, quote 0.0749 @ 0.3. The config knob + sweep script
  stay (zero-cost on any future checkpoint, and the next backbone may sit elsewhere on this
  curve).

**Multi-seed (TODO near-term item 3, partial):** seed-123 of tokpool1+emb128 linear finished:
cross-phase median **0.0740**, mean 0.0963 (vs seed-42: 0.0720, mean 0.0963 — identical mean,
median within the noise band). **The best-config result is real, not a seed draw.** Seed-456
in flight; prediction-ensembling check after it lands.

Headline numbers on the clean (cal>=20) eval, all reproduced through the current code this
session: **hybrid tokpool1+emb128 median 0.072–0.074 (2 seeds), densenet 0.0782 (0.0749 @
lambda 0.3)** — closes the "re-report on clean eval" hygiene item.

## 2026-06-09 18:55 EDT — Solve-time lambda oracle: global lambda ~0.5–1.0 is a real win; per-subject LOO/GCV lambda is NULL

Ran the solve-time lambda oracle (`analysis/lambda_oracle.py`, new) on FROZEN hybrid
tokpool1+emb128 embeddings (checkpoint `8lsofasl`, the 0.0720 linear baseline), uf2sh
cross-phase (cal P1 -> test P2). Training lambda and solve-time lambda are separate knobs —
the per-subject ridge is re-solved from frozen embeddings at eval — so this needed no GPU
training. Clean eval (cal>=20, n=211), median target NED:

| solve lambda | 0.1 (deployed) | 0.3 | 0.5 | **0.7** | 1.0 | 1.5 | 3.0 |
|---|---|---|---|---|---|---|---|
| median | 0.0785 | 0.0740 | 0.0727 | **0.0725** | 0.0735 | 0.0743 | 0.0758 |
| mean | 0.1077 | 0.0992 | 0.0966 | 0.0953 | 0.0944 | 0.0937 | 0.0940 |
| worst | 0.4519 | 0.3719 | 0.3528 | 0.3389 | 0.3235 | 0.3117 | 0.2947 |

**Findings:**
- **Global solve-time lambda ~0.5–1.0 beats the trained 0.1 everywhere: median −0.006,
  mean −0.013, worst 0.45 → ~0.33.** The curve is shallow (0.5–1.0 within 0.001), so the
  pick isn't fragile. Mean/worst keep improving slightly past the median optimum (more
  shrinkage = tail insurance).
- **Per-subject lambda (closed-form ridge LOO and GCV on each subject's ~20 cal points):
  NULL — don't build it.** The CHEAT upper bound (per-subject best lambda picked on target
  error) is only −0.002/−0.003 below the best global lambda, and honest selectors capture
  none of it: LOO ties global-best at best, GCV is worse, and both FATTEN the tail
  (worst 0.40–0.44 vs 0.34) by under-regularizing easy-looking subjects. 20 cal points are
  too few to estimate a per-subject lambda that beats a good global one.
- Bias (intercept) column: byte-identical to no-bias at every lambda — re-confirms the R0 null.
- Oracle caveat: uniform calibration weights (the real solve uses the learned sigmoid
  weights), so the true-model optimum may shift slightly; confirmed via the real eval path
  (`analysis/sweep_eval_lambda.py` + new `eval_ridge_regularization` config field) — results
  in the next entry.

**Code:** `eval_ridge_regularization` (TrainConfig + `--eval-ridge-regularization`) decouples
solve-time lambda from training lambda; `_build_eval_model` rebuilds the ridge with it (lambda
is a constructor constant, not a weight, so it's free on existing checkpoints). Branch
`personalization-attn` (5f31f6c, 8523617).

## 2026-06-09 15:36 EDT — Calibration-centering/standardization oracle (on hybrid embeddings): NULL/negative — argues against the cross-attention refiner

Ran the calibration-centering oracle (`analysis/calibration_centering_oracle.py`) on FROZEN
**hybrid** tokpool1+emb128 embeddings (extracted from `full_model.weights.h5`), cross-phase
(cal P1 -> test P2, n=244). This is the cheap linear special-case of the proposed sample<->
calibration cross-attention refiner: re-center / standardize each subject's per-frame embeddings
relative to their own calibration set, then the same per-subject ridge. Bias-corrected conditions
(the meaningful ones), median, consistent across lambda {0.01, 0.1, 1.0}:

| condition | median (lam=0.1) | delta vs baseline |
|---|---|---|
| baseline | 0.0837 | — |
| +bias (R0) | 0.0837 | 0.000 |
| center+bias | 0.0832 | -0.0005 (wash) |
| standardize+bias | 0.1266 | +0.0429 (WORSE) |

(center/standardize WITHOUT bias are +0.42 broken -- the expected artifact: subtracting the cal-mean
injects a constant gaze offset a no-bias ridge can't fit. Bias rows are the real comparison.)

**Result: NULL/negative.** `center+bias` collapses to baseline exactly as the math predicted
(= add-an-intercept = R0, already marginal); `standardize+bias` actively HURTS (+0.043 median, mean
0.18 vs 0.117) -- dividing each dim by its per-subject calibration std destroys gaze-relevant scale.
The oracle baseline (0.0837 @ lam0.1, 0.0779 @ lam1.0) reproduces the model's ballpark, so the
extraction is sound. **Conclusion: don't build the cross-attention refiner.** One-sided caveat (a
learned refiner could co-adapt the backbone, invisible to a frozen-feature oracle), but stacked on
the reweighting NULL and the train-loss evidence that the ridge is already robust, three independent
probes now agree: the **per-subject calibration-manipulation axis has little exploitable signal in
the current features**. The floor looks data/hard-subject-bound, not personalization-bound.

## 2026-06-09 13:35 EDT — Context-aware calibration weighting (set-attention head): directionally positive, biggest move in the tail, but within single-seed noise

Tested whether the per-point calibration **weighting** can be improved. Replaced the
context-free `Dense(1, sigmoid)` ridge sample-weight head with a new
`CalibrationWeightingHead` (`et_util/custom_layers.py`): a small masked self-attention
over the calibration set so each point's weight is computed *relative to the others*
(outlier rejection / redundancy / coverage), then fed to the same weighted ridge.
Leak-proof (every query attends ONLY to calibration keys + self → verified bitwise-zero
target leakage), zero-init context branch (starts identical to the linear head), optional
`(x,y)` input. Gated by `cfg.calibration_weighting="set_attention"`. Branch
`personalization-attn`. (W&B project `eye-tracking-calib-attn`, entity `vassar-cogsci-lab`.)

Matched A/B: hybrid, **tokpool1 + emb128** (5.22M params), seed 42, 40ep, mixed_bfloat16,
v6e TPU. Backbone constructed before the head → identical backbone init, so the ONLY
difference is the weighting head. Cross-phase eval (cal P1 → test P2, n=211):

| metric | linear baseline | set_attention | Δ |
|---|---|---|---|
| **median** | 0.0720 | **0.0696** | −0.0024 (−3.3%) |
| mean   | 0.0963 | 0.0948 | −0.0015 |
| **worst**  | 0.3661 | **0.3193** | −0.0468 (−12.8%) |
| p90    | 0.1970 | 0.1977 | +0.0007 |
| within-phase median | 0.0548 | 0.0539 | −0.0009 |

**Result: encouraging but inconclusive (single seed).** set_attention is directionally
better on every cross-phase metric, and the *largest* move is the **worst-case subject
(−12.8%)** while the bulk (p90, IQR, within-phase) is flat — exactly the mechanism's
predicted signature (helps subjects with bad calibration points most). Train loss matched
(0.04), so no extra overfitting. BUT the headline **median Δ −0.0024 is within the ~0.005
single-seed noise band**. Notable side-finding: tokpool1+emb128 **linear baseline already
hits 0.0720 median** — a touch better than the full hybrid's 0.075 (better-conditioned
ridge at smaller D, as hypothesized).

**Follow-up — tuned-head screen (40ep, seed 42, same config), instead of multi-seed first:**
made the head tunable (depth + optional FFN) and screened stronger/ablated variants vs the
linear baseline (0.0720), cross-phase median:

| variant | median | worst | vs linear |
|---|---|---|---|
| base set_attention (orig code) | 0.0696 | 0.319 | −0.0024 |
| base2 (rewrite, 1L attn64 h4 coords) | 0.0713 | 0.323 | −0.0007 |
| **strong** (2L+FFN attn128 h8 coords) | 0.0739 | 0.368 | **+0.0019** |
| nocoords (1L attn64 h4) | 0.0736 | 0.337 | +0.0016 |

**Verdict: MARGINAL / NULL — shelve.** Every set_attention variant (0.0696–0.0739) sits inside
the ~0.005 single-seed noise band around linear (0.0720), and the base re-run (base2 0.0713 ≈
linear) shows the original 0.0696 edge was a noise draw, not a stable effect. Crucially, **adding
capacity/depth did NOT grow the effect — `strong` was the worst arm** (0.0739), so this isn't an
under-capacity story. coords help a hair (base2 0.0713 < nocoords 0.0736, within noise). Only
weakly-consistent signal: the *small coords-on* head shaves the worst subject (base 0.319 / base2
0.323 vs linear 0.366, two runs agree) without moving the median — not enough to justify the
complexity. Multi-seed confirmation NOT warranted (nothing rose above noise to confirm). Code stays
gated + default-OFF on `personalization-attn`. Consistent with the broader "wall": the eye-appearance
CNN already extracts the signal; reweighting the per-subject ridge doesn't move the floor.

Ran a per-subject diagnostic on the two-phase eval (densenet checkpoint `gf1zjs9v`, CPU Colab):
computed each subject's cross-phase loss WITH subject_id, then characterized the worst ones.

**The worst subject = no calibration data.** `2725071097112` (file `yrvoyzo8`): loss **0.5932**, only
**N=5 frames, ALL phase-2, ZERO phase-1 calibration points** (cal=0). With cal=0 the per-subject ridge
has nothing to fit → degenerate prediction → an error that is *identical regardless of model*. That is
EXACTLY why this number was byte-identical across ~15 experiments — no architecture/feature could move
it because it was never a model problem. 2nd-worst (0.553) is the same (cal=0).

**The metric was being distorted by under-calibrated subjects:**
| filter                        | n   | mean   | median | worst  |
|-------------------------------|----:|-------:|-------:|-------:|
| ALL scored (what we reported) | 246 | 0.1126 | 0.0839 | 0.5932 |
| **cal>=20 (full grid only)**  | 211 | 0.0999 | 0.0782 | 0.3436 |
| cal>0 (any calibration)       | 244 | 0.1088 | 0.0833 | 0.3436 |

Cal-coverage among 246 scored: 211 have the full 20-pt grid; ~33 partial; **2 have cal=0** (the 0.59/0.55
"worst"). Plus 11 subjects already unscored (no phase-2 targets), and degenerate subjects exist (N=1, N=6).

**Conclusions:**
1. **Median was always robust** (0.0839 -> 0.0782 when cleaned) -> every relative comparison this session
   STANDS. But the **mean we quoted (~0.11) was inflated ~0.013** by un-calibratable subjects.
2. **Honest fully-calibrated numbers: mean 0.10, median 0.078, worst 0.34.**
3. **Eval-hygiene fix needed:** `evaluate()`/`build_two_phase_eval_dataset` should require a minimum
   calibration count (e.g. cal>=15-20); right now it scores subjects with phase-2 targets even when they
   have no phase-1 calibration, producing garbage.
4. The real worst fully-calibrated subject (`22nkcpfw`, 0.34, full data) has visibly **terrible eye crops**
   (off-center, over/under-exposed, eye barely in frame) -> a genuine image-quality case, not a bug.

## 2026-06-09 11:09 EDT — Pre-flight image-quality gate: VIABLE as a coarse screen (spun off from the tail investigation)

Tested whether cheap, on-device image features can predict gaze-prediction quality (a no-reference QA
gate that warns users about bad setup before proceeding). We already have labels: 246 subjects with
(images -> cross-phase loss). Computed 7 cheap per-subject features (brightness, contrast, Laplacian
sharpness, dark/over/under-exposure fractions, brightness inconsistency); Spearman + 5-fold CV
RandomForest; "bad quartile" = top-25% loss.

| predictor set                              | rank-corr | bad-quartile AUC | prec@top25% |
|--------------------------------------------|----------:|-----------------:|------------:|
| image features only (cal>=20, n=211)       | 0.44      | 0.67             | 0.50        |
| + data-completeness (N frames, cal pts)    | 0.58      | 0.78             | 0.54        |

**Findings:**
- **Strongest single signals are the cheapest:** `N_frames` (rho -0.52) and `cal_points` (-0.32) —
  data-completeness — plus **brightness inconsistency** (+0.45) and overall **brightness** (-0.24, darker
  worse). **Blur/sharpness is surprisingly WEAK** (-0.12) — webcam crops are uniformly low-res.
- **Verdict: viable as a coarse pre-flight gate** (AUC 0.78 combined). Catches the egregious cases
  (too few frames, bad/uneven lighting) but explains only ~20-30% of rank variance -> a screen, not a
  precise classifier. The actionable axes: "capture enough frames / calibration", "use steady, even,
  adequate lighting".
- **Untested but predicted strong: landmark-derived framing/visibility features** (eye size in frame,
  eye openness, head-pose extremity, landmark-detection confidence, crop centering). The 22nkcpfw failure
  was framing/visibility, invisible to brightness stats but obvious to landmarks (which we already
  extract). Likely the missing signal to turn the coarse gate into a strong, actionable one.

## 2026-06-09 08:11 EDT — Spatial-preserving front end (high-res + soft-argmax): NULL — floor is data-bound, not architectural

Tested the equivariance hypothesis from the CNN-design research: gaze is a localization task,
but the hybrid's front end (early max-pools) is invariance-biased — maybe that caps the floor.
Two 20ep shared-seed (seed 42) runs vs the 20ep baseline (0.0822). Branch `hybrid-attn-pool`.

| run                                   | cross median | Δ vs base | worst  |
|---------------------------------------|-------------:|----------:|-------:|
| baseline (emb200, attn-pool)          | 0.0822       | —         | 0.5932 |
| **high-res** (`cnn_late_pool` + PatchMerging) | 0.0888 | **+0.0066** | 0.5932 |
| **soft-argmax** readout               | 0.0814       | −0.0008   | 0.5932 |

**Verdict: NULL/negative — hypothesis NOT supported.**
- **High-res HURT** (+0.0066, outside noise) at 2.5x the compute. Running all conv stages at
  full 36x144 and folding detail into tokens via Swin patch-merging (info-preserving, *not*
  max-pool) did not unlock signal — it degraded. The early-epoch tell (train loss fitting
  *looser* than baseline) was right. So early pooling was NOT starving the model of spatial
  precision.
- **Soft-argmax = TIE** (−0.0008). A position-valued readout (expected (x,y) per channel,
  DSNT-style) is *equivalent* to magnitude attention-pooling here. Conceptually cleaner, no
  gain → not adopting.
- **Worst subject 0.5932 in BOTH** — identical to ~15 prior experiments. The long tail is
  untouched by any front-end change (invariance vs equivariance, resolution, readout). This is
  the strongest evidence yet that **the floor is data/subject-bound, not architecture-bound**.

**Takeaway:** the equivariance angle was well-grounded but empirically doesn't hold for this
data. The ~0.075 (50ep) / ~0.082 (20ep) floor is robust to front-end architecture. Remaining
leverage is almost certainly in the DATA (the 0.5932 tail subject — bad crops/landmarks/pose?)
or the personalization mechanism, not the vision front end. All flags (`cnn_late_pool`,
`readout=soft_argmax`) stay default OFF; code kept (clean, gated) for reference.

**Infra note:** the autonomous queue worked end-to-end — detached driver waited for the high-res
run, pulled the soft-argmax commit, ran it, and `runtime.unassign()`'d the v6e from a *subprocess*
(confirmed: subprocess CAN self-terminate the Colab VM). Results recovered from W&B
(`eye-tracking-hybrid-highres` / `-softargmax`) after the VM shut down.

## 2026-06-08 13:56 EDT — Hybrid refinement screen (20ep, shared seed): factorized-pos & LayerScale NULL; stoch-depth inconclusive

Screened 3 transformer refinements with **short 20-epoch runs at a shared seed** (new
`--random-seed` flag seeds Keras init, so baseline vs variant differ ~only by the change —
makes short-run deltas trustworthy). Hybrid, bf16, full data, two-phase eval, v6e TPU. The
point: cheaply triage directions before spending 50-epoch runs.

| variant (20ep, seed 42)   | train loss | cross median | vs base |
|---------------------------|-----------:|-------------:|--------:|
| baseline                  | 0.0487     | 0.0822       | —       |
| factorized_2d pos-emb     | 0.0490     | 0.0847       | +0.0025 |
| layer_scale 1e-4          | 0.0496     | 0.0855       | +0.0033 |
| stochastic_depth 0.1      | 0.1855     | 0.2080       | undertrained |

**Verdict:**
- **factorized_2d & LayerScale = NULL.** Near-identical train loss AND ~equal-or-slightly-worse
  cross-phase at matched seed. The "two eyes side by side → factorized row/col pos-emb helps"
  idea didn't pan out (1-D learned pos-emb already suffices; attention-pool may reduce reliance
  on exact positional encoding). LayerScale does little on a shallow 4-layer transformer. **Both
  dropped** (flags stay default OFF).
- **stochastic_depth = inconclusive (and unpromising).** Train *and* eval both high → simply
  didn't converge in 20ep (the expected regularizer caveat — not an eval bug, train loss confirms).
  But the slowdown is *severe* for 5% avg drop. Likely architecture-specific: drop-path injects
  per-frame embedding noise that the **per-subject ridge then fits its calibration on**, and the
  linear solve *amplifies* it instead of averaging — so drop-path may be fundamentally mismatched
  to the calibration-ridge design. Would need a 50ep run to confirm; deprioritized.

**Method win:** shared-seed 20ep screening (~30 min/variant) killed 2 of 3 refinements in ~2 hrs
with zero 50ep runs. Caveat reaffirmed: short runs are decisive for *architectural* changes,
inconclusive for *regularizers* (they pay off only near convergence). Baseline 20ep cross-median
0.0822 vs the 50ep 0.0752 — the ~0.007 gap is why variants are judged against the 20ep baseline.

**Takeaway:** none of the three knobs help → the hybrid+attn-pool (0.075) stands as-is. Next
leverage is elsewhere (confirm 0.075 + merge; capacity scaling; the worst-subject 0.593 tail;
or the sample↔calibration cross-attention idea), not more transformer-refinement knobs.

## 2026-06-08 11:29 EDT — Hybrid backbone + attention-pooling readout: FIRST real cross-phase win (0.087 → 0.075)

First time the **hybrid** (CNN-stem → transformer) backbone has ever been seriously trained,
with one architectural fix: replaced the position-locked `per-token Dense(32) → Flatten →
Dense(10368→200)` readout with an **attention-pooling (MAP) head** — a single learnable query
cross-attends over the 324 tokens → one `(B, 256)` vector → the existing `Dense(200, tanh)`.
Removes ~1.8M params and, more importantly, is translation-robust (Flatten was position-locked
and fought the aug). Same change applied to ViT. Branch `hybrid-attn-pool`; bf16, bs5, 50ep,
full data, two-phase eval (W&B `eye-tracking-hybrid-attn-pool`, run `genial-sound-3`/`qqq8cs0l`).

| model                          | cross **median** | cross mean | p90    | worst  |
|--------------------------------|-----------------:|-----------:|-------:|-------:|
| **hybrid + attn-pool** (new)   | **0.0752**       | 0.1086     | 0.2256 | 0.5932 |
| densenet — current-code base   | 0.0872           | 0.1183     | 0.2369 | 0.5932 |
| densenet — historical best fp32| 0.0840           | 0.1126     | —      | 0.593  |

(within-phase also improved: median 0.0632 vs ~0.065.)

**Verdict: REAL improvement — the first to move the floor all session.** Cross-phase median
0.0752 vs ~0.087 densenet = **−0.012 (~14%)**, ≈ **2.4× the ~0.005 run-to-run noise band**, so
almost certainly not noise. Likely mechanism: fixing the flatten readout let the model actually
exploit the attention backbone, and CNN-stem+transformer has more useful capacity here than
densenet. **Caveats:** (1) single seed — outside noise but a 1–2 seed confirmation would bank
it (deferred); (2) **worst subject 0.5932 is unchanged** — the same stubborn tail no experiment
touches; the win is entirely in the bulk/median.

**Infra (the hard part — see CLAUDE.md):** the hybrid does 324-token attention over bs5×144=720
images/step. **T4 was a dead end:** fp32 OOMs (needs ~38 GB), and fp16 won't compile (`Unsupported
dot precision algorithm: 4` — the fp16 attention dot needs sm_80+; densenet never hit this because
convs use cuDNN, not `dot_general`). **Solved on a v6e-1 TPU (32 GB):** bf16 is native (no algorithm
error), fits bs5. Two TPU gotchas: (a) a TPU is **single-process** — had to restart the kernel to
release it (an early `jax.devices()` grabbed it) and run training as the sole TPU process, never
importing jax in the kernel again; (b) the per-subject `linalg.solve` in the ridge layer — the thing
most likely to break — **compiles and runs fine on XLA-TPU**. ~250 ms/step, ~87 s/epoch, ~75 min/run.

**Next:** confirm with a 2nd seed (deferred); push the hybrid further (overlapping conv tokenizer,
factorized row/col positional embeddings, stochastic depth/LayerScale); then decide on merge
(`hybrid` is already the config default, so merging makes the default backbone actually good).

## 2026-06-07 22:14 EDT — Landmark R1 fusion + R0 ridge intercept: NULL/marginal; keep flags OFF

Tested whether explicit head/eye **geometry** from the (already-stored) MediaPipe landmarks
helps cross-phase. 4-arm A/B, same batch, full data / 50ep / fp16 / L4, λ0.1, two-phase eval
(W&B `eye-tracking-landmark-r1`). Branch `landmark-r1`.

| arm                     | flags                       | cross median | mean   | worst  |
|-------------------------|-----------------------------|-------------:|-------:|-------:|
| baseline                | —                           | 0.0893       | 0.1186 | 0.5932 |
| **R0 ridge intercept**  | `--ridge-bias`              | **0.0873**   | 0.1179 | 0.5932 |
| R1 iris-corner fusion   | `--use-landmarks`           | 0.0885       | 0.1203 | 0.5932 |
| R1 + R0                 | both                        | 0.0879       | 0.1187 | 0.5932 |

**Verdict: NULL/marginal.** All four within 0.0893 ± 0.002 — inside the ~0.005 run-to-run
noise band, worst-subject 0.5932 identical everywhere. **R1 geometry adds nothing** the CNN
hadn't already extracted (0.0885 ≈ baseline) — exactly as the oracles predicted (R1-alone 0.145
≪ CNN 0.087; pose-interactions/nonlinearity didn't generalize). **R0 (intercept)** is the only
arm ahead (0.0873, −0.0020) and lands on the best baseline ever seen (0.0872) — directionally
positive + principled + near-free, but not a clear single-seed win. Decision: `use_landmarks`
and `ridge_bias` stay **default OFF**; keep R0 as a cheap option worth a multi-seed re-check.

**Methodology win:** the cheap NumPy "oracle" (replicate the per-subject linear ridge offline
on hand-built features) called this outcome before any GPU — R1 weak, pose additively ≈random,
interactions/nonlinearity overfit cross-phase. Saved training the raw-1434 version.

**Bugs fixed (commit a45c2ef) — caught only on real data, synthetic tests missed both:**
(1) `make_augment_fn` rebuilt the input dict and dropped `Input_All_Landmarks` → R1 arms
crashed under augmentation (now copies the dict, swaps only the image; landmarks pass through
un-augmented). (2) R0 intercept was left UNregularized → near-singular ~20-point solve → NaN
eval; now regularized like every dim (`XtX + λI` stays PD). Lesson: local synthetic shape
checks don't catch augmentation-path or numerical-conditioning bugs; smoke-test on 1 real
record before a full sweep.

**Convergence:** 50 epochs is sufficient — train loss flat over the last ~10 epochs (<1% of
total drop; LR annealed to ~1% of peak). It's a plateau, not a cutoff, so a longer schedule
is unlikely to help. (Periodic eval-during-training would be a worthwhile add to rule out
back-half eval overfitting.)

## 2026-06-07 19:25 EDT — L2-normalize embedding: NULL result (best-vs-best tie); keep default OFF

Tested whether unit-L2-normalizing the per-image embedding before the ridge solve helps.
Full λ-sweep, both arms, current code, full data, 50ep, fp16, L4. Eval = two-phase held-out
(uf2sh), cross-phase = the real metric. (W&B project `eye-tracking-l2norm-sweep`.)

| arm           | λ      | cross median | cross mean | p90    | worst  |
|---------------|--------|-------------:|-----------:|-------:|-------:|
| baseline      | 0.01   | 0.0890       | 0.1207     | 0.2426 | 0.5932 |
| baseline      | 0.1    | 0.0904       | 0.1196     | —      | 0.5932 |
| **baseline**  | **1.0**| **0.0872**   | 0.1183     | 0.2369 | 0.5932 |
| L2            | 0.0003 | 0.0890       | 0.1202     | 0.2432 | 0.5932 |
| L2            | 0.001  | 0.0897       | 0.1203     | —      | 0.5932 |
| **L2**        | **0.003**| **0.0865** | 0.1217     | 0.2499 | 0.5932 |
| L2            | 0.01   | 0.0882       | 0.1193     | 0.2466 | 0.5932 |

**Verdict: NULL.** Best-vs-best = baseline 0.0872 (λ1.0) vs L2 0.0865 (λ0.003) — a 0.0007 gap,
well within run-to-run noise. Both arms are flat ~0.087–0.090 across the whole λ range. The
worst-subject value is **0.5932 in all 7 runs** — the long tail is one stubborn subject that
neither normalization nor λ touches. The norm-uniformity check foreshadowed this (real-image
embedding-norm CV only ~7%, so there was little per-image imbalance for L2 to equalize).

**Decision:** `l2_normalize_embedding` stays **default OFF**. The flag/code stay in (harmless,
off). Takeaway: tweaking the embedding's *scale/conditioning* doesn't move cross-phase error —
the missing ingredient is geometry/head-pose signal, not embedding hygiene.

Side note: the freshly-trained current-code baseline (~0.087–0.090) is ~0.005 worse than the
historical densenet checkpoint (0.0840, old notebook code / fp32). Likely fp16 + pipeline
differences + init/calibration variance; the L2 A/B is judged against the current-code
baseline (0.0872), apples-to-apples.

## 2026-06-07 16:24 EDT — First REAL two-phase number (re-baseline): cross-phase median 0.084

Ran the just-ported two-phase eval on the canonical historical densenet checkpoint —
W&B run **`gf1zjs9v` (clean-totem-19)**, densenet/embed200/ridge0.1/50ep, in project
`eye-tracking-dense-full-data-set-single-eye` (NOT `...-serializable-model`; the analysis
notebook had the wrong project, and the run is the only saved densenet model in W&B).
Eval set = OSF `uf2sh` (257 files; 246 subjects had full phase-2 targets), L4 runtime.

| eval                              | mean   | median | IQR            | p90   | worst |
|-----------------------------------|-------:|-------:|----------------|------:|------:|
| **cross-phase** (cal P1 → test P2)| **0.1126** | **0.0840** | [0.050, 0.151] | 0.231 | 0.593 |
| within-phase (cal P2 → test P2)   | 0.0998 | 0.0648 | [0.040, 0.132] | 0.213 | 0.748 |

**Takeaways:**
- **The honest number is ~0.084 median / 0.113 mean cross-phase**, vs the optimistic
  within-session figure (~0.071) we'd been quoting. Accuracy experiments must beat the
  cross-phase median, not the within-session number.
- **Temporal generalization costs ~0.02** (cross 0.084 vs within 0.065 median): calibrating
  in an earlier phase and predicting a later one is genuinely harder, as designed.
- **Long right tail** — mean >> median (0.113 vs 0.084), worst subject ~0.59. The mean is
  dragged by a few bad subjects; reporting the full distribution (not just the mean) matters.

**Loading gotcha (for next time):** the full `full_model.keras` does NOT deserialize under
current Keras — `SimpleTimeDistributed`'s serialization drifted ("Cannot convert '144' to a
shape"). Parity-proven workaround: build the architecture from current code
(`create_masked_model(TrainConfig(backbone="densenet"))`) and `load_weights(
"full_model.weights.h5")` — the 143-weight list is version-stable even when the saved
architecture isn't. `evaluate()` then rebuilds at 288 and copies weights.

## 2026-06-07 16:12 EDT — Ported the two-phase (288-point) held-out eval back into the package

**Why:** the new `train.py`/`data.py` eval (`build_eval_dataset`) was a *simplified*,
optimistic eval — fixed 20-grid on the **training** records (project 6b5cd, no `phase`
field), `max_targets=144`, calibration and target points from the **same session**, and it
scored *all* points (including the calibration points). The original
`analysis-of-individual-model-run.ipynb` eval — the scientifically meaningful one — was
**lost in the monorepo port**. So our "final loss ≈ 0.071" baseline was the easy eval.

**What the real eval is:** a separate held-out set (OSF **`uf2sh`**) whose records carry a
`phase` field — 288 points/subject, 144 in each of two temporal phases that share the same
20-point grid. It **calibrates on one phase's grid points and scores the *other* phase's
non-grid points** → measures temporal generalization (user re-seats/re-lights between
phases). Run both directions: `cal_phase=1` = cross-phase (cal P1 → test P2, the real
number); `cal_phase=2` = within-phase sanity check.

**What I changed (local only, no Colab):**
- `config.py`: added `eval_tfrecord_dir`, `eval_osf_project="uf2sh"`, `eval_max_targets=288`
  (+ `OSF_EVAL_TFRECORDS` constant).
- `data.py`: `build_two_phase_eval_dataset(cfg, cal_phase)` + phase-aware load
  (`parse_single_eye_tfrecord_with_phase`, which was ported but dead until now), 288-grouping,
  and the phase-gated mask `target_mask=(1-cal_mask)*phase2_mask`, `cal_mask*=phase{cal_phase}_mask`.
- `train.py`: `evaluate()` now (1) **rebuilds the model at 288** and copies weights (the trained
  model bakes `max_targets=144` into its Input shapes; SimpleTimeDistributed folds time into
  batch so the 143-weight list is identical and transfers cleanly), (2) **scores only the
  held-out target points** (was scoring calibration points too), (3) runs both phase directions,
  (4) reports the **per-subject distribution** (mean/median/IQR/p90/worst), returns cross-phase mean.

**Verified locally (tensorflow CPU, no data):** both models build (144 vs 288 inputs, 143
weights each), weights copy. Synthetic 288-point subject → `cal_phase=1`: 20 cal / 124 targets;
`cal_phase=2`: 20 cal / 124 targets; zero cal/target overlap both ways. Matches the notebook.

**Not yet run on real data** — next Colab run must `osf -p uf2sh fetch
single_eye_tfrecords.tar.gz` into `eval_single_eye_tfrecords/`. Expect the cross-phase number
to be **worse (more realistic) than the 0.071 within-session figure**. The model is saved
*before* `evaluate()` runs, so a missing eval dir can't lose a trained model.

## 2026-06-07 11:00 EDT — L4 vs T4: real ~1.5× GPU speedup (model IS FLOP-bound)

**Why:** now that fp16 proved the model is tensor-core/FLOP-bound, tested whether an L4
(more compute) actually helps. Ran the clean GPU-bound benchmark (no aug, bs5, same
model/data) on an L4 runtime (23 GB, 12 vCPU) and compared to the T4 numbers.

| precision | T4 (step, img/s) | L4 (step, img/s) | L4 speedup |
|-----------|------------------|------------------|-----------:|
| fp32      | 226 ms, 3,186    | 133 ms, 5,414    | **1.70×**  |
| fp16      | 91 ms, 7,912     | 61 ms, 11,803    | **1.49×**  |

**Result: the L4 is genuinely faster** — 1.49× in fp16 (the real config), 1.70× in fp32.
Confirms the model is FLOP/tensor-core-bound (the L4's extra compute converts to wall-clock).
Best operating point: **L4 + fp16 = 61 ms/step** (~5× vs the original free-T4 fp32 baseline
of 303 ms/step end-to-end). fp16 loss on L4 (0.085 @ 4ep) matches fp32 (0.086) — works fine.

**Worth it? Cost-dependent.** The L4 burns ~2–3× the compute units/hour of the T4. So for
*budget/units* the T4 High-RAM + fp16 is better value-per-unit; for *wall-clock/iteration
speed* the L4 + fp16 wins (~1.5× sooner per run). Not a spec-sheet mirage — a real choice.

## 2026-06-07 10:35 EDT — fp16 convergence VERIFIED (50-epoch fp16 vs fp32, parity)

**Why:** the speedup runs were only 3–4 epochs (final loss ~0.25 ≈ undertrained), so they
didn't show whether fp16 *converges correctly* (loss-scaling underflow, fp16→fp32 ridge
hand-off, NaN). Ran a full 50-epoch fp16 vs fp32 head-to-head (augmentation ON, bs5,
160-subject subset, identical schedule).

| metric            | fp16   | fp32   |
|-------------------|-------:|-------:|
| final train loss  | 0.0767 | ~0.077 |
| **final eval loss** | **0.0712** | **0.0710** |
| NaN/Inf           | none   | none   |
| trajectory        | smooth, monotonic | smooth — tracks fp16 in lockstep at every epoch |

**Result: fp16 converges identically to fp32** (eval 0.0712 vs 0.0710 = 0.3% = noise),
matching last session's full-data fp32 (0.0753) and far below the ~0.25 random level. The
earlier 0.25 was pure undertraining, NOT an fp16 defect. No numerical instability — loss
scaling + the fp16-backbone/fp32-ridge split are sound. **fp16 is safe to use in
production training**, and combined with the High-RAM runtime delivers the ~2.5× step
speedup at no accuracy cost.

## 2026-06-07 10:05 EDT — MAJOR CORRECTION: fp16 IS a 2.48× speedup (was masked by CPU)

**Why:** the earlier "fp16 is not a speedup" claim (08:32 entry) was measured WITH
augmentation on the 2-vCPU box, where the ~266 ms aug ceiling was binding — so any GPU
compute speedup from fp16 was invisible. Re-tested cleanly in the GPU-bound regime
(no augmentation, 8-vCPU box, same VM/GPU clocks, bs5):

| precision | step time | throughput |
|-----------|----------:|-----------:|
| fp32      | 226 ms    | 3,186 img/s |
| **fp16**  | **91 ms** | **7,912 img/s** |

**fp16 = 2.48× faster.** The model **IS heavily tensor-core/FLOP-bound** — the opposite
of what the masked measurement implied. Accuracy unaffected (fp16 train loss 0.0815 vs
fp32 0.0855; eval losses matched earlier). The ridge layer stays fp32, so this is safe.

**This reverses two earlier conclusions:**
- fp16 is a BIG free win (~2.5×), not just a memory enabler — but only once you're not
  CPU-bound (i.e. on the High-RAM 8-vCPU runtime, where aug delivers 10,623 img/s > the
  7,912 img/s fp16 needs, so aug keeps up).
- **Stacked free speedup vs the original 2-vCPU fp32 baseline (303 ms/step): High-RAM T4
  + fp16 should land ~95 ms/step ≈ 3.2× — all free, no new hardware.** (with-aug fp16 on
  8 vCPU not yet directly measured; predicted GPU-bound since aug has headroom.)
- **L4 reconsidered:** since the model IS FLOP/tensor-core-bound, an L4 (~1.85× the T4's
  fp16 tensor throughput + 12× L2 cache) WOULD now help — likely another ~1.5× on top of
  fp16, for extra cost. Worth measuring before committing.

## 2026-06-07 09:40 EDT — CONFIRMED: High-RAM T4 (8 vCPU) = 1.36× faster, free

**Why:** the 09:05 analysis predicted that more vCPUs would clear the ~20% augmentation
overhead (aug parallelism scales with cores). Switched the Colab runtime to **T4
High-RAM (8 vCPU, 50 GB RAM, same T4 GPU)** and re-measured.

| metric                | 2 vCPU (free T4) | 8 vCPU (High-RAM T4) |
|-----------------------|-----------------:|---------------------:|
| aug pipeline (isolated) |     2,709 img/s |        **10,623 img/s** (3.9×) |
| with-aug bs5 step time  |   303 ms/step   |        **223 ms/step** |
| with-aug bs5 throughput |    2,376 img/s  |        **3,229 img/s** |
| GPU utilization         |     ~79%        |        **100%** |

**Result:** aug throughput scaled ~linearly with cores (2 → 8 vCPU = 3.9×), lifting it
(10,623) well above the GPU-compute ceiling (~3,000 img/s) → training is now **fully
GPU-bound (100% util)**. With-aug step time dropped **303 → 223 ms (1.36×)** with **zero
code change** — just the runtime choice. (223 even beat the 2-core no-aug floor of 239,
so this VM's GPU clocks a bit higher too.)

**Recommendation: default to the T4 High-RAM runtime** — same GPU, 4× cores, ~1.36×
faster training, free. The 2-vCPU free T4 was wasting ~20% of the GPU on aug starvation.

## 2026-06-07 09:05 EDT — GPU-headroom measurement CORRECTS the aug-bottleneck read

**Why:** the 08:40 isolation profile concluded training was "~100% augmentation-bound,
GPU idle." That ignored `tf.data` prefetch **overlap** (aug on CPU runs concurrently
with GPU training). Measured the GPU-compute-bound step time directly by training on
cached, NON-augmented data (feeds at 636k img/s, so GPU is the only limit), fp32:

| batch | GPU-bound (no aug) | with aug | aug overhead |
|-------|-------------------:|---------:|--------------|
| bs5   | 239 ms (3013 img/s)| 303 ms (2376) | +27% time, GPU ~79% busy |
| bs10  | 529 ms (2722 img/s)| 612 ms (2353) | +16% time, GPU ~86% busy |

**Correction:** the GPU is **~79–86% utilized**, NOT idle. Augmentation overlaps with
training and adds only ~16–27% overhead — the isolated 2,709 img/s aug number was
misleading because prefetch hides most of it. **Any** aug fix caps at ~1.27× (bs5
303→239 ms is the GPU-compute floor).

**Implications:**
- **GPU augmentation: not worth it.** Only ~64 ms/step of idle GPU at bs5; moving aug
  onto the GPU serializes it with training (loses the CPU/GPU overlap), so best case
  ≤1.27× and a real risk of net-slower if GPU-side aug > 64 ms. High-effort JAX warp
  rewrite for ≤1.27× = bad trade.
- **More vCPUs: the better lever** — captures the same ~20% for free (no code), capped
  at the GPU ceiling. Pick a High-RAM/L4 runtime.
- **bs5 is the GPU sweet spot** (3013 img/s) — slightly more efficient than bs10 (2722).
- Real remaining speed lever is **GPU compute** (faster GPU / cheaper model), and fp16
  didn't reduce it → on T4 this model is bandwidth/overhead-bound, not FLOP-bound.

## 2026-06-07 08:40 EDT — Input-pipeline profile: augmentation is THE bottleneck (see 09:05 correction)

**Why:** the mixed-precision benchmark showed throughput flat (~2350 img/s) across
precision and batch size → suspected the GPU is starved by the CPU input pipeline.
Profiled the `tf.data` pipeline in isolation (GPU hidden, bs5, 160-subject subset,
cache-warm), measuring img/s at each stage:

| pipeline stage                        |        img/s |
|---------------------------------------|-------------:|
| raw cached replay (decode+group once) |    1,009,000 |
| + masking, no aug                     |      636,000 |
| **+ augmentation (what training feeds on)** | **2,709** |
| affine-only (2× ImageProjectiveTransformV3) |    4,384 |
| training observed (fp16/fp32 bs5)     |       ~2,350 |

**Conclusion:** training is **~100% augmentation-bound**. Augmentation is ~235× slower
than the rest of the pipeline combined and caps it at ~2,700 img/s; observed training
throughput (~2,350) sits right under that ceiling — the GPU idles waiting for it. This
is *why* mixed precision gave no speedup and batch size didn't matter.

**Root cause:** this Colab GPU runtime has only **2 vCPUs** (`os.cpu_count()==2`), and
augmentation per subject (144 frames) costs ~**33 ms affine** (two bilinear projective
warps: affine + perspective) + ~**20 ms photometric**. Both are already vectorized; the
two `ImageProjectiveTransformV3` warps are the single biggest cost.

**Levers (next):**
- *Quick CPU win:* fold the affine+perspective into ONE projective warp (compose the
  homographies; perspective ⊇ affine) — one `ImageProjectiveTransformV3` instead of two,
  ~1.4× pipeline.
- *Big win:* move augmentation onto the idle GPU (it sits ~0% util). Largest potential
  speedup but a real change under the JAX backend (reimplement aug in keras.ops, or let
  TF aug use the GPU and manage JAX/TF memory).
- *Free win if available:* a runtime with more vCPUs (aug parallelism scales with cores).

## 2026-06-07 08:32 EDT — Mixed precision (fp16) on T4 — memory win, NOT a speedup

**Change:** added a `mixed_precision` flag to `TrainConfig`/`train.py` (`float32` |
`mixed_float16` | `mixed_bfloat16`); sets the Keras global policy before building the
model so Keras auto-wraps the optimizer in a `LossScaleOptimizer`. The ridge layer
already hard-casts to fp32, so the solve/output stay fp32. Also fixed a latent bug:
`decay_steps = max(1, ...)` so `train_epochs <= warmup_epochs` no longer crashes.

**Benchmark** (DenseNet, JAX/T4, 160-subject subset, 3 epochs, epoch-3 cache-warm):

| config      | ms/step | img/s | OOM   | final loss |
|-------------|--------:|------:|-------|-----------:|
| fp32  bs5   |     303 |  2376 | —     |     0.2580 |
| fp16  bs5   |     306 |  2353 | —     |     0.2594 |
| fp16  bs10  |     612 |  2353 | —     |     0.2590 |
| fp16  bs20  |     898 |  3207 | soft⚠️ |     0.2571 |

**Findings:**
- **fp16 is NOT a compute speedup here:** fp16 bs5 (306 ms/step) ≈ fp32 bs5 (303). The
  DenseNet is tiny (1.58M params, 36×144 imgs, few channels) so its small convs don't
  saturate T4 tensor cores, and the ridge solve stays fp32. Throughput is flat at
  ~2350 img/s across fp32-bs5 / fp16-bs5 / fp16-bs10 → the bottleneck is per-step fixed
  overhead + the CPU aug pipeline, not fp16-able FLOPs.
- **fp16's real win is MEMORY:** it fits **bs10 cleanly** (fp32 OOM'd at bs10 last
  session) and **bs20 at the edge** (runs, but a soft BFC allocator fallback —
  `rc=0`, not a hard crash). So fp16 ≈ halves activation memory as expected.
- **Throughput only improves via larger batch** (bs20: 3207 img/s, +35% vs bs5) by
  amortizing per-step overhead — and fp16 is what makes bs20 reachable at all.
- **Accuracy unaffected** (loss 0.257–0.259 across all; fp16 == fp32).

**Implication for the GPU-upgrade question:** reinforces last session — the T4 is
*memory*-limited, not FLOP-limited (fp16 bought no compute speed). A bigger GPU's value
is mostly more memory (bigger batches) + faster input feeding, not raw FLOPs. The real
wall-clock lever now is the **CPU augmentation/input pipeline**, not precision.
Keep `mixed_precision` as a memory enabler (use `mixed_float16` + bs10–16), not a speed knob.

## 2026-06-06 21:52 EDT — Batch-size sweep on T4 (GPU-upgrade question)

**Goal:** now that the T4 is compute-bound (~98% util) at the default `batch_size=5`,
test whether larger batches raise throughput (→ a bigger GPU would help) or plateau
(→ overhead-bound, upgrade buys little).

**Result:** couldn't scale — the T4 **OOMs at `batch_size=10`** (RESOURCE_EXHAUSTED).
- bs=5: ~322 ms/step, ~2240 img/s, ~98% util (works).
- bs=10 / 20 / 40: out of memory.

**Conclusion:** at this config the T4 is **memory-limited**, not throughput-testable
beyond ~bs 5–8. The footprint is dominated by the `SimpleTimeDistributed` embedding
running over (batch × 144) eye images at once. So a GPU upgrade's value here is mostly
**more memory to fit larger batches**, not raw FLOPs. Higher-leverage next test:
**mixed precision** (fp16/bf16) — frees ~half the activation memory (may let the batch
grow on the T4) and uses tensor cores for compute.

**Also:** `train.py` raises `decay_steps=0` when `train_epochs <= warmup_epochs`
(warmup is hardcoded to 2) — latent bug; use ≥3 epochs until fixed.

## 2026-06-06 21:23 EDT — Training pipeline speedups (input-bound fix)

**Context:** the first real Colab runs (DenseNet, JAX backend, T4) were GPU-starved
— `nvidia-smi` showed the GPU near 0% util while the CPU input pipeline fed it.

**What changed** (`et_util/data.py`, `train.py`, `augment.py`):
- `.cache()` the decoded/grouped images → TFRecords parsed once, not every epoch.
- Hide the GPU from TensorFlow so JAX owns it and `tf.data` augmentation runs on CPU
  (removes the JAX/TF GPU-memory contention + BFC OOM warnings).
- `.repeat()` + file-based subject count → `steps_per_epoch`; drops the full counting
  pass and the "ran out of data" warning.
- Vectorized the per-frame augmentation (removed `tf.map_fn`); dropped blur, the only
  op that forced a per-frame loop.

**Result** (full 4-epoch benchmark vs the prior run, confirmed):
- ms/step: ~1000 → ~325 (cache-warm epochs 2–4); GPU util: ~0% → ~98% (pegged).
- sec/epoch: ~382 → ~113 — roughly **3.4× faster**.
- Final eval (fixed 20-pt grid): mean normalized Euclidean distance **0.0753**.

**Learned:** the dominant cost was the `map_fn` per-image augmentation + JAX/TF GPU
contention — the speedup shows even in epoch 1, before the cache helps; `.cache()` is a
smaller incremental win on epochs 2+.

**Also noted this session:**
- `WandbMetricsLogger` crashes under `KERAS_BACKEND=jax` (calls `.numpy()` on the LR);
  replaced with a backend-agnostic epoch logger.
- The saved `full_model.weights.h5` is a **DenseNet** checkpoint, not the config-default
  `hybrid` — load it with `TrainConfig(backbone="densenet")`.
