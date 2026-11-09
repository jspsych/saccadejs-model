# web/ARCHITECTURE.md — how a trained model reaches a browser

How a model gets from a training run in this repo to a participant's browser, and how anyone
reading a dataset years later can tell which model produced it.

Two repositories, one boundary:

- **`jspsych/eye-tracking`** (here) trains the model, exports it, gates it, and **publishes**
  the artifact. `web/export/` is that publisher; `web/rig/` is the hardware timing validation.
- **`jspsych/saccadejs`** ships the browser library (**saccade.js**) and **consumes** the
  artifact. It has no dependency on `et_util`.

The boundary is the artifact, not an import. The only thing that genuinely couples the two is
the preprocessing contract — which is why it is written down in saccadejs's `CONTRACTS.md` and
enforced by fixtures generated here.

Status: written 2026-09-08, rewritten 2026-09-09 once most of it was built. What follows is
what exists unless a section says otherwise; the open items are collected at the end.

## The core idea: version the contract, not the weights file

`web/export/export_onnx.py` emits far more than weights — preprocessing (`crop → gray →
resize → clahe`, landmark indices 226/446/[27,257]/[23,253], CLAHE clip 2.0 with 8×8 tiles,
`uint8_0_255` input range), I/O names and shapes, the embedding dim, ridge λ defaults, and the
W&B provenance.

**That bundle is the model.** A browser implementing a different crop runs a different model
whatever the weights say. So the unit of release is a version, not a file, and the registry
that describes it travels with the code that loads it.

### Three version numbers

| Version | Bumps when | Lives in |
|---|---|---|
| `@saccadejs/core` semver | library code changes | saccadejs `package.json` |
| model semver | **minor** = new weights, identical interface (drop-in). **major** = preprocessing or I/O changed | `models/releases.json` |
| `contract: <int>` | anything the runtime must implement changes | the registry, checked by the library |

Model version and library version are **not** coupled and must not be. A weights swap is safe
by construction; a contract change is loud by construction.

## Where the artifact lives

The registry is `packages/core/models/releases.json` in saccadejs: one entry per published
version with its sha256, size, contract, ONNX I/O, and the W&B run it came from. It ships
inside the npm package and is importable as `@saccadejs/core/models/releases.json`, so the
library can identify a model without a network call.

Three homes, three jobs:

| where | job | state |
|---|---|---|
| `saccade.jspsych.org/models/eye-embedding/<version>/` | the citable runtime URL | **live** |
| npm `@saccadejs/core`, model committed in `models/` | zero-config default for bundler users | awaiting first publish |
| `assets.modelUrl` | self-hosting, for labs that must serve everything themselves | **live** |

`docs/scripts/copy-model.mjs` stages **every** served release to the site and verifies each
recorded sha256 on every build, so a swapped or corrupted model fails the docs build rather
than producing quietly wrong gaze in a browser.

There is also an unversioned `/models/eye_embedding.onnx`, which every deploy rewrites. It
exists only so older links keep working; the docs tell people not to point studies at it.

### Retention: every version stays up (decided 2026-09-09)

Once a versioned URL is published it keeps resolving. By the time anyone asks whether an old
version is still needed, it is already in someone's methods section — and 20 MB per version on
GitHub's own infrastructure is not a cost worth optimising against that. `served: false` is
for a version withdrawn deliberately, not for tidying up, and `copy-model.mjs` fails the build
if a version marked served has lost its file.

Layout in `packages/core/models/`: the current model sits at the top level as
`eye_embedding.onnx`, because `packageModelUrl()` in `assets.ts` resolves exactly that path
for bundler users. Superseded versions move into `<version>/eye_embedding.onnx`, and each
release's `file` field names where its bytes actually live. A release therefore means: move
the outgoing model down, put the new one at the top level, update both `file` fields.

### Why not a model host

Hugging Face was proposed and rejected (2026-09-08). Every argument for it — free CDN,
permissive CORS, versioned revisions, a citable landing page — is already covered by Pages plus
the registry, and saccade.js uses no HF-specific feature: it fetches a URL. Adding it would
mean another account, another credential, and a fourth copy of every model.

Zenodo has a real but narrow role: an archived, DOI-bearing copy for any version eventually
withdrawn from the site. Under the retention policy above that is not needed yet. Zenodo is
an archive, not a CDN — never fetch a model from it at experiment time.

## Provenance in the recorded data

The part that matters most for research software, and the reason the rest exists.

The tracker hashes the model bytes it actually loaded — they are already contiguous in memory
for progress reporting, so it costs one pass at startup — and looks that hash up in the
registry. The extension then writes a single column, once per session:

```
saccade_model: "eye-embedding@1.0.0"     // the bytes hash-matched that published release
saccade_model: "sha256:f4669a8398d9"     // not a published release; identified by its hash
```

A version string here is **not** copied from a URL or a filename: it is recorded because the
bytes matched. That is what makes one column sufficient, and why custom weights identify
themselves instead of reading `unknown`. `SaccadeTracker.getModelIdentity()` returns the full
picture (`sha256`, `version`, `contract`, `url`, `resolvedFrom`) for anyone who wants it.

Only trials after the model loads carry the column — the camera, and so the model, is not
touched until the first `saccade-preview` or an explicit `start()`. Trials before that have no
gaze data either.

## Promotion pipeline — half built

`.github/workflows/export-model.yml` (added 2026-09-09) is stage one: a `workflow_dispatch`
job taking a W&B run id, running on `ubuntu-latest` so the build is x86 by construction.

```
W&B run id (a workflow input)
  └─ CI, linux/x86_64 only                                    ← built
       ├─ download_checkpoint.py   (entity/project/run from the environment)
       ├─ export_onnx.py           → eye_embedding.onnx + export manifest
       ├─ make_fixtures.py         → model-I/O + ridge fixtures
       ├─ GATE check_parity_ort.py (ORT-CPU vs Keras < 1e-4)
       ├─ GATE committed .npy fixtures unchanged (same run only — Keras-side drift)
       ├─ make_preproc_fixtures.py → the crop/CLAHE ground truth, from a good OpenCV
       ├─ propose_version.py       → proposed semver + a ready-to-paste registry entry
       └─ upload the bundle as a job artifact
                                                              ← not built
       ├─ GATE saccadejs jest      (crop/CLAHE/ridge fixture tests)
       ├─ GATE held-out eval       (cross-phase median, recorded in the registry entry)
       └─ open a PR against saccadejs: model bytes + a releases.json entry
```

The two remaining gates and the PR each need something the dispatch job does not have: the
saccadejs checkout, `v1_eval_frame0` (it lives on HPC `/work`, not anywhere a runner can
reach), and a cross-repo token. Until they exist, the copy-across in `web/export/README.md`
is still done by hand — but from a gated artifact rather than from someone's laptop.

### The export is reproducible (as of 2026-09-09)

It was not, and the first CI run caught it: the same checkpoint re-exported came out with a
different sha256, which `propose_version.py` dutifully called a **minor**. It wasn't one. The
weights were bit-identical; what differed was tf2onnx's bookkeeping — nodes numbered from a
process-global counter, `id()` values baked into einsum-derived names, node and attribute
emission order, and which of several identical constants survived deduplication (taking its
TF variable path with it). `PYTHONHASHSEED` is not the cause and does not help.

`canonicalize()` in `export_onnx.py` now runs last: a topological sort with a structural
signature as the tie-break, generated names replaced by their position in that order, small
constants merged and numbered, attributes sorted. Large tensors keep their Keras-derived
names, which is what makes the graph legible in Netron. Two exports on one machine are now
byte-identical, and `export-model.yml` exports twice and compares, so it stays that way.

Across platforms one folded constant still differs in the last ULP (58 of 256 elements,
~6e-08 — TF's own constant folding on arm64 versus x86). That is not fixable from here, and
does not need to be: releases are cut only by CI, on linux/x86.

Because the identity of a release is its bytes, `propose_version.py` also refuses to call a
re-export a new version: same checkpoint + unchanged contract + different bytes is `flagged`,
not `minor`. Publishing that would change the model recorded in participants' data without
changing the model they saw.

### How the version number is decided

`web/export/propose_version.py` diffs the fresh export manifest against the published
registry and applies the rule above mechanically:

| new manifest vs. newest release | verdict |
|---|---|
| `io` (names, shapes, dtypes, `inputRange`) or `contractHash` differ | **major**, `contract` +1 |
| both identical, model sha256 differs | **minor** |
| sha256 identical | none — this is already published |
| opset differs, or the baseline predates `contractHash` | **flagged**: exits non-zero, decides nothing |

`contractHash` is a new field on each release entry: sha256 over everything a browser must
implement — the whole `preprocess` block (minus its prose `note`), the I/O names/shapes/dtypes,
`inputRange`, and the embedding dim. It exists because `releases.json` recorded the I/O but
not the preprocessing, so half of the major-vs-minor question was not answerable from the
registry alone. Opset sits outside the hash on purpose: a bump changes which onnxruntime-web
versions can *load* the file, not what the runtime must implement, so it is flagged for a
person rather than classified.

Model 1.0.0 predates the field. The first dispatch prints the hash for its bytes; backfill
that one value into its entry and every later release classifies itself.

A `minor` or `major` verdict also writes `out/generated/release_entry.json` — the registry
entry, complete but for `notes`. It is what the eventual auto-PR will commit.

**linux/x86 is a hard requirement, not a preference.** The arm64 opencv-python wheels ship the
KleidiCV HAL, whose `resize` INTER_LINEAR and CLAHE deviate from the portable C++ path the
training pipeline ran on by up to 3 gray levels (see `web/export/README.md`). A release built
on an M-series Mac would be wrong in exactly the way the fixtures exist to catch. Making CI the
only thing that can publish removes the failure mode rather than documenting it.

`WANDB_API_KEY` is an organisation-level secret as of 2026-09-09; the job reads it to pull the
checkpoint, and needs nothing else.

## Non-goals

- **No canary or gradual rollout.** The deployment surface is "which pinned version does this
  release point at" — a version bump already says it, and staged rollout would work against
  the reproducibility goal above.
- **No model-version ↔ library-version lockstep.** They move for different reasons.
- **No `.onnx` in this repo's history.** Only the web path needs it; everyone working on
  `et_util/` would pay the clone cost.

## Still open

1. **The rest of the publisher workflow**: the saccadejs fixture tests, the held-out eval
   gate, and the auto-PR. Stage one — export, gate, version proposal — is built and manual.
   Backfill `contractHash` into the 1.0.0 entry on its first run.
2. **Contract enforcement at load time.** The registry carries `contract`, and identity
   resolution reads it, but the library does not yet refuse a model whose contract it does not
   implement. Cheap, and worth doing before a second contract exists.
3. **fp16/int8 variants** (`TODO.md`). These are variants *within* a model version — same
   weights, same contract, own sha256 and own measured accuracy — selectable by EP capability
   and recorded in the provenance. No release-process change needed.
4. **First npm publish**, which lights up the `@saccadejs/core` and jsDelivr paths in
   `assets.ts`. Those defaults have never been exercised; fetch the jsDelivr URL by hand once
   after publishing.
