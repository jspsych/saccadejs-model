# TODO — Open Experiment Ideas

Forward-looking backlog of experiments to try. The backward-looking record of what
we ran and learned lives in `EXPERIMENT_LOG.md` — when an item here is finished,
write its result there and check it off (or delete it) here.

Keep this list honest: note prior evidence / de-risk status on each item so we don't
re-run something the oracles already killed.

_Last updated: 2026-09-08._

## Browser follow-ups (2026-09-05; see EXPERIMENT_LOG 09-05)

> The browser runtime now lives in **`jspsych/saccadejs`** (`packages/core`). Items here that
> touch the shipping library are done *there*; what stays in this repo is the export pipeline
> (`web/export/`) and the timing rig (`web/rig/`).

- [ ] **Real-frame preprocessing fixtures.** Run `web/export/make_fixtures.py --cv-python
  /tmp/cvx86/bin/python` from an interactive terminal (camera prompt) to un-skip the last
  `crop.test.ts` case — the composed crop→gray→resize→CLAHE chain on real webcam frames.
- [ ] **Quantify the FaceMesh → FaceLandmarker landmark shift.** Training crops came from
  legacy `mp.solutions` FaceMesh; the browser uses the Tasks model. Same indices, slightly
  different coordinates. Measure on a few frames (crop bbox delta in px) and, if material,
  either re-extract eval embeddings with Tasks landmarks or nudge the pad.
- [ ] **ORT graph capture** (`enableGraphCapture`) needs GPU-resident I/O tensors
  (`preferredOutputLocation: "gpu-buffer"` + a persistent input GPUBuffer). Isolated embed
  is 7 ms; live it's ~15 ms under contention with the MediaPipe GPU delegate — capture
  may claw some back. Low priority: already camera-capped at 30 fps.
- [ ] **Multi-subject validation of the live pipeline** (a few people, 13-pt cal, 9-pt
  validation, λ∈{1,3}, TTA N∈{1,5}) before quoting the 0.036 number anywhere.
- [ ] fp16/int8 quantization of the ONNX for download size (20.6 MB) — only if deployment
  needs it; WebGPU speed is not the bottleneck.

## Temporal synchrony (2026-09-05; see EXPERIMENT_LOG 09-05 10:16)

- [ ] **Spread across hardware.** One Mac says 82 ms (sparse) / 81.5 ms (m-seq). Run
  the rig panel (`web/rig/panel`, `npm run dev`; sparse default, 15 s) on other laptops, external webcams, Windows,
  Safari/Firefox; collect the JSON exports. Tight spread → a fixed constant per device
  class; wide → run the loopback per session (15 s, no flicker, so it's cheap).
- [ ] **Lab validation of the loopback method** (Bridges et al. 2020 style): photodiode on
  the screen + camera-visible LED on a Black Box Toolkit, to decompose display vs camera
  lag and confirm macOS Chrome `captureTime` tracks exposure, not delivery. The whole field
  method rests on that assumption. Also check exposure duration (motion blur window).
- [ ] **Recorder** producing the {x, y, t} series with t = `FrameTime.capture`, raw
  per-frame (no ring-buffer smoothing), plus per-session QC (loopback lag, latency,
  dropped frames). Feed the jsPsych extension packaging item above; store t in the
  `performance.now()` clock and convert to trial-relative like the WebGazer extension.
- [ ] Behavioral gross-error check: target jumps, gaze should follow after ~200 ms; only
  catches sign errors / multi-hundred-ms offsets, not calibration-grade.
- Closed: webcam auto-exposure *did* break the sparse mode (biased low); fixed by the
  edge-difference estimator rather than exposure locking (macOS Chrome doesn't expose
  `exposureMode`). See EXPERIMENT_LOG 09-05.

## Deferred follow-ups (2026-06-17)

- **Contrastive embedding-distance loss (WebEyeTrack borrow #1): SMALL BENIGN POSITIVE, low
  headroom — diagnosed, mostly closed.** Branch `contrastive-embedding-loss` (`aux_contrastive_weight`
  knob). Matched A/B (seed 42/20ep/uf2sh): aux=0.3 median **0.0727** vs baseline **0.0743** (−0.0016),
  better on every percentile. HPC ridge-map diagnostic (06-19) explains why it's small: the embedding
  is **already ~3-4D and 0.87 gaze-correlated** in the baseline, so little headroom; and the aux loss
  does NOT harm per-subject structure (per_subject_gain unchanged, map_specificity ~flat) — so the
  per-subject-friendly relaxation (triplet/rank) is a **non-problem, DROPPED**. Remaining only IF
  curiosity-priced: one 2nd-seed confirmation @ w=0.3 (effect is ~1.5× seed noise). Default stays
  `aux_contrastive_weight=0`.
- **Drop `embedding_dim` below 128 — BLOCKED by init-collapse (06-20), needs an init fix first.**
  Motivation holds (eff-dim ~4, 128 oversized), but emb {64,32,16} @ seed 42 / default LR 3e-4 ALL
  init-collapsed (median 0.2514; emb128 same seed trains fine). Narrow widths are init-fragile.
  To revisit: lower LR (~1e-4) and/or warmup 2→5ep (expose `warmup_epochs` on the CLI first) or a
  check-and-reroll seed loop, AND re-baseline emb128 at the matched setting. Lower priority than
  data-side levers given the small contrastive effect. (EXPERIMENT_LOG 06-20.) Secondary WebEyeTrack
  borrow — the spatial-frequency-weighted regression loss — still open, lower prior.
- **Eval-λ optimum sweep** — DEFERRED. Sweep eval λ ∈ {0.1,0.5,1,1.5,2,3} over the 4
  good centered checkpoints (train λ{0.1,2.0} × seed{42,43}; all recoverable from
  W&B `lambda-center-*` / `lambda-confirm-s43-*`) to confirm λ=2 is the median-optimal
  eval point (λ-oracle hinted ~0.5–1.0 on frozen embeddings). **Run on the HPC**
  (`emc`/SLURM) per the job-routing rule, not Colab — two Colab attempts thrashed.
  Eval-only, ~free. Script staged: `/tmp/eval_lambda_sweep.py` (logs incrementally).
- **3rd clean confirmation seed** (optional) — train λ{0.1,2.0}+center at seed 45 with
  `--clipnorm 1.0`; seed 44 init-collapsed. 2/2 already confirms the win, so low priority.
- **First-time HPC eval setup** — verify jax/keras-3 in `~/eye-tracking/.venv`, stage
  `v1_eval_frame0` on `/work`, confirm W&B/outbound access; then eval jobs go there.

## Where things stand

- **Headline metric:** cross-phase median normalized Euclidean distance on the
  two-phase held-out set (OSF `uf2sh`), `cal_phase=1` (cal P1 → test P2), restricted
  to fully-calibrated subjects (the complete-calibration filter now lives in
  `train.py::_per_subject_target_loss`). Beat *this*, not the within-session figure.
- **Current best: multi-seed median 0.073 ± 0.001 (3 healthy seeds, 06-09)** — hybrid +
  attention-pooling at **tokpool1+emb128**, NOW THE DEFAULT CONFIG (promoted + merged to
  main 06-10; `TrainConfig()` builds it, historical checkpoints need explicit dims).
  3-seed prediction ensemble: median 0.0721 / mean 0.0934. Densenet clean baseline:
  0.0782 @ λ0.1 (0.0749 @ solve-λ0.3).
- **Solve-time λ is settled (06-09): a TAIL knob, not a median knob.** Median is already
  optimal at the trained λ=0.1 for the hybrid (the learned calibration weights cover what
  shrinkage would add — redundant mechanisms); λ≈0.5 buys mean −0.004 / p90 −0.015 / worst
  0.37→0.30 for ~0.002 median. `eval_ridge_regularization` + `analysis/sweep_eval_lambda.py`
  make this free on any checkpoint. Default stays None. Densenet still gains median (0.0749
  @ 0.3). Per-subject λ (LOO/GCV): NULL (see Closed/null).
- **The "0.5932 worst-subject tail" is SOLVED (06-09): it was an eval artifact**, not a model
  limit — un-calibratable subjects (cal=0 phase-1 points) fed into an ill-posed ridge. Clean
  fully-calibrated (cal>=20) numbers: **mean 0.10, median 0.078, worst 0.34** (densenet).
- **The wall (updated 06-09):** everything operating on the embedding→ridge *interface* is
  null — L2 norm, centering/standardizing, set-attention reweighting, explicit geometry (R1),
  readout variants, front-end equivariance. Three independent probes agree the
  per-subject-calibration axis has little exploitable signal in the current features. The
  floor looks **data-bound**. The "session-shift augmentation" idea was retired on analysis
  before running (see Closed/null): the geometric half breaks the per-subject coherence that
  makes uniform affine aug valid, and the photometric half is already covered by per-image
  augmentation. What remains on that axis is to *measure* the real phase-1→2 shift (near-term
  item 1), plus the two cheap levers the existing results already point at (solve-time λ,
  seed ensembling).

## Methodology — cheap before expensive

Before any GPU build, replicate the per-subject linear ridge offline in numpy on
FROZEN embeddings / hand-built features ("the oracle") and score it cross-phase. It
costs ~2 min and no GPU, and has correctly predicted the null results so far. Only
graduate an idea to a trained run if the oracle shows signal. See
`analysis/calibration_centering_oracle.py` for the current template.
**Limit:** the oracle can't see backbone *co-adaptation* — anything that changes what
the CNN learns (e.g. augmentation changes) needs a 20ep shared-seed screen instead.

## Open experiments

### Near-term, ranked (cheap, high prior)

- [ ] **0. LR recipe for the long retraining runs (notes from 06-11 ext-to-70ep probe).**
  Context: arm-B 20ep→0.0775, then a warm restart (+50ep) dipped to **0.0706-0.0713 by
  ep 60-70** — but spent ep 30-40 WORSE than the restart point (restart tax), and the
  arm-A control was never run, so "more epochs" vs "multi-frame data" vs "cycling" is
  unresolved. When the longer retrain happens, explore in this order:
  1. **Planned single 70ep cosine** (no restart stitching) as the baseline arm — also
     doubles as the arm-A control question.
  2. **Weight EMA** — `AdamW(use_ema=True, ema_momentum≈0.99)`, one flag; smooths the
     end-of-run checkpoint noise we observed (0.0706@60 vs 0.0713@70) and steadies the
     ridge-sensitive embeddings. Subsumes the old "EMA/SWA last ~10 epochs" note (item 3a).
  3. **SGDR restarts ONLY with snapshot ensembling** (average predictions of per-cycle
     minima) — restarts must earn their tax; seed-ensembling already proven (06-09), one
     90ep/3-cycle run ≈ a 3-member ensemble.
  4. **`clipnorm=1.0`** (and/or warmup 2→5ep) to attack the 1/3 init-collapse rate —
     sweep-throughput insurance, not a median knob.
  Keep the every-10-epochs eval from the 06-11 extension run (it's what made the
  trajectory visible); skip decay-shape/one-cycle experiments (noise at this scale).

- [x] **1. Measure the real phase-1→2 shift — DONE (06-11), DOUBLE NULL.** No recoverable
  affine-nuisance component (aligned crops slightly WORSE; the landmark-anchored crop box
  already removes what an affine can remove) and head-pose deltas don't predict either the
  cross-phase penalty (ρ≤0.14) or the per-subject mapping distortion (CV R²≈0). The gap is
  mapping-level AND invisible to landmarks — no appearance augmentation can touch it, and
  the head-context stream's mechanism gate failed (see Data/input section). The distortion
  itself is large (gain sd ~0.2) — cause unknown (kappa angle? lighting? cal quality).
  (`analysis/phase_shift_gate.py`, `EXPERIMENT_LOG` 06-11 20:42.)
  ~~Measure the real phase-1→2 shift (oracle-style gate) — before any more aug work.~~
  The cross-phase penalty is ~0.012–0.02 (within-phase 0.063 vs cross-phase 0.075); before
  trying to train against it, characterize what actually changes between phases in `uf2sh`,
  per subject: eye-region landmark geometry deltas (roll, scale, translation), brightness/
  contrast deltas, and — the decisive test — fit a per-subject affine alignment from phase-2
  landmarks to phase-1, warp phase-2 crops into phase-1's frame, and check whether cross-phase
  error on FROZEN embeddings drops. If alignment helps, there is an affine-like nuisance
  component worth training against (or correcting at inference); if it doesn't, the gap is
  mapping-level (parallax / head-pose changing the appearance→screen relationship) and NO
  appearance augmentation can touch it. Cheap (landmarks are stored; oracle template exists).
  Context: this replaces the broader "session-shift augmentation" idea, retired on analysis —
  see Closed/null.
- [x] **2. Solve-time λ retune — DONE (06-09).** Oracle said median −0.006 @ λ≈0.7; the real
  eval (learned weights) showed the median gain was already captured by the weighting head —
  λ is a tail knob for the hybrid (worst 0.37→0.30 @ 0.5), a real median gain only for
  densenet (0.0782→0.0749 @ 0.3). Per-subject λ NULL. Knob + sweep script merged; default
  unchanged. Full numbers in `EXPERIMENT_LOG` 06-09 19:45.
- [ ] **3. Multi-seed baseline + ensembling — MOSTLY DONE (06-09).** 3 healthy seeds (42/123/789):
  **multi-seed baseline median 0.073 ± 0.001**, judge future deltas against THIS. 3-seed
  prediction ensemble: median 0.0721, mean 0.0934 (beats every member's mean/p90) —
  `analysis/ensemble_eval.py`. Seed-456 COLLAPSED at init (1/3 observed rate; detectable by
  epoch 2, loss stuck ~0.22 — check-and-reroll in future sweeps). Default-config promotion +
  branch merges to main: DONE 06-10. REMAINING: (a) within-run EMA/SWA of the last ~10
  epochs; (b) R0 ridge-intercept multi-seed re-check (needs `--ridge-bias` merged from
  `landmark-r1`).

- [ ] **Loss-mask low-EAR target frames in training (QA gating done right) — batch with
  the v1 retrain screens.** Don't REMOVE bad frames from training (the weighting head
  needs to see blinks as cal candidates to learn to discount them, and input noise with
  correct labels is benign); instead stop them from GENERATING LOSS as targets — the
  supervision "predict gaze from closed eyes" is unlearnable and just teaches
  predict-the-prior. Mechanically: store an EAR-ok bit per candidate in
  `repack_v1_for_training.py`, then `target_mask &= ear_ok` in the masking step.
  Expected effect small (blinks ≈ 2-5% of target frames; this week's marginal training
  interventions all tied) — 20ep shared-seed screen, NOT oracle-testable (co-adaptation).
  Threshold from the 06-11 QA inspection: EAR < ~0.75 × subject median ≈ the blink band
  (`analysis/qa_gate_inspection.py --render`; the qa_*.png figures are no longer committed).
  Related but separate:
  QA-gated TTA at INFERENCE — swept 06-12, NULL (see Closed/null); doesn't change the
  training-side rationale here (loss masking is about supervision quality, not dilution).

- [x] **Pose-token run — NULL (06-12).** 150ep ties the no-pose arms at matched epochs
  (plateau 0.0693-0.0698 vs 0.0706-0.0713); co-adaptation extracts nothing from the 7
  pose features either. Pose-as-model-input is closed at every level (warp, additive,
  kernel-attention, residual ceiling, co-adapted token). Side results: plateau from
  ep~80-90 settles the training-budget question (~100ep single cosine suffices); the
  no-pose/no-affine control arm is moot given the tie. Pose's value stays where 06-11
  put it: the pre-flight gate. (`EXPERIMENT_LOG` 06-12.) Original in-flight notes:
  (a) AFFINE AUG IS FUNDAMENTALLY INCOMPATIBLE with pose conditioning, not just
  unimplemented: camera is rigid to screen, so no physical event produces "pose changed,
  appearance warped, label unchanged" — pose-adjusted warps would TRAIN THE
  POSE-CONDITIONED CORRECTION TOWARD ZERO (same 3D-unidentifiability that retired
  session-shift aug). Exception: horizontal mirror-flip IS scene-consistent if the label
  is mirrored too (x→1−x) and odd pose features negated (roll, width_asym, nose_dx) —
  a legitimate 2× aug for pose runs, not yet implemented.
  (b) The 150ep pose run differs from arm-B in TWO ways (pose token + no affine aug) —
  if it ties/loses, run the no-pose/no-affine control arm before concluding anything
  about pose. (c) TTA eval of a pose model needs per-frame pose fed alongside frame
  variants (extend eval_singleframe_vs_tta.py; landmarks per frame are in the v1 records).

### Backbone / capacity

- [x] **Transformer readout fix — DONE (the session's win).** Attention-pooling (MAP) head →
  hybrid hit cross-phase **0.075** (vs ~0.087 densenet). Settles the backbone bake-off:
  hybrid > densenet. (`EXPERIMENT_LOG`, 06-08.)
- [x] **Confirm 0.075 + merge `hybrid-attn-pool` → main — DONE (06-10).** Superseded by the
  stronger tokpool1+emb128 result (multi-seed 0.073 ± 0.001); both branches merged to main,
  tokpool1+emb128 promoted to the config default.
- [ ] **Hyperparameter sweep — partially superseded; finish only if curiosity-priced.**
  tokpool1+emb128 won and is now the default. Open ends from the original screen:
  `embedding_dim` {64, 320} arms, `heads` {16}, `layers` {6} — only worth running as cheap
  20ep screens if idle GPU; the D-vs-conditioning question that motivated the sweep is
  answered (smaller D + fewer tokens = better-conditioned ridge).

### Data / input information (the strategic bets — aligned with the "data-bound" diagnosis)

The high-res experiment (`cnn_late_pool`) only changed how the network *processed* the same
36×144 input — it never tested whether the input itself is the bottleneck. If the floor is
data-bound, the lever is what goes IN. These are the expensive items; reach for them when
the near-term list plateaus.

- [x] **Higher-resolution eye crops — KILLED by the native-resolution audit (06-11).**
  The premise was false: per-subject median native crop is 146×38 (train corpus) vs the
  current 144×36 input — 0% of training subjects have 2× pixels, 3% have 1.5×. Upscaling
  isn't information. Revisit only if future data collection uses higher-res cameras.
  Surviving variant: **RGB color at the same resolution** (hires crops are color;
  grayscale is a real discard) — cheap 20ep screen, needs a color-crop repack + first
  conv layer change. (`EXPERIMENT_LOG` 06-11 21:05.)
- [ ] **Face/head-context input stream (iTracker-style) — TRIPLE-GATED NULL (06-11),
  deprioritized with confidence.** Three independent probes: (1) affine warp
  decomposition — no recoverable geometric nuisance; (2) pose deltas (incl. verified
  yaw/pitch proxies) + distributional coverage — predict error LEVEL (ρ≈0.5, → gate
  features) but not the penalty or the distortion structure; (3) pose-fusion oracle —
  additive fusion, pose-keyed kernel attention, AND the population residual ceiling
  (leave-subject-out, linear+RBF) all ≤ +0.0005: **no function of landmark-visible pose
  corrects predictions.** Remaining blind spot: co-adapted full-face appearance (an
  oracle can't see it) — but the prior is now very low. `frame_small_jpeg` stays in the
  v1 records if a new angle appears. (`EXPERIMENT_LOG` 06-11 20:42 + 21:26.)
- [ ] **External pretraining.** With only a few hundred training subjects, pretrain the
  backbone on GazeCapture (~1,450 subjects) or MPIIFaceGaze, then fine-tune. Standard move
  for exactly this regime.

- [ ] **WebEyeTrack-inspired ideas (compared 2026-06-17; arXiv:2508.19544 /
  github.com/RedForestAI/WebEyeTrack).** BlazeGaze is a tiny (0.16M-param, 670KB) BlazeBlock
  CNN trained on public benchmarks (GazeCapture 2.32cm, MPIIFaceGaze 4.56cm), personalized
  via **MAML** (k=9 support, 5 inner SGD steps), deployed in-browser via TF.js. Their
  personalization is gradient-based; ours is a closed-form ridge on frozen embeddings — ours
  is arguably simpler/more robust for the browser, so don't chase MAML. The two borrowable
  pieces, both fitting our two-phase eval harness and oracle-testable on frozen embeddings:
  1. **⭐ Contrastive embedding-distance loss (HIGHEST PRIOR).** Their Stage-1 loss aligns
     pairwise *embedding* distances with pairwise *gaze* distances. Our entire method is a
     LINEAR solve on frozen embeddings, so an embedding space where Euclidean distance ∝ gaze
     distance is exactly what makes the ridge fit well — this is the rare auxiliary loss that
     targets our architecture's actual dependency, unlike the embedding→ridge *interface*
     tweaks that are all null (it changes what the backbone LEARNS, so it's a co-adaptation
     bet, not oracle-killable a priori). Add as a `TrainConfig` knob (aux loss weight) + 20ep
     shared-seed screen vs the 0.073 baseline. NOTE: distinct from the null contrastive/
     centering work — those reshaped frozen embeddings; this trains the geometry in.
  2. **Spatial-frequency-weighted regression loss.** They reweight gaze loss by a 30×30
     inverse-frequency grid to fix ground-truth target imbalance. Our calibration-grid data
     likely has the same imbalance; cheap to add as a per-target weight in the loss. Lower
     prior than (1) but trivial.
  Also noted (NOT to copy): explicit 3D head-pose via MediaPipe reconstruction + radial
  Procrustes — our pose work is triple-gated NULL (see above), though theirs is metric-3D not
  a learned token, so only revisit if we go cross-device. Metric caveat: their cm errors are
  not comparable to our normalized-screen 0.073 without converting via eval-set screen dims.

### Personalization mechanism (mostly de-prioritized after three null probes)

- [x] **Multi-frame TTA — RESOLVED (06-11): calibration-side NULL, target-side REAL WIN.**
  Oracle on the v1 all-frames extraction (frozen hybrid `8lsofasl`, n=230): k-frame
  calibration adds nothing beyond solve-λ retuning (see Closed/null), but **averaging the
  target trial's frame embeddings: median 0.0778 → 0.0646 (−0.013) at deployed λ, 0.0615
  with λ=1 — every stat improves**. Frame 0 is the worst single target frame (settling is
  real); mid-trial best, but TTA beats any single frame. (`analysis/frame_index_sweep.py`,
  `EXPERIMENT_LOG` 06-11.) REMAINING: real-model confirmation DONE (06-11, 0.0717 → 0.0601);
  TTA × QA gating swept 06-12 — NULL, plain mean already optimal (see Closed/null). Still
  open: wire plain embedding-mean TTA into deployment (λ≈1). Discovery from the raw-data
  pipeline: each 0.5 s trial webm has **~24 frames** (640×480, ~60 fps nominal) and the server
  scripts keep only frame 1 (`extractjpg.sh`, `-vframes 1`). At deployment the system records
  video per cal point anyway, so a k-frame calibration solve is FREE for users — and strictly
  more realistic than jitter. **Pilot (no retraining):** server-side ffmpeg extracts k≈5
  frames for the EVAL set's 40 grid trials only (~50k jpgs, 1–2 GB), embed with the frozen
  checkpoint, ridge on k×20 rows, score cross-phase vs the 1-frame baseline (oracle template
  + embedding-extraction script from 06-09 are directly reusable; interacts with solve-λ —
  re-tune it on the enriched solve). **Fold in for free: frame-INDEX check** — frame 1 is at
  fixation onset + webcam autoexposure settle, plausibly the WORST frame (over/under-exposure
  was the `22nkcpfw` failure mode and a top pre-flight-gate signal); compare frame ~12 vs
  frame 1 on the same extraction. Caveats: raw webms include non-consented subjects (apply
  the notebook-01 filter list server-side); eyedata24 webms are server-only (19 GB, 95,910
  files; eyedata23 = 56 GB — see the storage-migration discussion, GCS recommended).
  Note the eval webm probed is h264-in-webm — `-vframes`-style extraction works fine.
- [ ] **Calibration-set sampling: is the random count + random locations helping or
  hurting?** Training draws a *random* number of calibration points (`MIN_CAL_POINTS=8`
  .. `MAX_CAL_POINTS=40`) at *random* frame locations, re-rolled fresh every epoch
  (`data.py` `_add_masks`), while eval always uses the fixed 20-point grid. Disentangle
  with 20ep shared-seed screens vs the current baseline:
  1. **Fixed count = 20** (still random locations) vs random 8–40 — isolates count variety.
  2. **Fixed locations = the eval 20-grid** (matches train to eval) vs random locations —
     isolates the train/eval distribution mismatch.
  3. **Range width**: narrow (e.g. 16–24) vs wide (8–40) — sensitivity of the curriculum.
  Cheap (knobs exist; locations need a small data.py option).
- [ ] **Sample ↔ calibration cross-attention (in-context refiner) — GATE FAILED, deprioritized.**
  The cheap linear special-case (centering oracle) came back null/negative, and the reweighting
  head was null too. Only revisit with a NEW angle the oracle couldn't see — e.g. backbone
  *co-adaptation*, or feeding reliability features the gaze embedding discards. Original design
  notes preserved in `EXPERIMENT_LOG` (06-09) and git history of this file.

### Eval / training hygiene

- [x] **Require minimum calibration coverage in the two-phase eval — DONE.** Implemented as the
  complete-calibration filter in `train.py::_per_subject_target_loss` (skips subjects without
  a frame covering every grid point; derived from `calibration_points` so it adapts).
- [x] **Re-report headline numbers on the clean eval — DONE (06-09).** Hybrid tokpool1+emb128:
  median 0.0720/0.0740 (seeds 42/123), mean 0.0963; densenet `gf1zjs9v`: median 0.0782, mean
  0.0999 @ λ0.1 (0.0749/0.0962 @ solve-λ0.3). All through the current code + complete-calibration
  filter (`EXPERIMENT_LOG` 06-09 19:45).
- [x] **Multi-frame (TTA) eval variant — BUILT + CONFIRMED (06-11 evening).**
  `analysis/eval_singleframe_vs_tta.py` (sharded, resumable, runs the real model): on the
  70-ep arm-B checkpoint, cross-phase median **0.0717 single → 0.0601 TTA**, 202/230
  subjects improved, every stat better — the oracle gain transfers ~fully. REMAINING:
  (a) wire frame-averaged inference into the deployment path, (b) ~~QA-gated TTA~~ NULL
  (06-12, see Closed/null), (c) report single+TTA in all future evals. ~11 subjects have no phase-2 targets, 2 have
  cal=0, some have N=1/5/6 frames. Worth reporting upstream / excluding from the released split.
- [ ] **Periodic eval-during-training.** Score the two-phase eval every N epochs to
  rule out back-half overfitting and confirm the 50-epoch plateau is real on the eval
  metric (not just train loss). Cheap instrumentation, de-risks all future runs.
- [ ] **Off-by-one in cal-count sampling (trivial).** `tf.random.uniform(maxval=cfg.max_cal_points)`
  in `data.py::_add_masks` is exclusive — training never actually sees 40 cal points (max 39).
  Fix to `maxval=cfg.max_cal_points + 1` (or document the intent).

### Infrastructure / data logistics

- [ ] **Retire the EC2 box as the sole raw-video backup.** GCS migration is done
  (2026-06-10) — raw webms are now in `gs://webcam-jspsych-webm` (56 GB eyedata23 +
  19 GB eyedata24, per-subject tars). EC2 still holds the only copy of the source
  `webm/` flat files; once confident GCS is solid, the EC2 webm dirs can be deleted
  (or the instance downsized/terminated).

### Pre-flight image-quality gate (product direction — viability confirmed 06-09)

A cheap, on-device, no-reference quality check that warns a user their setup/data is low-quality
*before* they rely on gaze predictions. We have labels (subject images → cross-phase loss).

- [x] **Viability check — DONE, VIABLE as a coarse screen.** 7 cheap image features + data-completeness
  → 5-fold CV RandomForest predicts the bad-quartile at **AUC 0.78** (image-only 0.67). Strongest
  signals (cheapest): **N frames captured** (rho -0.52), **cal points** (-0.32), **brightness
  inconsistency** (+0.45), brightness (-0.24). **Blur/sharpness weak** (-0.12, low-res crops). It's a
  screen, not a precise classifier (~20-30% rank variance). (`EXPERIMENT_LOG`, 06-09.)
- [ ] **Add cal-pose instability (ρ=+0.47) + embedding coverage (ρ=+0.52) to the gate
  (06-11 evening).** Calibration-pose variance is the strongest image-side feature found
  (beats blink_frac/sharp_cv) and maps to actionable feedback ("hold still during
  calibration"); embedding nearest-cal-neighbor coverage is similar (check overlap with
  cluster spread before counting it separately). Both predict error LEVEL, not the
  cross-phase penalty — reliability-grade, not correction-grade.
  (`analysis/phase_shift_gate.py` arm 3, `EXPERIMENT_LOG` 06-11 21:26.)
- [ ] **Fold prediction-cluster spread into the gate as the PRIMARY feature (06-11).**
  Per-subject within-trial prediction spread (no ground truth needed — any fixation
  video) predicts per-subject cross-phase error at r=0.88/ρ=0.84, and even predicts the
  bias component (r=0.84). Dwarfs every feature in the 06-09 screen (best ρ≈−0.52).
  Needs a calibrated model first, so it's a *post-calibration* gate signal — pairs with
  the image-only features for the pre-calibration tier.
  (`analysis/prediction_cluster_analysis.py`, `EXPERIMENT_LOG` 06-11.)
- [ ] **Add landmark-derived framing/visibility features — partially measured 06-11:**
  eye openness LEVEL is weak (ρ=−0.19) but **blink_frac ρ=+0.32 and sharp_cv ρ=+0.36**
  are the two strongest image-side features found so far — inconsistency beats level
  across every signal. Fold both into the pre-calibration tier; remaining from the
  original list: head-pose extremity, crop centering, landmark confidence.
  (`EXPERIMENT_LOG` 06-11 18:57.) Original item:
  eye size in frame, eye openness, head-pose extremity, landmark-detection confidence, crop centering.
  We already extract MediaPipe landmarks, so still cheap. The `22nkcpfw` failure was framing/visibility,
  invisible to brightness stats. Test if AUC jumps; these also map directly to actionable user feedback
  ("center your face", "move closer", "open your eyes", "improve/steady lighting").
- [ ] **(if pursued) Per-frame gate + UX.** Two-tier design: (1) instant data-completeness + lighting
  check, (2) landmark framing check. Per-frame labels are weak (we only have per-session loss) — would
  need per-frame proxy labels or heuristic bootstrapping. Real-time bad-frame rejection + session-level
  "redo calibration / fix setup" prompt.

## Closed / null (don't re-run without a new angle)

**Stop investing in the embedding→ridge interface.** The consistent pattern across this whole
list: manipulating embeddings or the per-subject solve *given fixed features* doesn't move the
floor. New work should change what the backbone learns (augmentation, input information,
pretraining) or exploit levers the existing results already revealed (λ, ensembling).

- **QA-gated / robust target-side TTA** — NULL (06-12, frozen-embedding sweep, n=230).
  EAR gates (0.5–0.85 × subject median), rel-sharpness gates, combined, pred-median, and
  pred-trim50 are all within noise of plain all-frames embedding-mean at both λ; the
  blink-band gate drops only ~5% of frames and the mean over ~15-24 frames already
  dilutes them. The tail is unmoved — tail subjects fail at the solve/mapping level, not
  the frame level (consistent with the 75%-bias finding). Cal-side "first EAR-ok frame"
  repair actively HURTS the worst subjects (blink swap → fixation-drift exposure). Ship
  plain embedding-mean TTA; QA bits stay in the pre-flight gate only.
  (`analysis/qa_gated_tta_sweep.py`, `EXPERIMENT_LOG` 06-12 14:25.)
- **Multi-frame TRAINING (real-frame sampling as augmentation)** — NULL (06-11, 20ep
  shared-seed screen on v1, seed 42): exact cross-phase median tie (0.0775) vs the
  single-frame baseline, tail marginally worse, train loss LOWER. Per-image synthetic aug
  likely already covers the photometric nuisance real frames carry (same redundancy that
  retired session-shift photometric aug). New angle required before re-running: temporal-
  consistency loss across a trial's frames, or an aug-off ablation. Infra is in place and
  cheap to reuse (`et_util/frames_data.py`, `v1_train_repack`, ~45 min/arm on TPU v5e).
  (`EXPERIMENT_LOG` 06-11 13:36.)
- **Multi-frame (k-frame) calibration solve** — NULL beyond solve-λ (06-11, v1 oracle,
  n=230). Best cell (k=2, λ=1.0) ties the k=1 λ-retuned solve on median AND tail; stacking
  at the deployed λ=0.1 actively hurts (under-regularization); sharpness-based frame
  selection is the worst policy; >3 frames/trial dilutes (late-frame fixation drift).
  Don't revisit without a mechanism that isn't row-enrichment of the same ridge.
  (`analysis/multiframe_calibration_oracle.py`, `EXPERIMENT_LOG` 06-11.)
- **Session-shift augmentation (split cal/target transforms)** — retired on ANALYSIS, never
  run (06-09). Geometric half: uniform affine aug is only valid because cal+target share the
  warp and the per-subject ridge absorbs it (a coherent synthetic subject); splitting the
  transforms asserts the phase-1 mapping applies unchanged to geometrically different images,
  which is false in the real generative process — a real head movement changes the
  appearance→screen mapping itself (parallax, gain/offset), and a 2D crop warp with frozen
  labels can't represent that (the correct label change is unidentifiable without 3D
  geometry). It would also force the backbone to discount iris-position cues that carry the
  gaze signal, while landmark-anchored crops make systematic affine phase shifts small
  anyway. Photometric half: already covered — default `per_image_augmentation=True` draws
  brightness/contrast/gamma/noise per FRAME, so the ridge can never absorb a shared lighting
  state and the backbone is already pushed toward per-image photometric invariance (a
  stronger signal than a phase-level split). What's left — shift *magnitude* beyond the mild
  ranges, and spatially-structured lighting (shadows/specularities) — is gated behind the
  phase-shift measurement (near-term item 1).
- **Per-subject solve-time λ via closed-form LOO/GCV** — NULL (06-09). The cheat upper
  bound is only ~−0.003 vs the best global λ, honest selectors capture none of it and
  fatten the worst-subject tail by under-regularizing. 20 cal points are too few to beat
  a good global λ. (`analysis/lambda_oracle.py`, `EXPERIMENT_LOG` 06-09.)
- **Calibration centering / standardization on frozen embeddings** — NULL/negative.
  center+bias ≡ R0 (wash); standardize+bias actively HURTS (+0.043 median). Killed the
  cross-attention refiner gate. (`EXPERIMENT_LOG`, 06-09.)
- **Context-aware calibration weighting (set-attention head)** — MARGINAL/NULL at seed 42, 40ep
  (tokpool1+emb128). All variants 0.0696–0.0739 vs linear 0.0720 (inside ~0.005 noise); base re-run
  ≈ linear, and **more capacity/depth made it worse** (`strong` 0.0739), so not under-capacity. Weak
  worst-subject shave (0.32 vs 0.366) only. Code gated + default-OFF on `personalization-attn`.
  (`EXPERIMENT_LOG`, 06-09.) New angle needed (e.g. residual-aware/IRLS reweighting, or the
  augmented-feature oracle to check if reliability features even exist to exploit).
- **Spatial-preserving front end (equivariance hypothesis)** — NULL. High-res
  (`cnn_late_pool`+PatchMerging) WORSE (+0.0066 at 2.5× compute); soft-argmax readout TIE.
  Early pooling was not the cap; floor is not architectural. Low-prior stragglers (CoordConv,
  BlurPool) not worth running unless the data work surfaces something. (06-09.)
- **L2-normalize embedding** — NULL. Flat ~0.087–0.090 across full λ sweep, both arms.
  Embedding norm CV was only ~7%, so little imbalance to fix. (06-07.)
- **Landmark R1 (iris-corner geometry) fusion** — NULL. 0.0885 ≈ baseline; CNN already
  has the geometry. Oracle predicted it (R1-alone 0.145 ≪ CNN 0.087). (06-07.)
- **Absolute head pose/position, pose×appearance interactions, RBF kernel ridge** —
  oracle-killed (additively ≈random; interactions/nonlinearity overfit cross-phase).
- **Factorized 2D positional embedding & LayerScale (hybrid)** — NULL at 20ep shared-seed
  (≈ baseline). 1D pos-emb already suffices; LayerScale does little on a 4-layer transformer. (06-08.)
- **Stochastic depth (hybrid)** — inconclusive/unpromising: undertrains at 20ep, and likely
  mismatched to the ridge (per-frame drop-path noise the calibration solve amplifies). (06-08.)
- **Worst-subject 0.5932 investigation** — RESOLVED as eval artifact (cal=0 subjects); see
  "Where things stand". The real worst fully-calibrated subject (`22nkcpfw`, 0.34) is an
  image-quality case → handled by the pre-flight gate direction, not a model mystery. (06-09.)
