#!/bin/bash
# Multi-frame training screen (2026-06-11) — run on a Colab TPU runtime.
#
# Two 20-epoch shared-seed arms that differ ONLY in which real frame each
# trial contributes per epoch:
#   ARM A (baseline):    --frame-mode first   (legacy single-frame recipe,
#                                              v1/RGB-correct extraction)
#   ARM B (multi-frame): --frame-mode random  (fresh draw from the trial's
#                                              settling window every epoch)
# Eval for BOTH arms: the v1-derived frame-0 two-phase set (v1_eval_frame0) —
# comparisons stay within one extraction; numbers are NOT comparable to the
# old-record baselines (BGR-era).
#
# Colab setup (cells before running this script):
#   from google.colab import auth; auth.authenticate_user()
#   !git clone https://github.com/jspsych/eye-tracking && cd eye-tracking
#   !pip install -q keras keras-hub wandb osfclient
#   import wandb; wandb.login()
#
# Init-collapse check (06-09 lesson, ~1/3 of seeds): by epoch 2 the loss must
# be clearly below ~0.22; if it's stuck there, reroll the seed for BOTH arms.

set -euo pipefail

# data: ~7 GB from GCS (GCP-internal transfer; fast + free on Colab)
mkdir -p v1_train_repack
gcloud storage rsync -r gs://webcam-jspsych-frames/v1_train_repack v1_train_repack

SEED=42
COMMON=(--frames-tfrecord-dir v1_train_repack/v1_train_frames
        --eval-tfrecord-dir v1_train_repack/v1_eval_frame0
        --train-epochs 20 --random-seed "$SEED"
        --wandb-project eye-tracking-multiframe)

for MODE in first random; do
    echo "===== ARM: frame_mode=$MODE (seed $SEED) ====="
    python -m et_util.train "${COMMON[@]}" --frame-mode "$MODE"
    mv full_model.weights.h5 "multiframe_${MODE}_s${SEED}.weights.h5"
done
