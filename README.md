# eye-tracking

[![CI](https://github.com/jspsych/eye-tracking/actions/workflows/ci.yml/badge.svg)](https://github.com/jspsych/eye-tracking/actions/workflows/ci.yml)

Training code for the gaze model behind [SaccadeJS](https://github.com/jspsych/saccadejs):
a network that turns a 144 × 36 grayscale image of both eyes into a 128-dimensional
embedding plus a calibration weight, and a weighted ridge regression, fit per participant,
that maps embeddings to gaze. The browser library that runs the model lives in
[`jspsych/saccadejs`](https://github.com/jspsych/saccadejs); this repo trains the model
and exports it to ONNX for that library.

## Where things are

| Path | What |
|------|------|
| `et_util/` | The installable package. `models.py` and `custom_layers.py` build the network (the default backbone is the hybrid CNN-transformer; `create_masked_model` is the entry point), `custom_loss.py` is the loss, `augment.py` the augmentation, `frames_data.py` the input pipeline for the multi-frame training records, `train.py` the training CLI, and `config.py` every setting with its default. `process_functions.py` holds the eye-crop geometry the browser port must match. |
| `server/` | Scripts from the data-collection server: frame extraction, OSF upload, and `repack_webm.py`, which packs each participant's trial videos into one tar with a manifest. |
| `preprocessing/` | Building the training records from the videos (below), plus the two notebooks of the original pipeline: `01_process_subject_data.ipynb` (R; consent and completeness filtering) and `02_generate_tfrecords.ipynb` (the earlier one-frame-per-trial TFRecords on OSF). |
| `training/` | Scripts that ran training on a Colab TPU v5e. |
| `analysis/` | `extract_v1_embeddings.py`: runs a trained checkpoint over every frame and saves the per-frame embeddings. |
| `web/export/` | Keras to ONNX export, parity checks and fixtures; see its README. `web/ARCHITECTURE.md` describes how a trained model reaches the browser. |
| `web/rig/` | The hardware timing rig (RP2040 firmware and a Web Serial browser panel) used to measure camera latency. |
| `tests/` | pytest suite run by CI. |
| `EXPERIMENT_LOG.md` | Dated log of every experiment and change, newest first. Most entries are about approaches the final model does not use. |

## How the deployed model was trained

The model SaccadeJS ships (version 1.0.0) is the W&B run `kwppnkuj`, with the weights file
`multiframe_random_s42_ext70.weights.h5`. `web/export/download_checkpoint.py` defaults to
that run.

1. **Video to frames.** `server/repack_webm.py` packed each participant's videos into a tar.
   `preprocessing/extract_frames_to_gcs.py` (run as a SLURM array by
   `preprocessing/extract_frames.sbatch`) decodes every frame of every trial video, runs
   MediaPipe face mesh, and writes per-frame records with the eye crop, for the participants
   in `preprocessing/include_list_v1.csv`. `preprocessing/fix_v1_labels.py` corrects a
   swapped x/y label bug in those records.
2. **Training records.** `preprocessing/repack_v1_for_training.py --train` writes one example
   per trial with up to eight candidate frames (the first frame with a face, and up to seven
   from frame 3 through the fourth-to-last); `--eval` writes the held-out participants'
   first-face frames for the two-phase evaluation.
3. **Training.** `training/run_multiframe_screen_colab.sh` ran the first 20 epochs: the
   `--frame-mode random` arm, seed 42, on those records (`training/colab_bootstrap.sh` sets
   up a headless TPU VM). Training then continued from those weights for 50 more epochs,
   to 70 in total, with a fresh warmup and cosine schedule. The extension was run from a
   script that is not in this repo; `--init-weights` in `et_util/train.py`, added later,
   loads weights before training for the same purpose.
4. **Export.** `web/export/` converts the checkpoint to ONNX and checks parity.

Several defaults in `et_util/config.py` changed after this run, so the current defaults do
not reproduce it. The run used a training ridge penalty of 0.1 (`--ridge-regularization 0.1`;
the default became 2.0 on 2026-06-17), no screen-center offset in the ridge
(`--no-ridge-center-targets`; on by default since 2026-06-17), no gradient clipping
(`--clipnorm 0`; 1.0 since 2026-06-17), and a warmup measured in epochs (in `TrainConfig`,
`warmup_steps=None` with `warmup_epochs`; measured in steps since 2026-08-06).

The training records and raw videos used here are stored in the lab's private storage. The
released SaccadeJS dataset (videos and per-participant eye crops) is on Zenodo,
<https://doi.org/10.5281/zenodo.23817713>, under a data use agreement.

## Setup

```bash
pip install -e ".[dev]"   # add jax[tpu] or jax[cuda] for training
pytest
cp .env.example .env      # OSF and W&B credentials, read from the environment
```

Keras defaults to the JAX backend (set in `et_util/__init__.py`); export `KERAS_BACKEND` to
override it.

```bash
python -m et_util.train --help
```
