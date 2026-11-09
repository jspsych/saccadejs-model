"""Config-driven training entrypoint.

    python -m et_util.train --backbone hybrid --embedding-dim 200

The Keras backend defaults to jax (matching the original notebook), but it is
actually set in ``et_util/__init__.py``: the package __init__ runs before this
module's body and already imports keras, so the setdefault below is a backstop for
direct-file execution paths, not the thing that takes effect under ``python -m``.
Either way, an explicitly exported ``KERAS_BACKEND`` wins. Credentials are read
from the environment.
"""

import os

# Backstop only -- et_util/__init__.py has already done this by the time we get here.
os.environ.setdefault("KERAS_BACKEND", "jax")

import argparse
import dataclasses

import numpy as np
import keras

from et_util.config import TrainConfig
from et_util import data as data_mod
from et_util.models import create_masked_model
from et_util.custom_loss import normalized_weighted_euc_dist


def _hide_gpu_from_tf():
    """Make TensorFlow ignore the GPU so JAX (the training backend) owns it.

    Training runs on JAX; TF is only the tf.data input/augmentation pipeline, which
    runs fine on CPU. Hiding the GPU from TF removes the GPU-memory contention (the
    'ran out of memory' BFC warnings) and lets CPU augmentation overlap with GPU
    training. Must run before TF initializes the GPU.
    """
    import tensorflow as tf
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception as e:  # already initialized, or no GPU present -- non-fatal
        print(f"[train] could not hide GPU from TF ({e}); continuing")


class _WandbEpochLogger(keras.callbacks.Callback):
    """Log epoch metrics to W&B. Backend-agnostic -- avoids WandbMetricsLogger's
    TF-only ``optimizer.learning_rate.numpy()`` call, which raises on the JAX backend.
    """

    def __init__(self, wandb_module):
        super().__init__()
        self._wandb = wandb_module

    def on_epoch_end(self, epoch, logs=None):
        data = {f"train/{k}": float(v) for k, v in (logs or {}).items()}
        data["epoch"] = epoch
        self._wandb.log(data)


class PeriodicTwoPhaseEval(keras.callbacks.Callback):
    """Run the two-phase eval every N epochs during training (trajectory
    visibility for long runs -- piloted on the 06-11 ext-to-70ep probe)."""

    def __init__(self, cfg, model_ref, every, wandb_module=None):
        super().__init__()
        self._cfg = cfg
        self._model_ref = model_ref
        self._every = every
        self._wandb = wandb_module

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self._every != 0:
            return
        print(f"\n[periodic-eval] epoch {epoch + 1}")
        # Log the periodic median to W&B (not just stdout) so the eval trajectory
        # is recoverable off-VM -- a headless Colab run's result must never live
        # only on the session (see CLAUDE.md "Robust headless Colab execution").
        evaluate(self._cfg, self._model_ref,
                 wandb=self._wandb, epoch=epoch + 1)
        # checkpoint alongside the eval so long unattended runs survive a
        # runtime death (wandb.save syncs the file incrementally)
        ckpt = self._cfg.weights_path + ".ckpt.weights.h5"
        self._model_ref.save_weights(ckpt)
        if self._wandb is not None:
            self._wandb.save(ckpt)


def build_arg_parser() -> argparse.ArgumentParser:
    """Expose the commonly-swept knobs on the CLI; everything else uses defaults."""
    # Each knob accepts both --dash-case and --underscore_case so the parser works
    # whether invoked by hand or by a W&B sweep (which passes --underscore_case).
    d = TrainConfig()
    p = argparse.ArgumentParser(description="Train the masked calibration gaze model")
    p.add_argument("--tfrecord-dir", "--tfrecord_dir", default=d.tfrecord_dir)
    p.add_argument("--eval-tfrecord-dir", "--eval_tfrecord_dir", default=d.eval_tfrecord_dir)
    p.add_argument("--frames-tfrecord-dir", "--frames_tfrecord_dir",
                   default=d.frames_tfrecord_dir,
                   help="train from v1 per-trial candidate records (frames_data.py) "
                        "instead of --tfrecord-dir")
    p.add_argument("--frame-mode", "--frame_mode", default=d.frame_mode,
                   choices=["first", "random"],
                   help="v1 frames: 'first' = single-frame baseline arm, "
                        "'random' = per-epoch real-frame sampling arm")
    p.add_argument("--pose-token", "--pose_token", dest="pose_token",
                   action="store_true", default=d.pose_token,
                   help="append a projected head-pose token to the transformer "
                        "(requires *_pose records; forces affine aug OFF)")
    p.add_argument("--no-affine-augmentation", dest="affine_augmentation",
                   action="store_false", default=d.affine_augmentation)
    p.add_argument("--clipnorm", type=float, default=d.clipnorm)
    p.add_argument("--eval-every-epochs", "--eval_every_epochs", type=int,
                   default=d.eval_every_epochs)
    p.add_argument("--backbone", default=d.backbone, choices=["densenet", "vit", "hybrid"])
    p.add_argument("--embedding-dim", "--embedding_dim", type=int, default=d.embedding_dim)
    # Hybrid transformer shape (hybrid backbone only; hidden_dim is locked to the
    # CNN output channels by the token reshape, so it is not exposed here).
    p.add_argument("--hybrid-num-layers", "--hybrid_num_layers", type=int,
                   default=d.hybrid_transformer_num_layers)
    p.add_argument("--hybrid-num-heads", "--hybrid_num_heads", type=int,
                   default=d.hybrid_transformer_num_heads)
    p.add_argument("--hybrid-mlp-dim", "--hybrid_mlp_dim", type=int,
                   default=d.hybrid_transformer_mlp_dim)
    # Downsizing knobs (hybrid): comma-separated tuples + extra token pools.
    p.add_argument("--hybrid-cnn-filters", "--hybrid_cnn_filters", default=None,
                   help="3 ints e.g. 64,128,128 -- narrows CNN stages AND the transformer hidden dim")
    p.add_argument("--hybrid-blocks-per-stage", "--hybrid_blocks_per_stage", default=None,
                   help="3 ints e.g. 1,1,1 -- residual blocks per CNN stage")
    p.add_argument("--token-pool-extra", "--token_pool_extra", type=int, default=d.token_pool_extra,
                   help="extra 2x2 pools before tokenizing: 0=324 tokens, 1=~72, 2=~18")
    p.add_argument("--cnn-late-pool", "--cnn_late_pool", dest="cnn_late_pool",
                   action="store_true", default=d.cnn_late_pool,
                   help="run all conv stages at full 36x144, defer pooling to after stage3 (HRNet-style)")
    p.add_argument("--readout", default=d.readout, choices=["attention_pool", "soft_argmax"],
                   help="token readout: attention_pool (default) or soft_argmax (spatial-softmax)")
    # Calibration weighting head
    p.add_argument("--calibration-weighting", "--calibration_weighting",
                   default=d.calibration_weighting, choices=["linear", "set_attention"],
                   help="ridge sample-weight head: linear (default) or set_attention (context-aware)")
    p.add_argument("--cal-weight-attn-dim", "--cal_weight_attn_dim", type=int,
                   default=d.cal_weight_attn_dim)
    p.add_argument("--cal-weight-heads", "--cal_weight_heads", type=int,
                   default=d.cal_weight_heads)
    p.add_argument("--cal-weight-layers", "--cal_weight_layers", type=int,
                   default=d.cal_weight_layers, help="set_attention: stacked attention blocks (depth)")
    p.add_argument("--cal-weight-mlp-ratio", "--cal_weight_mlp_ratio", type=float,
                   default=d.cal_weight_mlp_ratio, help="set_attention: >0 adds a transformer FFN per block")
    p.add_argument("--no-cal-weight-coords", dest="cal_weight_use_coords",
                   action="store_false", default=d.cal_weight_use_coords,
                   help="set_attention: do NOT feed each point's (x,y) to the weighting head")
    p.add_argument("--ridge-regularization", "--ridge_regularization", type=float, default=d.ridge_regularization)
    p.add_argument("--eval-ridge-regularization", "--eval_ridge_regularization", type=float,
                   default=d.eval_ridge_regularization,
                   help="solve-time ridge lambda for eval/deploy (default: ridge-regularization)")
    p.add_argument("--ridge-center-targets", "--ridge_center_targets",
                   dest="ridge_center_targets", action="store_true", default=d.ridge_center_targets,
                   help="solve the ridge on coords relative to screen center (0.5); "
                        "predictions fall back to center, not the corner (DEFAULT: on)")
    p.add_argument("--no-ridge-center-targets", "--no_ridge_center_targets",
                   dest="ridge_center_targets", action="store_false", default=argparse.SUPPRESS,
                   help="disable center-targets; solve the ridge from the origin (legacy)")
    p.add_argument("--eval-ridge-center-targets", "--eval_ridge_center_targets",
                   dest="eval_ridge_center_targets", action="store_true", default=None,
                   help="force center-targets ON at eval/deploy only (free on any checkpoint)")
    p.add_argument("--no-eval-ridge-center-targets", "--no_eval_ridge_center_targets",
                   dest="eval_ridge_center_targets", action="store_false", default=argparse.SUPPRESS,
                   help="force center-targets OFF at eval/deploy only (overrides the training value)")
    p.add_argument("--l2-normalize-embedding", "--l2_normalize_embedding",
                   dest="l2_normalize_embedding", action="store_true", default=d.l2_normalize_embedding)
    p.add_argument("--aux-contrastive-weight", "--aux_contrastive_weight", type=float,
                   default=d.aux_contrastive_weight,
                   help="weight on the WebEyeTrack-style contrastive embedding-distance "
                        "aux loss (0 = off; aligns embedding geometry with gaze geometry)")
    p.add_argument("--min-cal-points", "--min_cal_points", type=int, default=d.min_cal_points)
    p.add_argument("--max-cal-points", "--max_cal_points", type=int, default=d.max_cal_points)
    p.add_argument("--batch-size", "--batch_size", type=int, default=d.batch_size)
    p.add_argument("--random-seed", "--random_seed", type=int, default=d.random_seed)
    p.add_argument("--init-weights", "--init_weights", default=d.init_weights,
                   help="warm start: load these weights before training "
                        "(weights only, not optimizer state). Default: random init.")
    p.add_argument("--train-epochs", "--train_epochs", type=int, default=d.train_epochs)
    p.add_argument("--learning-rate", "--learning_rate", type=float, default=d.learning_rate)
    p.add_argument("--warmup-steps", "--warmup_steps", type=int, default=d.warmup_steps,
                   help="LR warmup length in OPTIMIZER STEPS (default %(default)s). "
                        "Absolute on purpose: warmup used to be warmup_epochs x "
                        "steps_per_epoch, so it shrank with the dataset (698 steps "
                        "at ~1747 subjects, 80 at 200) and left small runs least "
                        "protected against init-collapse. Pass 0 to restore the "
                        "legacy epoch-proportional behaviour.")
    p.add_argument("--mixed-precision", "--mixed_precision", default=d.mixed_precision,
                   choices=["float32", "mixed_float16", "mixed_bfloat16"])
    # Transformer refinements (vit/hybrid only; no-ops for densenet)
    p.add_argument("--pos-embedding", "--pos_embedding", default=d.pos_embedding,
                   choices=["learned_1d", "factorized_2d"])
    p.add_argument("--stochastic-depth-rate", "--stochastic_depth_rate", type=float,
                   default=d.stochastic_depth_rate)
    p.add_argument("--layer-scale-init", "--layer_scale_init", type=float,
                   default=d.layer_scale_init)
    p.add_argument("--no-augmentation", dest="augmentation", action="store_false", default=d.augmentation)
    # Per-op photometric control, so the four ops the vectorized path actually
    # applies (brightness -> contrast -> gamma -> noise, then clip) can be
    # neutralised one at a time. Identity values: brightness "0,0",
    # contrast "1,1", gamma "1,1", noise-prob 0.
    p.add_argument("--brightness-delta-range", "--brightness_delta_range", default=None,
                   help="two floats 'lo,hi' (uint8 additive). '0,0' disables.")
    p.add_argument("--contrast-range", "--contrast_range", default=None,
                   help="two floats 'lo,hi'. '1,1' disables.")
    p.add_argument("--gamma-range", "--gamma_range", default=None,
                   help="two floats 'lo,hi'. '1,1' disables.")
    p.add_argument("--noise-prob", "--noise_prob", type=float, default=None,
                   help="per-frame gaussian-noise probability. 0 disables.")
    p.add_argument("--no-per-image-augmentation", dest="per_image_augmentation",
                   action="store_false", default=d.per_image_augmentation,
                   help="disable the PHOTOMETRIC half (brightness/contrast/gamma/"
                        "noise/blur), leaving affine on. The affine half has its "
                        "own --no-affine-augmentation.")
    p.add_argument("--no-wandb", dest="use_wandb", action="store_false", default=d.use_wandb)
    p.add_argument("--wandb-project", "--wandb_project", default=d.wandb_project)
    return p


def _parse_int_tuple(s, default):
    """Parse a comma-separated int string into a tuple; None -> the config default."""
    if s is None:
        return default
    return tuple(int(v) for v in str(s).split(","))


def _parse_float_tuple(text, default):
    """'0.9,1.1' -> (0.9, 1.1); None/'' -> default."""
    if not text:
        return default
    parts = [float(v) for v in str(text).replace(" ", "").split(",") if v != ""]
    if len(parts) != 2:
        raise ValueError(f"expected two comma-separated floats, got {text!r}")
    return (parts[0], parts[1])


def config_from_args(args) -> TrainConfig:
    d = TrainConfig()
    return TrainConfig(
        tfrecord_dir=args.tfrecord_dir,
        eval_tfrecord_dir=args.eval_tfrecord_dir,
        frames_tfrecord_dir=args.frames_tfrecord_dir,
        frame_mode=args.frame_mode,
        pose_token=args.pose_token,
        affine_augmentation=args.affine_augmentation and not args.pose_token,
        clipnorm=args.clipnorm,
        eval_every_epochs=args.eval_every_epochs,
        backbone=args.backbone,
        embedding_dim=args.embedding_dim,
        hybrid_cnn_filters=_parse_int_tuple(args.hybrid_cnn_filters, d.hybrid_cnn_filters),
        hybrid_cnn_blocks_per_stage=_parse_int_tuple(args.hybrid_blocks_per_stage, d.hybrid_cnn_blocks_per_stage),
        hybrid_transformer_num_layers=args.hybrid_num_layers,
        hybrid_transformer_num_heads=args.hybrid_num_heads,
        hybrid_transformer_mlp_dim=args.hybrid_mlp_dim,
        token_pool_extra=args.token_pool_extra,
        cnn_late_pool=args.cnn_late_pool,
        readout=args.readout,
        calibration_weighting=args.calibration_weighting,
        cal_weight_attn_dim=args.cal_weight_attn_dim,
        cal_weight_heads=args.cal_weight_heads,
        cal_weight_layers=args.cal_weight_layers,
        cal_weight_mlp_ratio=args.cal_weight_mlp_ratio,
        cal_weight_use_coords=args.cal_weight_use_coords,
        ridge_regularization=args.ridge_regularization,
        eval_ridge_regularization=args.eval_ridge_regularization,
        ridge_center_targets=args.ridge_center_targets,
        eval_ridge_center_targets=args.eval_ridge_center_targets,
        l2_normalize_embedding=args.l2_normalize_embedding,
        aux_contrastive_weight=args.aux_contrastive_weight,
        min_cal_points=args.min_cal_points,
        max_cal_points=args.max_cal_points,
        batch_size=args.batch_size,
        random_seed=args.random_seed,
        # MUST be listed here: config_from_args builds TrainConfig from an
        # EXPLICIT field list, so a new argparse flag that is not added to this
        # call is parsed and silently discarded -- the model trains from random
        # init while the log looks fine. Same shape as the `--restlessness` dead
        # config in simulated-eyes FINDINGS 2.6. Caught because the warm-start
        # print did not appear and epoch-1 loss sat at the random-init value.
        init_weights=args.init_weights,
        train_epochs=args.train_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        mixed_precision=args.mixed_precision,
        pos_embedding=args.pos_embedding,
        stochastic_depth_rate=args.stochastic_depth_rate,
        layer_scale_init=args.layer_scale_init,
        augmentation=args.augmentation,
        # Must be listed here or the flag is silently dropped -- see the
        # init_weights comment above for the same trap.
        per_image_augmentation=args.per_image_augmentation,
        # Same trap as init_weights: absent from this list == silently ignored.
        brightness_delta_range=_parse_float_tuple(args.brightness_delta_range, d.brightness_delta_range),
        contrast_range=_parse_float_tuple(args.contrast_range, d.contrast_range),
        gamma_range=_parse_float_tuple(args.gamma_range, d.gamma_range),
        noise_prob=d.noise_prob if args.noise_prob is None else args.noise_prob,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
    )


def run(cfg: TrainConfig):
    """Build data + model, train, save, and evaluate on the fixed calibration grid."""
    wandb = None
    if cfg.use_wandb:
        import wandb as _wandb
        wandb = _wandb
        wandb.init(project=cfg.wandb_project, config=cfg.to_dict())

    # --- Determinism --- (seed Keras/NumPy/backend before building anything so
    # weight init is reproducible. For A/B screening this is what makes baseline
    # vs variant differ ~only by the change, not by random init -- essential for
    # trusting short-run deltas. tf.data augmentation ordering isn't fully
    # deterministic, but init is the dominant variance source.)
    keras.utils.set_random_seed(cfg.random_seed)
    print(f"[train] random seed = {cfg.random_seed}")

    # --- Mixed precision --- (set the global policy BEFORE building the model so
    # its dtype_policy is picked up; Keras then auto-wraps the optimizer in a
    # LossScaleOptimizer for mixed_float16. The ridge layer stays fp32 internally.)
    if cfg.mixed_precision != "float32":
        keras.mixed_precision.set_global_policy(cfg.mixed_precision)
        print(f"[train] mixed precision: global policy = {cfg.mixed_precision}")

    # On the JAX/TPU backend the fused splash-attention kernel rejects the hybrid
    # model's token counts (the MAP head's q_seq_len=1; 72 image tokens not
    # divisible by 128) -- it raises per-call, spams benign fallback tracebacks,
    # and runs ~3x slower than the standard attention path. Force standard
    # attention. (Was previously done in the Colab notebook cell; needed here for
    # headless `python -m et_util.train` runs.)
    if keras.backend.backend() == "jax":
        keras.config.disable_flash_attention()
        print("[train] jax backend: flash/splash attention disabled")

    # --- Data --- (hide the GPU from TF first so JAX owns it; aug runs on CPU)
    _hide_gpu_from_tf()
    if cfg.frames_tfrecord_dir:
        from et_util import frames_data
        print(f"[train] v1 frames training: dir={cfg.frames_tfrecord_dir} "
              f"frame_mode={cfg.frame_mode} pose_token={cfg.pose_token}")
        builder = (frames_data.build_train_dataset_v1_pose if cfg.pose_token
                   else frames_data.build_train_dataset_v1)
        train_ds, n_individuals = builder(
            cfg, cfg.frames_tfrecord_dir, frame_mode=cfg.frame_mode,
            cache_file=cfg.frames_cache_file)
    else:
        train_ds, n_individuals = data_mod.build_train_dataset(cfg)
    steps_per_epoch = max(1, n_individuals // cfg.batch_size)

    # --- LR schedule (warmup -> cosine decay) ---
    # Warmup is an ABSOLUTE number of steps (cfg.warmup_steps), not a count of
    # epochs, so it does not shrink when the dataset does -- see the comment on
    # TrainConfig.warmup_steps for why that mattered. Decay then fills whatever
    # is left of the run, so warmup + decay == the actual run length and the LR
    # still lands on decay_alpha at the final step.
    total_steps = max(1, steps_per_epoch * cfg.train_epochs)
    if cfg.warmup_steps:
        # Clamp so a short smoke run (a few subjects, 3 epochs) cannot be ENTIRELY
        # warmup, which would train at ~initial_learning_rate the whole way.
        warmup_steps = min(int(cfg.warmup_steps), max(1, total_steps // 2))
    else:
        warmup_steps = steps_per_epoch * cfg.warmup_epochs          # legacy path
    # Guard decay_steps >= 1: CosineDecay rejects a non-positive decay window.
    decay_steps = max(1, total_steps - warmup_steps)
    print(f"[train] schedule: {steps_per_epoch} steps/epoch x {cfg.train_epochs} "
          f"epochs = {total_steps} steps; warmup {warmup_steps}, decay {decay_steps}")
    if cfg.warmup_steps and warmup_steps < int(cfg.warmup_steps):
        print(f"[train] WARNING warmup clamped {cfg.warmup_steps} -> {warmup_steps} "
              "(half the run) -- this run is too short to warm up properly")
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=cfg.initial_learning_rate,
        decay_steps=decay_steps,
        alpha=cfg.decay_alpha,
        warmup_target=cfg.learning_rate,
        warmup_steps=warmup_steps,
    )

    # --- Model ---
    model = create_masked_model(cfg)
    if cfg.init_weights:
        # Warm start. AFTER set_random_seed so the (discarded) random init is
        # still reproducible, and BEFORE compile so the optimizer is fresh --
        # this loads weights only, never optimizer state, so the LR schedule
        # starts from step 0 as the printed schedule says it does.
        model.load_weights(cfg.init_weights)
        print(f"[train] WARM START from {cfg.init_weights}")
    opt_kwargs = {}
    if cfg.clipnorm:
        opt_kwargs["global_clipnorm"] = cfg.clipnorm
        print(f"[train] gradient clipping: global_clipnorm={cfg.clipnorm}")
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=lr_schedule, **opt_kwargs),
        loss=normalized_weighted_euc_dist,
        jit_compile=True,
    )
    model.summary()

    callbacks = []
    if wandb is not None:
        callbacks.append(_WandbEpochLogger(wandb))
    if cfg.eval_every_epochs:
        callbacks.append(PeriodicTwoPhaseEval(cfg, model, cfg.eval_every_epochs, wandb))

    model.fit(train_ds.batch(cfg.batch_size), epochs=cfg.train_epochs,
              steps_per_epoch=steps_per_epoch, callbacks=callbacks)

    # --- Save ---
    model.save(cfg.keras_model_path)
    model.save_weights(cfg.weights_path)
    if wandb is not None:
        wandb.save(cfg.keras_model_path)
        wandb.save(cfg.weights_path)

    # --- Evaluate on the fixed 20-point calibration grid ---
    mean_loss = evaluate(cfg, model, wandb=wandb)

    if wandb is not None:
        wandb.finish()
    return model, mean_loss


def _build_eval_model(cfg: TrainConfig, model):
    """Rebuild the model at ``eval_max_targets`` (288) and copy the trained weights.

    The training model bakes ``max_targets`` (144) into its Input shapes, but the
    two-phase eval feeds 288-length sequences. None of the weights depend on the
    sequence length (SimpleTimeDistributed folds time into the batch axis), so the
    weight list is identical and can be copied directly.
    """
    solve_lam = (cfg.eval_ridge_regularization
                 if cfg.eval_ridge_regularization is not None
                 else cfg.ridge_regularization)
    solve_center = (cfg.eval_ridge_center_targets
                    if cfg.eval_ridge_center_targets is not None
                    else cfg.ridge_center_targets)
    eval_cfg = dataclasses.replace(cfg, max_targets=cfg.eval_max_targets,
                                   ridge_regularization=solve_lam,
                                   ridge_center_targets=solve_center)
    eval_model = create_masked_model(eval_cfg)
    eval_model.set_weights(model.get_weights())
    if solve_lam != cfg.ridge_regularization:
        print(f"[eval] solve-time ridge lambda = {solve_lam} "
              f"(training lambda was {cfg.ridge_regularization})")
    if solve_center != cfg.ridge_center_targets:
        print(f"[eval] solve-time ridge center_targets = {solve_center} "
              f"(training value was {cfg.ridge_center_targets})")
    return eval_model


def _per_subject_target_loss(eval_ds, predictions, calibration_points):
    """Mean normalized Euclidean error over each subject's held-out TARGET points only.

    Calibration points (target_mask==0) and padding are excluded, so the score
    reflects only genuinely held-out predictions.

    Subjects WITHOUT a complete calibration set are skipped: a subject is scored only
    if it has a calibration frame covering EVERY point in ``calibration_points`` (in the
    chosen phase). Under-calibrated subjects make the per-subject ridge ill-posed -- with
    no/partial calibration the solve is degenerate and the "error" is meaningless and
    model-independent, which was inflating the mean/worst (see EXPERIMENT_LOG 2026-06-09).
    The requirement is derived from ``calibration_points`` itself, so it adapts if the
    calibration grid is changed.
    """
    cp = np.asarray(calibration_points, dtype=np.float32)
    per_subject = []
    n_incomplete = 0
    for i, (inputs, labels, target_mask) in enumerate(eval_ds.as_numpy_iterator()):
        # require a COMPLETE calibration set: every calibration point covered by a cal frame
        cal_mask = np.asarray(inputs["Input_Calibration_Mask"]).reshape(-1).astype(bool)
        cal_coords = np.asarray(inputs["Input_All_Coords"]).reshape(-1, 2)[cal_mask]
        covered = all((np.abs(cal_coords - p).sum(axis=1) < 1e-4).any() for p in cp) if len(cal_coords) else False
        if not covered:
            n_incomplete += 1
            continue
        mask = np.asarray(target_mask).reshape(-1).astype(bool)
        if not mask.any():
            continue
        y_true = np.asarray(labels)[mask]
        y_pred = np.asarray(predictions[i])[mask]
        errs = np.asarray(normalized_weighted_euc_dist(y_true, y_pred)).reshape(-1)
        per_subject.append(float(np.mean(errs)))
    if n_incomplete:
        print(f"[eval] skipped {n_incomplete} subjects without a complete "
              f"{len(cp)}-point calibration set")
    return np.array(per_subject, dtype=np.float32)


def _summarize(name, per_subject):
    """Print + return the per-subject distribution (mean hides the long tail)."""
    s = {
        "n_subjects": int(per_subject.size),
        "mean": float(np.mean(per_subject)),
        "median": float(np.median(per_subject)),
        "p25": float(np.percentile(per_subject, 25)),
        "p75": float(np.percentile(per_subject, 75)),
        "p90": float(np.percentile(per_subject, 90)),
        "worst": float(np.max(per_subject)),
    }
    print(f"[eval] {name}: n={s['n_subjects']}  mean={s['mean']:.4f}  "
          f"median={s['median']:.4f}  IQR=[{s['p25']:.4f}, {s['p75']:.4f}]  "
          f"p90={s['p90']:.4f}  worst={s['worst']:.4f}")
    return s


def evaluate(cfg: TrainConfig, model, wandb=None, calibration_points=None, epoch=None):
    """Two-phase held-out eval: calibrate on one phase's grid, score phase-2 targets.

    Runs both directions -- cal_phase=1 is the real cross-phase (temporal
    generalization) number; cal_phase=2 is the within-phase sanity check.
    Reports the per-subject distribution and returns the cross-phase mean.

    Only subjects with a COMPLETE calibration set (a frame covering every point in
    ``calibration_points``) are scored -- under-calibrated subjects give a degenerate
    ridge solve. ``calibration_points`` (default: the fixed grid) drives both the
    dataset's calibration mask and this filter, so they stay in lock-step if changed.
    """
    if calibration_points is None:
        calibration_points = data_mod.default_calibration_grid()
    eval_model = _build_eval_model(cfg, model)

    phase_names = {1: "cross-phase (cal P1 -> test P2)", 2: "within-phase (cal P2 -> test P2)"}
    summaries = {}
    per_subject_by_phase = {}
    for cal_phase in (1, 2):
        if cfg.pose_token:
            from et_util import frames_data
            eval_ds = frames_data.build_two_phase_eval_dataset_pose(
                cfg, cal_phase=cal_phase, calibration_points=calibration_points)
        else:
            eval_ds = data_mod.build_two_phase_eval_dataset(
                cfg, cal_phase=cal_phase, calibration_points=calibration_points)
        predictions = eval_model.predict(eval_ds.batch(1))
        per_subject = _per_subject_target_loss(eval_ds, predictions, calibration_points)
        per_subject_by_phase[cal_phase] = per_subject
        summaries[cal_phase] = _summarize(phase_names[cal_phase], per_subject)

    # Headline number = cross-phase mean (the meaningful generalization metric).
    mean_loss = summaries[1]["mean"]
    print(f"Final cross-phase mean normalized Euclidean distance: {mean_loss:.4f}")

    if wandb is not None:
        log = {"final_loss_mean": mean_loss}
        # When called from PeriodicTwoPhaseEval, tag the epoch in the SAME log
        # call so the median trajectory plots against epoch (one W&B step).
        if epoch is not None:
            log["periodic_eval/epoch"] = epoch
        for cal_phase, name in phase_names.items():
            key = "cross_phase" if cal_phase == 1 else "within_phase"
            for stat, val in summaries[cal_phase].items():
                log[f"eval/{key}/{stat}"] = val
            table = wandb.Table(
                data=[[s] for s in per_subject_by_phase[cal_phase]], columns=["Loss"]
            )
            log[f"eval/{key}/hist"] = wandb.plot.histogram(
                table, value="Loss", title=f"Per-subject loss: {name}"
            )
        wandb.log(log)
    return mean_loss


def main():
    args = build_arg_parser().parse_args()
    run(config_from_args(args))


if __name__ == "__main__":
    main()
