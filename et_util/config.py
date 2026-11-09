"""Central configuration: experiment hyperparameters, OSF component IDs, constants.

Credentials (OSF token/username, W&B key) are NOT stored here -- they are read
from the environment / Colab secrets at runtime.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple

# --- OSF components -------------------------------------------------------
# Private project (raw data; access-controlled)
OSF_PRIVATE_OVERALL = "2wcgp"
OSF_PRIVATE_JSON = "vmdu6"
OSF_PRIVATE_JPG = "3b6et"
OSF_PRIVATE_WEBM = "34t8d"

# Public components (consented, released dataset)
OSF_PUBLIC_JPG = "7s2ae"
OSF_PUBLIC_JSON = "6cn7a"
OSF_GLASSES_INFO = "yezk8"

# TFRecords project (final, model-ready datasets)
OSF_TFRECORDS = "6b5cd"

# Two-phase evaluation TFRecords (held-out test set with a `phase` field; see
# eval_* fields in TrainConfig). 288 points/subject, 144 per temporal phase.
OSF_EVAL_TFRECORDS = "uf2sh"

# --- Dataset constants ----------------------------------------------------
EYE_IMAGE_HEIGHT = 36
EYE_IMAGE_WIDTH = 144
IMAGE_SHAPE = (EYE_IMAGE_HEIGHT, EYE_IMAGE_WIDTH, 1)  # stitched two-eye grayscale


@dataclass
class TrainConfig:
    """All knobs for a training run. Override fields via CLI or a config file."""

    # --- Dataset ----------------------------------------------------------
    tfrecord_dir: str = "single_eye_tfrecords"
    osf_tfrecords_file: str = "single_eye_tfrecords.tar.gz"
    max_targets: int = 144            # images per subject (padding length)
    random_seed: int = 12604

    # --- Two-phase evaluation set -----------------------------------------
    # A SEPARATE held-out test set (OSF project ``uf2sh``) whose records carry
    # a ``phase`` field: 288 points/subject, 144 in each of two temporal phases
    # that share the same 20-point calibration grid. Eval calibrates on one
    # phase's grid points and scores the *other* phase's non-grid points, which
    # measures temporal generalization (the user re-seats/re-lights between
    # phases). cal_phase=1 -> cross-phase (cal P1, test P2); cal_phase=2 ->
    # within-phase (cal P2, test P2). This is the scientifically meaningful eval.
    eval_tfrecord_dir: str = "eval_single_eye_tfrecords"
    eval_osf_project: str = OSF_EVAL_TFRECORDS
    eval_osf_tfrecords_file: str = "single_eye_tfrecords.tar.gz"
    eval_max_targets: int = 288       # 144 per phase

    # --- v1 multi-frame training records (2026-06-11) --------------------
    # When set, training reads the per-trial candidate records written by
    # preprocessing/repack_v1_for_training.py (et_util/frames_data.py) instead
    # of tfrecord_dir. frame_mode: "first" = legacy-equivalent single-frame
    # baseline (v1/RGB-correct extraction); "random" = fresh draw from the
    # trial's settling-window frames every epoch (the multi-frame arm).
    # Pair with eval_tfrecord_dir pointing at the v1_eval_frame0 set so the
    # whole screen stays within one extraction.
    frames_tfrecord_dir: Optional[str] = None
    frame_mode: str = "random"        # "first" | "random"

    # --- Pose-token model (2026-06-11 maximalist bet) ---------------------
    # When True, the 7 verified head-pose features (et_util/pose.py) are
    # projected to an extra TOKEN that joins the image tokens before the
    # transformer layers, so pose participates in self-attention at every
    # layer and the pooled embedding is pose-conditioned. Requires the
    # *_pose repacked records (cand_pose / pose features) and
    # affine_augmentation=False (warps would desynchronize pose from pixels).
    pose_token: bool = False
    pose_dim: int = 7
    # tf.data cache for the grouped candidate stacks: "" = in-memory (~10 GB),
    # a path = disk-backed (safer on RAM-lean hosts), None = no cache (slow).
    frames_cache_file: Optional[str] = ""
    # Gate the affine half of augmentation (flip/rotation/zoom/translation/
    # perspective); photometric per-image augmentation is unaffected.
    affine_augmentation: bool = True
    # AdamW global-norm gradient clipping; 0 = off. DEFAULT 1.0 (2026-06-17):
    # insurance against the ~1/3 init-collapse rate (both seed-44 confirm arms
    # collapsed identically with clipnorm=0; EXPERIMENT_LOG 2026-06-17).
    clipnorm: float = 1.0
    # Run the two-phase eval every N epochs during training (0 = off).
    eval_every_epochs: int = 0

    # --- Model ------------------------------------------------------------
    # Defaults = the proven best config (promoted 2026-06-10): hybrid backbone,
    # embedding_dim=128 + token_pool_extra=1 ("tokpool1+emb128") — cross-phase
    # median 0.073 ± 0.001 across 3 seeds on the clean two-phase eval, vs ~0.075
    # for the old emb200/tokpool0 hybrid and ~0.078 densenet (EXPERIMENT_LOG
    # 2026-06-09). NOTE: historical checkpoints need their original dims passed
    # explicitly — e.g. the densenet full_model.weights.h5 was trained at
    # embedding_dim=200 (and token_pool_extra is hybrid-only).
    embedding_dim: int = 128
    # DEFAULT 2.0 (2026-06-16): training at the eval-time ridge lambda ("train-
    # matched") beat the old train-0.1 default by ~9-11% cross-phase MEDIAN at
    # eval lambda=2, robustly across center off/on (EXPERIMENT_LOG 2026-06-16).
    # Single seed, but the effect (~0.01) >> ~0.001 seed noise. eval_ridge_
    # regularization stays None so eval INHERITS this -> train & eval both lambda=2.
    ridge_regularization: float = 2.0
    # Solve-time ridge strength used at EVALUATION (and deployment). The per-subject
    # ridge is re-solved from frozen embeddings at eval, so this is a separate knob
    # from the training-time ridge_regularization and costs nothing to change on an
    # existing checkpoint. None = use ridge_regularization (the old behavior). The
    # lambda oracle (EXPERIMENT_LOG 2026-06-09; the script is in git history) put the cross-phase
    # optimum near 0.5-1.0 on frozen hybrid embeddings vs the trained 0.1.
    eval_ridge_regularization: Optional[float] = None
    # Solve the per-subject ridge on coords RELATIVE to screen center (0.5), adding
    # it back on output -- the regression predicts offsets from center, so an
    # under-determined / heavily-regularized solve falls back to (0.5, 0.5) instead
    # of the corner. A fixed prior mean (zero DOF, unlike an estimated intercept).
    # DEFAULT ON (2026-06-16): from first principles a heavily-regularized /
    # under-determined solve should fall back to screen center, not the (0,0)
    # corner -- correct regardless of the co-adaptation null. Disable with
    # --no-ridge-center-targets. Co-adaptation screen was ~null (EXPERIMENT_LOG
    # 2026-06-16); post-hoc gain is tail-grade and strict at small cal counts
    # (2026-06-15): n=5 center+lambda3 beats baseline on median/p90/worst together.
    ridge_center_targets: bool = True
    # Eval/deploy override for ridge_center_targets (None = use the training value).
    eval_ridge_center_targets: Optional[bool] = None
    min_cal_points: int = 8
    max_cal_points: int = 40
    backbone: str = "hybrid"          # "densenet" | "vit" | "hybrid"

    # L2-normalize each per-image embedding to unit norm before the ridge solve.
    # Equalizes every image's contribution to the ridge Gram matrix (X^T W X) and
    # improves conditioning. NOTE: unit norm shrinks the embedding scale by
    # ~sqrt(embedding_dim), so the effective ridge strength rises ~embedding_dim-fold
    # -- retune ridge_regularization (sweep ~100x smaller) when enabling this.
    l2_normalize_embedding: bool = False

    # --- Auxiliary contrastive embedding-distance loss --------------------
    # WebEyeTrack Stage-1 idea (arXiv:2508.19544): an auxiliary loss that aligns
    # pairwise EMBEDDING distances with pairwise GAZE-target distances, so Euclidean
    # distance in the frozen embedding space tracks gaze distance -- exactly the
    # geometry the per-subject linear ridge solve depends on. Weight on the aux term
    # added to the main ridge loss; 0.0 = OFF (no layer inserted, default model
    # byte-identical, historical checkpoints load). A co-adaptation bet (changes what
    # the backbone learns), distinct from the null frozen-embedding interface tweaks.
    aux_contrastive_weight: float = 0.0

    # --- Calibration weighting head ---------------------------------------
    # How each calibration point's ridge sample-weight is produced.
    #   "linear"        = the original context-free Dense(1, sigmoid) on the
    #                     per-frame embedding (proven default).
    #   "set_attention" = CalibrationWeightingHead: a small masked self-attention
    #                     over the calibration set, so a point's weight is computed
    #                     relative to the others (outlier/redundancy/coverage), then
    #                     fed to the same ridge. Starts identical to "linear"
    #                     (zero-init context branch) and earns the context if it helps.
    calibration_weighting: str = "linear"  # "linear" | "set_attention"
    cal_weight_attn_dim: int = 64          # set_attention: attention width
    cal_weight_heads: int = 4              # set_attention: number of heads
    cal_weight_layers: int = 1             # set_attention: stacked attention blocks (depth)
    cal_weight_mlp_ratio: float = 0.0      # set_attention: >0 adds a transformer FFN per block
    cal_weight_use_coords: bool = True     # set_attention: feed each point's (x,y)

    # DenseNet
    densenet_stackwise_num_repeats: Tuple[int, ...] = (4, 4, 4)

    # ViT
    vit_patch_size: int = 4
    vit_hidden_dim: int = 256
    vit_num_layers: int = 6
    vit_num_heads: int = 8
    vit_mlp_dim: int = 1024
    vit_dropout: float = 0.1
    vit_attention_dropout: float = 0.0

    # Hybrid CNN-Transformer. NOTE: the transformer hidden_dim is DERIVED from
    # hybrid_cnn_filters[2] (the token reshape forces them equal), so narrowing
    # the last CNN stage narrows the whole transformer -- there is no separate
    # hidden_dim knob.
    hybrid_cnn_filters: Tuple[int, ...] = (64, 128, 256)
    hybrid_cnn_blocks_per_stage: Tuple[int, ...] = (2, 2, 2)
    hybrid_transformer_num_layers: int = 4
    hybrid_transformer_num_heads: int = 8
    hybrid_transformer_mlp_dim: int = 512
    hybrid_transformer_dropout: float = 0.1
    # Extra 2x2 pools before tokenizing the CNN features (fewer transformer tokens;
    # attention is O(tokens^2) so this is the biggest compute/memory lever). 0 = 324
    # tokens; 1 -> ~72 (default, part of the proven tokpool1+emb128 config); 2 -> ~18.
    token_pool_extra: int = 1
    # HRNet-style high-resolution front end (equivariance hypothesis). False = the
    # default progressive pooling (pool after stage1 & stage2, so stages 2-3 run at
    # reduced resolution). True = run ALL conv stages at full 36x144 (preserving fine
    # spatial/"where" detail), then defer both 2x2 pools to AFTER stage3 to reach the
    # same 9x36 token grid. Downstream (tokens, transformer, head) is identical -- only
    # the CNN resolution schedule changes -- so it is a clean A/B. Much heavier FLOPs
    # (stage3 at 16x the positions), but that's the point: keep spatial precision for a
    # localization task before dropping down for attention.
    cnn_late_pool: bool = False

    # Final readout over the transformer tokens. "attention_pool" = a learned query
    # pools by magnitude (default, the proven 0.075 head). "soft_argmax" = spatial-
    # softmax: the expected (x,y) location per channel -- a position-valued readout
    # for the equivariant localization view of gaze (replaces the attention-pool).
    readout: str = "attention_pool"  # "attention_pool" | "soft_argmax"

    # --- Transformer refinements (apply to vit + hybrid) ------------------
    # All default to the proven baseline (the hybrid+attn-pool run that hit
    # cross-phase 0.075); each is an independent A/B knob.
    #
    # Positional embedding: "learned_1d" = the original flat table over all
    # tokens; "factorized_2d" = separate row + column tables (the token grid is
    # 9x36 and the image is two eyes side by side, so column position carries a
    # lot of signal). The CNN/patch stem produces a 9x36 grid for both backbones.
    pos_embedding: str = "learned_1d"          # "learned_1d" | "factorized_2d"
    pos_grid_rows: int = 9                      # IMAGE_SHAPE[0] // 4
    pos_grid_cols: int = 36                     # IMAGE_SHAPE[1] // 4
    # Stochastic depth (drop-path): max rate, scaled linearly 0..rate across the
    # transformer layers. 0.0 disables. Regularizes (the hybrid fits tighter than
    # densenet, hinting at mild overfitting).
    stochastic_depth_rate: float = 0.0
    # LayerScale: per-channel learnable residual scale, init to this constant.
    # 0.0 disables (no LayerScale layer is inserted); a small value like 1e-4
    # enables it. Stabilizes pre-norm transformer training.
    layer_scale_init: float = 0.0

    # --- Augmentation -----------------------------------------------------
    # All ranges are consumed by augment.py. Defaults reproduce the original
    # notebook's behavior exactly; ranges are in uint8 image units (0-255).
    augmentation: bool = True
    per_image_augmentation: bool = True
    blur_prob: float = 0.3
    blur_sigma_range: Tuple[float, float] = (0.5, 1.5)
    brightness_delta_range: Tuple[float, float] = (-25.0, 25.0)  # additive, uint8
    contrast_range: Tuple[float, float] = (0.9, 1.1)             # multiplicative around 127.5
    gamma_range: Tuple[float, float] = (0.9, 1.1)
    noise_prob: float = 0.4
    noise_std_range: Tuple[float, float] = (1.0, 5.0)            # gaussian std, uint8

    # --- Training ---------------------------------------------------------
    batch_size: int = 5               # subjects per batch
    train_epochs: int = 50
    initial_learning_rate: float = 1e-6
    learning_rate: float = 3e-4
    # WARMUP IS IN OPTIMIZER STEPS, NOT EPOCHS, and that is the whole point.
    # It used to be `steps_per_epoch * warmup_epochs`, so its length scaled with
    # the number of subjects:
    #     ~1747 subjects -> 349 steps/epoch -> 698 warmup steps
    #       750 subjects -> 150             -> 300
    #       200 subjects ->  40             ->  80
    # Warmup is what protects the first few hundred steps from blowing up, and
    # init-collapse (the "dead-seed plateau": loss flat ~0.22 from epoch 0, eval
    # pinned at 0.2514 with a zero-width IQR) is an early-training failure. So
    # the guard was weakest exactly where it was needed most. Worse, two arms of
    # different SIZE were never on the same schedule, which silently confounds
    # any comparison between corpora -- including synthetic (750) vs real
    # (1747), and the participant-percentage sweep in the (since removed)
    # training-set-size notebook, which varies the very quantity
    # warmup was riding on.
    #
    # 700 reproduces the promoted config to within two steps (698), so runs at
    # the historical ~1747-subject scale are unchanged; smaller runs now get the
    # same warmup instead of a proportionally shorter one.
    #
    # Set to None to restore the legacy epoch-proportional behaviour.
    warmup_steps: Optional[int] = 700
    warmup_epochs: int = 2            # legacy path; used only when warmup_steps is None
    decay_alpha: float = 0.01

    # Keras global dtype policy. "float32" (off) | "mixed_float16" | "mixed_bfloat16".
    # mixed_float16 uses fp16 activations + tensor cores (T4/Turing) and frees ~half
    # the activation memory; Keras auto-wraps the optimizer in a LossScaleOptimizer.
    # The ridge layer hard-casts to fp32 internally, so the solve/output stay fp32.
    mixed_precision: str = "float32"

    # --- Tracking / output ------------------------------------------------
    wandb_project: str = "eye-tracking-serializable-model"
    use_wandb: bool = True
    keras_model_path: str = "full_model.keras"
    weights_path: str = "full_model.weights.h5"
    # Warm start: load these weights into the freshly built model before
    # training. None = train from random init (the default, unchanged).
    #
    # Fine-tuning from a real-trained checkpoint has been a named direction
    # since EXPERIMENT_LOG 2026-08-08 ("warm-start from a real-trained
    # checkpoint, fine-tune on synthetic") and every run of it so far went
    # through an ad-hoc script in scratch, which is why none of those
    # checkpoints survive. The dims must match the checkpoint's TrainConfig --
    # load_weights fails loudly if they disagree.
    init_weights: Optional[str] = None

    @property
    def decay_epochs(self) -> int:
        """LEGACY. Only meaningful on the `warmup_steps is None` path.

        With warmup in steps, the decay window is whatever is left of the run
        after warmup (train.py computes it), so that warmup + decay still spans
        exactly `train_epochs` and the LR lands on `decay_alpha` at the end.
        """
        return self.train_epochs - self.warmup_epochs

    @property
    def image_shape(self) -> Tuple[int, int, int]:
        return IMAGE_SHAPE

    def to_dict(self) -> dict:
        return asdict(self)
