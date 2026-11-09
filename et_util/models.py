"""Model architectures: backbones, embedding model, and the masked calibration model.

IMPORTANT: set the Keras backend (e.g. ``os.environ["KERAS_BACKEND"] = "jax"``)
BEFORE importing this module -- importing it imports keras.

Ported from models/densenet-regression-mask-trick-model.ipynb, parameterized by
``TrainConfig`` instead of module-level globals. Behavior is intended to match
the original notebook exactly so existing weights (full_model.weights.h5) load.
"""

import keras

from et_util.config import TrainConfig, IMAGE_SHAPE
from et_util.custom_layers import (
    SimpleTimeDistributed,
    PairTimeDistributed,
    MaskedWeightedRidgeRegressionLayer,
    ResidualBlock,
    AddPositionalEmbedding,
    AttentionPooling,
    CalibrationWeightingHead,
    Factorized2DPositionalEmbedding,
    LayerScale,
    StochasticDepth,
    PatchMerging,
    SpatialSoftmaxReadout,
    ContrastiveEmbeddingDistanceLoss,
)


def _readout(x, rows, cols, num_heads, cfg):
    """Final token-sequence readout. Default attention-pool (magnitude); soft_argmax
    reads the expected (x,y) location per channel over the (rows, cols) token grid."""
    if cfg.readout == "soft_argmax":
        return SpatialSoftmaxReadout(rows, cols, name="soft_argmax")(x)
    return AttentionPooling(num_heads=num_heads, name="attn_pool")(x)


def _add_positional_embedding(x, num_tokens, hidden_dim, cfg):
    """Config-gated positional embedding. The default ("learned_1d") keeps the
    original layer + name so existing checkpoints still load; "factorized_2d"
    swaps in separable row/col tables over the configured grid.
    """
    if cfg.pos_embedding == "factorized_2d":
        if cfg.pos_grid_rows * cfg.pos_grid_cols != num_tokens:
            raise ValueError(
                f"factorized_2d grid {cfg.pos_grid_rows}x{cfg.pos_grid_cols} "
                f"!= {num_tokens} tokens"
            )
        return Factorized2DPositionalEmbedding(
            cfg.pos_grid_rows, cfg.pos_grid_cols, hidden_dim, name="add_position_embedding_2d"
        )(x)
    return AddPositionalEmbedding(num_tokens, hidden_dim, name="add_position_embedding")(x)


def _branch_postprocess(branch, cfg, sd_rate, name):
    """Optional LayerScale then StochasticDepth on a residual branch output.
    Both are no-ops (no layer inserted) when their flag is at its default 0.0,
    so the default model topology is unchanged.
    """
    if cfg.layer_scale_init > 0.0:
        branch = LayerScale(cfg.layer_scale_init, name=f"{name}_ls")(branch)
    if cfg.stochastic_depth_rate > 0.0:
        branch = StochasticDepth(sd_rate, name=f"{name}_sd")(branch)
    return branch


def create_densenet_backbone(cfg: TrainConfig) -> keras.Model:
    """DenseNet backbone (keras-hub) for 36x144x1 grayscale eye images."""
    import keras_hub  # lazy: only the densenet backbone needs it

    return keras_hub.models.DenseNetBackbone(
        stackwise_num_repeats=list(cfg.densenet_stackwise_num_repeats),
        image_shape=IMAGE_SHAPE,
    )


def create_vit_backbone(cfg: TrainConfig) -> keras.Model:
    """Vision Transformer backbone. 4x4 patches -> 324 tokens, pre-norm encoder.

    Output: (batch, vit_hidden_dim) -- a single attention-pooled vector. The old
    per-token Dense(vit_patch_projection_dim) + downstream Flatten readout was
    position-locked and fought the translation augmentation; an attention-pooling
    head reads one vector that is permutation/translation robust instead.
    """
    inputs = keras.layers.Input(shape=IMAGE_SHAPE)

    patch_embeddings = keras.layers.Conv2D(
        filters=cfg.vit_hidden_dim,
        kernel_size=cfg.vit_patch_size,
        strides=cfg.vit_patch_size,
        padding="valid",
        activation=None,
        name="patch_embedding",
    )(inputs)

    num_patches_h = IMAGE_SHAPE[0] // cfg.vit_patch_size
    num_patches_w = IMAGE_SHAPE[1] // cfg.vit_patch_size
    num_patches = num_patches_h * num_patches_w

    x = keras.layers.Reshape(
        target_shape=(num_patches, cfg.vit_hidden_dim), name="flatten_patches"
    )(patch_embeddings)

    x = _add_positional_embedding(x, num_patches, cfg.vit_hidden_dim, cfg)

    for i in range(cfg.vit_num_layers):
        sd_rate = cfg.stochastic_depth_rate * i / max(1, cfg.vit_num_layers - 1)
        x_norm = keras.layers.LayerNormalization(epsilon=1e-6, name=f"layer_norm_{i}_1")(x)
        attn_output = keras.layers.MultiHeadAttention(
            num_heads=cfg.vit_num_heads,
            key_dim=cfg.vit_hidden_dim // cfg.vit_num_heads,
            dropout=cfg.vit_attention_dropout,
            name=f"attention_{i}",
        )(x_norm, x_norm)
        attn_output = keras.layers.Dropout(cfg.vit_dropout, name=f"attn_dropout_{i}")(attn_output)
        attn_output = _branch_postprocess(attn_output, cfg, sd_rate, name=f"attn_{i}")
        x = keras.layers.Add(name=f"attn_add_{i}")([x, attn_output])

        x_norm = keras.layers.LayerNormalization(epsilon=1e-6, name=f"layer_norm_{i}_2")(x)
        mlp = keras.Sequential(
            [
                keras.layers.Dense(cfg.vit_mlp_dim, activation="gelu", name=f"mlp_{i}_dense_1"),
                keras.layers.Dropout(cfg.vit_dropout, name=f"mlp_{i}_dropout_1"),
                keras.layers.Dense(cfg.vit_hidden_dim, name=f"mlp_{i}_dense_2"),
                keras.layers.Dropout(cfg.vit_dropout, name=f"mlp_{i}_dropout_2"),
            ],
            name=f"mlp_{i}",
        )
        mlp_output = _branch_postprocess(mlp(x_norm), cfg, sd_rate, name=f"mlp_{i}")
        x = keras.layers.Add(name=f"mlp_add_{i}")([x, mlp_output])

    x = keras.layers.LayerNormalization(epsilon=1e-6, name="final_ln")(x)
    output = _readout(x, num_patches_h, num_patches_w, cfg.vit_num_heads, cfg)

    return keras.Model(inputs=inputs, outputs=output, name="ViT_Backbone")


def create_hybrid_backbone(cfg: TrainConfig) -> keras.Model:
    """Hybrid CNN (ResNet blocks, LayerNorm) -> Transformer backbone.

    CNN: stem + 3 stages (64/128/256 filters) with 2x2 pooling -> 9x36x256.
    Reshape to 324 tokens, add positional embeddings, 4 pre-norm transformer
    layers, then an attention-pooling head. Output: (batch, hybrid_transformer_hidden_dim)
    -- one pooled vector (replaces the position-locked per-patch + Flatten readout).
    """
    inputs = keras.layers.Input(shape=IMAGE_SHAPE)

    # ===== CNN STAGE =====
    x = keras.layers.Conv2D(
        cfg.hybrid_cnn_filters[0], 3, strides=1, padding="same",
        use_bias=False, name="stem_conv",
    )(inputs)
    x = keras.layers.LayerNormalization(epsilon=1e-6, name="stem_ln")(x)
    x = keras.layers.Activation("relu", name="stem_relu")(x)

    # Progressive (default): pool between stages, so stages 2-3 run at reduced
    # resolution. Late (cnn_late_pool): run every stage at full 36x144 to preserve
    # fine spatial detail, then defer both pools to after stage3. Either way the
    # feature map reaches 9x36 before tokenizing.
    late = cfg.cnn_late_pool

    for i in range(cfg.hybrid_cnn_blocks_per_stage[0]):
        x = ResidualBlock(cfg.hybrid_cnn_filters[0], stride=1, name=f"stage1_block{i}")(x)
    if not late:
        x = keras.layers.MaxPooling2D(pool_size=2, name="pool1")(x)

    for i in range(cfg.hybrid_cnn_blocks_per_stage[1]):
        x = ResidualBlock(cfg.hybrid_cnn_filters[1], stride=1, name=f"stage2_block{i}")(x)
    if not late:
        x = keras.layers.MaxPooling2D(pool_size=2, name="pool2")(x)

    for i in range(cfg.hybrid_cnn_blocks_per_stage[2]):
        x = ResidualBlock(cfg.hybrid_cnn_filters[2], stride=1, name=f"stage3_block{i}")(x)

    if late:
        # Deferred downsampling: high-res convs done. Drop 36x144 -> 9x36 with
        # information-preserving patch-merging (NOT max-pool), so the fine spatial
        # detail the high-res convs preserved is folded into the tokens, not discarded.
        merge_dim = cfg.hybrid_cnn_filters[2]
        x = PatchMerging(merge_dim, name="patch_merge1")(x)   # 36x144 -> 18x72
        x = PatchMerging(merge_dim, name="patch_merge2")(x)   # 18x72  -> 9x36

    # ===== TRANSITION TO TRANSFORMER =====
    # hidden_dim is the CNN's stage-3 channel count (the token reshape requires it),
    # so it is DERIVED from cnn_filters[2] -- narrowing the CNN narrows the whole
    # transformer consistently. At the default (256) this is byte-identical.
    hidden_dim = cfg.hybrid_cnn_filters[2]

    # Optional extra 2x2 pools before tokenizing -> fewer tokens (attention is
    # O(tokens^2), so this is the biggest compute/memory lever). 0 = 9x36=324 tokens;
    # each extra pool ~quarters the token count (1 -> 4x18=72, 2 -> 2x9=18).
    h, w = IMAGE_SHAPE[0] // 4, IMAGE_SHAPE[1] // 4   # 9, 36 after the two base pools
    for j in range(cfg.token_pool_extra):
        x = keras.layers.MaxPooling2D(pool_size=2, name=f"token_pool{j}")(x)
        h, w = h // 2, w // 2
    num_patches = h * w
    x = keras.layers.Reshape(
        target_shape=(num_patches, hidden_dim), name="reshape_to_sequence",
    )(x)
    x = _add_positional_embedding(x, num_patches, hidden_dim, cfg)

    # Pose token (2026-06-11 maximalist bet): project the standardized pose
    # vector to hidden_dim and append it as an extra token AFTER positional
    # embedding, so pose participates in self-attention at every transformer
    # layer and the pooled embedding is pose-conditioned.
    pose_in = None
    if cfg.pose_token:
        pose_in = keras.layers.Input(shape=(cfg.pose_dim,), name="pose_input")
        ptok = keras.layers.Dense(hidden_dim, name="pose_token_proj")(pose_in)
        ptok = keras.layers.Reshape((1, hidden_dim), name="pose_token_reshape")(ptok)
        x = keras.layers.Concatenate(axis=1, name="append_pose_token")([x, ptok])

    # ===== TRANSFORMER STAGE =====
    for i in range(cfg.hybrid_transformer_num_layers):
        sd_rate = cfg.stochastic_depth_rate * i / max(1, cfg.hybrid_transformer_num_layers - 1)
        x_norm = keras.layers.LayerNormalization(epsilon=1e-6, name=f"transformer_ln1_{i}")(x)
        attn_output = keras.layers.MultiHeadAttention(
            num_heads=cfg.hybrid_transformer_num_heads,
            key_dim=hidden_dim // cfg.hybrid_transformer_num_heads,
            dropout=0.0,
            name=f"transformer_mha_{i}",
        )(x_norm, x_norm)
        attn_output = keras.layers.Dropout(
            cfg.hybrid_transformer_dropout, name=f"transformer_attn_dropout_{i}"
        )(attn_output)
        attn_output = _branch_postprocess(attn_output, cfg, sd_rate, name=f"transformer_attn_{i}")
        x = keras.layers.Add(name=f"transformer_add1_{i}")([x, attn_output])

        x_norm = keras.layers.LayerNormalization(epsilon=1e-6, name=f"transformer_ln2_{i}")(x)
        mlp = keras.Sequential(
            [
                keras.layers.Dense(
                    cfg.hybrid_transformer_mlp_dim, activation="gelu",
                    name=f"transformer_mlp_dense1_{i}",
                ),
                keras.layers.Dropout(
                    cfg.hybrid_transformer_dropout, name=f"transformer_mlp_dropout1_{i}"
                ),
                keras.layers.Dense(
                    hidden_dim, name=f"transformer_mlp_dense2_{i}"
                ),
                keras.layers.Dropout(
                    cfg.hybrid_transformer_dropout, name=f"transformer_mlp_dropout2_{i}"
                ),
            ],
            name=f"transformer_mlp_{i}",
        )
        mlp_output = _branch_postprocess(mlp(x_norm), cfg, sd_rate, name=f"transformer_mlp_branch_{i}")
        x = keras.layers.Add(name=f"transformer_add2_{i}")([x, mlp_output])

    x = keras.layers.LayerNormalization(epsilon=1e-6, name="final_ln")(x)
    output = _readout(x, h, w, cfg.hybrid_transformer_num_heads, cfg)

    model_inputs = inputs if pose_in is None else [inputs, pose_in]
    return keras.Model(inputs=model_inputs, outputs=output, name="Hybrid_CNN_Transformer_Backbone")


_BACKBONES = {
    "densenet": create_densenet_backbone,
    "vit": create_vit_backbone,
    "hybrid": create_hybrid_backbone,
}


def create_embedding_model(cfg: TrainConfig) -> keras.Model:
    """Per-image embedding: rescale -> backbone -> flatten -> Dense(embedding_dim, tanh).

    With cfg.pose_token (hybrid only) the model takes [image, pose] and the
    backbone attends over image tokens + a projected pose token.
    """
    input_eyes = keras.layers.Input(shape=IMAGE_SHAPE)
    eyes_rescaled = keras.layers.Rescaling(scale=1.0 / 255)(input_eyes)

    try:
        backbone = _BACKBONES[cfg.backbone](cfg)
    except KeyError:
        raise ValueError(f"Unknown backbone: {cfg.backbone}")

    if cfg.pose_token:
        if cfg.backbone != "hybrid":
            raise ValueError("pose_token requires the hybrid backbone")
        input_pose = keras.layers.Input(shape=(cfg.pose_dim,))
        encoded = backbone([eyes_rescaled, input_pose])
        flattened = keras.layers.Flatten()(encoded)
        embedding = keras.layers.Dense(units=cfg.embedding_dim, activation="tanh")(flattened)
        if cfg.l2_normalize_embedding:
            embedding = keras.layers.UnitNormalization(axis=-1, name="embedding_l2norm")(embedding)
        return keras.Model(inputs=[input_eyes, input_pose], outputs=embedding,
                           name="Eye_Image_Embedding")

    encoded = backbone(eyes_rescaled)
    # Flatten the densenet spatial map (B,H,W,C); a no-op for the transformer
    # backbones, which already attention-pool to a single (B, hidden) vector.
    flattened = keras.layers.Flatten()(encoded)
    embedding = keras.layers.Dense(units=cfg.embedding_dim, activation="tanh")(flattened)
    if cfg.l2_normalize_embedding:
        embedding = keras.layers.UnitNormalization(axis=-1, name="embedding_l2norm")(embedding)
    return keras.Model(inputs=input_eyes, outputs=embedding, name="Eye_Image_Embedding")


def create_masked_model(cfg: TrainConfig) -> keras.Model:
    """Full calibration model: per-image embeddings -> learned calibration weights
    -> masked weighted ridge regression -> predicted (x, y) for every image.
    """
    input_all_images = keras.layers.Input(
        shape=(cfg.max_targets, *IMAGE_SHAPE), name="Input_All_Images"
    )
    input_all_coords = keras.layers.Input(shape=(cfg.max_targets, 2), name="Input_All_Coords")
    input_cal_mask = keras.layers.Input(shape=(cfg.max_targets,), name="Input_Calibration_Mask")

    embedding_model = create_embedding_model(cfg)
    extra_inputs = []
    if cfg.pose_token:
        input_all_pose = keras.layers.Input(
            shape=(cfg.max_targets, cfg.pose_dim), name="Input_All_Pose")
        extra_inputs.append(input_all_pose)
        all_embeddings = PairTimeDistributed(embedding_model, name="Image_Embeddings")(
            input_all_images, input_all_pose)
    else:
        all_embeddings = SimpleTimeDistributed(embedding_model, name="Image_Embeddings")(input_all_images)

    # Auxiliary contrastive embedding-distance loss (WebEyeTrack Stage-1): shape the
    # embedding geometry to match the gaze geometry. Identity passthrough -- only
    # inserted when weight > 0, so the default model topology is unchanged.
    if cfg.aux_contrastive_weight > 0.0:
        all_embeddings = ContrastiveEmbeddingDistanceLoss(
            cfg.aux_contrastive_weight, name="contrastive_aux"
        )([all_embeddings, input_all_coords])

    if cfg.calibration_weighting == "set_attention":
        head_inputs = [all_embeddings, input_cal_mask]
        if cfg.cal_weight_use_coords:
            head_inputs.append(input_all_coords)
        calibration_weights = CalibrationWeightingHead(
            attn_dim=cfg.cal_weight_attn_dim,
            num_heads=cfg.cal_weight_heads,
            num_layers=cfg.cal_weight_layers,
            mlp_ratio=cfg.cal_weight_mlp_ratio,
            use_coords=cfg.cal_weight_use_coords,
            name="Calibration_Weights",
        )(head_inputs)
    else:
        calibration_weights = keras.layers.Dense(
            1, activation="sigmoid", name="Calibration_Weights"
        )(all_embeddings)

    ridge = MaskedWeightedRidgeRegressionLayer(
        cfg.ridge_regularization, center_targets=cfg.ridge_center_targets,
        name="Regression")(
        [all_embeddings, input_all_coords, calibration_weights, input_cal_mask]
    )

    return keras.Model(
        inputs=[input_all_images, input_all_coords, input_cal_mask] + extra_inputs,
        outputs=ridge,
        name="MaskedEyePredictionModel",
    )
