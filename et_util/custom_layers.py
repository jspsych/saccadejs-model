import keras
from keras import ops


@keras.saving.register_keras_serializable(package="et_util")
class SimpleTimeDistributed(keras.layers.Wrapper):
    """A simplified version of TimeDistributed that applies a layer to every temporal slice of an input.

    This implementation avoids for loops by using reshape operations to apply the wrapped layer
    to all time steps at once.

    Args:
        layer: a `keras.layers.Layer` instance.
    """

    def __init__(self, layer, **kwargs):
        super().__init__(layer, **kwargs)
        self.supports_masking = getattr(layer, "supports_masking", False)

    def build(self, input_shape):
        # Validate input shape has at least 3 dimensions (batch, time, ...)
        if not isinstance(input_shape, (tuple, list)) or len(input_shape) < 3:
            raise ValueError(
                "`SimpleTimeDistributed` requires input with at least 3 dimensions"
            )

        # Build the wrapped layer with shape excluding the time dimension
        super().build((input_shape[0], *input_shape[2:]))
        self.built = True

    def compute_output_shape(self, input_shape):
        # Get output shape by applying the layer to a single time slice
        child_output_shape = self.layer.compute_output_shape(
            (input_shape[0], *input_shape[2:])
        )
        # Include time dimension in the result
        return (child_output_shape[0], input_shape[1], *child_output_shape[1:])

    def call(self, inputs, training=None):
        input_shape = ops.shape(inputs)
        batch_size = input_shape[0]
        time_steps = input_shape[1]

        # Reshape inputs to combine batch and time dimensions: (batch*time, ...)
        reshaped_inputs = ops.reshape(inputs, (-1, *input_shape[2:]))

        # Apply the layer to all time steps at once
        outputs = self.layer.call(reshaped_inputs, training=training)

        # Get output dimensions
        output_shape = ops.shape(outputs)

        # Reshape back to include the separate batch and time dimensions: (batch, time, ...)
        return ops.reshape(outputs, (batch_size, time_steps, *output_shape[1:]))

    def get_config(self):
        config = super().get_config()
        # The parent Wrapper class handles serialization of self.layer
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        # Keras Wrapper.from_config handles deserialization of the wrapped layer
        from keras.layers import deserialize as deserialize_layer

        layer_config = config.pop("layer")
        layer = deserialize_layer(layer_config, custom_objects=custom_objects)
        return cls(layer, **config)


@keras.saving.register_keras_serializable(package="et_util")
class MaskedWeightedRidgeRegressionLayer(keras.layers.Layer):
    """
    A custom layer that performs weighted ridge regression with proper masking support.

    This layer takes embeddings, coordinates, weights, and calibration mask as explicit inputs,
    while using Keras' masking system to handle target masking. This separation allows more
    precise control over calibration points while leveraging Keras' built-in mask propagation
    for target predictions.

    Args:
        lambda_ridge (float): Regularization parameter for ridge regression
        epsilon (float): Small constant for numerical stability inside sqrt
    """

    def __init__(self, lambda_ridge, epsilon=1e-7, center_targets=False, **kwargs):
        self.lambda_ridge = lambda_ridge
        self.epsilon = epsilon
        # center_targets: solve the ridge on coords RELATIVE to screen center (0.5)
        # and add it back on output, so the regression predicts offsets from center.
        # As the solve shrinks (heavy lambda / few cal points) predictions fall back
        # to (0.5, 0.5) -- the screen middle -- instead of the origin/corner. A fixed
        # prior mean, costs zero degrees of freedom (vs an estimated intercept).
        # Free on any checkpoint (flip on at eval); bake into a retrain to let the
        # backbone co-adapt to residuals-from-center. See EXPERIMENT_LOG 2026-06-15.
        self.center_targets = center_targets
        super(MaskedWeightedRidgeRegressionLayer, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        """
        The forward pass of the layer.

        Args:
            inputs: A list containing:
                - embeddings: Embeddings for all points (batch_size, n_points, embedding_dim)
                - coords: Coordinates for all points (batch_size, n_points, 2)
                - calibration_weights: Importance weights (batch_size, n_points, 1)
                - cal_mask: Mask for calibration points (batch_size, n_points) [EXPLICIT]

        Returns:
            Predicted coordinates for the target points (batch_size, n_points, 2)
        """
        # Unpack inputs
        embeddings, coords, calibration_weights, cal_mask = inputs

        # Ensure correct dtype, especially important for JIT
        embeddings = ops.cast(embeddings, "float32")
        coords = ops.cast(coords, "float32")
        calibration_weights = ops.cast(calibration_weights, "float32")
        cal_mask = ops.cast(cal_mask, "float32")

        # reshape weights to (batch, calibration)
        w = ops.squeeze(calibration_weights, axis=-1)

        # Pre-compute masked weights for calibration points
        w_masked = w * cal_mask
        # Add epsilon inside sqrt for numerical stability
        w_sqrt = ops.sqrt(w_masked + self.epsilon)
        w_sqrt = ops.expand_dims(w_sqrt, -1)

        # Apply calibration mask to embeddings
        cal_mask_expand = ops.expand_dims(cal_mask, -1)
        X = embeddings * cal_mask_expand

        # Optionally solve for offsets from screen center (0.5); the constant is
        # added back to the output below. Non-cal rows stay zero after the mask.
        center = 0.5 if self.center_targets else 0.0
        coords_solve = coords - center

        # Weight calibration embeddings and coordinates using the masked weights
        X_weighted = X * w_sqrt
        y_weighted = coords_solve * w_sqrt * cal_mask_expand

        # Matrix operations
        X_t = ops.transpose(X_weighted, axes=[0, 2, 1])
        X_t_X = ops.matmul(X_t, X_weighted)

        # Add regularization
        identity_matrix = ops.cast(ops.eye(ops.shape(embeddings)[-1]), "float32")
        lhs = X_t_X + self.lambda_ridge * identity_matrix

        # Compute RHS
        rhs = ops.matmul(X_t, y_weighted)

        # Solve the system
        kernel = ops.linalg.solve(lhs, rhs)

        # Apply regression using the *original* full embeddings
        output = ops.matmul(embeddings, kernel) + center

        return output

    def compute_output_shape(self, input_shapes):
        """
        Computes the output shape of the layer.

        Args:
            input_shapes: List of input shapes

        Returns:
            Output shape tuple
        """
        # Output shape matches coordinates: (batch_size, n_points, 2)
        return (input_shapes[0][0], input_shapes[0][1], 2)

    def get_config(self):
        """
        Returns the configuration of the layer for serialization.

        Returns:
            Dictionary containing the layer configuration
        """
        config = super(MaskedWeightedRidgeRegressionLayer, self).get_config()
        config.update({"lambda_ridge": self.lambda_ridge, "epsilon": self.epsilon,
                       "center_targets": self.center_targets})
        return config


@keras.saving.register_keras_serializable(package="et_util")
class ResidualBlock(keras.layers.Layer):
    """
    ResNet-style residual block with LayerNorm instead of BatchNorm.

    Architecture:
        x -> Conv2D -> LN -> ReLU -> Conv2D -> LN -> Add(x) -> ReLU

    If input/output channels differ, a 1x1 conv projection is applied to the skip.
    Uses LayerNorm for consistent behavior and better Transformer compatibility.
    """

    def __init__(self, filters, stride=1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.stride = stride

    def build(self, input_shape):
        input_filters = input_shape[-1]

        self.conv1 = keras.layers.Conv2D(
            self.filters, 3, strides=self.stride,
            padding='same', use_bias=False
        )
        self.ln1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.conv2 = keras.layers.Conv2D(
            self.filters, 3, strides=1,
            padding='same', use_bias=False
        )
        self.ln2 = keras.layers.LayerNormalization(epsilon=1e-6)

        self.use_projection = (input_filters != self.filters) or (self.stride != 1)
        if self.use_projection:
            self.proj_conv = keras.layers.Conv2D(
                self.filters, 1, strides=self.stride,
                padding='same', use_bias=False
            )
            self.proj_ln = keras.layers.LayerNormalization(epsilon=1e-6)

        super().build(input_shape)

    def call(self, x, training=None):
        out = self.conv1(x)
        out = self.ln1(out)
        out = keras.activations.relu(out)
        out = self.conv2(out)
        out = self.ln2(out)

        if self.use_projection:
            shortcut = self.proj_conv(x)
            shortcut = self.proj_ln(shortcut)
        else:
            shortcut = x

        out = keras.layers.Add()([out, shortcut])
        out = keras.activations.relu(out)
        return out

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters, "stride": self.stride})
        return config


@keras.saving.register_keras_serializable(package="et_util")
class AddPositionalEmbedding(keras.layers.Layer):
    """
    Adds learnable positional embeddings to a sequence of tokens.

    Used in transformer architectures to inject positional information.
    """

    def __init__(self, num_positions, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_positions = num_positions
        self.hidden_dim = hidden_dim

    def build(self, input_shape):
        self.pos_embedding = self.add_weight(
            name="pos_embedding",
            shape=(1, self.num_positions, self.hidden_dim),
            initializer=keras.initializers.RandomNormal(stddev=0.02),
            trainable=True
        )

    def call(self, x):
        return x + self.pos_embedding

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_positions": self.num_positions,
            "hidden_dim": self.hidden_dim
        })
        return config


@keras.saving.register_keras_serializable(package="et_util")
class AttentionPooling(keras.layers.Layer):
    """Multihead attention-pooling (MAP) head: a single learnable query token
    cross-attends over a token sequence to produce ONE pooled vector.

    Replaces a ``Flatten``-based readout for the transformer backbones. Flatten
    concatenates every token position into a fixed-length vector, so the
    downstream Dense layer is position-locked (each patch gets its own weights)
    and large; it also fights translation augmentation, since shifting the eye
    moves features into different patch slots. Attention pooling is instead
    permutation-invariant over tokens and translation-robust: the query learns
    *what* to read, not *where*, and the head is small (one query + one MHA).

    Input:  (batch, num_tokens, hidden_dim)
    Output: (batch, hidden_dim)
    """

    def __init__(self, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads

    def build(self, input_shape):
        hidden_dim = input_shape[-1]
        # one learnable query token, broadcast over the batch at call time
        self.query = self.add_weight(
            name="query",
            shape=(1, 1, hidden_dim),
            initializer=keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
        )
        self.mha = keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=hidden_dim // self.num_heads,
            name="pool_attention",
        )
        self.ln = keras.layers.LayerNormalization(epsilon=1e-6, name="pool_ln")
        # Build the weight-bearing children eagerly. Functional construction only
        # builds them lazily on first call, which is too late if weights are loaded
        # into a model that hasn't been called yet (load_weights then fails with
        # "layer was never built"). The query is (batch, 1, hidden); values are the
        # full token sequence (batch, num_tokens, hidden).
        q_shape = (input_shape[0], 1, hidden_dim)
        self.mha.build(query_shape=q_shape, value_shape=input_shape)
        self.ln.build((input_shape[0], hidden_dim))
        super().build(input_shape)

    def call(self, x):
        batch = ops.shape(x)[0]
        hidden_dim = ops.shape(x)[-1]
        q = ops.broadcast_to(self.query, (batch, 1, hidden_dim))  # (batch, 1, hidden)
        pooled = self.mha(q, x)                                    # (batch, 1, hidden)
        pooled = ops.squeeze(pooled, axis=1)                      # (batch, hidden)
        return self.ln(pooled)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        config = super().get_config()
        config.update({"num_heads": self.num_heads})
        return config


@keras.saving.register_keras_serializable(package="et_util")
class CalibrationWeightingHead(keras.layers.Layer):
    """Context-aware calibration weighting head.

    Replaces the context-free ``Dense(1, sigmoid)`` weighting head. That head
    scores each frame's importance from its own embedding alone, so it can catch
    pointwise garbage (a blink) but cannot reason about a calibration point
    *relative to the others* -- outlier rejection, redundancy, or spatial
    coverage. This head runs a small masked self-attention over the calibration
    set first, so each weight is computed in the context of the whole set, then
    feeds the same weighted ridge.

    Inputs (list):
        - embeddings: (B, T, D)   per-frame embeddings
        - cal_mask:   (B, T)      1 = calibration point, 0 = target/padding
        - coords:     (B, T, 2)   optional (only if ``use_coords``); each point's
                                  (x, y). Calibration coords are known at fit time,
                                  so this lets the head reason about spatial
                                  coverage/redundancy. Gated by cal_mask.
    Output:
        - weights:    (B, T, 1) in (0, 1)   ridge sample weights

    Leakage safety: the attention mask lets every query attend ONLY to
    calibration keys (plus itself, so no row is fully masked -> no NaN softmax).
    A *calibration* query therefore never attends to a target key, so a
    calibration point's weight depends only on the calibration set -- no target
    information can leak into the per-subject ridge kernel (which would make the
    held-out eval fake-good). Target queries' weights are computed but discarded
    by ``cal_mask`` in the ridge.

    Init safety: the context contribution is a zero-initialized Dense, so at step
    0 ``weights == sigmoid(pointwise_logit)`` -- identical to the original linear
    head. The network only spends the attention branch if it earns its keep.
    """

    def __init__(self, attn_dim=64, num_heads=4, num_layers=1, mlp_ratio=0.0,
                 use_coords=True, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.num_layers = num_layers       # stacked pre-norm self-attention blocks
        self.mlp_ratio = mlp_ratio         # >0 adds a transformer FFN per block
        self.use_coords = use_coords
        self.dropout = dropout

    def build(self, input_shapes):
        emb_shape = input_shapes[0]
        batch, num_tokens, emb_dim = emb_shape[0], emb_shape[1], emb_shape[-1]
        self.num_tokens = num_tokens  # static (= max_targets); needed for jit-safe eye
        in_dim = emb_dim + (2 if self.use_coords else 0)
        feat_shape = (batch, num_tokens, in_dim)
        proj_shape = (batch, num_tokens, self.attn_dim)

        self.in_proj = keras.layers.Dense(self.attn_dim, name="cw_in_proj")
        self.in_proj.build(feat_shape)

        # Stacked pre-norm self-attention (+ optional FFN) blocks over the
        # calibration set. Lists of layers are tracked by Keras for weights.
        self.ln1 = [keras.layers.LayerNormalization(epsilon=1e-6, name=f"cw_ln1_{i}")
                    for i in range(self.num_layers)]
        self.mha = [keras.layers.MultiHeadAttention(
                        num_heads=self.num_heads,
                        key_dim=max(1, self.attn_dim // self.num_heads),
                        dropout=self.dropout, name=f"cw_attn_{i}")
                    for i in range(self.num_layers)]
        self.use_mlp = self.mlp_ratio and self.mlp_ratio > 0
        if self.use_mlp:
            hidden = max(1, int(self.attn_dim * self.mlp_ratio))
            self.ln2 = [keras.layers.LayerNormalization(epsilon=1e-6, name=f"cw_ln2_{i}")
                        for i in range(self.num_layers)]
            self.fc1 = [keras.layers.Dense(hidden, activation="gelu", name=f"cw_fc1_{i}")
                        for i in range(self.num_layers)]
            self.fc2 = [keras.layers.Dense(self.attn_dim, name=f"cw_fc2_{i}")
                        for i in range(self.num_layers)]
        for i in range(self.num_layers):
            self.ln1[i].build(proj_shape)
            self.mha[i].build(query_shape=proj_shape, value_shape=proj_shape)
            if self.use_mlp:
                self.ln2[i].build(proj_shape)
                self.fc1[i].build(proj_shape)
                self.fc2[i].build((batch, num_tokens, max(1, int(self.attn_dim * self.mlp_ratio))))

        # The context-free baseline path (== the original Dense(1) logit).
        self.pointwise_logit = keras.layers.Dense(1, name="cw_pointwise_logit")
        self.pointwise_logit.build(feat_shape)
        # Zero-initialized so the context branch starts inert (see Init safety).
        self.context_logit = keras.layers.Dense(
            1, kernel_initializer="zeros", bias_initializer="zeros",
            name="cw_context_logit")
        self.context_logit.build(proj_shape)
        super().build(input_shapes)

    def call(self, inputs, training=None):
        if self.use_coords:
            embeddings, cal_mask, coords = inputs
        else:
            embeddings, cal_mask = inputs
            coords = None

        cm = ops.cast(cal_mask, embeddings.dtype)  # (B, T)
        feat = embeddings
        if self.use_coords:
            coords = ops.cast(coords, embeddings.dtype)
            # gate coords by cal_mask so only calibration coords enter the head
            feat = ops.concatenate([embeddings, coords * cm[..., None]], axis=-1)

        # attention mask: query i may attend key j iff (j is calibration) OR (i == j).
        # Same mask for every layer -> calibration queries only ever see calibration
        # keys, so the no-target-leakage guarantee holds at any depth.
        key_is_cal = cm > 0.5                                   # (B, T)
        eye = ops.cast(ops.eye(self.num_tokens), "bool")        # (T, T)
        allowed = ops.logical_or(
            ops.expand_dims(key_is_cal, 1),   # (B, 1, T) -- broadcast over queries
            ops.expand_dims(eye, 0),          # (1, T, T) -- self-attention diagonal
        )                                     # -> (B, T, T)

        z = self.in_proj(feat)  # (B, T, attn_dim)
        for i in range(self.num_layers):
            h = self.ln1[i](z)
            z = z + self.mha[i](query=h, value=h, key=h, attention_mask=allowed, training=training)
            if self.use_mlp:
                z = z + self.fc2[i](self.fc1[i](self.ln2[i](z)))

        logit = self.pointwise_logit(feat) + self.context_logit(z)  # (B, T, 1)
        return ops.sigmoid(logit)

    def compute_output_shape(self, input_shapes):
        emb_shape = input_shapes[0]
        return (emb_shape[0], emb_shape[1], 1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "attn_dim": self.attn_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "mlp_ratio": self.mlp_ratio,
            "use_coords": self.use_coords,
            "dropout": self.dropout,
        })
        return config


@keras.saving.register_keras_serializable(package="et_util")
class Factorized2DPositionalEmbedding(keras.layers.Layer):
    """Separable row+col learnable positional embeddings for a token grid.

    Replaces a flat 1-D embedding over ``rows*cols`` positions with two small
    tables -- one per row, one per column -- that are broadcast-added. The eye
    image is two eyes side by side on a ``rows x cols`` grid, so horizontal
    (column) position is highly informative; factorizing gives the model that
    2-D structure explicitly (and uses far fewer params than the flat table).

    Input/Output: (batch, rows*cols, hidden_dim) -- the flattened token sequence,
    row-major (row index varies slowest), matching a ``Reshape((rows*cols, h))``
    of a ``(rows, cols, h)`` feature map.
    """

    def __init__(self, num_rows, num_cols, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.hidden_dim = hidden_dim

    def build(self, input_shape):
        self.row_emb = self.add_weight(
            name="row_emb", shape=(self.num_rows, self.hidden_dim),
            initializer=keras.initializers.RandomNormal(stddev=0.02), trainable=True,
        )
        self.col_emb = self.add_weight(
            name="col_emb", shape=(self.num_cols, self.hidden_dim),
            initializer=keras.initializers.RandomNormal(stddev=0.02), trainable=True,
        )

    def call(self, x):
        # (rows,1,h) + (1,cols,h) -> (rows,cols,h) -> (rows*cols, h), row-major
        pos = self.row_emb[:, None, :] + self.col_emb[None, :, :]
        pos = ops.reshape(pos, (self.num_rows * self.num_cols, self.hidden_dim))
        return x + pos[None]

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_rows": self.num_rows,
            "num_cols": self.num_cols,
            "hidden_dim": self.hidden_dim,
        })
        return config


@keras.saving.register_keras_serializable(package="et_util")
class LayerScale(keras.layers.Layer):
    """Per-channel learnable scale on a residual branch (CaiT-style).

    Multiplies the branch output by a learned vector initialized to a small
    constant, so each residual branch starts near-identity and the network
    learns how much of it to admit -- stabilizes pre-norm transformer training.
    """

    def __init__(self, init_value, **kwargs):
        super().__init__(**kwargs)
        self.init_value = init_value

    def build(self, input_shape):
        self.gamma = self.add_weight(
            name="gamma", shape=(input_shape[-1],),
            initializer=keras.initializers.Constant(self.init_value), trainable=True,
        )

    def call(self, x):
        return x * self.gamma

    def get_config(self):
        config = super().get_config()
        config.update({"init_value": self.init_value})
        return config


@keras.saving.register_keras_serializable(package="et_util")
class StochasticDepth(keras.layers.Layer):
    """Drop-path: at training time, randomly zero an entire residual branch for
    a fraction of samples (per-sample), scaling survivors by 1/keep so the
    expectation is preserved. A no-op at inference and when ``drop_rate==0``.
    Applied to the branch output *before* the residual Add.
    """

    def __init__(self, drop_rate, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.drop_rate = drop_rate
        self.seed = seed
        self.seed_generator = keras.random.SeedGenerator(seed)

    def call(self, x, training=None):
        if not training or self.drop_rate == 0.0:
            return x
        keep = 1.0 - self.drop_rate
        # per-sample mask: broadcast over all axes except batch
        shape = (ops.shape(x)[0],) + (1,) * (len(x.shape) - 1)
        rnd = keras.random.uniform(shape, dtype=x.dtype, seed=self.seed_generator)
        mask = ops.cast(rnd < keep, x.dtype)
        return x / keep * mask

    def get_config(self):
        config = super().get_config()
        config.update({"drop_rate": self.drop_rate, "seed": self.seed})
        return config


@keras.saving.register_keras_serializable(package="et_util")
class PatchMerging(keras.layers.Layer):
    """Swin-style patch merging: an *information-preserving* 2x downsample.

    Instead of max-pooling (which discards where-in-the-window a feature was),
    each 2x2 spatial neighborhood is folded into the channel axis -- lossless,
    spatial detail becomes channel detail -- then LayerNorm + a linear projection
    learn how to compress it. Used to drop the high-resolution CNN feature map
    down to the transformer's token grid without throwing away the fine spatial
    (equivariant) information the high-res convs worked to preserve.

    Input (B, H, W, C) with H, W even -> output (B, H//2, W//2, out_dim).
    """

    def __init__(self, out_dim, **kwargs):
        super().__init__(**kwargs)
        self.out_dim = out_dim

    def build(self, input_shape):
        c = input_shape[-1]
        self.ln = keras.layers.LayerNormalization(epsilon=1e-6, name="merge_ln")
        self.proj = keras.layers.Dense(self.out_dim, use_bias=False, name="merge_proj")
        # build children eagerly (load_weights-before-call safety); they see 4*C channels
        merged_shape = (input_shape[0], None, None, 4 * c)
        self.ln.build(merged_shape)
        self.proj.build(merged_shape)
        super().build(input_shape)

    def call(self, x):
        b = ops.shape(x)[0]
        h, w, c = x.shape[1], x.shape[2], x.shape[3]   # static in the functional graph
        # group each 2x2 neighborhood into the channel axis (lossless)
        x = ops.reshape(x, (b, h // 2, 2, w // 2, 2, c))
        x = ops.transpose(x, (0, 1, 3, 2, 4, 5))        # (b, h/2, w/2, 2, 2, c)
        x = ops.reshape(x, (b, h // 2, w // 2, 4 * c))   # concat the four sub-pixels
        return self.proj(self.ln(x))

    def compute_output_shape(self, input_shape):
        b, h, w, _ = input_shape
        return (b, None if h is None else h // 2, None if w is None else w // 2, self.out_dim)

    def get_config(self):
        config = super().get_config()
        config.update({"out_dim": self.out_dim})
        return config


@keras.saving.register_keras_serializable(package="et_util")
class SpatialSoftmaxReadout(keras.layers.Layer):
    """Spatial-softmax / soft-argmax readout: the *expected (x,y) location* of each
    channel's activation, instead of pooling magnitude.

    For a localization task (gaze), where a feature fires matters more than whether
    it fired. This reshapes the token sequence back to its (rows, cols) grid, takes a
    softmax over spatial positions per channel (a normalized attention map), and
    returns the probability-weighted mean coordinate per channel -- a differentiable
    argmax. Output is position-valued, not magnitude-valued. A learnable temperature
    controls how peaked the spatial softmax is. Refs: Levine et al. 2016 (spatial
    feature points); DSNT, Nibali et al. 2018.

    Input  (B, rows*cols, C) -- transformer tokens in row-major grid order.
    Output (B, 2*C) -- [expected_row_per_channel, expected_col_per_channel].
    """

    def __init__(self, rows, cols, **kwargs):
        super().__init__(**kwargs)
        self.rows = rows
        self.cols = cols

    def build(self, input_shape):
        # learnable temperature (softplus-free; clamp via abs at call) controlling peakiness
        self.log_temperature = self.add_weight(
            name="log_temperature", shape=(),
            initializer=keras.initializers.Constant(0.0), trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        c = x.shape[-1]
        n = self.rows * self.cols
        temp = ops.exp(self.log_temperature)                 # > 0
        attn = ops.softmax(x / temp, axis=1)                 # (B, n, C) -- per-channel spatial map
        # normalized coordinate grids in [0,1], row-major over (rows, cols)
        rr = ops.arange(self.rows, dtype="float32") / max(1, self.rows - 1)
        cc = ops.arange(self.cols, dtype="float32") / max(1, self.cols - 1)
        R, C = ops.meshgrid(rr, cc, indexing="ij")           # (rows, cols) each
        row_coords = ops.reshape(R, (n, 1))                  # (n, 1)
        col_coords = ops.reshape(C, (n, 1))
        exp_row = ops.sum(attn * row_coords[None], axis=1)   # (B, C)
        exp_col = ops.sum(attn * col_coords[None], axis=1)   # (B, C)
        return ops.concatenate([exp_row, exp_col], axis=-1)  # (B, 2C)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 2 * input_shape[-1])

    def get_config(self):
        config = super().get_config()
        config.update({"rows": self.rows, "cols": self.cols})
        return config


@keras.saving.register_keras_serializable(package="et_util")
class MaskInspectorLayer(keras.layers.Layer):
    """Debug utility layer that prints mask information during execution."""

    def __init__(self, **kwargs):
        super(MaskInspectorLayer, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, mask=None):
        # Print mask information (for debugging)
        print("Layer mask:", mask)
        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(MaskInspectorLayer, self).get_config()
        return config


class PairTimeDistributed(keras.layers.Layer):
    """Apply a two-input inner model to every timestep of a (sequence, sequence) pair.

    SimpleTimeDistributed for the pose-token embedding model (2026-06-11):
    folds the time axis of BOTH inputs into the batch axis, calls the inner
    model once, and unfolds. Time dimension must be static (it is: max_targets
    is baked into the Input shapes).
    """

    def __init__(self, inner, **kwargs):
        super().__init__(**kwargs)
        self.inner = inner

    def call(self, images, pose):
        t = images.shape[1]
        img_flat = ops.reshape(images, (-1, *images.shape[2:]))
        pose_flat = ops.reshape(pose, (-1, pose.shape[-1]))
        out = self.inner([img_flat, pose_flat])
        return ops.reshape(out, (-1, t, out.shape[-1]))

    def compute_output_shape(self, images_shape, pose_shape):
        inner_out = self.inner.compute_output_shape(
            [(None, *images_shape[2:]), (None, pose_shape[-1])])
        return (images_shape[0], images_shape[1], inner_out[-1])

    def get_config(self):
        config = super().get_config()
        config["inner"] = keras.saving.serialize_keras_object(self.inner)
        return config

    @classmethod
    def from_config(cls, config):
        inner = keras.saving.deserialize_keras_object(config.pop("inner"))
        return cls(inner, **config)


@keras.saving.register_keras_serializable(package="et_util")
class ContrastiveEmbeddingDistanceLoss(keras.layers.Layer):
    """Auxiliary loss (WebEyeTrack Stage-1 idea): push the per-image embedding
    geometry toward the gaze geometry, so pairwise EMBEDDING distance is
    proportional to pairwise GAZE-target distance.

    Why this and not the null embedding->ridge interface tweaks: our whole
    personalization is a LINEAR ridge solve on these frozen embeddings, so a space
    where Euclidean distance tracks gaze distance is exactly what makes that solve
    fit well. Unlike centering/normalization/reweighting (all null on frozen
    features), this changes what the BACKBONE LEARNS -- a co-adaptation bet.

    Identity passthrough: returns the embeddings unchanged and registers a scaled
    ``add_loss``. ``models.py`` only inserts this layer when the weight is > 0, so
    the default model stays byte-identical and existing checkpoints load.

    Inputs ``[embeddings (B, N, D), coords (B, N, 2)]``. Every training subject has
    exactly N real frames (the <144-image filter drops short subjects), so no
    validity mask is needed. Both distance matrices are normalized to unit mean
    over the off-diagonal pairs before comparison, making the alignment invariant
    to the (arbitrary) global scale of either space -- the ridge absorbs scale.
    """

    def __init__(self, weight, epsilon=1e-7, **kwargs):
        super().__init__(**kwargs)
        self.weight = weight
        self.epsilon = epsilon

    def _pairwise(self, x):
        """Euclidean distance matrix (B, N, N) from points (B, N, C)."""
        sq = ops.sum(ops.square(x), axis=-1)                    # (B, N)
        gram = ops.matmul(x, ops.transpose(x, (0, 2, 1)))       # (B, N, N)
        d2 = ops.expand_dims(sq, 2) + ops.expand_dims(sq, 1) - 2.0 * gram
        d2 = ops.maximum(d2, 0.0)                               # numerical floor
        return ops.sqrt(d2 + self.epsilon)

    def call(self, inputs):
        embeddings, coords = inputs
        embeddings = ops.cast(embeddings, "float32")
        coords = ops.cast(coords, "float32")
        # Match the main loss's 16:9 x-weighting so "gaze distance" is in the same
        # anisotropic units the model is actually scored on.
        coords = coords * ops.array([1.778, 1.0], dtype="float32")

        n = embeddings.shape[1]                                 # static (144 / 288)
        off = 1.0 - ops.eye(n, dtype="float32")                 # zero the diagonal
        off = ops.expand_dims(off, 0)                           # (1, N, N)
        npairs = float(n * (n - 1))

        def unit_mean(d):
            mean = ops.sum(d * off, axis=[1, 2], keepdims=True) / npairs
            return d / (mean + self.epsilon)

        d_emb = unit_mean(self._pairwise(embeddings))
        d_gaze = unit_mean(self._pairwise(coords))

        sq_err = ops.square(d_emb - d_gaze) * off               # drop the diagonal
        per_subject = ops.sum(sq_err, axis=[1, 2]) / npairs
        self.add_loss(self.weight * ops.mean(per_subject))
        return embeddings

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super().get_config()
        config.update({"weight": self.weight, "epsilon": self.epsilon})
        return config
