"""Sequence augmentation for eye-image training data.

Two stages:
  - non-affine photometric: brightness, contrast, gamma, noise. With
    ``per_image_augmentation=True`` these vary per frame (vectorized, no blur);
    with False the same values -- plus blur -- are applied across the whole
    sequence.
  - affine: horizontal flip, rotation, zoom, translation, perspective -- always
    applied consistently across the sequence (one shared seed).

Ranges come from ``TrainConfig``.
"""

import math
import tensorflow as tf

from et_util.config import TrainConfig


# --- Affine transform matrix helpers -------------------------------------
def get_translation_matrix(tx, ty):
    zero = tf.constant(0.0, dtype="float32")
    one = tf.constant(1.0, dtype="float32")
    return tf.stack([
        tf.stack([one, zero, tx]),
        tf.stack([zero, one, ty]),
        tf.stack([zero, zero, one]),
    ])


def get_rotation_matrix(angle, center_x, center_y):
    center_x = tf.cast(center_x, "float32")
    center_y = tf.cast(center_y, "float32")
    cos_a = tf.cos(angle)
    sin_a = tf.sin(angle)
    zero = tf.constant(0.0, dtype="float32")
    one = tf.constant(1.0, dtype="float32")
    m_rot = tf.stack([
        tf.stack([cos_a, -sin_a, zero]),
        tf.stack([sin_a, cos_a, zero]),
        tf.stack([zero, zero, one]),
    ])
    m_trans1 = get_translation_matrix(-center_x, -center_y)
    m_trans2 = get_translation_matrix(center_x, center_y)
    return m_trans2 @ m_rot @ m_trans1


def get_zoom_matrix(zx, zy, center_x, center_y):
    center_x = tf.cast(center_x, "float32")
    center_y = tf.cast(center_y, "float32")
    zero = tf.constant(0.0, dtype="float32")
    one = tf.constant(1.0, dtype="float32")
    m_scale = tf.stack([
        tf.stack([zx, zero, zero]),
        tf.stack([zero, zy, zero]),
        tf.stack([zero, zero, one]),
    ])
    m_trans1 = get_translation_matrix(-center_x, -center_y)
    m_trans2 = get_translation_matrix(center_x, center_y)
    return m_trans2 @ m_scale @ m_trans1


def get_flip_matrix(horizontal, center_x, center_y):
    center_x = tf.cast(center_x, "float32")
    center_y = tf.cast(center_y, "float32")
    zero = tf.constant(0.0, dtype="float32")
    one = tf.constant(1.0, dtype="float32")
    neg_one = tf.constant(-1.0, dtype="float32")
    m_flip = tf.stack([
        tf.stack([neg_one, zero, zero]),
        tf.stack([zero, one, zero]),
        tf.stack([zero, zero, one]),
    ])
    m_trans1 = get_translation_matrix(-center_x, -center_y)
    m_trans2 = get_translation_matrix(center_x, center_y)
    flip_result = m_trans2 @ m_flip @ m_trans1
    identity = tf.eye(3, dtype="float32")
    return tf.where(horizontal, flip_result, identity)


def matrix_to_affine_params(matrix):
    return tf.stack([
        matrix[0, 0], matrix[0, 1], matrix[0, 2],
        matrix[1, 0], matrix[1, 1], matrix[1, 2],
        matrix[2, 0], matrix[2, 1],
    ])


def apply_affine_augmentation(sequence):
    """Apply one consistent random affine + perspective transform across a sequence.

    sequence: (timesteps, H, W, C) tensor. Returns the augmented sequence.
    """
    sequence_shape = tf.shape(sequence)
    timesteps = sequence_shape[0]
    img_height = sequence_shape[1]
    img_width = sequence_shape[2]
    img_height_f = tf.cast(img_height, "float32")
    img_width_f = tf.cast(img_width, "float32")
    center_x = img_width_f / 2.0
    center_y = img_height_f / 2.0

    base_seed = tf.random.uniform([2], minval=0, maxval=2147483647, dtype="int32")

    # Flip
    flip_seed = base_seed + [0, 1]
    random_flip_val = tf.random.stateless_uniform(shape=(), seed=flip_seed, minval=0.0, maxval=1.0)
    apply_flip = random_flip_val < 0.5

    # Rotation
    rotation_factor = 0.05
    max_angle = rotation_factor * math.pi
    rotation_seed = base_seed + [0, 2]
    rotation_angle = tf.random.stateless_uniform(
        shape=(), seed=rotation_seed, minval=-max_angle, maxval=max_angle
    )

    # Zoom
    zoom_factor_range = (0.9, 1.1)
    zoom_height_factor = tf.random.stateless_uniform(
        shape=(), seed=base_seed + [0, 3], minval=zoom_factor_range[0], maxval=zoom_factor_range[1]
    )
    zoom_width_factor = tf.random.stateless_uniform(
        shape=(), seed=base_seed + [0, 4], minval=zoom_factor_range[0], maxval=zoom_factor_range[1]
    )

    # Translation
    translation_height_factor = 0.1
    translation_width_factor = 0.1
    random_shift_h = tf.random.stateless_uniform(
        shape=(), seed=base_seed + [0, 5],
        minval=-translation_height_factor, maxval=translation_height_factor,
    )
    random_shift_w = tf.random.stateless_uniform(
        shape=(), seed=base_seed + [0, 6],
        minval=-translation_width_factor, maxval=translation_width_factor,
    )
    shift_height = random_shift_h * img_height_f
    shift_width = random_shift_w * img_width_f

    flip_matrix = get_flip_matrix(apply_flip, center_x, center_y)
    zoom_matrix = get_zoom_matrix(zoom_width_factor, zoom_height_factor, center_x, center_y)
    rotation_matrix = get_rotation_matrix(rotation_angle, center_x, center_y)
    translation_matrix = get_translation_matrix(shift_width, shift_height)
    combined_matrix = translation_matrix @ rotation_matrix @ zoom_matrix @ flip_matrix

    affine_params = matrix_to_affine_params(combined_matrix)
    affine_params_tiled = tf.tile(tf.expand_dims(affine_params, 0), [timesteps, 1])

    sequence_float = tf.cast(sequence, "float32")
    augmented_sequence = tf.raw_ops.ImageProjectiveTransformV3(
        images=sequence_float,
        transforms=affine_params_tiled,
        output_shape=tf.stack([img_height, img_width]),
        fill_value=0.0,
        interpolation="BILINEAR",
        fill_mode="CONSTANT",
    )

    # === PERSPECTIVE AUGMENTATION ===
    perspective_factor = 0.1
    start_points = tf.constant(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype="float32"
    )
    scale_factors = tf.stack([img_width_f, img_height_f])
    start_points_scaled = start_points * scale_factors

    max_shift_x = perspective_factor * img_width_f
    max_shift_y = perspective_factor * img_height_f
    random_shifts = tf.random.stateless_uniform(
        shape=(4, 2), seed=base_seed + [0, 7], minval=-1.0, maxval=1.0
    )
    scaled_shifts = random_shifts * tf.stack([[max_shift_x, max_shift_y]])
    end_points = start_points_scaled + scaled_shifts

    def compute_perspective_transform(src_pts, dst_pts):
        A_rows = []
        for i in range(4):
            x, y = src_pts[i, 0], src_pts[i, 1]
            u, v = dst_pts[i, 0], dst_pts[i, 1]
            A_rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y])
            A_rows.append([0, 0, 0, -x, -y, -1, v * x, v * y])
        A = tf.stack([tf.stack(row) for row in A_rows])
        b = tf.reshape(tf.stack([dst_pts[:, 0], dst_pts[:, 1]], axis=1), [-1])
        b = -b
        h = tf.linalg.lstsq(A, tf.expand_dims(b, 1))
        return tf.squeeze(h)

    perspective_params = compute_perspective_transform(start_points_scaled, end_points)
    perspective_params_tiled = tf.tile(tf.expand_dims(perspective_params, 0), [timesteps, 1])

    augmented_sequence = tf.raw_ops.ImageProjectiveTransformV3(
        images=augmented_sequence,
        transforms=perspective_params_tiled,
        output_shape=tf.stack([img_height, img_width]),
        fill_value=0.0,
        interpolation="BILINEAR",
        fill_mode="CONSTANT",
    )

    return tf.cast(augmented_sequence, sequence.dtype)


def make_augment_fn(cfg: TrainConfig):
    """Build a tf.data map fn applying non-affine + affine augmentation.

    Returns ``augment_sequence_complete(inputs, targets, mask)`` closing over cfg.
    """
    blur_prob_thresh = cfg.blur_prob
    blur_sigma_range = cfg.blur_sigma_range
    noise_prob_thresh = cfg.noise_prob
    per_image = cfg.per_image_augmentation
    brightness_delta_range = cfg.brightness_delta_range
    contrast_range = cfg.contrast_range
    gamma_range = cfg.gamma_range
    noise_std_range = cfg.noise_std_range

    def apply_non_affine_augmentations(sequence):
        sequence_shape = tf.shape(sequence)
        timesteps = sequence_shape[0]
        base_seed = tf.random.uniform([2], minval=0, maxval=2147483647, dtype="int32")
        augmented_sequence = tf.cast(sequence, "float32")

        if per_image:
            # Vectorized per-frame photometric augmentation (no map_fn; blur dropped).
            # Each frame gets its own brightness/contrast/gamma/noise, applied across
            # the whole (T, H, W, C) stack by broadcasting over the time axis.
            def _per_frame(v):
                return v[:, tf.newaxis, tf.newaxis, tf.newaxis]

            brightness_deltas = tf.random.stateless_uniform(
                shape=(timesteps,), seed=base_seed + [2, 0],
                minval=brightness_delta_range[0], maxval=brightness_delta_range[1],
            )
            contrast_factors = tf.random.stateless_uniform(
                shape=(timesteps,), seed=base_seed + [3, 0],
                minval=contrast_range[0], maxval=contrast_range[1],
            )
            gamma_values = tf.random.stateless_uniform(
                shape=(timesteps,), seed=base_seed + [4, 0],
                minval=gamma_range[0], maxval=gamma_range[1],
            )
            noise_seed = base_seed + [5, 0]
            noise_probs = tf.random.stateless_uniform(shape=(timesteps,), seed=noise_seed, minval=0.0, maxval=1.0)
            noise_stds = tf.random.stateless_uniform(
                shape=(timesteps,), seed=noise_seed + [0, 1],
                minval=noise_std_range[0], maxval=noise_std_range[1],
            )

            augmented_sequence = augmented_sequence + _per_frame(brightness_deltas)
            augmented_sequence = (augmented_sequence - 127.5) * _per_frame(contrast_factors) + 127.5
            augmented_sequence = tf.maximum(augmented_sequence / 255.0, 1e-8) ** _per_frame(gamma_values) * 255.0

            # Per-frame noise, gated by a per-frame probability (vectorized tf.cond).
            noise = tf.random.stateless_normal(tf.shape(augmented_sequence), seed=noise_seed + [0, 2], stddev=1.0)
            noise_gate = _per_frame(tf.cast(noise_probs < noise_prob_thresh, augmented_sequence.dtype))
            augmented_sequence = augmented_sequence + noise * _per_frame(noise_stds) * noise_gate
            augmented_sequence = tf.clip_by_value(augmented_sequence, 0.0, 255.0)
        else:
            blur_seed = base_seed + [1, 0]
            blur_prob = tf.random.stateless_uniform(shape=(), seed=blur_seed, minval=0.0, maxval=1.0)
            blur_sigma = tf.random.stateless_uniform(
                shape=(), seed=blur_seed + [0, 1], minval=blur_sigma_range[0], maxval=blur_sigma_range[1]
            )
            brightness_delta = tf.random.stateless_uniform(
                shape=(), seed=base_seed + [2, 0],
                minval=brightness_delta_range[0], maxval=brightness_delta_range[1])
            contrast_factor = tf.random.stateless_uniform(
                shape=(), seed=base_seed + [3, 0], minval=contrast_range[0], maxval=contrast_range[1])
            gamma_value = tf.random.stateless_uniform(
                shape=(), seed=base_seed + [4, 0], minval=gamma_range[0], maxval=gamma_range[1])
            noise_seed = base_seed + [5, 0]
            noise_prob = tf.random.stateless_uniform(shape=(), seed=noise_seed, minval=0.0, maxval=1.0)
            noise_std = tf.random.stateless_uniform(
                shape=(), seed=noise_seed + [0, 1], minval=noise_std_range[0], maxval=noise_std_range[1])

            def apply_blur_all():
                kernel_size = tf.cast(tf.math.ceil(blur_sigma * 3) * 2 + 1, "int32")
                kernel_size = tf.minimum(kernel_size, 15)
                half_size = kernel_size // 2
                x = tf.cast(tf.range(-half_size, half_size + 1), "float32")
                g = tf.exp(-0.5 * tf.square(x / blur_sigma))
                g = g / tf.reduce_sum(g)
                g2d = tf.tensordot(g, g, axes=0)
                g2d = tf.expand_dims(tf.expand_dims(g2d, -1), -1)
                return tf.nn.depthwise_conv2d(augmented_sequence, g2d, strides=[1, 1, 1, 1], padding="SAME")

            augmented_sequence = tf.cond(blur_prob < blur_prob_thresh, apply_blur_all, lambda: augmented_sequence)
            augmented_sequence = augmented_sequence + brightness_delta
            augmented_sequence = (augmented_sequence - 127.5) * contrast_factor + 127.5
            seq_norm = augmented_sequence / 255.0
            seq_norm = tf.pow(tf.maximum(seq_norm, 1e-8), gamma_value)
            augmented_sequence = seq_norm * 255.0

            def apply_noise_all():
                noise = tf.random.stateless_normal(
                    shape=tf.shape(augmented_sequence), seed=noise_seed + [0, 2], stddev=noise_std
                )
                return augmented_sequence + noise

            augmented_sequence = tf.cond(noise_prob < noise_prob_thresh, apply_noise_all, lambda: augmented_sequence)
            augmented_sequence = tf.clip_by_value(augmented_sequence, 0.0, 255.0)

        return tf.cast(augmented_sequence, "uint8")

    affine = getattr(cfg, "affine_augmentation", True)

    def augment_sequence_complete(all_inputs, targets, mask):
        images = all_inputs["Input_All_Images"]
        augmented = apply_non_affine_augmentations(images)
        if affine:
            augmented = apply_affine_augmentation(augmented)
        # preserve any extra inputs (e.g. Input_All_Pose) untouched -- pose
        # describes the ORIGINAL geometry, which is also why affine warps must
        # be disabled when pose inputs are in play (cfg.affine_augmentation)
        inputs = dict(all_inputs)
        inputs["Input_All_Images"] = augmented
        return inputs, targets, mask

    return augment_sequence_complete
