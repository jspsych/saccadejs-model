"""Gaze-invariant head-pose features from MediaPipe landmarks.

Extracted from analysis/phase_shift_gate.py (2026-06-11) so the training
repack and the input pipeline can share them. Feature-by-feature visual
verification: analysis/pose_feature_overlay.py. (Both scripts have since been
removed; they are in git history.) depth (4) and z_asym (6) ride
MediaPipe's weak z channel — drop them for model input (GOOD_POSE_IDX).
"""

import numpy as np

POSE_NAMES = ["center_x", "center_y", "logscale", "roll", "depth",
              "width_asym", "z_asym", "nose_dx", "nose_dy"]
GOOD_POSE_IDX = [0, 1, 2, 3, 5, 7, 8]   # the 7 visually-verified features

# Pooled per-feature mean/std over the v1 eyedata24 extraction (250 subjects,
# first face-found frame per trial; computed 2026-06-11) — used to standardize
# pose inputs to the model so a Dense projection sees ~unit-scale features.
POSE_MEAN = np.array([0.5324, 0.5146, -1.6161, -0.0086, 0.0180,
                      -0.0035, 0.0008, 0.0017, -0.0639], np.float32)
POSE_STD = np.array([0.0794, 0.1327, 0.2723, 0.1289, 0.0124,
                     0.1014, 0.0144, 0.0331, 0.0479], np.float32)


def pose_features(lm):
    """(478, 3) normalized landmarks -> 9-D pose vector (see POSE_NAMES).

    First five = translation/scale/roll/depth; last four are yaw/pitch
    proxies: eye-width asymmetry and corner-depth asymmetry rotate with yaw;
    the nose-bridge offset relative to the eye midpoint (in interocular
    units) moves with yaw (x) and pitch (y).
    """
    l_out, r_out = lm[33, :2], lm[263, :2]
    vec = r_out - l_out
    iod = max(np.linalg.norm(vec), 1e-6)
    cx, cy = (l_out[0] + r_out[0]) / 2, (l_out[1] + r_out[1]) / 2
    w_l = max(np.linalg.norm(lm[133, :2] - lm[33, :2]), 1e-6)
    w_r = max(np.linalg.norm(lm[263, :2] - lm[362, :2]), 1e-6)
    return np.array([
        cx, cy,
        np.log(iod),
        np.arctan2(vec[1], vec[0]),
        float(np.mean(lm[[33, 133, 362, 263], 2])),
        np.log(w_r / w_l),
        float(np.mean(lm[[362, 263], 2]) - np.mean(lm[[33, 133], 2])),
        (lm[168, 0] - cx) / iod,
        (lm[168, 1] - cy) / iod,
    ])
