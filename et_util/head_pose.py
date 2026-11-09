"""Head yaw and pitch in degrees from MediaPipe's face landmarks.

saccade.js reads the head's pose from MediaPipe's facial transformation matrix, which fits
the canonical face model to a frame's landmarks. The v1 frame records keep the landmarks
but not the matrix, so this refits it: a scaled rigid fit (Umeyama) of the canonical
model's 468 vertices to the landmarks, under an orthographic camera. MediaPipe's own fit
also corrects for perspective, so a head far off the image's center reads a few degrees
differently; sizes and changes within a participant, which is what the paper uses, are
little affected.

The landmarks are normalized to the image's width (x, z) and height (y), so the fit needs
the image's shape. The v1 records do not store it, but each frame's eye crop is cut at
fixed landmarks (et_util.process_functions.getBothEyesAsSingleImage), so the crop's size in
pixels, over the landmarks' extent, gives the image's width and height:
`image_size_from_crop`.
"""

from pathlib import Path

import numpy as np

# x to the face's left (the viewer's right), y up, z out of the face; centimetres.
CANONICAL = np.loadtxt(Path(__file__).with_name("canonical_face_model.txt"))

# getBothEyesAsSingleImage's padding, in normalized image units.
_CROP_PADDING = 0.01


def image_size_from_crop(lm, crop_width, crop_height):
    """The image's (width, height) in pixels implied by one frame's eye crop.

    `lm` is the (478, 3) normalized landmarks; `crop_width` and `crop_height` the crop's
    size in pixels. Each edge of the crop was truncated to a whole pixel, so a single
    frame is off by up to a pixel or two; take the median over many frames.
    """
    span_x = lm[446, 0] - lm[226, 0] + 2 * _CROP_PADDING
    top = min(lm[27, 1], lm[257, 1])
    bottom = max(lm[23, 1], lm[253, 1])
    span_y = bottom - top + 2 * _CROP_PADDING
    return crop_width / span_x, crop_height / span_y


def head_angles(lm, width, height):
    """(yaw, pitch) in degrees from (478, 3) normalized landmarks in a width x height image.

    Yaw is positive with the face turned toward the image's right (the participant's left);
    pitch is positive with the head tipped up. Both are zero facing the camera squarely.
    """
    # Into the canonical model's axes: x right, y up, z toward the camera. MediaPipe's z is
    # in the same units as x and grows away from the camera.
    points = np.c_[lm[:468, 0] * width, -lm[:468, 1] * height, -lm[:468, 2] * width]
    R = _rotation(CANONICAL, points)
    # The face's forward direction (its +z) in the camera's axes.
    forward = R[:, 2]
    yaw = np.degrees(np.arctan2(forward[0], forward[2]))
    pitch = np.degrees(np.arcsin(np.clip(forward[1], -1, 1)))
    return yaw, pitch


def _rotation(source, target):
    """The rotation of the best scaled rigid fit of `source` points onto `target` (Umeyama,
    1991): a proper rotation, never a reflection."""
    a = source - source.mean(axis=0)
    b = target - target.mean(axis=0)
    U, _, Vt = np.linalg.svd(b.T @ a)
    d = np.sign(np.linalg.det(U @ Vt))
    return U @ np.diag([1.0, 1.0, d]) @ Vt
