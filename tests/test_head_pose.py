"""The head-pose refit, against the canonical face posed by known angles.

et_util.head_pose recovers yaw and pitch from landmarks the way saccade.js reads them from
MediaPipe's transformation matrix. These tests pose the canonical model itself, project it
into landmarks as MediaPipe reports them, and check the angles and the image size come back.
"""

import numpy as np
import pytest

from et_util.head_pose import CANONICAL, head_angles, image_size_from_crop


def rotation(yaw, pitch, roll=0.0):
    """Turn toward the viewer's right by `yaw`, tip up by `pitch`, tilt by `roll` (degrees)."""
    y, p, r = np.radians([yaw, pitch, roll])
    # Yaw about the vertical axis, carrying the face's forward (+z) toward +x.
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    # Pitch about the horizontal axis, carrying forward toward +y (up).
    Rx = np.array([[1, 0, 0], [0, np.cos(p), np.sin(p)], [0, -np.sin(p), np.cos(p)]])
    Rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


def landmarks(R, width=640, height=480, scale=6.0, center=(0.45, 0.55)):
    """The posed canonical face as MediaPipe's normalized landmarks (478 of them; the iris
    points, which the fit does not use, repeat the first ten)."""
    points = CANONICAL @ R.T * scale
    lm = np.c_[points[:, 0] / width + center[0],
               -points[:, 1] / height + center[1],
               -points[:, 2] / width]
    return np.r_[lm, lm[:10]]


def test_facing_the_camera():
    assert head_angles(landmarks(np.eye(3)), 640, 480) == pytest.approx((0, 0), abs=1e-6)


@pytest.mark.parametrize("yaw, pitch, roll", [(20, 0, 0), (0, 15, 0), (-25, -10, 0),
                                              (10, 12, 15)])
def test_known_angles(yaw, pitch, roll):
    got = head_angles(landmarks(rotation(yaw, pitch, roll)), 640, 480)
    # With roll, yaw and pitch read off the forward direction differ from the angles they
    # were composed from by a little; they still name the same pose.
    assert got == pytest.approx((yaw, pitch), abs=0.5 if roll == 0 else 2.5)


def test_pitch_up_is_positive():
    _, pitch = head_angles(landmarks(rotation(0, 15)), 640, 480)
    assert pitch > 0


def test_wrong_aspect_distorts_pitch():
    # Why the image's shape matters: a 16:9 face read as 4:3 tips by several degrees.
    lm = landmarks(rotation(0, 15), width=1280, height=720)
    assert head_angles(lm, 1280, 720)[1] == pytest.approx(15, abs=0.5)
    assert abs(head_angles(lm, 960, 720)[1] - 15) > 2


def test_image_size_from_crop():
    width, height = 1280, 720
    lm = landmarks(np.eye(3), width, height, scale=12.0)
    pad = 0.01
    crop_width = int((lm[446, 0] + pad) * width) - int((lm[226, 0] - pad) * width)
    top = min(int((lm[27, 1] - pad) * height), int((lm[257, 1] - pad) * height))
    bottom = max(int((lm[23, 1] + pad) * height), int((lm[253, 1] + pad) * height))
    w, h = image_size_from_crop(lm, crop_width, bottom - top)
    assert w == pytest.approx(width, rel=0.02)
    assert h == pytest.approx(height, rel=0.05)
