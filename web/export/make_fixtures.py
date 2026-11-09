"""Model I/O + staged-preprocessing + ridge fixtures for the browser demo.

The staged crops (crop_gray/resized/clahe) must come from an OpenCV whose imgproc
uses the portable C++ path -- the one the training pipeline saw on linux/x86. The
Apple-silicon opencv-python wheels ship the KleidiCV HAL, whose resize/CLAHE
deviate by up to 3 gray levels. check_opencv() (shared with
make_preproc_fixtures.py) detects that; pass --cv-python to run just the imgproc
stages in a good interpreter:

    uv python install cpython-3.11-macos-x86_64
    uv venv --python cpython-3.11-macos-x86_64 /tmp/cvx86
    VIRTUAL_ENV=/tmp/cvx86 uv pip install "opencv-python==4.10.0.84" "numpy<2"
"""

import argparse
import json
import os
import subprocess
import urllib.request
import zlib
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import numpy as np

from download_checkpoint import download
from export_onnx import build_models

HERE = Path(__file__).resolve().parent
# Default output; --out redirects it (e.g. at the saccade.js core test fixtures).
FIX = HERE / "fixtures"
TASK = HERE / "face_landmarker.task"
TASK_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/1/face_landmarker.task")

PAD = 0.01
EPS = 1e-7
LAMBDA = 1.0
CENTER = 0.5
N_RANDOM = 16
N_REAL = 24


RESIZE_CRC = 2693661899
CLAHE_CRC = 3109979956
GRAY_REF = [145, 34, 80]


def clahe_apply(img):
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)


def opencv_ok():
    small = (np.arange(9 * 5).reshape(5, 9) * 7 % 256).astype(np.uint8)
    big = (np.arange(151 * 37).reshape(37, 151) * 37 % 256).astype(np.uint8)
    img = np.zeros((1, 3, 3), np.uint8)
    img[0, 0] = [10, 200, 90]
    img[0, 1] = [255, 0, 17]
    img[0, 2] = [7, 7, 250]
    return (
        zlib.crc32(cv2.resize(small, (23, 13)).tobytes()) == RESIZE_CRC
        and zlib.crc32(clahe_apply(cv2.resize(big, (144, 36))).tobytes()) == CLAHE_CRC
        and cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).ravel().tolist() == GRAY_REF
    )


def crop_bbox(lm, w, h):
    left = int((lm[226][0] - PAD) * w)
    right = int((lm[446][0] + PAD) * w)
    top = min(int((lm[27][1] - PAD) * h), int((lm[257][1] - PAD) * h))
    bottom = max(int((lm[23][1] + PAD) * h), int((lm[253][1] + PAD) * h))
    return left, top, right, bottom


def stage_one(k):
    """Run the imgproc stages for frame k; requires a portable-path OpenCV."""
    rgb = cv2.cvtColor(cv2.imread(str(FIX / f"frame_{k}.png")), cv2.COLOR_BGR2RGB)
    lm = json.loads((FIX / f"landmarks_{k}.json").read_text())
    h, w = rgb.shape[:2]
    left, top, right, bottom = crop_bbox(lm, w, h)
    crop = rgb[top:bottom, left:right]
    if crop.size == 0 or 0 in crop.shape:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (144, 36))
    cv2.imwrite(str(FIX / f"crop_gray_{k}.png"), gray)
    cv2.imwrite(str(FIX / f"resized_{k}.png"), resized)
    cv2.imwrite(str(FIX / f"clahe_{k}.png"), clahe_apply(resized))
    return True


def stage_all(n, cv_python):
    if n == 0:
        return []
    if opencv_ok():
        return [k for k in range(n) if stage_one(k)]
    if not cv_python:
        print(f"WARNING: this OpenCV build ({cv2.__version__}) does not match the "
              "portable imgproc path (KleidiCV HAL on Apple silicon). Staged crop "
              "fixtures and real-crop model inputs SKIPPED; pass --cv-python "
              "</path/to/x86_64/python> (see this module's docstring).")
        return []
    subprocess.run([cv_python, str(Path(__file__).resolve()), "--stage-only",
                    "--n-frames", str(n)], check=True)
    return [k for k in range(n) if (FIX / f"clahe_{k}.png").exists()]


def structured():
    imgs = []
    x = np.linspace(0, 255, 144, dtype=np.float32)
    y = np.linspace(0, 255, 36, dtype=np.float32)
    imgs.append(np.tile(x, (36, 1)))
    imgs.append(np.tile(y[:, None], (1, 144)))
    cb = np.indices((36, 144)).sum(0) // 6 % 2 * 255
    imgs.append(cb.astype(np.float32))
    gy, gx = np.mgrid[0:36, 0:144]
    blob = np.zeros((36, 144), np.float32)
    for cx, cy, s, a in [(30, 18, 8, 200), (72, 12, 5, 255), (110, 24, 10, 160)]:
        blob += a * np.exp(-(((gx - cx) ** 2 + (gy - cy) ** 2) / (2.0 * s * s)))
    imgs.append(np.clip(blob, 0, 255))
    imgs.append(np.zeros((36, 144), np.float32))
    imgs.append(np.full((36, 144), 255.0, np.float32))
    imgs.append(np.full((36, 144), 128.0, np.float32))
    imgs.append(((gx // 4 + gy // 4) % 2 * 200 + 20).astype(np.float32))
    return [im.astype(np.uint8) for im in imgs]


def ensure_task():
    if not TASK.exists():
        print(f"downloading {TASK_URL}")
        urllib.request.urlretrieve(TASK_URL, TASK)
    return TASK


def landmarker():
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    ensure_task()
    opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(TASK)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(opts), mp


def detect(det, mp, rgb, ts_ms):
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    res = det.detect_for_video(img, ts_ms)
    if not res.face_landmarks:
        return None
    return [[p.x, p.y, p.z] for p in res.face_landmarks[0]]


def capture_camera(seconds=5.0):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("camera unavailable (no device or permission denied); "
              "re-run with --frames-dir <dir> or from an interactive shell")
        return []
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    det, mp = landmarker()
    frames, t = [], 0
    import time

    t0 = time.time()
    while time.time() - t0 < seconds:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t += 33
        lm = detect(det, mp, rgb, t)
        if lm is not None:
            frames.append((rgb, lm))
    cap.release()
    det.close()
    print(f"camera: {len(frames)} face-found frames")
    return frames


def capture_dir(d):
    paths = sorted(p for p in Path(d).iterdir()
                   if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
    if not paths:
        print(f"no images in {d}")
        return []
    det, mp = landmarker()
    frames, t = [], 0
    for p in paths:
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t += 33
        lm = detect(det, mp, rgb, t)
        if lm is not None:
            frames.append((rgb, lm))
    det.close()
    print(f"{d}: {len(frames)}/{len(paths)} face-found frames")
    return frames


def evenly(items, n):
    if len(items) <= n:
        return items
    idx = np.linspace(0, len(items) - 1, n).round().astype(int)
    return [items[i] for i in idx]


def solve_ridge(E, Y, W, lam=LAMBDA, center=CENTER):
    ws = np.sqrt(W + EPS)[:, None]
    Xw = E * ws
    yw = (Y - center) * ws
    lhs = Xw.T @ Xw + lam * np.eye(E.shape[1], dtype=np.float64)
    return np.linalg.solve(lhs, Xw.T @ yw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", default=None)
    ap.add_argument("--no-camera", action="store_true")
    ap.add_argument("--cv-python", default=None)
    ap.add_argument("--stage-only", action="store_true")
    ap.add_argument("--n-frames", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None,
                    help="fixture output directory (default web/export/fixtures)")
    args = ap.parse_args()

    if args.out:
        global FIX
        FIX = Path(args.out).resolve()

    if args.stage_only:
        if not opencv_ok():
            raise SystemExit(f"OpenCV {cv2.__version__} does not match the portable "
                             "imgproc path; see this module's docstring")
        print(f"staged {len(stage_all(args.n_frames, None))} frames with "
              f"OpenCV {cv2.__version__}")
        return

    FIX.mkdir(parents=True, exist_ok=True)
    weights = download()
    _, emb_model, cal = build_models(weights)

    rng = np.random.default_rng(args.seed)
    images = [rng.integers(0, 256, (36, 144), dtype=np.uint8) for _ in range(N_RANDOM)]
    images += structured()

    if args.frames_dir:
        frames = capture_dir(args.frames_dir)
    elif args.no_camera:
        frames = []
    else:
        frames = capture_camera()

    kept = []
    for rgb, lm in frames:
        h, w = rgb.shape[:2]
        left, top, right, bottom = crop_bbox(lm, w, h)
        if right > left and bottom > top and rgb[top:bottom, left:right].size:
            kept.append((rgb, lm))
    kept = evenly(kept, N_REAL)

    for k, (rgb, lm) in enumerate(kept):
        cv2.imwrite(str(FIX / f"frame_{k}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        (FIX / f"landmarks_{k}.json").write_text(json.dumps(
            [[round(float(v), 7) for v in p] for p in lm]))

    staged = stage_all(len(kept), args.cv_python)
    for k in staged:
        images.append(cv2.imread(str(FIX / f"clahe_{k}.png"), cv2.IMREAD_GRAYSCALE))

    X = np.stack(images).astype(np.float32)[..., None]
    E = np.asarray(emb_model.predict(X, batch_size=8, verbose=0), dtype=np.float32)
    np.save(FIX / "model_inputs.npy", X)
    np.save(FIX / "model_embeddings.npy", E)
    # The calibration head now ships as a second graph output, so parity has to cover it too.
    # Computed here from the Keras layer, which is the reference the ONNX must match.
    np.save(FIX / "model_cal_weights.npy",
            np.asarray(cal(E), dtype=np.float32).reshape(len(E), 1))
    print(f"fixtures: N={len(images)} (random {N_RANDOM}, structured 8, "
          f"real {len(staged)}/{len(kept)}) -> {FIX}")

    ck, cb = [np.asarray(w) for w in cal.get_weights()]
    ck, cb = ck[:, 0].astype(np.float64), float(cb[0])

    n_rows = 13
    src = E[-len(staged):] if len(staged) >= n_rows else E
    if len(src) < n_rows:
        reps = int(np.ceil(n_rows / len(src)))
        src = np.concatenate([src] * reps, 0)
    rows_e = src[:n_rows].astype(np.float64)
    grid = [(5, 5), (50, 5), (95, 5), (5, 50), (50, 50), (95, 50), (5, 95),
            (50, 95), (95, 95), (27.5, 27.5), (72.5, 27.5), (27.5, 72.5),
            (72.5, 72.5)]
    Y = np.array([[x / 100.0, y / 100.0] for x, y in grid])
    W = 1.0 / (1.0 + np.exp(-(rows_e @ ck + cb)))
    kernel = solve_ridge(rows_e, Y, W)
    pred = rows_e @ kernel + CENTER

    (FIX / "ridge_fixture.json").write_text(json.dumps({
        "lambda": LAMBDA,
        "center": CENTER,
        "epsilon": EPS,
        "embedding_dim": int(rows_e.shape[1]),
        "rows": [{"e": [float(v) for v in rows_e[i]],
                  "x": float(Y[i, 0]), "y": float(Y[i, 1]),
                  "w": float(W[i])} for i in range(n_rows)],
        "kernel": [[float(v) for v in row] for row in kernel],
        "predictions": [{"x": float(pred[i, 0]), "y": float(pred[i, 1])}
                        for i in range(n_rows)],
    }, indent=1))
    print(f"wrote {FIX / 'ridge_fixture.json'}")


if __name__ == "__main__":
    main()
