"""Generate OpenCV ground-truth fixtures for the TS preprocessing/ridge port.

Must run against an OpenCV whose imgproc uses the portable C++ path -- i.e. the
same behaviour the training pipeline saw on linux/x86. On Apple silicon the
opencv-python wheels ship the KleidiCV custom HAL, whose cv2.resize deviates from
that path by up to 3 gray levels; there is no env var to turn it off. Use an
x86_64 interpreter under Rosetta:

    uv python install cpython-3.11-macos-x86_64
    uv venv --python cpython-3.11-macos-x86_64 /tmp/cvx86
    VIRTUAL_ENV=/tmp/cvx86 uv pip install "opencv-python==4.10.0.84" "numpy<2"
    /tmp/cvx86/bin/python web/export/make_preproc_fixtures.py

check_opencv() below aborts if the local build disagrees.
"""

import json
import os
import zlib

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# Default output; --out redirects it (e.g. at the saccade.js core test fixtures).
OUT = os.path.join(HERE, "fixtures", "preproc")

EYE_W, EYE_H = 144, 36


RESIZE_CRC = 2693661899
CLAHE_CRC = 3109979956
GRAY_REF = [145, 34, 80]


def check_opencv():
    small = (np.arange(9 * 5).reshape(5, 9) * 7 % 256).astype(np.uint8)
    big = (np.arange(151 * 37).reshape(37, 151) * 37 % 256).astype(np.uint8)
    img = np.zeros((1, 3, 3), np.uint8)
    img[0, 0] = [10, 200, 90]
    img[0, 1] = [255, 0, 17]
    img[0, 2] = [7, 7, 250]
    ok = (
        zlib.crc32(cv2.resize(small, (23, 13)).tobytes()) == RESIZE_CRC
        and zlib.crc32(clahe_apply(cv2.resize(big, (EYE_W, EYE_H))).tobytes())
        == CLAHE_CRC
        and cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).ravel().tolist() == GRAY_REF
    )
    if not ok:
        raise SystemExit(
            "this OpenCV build (%s) does not match the portable imgproc path; "
            "see the module docstring for the x86_64 recipe" % cv2.__version__
        )


def clahe_apply(img):
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)


def both_eyes_bbox(lm, W, H, padding=0.01):
    left = int((lm[226][0] - padding) * W)
    right = int((lm[446][0] + padding) * W)
    top = min(int((lm[27][1] - padding) * H), int((lm[257][1] - padding) * H))
    bottom = max(int((lm[23][1] + padding) * H), int((lm[253][1] + padding) * H))
    return left, top, right, bottom


def make_source(name, w, h, rng):
    if name == "noise":
        return rng.integers(0, 256, size=(h, w), dtype=np.uint8)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    v = (
        127.5
        + 60.0 * np.sin(xx / 13.0)
        + 40.0 * np.cos(yy / 7.0)
        + 25.0 * np.sin((xx + yy) / 23.0)
        + 90.0 * (xx / max(w - 1, 1) - 0.5)
    )
    if name == "structured_noise":
        v = v + rng.normal(0, 12.0, size=(h, w))
    return np.clip(v, 0, 255).astype(np.uint8)


def main():
    check_opencv()
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(20260904)

    cases = []
    specs = [
        ("odd_small", 151, 37, "noise"),
        ("upscale", 80, 20, "noise"),
        ("identity", 144, 36, "noise"),
        ("downscale", 1200, 300, "structured"),
        ("wide_flat", 420, 41, "structured_noise"),
        ("tiny", 23, 9, "noise"),
    ]
    for name, w, h, kind in specs:
        src = make_source(kind, w, h, rng)
        resized = cv2.resize(src, (EYE_W, EYE_H))
        claheed = clahe_apply(resized)
        cv2.imwrite(os.path.join(OUT, f"{name}_src.png"), src)
        cv2.imwrite(os.path.join(OUT, f"{name}_resized.png"), resized)
        cv2.imwrite(os.path.join(OUT, f"{name}_clahe.png"), claheed)
        cases.append({"name": name, "w": w, "h": h})

    # CLAHE directly on odd-sized images (exercises reflect-101 padding on both axes)
    clahe_cases = []
    for name, w, h in [("clahe_37x151", 151, 37), ("clahe_40x144", 144, 40), ("clahe_13x29", 29, 13)]:
        src = make_source("noise", w, h, rng)
        cv2.imwrite(os.path.join(OUT, f"{name}_src.png"), src)
        cv2.imwrite(os.path.join(OUT, f"{name}_out.png"), clahe_apply(src))
        clahe_cases.append({"name": name, "w": w, "h": h})

    # Full pipeline on a synthetic RGB frame + fake landmarks.
    W, H = 320, 240
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    r = 128 + 90 * np.sin(xx / 11.0) + 30 * np.cos(yy / 5.0)
    g = 128 + 70 * np.cos(xx / 17.0) + 50 * np.sin((xx * yy) / 900.0)
    b = 128 + 80 * np.sin((xx + 2 * yy) / 19.0)
    frame_rgb = np.clip(np.stack([r, g, b], axis=-1), 0, 255).astype(np.uint8)
    frame_rgb = np.clip(
        frame_rgb.astype(np.int16) + rng.integers(-10, 11, size=(H, W, 3)), 0, 255
    ).astype(np.uint8)

    lm = np.zeros((478, 3), dtype=np.float64)
    lm[:, 0] = 0.5
    lm[:, 1] = 0.5
    lm[226] = [0.3123, 0.4200, 0.0]
    lm[446] = [0.6871, 0.4200, 0.0]
    lm[27] = [0.36, 0.3711, 0.0]
    lm[257] = [0.62, 0.3655, 0.0]
    lm[23] = [0.36, 0.4402, 0.0]
    lm[253] = [0.62, 0.4479, 0.0]

    left, top, right, bottom = both_eyes_bbox(lm, W, H)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    crop_bgr = frame_bgr[top:bottom, left:right]
    crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    pipe_resized = cv2.resize(crop_gray, (EYE_W, EYE_H))
    pipe_clahe = clahe_apply(pipe_resized)

    cv2.imwrite(os.path.join(OUT, "pipe_frame.png"), frame_bgr)
    cv2.imwrite(os.path.join(OUT, "pipe_crop_gray.png"), crop_gray)
    cv2.imwrite(os.path.join(OUT, "pipe_resized.png"), pipe_resized)
    cv2.imwrite(os.path.join(OUT, "pipe_clahe.png"), pipe_clahe)

    # Standalone gray conversion check on the whole frame.
    cv2.imwrite(os.path.join(OUT, "pipe_frame_gray.png"), cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY))

    with open(os.path.join(OUT, "cases.json"), "w") as f:
        json.dump(
            {
                "resize_cases": cases,
                "clahe_cases": clahe_cases,
                "pipeline": {
                    "W": W,
                    "H": H,
                    "landmarks": lm.tolist(),
                    "bbox": {"left": left, "top": top, "right": right, "bottom": bottom},
                    "crop_w": right - left,
                    "crop_h": bottom - top,
                },
            },
            f,
        )

    # Ridge fixture.
    n = 13
    d = 128
    f32 = lambda a: a.astype(np.float32).astype(np.float64)
    emb = f32(rng.uniform(-1.0, 1.0, size=(n, d)))
    xs = [0.05, 0.35, 0.65, 0.95]
    ys = [0.05, 0.275, 0.50, 0.725, 0.95]
    grid = [(x, y) for y in ys for x in xs]
    idx = rng.choice(len(grid), size=n, replace=False)
    coords = np.array([grid[i] for i in idx], dtype=np.float64)
    w = rng.uniform(0.05, 0.95, size=n)

    lam = 1.0
    center = 0.5
    eps = 1e-7
    ws = np.sqrt(w + eps)[:, None]
    Xw = emb * ws
    yw = (coords - center) * ws
    lhs = Xw.T @ Xw + lam * np.eye(d)
    rhs = Xw.T @ yw
    kernel = np.linalg.solve(lhs, rhs)

    probes = f32(rng.uniform(-1.0, 1.0, size=(5, d)))
    preds = probes @ f32(kernel) + center

    head_kernel = rng.normal(0, 0.3, size=d)
    head_bias = float(rng.normal(0, 0.2))
    weights = 1.0 / (1.0 + np.exp(-(emb @ head_kernel + head_bias)))

    with open(os.path.join(OUT, "ridge_case.json"), "w") as f:
        json.dump(
            {
                "lambda": lam,
                "center": center,
                "epsilon": eps,
                "embeddings": emb.tolist(),
                "coords": coords.tolist(),
                "weights": w.tolist(),
                "kernel": kernel.tolist(),
                "probes": probes.tolist(),
                "predictions": preds.tolist(),
                "head": {"kernel": head_kernel.tolist(), "bias": head_bias},
                "head_weights": weights.tolist(),
            },
            f,
        )

    print("wrote fixtures to", OUT)
    total = sum(
        os.path.getsize(os.path.join(OUT, p)) for p in os.listdir(OUT)
    )
    print("total bytes:", total)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None,
                    help="fixture output directory (default web/export/fixtures/preproc)")
    a = ap.parse_args()
    if a.out:
        OUT = os.path.abspath(a.out)
    main()
