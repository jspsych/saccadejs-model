# SaccadeJS held-out evaluation embeddings

Per-frame output of the SaccadeJS eye-embedding model, version 1.0.0 (W&B run
`vassar-cogsci-lab/eye-tracking-multiframe/kwppnkuj`, checkpoint
`multiframe_random_s42_ext70.weights.h5`), on the held-out evaluation set ("eyedata24"). No
participant in this set was used to train the model.

No video or images are included. Each frame is stored as numbers: the model's 128-number eye
embedding, its calibration weight, head pose, and image-quality measures.

Built by `analysis/build_paper_eval_dataset.py` in
[jspsych/saccadejs-model](https://github.com/jspsych/saccadejs-model), from embeddings written by
`analysis/extract_v1_embeddings.py --save-pose`.

## Contents

| Path | What it is |
|---|---|
| `participants/<id>.json.gz` | One gzipped JSON file per participant with usable frames |
| `participants.csv` | One row per participant, including those with no usable frames |
| `trials.csv` | When each trial was shown in its session |
| `extraction_meta.json` | The checkpoint and settings the embeddings were extracted with |
| `README.md` | This file |

## The task

Each participant completed two phases of the same task, one after the other. In each phase a
fixation target appeared once at each of 144 positions, a 16 × 9 lattice (x at 5, 11, …, 95%
of the screen width; y at 5, 16.25, …, 95% of its height), in three blocks of 48 with a break
between blocks. Positions were assigned to blocks at random and shown in random order within
a block. Video was recorded from 1000 to 1500 ms after the target appeared, after the saccade
to it, so every frame is fixation. On 10 trials per block the target then changed color and
the participant pressed the space bar before the next trial.

Before the task, participants said whether their webcam was built in or separate, and whether
they would wear glasses during the task.

## participants.csv

| Column | Meaning |
|---|---|
| `participant` | Participant code, the file name in `participants/` |
| `n_trials` | Trials with a video |
| `n_frames` | Frames decoded from those videos |
| `n_face_found` | Frames in which MediaPipe found a face |
| `n_trials_embedded` | Trials with at least one embedded frame |
| `n_frames_embedded` | Embedded frames (every face-found frame) |
| `webcam` | `integrated` or `external`, self-reported |
| `glasses` | `yes` or `no`: whether they wore glasses during the task, self-reported |
| `screen_width`, `screen_height` | The browser window in CSS pixels after it went fullscreen; target positions are percentages of these |
| `camera_width`, `camera_height` | The camera image in pixels, estimated from the eye crops' sizes (median over frames); format 2 only |

## trials.csv

| Column | Meaning |
|---|---|
| `participant`, `trial` | As in the participant files |
| `order` | The trial's position among the participant's 288 recordings, from 1 |
| `block` | 0–5: three blocks per phase |
| `time_s` | When the trial was shown, in seconds from the start of the session |

## Participant files

```json
{ "format": "saccadejs-eval/2",
  "participant": "065fta2n",
  "trials": [ ... ] }   // one object per trial, by phase and then position (order: trials.csv)
```

Each trial has:

| Field | Meaning |
|---|---|
| `trial` | `<participant>_<phase>_<x>_<y>` |
| `phase` | 1 or 2 |
| `x`, `y` | Target position, in percent of the screen's width and height |
| `n_frames` | Frames decoded from the trial's video, face found or not |

and per-frame streams, one row per embedded frame in recording order, packed as base64 over a
little-endian typed array, row after row:

| Field | Type | Values per frame | What one row is |
|---|---|---|---|
| `frame_index` | int16 | 1 | 0-based position of the frame in the trial's video |
| `time_ms` | float16 | 1 | The frame's timestamp in the video, ms from the start of recording |
| `embedding` | float16 | 128 | The eye embedding (tanh output, −1 to 1) |
| `weight` | float16 | 1 | The model's 0–1 calibration weight for the frame |
| `eye_aspect_ratio` | float16 | 2 | Left then right eye; low values mean a closing eye |
| `eye_image_stats` | float16 | 6 | Of the gray eye crop at the camera's resolution: sharpness (variance of the Laplacian), mean brightness (0–255), contrast (SD of brightness), mean brightness of the participant's left eye (the crop's right half; the image is unmirrored) and right eye, and noise: the mean absolute change from the trial's previous frame, at 144 × 36, NaN when that frame was not the one just before |
| `head_pose` | float16 | 9 | From MediaPipe's landmarks, in normalized image coordinates: `center_x`, `center_y` (midpoint of the outer eye corners), `log_iod` (log interocular distance; larger is nearer the camera), `roll` (radians), `depth`, `width_asym` (log ratio of the eyes' widths; turns with yaw), `z_asym`, `nose_dx`, `nose_dy` (nose bridge offset from the eye midpoint in interocular distances; turn with yaw and pitch). `depth` and `z_asym` use MediaPipe's weak z coordinate. |
| `head_angles` | float16 | 2 | Yaw and pitch in degrees, from a fit of MediaPipe's canonical face model to the landmarks (`et_util/head_pose.py` in jspsych/saccadejs-model). Yaw is positive turned toward the image's right (the participant's left), pitch positive tipped up |
| `eye_crop_px` | float16 | 2 | The eye crop's width and height in camera pixels |

Format 1, the first release, had only sharpness and brightness in `eye_image_stats` (2 values
per frame), and no `head_angles`, `eye_crop_px`, `camera_width` or `camera_height`.

```python
import base64, gzip, json
import numpy as np

def unpack(b64, dtype, width):
    raw = np.frombuffer(base64.b64decode(b64), dtype=np.dtype(dtype).newbyteorder("<"))
    return raw.reshape(-1, width)

with gzip.open("participants/065fta2n.json.gz", "rt") as f:
    p = json.load(f)
t = p["trials"][0]
E = unpack(t["embedding"], "f2", 128)
```

## Using the embeddings

The model predicts gaze by ridge regression from embeddings to target positions, fit on a
participant's own calibration frames and weighted by each frame's calibration weight. A
held-out evaluation calibrates on one phase and tests on the other.
