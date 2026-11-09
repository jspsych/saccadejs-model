# SaccadeJS dataset summary

Format `{FORMAT}`. Built by `analysis/build_dataset_summary.py` in jspsych/eye-tracking.

Who is in the SaccadeJS dataset and how they got there, without identifiers. The dataset has two
collections:

- `eyedata23`, the training set (2023). Each participant looked at all 144 positions of a 16 × 9
  lattice once.
- `eyedata24`, the held-out evaluation set (December 2024 and March 2025). Each participant did the
  same task twice, in two phases, for 288 trials.

## participants.csv

One row per released participant.

| column | meaning |
|---|---|
| `participant` | the experiment's random 8-character code, the same one the dataset's files use |
| `collection` | `eyedata23` or `eyedata24` |
| `split` | `training` or `held-out` |
| `n_trials` | trials with a video |
| `n_frames` | frames decoded from those videos |
| `n_face_found` | frames in which MediaPipe found a face |
| `webcam` | `integrated` or `external`, as the participant answered; blank if not saved |
| `glasses` | `yes` or `no`, whether they wore glasses for the task; blank if not saved |
| `screen_width`, `screen_height` | the browser window in CSS pixels once fullscreen |

The remaining columns describe the recordings. They come from decoding every trial video
(`analysis/probe_clips.py`) and from the frame records (`analysis/extract_dataset_frame_stats.py`),
reduced by `analysis/dataset_conditions.py`. The frame measures are medians over the participant's
frames in which MediaPipe found a face, defined as in the paper's section 4; they are blank for
the participants with no such frame. The frame rate and clip length are blank for participants
whose videos carry no readable frame timestamps.

| column | meaning |
|---|---|
| `video_codec` | the videos' codec as OpenCV reports it: `h264` or `VP80` |
| `video_width`, `video_height` | the videos' frame size in pixels |
| `clips_timed` | trial videos whose frame timestamps are all readable |
| `frame_rate` | frames per second: 1000 over the median gap between consecutive readable timestamps, leaving out each video's first gap, which is often long |
| `clip_ms` | the median video's length in ms: its last frame's timestamp plus one gap. The experiment asked for 500 ms |
| `eye_crop_width` | the eye crop's width in camera pixels |
| `brightness` | mean gray level of the eye crop, 0–255 |
| `contrast` | its standard deviation |
| `brightness_difference` | the difference in mean gray level between the two eyes, over all frames |
| `noise` | mean absolute change of the 144 × 36 eye image from the frame just before |
| `sharpness` | variance of the eye crop's Laplacian |
| `eye_aspect_ratio` | the less open eye's aspect ratio |
| `blink_rate` | share of frames in which that is below 0.7 of the participant's median |
| `interocular_distance` | between the outer eye corners, as a share of the image's width |
| `head_x_sd`, `head_y_sd` | standard deviation of the eyes' midpoint, as a share of the image's width and height |
| `distance_sd` | standard deviation of the log interocular distance: relative movement toward and away from the camera |
| `yaw`, `pitch` | the head's turn and tilt in degrees; pitch positive tipped up |
| `roll`, `roll_sd` | the head's roll in degrees (the line between the outer eye corners), and its standard deviation |

## consent.csv

Counts only, per collection, so that no one who withdrew appears as a row.

| count | meaning |
|---|---|
| `recorded videos` | participants with any video on the server |
| `saved sessions` | participants whose session data was saved |
| `answered the consent question` | sessions with an answer to the final public-release question |
| `consent: public`, `consent: opted out` | that answer |
| `opted out: research only`, `opted out: delete` | the follow-up: may the research team keep the videos privately |
| `removed on request` | removed later, after asking us |
| `public, not removed: complete videos` / `incomplete videos` | of those who agreed, whether all their trials' videos are on the server |
| `public, not removed, not released` | agreed but not in the released dataset |
| `public, not removed, complete videos, not released` | of those, the ones with every trial's video on the server, left out because at least one of those videos had no decodable frame |
| `released, with a saved session` | released participants whose session data was saved |
| `released` | participants in the released dataset |
| `asked by Prolific message` | `eyedata24` only: the final consent answer was not saved for every session, so participants were asked again through Prolific's messages; how many were asked |
| `by message: agreed`, `declined`, `no response` | their answers |
| `released, consent in the experiment` / `by Prolific message` | `eyedata24` only: on which of the two each released participant is in; every released participant has a yes from one of them and no later no |

## demographics.csv

Counts only, per collection, of the released participants' demographics as Prolific records them
for each participant. These are not linked to the rows of `participants.csv`.

| column | meaning |
|---|---|
| `collection` | `eyedata23` or `eyedata24` |
| `variable` | `age` (in bands), `age (years)`, `sex`, `ethnicity`, `country of residence`, `first language`, `student status` or `employment status` |
| `value` | a category, as Prolific words it; for `age (years)`, `mean`, `sd`, `median`, `min` or `max` |
| `n` | released participants in that category; for `age (years)`, the statistic, in years |

`not available` counts participants whose value Prolific no longer has, or who couldn't be looked
up because their session data wasn't saved. Countries and languages with fewer than 5 released
participants in a collection are counted together as `other (fewer than 5 each)`; Prolific's own
`Other` is a separate answer.
