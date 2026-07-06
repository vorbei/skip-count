# tools — offline algorithm development & validation

Python scripts used to develop and validate the counting algorithm against real
jump-rope videos. They are research/analysis code (not needed to run the app).

> Note: paths inside these scripts are hard-coded to the original dev machine's
> scratch directory — adjust `BASE` / `VIDEO` / `OUT` before running.

## Setup

```bash
python3.12 -m venv venv
venv/bin/pip install mediapipe opencv-python-headless numpy scipy matplotlib
# pose model:
curl -L -o pose_landmarker_lite.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task
```

## Scripts

- **extract_pose.py** — run MediaPipe pose over a video → per-frame landmark JSON
  (`VIDEO=… OUT=… python extract_pose.py`). Aspect-correct portrait resize.
- **render_debug.py** — render a debug video: skeleton overlay + live count + frame#
  + status, driving the exact browser detector (`ALGO=A|B VIDEO=… OUT=…`).
- **test_db.py / test_C.py / test_A_db.py** — detector ports (A / B / combined) with
  the double-bounce guard; sweep params, compare counts on single- vs double-bounce
  clips vs ground truth.
- **final_validate2.py / analyze_A.py** — accuracy + false-positive validation, miss
  attribution, improvement tests.
- **export_video_data.py** — produce the app's JSON export from a video offline.
- **pace_chart.py** — the pace + vertical-height + trip chart (matplotlib reference of
  the in-app canvas chart).
- **generate_icon.py** — render the app icons.

Ground-truth cadence was cross-checked with Welch PSD, autocorrelation, and a raw-pixel
kymograph, all converging on the body oscillation frequency.
