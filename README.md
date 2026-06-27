# Real-Time Pose Recognition

A compact Python app for real-time human pose detection and matching with MediaPipe and OpenCV. It compares a live webcam pose against saved references and keeps the experience smooth, fast, and easy to use, with a playful nod to BhAAi aka Allu Arjun's iconic laugh.

## What it does

- Detects 33 pose landmarks from a webcam feed
- Normalizes poses for scale and position handling
- Matches against stored reference poses with weighted scoring
- Shows live feedback, confidence, and stability across frames
- Lets you capture new reference poses when needed

## Quick start

```bash
pip install -r requirements.txt
python quickstart.py
python main.py --mode capture
python main.py --mode live
```

## Controls

### Live mode
- `SPACE` capture a new reference pose
- `M` show detailed scores
- `I` display the matched reference image
- `+` / `-` adjust threshold
- `Q` quit

### Capture mode
- `SPACE` capture the current pose
- `S` save and continue
- `Q` quit without saving

## Project layout

- `main.py` live app and UI
- `pose_detector.py` MediaPipe pose detection
- `pose_utils.py` normalization and features
- `matcher.py` pose matching and scoring
- `quickstart.py` setup and validation helper
- `calibration.py` testing and benchmarking

## Notes

- Keep a few clear reference images in `datasets/` for best results.
- Threshold tuning usually works well around `0.60`.
- For the full technical breakdown, see `PROJECT_SUMMARY.md` and `SETUP_GUIDE.md`.