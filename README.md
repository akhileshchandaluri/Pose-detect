<div align="center">

# 🕺 Real-Time Pose Recognition & Matching

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?logo=python&logoColor=white)](https://www.python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8.1+-green.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.9-blueviolet.svg?logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A powerful and compact Python application for **real-time human pose detection and matching**. Built on top of MediaPipe and OpenCV, it analyzes a live webcam feed, extracts 33 pose landmarks, normalizes them, and matches them against stored reference poses with a playful nod to BhAAi aka Allu Arjun's iconic style!

<br>

<img src="demo.png" alt="Pose Match Demo" width="600" style="border-radius:10px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);">

</div>

---

## ✨ Features

*   **🎯 High Precision Detection**: Tracks 33 body keypoints in real-time using MediaPipe.
*   **⚖️ Robust Normalization**: Scale and position invariant—you don't have to stand at an exact distance.
*   **🧠 Smart Matching Algorithms**: Utilizes Euclidean distance, Angle Difference, and Cosine Similarity for weighted, accurate matching (85%+ accuracy).
*   **⚡ Blazing Fast**: Runs at 25-35 FPS on modern hardware with a highly optimized pipeline.
*   **📊 Live Feedback**: Displays real-time matching confidence, detailed metrics, and a skeleton overlay.
*   **📸 Custom Reference Poses**: Easily capture your own poses directly from the app and build your database instantly.

## 🚦 Quick Start

### 1. Install Dependencies
Make sure you have Python 3.8+ installed. Then install the required packages:

```bash
pip install -r requirements.txt
```

### 2. Verify Setup
Run the quickstart wizard to ensure your camera and dependencies are working perfectly:
```bash
python quickstart.py
```

### 3. Capture Reference Poses (Optional)
Add your own reference poses to the database:
```bash
python main.py --mode capture
```

### 4. Run Live Matching
Start the real-time pose matching engine:
```bash
python main.py --mode live
```

---

## 🎮 Controls

### 🎥 Live Mode
| Key | Action |
| :---: | --- |
| `SPACE` | Capture a new reference pose dynamically |
| `M` | Toggle detailed scoring & metrics display |
| `I` | Display the current matched reference image |
| `+` / `-` | Increase / Decrease the matching threshold (default: 0.60) |
| `Q` | Quit application |

### 📸 Capture Mode
| Key | Action |
| :---: | --- |
| `SPACE` | Capture the current frame/pose |
| `S` | Save captured pose and continue |
| `Q` | Quit without saving |

---

## 🏗️ Project Architecture

| Component | Description |
| --- | --- |
| 🎛️ `main.py` | The main live application and UI loop. |
| 👁️ `pose_detector.py` | Handles the MediaPipe pose detection extraction. |
| 📏 `pose_utils.py` | Core engine for scale/position normalization and feature extraction. |
| 🤝 `matcher.py` | Houses the pose matching logic, scoring algorithms, and database management. |
| 🚀 `quickstart.py` | Setup, validation, and environment helper. |
| 🔧 `calibration.py` | System testing, debugging, and performance benchmarking. |

---

## 💡 How it Works

1.  **Detection**: MediaPipe extracts 33 3D landmarks from your webcam feed.
2.  **Normalization**: Keypoints are centered relative to the hips and scaled based on torso length, removing distance/position bias.
3.  **Feature Extraction**: Calculates 8 specific joint angles and a flattened feature vector.
4.  **Scoring**: The live pose is compared to the JSON database (`reference_poses.json`) using a combined metric:
    *   *40% Euclidean Distance*
    *   *35% Angle Difference*
    *   *25% Cosine Similarity*
5.  **Temporal Tracking**: A match is considered "stable" only if it consistently hits the confidence threshold across multiple frames, avoiding jitter.

## 📚 Advanced Documentation
For a deeper dive into tuning, system limits, and the architecture, check out the supplementary docs:
*   [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)
*   [SETUP_GUIDE.md](SETUP_GUIDE.md)

---
<div align="center">
Made with ❤️ for computer vision enthusiasts.
</div>