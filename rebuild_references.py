"""
Rebuild the reference pose database from images in datasets/.

Run this after adding, removing, or replacing reference images:
    python rebuild_references.py
"""

import os

import cv2
import numpy as np

from matcher import ReferencePoseDatabase
from pose_detector import PoseDetector
from pose_utils import FeatureVector


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def load_image(path: str):
    image = cv2.imread(path)
    if image is not None:
        return image

    try:
        from PIL import Image

        pil_image = Image.open(path)
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def label_from_filename(filename: str) -> str:
    base_name = os.path.splitext(filename)[0]
    return base_name.replace("_", " ").replace("-", " ").title()


def main():
    datasets_dir = "datasets"
    db = ReferencePoseDatabase("reference_poses.json")
    detector = PoseDetector(model_complexity=1, min_detection_confidence=0.5)

    db.poses = {}
    loaded = 0

    if not os.path.isdir(datasets_dir):
        print("[ERR] datasets/ folder does not exist")
        return

    for filename in sorted(os.listdir(datasets_dir)):
        if not filename.lower().endswith(IMAGE_EXTENSIONS):
            continue

        image_path = os.path.join(datasets_dir, filename)
        image = load_image(image_path)
        if image is None:
            print(f"[ERR] Could not read image: {filename}")
            continue

        landmarks, confidences, success = detector.detect(image)
        if not success:
            print(f"[ERR] No pose detected: {filename}")
            continue

        features = FeatureVector.extract_features(landmarks, confidences)
        if features is None:
            print(f"[ERR] Pose not usable: {filename}")
            continue

        base_name = os.path.splitext(filename)[0]
        pose_id = f"dataset_{base_name}"
        label = label_from_filename(filename)

        db.add_pose(
            pose_id=pose_id,
            image_path=image_path,
            label=label,
            landmarks=features["landmarks"],
            features=features,
        )
        loaded += 1
        print(f"[OK] Added {filename} -> {label}")

    db.save_database()
    detector.release()
    print(f"[OK] Rebuilt {loaded} reference pose(s)")


if __name__ == "__main__":
    main()
