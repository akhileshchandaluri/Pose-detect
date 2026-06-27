"""
Simple pose capture tool for recording new reference poses.

Usage:
    python capture.py
    
Controls:
    SPACE - Capture current pose
    S     - Save pose with label and continue
    Q     - Quit without saving
"""

import cv2
import numpy as np
import sys
import time
from pose_detector import PoseDetector
from pose_utils import PoseNormalizer, FeatureVector
from matcher import ReferencePoseDatabase

def main():
    """Main capture application."""
    print("\n" + "="*70)
    print("POSE CAPTURE TOOL - Record New Reference Poses")
    print("="*70)
    print("\nInstructions:")
    print("  1. Position yourself in front of the camera")
    print("  2. Hold a pose")
    print("  3. Press SPACE to capture")
    print("  4. Type a label when prompted")
    print("  5. Repeat for multiple poses")
    print("\nControls:")
    print("  SPACE  - Capture current pose")
    print("  Q      - Quit")
    print("="*70 + "\n")
    
    # Initialize
    detector = PoseDetector(model_complexity=1, min_detection_confidence=0.5)
    database = ReferencePoseDatabase("reference_poses.json")
    
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    camera.set(cv2.CAP_PROP_FPS, 30)
    
    poses_captured = 0
    
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)  # Mirror
        
        # Detect pose
        landmarks, confidences, success = detector.detect(frame)
        
        # Draw frame
        h, w = frame.shape[:2]
        
        if success and landmarks is not None:
            # Draw skeleton
            frame = detector.draw_pose(frame, landmarks, confidences)
            
            # Draw status
            cv2.putText(frame, "POSE DETECTED!", (50, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(frame, "Press SPACE to capture", (50, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 100), 2)
        else:
            # Draw status
            cv2.putText(frame, "No pose detected", (50, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(frame, "Position yourself in frame", (50, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 1)
        
        # Draw controls info
        cv2.putText(frame, f"Captured: {poses_captured} pose(s) | Q: Quit", (50, h-30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Display
        cv2.imshow("Pose Capture", frame)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\n[OK] Capture complete. Quitting...")
            break
        
        elif key == ord(' '):  # Space - capture
            if not success or landmarks is None:
                print("[ERR] No pose detected! Cannot capture.")
                continue
            
            # Extract features
            features = FeatureVector.extract_features(landmarks, confidences)
            if features is None:
                print("[ERR] Failed to extract features")
                continue
            
            # Normalize pose
            normalized_landmarks, _ = PoseNormalizer.normalize_pose(landmarks, confidences)
            
            # Get label from user
            cv2.destroyWindow("Pose Capture")
            
            print("\n" + "-"*70)
            label = input("Enter pose label (e.g., 'standing', 'sitting', 'arms_up'): ").strip()
            
            if not label:
                print("[ERR] No label provided. Cancelled.")
                cv2.imshow("Pose Capture", frame)
                continue
            
            # Generate pose ID
            pose_id = f"captured_{label}_{int(time.time())}"
            
            # Add to database
            database.add_pose(
                pose_id=pose_id,
                image_path="",
                label=label,
                landmarks=normalized_landmarks,
                features=features
            )
            
            database.save_database()
            
            print(f"[OK] Captured pose: '{label}'")
            print("-"*70)
            
            poses_captured += 1
            
            # Re-open camera window
            cv2.imshow("Pose Capture", frame)
    
    # Cleanup
    camera.release()
    detector.release()
    cv2.destroyAllWindows()
    
    print(f"\n[OK] Captured {poses_captured} pose(s) total!")
    print("[OK] Poses saved to reference_poses.json")
    print("\nRun: python main.py --mode live")
    print("     to start matching!\n")

if __name__ == "__main__":
    main()

