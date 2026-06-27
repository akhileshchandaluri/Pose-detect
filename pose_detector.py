"""
Pose Detector Module
=====================
Handles real-time pose detection using MediaPipe Pose.
Extracts 33 keypoints from the human body with normalized coordinates.

Key Features:
- Real-time pose detection from video frames
- Normalized coordinate system (0-1 range)
- Confidence scores for each keypoint
- Handles multiple people (uses the most confident one)
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, List, Dict, Optional


class PoseDetector:
    """
    MediaPipe-based pose detector for extracting human keypoints.
    
    Attributes:
        KEYPOINT_NAMES: Names of all 33 MediaPipe pose keypoints
        mp_pose: MediaPipe Pose model
        pose: Pose inference pipeline
    """
    
    # MediaPipe Pose Keypoint names (33 keypoints)
    KEYPOINT_NAMES = [
        "nose", "left_eye_inner", "left_eye", "left_eye_outer",
        "right_eye_inner", "right_eye", "right_eye_outer",
        "left_ear", "right_ear",
        "mouth_left", "mouth_right",
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
        "left_pinky", "right_pinky",
        "left_index", "right_index",
        "left_thumb", "right_thumb",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle",
        "left_heel", "right_heel",
        "left_foot_index", "right_foot_index"
    ]
    
    def __init__(self, model_complexity: int = 1, min_detection_confidence: float = 0.7):
        """
        Initialize the pose detector.
        
        Args:
            model_complexity (int): 0 (faster), 1 (balanced), 2 (accurate)
            min_detection_confidence (float): Minimum confidence for detection
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def detect(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], bool]:
        """
        Detect pose in a single frame.
        
        Args:
            frame (np.ndarray): Input frame (BGR format from OpenCV)
            
        Returns:
            Tuple containing:
                - landmarks (np.ndarray): Array of shape (33, 2) with normalized (x, y) coordinates
                - confidences (np.ndarray): Confidence scores for each keypoint
                - success (bool): True if pose detected, False otherwise
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        if not results.pose_landmarks:
            return None, None, False
        
        # Extract landmarks (33 keypoints with x, y, z, visibility)
        landmarks = results.pose_landmarks.landmark
        
        # Convert to normalized coordinates (x, y only)
        pose_points = np.array([
            [landmark.x, landmark.y] for landmark in landmarks
        ], dtype=np.float32)
        
        # Extract visibility/confidence scores
        confidences = np.array([
            landmark.visibility for landmark in landmarks
        ], dtype=np.float32)
        
        return pose_points, confidences, True
    
    def draw_pose(self, frame: np.ndarray, landmarks: np.ndarray, 
                  confidences: np.ndarray = None, threshold: float = 0.5) -> np.ndarray:
        """
        Draw pose skeleton on frame.
        
        Args:
            frame (np.ndarray): Input frame
            landmarks (np.ndarray): Pose keypoints (33, 2)
            confidences (np.ndarray): Confidence scores for filtering
            threshold (float): Minimum confidence to draw keypoint
            
        Returns:
            np.ndarray: Frame with drawn pose skeleton
        """
        frame_h, frame_w = frame.shape[:2]
        
        # Draw connections between keypoints
        connections = self.mp_pose.POSE_CONNECTIONS
        for connection in connections:
            start_idx, end_idx = connection
            
            # Check confidence threshold
            if confidences is not None:
                if confidences[start_idx] < threshold or confidences[end_idx] < threshold:
                    continue
            
            start_point = landmarks[start_idx]
            end_point = landmarks[end_idx]
            
            # Convert normalized coordinates to pixel coordinates
            start_pos = (int(start_point[0] * frame_w), int(start_point[1] * frame_h))
            end_pos = (int(end_point[0] * frame_w), int(end_point[1] * frame_h))
            
            cv2.line(frame, start_pos, end_pos, (0, 255, 0), 2)
        
        # Draw keypoints (circles)
        for i, (landmark, name) in enumerate(zip(landmarks, self.KEYPOINT_NAMES)):
            if confidences is not None and confidences[i] < threshold:
                continue
            
            x = int(landmark[0] * frame_w)
            y = int(landmark[1] * frame_h)
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        
        return frame
    
    def get_keypoint_index(self, name: str) -> int:
        """Get the index of a keypoint by name."""
        return self.KEYPOINT_NAMES.index(name)
    
    def release(self):
        """Release resources."""
        self.pose.close()

