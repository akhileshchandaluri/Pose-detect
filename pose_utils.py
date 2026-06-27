"""
Pose Utilities Module
====================
Handles pose normalization, feature extraction, and angle calculations.
Critical for making pose matching robust to scale, position, and rotation variations.

Key Functions:
- Pose normalization (scale & position invariance)
- Angle calculations at joints (elbow, shoulder, knee, hip)
- Feature vector generation
- Cosine similarity computation
"""

import numpy as np
from typing import Tuple, List, Dict, Optional


class PoseNormalizer:
    """Handles all pose normalization operations for robustness."""
    
    # Key body part indices (for reference_detector.py)
    LEFT_SHOULDER_IDX = 11
    RIGHT_SHOULDER_IDX = 12
    LEFT_HIP_IDX = 23
    RIGHT_HIP_IDX = 24
    LEFT_ELBOW_IDX = 13
    RIGHT_ELBOW_IDX = 14
    LEFT_WRIST_IDX = 15
    RIGHT_WRIST_IDX = 16
    LEFT_KNEE_IDX = 25
    RIGHT_KNEE_IDX = 26
    LEFT_ANKLE_IDX = 27
    RIGHT_ANKLE_IDX = 28

    UPPER_BODY_INDICES = np.array([
        0, 7, 8, 9, 10,          # head / face anchors
        11, 12, 13, 14, 15, 16,  # shoulders, elbows, wrists
        17, 18, 19, 20, 21, 22,  # hand hints from MediaPipe
        23, 24                   # hips for torso orientation
    ], dtype=np.int32)

    MIRROR_PAIRS = [
        (1, 4), (2, 5), (3, 6), (7, 8), (9, 10),
        (11, 12), (13, 14), (15, 16), (17, 18),
        (19, 20), (21, 22), (23, 24), (25, 26),
        (27, 28), (29, 30), (31, 32)
    ]
    
    @staticmethod
    def compute_torso_length(landmarks: np.ndarray) -> float:
        """
        Compute torso length as the distance between shoulders.
        Used for scale normalization.
        
        Args:
            landmarks (np.ndarray): Pose keypoints (33, 2)
            
        Returns:
            float: Distance between shoulders
        """
        left_shoulder = landmarks[PoseNormalizer.LEFT_SHOULDER_IDX]
        right_shoulder = landmarks[PoseNormalizer.RIGHT_SHOULDER_IDX]
        
        distance = np.linalg.norm(left_shoulder - right_shoulder)
        return max(distance, 1e-6)  # Avoid division by zero

    @staticmethod
    def get_valid_mask(confidences: np.ndarray = None,
                       confidence_threshold: float = 0.35) -> np.ndarray:
        """Return a per-landmark visibility mask."""
        if confidences is None:
            return np.ones(33, dtype=bool)
        return np.asarray(confidences) >= confidence_threshold

    @staticmethod
    def mirror_pose(landmarks: np.ndarray) -> np.ndarray:
        """Mirror a normalized pose and swap left/right semantic landmarks."""
        mirrored = landmarks.copy()
        mirrored[:, 0] *= -1.0
        for left_idx, right_idx in PoseNormalizer.MIRROR_PAIRS:
            mirrored[[left_idx, right_idx]] = mirrored[[right_idx, left_idx]]
        return mirrored
    
    @staticmethod
    def normalize_pose(landmarks: np.ndarray, confidences: np.ndarray = None,
                       confidence_threshold: float = 0.35) -> Tuple[np.ndarray, bool]:
        """
        Normalize pose to be invariant to scale and position.
        
        Process:
        1. Position normalization: Translate to origin using hip center
        2. Scale normalization: Divide by torso length
        3. Filter by confidence if provided
        
        Args:
            landmarks (np.ndarray): Pose keypoints (33, 2)
            confidences (np.ndarray): Optional confidence scores
            confidence_threshold (float): Minimum confidence to keep keypoint
            
        Returns:
            Tuple of:
                - normalized_pose (np.ndarray): Normalized keypoints (33, 2)
                - valid (bool): True if pose has enough valid keypoints
        """
        if landmarks is None:
            return None, False
        
        landmarks = landmarks.astype(np.float32).copy()
        
        valid_mask = PoseNormalizer.get_valid_mask(confidences, confidence_threshold)
        valid_mask &= np.isfinite(landmarks).all(axis=1)

        left_shoulder = landmarks[PoseNormalizer.LEFT_SHOULDER_IDX]
        right_shoulder = landmarks[PoseNormalizer.RIGHT_SHOULDER_IDX]
        shoulders_valid = (
            valid_mask[PoseNormalizer.LEFT_SHOULDER_IDX] and
            valid_mask[PoseNormalizer.RIGHT_SHOULDER_IDX]
        )
        if not shoulders_valid:
            return None, False

        torso_indices = [
            PoseNormalizer.LEFT_SHOULDER_IDX, PoseNormalizer.RIGHT_SHOULDER_IDX,
            PoseNormalizer.LEFT_HIP_IDX, PoseNormalizer.RIGHT_HIP_IDX
        ]
        center_points = landmarks[[i for i in torso_indices if valid_mask[i]]]
        body_center = np.mean(center_points, axis=0) if len(center_points) else (left_shoulder + right_shoulder) / 2.0

        # Step 1: Translate to origin (position invariance)
        landmarks_centered = landmarks - body_center

        # Step 2: Rotate so the shoulder line is horizontal. This reduces camera tilt
        # and reference-image crop differences without destroying arm geometry.
        shoulder_vec = right_shoulder - left_shoulder
        shoulder_angle = np.arctan2(shoulder_vec[1], shoulder_vec[0])
        cos_a = np.cos(-shoulder_angle)
        sin_a = np.sin(-shoulder_angle)
        rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
        landmarks_centered = landmarks_centered @ rotation.T

        # Step 3: Scale normalization. Shoulder width alone is noisy for angled
        # photos, so include visible torso height and visible upper-body spread.
        scale_candidates = [np.linalg.norm(shoulder_vec)]
        if valid_mask[PoseNormalizer.LEFT_HIP_IDX]:
            scale_candidates.append(np.linalg.norm(left_shoulder - landmarks[PoseNormalizer.LEFT_HIP_IDX]))
        if valid_mask[PoseNormalizer.RIGHT_HIP_IDX]:
            scale_candidates.append(np.linalg.norm(right_shoulder - landmarks[PoseNormalizer.RIGHT_HIP_IDX]))
        visible_upper = landmarks_centered[PoseNormalizer.UPPER_BODY_INDICES[
            valid_mask[PoseNormalizer.UPPER_BODY_INDICES]
        ]]
        if len(visible_upper) > 1:
            spread = np.max(visible_upper, axis=0) - np.min(visible_upper, axis=0)
            scale_candidates.append(float(np.linalg.norm(spread) * 0.35))

        scale = max([float(s) for s in scale_candidates if np.isfinite(s)], default=1e-6)
        scale = max(scale, 1e-6)
        landmarks_normalized = landmarks_centered / scale

        # Step 4: Filter by confidence if provided
        if confidences is not None:
            low_confidence_mask = ~valid_mask
            landmarks_normalized[low_confidence_mask] = 0.0  # Zero out low-confidence points
        
        # Check if we have enough valid keypoints
        upper_valid_count = int(np.sum(valid_mask[PoseNormalizer.UPPER_BODY_INDICES]))
        is_valid = upper_valid_count >= 8
        
        return landmarks_normalized, is_valid
    
    @staticmethod
    def compute_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """
        Compute angle at p2 formed by vectors p1->p2 and p2->p3.
        Returns angle in degrees (0-180).
        
        Args:
            p1, p2, p3 (np.ndarray): Points with shape (2,) for (x, y)
            
        Returns:
            float: Angle in degrees
        """
        # Create vectors
        v1 = p1 - p2
        v2 = p3 - p2
        
        # Compute dot product and magnitudes
        dot_product = np.dot(v1, v2)
        magnitude_v1 = np.linalg.norm(v1)
        magnitude_v2 = np.linalg.norm(v2)
        
        # Avoid division by zero
        if magnitude_v1 < 1e-6 or magnitude_v2 < 1e-6:
            return 0.0
        
        # Compute cosine and clamp to [-1, 1]
        cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        # Convert to degrees
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg


class AngleFeatureExtractor:
    """Extract angle-based features from normalized poses."""
    
    @staticmethod
    def extract_joint_angles(landmarks: np.ndarray) -> Dict[str, float]:
        """
        Extract angles at major joints.
        
        Args:
            landmarks (np.ndarray): Normalized pose keypoints (33, 2)
            
        Returns:
            Dict with angle names and values
        """
        angles = {}
        
        # Left side angles
        angles['left_elbow'] = PoseNormalizer.compute_angle(
            landmarks[11],  # left shoulder
            landmarks[13],  # left elbow
            landmarks[15]   # left wrist
        )
        
        angles['left_shoulder'] = PoseNormalizer.compute_angle(
            landmarks[12],  # right shoulder (front reference)
            landmarks[11],  # left shoulder
            landmarks[13]   # left elbow
        )
        
        angles['left_hip'] = PoseNormalizer.compute_angle(
            landmarks[12],  # right shoulder
            landmarks[23],  # left hip
            landmarks[25]   # left knee
        )
        
        angles['left_knee'] = PoseNormalizer.compute_angle(
            landmarks[23],  # left hip
            landmarks[25],  # left knee
            landmarks[27]   # left ankle
        )
        
        # Right side angles
        angles['right_elbow'] = PoseNormalizer.compute_angle(
            landmarks[12],  # right shoulder
            landmarks[14],  # right elbow
            landmarks[16]   # right wrist
        )
        
        angles['right_shoulder'] = PoseNormalizer.compute_angle(
            landmarks[11],  # left shoulder (front reference)
            landmarks[12],  # right shoulder
            landmarks[14]   # right elbow
        )
        
        angles['right_hip'] = PoseNormalizer.compute_angle(
            landmarks[11],  # left shoulder
            landmarks[24],  # right hip
            landmarks[26]   # right knee
        )
        
        angles['right_knee'] = PoseNormalizer.compute_angle(
            landmarks[24],  # right hip
            landmarks[26],  # right knee
            landmarks[28]   # right ankle
        )
        
        return angles

    @staticmethod
    def extract_joint_angles_masked(landmarks: np.ndarray, valid_mask: np.ndarray) -> Dict[str, float]:
        """Extract joint angles, using NaN for joints whose landmarks are hidden."""
        angle_specs = {
            'left_elbow': (11, 13, 15),
            'left_shoulder': (12, 11, 13),
            'left_hip': (12, 23, 25),
            'left_knee': (23, 25, 27),
            'right_elbow': (12, 14, 16),
            'right_shoulder': (11, 12, 14),
            'right_hip': (11, 24, 26),
            'right_knee': (24, 26, 28),
        }
        angles = {}
        for name, (p1, p2, p3) in angle_specs.items():
            if valid_mask[p1] and valid_mask[p2] and valid_mask[p3]:
                angles[name] = PoseNormalizer.compute_angle(landmarks[p1], landmarks[p2], landmarks[p3])
            else:
                angles[name] = np.nan
        return angles
    
    @staticmethod
    def angles_to_vector(angles: Dict[str, float]) -> np.ndarray:
        """Convert angle dictionary to feature vector."""
        angle_order = [
            'left_elbow', 'left_shoulder', 'left_hip', 'left_knee',
            'right_elbow', 'right_shoulder', 'right_hip', 'right_knee'
        ]
        return np.array([angles[name] for name in angle_order], dtype=np.float32)


class FeatureVector:
    """Generate complete feature vectors for pose matching."""
    
    @staticmethod
    def flatten_pose(landmarks: np.ndarray) -> np.ndarray:
        """
        Flatten normalized pose to 1D vector.
        
        Args:
            landmarks (np.ndarray): Pose keypoints (33, 2)
            
        Returns:
            np.ndarray: Flattened vector (66,)
        """
        return landmarks.flatten().astype(np.float32)
    
    @staticmethod
    def extract_features(landmarks: np.ndarray, confidences: np.ndarray = None) -> Dict:
        """
        Extract all features from a pose.
        
        Args:
            landmarks (np.ndarray): Raw pose keypoints
            confidences (np.ndarray): Confidence scores
            
        Returns:
            Dict with normalized pose, angles, and feature vector
        """
        valid_mask = PoseNormalizer.get_valid_mask(confidences)

        # Normalize pose
        normalized_landmarks, is_valid = PoseNormalizer.normalize_pose(
            landmarks, confidences
        )
        
        if not is_valid:
            return None
        
        valid_mask &= np.linalg.norm(normalized_landmarks, axis=1) > 0

        # Extract angles. Missing joints are represented as NaN and ignored during
        # matching, which is essential for half-body reference photos.
        angles = AngleFeatureExtractor.extract_joint_angles_masked(normalized_landmarks, valid_mask)
        angle_vector = AngleFeatureExtractor.angles_to_vector(angles)
        
        # Flatten pose
        pose_vector = FeatureVector.flatten_pose(normalized_landmarks)
        upper_body_vector = normalized_landmarks[PoseNormalizer.UPPER_BODY_INDICES].flatten().astype(np.float32)
        
        return {
            'landmarks': normalized_landmarks,
            'valid_mask': valid_mask.astype(bool),
            'angles': angles,
            'angle_vector': angle_vector,
            'pose_vector': pose_vector,
            'upper_body_vector': upper_body_vector,
            'is_valid': is_valid
        }


class SimilarityMetrics:
    """Compute similarity metrics between two poses."""
    
    @staticmethod
    def euclidean_distance(landmarks_1: np.ndarray, landmarks_2: np.ndarray,
                          confidences_1: np.ndarray = None,
                          confidences_2: np.ndarray = None) -> float:
        """
        Compute average Euclidean distance between corresponding keypoints.
        Lower is better.
        
        Args:
            landmarks_1, landmarks_2 (np.ndarray): Pose keypoints (33, 2)
            confidences_1, confidences_2: Optional confidence scores
            
        Returns:
            float: Average distance (normalized)
        """
        if landmarks_1 is None or landmarks_2 is None:
            return float('inf')
        
        valid_mask = np.ones(landmarks_1.shape[0], dtype=bool)
        if confidences_1 is not None:
            valid_mask &= np.asarray(confidences_1).astype(bool)
        if confidences_2 is not None:
            valid_mask &= np.asarray(confidences_2).astype(bool)

        upper_mask = np.zeros(landmarks_1.shape[0], dtype=bool)
        upper_mask[PoseNormalizer.UPPER_BODY_INDICES] = True
        valid_mask &= upper_mask

        if np.sum(valid_mask) < 6:
            return float('inf')

        distances = np.linalg.norm(landmarks_1[valid_mask] - landmarks_2[valid_mask], axis=1)
        weights = np.ones_like(distances)
        valid_indices = np.where(valid_mask)[0]
        for idx, landmark_idx in enumerate(valid_indices):
            if landmark_idx in (11, 12, 13, 14, 15, 16):
                weights[idx] = 1.35
            elif landmark_idx in (23, 24):
                weights[idx] = 0.85

        return float(np.average(distances, weights=weights))
    
    @staticmethod
    def angle_difference(angles_1: Dict[str, float], angles_2: Dict[str, float]) -> float:
        """
        Compute average absolute angle difference across all joints.
        Lower is better.
        
        Args:
            angles_1, angles_2 (Dict): Joint angles
            
        Returns:
            float: Average angle difference in degrees
        """
        if not angles_1 or not angles_2:
            return float('inf')
        
        differences = []
        for key in angles_1:
            if key in angles_2:
                a = angles_1[key]
                b = angles_2[key]
                if np.isfinite(a) and np.isfinite(b):
                    differences.append(abs(a - b))
        
        if not differences:
            return float('inf')
        
        return float(np.mean(differences))
    
    @staticmethod
    def cosine_similarity(vector_1: np.ndarray, vector_2: np.ndarray) -> float:
        """
        Compute cosine similarity between two feature vectors.
        Returns value in [-1, 1]. Higher is better.
        
        Args:
            vector_1, vector_2 (np.ndarray): Feature vectors
            
        Returns:
            float: Cosine similarity score in [-1, 1]
        """
        if vector_1 is None or vector_2 is None:
            return 0.0
        
        vector_1 = np.nan_to_num(vector_1, nan=0.0, posinf=0.0, neginf=0.0)
        vector_2 = np.nan_to_num(vector_2, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize vectors
        norm_1 = np.linalg.norm(vector_1)
        norm_2 = np.linalg.norm(vector_2)
        
        if norm_1 < 1e-6 or norm_2 < 1e-6:
            return 0.0
        
        # Compute cosine similarity
        similarity = np.dot(vector_1, vector_2) / (norm_1 * norm_2)
        return float(np.clip(similarity, -1.0, 1.0))
    
    @staticmethod
    def compute_combined_score(
        euclidean_dist: float,
        angle_diff: float,
        cosine_sim: float,
        weights: Tuple[float, float, float] = (0.4, 0.35, 0.25)
    ) -> float:
        """
        Combine multiple similarity metrics into a single score.
        Lower euclidean and angle difference = better.
        Higher cosine similarity = better.
        
        Args:
            euclidean_dist (float): Average keypoint distance
            angle_diff (float): Average angle difference (degrees)
            cosine_sim (float): Cosine similarity [-1, 1]
            weights (Tuple): Weights for each metric
            
        Returns:
            float: Combined score (0-1, higher is better match)
        """
        w_euclidean, w_angle, w_cosine = weights
        
        if not np.isfinite(euclidean_dist):
            euclidean_score = 0.0
        else:
            euclidean_score = np.exp(-euclidean_dist * 4.0)
        
        if not np.isfinite(angle_diff):
            angle_score = 0.0
        else:
            angle_score = np.exp(-(angle_diff / 45.0))
        
        # Cosine similarity already in [-1, 1], convert to [0, 1]
        cosine_score = (cosine_sim + 1.0) / 2.0
        
        # Combine
        combined = (w_euclidean * euclidean_score +
                   w_angle * angle_score +
                   w_cosine * cosine_score)
        
        return float(np.clip(combined, 0.0, 1.0))

