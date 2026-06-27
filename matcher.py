"""
Pose Matching Module
====================
Handles matching of live poses against stored reference poses.
Implements the complete matching pipeline with multiple algorithms.

Key Features:
- Load and manage reference poses from JSON
- Match live pose against all references
- Generate confidence scores
- Track best matches
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from pose_utils import PoseNormalizer, AngleFeatureExtractor, FeatureVector, SimilarityMetrics
import os


class ReferencePoseDatabase:
    """Manages stored reference poses."""
    
    def __init__(self, db_path: str = "reference_poses.json"):
        """
        Initialize the reference pose database.
        
        Args:
            db_path (str): Path to JSON file storing reference poses
        """
        self.db_path = db_path
        self.poses = {}  # Dict: pose_id -> pose data
        self.load_database()
    
    def load_database(self):
        """Load reference poses from JSON file."""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r') as f:
                    data = json.load(f)
                    
                # Convert lists back to numpy arrays
                for pose_id, pose_data in data.items():
                    pose_data['landmarks'] = np.array(
                        pose_data['landmarks'], dtype=np.float32
                    )
                    if 'valid_mask' in pose_data:
                        pose_data['valid_mask'] = np.array(
                            pose_data['valid_mask'], dtype=bool
                        )
                    else:
                        pose_data['valid_mask'] = (
                            np.linalg.norm(pose_data['landmarks'], axis=1) > 0
                        )
                    pose_data['angle_vector'] = np.array(
                        pose_data['angle_vector'], dtype=np.float32
                    )
                    pose_data['pose_vector'] = np.array(
                        pose_data['pose_vector'], dtype=np.float32
                    )
                    if 'upper_body_vector' in pose_data:
                        pose_data['upper_body_vector'] = np.array(
                            pose_data['upper_body_vector'], dtype=np.float32
                        )
                    else:
                        pose_data['upper_body_vector'] = pose_data['landmarks'][
                            PoseNormalizer.UPPER_BODY_INDICES
                        ].flatten().astype(np.float32)
                    pose_data['angles'] = AngleFeatureExtractor.extract_joint_angles_masked(
                        pose_data['landmarks'],
                        pose_data['valid_mask']
                    )
                    pose_data['angle_vector'] = AngleFeatureExtractor.angles_to_vector(
                        pose_data['angles']
                    )
                
                self.poses = data
                print(f"[OK] Loaded {len(self.poses)} reference poses")
            except Exception as e:
                print(f"[ERR] Error loading database: {e}")
                self.poses = {}
        else:
            print(f"[INFO] Database not found at {self.db_path}. Starting with empty database.")
            self.poses = {}
    
    def save_database(self):
        """Save reference poses to JSON file."""
        try:
            # Convert numpy arrays to lists for JSON serialization
            save_data = {}
            for pose_id, pose_data in self.poses.items():
                save_data[pose_id] = {
                    'image_path': pose_data['image_path'],
                    'label': pose_data['label'],
                    'landmarks': pose_data['landmarks'].tolist(),
                    'valid_mask': pose_data.get(
                        'valid_mask',
                        np.linalg.norm(pose_data['landmarks'], axis=1) > 0
                    ).tolist(),
                    'angle_vector': pose_data['angle_vector'].tolist(),
                    'pose_vector': pose_data['pose_vector'].tolist(),
                    'upper_body_vector': pose_data.get(
                        'upper_body_vector',
                        pose_data['landmarks'][PoseNormalizer.UPPER_BODY_INDICES].flatten()
                    ).tolist(),
                    'angles': pose_data['angles']
                }
            
            with open(self.db_path, 'w') as f:
                json.dump(save_data, f, indent=2)
            print(f"[OK] Saved database with {len(self.poses)} poses")
        except Exception as e:
            print(f"[ERR] Error saving database: {e}")
    
    def add_pose(self, pose_id: str, image_path: str, label: str,
                 landmarks: np.ndarray, features: Dict):
        """
        Add a reference pose to the database.
        
        Args:
            pose_id (str): Unique identifier
            image_path (str): Path to reference image
            label (str): Human-readable label
            landmarks (np.ndarray): Normalized landmarks
            features (Dict): Feature dictionary from FeatureVector.extract_features
        """
        self.poses[pose_id] = {
            'image_path': image_path,
            'label': label,
            'landmarks': landmarks,
            'valid_mask': features.get(
                'valid_mask',
                np.linalg.norm(landmarks, axis=1) > 0
            ),
            'angle_vector': features['angle_vector'],
            'pose_vector': features['pose_vector'],
            'upper_body_vector': features.get(
                'upper_body_vector',
                landmarks[PoseNormalizer.UPPER_BODY_INDICES].flatten().astype(np.float32)
            ),
            'angles': features['angles']
        }
    
    def get_pose(self, pose_id: str) -> Optional[Dict]:
        """Get a specific reference pose."""
        return self.poses.get(pose_id)
    
    def get_all_poses(self) -> Dict:
        """Get all reference poses."""
        return self.poses
    
    def delete_pose(self, pose_id: str) -> bool:
        """Delete a reference pose."""
        if pose_id in self.poses:
            del self.poses[pose_id]
            return True
        return False
    
    def count(self) -> int:
        """Get number of stored poses."""
        return len(self.poses)


class PoseMatcher:
    """Matches live poses against reference poses."""
    
    def __init__(self, db_path: str = "reference_poses.json",
                 euclidean_weight: float = 0.45,
                 angle_weight: float = 0.45,
                 cosine_weight: float = 0.10,
                 match_threshold: float = 0.72,
                 min_margin: float = 0.06):
        """
        Initialize the pose matcher.
        
        Args:
            db_path (str): Path to reference pose database
            euclidean_weight (float): Weight for Euclidean distance
            angle_weight (float): Weight for angle difference
            cosine_weight (float): Weight for cosine similarity
            match_threshold (float): Score threshold for positive match (0-1)
        """
        self.database = ReferencePoseDatabase(db_path)
        self.weights = (euclidean_weight, angle_weight, cosine_weight)
        self.match_threshold = match_threshold
        self.min_margin = min_margin
        
        # Normalize weights
        total = sum(self.weights)
        self.weights = tuple(w / total for w in self.weights)
    
    def match_pose(self, live_landmarks: np.ndarray,
                   live_confidences: np.ndarray = None) -> Tuple[Optional[Dict], float]:
        """
        Match a live pose against all reference poses.
        
        Args:
            live_landmarks (np.ndarray): Live pose keypoints (33, 2)
            live_confidences (np.ndarray): Confidence scores for keypoints
            
        Returns:
            Tuple of:
                - best_match (Dict): Best matching pose or None
                - confidence (float): Match confidence score (0-1)
        """
        if live_landmarks is None or self.database.count() == 0:
            return None, 0.0
        
        # Extract features from live pose
        live_features = FeatureVector.extract_features(live_landmarks, live_confidences)
        if live_features is None:
            return None, 0.0
        
        best_match = None
        best_score = 0.0
        second_score = 0.0
        
        # Compare against all reference poses
        for pose_id, ref_pose in self.database.get_all_poses().items():
            score = self._compute_match_score(live_features, ref_pose)
            
            if score > best_score:
                second_score = best_score
                best_score = score
                best_match = ref_pose
                best_match['pose_id'] = pose_id
            elif score > second_score:
                second_score = score
        
        # Check if above threshold
        margin = best_score - second_score
        confident_margin = margin >= self.min_margin or best_score >= self.match_threshold + 0.10
        if best_score >= self.match_threshold and confident_margin:
            return best_match, best_score
        else:
            return None, best_score
    
    def _compute_match_score(self, live_features: Dict, ref_pose: Dict) -> float:
        """
        Compute match score between live and reference pose.
        
        Args:
            live_features (Dict): Features from live pose
            ref_pose (Dict): Reference pose data
            
        Returns:
            float: Combined match score (0-1)
        """
        normal_score, _ = self._compute_feature_score(live_features, ref_pose)
        mirrored_features = self._mirror_features(live_features)
        mirrored_score, _ = self._compute_feature_score(mirrored_features, ref_pose)
        return max(normal_score, mirrored_score)

    def _compute_feature_score(self, live_features: Dict, ref_pose: Dict) -> Tuple[float, Dict]:
        """Compute score and metric details for one orientation."""
        euclidean_dist = SimilarityMetrics.euclidean_distance(
            live_features['landmarks'],
            ref_pose['landmarks'],
            live_features.get('valid_mask'),
            ref_pose.get('valid_mask')
        )
        
        # Method 2: Angle difference at joints
        angle_diff = SimilarityMetrics.angle_difference(
            live_features['angles'],
            ref_pose['angles']
        )
        
        # Method 3: Cosine similarity between pose vectors
        cosine_sim = SimilarityMetrics.cosine_similarity(
            live_features.get('upper_body_vector', live_features['pose_vector']),
            ref_pose.get('upper_body_vector', ref_pose['pose_vector'])
        )
        
        # Combine methods
        combined_score = SimilarityMetrics.compute_combined_score(
            euclidean_dist, angle_diff, cosine_sim, self.weights
        )
        
        return combined_score, {
            'euclidean_distance': euclidean_dist,
            'angle_difference': angle_diff,
            'cosine_similarity': cosine_sim,
            'combined_score': combined_score
        }

    def _mirror_features(self, features: Dict) -> Dict:
        """Return a mirrored copy of extracted features for camera/reference parity."""
        mirrored_landmarks = PoseNormalizer.mirror_pose(features['landmarks'])
        mirrored_mask = features.get('valid_mask', np.ones(33, dtype=bool)).copy()
        for left_idx, right_idx in PoseNormalizer.MIRROR_PAIRS:
            mirrored_mask[[left_idx, right_idx]] = mirrored_mask[[right_idx, left_idx]]
        mirrored_angles = AngleFeatureExtractor.extract_joint_angles_masked(
            mirrored_landmarks, mirrored_mask
        )
        return {
            'landmarks': mirrored_landmarks,
            'valid_mask': mirrored_mask,
            'angles': mirrored_angles,
            'angle_vector': AngleFeatureExtractor.angles_to_vector(mirrored_angles),
            'pose_vector': mirrored_landmarks.flatten().astype(np.float32),
            'upper_body_vector': mirrored_landmarks[PoseNormalizer.UPPER_BODY_INDICES].flatten().astype(np.float32),
            'is_valid': features.get('is_valid', True)
        }
    
    def match_pose_detailed(self, live_landmarks: np.ndarray,
                           live_confidences: np.ndarray = None) -> Dict:
        """
        Get detailed matching information for all reference poses.
        
        Args:
            live_landmarks (np.ndarray): Live pose keypoints (33, 2)
            live_confidences (np.ndarray): Confidence scores
            
        Returns:
            Dict with detailed scores for all poses
        """
        if live_landmarks is None or self.database.count() == 0:
            return {}
        
        # Extract features
        live_features = FeatureVector.extract_features(live_landmarks, live_confidences)
        if live_features is None:
            return {}
        
        results = {}
        
        # Score each reference pose
        for pose_id, ref_pose in self.database.get_all_poses().items():
            normal_score, normal_details = self._compute_feature_score(live_features, ref_pose)
            mirrored_score, mirrored_details = self._compute_feature_score(
                self._mirror_features(live_features), ref_pose
            )
            if mirrored_score > normal_score:
                details = mirrored_details
                details['orientation'] = 'mirrored'
            else:
                details = normal_details
                details['orientation'] = 'normal'

            combined_score = details['combined_score']
            
            results[pose_id] = {
                'label': ref_pose['label'],
                'image_path': ref_pose['image_path'],
                'euclidean_distance': details['euclidean_distance'],
                'angle_difference': details['angle_difference'],
                'cosine_similarity': details['cosine_similarity'],
                'combined_score': combined_score,
                'orientation': details['orientation'],
                'is_match': combined_score >= self.match_threshold
            }
        
        sorted_scores = sorted(
            ((pose_id, row['combined_score']) for pose_id, row in results.items()),
            key=lambda item: item[1],
            reverse=True
        )
        for index, (pose_id, score) in enumerate(sorted_scores):
            second = sorted_scores[1][1] if index == 0 and len(sorted_scores) > 1 else (
                sorted_scores[0][1] if index != 0 and sorted_scores else 0.0
            )
            margin = score - second
            results[pose_id]['margin'] = margin
            results[pose_id]['is_match'] = (
                score >= self.match_threshold and
                (margin >= self.min_margin or score >= self.match_threshold + 0.10)
            )
        
        return results
    
    def set_threshold(self, threshold: float):
        """Update match threshold."""
        self.match_threshold = np.clip(threshold, 0.0, 1.0)
    
    def set_weights(self, euclidean_weight: float, angle_weight: float, cosine_weight: float):
        """Update combination weights."""
        self.weights = (euclidean_weight, angle_weight, cosine_weight)
        total = sum(self.weights)
        self.weights = tuple(w / total for w in self.weights)


class MatchTracker:
    """Tracks pose matches over multiple frames for stability."""
    
    def __init__(self, history_size: int = 5, stability_threshold: float = 0.6):
        """
        Initialize match tracker.
        
        Args:
            history_size (int): Number of frames to track
            stability_threshold (float): Confidence threshold for stable match
        """
        self.history_size = history_size
        self.stability_threshold = stability_threshold
        self.match_history = []  # List of (pose_id, confidence) tuples
    
    def update(self, pose_id: Optional[str], confidence: float):
        """
        Update match history.
        
        Args:
            pose_id (str): ID of matched pose or None
            confidence (float): Match confidence
        """
        self.match_history.append((pose_id, confidence))
        
        # Keep only recent history
        if len(self.match_history) > self.history_size:
            self.match_history.pop(0)
    
    def get_stable_match(self) -> Tuple[Optional[str], float]:
        """
        Get stable match if majority of recent frames agree.
        
        Returns:
            Tuple of (pose_id, average_confidence) or (None, 0.0)
        """
        if not self.match_history:
            return None, 0.0
        
        # Count pose occurrences
        pose_counts = {}
        for pose_id, conf in self.match_history:
            if pose_id is not None:
                if pose_id not in pose_counts:
                    pose_counts[pose_id] = {'count': 0, 'confidence': 0.0}
                pose_counts[pose_id]['count'] += 1
                pose_counts[pose_id]['confidence'] += conf
        
        if not pose_counts:
            return None, 0.0
        
        # Find most frequent pose
        best_pose_id = max(pose_counts.keys(), 
                          key=lambda x: pose_counts[x]['count'])
        count = pose_counts[best_pose_id]['count']
        avg_confidence = pose_counts[best_pose_id]['confidence'] / count
        
        # Check if stable (majority agreement + above threshold)
        if count >= self.history_size / 2 and avg_confidence >= self.stability_threshold:
            return best_pose_id, avg_confidence
        
        return None, avg_confidence
    
    def is_stable(self) -> bool:
        """Check if current match is stable."""
        pose_id, _ = self.get_stable_match()
        return pose_id is not None

