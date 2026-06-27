"""
Calibration and Testing Script
===============================
Tools to test and calibrate the pose matching system.

Usage:
    python calibration.py --test detection     # Test pose detection
    python calibration.py --test matching      # Test pose matching
    python calibration.py --test database      # Check database
    python calibration.py --benchmark          # Benchmark performance
"""

import cv2
import numpy as np
import json
import time
import argparse
from typing import List, Tuple
from pose_detector import PoseDetector
from pose_utils import FeatureVector, SimilarityMetrics, PoseNormalizer
from matcher import PoseMatcher


class SystemTester:
    """Test and calibrate the pose matching system."""
    
    def __init__(self):
        """Initialize tester."""
        self.detector = PoseDetector(model_complexity=1, min_detection_confidence=0.7)
        self.matcher = PoseMatcher()
    
    def test_detection(self, duration: int = 10):
        """
        Test pose detection in real-time.
        
        Args:
            duration (int): Test duration in seconds
        """
        print("\n" + "="*70)
        print("POSE DETECTION TEST")
        print("="*70)
        print(f"Testing for {duration} seconds...")
        print("Controls: Q to quit early, SPACE to analyze single frame")
        print("-"*70)
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERR] Cannot open webcam")
            return
        
        start_time = time.time()
        frame_count = 0
        detection_count = 0
        keypoint_counts = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            frame_count += 1
            
            # Detect pose
            t0 = time.time()
            landmarks, confidences, success = self.detector.detect(frame)
            detection_time = (time.time() - t0) * 1000  # Convert to ms
            
            if success:
                detection_count += 1
                valid_keypoints = np.sum(confidences > 0.5)
                keypoint_counts.append(valid_keypoints)
                
                # Draw pose
                frame = self.detector.draw_pose(frame, landmarks, confidences)
            
            # Display stats
            h, w = frame.shape[:2]
            cv2.putText(frame, f"Detection time: {detection_time:.1f}ms", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Detected: {'Yes' if success else 'No'}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                       (0, 255, 0) if success else (0, 0, 255), 2)
            
            if success and confidences is not None:
                valid = np.sum(confidences > 0.5)
                cv2.putText(frame, f"Valid keypoints: {valid}/33", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            
            elapsed = time.time() - start_time
            cv2.putText(frame, f"Elapsed: {elapsed:.1f}s", (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 1)
            
            cv2.imshow("Detection Test", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            if elapsed > duration:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Print results
        print("\nRESULTS:")
        print("-"*70)
        print(f"Total frames: {frame_count}")
        print(f"Detected poses: {detection_count}")
        print(f"Detection rate: {detection_count/frame_count*100:.1f}%")
        print(f"Average detection time: {np.mean([t for t in range(detection_count)])*1000/frame_count:.1f}ms")
        
        if keypoint_counts:
            print(f"Average valid keypoints: {np.mean(keypoint_counts):.1f}/33")
            print(f"Min valid keypoints: {np.min(keypoint_counts)}/33")
            print(f"Max valid keypoints: {np.max(keypoint_counts)}/33")
        
        print("="*70 + "\n")
    
    def test_matching(self):
        """
        Test pose matching with database.
        Compares live pose against stored references.
        """
        print("\n" + "="*70)
        print("POSE MATCHING TEST")
        print("="*70)
        
        if self.matcher.database.count() == 0:
            print("[ERR] No reference poses in database!")
            print("Run: python main.py --mode capture")
            return
        
        print(f"Database contains {self.matcher.database.count()} reference poses")
        print("Position yourself and press SPACE to test match")
        print("Controls: SPACE to test, M for detailed scores, Q to quit")
        print("-"*70)
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERR] Cannot open webcam")
            return
        
        test_results = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Detect pose
            landmarks, confidences, success = self.detector.detect(frame)
            
            if success:
                frame = self.detector.draw_pose(frame, landmarks, confidences)
                cv2.putText(frame, "Pose detected. Press SPACE to test", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No pose detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow("Matching Test", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # Space - test match
                if success:
                    self._test_single_match(landmarks, confidences, test_results)
            elif key == ord('m'):  # M - detailed scores
                if success:
                    self._show_detailed_match(landmarks, confidences)
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Summary
        if test_results:
            print("\n" + "="*70)
            print("TEST SUMMARY")
            print("-"*70)
            print(f"Tests performed: {len(test_results)}")
            
            scores = [r['score'] for r in test_results]
            matches = [r['is_match'] for r in test_results]
            
            print(f"Average score: {np.mean(scores):.3f}")
            print(f"Min score: {np.min(scores):.3f}")
            print(f"Max score: {np.max(scores):.3f}")
            print(f"Matches found: {np.sum(matches)}/{len(matches)}")
            print("="*70 + "\n")
    
    def _test_single_match(self, landmarks: np.ndarray, confidences: np.ndarray,
                           results: List):
        """Test single pose match."""
        matched_pose, confidence = self.matcher.match_pose(landmarks, confidences)
        
        is_match = confidence >= self.matcher.match_threshold
        results.append({'score': confidence, 'is_match': is_match})
        
        print(f"\nMatch result:")
        if is_match:
            print(f"  [OK] MATCHED: {matched_pose['label']}")
            print(f"  Confidence: {confidence:.3f}")
        else:
            print(f"  [ERR] NO MATCH (score: {confidence:.3f})")
        print(f"  Threshold: {self.matcher.match_threshold:.3f}")
    
    def _show_detailed_match(self, landmarks: np.ndarray, confidences: np.ndarray):
        """Show detailed matching scores."""
        results = self.matcher.match_pose_detailed(landmarks, confidences)
        
        print("\n" + "="*90)
        print("DETAILED MATCH SCORES")
        print("="*90)
        print(f"{'Rank':<6} {'Label':<20} {'Euclidean':<12} {'Angle Diff':<12} {'Cosine':<10} {'Score':<10} {'Match':<8}")
        print("-"*90)
        
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1]['combined_score'],
            reverse=True
        )
        
        for rank, (pose_id, scores) in enumerate(sorted_results[:15], 1):
            label = scores['label'][:19]
            euclidean = scores['euclidean_distance']
            angle_diff = scores['angle_difference']
            cosine = scores['cosine_similarity']
            combined = scores['combined_score']
            is_match = "[OK]" if scores['is_match'] else "[ERR]"
            
            print(f"{rank:<6} {label:<20} {euclidean:<12.4f} {angle_diff:<12.2f} deg {cosine:<10.3f} {combined:<10.3f} {is_match:<8}")
        
        print("="*90 + "\n")
    
    def test_database(self):
        """Test and display database information."""
        print("\n" + "="*70)
        print("DATABASE TEST")
        print("="*70)
        
        db = self.matcher.database
        count = db.count()
        
        print(f"Database file: reference_poses.json")
        print(f"Total poses: {count}")
        
        if count == 0:
            print("[ERR] Database is empty!")
            print("Capture some poses first: python main.py --mode capture")
            return
        
        print("\nPoses in database:")
        print("-"*70)
        print(f"{'ID':<25} {'Label':<20} {'Keypoints':<12}")
        print("-"*70)
        
        for pose_id, pose_data in db.poses.items():
            label = pose_data['label']
            landmarks = pose_data['landmarks']
            print(f"{pose_id:<25} {label:<20} {landmarks.shape}")
        
        print("="*70 + "\n")
    
    def benchmark(self):
        """Benchmark system performance."""
        print("\n" + "="*70)
        print("PERFORMANCE BENCHMARK")
        print("="*70)
        print("Recording 60 frames for benchmark...")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERR] Cannot open webcam")
            return
        
        detection_times = []
        matching_times = []
        total_times = []
        
        frame_count = 0
        while frame_count < 60:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Benchmark detection
            t0 = time.time()
            landmarks, confidences, success = self.detector.detect(frame)
            det_time = (time.time() - t0) * 1000
            detection_times.append(det_time)
            
            # Benchmark matching
            t0 = time.time()
            if success and self.matcher.database.count() > 0:
                matched_pose, confidence = self.matcher.match_pose(landmarks, confidences)
            match_time = (time.time() - t0) * 1000
            matching_times.append(match_time)
            
            total_times.append(det_time + match_time)
            
            frame_count += 1
        
        cap.release()
        
        # Results
        print("\n" + "-"*70)
        print("RESULTS:")
        print("-"*70)
        
        print("\nDetection Performance:")
        print(f"  Min: {np.min(detection_times):.2f} ms")
        print(f"  Max: {np.max(detection_times):.2f} ms")
        print(f"  Mean: {np.mean(detection_times):.2f} ms")
        print(f"  Std: {np.std(detection_times):.2f} ms")
        
        print("\nMatching Performance:")
        print(f"  Min: {np.min(matching_times):.2f} ms")
        print(f"  Max: {np.max(matching_times):.2f} ms")
        print(f"  Mean: {np.mean(matching_times):.2f} ms")
        print(f"  Std: {np.std(matching_times):.2f} ms")
        
        print("\nTotal Pipeline:")
        print(f"  Min: {np.min(total_times):.2f} ms")
        print(f"  Max: {np.max(total_times):.2f} ms")
        print(f"  Mean: {np.mean(total_times):.2f} ms")
        print(f"  Average FPS: {1000/np.mean(total_times):.1f} fps")
        
        print("\n" + "="*70 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Pose Matching System Tester")
    parser.add_argument(
        "--test",
        choices=["detection", "matching", "database"],
        help="Run specific test"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmark"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=10,
        help="Test duration in seconds (for detection test)"
    )
    
    args = parser.parse_args()
    
    tester = SystemTester()
    
    if args.benchmark:
        tester.benchmark()
    elif args.test == "detection":
        tester.test_detection(args.duration)
    elif args.test == "matching":
        tester.test_matching()
    elif args.test == "database":
        tester.test_database()
    else:
        print("\nPose Matching System Tester")
        print("="*70)
        print("Tests available:")
        print("  python calibration.py --test detection    (Test pose detection)")
        print("  python calibration.py --test matching     (Test pose matching)")
        print("  python calibration.py --test database     (Check database)")
        print("  python calibration.py --benchmark         (Performance benchmark)")
        print("="*70 + "\n")


if __name__ == "__main__":
    main()

