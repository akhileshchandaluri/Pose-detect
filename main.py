"""
Main Application Module
=======================
Real-time pose detection and matching using webcam.

Features:
- Live webcam capture
- Real-time pose detection
- Pose matching with visual feedback
- Countdown before capture
- Sound and visual notifications
- Display matched reference image

Usage:
    python main.py --mode live          # Start live matching
    python main.py --mode capture       # Capture new reference pose
    python main.py --mode calibrate     # Calibrate matching thresholds
"""

import cv2
import numpy as np
import argparse
import json
import os
import subprocess
import sys
import threading
from datetime import datetime
from typing import Tuple, Optional
import time

from pose_detector import PoseDetector
from pose_utils import PoseNormalizer, FeatureVector
from matcher import PoseMatcher, MatchTracker, ReferencePoseDatabase


class LaughDetector:
    """Detect a laugh/smile using MediaPipe Face Mesh (Mouth Aspect Ratio)."""

    def __init__(self, threshold: float = 0.22):
        self.threshold = threshold
        import mediapipe as mp
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def detect(self, frame: np.ndarray) -> Tuple[bool, float]:
        import math
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            return False, 0.0

        landmarks = results.multi_face_landmarks[0].landmark

        # Inner lip indices: top=13, bottom=14, left_corner=78, right_corner=308
        top_lip = landmarks[13]
        bottom_lip = landmarks[14]
        left_mouth = landmarks[78]
        right_mouth = landmarks[308]

        # Calculate vertical and horizontal distances
        vert_dist = math.hypot(top_lip.x - bottom_lip.x, top_lip.y - bottom_lip.y)
        horiz_dist = math.hypot(left_mouth.x - right_mouth.x, left_mouth.y - right_mouth.y)

        if horiz_dist == 0:
            return False, 0.0

        mar = vert_dist / horiz_dist
        score = float(mar)

        return score >= self.threshold, score

    def release(self):
        self.face_mesh.close()


class PoseMatchingApp:
    """Main application for pose detection and matching."""
    
    def __init__(self, reference_db: str = "reference_poses.json",
                 match_threshold: float = 0.72,
                 camera_id: int = 0,
                 laugh_image: str = "",
                 laugh_sound: str = "",
                 laugh_video: str = "",
                 laugh_threshold: float = 0.22,
                 laugh_hold_frames: int = 3,
                 laugh_cooldown: float = 8.0):
        """
        Initialize the application.
        
        Args:
            reference_db (str): Path to reference pose database
            match_threshold (float): Confidence threshold for matches
            camera_id (int): Camera device ID
        """
        self.detector = None
        self.pose_detection_enabled = True
        try:
            self.detector = PoseDetector(model_complexity=1, min_detection_confidence=0.5)
        except Exception as exc:
            self.pose_detection_enabled = False
            print(f"[WARN] Pose detection disabled: {exc}")
            print("[WARN] Laugh video trigger will still work.")

        self.matcher = PoseMatcher(
            db_path=reference_db,
            match_threshold=match_threshold
        )
        self.tracker = MatchTracker(history_size=6, stability_threshold=match_threshold)
        
        self.camera = cv2.VideoCapture(camera_id)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        
        self.fps = 0
        self.frame_count = 0
        self.last_time = time.time()

        self.laugh_image_path = self.find_media_path(
            laugh_image,
            ["datasets/laugh.jpg", "datasets/laugh.png", "datasets/laugh.jpeg", "datasets/laugh.webp"]
        )
        self.laugh_sound_path = self.find_media_path(
            laugh_sound,
            ["datasets/laugh.wav", "datasets/laugh.mp3"]
        )
        self.laugh_video_path = self.find_video_path(
            laugh_video,
            ["datasets/laugh.mp4", "datasets/laugh.mov", "datasets/laugh.avi", "datasets/laugh.mkv"]
        )
        self.laugh_enabled = any([self.laugh_image_path, self.laugh_sound_path, self.laugh_video_path])
        self.laugh_detector = LaughDetector(threshold=laugh_threshold) if self.laugh_enabled else None
        self.laugh_hold_frames = max(1, laugh_hold_frames)
        self.laugh_cooldown = max(1.0, laugh_cooldown)
        self.laugh_frame_count = 0
        self.laugh_was_active = False
        self.last_laugh_trigger_time = 0.0
        self.laugh_window_title = "Laugh Trigger"
        self.laugh_image_visible = False
        self.laugh_image_expires_at = 0.0
        self.cached_laugh_image = None
        
        # Video playback state
        self.video_cap = None
        self.audio_process = None
        self.video_window_title = "Laugh Video"
        self.is_video_playing = False
        self.video_frame_index = 0
        
        # Load manual pose-to-image mapping
        self.pose_image_mapping = {}
        try:
            with open("pose_image_mapping.json", "r") as f:
                mapping_data = json.load(f)
                self.pose_image_mapping = mapping_data.get("pose_mapping", {})
        except:
            pass  # No manual mapping file
        
        # UI state
        self.show_stats = True
        self.show_skeleton = True
        self.show_angles = False
        self.paused = False
        
        # Auto-load dataset images
        self.auto_load_dataset_images()

        if self.laugh_enabled:
            print("\nLaugh trigger enabled")
            if self.laugh_video_path:
                print(f"  Video: {self.laugh_video_path}")
            if self.laugh_image_path:
                print(f"  Image: {self.laugh_image_path}")
            if self.laugh_sound_path:
                print(f"  Sound: {self.laugh_sound_path}")
    
    def get_fps(self) -> float:
        """Calculate and return FPS."""
        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.last_time
        
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_time = current_time
        
        return self.fps

    def find_media_path(self, explicit_path: str, default_paths: list) -> str:
        """Return a configured media path, or a default file if it exists."""
        if explicit_path:
            if os.path.exists(explicit_path):
                return explicit_path
            print(f"[WARN] Media file not found: {explicit_path}")
            return ""

        for path in default_paths:
            if os.path.exists(path):
                return path

        return ""

    def find_video_path(self, explicit_path: str, default_paths: list) -> str:
        """Return the configured video, or automatically use a video from datasets/."""
        video_path = self.find_media_path(explicit_path, default_paths)
        if video_path:
            return video_path

        datasets_dir = "datasets"
        video_extensions = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".wmv")
        if not os.path.isdir(datasets_dir):
            return ""

        video_files = sorted(
            file_name for file_name in os.listdir(datasets_dir)
            if file_name.lower().endswith(video_extensions)
        )
        if not video_files:
            return ""

        return os.path.join(datasets_dir, video_files[0])
    
    def auto_load_dataset_images(self):
        """Auto-load and extract poses from images in datasets/ folder."""
        if not self.pose_detection_enabled:
            return

        # Only auto-load if database is empty
        if self.matcher.database.count() > 0:
            return
        
        datasets_dir = "datasets"
        
        # Create datasets directory if it doesn't exist
        if not os.path.exists(datasets_dir):
            os.makedirs(datasets_dir)
            print(f"Created {datasets_dir}/ folder")
            return
        
        # Scan for image files
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        image_files = [
            f for f in os.listdir(datasets_dir)
            if f.lower().endswith(image_extensions)
        ]
        
        if not image_files:
            print(f"No images found in {datasets_dir}/")
            return
        
        print(f"\nAuto-loading {len(image_files)} image(s) from {datasets_dir}/...")
        
        # Use SAME detector settings as live mode for consistency
        lenient_detector = PoseDetector(model_complexity=1, min_detection_confidence=0.5)
        loaded_count = 0
        
        for image_file in image_files:
            image_path = os.path.join(datasets_dir, image_file)
            
            try:
                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    # Try loading with PIL for .webp
                    try:
                        from PIL import Image
                        pil_image = Image.open(image_path)
                        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                    except:
                        print(f"  [ERR] Failed to load: {image_file}")
                        continue
                
                if image is None:
                    print(f"  [ERR] Failed to load: {image_file}")
                    continue
                
                # Detect pose in image using lenient detector
                landmarks, confidences, success = lenient_detector.detect(image)
                
                if not success or landmarks is None:
                    print(f"  [ERR] No pose detected in: {image_file}")
                    continue
                
                # Extract features
                features = FeatureVector.extract_features(landmarks, confidences)
                if features is None:
                    print(f"  [ERR] Failed to extract features from: {image_file}")
                    continue
                
                # Normalize pose
                normalized_landmarks, _ = PoseNormalizer.normalize_pose(landmarks, confidences)
                
                # Generate ID from filename (without extension)
                base_name = os.path.splitext(image_file)[0]
                pose_id = f"dataset_{base_name}_{int(time.time() * 1000) % 10000}"
                label = base_name.replace('_', ' ').title()
                
                # Add to database
                self.matcher.database.add_pose(
                    pose_id=pose_id,
                    image_path=image_path,
                    label=label,
                    landmarks=normalized_landmarks,
                    features=features
                )
                
                print(f"  [OK] Loaded: {image_file} -> {label}")
                loaded_count += 1
                
            except Exception as e:
                print(f"  [ERR] Error loading {image_file}: {str(e)}")
        
        # Clean up lenient detector
        lenient_detector.release()
        
        if loaded_count > 0:
            self.matcher.database.save_database()
            print(f"\n[OK] Auto-loaded {loaded_count} pose(s) from datasets/\n")
        else:
            print(f"[ERR] No poses were successfully loaded from datasets/\n")
    
    def get_fps(self) -> float:
        """Calculate and return FPS."""
        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.last_time
        
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_time = current_time
        
        return self.fps
    
    def draw_info_panel(self, frame: np.ndarray, pose_landmarks: np.ndarray = None,
                       pose_confidences: np.ndarray = None,
                       match_result: Optional[Tuple] = None,
                       detailed_scores: dict = None) -> np.ndarray:
        """
        Draw information panel on frame.
        
        Args:
            frame (np.ndarray): Video frame
            pose_landmarks (np.ndarray): Detected pose keypoints
            pose_confidences (np.ndarray): Confidence scores
            match_result (Tuple): (pose_id, confidence) or None
            
        Returns:
            np.ndarray: Frame with info panel
        """
        h, w = frame.shape[:2]
        
        # Semi-transparent background for stats
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # FPS
        fps = self.get_fps()
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Pose detection status
        if pose_landmarks is not None:
            cv2.putText(frame, "[OK] Pose Detected", (20, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Valid keypoints count
            valid_count = np.sum(pose_confidences > 0.5)
            cv2.putText(frame, f"Keypoints: {valid_count}/33", (20, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        else:
            cv2.putText(frame, "[ERR] No Pose", (20, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Database info
        db_count = self.matcher.database.count()
        cv2.putText(frame, f"Ref Poses: {db_count}", (20, 145),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 1)
        
        # Match info - show on screen when matched
        if match_result is not None:
            pose_id, confidence = match_result
            if pose_id is not None:
                label = self.matcher.database.get_pose(pose_id).get('label', 'Unknown')
                cv2.putText(frame, f"MATCH: {label} ({confidence:.0%})", (20, 180),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)
        
        # Remove the scoring display - only show image
        
        # Instructions at bottom
        instruction_y = h - 20
        instructions = [
            "SPACE: Capture | M: Match | A: Angles | T: Stats | +/-: Threshold | Q: Quit"
        ]
        cv2.putText(frame, instructions[0], (10, instruction_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def draw_match_panel(self, frame: np.ndarray, match_result: Optional[Tuple]) -> np.ndarray:
        """Draw match result panel with reference image if available."""
        if match_result is None or match_result[0] is None:
            return frame
        
        pose_id, confidence = match_result
        pose_data = self.matcher.database.get_pose(pose_id)
        
        h, w = frame.shape[:2]
        
        # Draw colored border for match
        color = (0, 255, 0) if confidence > 0.75 else (0, 165, 255) if confidence > 0.65 else (255, 100, 0)
        cv2.rectangle(frame, (5, 5), (w-5, h-5), color, 4)
        
        # Draw match info panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - 350, h - 200), (w - 10, h - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        
        label = pose_data.get('label', 'Unknown')
        cv2.putText(frame, "MATCH FOUND!", (w - 340, h - 160),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Pose: {label}", (w - 340, h - 125),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 1)
        cv2.putText(frame, f"Confidence: {confidence:.1%}", (w - 340, h - 95),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)
        cv2.putText(frame, f"Threshold: {self.matcher.match_threshold:.2f}", (w - 340, h - 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
        cv2.putText(frame, "Image showing ->", (w - 340, h - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
        
        return frame
    
    def display_reference_image(self, match_result: Tuple):
        """Display the reference image for a matched pose."""
        if match_result is None or match_result[0] is None:
            return
        
        pose_id, confidence = match_result
        pose_data = self.matcher.database.get_pose(pose_id)
        
        if pose_data is None:
            return
        
        label = pose_data.get('label', '')
        
        # Check manual mapping first (normalized label)
        label_normalized = label.lower().replace(' ', '_')
        image_path = self.pose_image_mapping.get(label_normalized)
        
        # If no manual mapping, use auto-detected path
        if not image_path:
            image_path = pose_data.get('image_path', '')
        
        if not image_path or not os.path.exists(image_path):
            return
        
        try:
            # Load reference image (support .webp and other formats)
            ref_image = cv2.imread(image_path)
            
            # If imread fails (e.g., .webp), try with PIL
            if ref_image is None:
                try:
                    from PIL import Image
                    pil_image = Image.open(image_path)
                    ref_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                except:
                    return
            
            if ref_image is None:
                return
            
            label = pose_data.get('label', 'Reference Pose')
            window_title = f"Match: {label}"
            
            # Resize to reasonable size if too large
            h, w = ref_image.shape[:2]
            if h > 600 or w > 600:
                scale = min(600/h, 600/w)
                ref_image = cv2.resize(ref_image, None, fx=scale, fy=scale)
            
            # Add label at top of image
            cv2.putText(ref_image, f"{label} (Match: {confidence:.0%})", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Display in window with pose-specific title
            cv2.imshow(window_title, ref_image)
        except Exception as e:
            pass

    def load_image_file(self, image_path: str) -> Optional[np.ndarray]:
        """Load an image with OpenCV first, then PIL for formats like webp."""
        if not image_path or not os.path.exists(image_path):
            return None

        image = cv2.imread(image_path)
        if image is not None:
            return image

        try:
            from PIL import Image
            pil_image = Image.open(image_path)
            return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        except Exception:
            return None

    def open_with_default_player(self, media_path: str) -> bool:
        """Open media in the OS default app so videos can play with audio."""
        if not media_path or not os.path.exists(media_path):
            print(f"[ERR] Media file not found: {media_path}")
            return False

        abs_path = os.path.abspath(media_path)

        try:
            if os.name == "nt":
                os.startfile(abs_path)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", abs_path])
            else:
                subprocess.Popen(["xdg-open", abs_path])
            return True
        except Exception as exc:
            print(f"[ERR] Could not open media: {exc}")
            return False

    def play_laugh_sound(self):
        """Play a configured laugh sound once."""
        if not self.laugh_sound_path:
            return

        try:
            if os.name == "nt" and self.laugh_sound_path.lower().endswith(".wav"):
                import winsound
                winsound.PlaySound(
                    os.path.abspath(self.laugh_sound_path),
                    winsound.SND_FILENAME | winsound.SND_ASYNC
                )
            else:
                self.open_with_default_player(self.laugh_sound_path)
        except Exception as exc:
            print(f"[ERR] Could not play laugh sound: {exc}")

    def start_video_playback(self):
        """Start playing the laugh video in a small window with audio using ffplay."""
        if not self.laugh_video_path or not os.path.exists(self.laugh_video_path):
            return
        
        self.stop_video_playback()
        
        abs_path = os.path.abspath(self.laugh_video_path)
        
        try:
            import subprocess
            # Use ffplay to play video and audio together in a small window
            # -x 420 -y 320 limits the size
            # -loop 0 makes it loop while laughing
            cmd = ["ffplay", "-x", "420", "-y", "320", "-loop", "0", abs_path]
            
            self.audio_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            self.is_video_playing = True
            print(f"[OK] Playing video: {abs_path}")
            
        except Exception as exc:
            print(f"[ERR] Could not play video with ffplay: {exc}")
            self.is_video_playing = False

    def start_audio_playback(self, video_path: str):
        """Deprecated. Video and audio are now both handled by ffplay."""
        pass

    def stop_video_playback(self):
        """Stop video playback and close the video window."""
        if hasattr(self, 'audio_process') and self.audio_process is not None:
            try:
                self.audio_process.terminate()
                try:
                    self.audio_process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    self.audio_process.kill()
            except Exception:
                pass
            self.audio_process = None
        
        self.is_video_playing = False

    def display_video_frame(self):
        """Display the next frame of the video. Called in main loop."""
        if not self.is_video_playing or not hasattr(self, 'video_cap') or self.video_cap is None:
            return
        
        try:
            ret, frame = self.video_cap.read()
            
            if not ret:
                # Video ended, restart from beginning
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.video_cap.read()
                if not ret:
                    self.stop_video_playback()
                    return
            
            # Resize to small window (420x320)
            h, w = frame.shape[:2]
            aspect_ratio = w / h
            display_w = 420
            display_h = int(display_w / aspect_ratio)
            
            if display_h > 320:
                display_h = 320
                display_w = int(display_h * aspect_ratio)
            
            frame = cv2.resize(frame, (display_w, display_h))
            
            # Add border
            border_color = (0, 255, 0)
            cv2.rectangle(frame, (0, 0), (display_w - 1, display_h - 1), border_color, 3)
            
            # Display
            cv2.imshow(self.video_window_title, frame)
            
        except Exception as exc:
            print(f"[ERR] Error displaying video: {exc}")
            self.stop_video_playback()

    def display_laugh_image(self):
        """Show the configured laugh image for a few seconds."""
        if not self.laugh_image_path:
            return

        if self.cached_laugh_image is None:
            self.cached_laugh_image = self.load_image_file(self.laugh_image_path)

        if self.cached_laugh_image is None:
            print(f"[ERR] Could not load laugh image: {self.laugh_image_path}")
            return

        laugh_image = self.cached_laugh_image.copy()
        h, w = laugh_image.shape[:2]
        if h > 700 or w > 900:
            scale = min(700 / h, 900 / w)
            laugh_image = cv2.resize(laugh_image, None, fx=scale, fy=scale)

        cv2.putText(laugh_image, "LAUGH DETECTED", (20, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
        cv2.imshow(self.laugh_window_title, laugh_image)
        self.laugh_image_visible = True
        self.laugh_image_expires_at = time.time() + 4.0

    def close_expired_laugh_image(self):
        """Close the laugh image window after its display time expires."""
        if not self.laugh_image_visible:
            return

        if time.time() >= self.laugh_image_expires_at:
            try:
                cv2.destroyWindow(self.laugh_window_title)
            except Exception:
                pass
            self.laugh_image_visible = False

    def trigger_laugh_media(self):
        """Play the configured media for a laugh event."""
        if self.laugh_video_path:
            self.start_video_playback()
            return

        self.display_laugh_image()
        self.play_laugh_sound()

    def update_laugh_trigger(self, frame: np.ndarray) -> Tuple[bool, float]:
        """Detect laughter and fire configured media when stable."""
        if not self.laugh_detector:
            return False, 0.0

        is_laughing, laugh_score = self.laugh_detector.detect(frame)
        
        # Debug: Print laugh detection every 30 frames
        if self.frame_count % 30 == 0:
            print(f"[DEBUG] Laugh score: {laugh_score:.3f}, is_laughing: {is_laughing}, frame_count: {self.laugh_frame_count}/{self.laugh_hold_frames}")
        
        if is_laughing:
            self.laugh_frame_count += 1
            self.not_laughing_count = 0
        else:
            self.not_laughing_count = getattr(self, 'not_laughing_count', 0) + 1
            if self.not_laughing_count > 15:  # Grace period of 15 frames
                self.laugh_frame_count = 0

        now = time.time()
        is_stable_laugh = self.laugh_frame_count >= self.laugh_hold_frames
        can_trigger = (now - self.last_laugh_trigger_time) >= self.laugh_cooldown

        if is_stable_laugh and not self.laugh_was_active and can_trigger:
            print(f"[INFO] Laugh triggered! Playing video...")
            self.trigger_laugh_media()
            self.last_laugh_trigger_time = now
        
        # Stop video when user stops laughing
        if not is_stable_laugh and self.laugh_was_active and self.is_video_playing:
            print(f"[INFO] Laugh ended. Stopping video...")
            self.stop_video_playback()

        self.laugh_was_active = is_stable_laugh

        return is_laughing, laugh_score

    def draw_laugh_status(self, frame: np.ndarray, is_laughing: bool, laugh_score: float) -> np.ndarray:
        """Draw laugh detection state on the live camera frame."""
        if not self.laugh_enabled:
            return frame

        h, w = frame.shape[:2]
        status = "LAUGH DETECTED" if is_laughing else "Laugh trigger ready"
        color = (0, 255, 255) if is_laughing else (180, 180, 180)

        cv2.putText(frame, f"{status} ({laugh_score:.2f})", (20, h - 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        return frame
    
    def run_live_matching(self):
        """Run real-time pose matching."""
        print("\n" + "="*60)
        print("REAL-TIME POSE MATCHING")
        print("="*60)
        print(f"Loaded {self.matcher.database.count()} reference poses")
        print("\nControls:")
        print("  SPACE  - Capture current pose")
        print("  M      - Show detailed match scores")
        print("  A      - Toggle angle visualization")
        print("  T      - Toggle stats")
        print("  +/-    - Adjust threshold")
        print("  Q      - Quit")
        if self.laugh_enabled:
            print("  Laugh  - Auto-play configured laugh media")
        print("="*60 + "\n")
        
        last_displayed_pose_id = None
        last_window_title = None
        debug_counter = 0
        
        while True:
            ret, frame = self.camera.read()
            if not ret:
                print("[ERR] Failed to read from camera")
                break
            
            try:
                frame = cv2.flip(frame, 1)  # Mirror effect
                
                # Detect pose
                landmarks, confidences, success = None, None, False
                if self.detector is not None:
                    landmarks, confidences, success = self.detector.detect(frame)
                laugh_detected, laugh_score = self.update_laugh_trigger(frame)
                
                match_result = None
                detailed_scores = None
                if success:
                    # Match pose - get detailed scores to see what's happening
                    detailed_scores = self.matcher.match_pose_detailed(landmarks, confidences)
                    
                    # Get best match, then require both matcher confidence and short-term stability.
                    best_match = None
                    best_score = 0.0
                    
                    if detailed_scores:
                        sorted_scores = sorted(
                            detailed_scores.items(),
                            key=lambda x: x[1]['combined_score'],
                            reverse=True
                        )
                        pose_id, scores = sorted_scores[0]
                        best_score = scores['combined_score']
                        best_match = {
                            'pose_id': pose_id,
                            'combined_score': best_score,
                            'label': scores['label'],
                            'is_match': scores.get('is_match', False),
                            'margin': scores.get('margin', 0.0)
                        }
                    
                    # DEBUG: Print matching info every 30 frames (COMMENTED OUT - use terminal output if needed)
                    debug_counter += 1
                    # if debug_counter % 30 == 0 and detailed_scores:
                    #     # Show top 3 matches
                    #     sorted_scores = sorted(detailed_scores.items(), key=lambda x: x[1]['combined_score'], reverse=True)
                    #     print(f"\n[DEBUG] Top 3 matches:")
                    #     for i, (pid, scores) in enumerate(sorted_scores[:3], 1):
                    #         print(f"  {i}. {scores['label']}: {scores['combined_score']:.3f} (Euclidean: {scores['euclidean_distance']:.3f}, Angle: {scores['angle_difference']:.1f} deg)")
                    #     print(f"  Threshold: {self.matcher.match_threshold:.2f}")
                    
                    candidate_pose_id = best_match['pose_id'] if best_match and best_match.get('is_match') else None
                    candidate_score = best_score if candidate_pose_id else 0.0
                    self.tracker.update(candidate_pose_id, candidate_score)
                    stable_pose_id, stable_score = self.tracker.get_stable_match()
                    if stable_pose_id is not None:
                        match_result = (stable_pose_id, stable_score)
                    
                    # Draw skeleton
                    if self.show_skeleton:
                        frame = self.detector.draw_pose(frame, landmarks, confidences)
                
                # Draw UI
                frame = self.draw_info_panel(frame, landmarks, confidences, match_result, detailed_scores)
                frame = self.draw_laugh_status(frame, laugh_detected, laugh_score)
                if match_result:
                    frame = self.draw_match_panel(frame, match_result)
                    
                    # Auto-display reference image when match found
                    pose_id = match_result[0]
                    if pose_id != last_displayed_pose_id:
                        # Close previous window if different pose
                        if last_window_title is not None:
                            try:
                                cv2.destroyWindow(last_window_title)
                            except Exception:
                                pass
                        
                        # Display new image and update window title
                        pose_data = self.matcher.database.get_pose(pose_id)
                        if pose_data:
                            label = pose_data.get('label', 'Reference Pose')
                            last_window_title = f"Match: {label}"
                        
                        self.display_reference_image(match_result)
                        last_displayed_pose_id = pose_id
                else:
                    # Close image window if no match
                    if last_window_title is not None:
                        try:
                            cv2.destroyWindow(last_window_title)
                        except Exception:
                            pass
                        last_window_title = None
                    last_displayed_pose_id = None
                self.close_expired_laugh_image()
                
                # Display video frame if playing
                self.display_video_frame()
                
                # Display
                cv2.imshow("Pose Matching - Live", frame)
                
                # Handle input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):  # Space - capture pose
                    self.capture_pose(landmarks, confidences)
                elif key == ord('m'):  # M - show detailed scores
                    self.show_detailed_scores(landmarks, confidences)
                elif key == ord('a'):  # A - toggle angles
                    self.show_angles = not self.show_angles
                    print(f"Angle visualization: {'ON' if self.show_angles else 'OFF'}")
                elif key == ord('t'):  # T - toggle stats
                    self.show_stats = not self.show_stats
                elif key == ord('+') or key == ord('='):
                    self.matcher.set_threshold(self.matcher.match_threshold + 0.05)
                    self.tracker.stability_threshold = self.matcher.match_threshold
                    print(f"Threshold: {self.matcher.match_threshold:.2f}")
                elif key == ord('-') or key == ord('_'):
                    self.matcher.set_threshold(self.matcher.match_threshold - 0.05)
                    self.tracker.stability_threshold = self.matcher.match_threshold
                    print(f"Threshold: {self.matcher.match_threshold:.2f}")
                
            except Exception as exc:
                print(f"[ERR] Error in main loop: {exc}")
                import traceback
                traceback.print_exc()
    
    def capture_pose(self, landmarks: np.ndarray, confidences: np.ndarray):
        """Capture and save current pose as reference."""
        if landmarks is None:
            print("[ERR] No pose detected!")
            return
        
        # Get label from user
        label = input("\nEnter pose label (e.g., 'standing', 'sitting'): ").strip()
        if not label:
            print("[ERR] Cancelled")
            return
        
        # Extract features
        features = FeatureVector.extract_features(landmarks, confidences)
        if features is None:
            print("[ERR] Failed to extract features")
            return
        
        # Normalize pose
        normalized_landmarks, _ = PoseNormalizer.normalize_pose(landmarks, confidences)
        
        # Generate ID
        pose_id = f"{label}_{int(time.time())}"
        
        # Save to database
        self.matcher.database.add_pose(
            pose_id=pose_id,
            image_path="",  # Would be filled if saving images
            label=label,
            landmarks=normalized_landmarks,
            features=features
        )
        
        self.matcher.database.save_database()
        print(f"[OK] Captured pose: {label}")
    
    def show_detailed_scores(self, landmarks: np.ndarray, confidences: np.ndarray):
        """Show detailed matching scores for all poses."""
        if landmarks is None:
            print("[ERR] No pose detected!")
            return
        
        results = self.matcher.match_pose_detailed(landmarks, confidences)
        
        print("\n" + "="*80)
        print("DETAILED MATCH SCORES")
        print("="*80)
        
        if not results:
            print("No reference poses to compare")
            return
        
        # Sort by combined score
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1]['combined_score'],
            reverse=True
        )
        
        print(f"{'Rank':<6} {'Label':<20} {'Euclidean':<12} {'Angle Diff':<12} {'Cosine':<10} {'Score':<10} {'Margin':<10} {'Match':<8}")
        print("-"*80)
        
        for rank, (pose_id, scores) in enumerate(sorted_results[:10], 1):
            label = scores['label'][:19]
            euclidean = scores['euclidean_distance']
            angle_diff = scores['angle_difference']
            cosine = scores['cosine_similarity']
            combined = scores['combined_score']
            margin = scores.get('margin', 0.0)
            is_match = "[OK]" if scores['is_match'] else "[ERR]"
            
            print(f"{rank:<6} {label:<20} {euclidean:<12.4f} {angle_diff:<12.2f} deg {cosine:<10.3f} {combined:<10.3f} {margin:<10.3f} {is_match:<8}")
        
        print("="*80 + "\n")
    
    def run_capture_mode(self):
        """Capture new reference poses."""
        print("\n" + "="*60)
        print("CAPTURE MODE - Add New Reference Poses")
        print("="*60)
        print("Controls:")
        print("  SPACE  - Capture pose")
        print("  S      - Save and go to next")
        print("  Q      - Quit without saving")
        print("="*60 + "\n")
        
        while True:
            ret, frame = self.camera.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Detect pose
            landmarks, confidences, success = self.detector.detect(frame)
            
            if success:
                frame = self.detector.draw_pose(frame, landmarks, confidences)
                cv2.putText(frame, "Pose detected! Press SPACE to capture", (20, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No pose detected", (20, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow("Capture Mode", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # Space
                if success:
                    self.capture_pose(landmarks, confidences)
    
    def cleanup(self):
        """Clean up resources."""
        self.stop_video_playback()
        self.camera.release()
        if self.detector is not None:
            self.detector.release()
        if self.laugh_detector:
            self.laugh_detector.release()
        cv2.destroyAllWindows()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Real-time Pose Detection and Matching System"
    )
    parser.add_argument(
        "--mode",
        choices=["live", "capture"],
        default="live",
        help="Application mode: live matching or capture new poses"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.72,
        help="Match confidence threshold (0-1)"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device ID"
    )
    parser.add_argument(
        "--db",
        type=str,
        default="reference_poses.json",
        help="Path to reference pose database"
    )
    parser.add_argument(
        "--laugh-video",
        type=str,
        default="",
        help="Video to open when laughter is detected. Defaults to datasets/laugh.mp4 if present"
    )
    parser.add_argument(
        "--laugh-image",
        type=str,
        default="",
        help="Image to show when laughter is detected. Defaults to datasets/laugh.jpg/png if present"
    )
    parser.add_argument(
        "--laugh-sound",
        type=str,
        default="",
        help="Sound to play with the laugh image. Defaults to datasets/laugh.wav/mp3 if present"
    )
    parser.add_argument(
        "--laugh-threshold",
        type=float,
        default=0.22,
        help="Mouth-open threshold for laughter detection. Lower is more sensitive"
    )
    parser.add_argument(
        "--laugh-hold-frames",
        type=int,
        default=3,
        help="Number of consecutive laughing frames needed before media plays"
    )
    parser.add_argument(
        "--laugh-cooldown",
        type=float,
        default=8.0,
        help="Seconds to wait before the laugh media can trigger again"
    )
    
    args = parser.parse_args()
    
    # Create app
    app = PoseMatchingApp(
        reference_db=args.db,
        match_threshold=args.threshold,
        camera_id=args.camera,
        laugh_image=args.laugh_image,
        laugh_sound=args.laugh_sound,
        laugh_video=args.laugh_video,
        laugh_threshold=args.laugh_threshold,
        laugh_hold_frames=args.laugh_hold_frames,
        laugh_cooldown=args.laugh_cooldown
    )
    
    try:
        if args.mode == "live":
            app.run_live_matching()
        elif args.mode == "capture":
            app.run_capture_mode()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        app.cleanup()
        print("Application closed")


if __name__ == "__main__":
    main()

