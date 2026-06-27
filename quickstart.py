"""
Quick Start Script
==================
Automated setup and testing to get started immediately.

Run this to:
1. Check Python version
2. Install dependencies
3. Test all components
4. Guide first-time users

Usage:
    python quickstart.py
"""

import sys
import subprocess
import os
import json
from pathlib import Path


def print_header(text: str):
    """Print formatted header."""
    print("\n" + "="*70)
    print(text.center(70))
    print("="*70)


def print_step(step: int, text: str):
    """Print step indicator."""
    print(f"\n[{step}] {text}")
    print("-" * 70)


def check_python_version():
    """Check Python version."""
    print_step(1, "Checking Python Version")
    
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("[ERR] Python 3.8+ required")
        return False
    
    print("[OK] Python version OK")
    return True


def check_dependencies():
    """Check if required packages are installed."""
    print_step(2, "Checking Dependencies")
    
    packages = ['cv2', 'mediapipe', 'numpy']
    all_ok = True
    
    for package in packages:
        try:
            __import__(package)
            print(f"[OK] {package} is installed")
        except ImportError:
            print(f"[ERR] {package} is NOT installed")
            all_ok = False
    
    return all_ok


def install_dependencies():
    """Install dependencies from requirements.txt."""
    print_step(3, "Installing Dependencies")
    
    print("Installing packages from requirements.txt...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "-r", "requirements.txt"
        ])
        print("[OK] Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("[ERR] Failed to install dependencies")
        return False


def test_imports():
    """Test if all modules can be imported."""
    print_step(4, "Testing Module Imports")
    
    try:
        import cv2
        print("[OK] cv2 imported")
        
        import mediapipe as mp
        print("[OK] mediapipe imported")
        
        import numpy as np
        print("[OK] numpy imported")
        
        from pose_detector import PoseDetector
        print("[OK] PoseDetector imported")
        
        from pose_utils import PoseNormalizer
        print("[OK] PoseUtils imported")
        
        from matcher import PoseMatcher
        print("[OK] Matcher imported")
        
        from main import PoseMatchingApp
        print("[OK] Main app imported")
        
        print("\n[OK] All modules imported successfully!")
        return True
        
    except Exception as e:
        print(f"[ERR] Import error: {e}")
        return False


def test_webcam():
    """Test if webcam is accessible."""
    print_step(5, "Testing Webcam")
    
    try:
        import cv2
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERR] Cannot access webcam")
            print("\nSolutions:")
            print("  1. Check if camera is connected")
            print("  2. Try different camera ID: --camera 1")
            print("  3. Check Device Manager (Windows) or lsusb (Linux)")
            return False
        
        ret, frame = cap.read()
        if not ret:
            print("[ERR] Cannot read from webcam")
            cap.release()
            return False
        
        h, w = frame.shape[:2]
        print(f"[OK] Webcam OK ({w}x{h})")
        cap.release()
        return True
        
    except Exception as e:
        print(f"[ERR] Webcam error: {e}")
        return False


def test_pose_detection():
    """Test pose detection."""
    print_step(6, "Testing Pose Detection")
    
    try:
        import cv2
        from pose_detector import PoseDetector
        
        print("Initializing detector...")
        detector = PoseDetector(model_complexity=1)
        
        print("Capturing frame...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERR] Cannot open camera")
            return False
        
        ret, frame = cap.read()
        if not ret:
            print("[ERR] Cannot capture frame")
            cap.release()
            return False
        
        print("Detecting pose (this may take a few seconds)...")
        landmarks, confidences, success = detector.detect(frame)
        
        if success:
            valid_points = sum(1 for c in confidences if c > 0.5)
            print(f"[OK] Pose detected! ({valid_points}/33 keypoints)")
        else:
            print("⚠ No pose detected (this is OK if not in frame)")
        
        detector.release()
        cap.release()
        return True
        
    except Exception as e:
        print(f"[ERR] Detection error: {e}")
        return False


def check_database():
    """Check if reference database exists."""
    print_step(7, "Checking Reference Database")
    
    db_path = "reference_poses.json"
    
    if os.path.exists(db_path):
        try:
            with open(db_path, 'r') as f:
                data = json.load(f)
            count = len(data)
            print(f"[OK] Database found with {count} poses")
            return True
        except:
            print("⚠ Database file corrupted")
            return False
    else:
        print("[INFO] No database found (this is OK for first run)")
        print("  You can capture poses using: python main.py --mode capture")
        return True


def show_next_steps():
    """Show next steps for the user."""
    print_header("NEXT STEPS")
    
    db_path = "reference_poses.json"
    has_db = os.path.exists(db_path) and os.path.getsize(db_path) > 0
    
    if has_db:
        print("\n[OK] System is ready to use!")
        print("\n1. START LIVE MATCHING:")
        print("   python main.py --mode live")
        print("\n2. DURING LIVE MODE:")
        print("   - SPACE: Capture new pose")
        print("   - M: Show match details")
        print("   - +/-: Adjust threshold")
        print("   - Q: Quit")
    else:
        print("\n⚠ No reference poses found. Let's capture some!")
        print("\n1. CAPTURE REFERENCE POSES:")
        print("   python main.py --mode capture")
        print("\n   Steps:")
        print("   - Position yourself in front of camera")
        print("   - Wait for 'Pose detected!' message")
        print("   - Press SPACE to capture")
        print("   - Enter a label (e.g., 'standing', 'sitting')")
        print("   - Repeat for 5-10 different poses")
        print("   - Press Q to exit")
        print("\n2. THEN RUN LIVE MATCHING:")
        print("   python main.py --mode live")
    
    print("\n3. TEST SYSTEM COMPONENTS:")
    print("   python calibration.py --test detection")
    print("   python calibration.py --test matching")
    print("   python calibration.py --benchmark")
    
    print("\n4. READ DOCUMENTATION:")
    print("   README.md - Complete documentation")
    print("   SETUP_GUIDE.md - Installation and configuration")
    print("   ADVANCED_GUIDE.md - Advanced tips and tricks")


def main():
    """Run quickstart wizard."""
    
    print_header("POSE MATCHING SYSTEM - QUICK START")
    
    # Check Python version
    if not check_python_version():
        print("\n[ERR] System check failed. Please upgrade Python.")
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        print("\nAttempting to install dependencies...")
        if not install_dependencies():
            print("\n[ERR] Could not install dependencies")
            print("Please run: pip install -r requirements.txt")
            sys.exit(1)
    
    # Test imports
    if not test_imports():
        print("\n[ERR] Module import failed")
        sys.exit(1)
    
    # Test webcam
    if not test_webcam():
        print("\n⚠ Webcam test failed")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Test pose detection
    try:
        response = input("\nTest pose detection (requires ~5 seconds)? (y/n): ")
        if response.lower() == 'y':
            test_pose_detection()
    except KeyboardInterrupt:
        print("\nSkipped")
    
    # Check database
    check_database()
    
    # Show next steps
    show_next_steps()
    
    print_header("READY TO START!")
    print("\n[OK] System is ready. Good luck!\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERR] Unexpected error: {e}")
        sys.exit(1)

