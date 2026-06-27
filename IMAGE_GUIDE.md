# 📸 Reference Images Guide

## Quick Answer: Where to Put Pictures

**Location:** `datasets/` folder in your project directory

```
pose_detect/
└── datasets/              ← PUT YOUR IMAGES HERE
    ├── standing.jpg
    ├── sitting.jpg
    ├── arms_up.jpg
    └── ... (your images)
```

---

## 🎯 How It Works

### Step 1: Prepare Your Images

Place reference images in `datasets/`:
- `.jpg` format preferred
- Clear, well-lit photos of the pose
- Full body visible in frame
- Recommended resolution: 640x480 or higher

**Example:**
```
datasets/
├── person_standing.jpg       (full body standing pose)
├── person_sitting.jpg        (full body sitting pose)
├── person_arms_raised.jpg    (arms up pose)
└── person_dancing.jpg        (dancing pose)
```

### Step 2: Link Images to Poses

When capturing a reference pose, note the image path:

**Via UI (Capture Mode):**
```bash
python main.py --mode capture
```
- Position yourself in front of camera
- Press SPACE to capture
- Enter label: `standing`
- ✓ Pose is saved
- Manually edit `reference_poses.json` to add image path

**Via JSON Directly (Edit reference_poses.json):**
```json
{
  "standing_pose_001": {
    "image_path": "datasets/standing.jpg",     ← Add this line
    "label": "standing",
    "landmarks": [...],
    "angle_vector": [...],
    "pose_vector": [...],
    "angles": {...}
  }
}
```

**Via Python Script (Recommended):**
```python
from matcher import ReferencePoseDatabase
from pose_detector import PoseDetector
from pose_utils import FeatureVector
import cv2

detector = PoseDetector()
db = ReferencePoseDatabase("reference_poses.json")

# Load image and detect pose
image = cv2.imread("datasets/standing.jpg")
landmarks, confidences, success = detector.detect(image)

if success:
    features = FeatureVector.extract_features(landmarks, confidences)
    
    # Add pose with image path linked
    db.add_pose(
        pose_id="standing_pose_001",
        image_path="datasets/standing.jpg",     ← Link here
        label="standing",
        landmarks=landmarks,
        features=features
    )
    
    db.save_database()
    print("✓ Pose added with image")
```

### Step 3: Display During Matching

**During Live Matching:**
```bash
python main.py --mode live
```

1. Perform a pose that matches your reference
2. Wait for "MATCH FOUND!" message
3. Press **I** to view the reference image
4. Press any key to close the image

**Output:**
```
LIVE WEBCAM
[Your live pose with skeleton overlay]

MATCH FOUND!
Pose: standing
Confidence: 87%
← Press 'I' to see reference
```

---

## 📝 Complete Workflow Example

### Scenario: Create Standing Pose Reference

**Step 1: Take a photo**
```bash
# Take a picture of you standing
# Save as: datasets/standing_reference.jpg
```

**Step 2: Capture pose in system**
```bash
python main.py --mode capture
# Strike the standing pose
# Press SPACE
# Enter label: "standing"
```

**Step 3: Link image to pose**

Edit `reference_poses.json`:
```json
{
  "standing_12345": {
    "image_path": "datasets/standing_reference.jpg",
    "label": "standing",
    "landmarks": [...],
    "angle_vector": [...],
    "pose_vector": [...],
    "angles": {...}
  }
}
```

**Step 4: Test in live mode**
```bash
python main.py --mode live
# Stand in same pose
# Wait for "MATCH FOUND!"
# Press I
# ✓ Reference image displays
```

---

## 🎨 Image Best Practices

### Good Images ✓
- [ ] Full body visible
- [ ] Clear lighting
- [ ] High contrast (not too dark/bright)
- [ ] Person centered in frame
- [ ] Resolution 640x480 or higher
- [ ] JPG or PNG format
- [ ] No obstructions (clear background)

### Bad Images ✗
- [ ] Cropped body parts
- [ ] Very dark/backlit
- [ ] Person at edge of frame
- [ ] Low resolution (<320 pixels)
- [ ] Cluttered background
- [ ] Multiple people
- [ ] Shadows on body

### Example Good Setup
```
📷 Camera
   ↓ (2-3 meters away)
┌──────────────────┐
│     Person       │  ← Full body visible
│   Standing       │  ← Well-lit
│  Neutral BG      │  ← Clean background
└──────────────────┘
```

---

## 🔧 Troubleshooting

### Image Not Displaying

**Problem:** Press 'I' but nothing happens

**Solutions:**
1. Check image path in `reference_poses.json`
   ```json
   "image_path": "datasets/standing.jpg"  ← Must be correct
   ```

2. Verify file exists
   ```bash
   # In Windows PowerShell
   Test-Path "datasets/standing.jpg"
   ```

3. Check file format
   ```bash
   # Should be .jpg or .png
   dir datasets/
   ```

4. Try absolute path in JSON
   ```json
   "image_path": "d:/btech/projects/pose detect/datasets/standing.jpg"
   ```

### Image Path Errors

**"Image not found" error:**
```
✗ Reference image not found: datasets/standing.jpg
  Expected location: datasets/standing.jpg
```

**Fix:**
1. Ensure file exists in `datasets/` folder
2. Use forward slashes: `datasets/image.jpg` (not backslashes)
3. Check capitalization on Linux/Mac

### Multiple Images Per Pose

**Strategy:** Create separate pose IDs for different angles

```json
{
  "standing_front": {
    "image_path": "datasets/standing_front.jpg",
    "label": "standing"
    ...
  },
  "standing_side": {
    "image_path": "datasets/standing_side.jpg",
    "label": "standing"
    ...
  },
  "standing_back": {
    "image_path": "datasets/standing_back.jpg",
    "label": "standing"
    ...
  }
}
```

---

## 📊 Image Database Organization

### Recommended Folder Structure

```
pose_detect/
└── datasets/
    ├── standing/
    │   ├── front.jpg
    │   ├── side_left.jpg
    │   ├── side_right.jpg
    │   └── back.jpg
    ├── sitting/
    │   ├── front.jpg
    │   └── side.jpg
    ├── arms_up/
    │   └── front.jpg
    └── custom_poses/
        ├── pose1.jpg
        └── pose2.jpg
```

**Update JSON paths accordingly:**
```json
"image_path": "datasets/standing/front.jpg"
"image_path": "datasets/sitting/front.jpg"
```

---

## 🎬 Programmatic Image Handling

### Load and Display Image
```python
import cv2
from matcher import PoseMatcher

matcher = PoseMatcher("reference_poses.json")
matched_pose, confidence = matcher.match_pose(landmarks, confidences)

if matched_pose:
    image_path = matched_pose['image_path']
    image = cv2.imread(image_path)
    
    if image is not None:
        cv2.imshow("Reference Pose", image)
        cv2.waitKey(0)  # Wait for key press
    else:
        print(f"Cannot load image: {image_path}")
```

### Resize Image Before Display
```python
import cv2

image = cv2.imread("datasets/standing.jpg")

# Resize to fit window
height, width = image.shape[:2]
if height > 720 or width > 1280:
    scale = min(1280/width, 720/height)
    image = cv2.resize(image, None, fx=scale, fy=scale)

cv2.imshow("Reference Pose", image)
cv2.waitKey(0)
```

### Side-by-Side Comparison
```python
import cv2
import numpy as np

# Load reference image
ref_image = cv2.imread("datasets/standing.jpg")

# Resize to same height
h1, w1 = live_frame.shape[:2]
h2, w2 = ref_image.shape[:2]

scale = h1 / h2
ref_image = cv2.resize(ref_image, (int(w2*scale), h1))

# Concatenate horizontally
comparison = np.hstack([live_frame, ref_image])

cv2.imshow("Live vs Reference", comparison)
cv2.waitKey(0)
```

---

## 🔄 Batch Processing Images

### Extract Poses from Image Folder
```python
import os
import cv2
from pose_detector import PoseDetector
from pose_utils import FeatureVector
from matcher import ReferencePoseDatabase

detector = PoseDetector()
db = ReferencePoseDatabase("reference_poses.json")

# Process all images in datasets folder
for filename in os.listdir("datasets"):
    if filename.endswith((".jpg", ".png")):
        image_path = os.path.join("datasets", filename)
        image = cv2.imread(image_path)
        
        # Detect pose
        landmarks, confidences, success = detector.detect(image)
        
        if success:
            # Extract features
            features = FeatureVector.extract_features(landmarks, confidences)
            
            # Extract label from filename (e.g., "standing.jpg" → "standing")
            label = os.path.splitext(filename)[0]
            
            # Add to database
            db.add_pose(
                pose_id=f"{label}_{int(time.time())}",
                image_path=image_path,
                label=label,
                landmarks=landmarks,
                features=features
            )
            
            print(f"✓ Added: {label}")
        else:
            print(f"✗ Failed to detect pose in {filename}")

db.save_database()
print("✓ Batch processing complete")
```

---

## 🎓 Advanced: Custom Display Panel

### Modify Main Window to Show Images

Edit `main.py` to show reference image in a panel:

```python
def run_live_matching(self):
    # ... existing code ...
    
    current_match = None
    
    while True:
        ret, frame = cap.read()
        # ... pose detection ...
        
        match_result = ...
        
        # Show reference image in corner if match
        if match_result and match_result[0]:
            current_match = match_result[0]
            pose_data = self.matcher.database.get_pose(current_match)
            
            if pose_data['image_path']:
                ref_img = cv2.imread(pose_data['image_path'])
                
                # Resize and embed in corner
                ref_img = cv2.resize(ref_img, (200, 200))
                
                # Place in top-right corner
                frame[10:210, frame.shape[1]-210:frame.shape[1]-10] = ref_img
```

---

## 📞 Quick Reference

| Action | Command |
|--------|---------|
| Put images here | `datasets/` folder |
| View reference image | Press **I** during live mode |
| Edit image paths | Edit `reference_poses.json` |
| Test image loading | `python calibration.py --test database` |
| Add image to pose | Edit JSON or use Python script |
| Change image folder | Edit image_path in JSON to different folder |

---

## ✅ Checklist

- [ ] Created `datasets/` folder
- [ ] Added reference images to `datasets/`
- [ ] Captured poses with `python main.py --mode capture`
- [ ] Updated `reference_poses.json` with image paths
- [ ] Tested with `python main.py --mode live`
- [ ] Pressed **I** to view reference image
- [ ] Verified images display correctly
- [ ] Organized images into subfolders (optional)

---

## 🎉 Done!

Your pose matching system now displays reference images when poses match. Press **I** to view the reference pose image during live matching.

**Enjoy! 📸✨**
