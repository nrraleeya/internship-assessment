# Computer Vision Project – Highway Car Counter

## 1. Project Overview
This sub-module contains a high-performance computer vision pipeline designed to detect, track, and count vehicles passing through a segment of a highway using video footage. 

Unlike basic frame-differencing techniques that fail under shifting shadows or light conditions, this solution pairs deep learning object detection with real-time multi-object tracking. To accommodate the camera's perspective and varying traffic speeds across near and far lanes, a two-dimensional spatial counting zone was engineered to ensure precise, non-duplicate tallies.

---

## 2. Technical Stack & Architecture
* **Python 3.13** - Primary execution environment.
* **OpenCV (cv2)** - Used for video I/O stream operations, rendering bounding boxes, dynamic frame color blends, and HUD dashboard overlays.
* **YOLOv8 (Ultralytics)** - Leveraged a pre-trained convolutional neural network model (`yolov8n.pt`) for fast, precise object detection and tracking.

### System Workflow
1. **Stream Ingestion:** Frames are loaded sequentially from the source video file while extracting metadata properties (FPS, dimensions).
2. **Object Tracking:** The YOLOv8 tracking architecture tracks individual vehicles frame-by-frame. Detections are strictly filtered to target indices (`2: car`, `5: bus`, `7: truck`) to ignore background elements like trees or signs.
3. **Spatial Zone Filtering:** Instead of a razor-thin geometric line where fast vehicles can go through across frames undetected, the system checks whether the tracking centroid falls within a 2D vertical bounding region covering the highway road surface.
4. **Deduplication:** Validated vehicle IDs are logged into a Python `set()`. The structure inherently handles deduplication, ensuring every vehicle is counted exactly once.
5. **HUD Rendering:** The system draws the active zone mask, visual tracking boundaries, and a continuous counting dashboard directly onto the frame before compiling the final video export.

---

## 3. Core Features & Edge-Case Optimizations
* **Unified UI Theme:** Removed distracting multicolored trails and complex multi-class labels. All valid targets receive a clean, unified green bounding box labeled strictly with their unique index (`Vehicle ID: X`).
* **Bidirectional Lane Coverage:** Dynamically calculates a broad spatial region ($45\%$ to $65\%$ of frame height) to perfectly capture both the distant, smaller vehicles in the far lane and larger, fast-moving vehicles in the near lane.
* **Zero-Dependency Core:** Implemented directly via the raw model tracking outputs rather than restrictive high-level wrapper APIs, ensuring seamless stability across different library updates.

---

## 4. Execution Instructions

### Prerequisites
Make sure your Python environment has the required computer vision dependencies installed:
```bash
pip install ultralytics opencv-python --user
