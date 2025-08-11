# YOLOv8 Multi-Person Tracking with Movement Trails

This project uses **Ultralytics YOLOv8** to detect and track multiple people in a video while drawing **unique IDs** and **movement trails** for each tracked individual.  
It demonstrates object detection, object tracking, and real-time visualization using **OpenCV**.

## 📌 Features
- **YOLOv8 Nano Model** (`yolov8n.pt`) for lightweight and fast detection.
- Tracks only **persons** (COCO class `0`).
- Assigns **custom sequential IDs** for stable tracking even if YOLO internal IDs change.
- Filters out **false positives** by confirming detections after 5 consecutive frames.
- Draws:
  - Bounding box around each tracked person.
  - Unique ID label.
  - Movement trail of last 30 positions.
  - Center point marker for each person.
- Saves processed output as **MP4**.
- Supports real-time display with OpenCV.

## 🖼 Example Output
- Red bounding boxes indicate tracked persons.
- Trails visualize recent movement paths.
- IDs remain stable for each individual.

## 📂 Project Structure
```

.
├── data/people_walking.mp4             # Input video file
├── people_tracking.py                  # Main tracking script
├── yolov8n.pt                          # YOLOv8 Nano model weights
├── data/people_walking_output.mp4      # Output video (generated)
└── README.md

````

## 🚀 Installation

1. **Clone this repository**:
```bash
git clone https://github.com/di37/yolov8-person-tracker.git
cd yolov8-person-tracker
````

2. **Install dependencies**:

```bash
pip install ultralytics opencv-python
```

3. **Download YOLOv8 model weights**:

```bash
yolo download model=yolov8n.pt
```

Alternatively, you can manually place `yolov8n.pt` in the project directory.

## ▶️ Usage

Run the script with:

```bash
python people_tracking.py
```

* Press **`q`** to quit early.
* The processed video will be saved as `people_with_trail_output.mp4`.

## ⚙️ Key Parameters

* `classes=[0]`: Detect only persons.
* `maxlen=30`: Number of recent points stored for trails.
* `appear[oid] >= 5`: Minimum frames before confirming a person.
* `fps`: Dynamically detected from input video; defaults to 30.

## 🧠 How It Works

1. **Detection & Tracking**: YOLOv8 detects persons and assigns tracking IDs (`model.track` with `persist=True`).
2. **Custom ID Mapping**: Internal YOLO IDs are mapped to our own stable sequential IDs.
3. **Trail Drawing**: A `deque` stores recent positions per ID; OpenCV draws connecting lines.
4. **False Positive Filtering**: Only objects appearing for 5+ consecutive frames get a stable ID.

# Reference

- Computer Vision Bootcamp from Vizuara - Dr.Sreedath Panat

## 🛠 Dependencies

* [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
* [OpenCV](https://opencv.org/)
* Python 3.10+

## 📜 License

This project is licensed under the MIT License — you are free to use, modify, and distribute it.

---

**Author:** Your Name
**GitHub:** [@di37](https://github.com/di37)