# GesturSync

**A real-time, vision-based multimodal HCI system for touchless presentation and media control.**

GesturSync bridges the gap between human spatial intent and machine execution using Google's MediaPipe for 3D hand landmark extraction and a Linear Support Vector Machine (SVM) for gesture classification. The system runs on standard consumer hardware—no depth sensors or GPU required—achieving **98.8% classification accuracy** with **45–55 FPS** real-time performance on CPU.

> Based on the research paper: _GesturSync: A Multimodal HCI System Design and Usability Evaluation Using MediaPipe and SVM_

## Authors

| Name                | ID     | Email                |
| ------------------- | ------ | -------------------- |
| Abdul Rafay Javed   | BSSE23 | bsse23022@itu.edu.pk |
| Mustafa Fawwaz      | BSSE23 | bsse23036@itu.edu.pk |
| Hafiz M. Saad Irfan | BSSE23 | bsse23082@itu.edu.pk |

**Institution:** Information Technology University, Lahore, Pakistan

---

## Features

- **Touchless gesture control** — Navigate presentations and control media without a mouse or keyboard
- **Hybrid classification pipeline** — Linear SVM for discrete commands + heuristic distance mapping for continuous control
- **Five gesture classes** — Swipe Left, Swipe Right, Pinch Open, Pinch Close, Neutral Rest
- **Real-time performance** — 18–22 ms per frame on mid-range Intel Core i5 (no GPU)
- **Temporal debouncing** — Cooldown timers prevent accidental rapid-fire triggers ("Midas Touch")
- **Lightweight deployment** — Runs on any standard webcam and consumer CPU

### Supported Actions

| Gesture / Heuristic   | System Action        |
| --------------------- | -------------------- |
| Swipe Left            | Previous slide (`←`) |
| Swipe Right           | Next slide (`→`)     |
| Thumb → Pinky (close) | Mute volume          |
| Thumb → Index (close) | Volume down          |
| Thumb → Index (far)   | Volume up            |

---

## System Architecture

GesturSync follows a sequential 5-step pipeline:

```
Video Ingestion → Feature Extraction → Classification → Heuristic Logic → System Execution
   (OpenCV)         (MediaPipe 3D)        (SVM)         (Distance Map)      (PyAutoGUI)
```

1. **Video Ingestion** — Captures 640×480 RGB frames via OpenCV webcam
2. **Feature Extraction** — MediaPipe Hands extracts 21 3D landmarks (63-dimensional feature vector per frame)
3. **Classification** — Linear SVM (C=1.0) predicts one of five gesture classes
4. **Heuristic Logic** — Euclidean distance between thumb/index/pinky tips drives continuous volume control
5. **System Execution** — PyAutoGUI simulates keyboard shortcuts for OS-level control

---

## Tech Stack

| Component         | Library                   |
| ----------------- | ------------------------- |
| Video capture     | OpenCV                    |
| Hand tracking     | MediaPipe Hands           |
| Classification    | scikit-learn (Linear SVM) |
| System control    | PyAutoGUI                 |
| Data handling     | pandas, NumPy             |
| Model persistence | joblib                    |
| Visualization     | matplotlib                |

---

## Requirements

- Python 3.11+
- Webcam
- Windows / macOS / Linux

---

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Clone the repository
git clone https://github.com/your-username/GesturSync_new.git
cd GesturSync_new

# Install dependencies
uv sync

# Install additional runtime dependencies not in pyproject.toml
uv pip install pyautogui matplotlib
```

Alternatively, with pip:

```bash
pip install mediapipe opencv-python scikit-learn pandas joblib pyautogui matplotlib
```

---

## Usage

GesturSync is used in three stages: collect training data, train the model, then run live inference.

### 1. Collect Gesture Data

```bash
uv run python data_collector.py
```

Hold your hand in each gesture and press the corresponding number key to record samples:

| Key | Gesture      |
| --- | ------------ |
| `0` | Swipe Left   |
| `1` | Swipe Right  |
| `2` | Pinch Open   |
| `3` | Pinch Close  |
| `4` | Neutral Rest |

Samples are saved to `hand_gestures.csv`. The paper uses **1,500 samples** (300 per class) with varied hand depth, wrist tilt, and spatial position.

Press `q` to quit.

### 2. Train the SVM Model

```bash
uv run python train_svm.py
```

This script:

- Loads `hand_gestures.csv`
- Performs an **80/20 train-test split** (random state 42)
- Trains a Linear SVM (`kernel='linear'`, `C=1.0`)
- Prints accuracy, precision, recall, and a classification report
- Saves the model to `gestursync_svm_model.pkl`

### 3. Run Live Gesture Control

```bash
uv run python gestursync_live.py
```

Point your webcam at your hand. The live window shows the hand skeleton, the AI's predicted gesture, and triggered actions. Press `q` to quit.

> **Note:** `gestursync_svm_model.pkl` must exist before running the live demo. Train the model first, or use a pre-trained checkpoint if provided.

---

## Project Structure

```
GesturSync_new/
├── data_collector.py          # Webcam-based gesture data collection
├── train_svm.py               # SVM training and evaluation
├── gestursync_live.py         # Real-time gesture recognition and system control
├── generate_architecture.py   # Generates architecture.png for documentation
├── generate_graph.py          # Generates accuracy_graph.png
├── generate_matrix.py         # Generates confusion_matrix.png
├── hand_gestures.csv          # Collected training data (generated)
├── gestursync_svm_model.pkl   # Trained model (generated)
├── gesture.mp4                # Demo video
├── gestursync_short.mp4       # Short demo clip
├── GesturSync.pdf             # Research paper
├── pyproject.toml
└── uv.lock
```

---

## Model Performance

Evaluated on a held-out 20% test set (300 samples from 1,500 total):

| Metric        | Score  |
| ------------- | ------ |
| **Accuracy**  | 98.80% |
| **Precision** | 98.86% |
| **Recall**    | 98.80% |

The Swipe Right and Pinch configurations achieved perfect precision (1.00). Minor confusion between Swipe Left and Neutral was observed due to overlapping transitional wrist angles.

Run the visualization scripts to regenerate paper figures:

```bash
uv run python generate_architecture.py   # → architecture.png
uv run python generate_graph.py          # → accuracy_graph.png
uv run python generate_matrix.py         # → confusion_matrix.png
```

---

## Usability Evaluation

Five participants (ages 20–25) completed two tasks after a one-minute tutorial:

1. **Discrete control** — Navigate a 10-slide PDF using swipe gestures
2. **Continuous control** — Adjust system volume to zero and back to maximum using pinch distance

| Evaluation Criteria                 | Avg Rating (out of 5) |
| ----------------------------------- | --------------------- |
| Intuitiveness of Swiping            | 4.8                   |
| Responsiveness (Latency Perception) | 4.7                   |
| Physical Fatigue after 5 minutes    | 3.4                   |
| Overall System Reliability          | 4.5                   |

---

## Known Limitations

- **Backlighting** — Bright windows behind the user can disrupt MediaPipe palm detection
- **Motion blur** — Very rapid hand movements cause landmark distortion and dropped frames
- **Gorilla Arm** — Prolonged mid-air use may cause mild physical discomfort (common in touchless interfaces)

---

## Future Work

- Integrate NLP for concurrent voice commands (e.g., "Search the web for…")
- Apply Kalman filters to smooth landmark coordinates and reduce motion-blur artifacts
- Expand gesture vocabulary for richer spatial navigation

---

## Comparison with Related Approaches

| Approach            | Method                     | Accuracy  | Limitation                             |
| ------------------- | -------------------------- | --------- | -------------------------------------- |
| Oudah et al. (2020) | CNN on raw RGB             | 92.4%     | Requires GPU for real-time             |
| Kumar et al. (2021) | Kinect + SVM               | 96.1%     | Expensive depth hardware               |
| Zhang et al. (2022) | MediaPipe + DTW            | 91.5%     | High temporal matching latency         |
| **GesturSync**      | **MediaPipe + Linear SVM** | **98.8%** | **Lightweight, CPU-only, low latency** |

---

## License

This project was developed as academic research at Information Technology University. Contact the authors for licensing inquiries.

---

## References

See [GesturSync.pdf](GesturSync.pdf) for the full bibliography and detailed methodology.
