# ADAS Pilot

A real-time **Advanced Driver Assistance System** built with a hybrid **Rust + Python** architecture. Performance-critical perception (lane detection, object tracking, traffic-light classification, ONNX inference) runs in compiled Rust via [PyO3](https://pyo3.rs) bindings, while the application layer orchestrates everything in Python with OpenCV and Ultralytics YOLO.

---

## Features

| Subsystem | Implementation | Description |
|---|---|---|
| **Lane Detection** | Rust (`lane_detect.rs`) | Canny edge detection → Hough transform with trapezoidal ROI masking. Returns left/right lane lines. |
| **Lane Management** | Rust (`lane_manager.rs`) | Exponential smoothing of lane lines across frames, left/right grouping, and ego-lane object filtering. Supports highway and two-way road modes. |
| **Object Detection** | Python (Ultralytics YOLOv8) | GPU-accelerated vehicle detection (cars, motorcycles, buses, trucks) via PyTorch CUDA. |
| **Object Tracking & TTC** | Rust (`object_proc.rs`) | Centroid-based frame-to-frame tracker. Estimates distance (pinhole camera model), relative speed, and time-to-collision (TTC). |
| **Traffic Light Detection** | Rust (`traffic_light.rs`) | HSV colour analysis on the upper third of each frame. Classifies as RED / YELLOW / GREEN / NONE. |
| **Traffic Sign Recognition** | Rust (`lib.rs` — `AdasBrain`) | Loads a custom ONNX model, runs YOLOv8-style inference entirely in Rust via the `ort` runtime, with NMS post-processing. |
| **Multi-threaded Camera** | Python (`main.py`) | Dedicated background thread for video decoding and resizing, keeping the main inference loop unblocked. |
| **Real-time HUD** | Python (`main.py`) | OpenCV overlay showing lane lines, bounding boxes colour-coded by threat level, distance/speed labels, traffic light status, sign labels, and FPS counter. |

---

## Architecture

```
Raw camera frame
     │
     ├──► lane_detect.rs ──► lane_manager.rs ──┐
     │    Canny + Hough       Smooth + filter   │
     │                                          │
     ├──► lib.rs (AdasBrain) ──────────────────►├──► lib.rs — PyO3 Bindings ──► Python app
     │    YOLO via ONNX runtime                 │    RustTracker
     │                                          │    RustLaneManager
     ├──► traffic_light.rs ────────────────────►│    AdasBrain
          HSV colour analysis                   │    detect_lanes_rust
                                                │    check_traffic_lights
```

### Rust → Python Bindings (PyO3)

The `adas_pilot` Python module exposes:

| Export | Type | Purpose |
|---|---|---|
| `detect_lanes_rust(frame)` | Function | Returns an `(N × 4)` NumPy array of detected lane line endpoints. |
| `check_traffic_lights(frame)` | Function | Returns `"RED"`, `"YELLOW"`, `"GREEN"`, or `"NONE"`. |
| `RustTracker` | Class | Centroid tracker — call `process_frame(detections, dt)` to get tracked objects with distance, speed, and TTC. |
| `RustLaneManager` | Class | Smoothed lane tracker — call `update_lanes(raw_lines, img_width)` and `filter_objects(detections)`. |
| `AdasBrain` | Class | ONNX inference engine — call `process_frame(bytes, w, h, conf_threshold)` for traffic sign detections. |

---

## Project Structure

```
adas_pilot/
├── src/                        # Rust source (compiled to a Python extension via PyO3)
│   ├── lib.rs                  #   Module root: AdasBrain (ONNX), PyO3 bindings, NMS
│   ├── lane_detect.rs          #   Canny + Hough lane detection
│   ├── lane_manager.rs         #   Lane smoothing & ego-lane filtering
│   ├── object_proc.rs          #   Centroid tracker, distance & TTC estimation
│   └── traffic_light.rs        #   HSV-based traffic light classifier
│
├── app/                        # Python application layer
│   ├── main.py                 #   Entry point — camera loop, AI orchestration, HUD
│   ├── display.py              #   (reserved) display utilities
│   ├── audio_alert.py          #   (reserved) audio warning system
│   └── tools/                  #   Training & export utilities
│       ├── train_signs.py      #     Fine-tune YOLOv8 on a custom traffic sign dataset
│       └── export_model.py     #     Export trained weights to ONNX
│
├── models/                     # Pre-trained model weights
│   ├── traffic_signs.onnx      #   Custom ONNX model for traffic sign recognition
│   ├── yolov8n.pt              #   YOLOv8-Nano (general object detection)
│   └── yolo11n.pt              #   YOLO11-Nano
│
├── assets/
│   ├── videos/                 #   Test video files (e.g. test_vid.mp4)
│   └── datasets/               #   Training datasets (gitignored)
│
├── Cargo.toml                  # Rust crate config & dependencies
├── pyproject.toml              # Python build config (maturin)
└── .gitignore
```

---

## Prerequisites

- **Rust** (stable toolchain) — [rustup.rs](https://rustup.rs)
- **Python ≥ 3.9**
- **NVIDIA GPU + CUDA toolkit** (recommended for real-time performance)
- An ONNX-compatible runtime (bundled via the `ort` crate)

---

## Setup

```bash
# 1. Clone the repository
git clone <repo-url> && cd adas_pilot

# 2. Create a Python virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# 3. Install Python dependencies
pip install maturin opencv-python numpy torch ultralytics onnxruntime-gpu

# 4. Build the Rust extension (compiles & installs into the active venv)
maturin develop --release

# 5. Place a test video
#    Put your dashcam video at assets/videos/test_vid.mp4

# 6. Run
cd app
python main.py
```

---

## Usage

```bash
cd app
python main.py
```

### Controls
- Press **`q`** to quit.

### Configuration (`main.py`)

| Variable | Default | Description |
|---|---|---|
| `IS_TWO_WAY_ROAD` | `False` | Set to `True` for two-way roads (changes lane-filtering logic and left-lane colour). |
| `AI_SKIP_FRAMES` | `3` | Run heavy AI (YOLO + ONNX) every N-th frame to maintain FPS. |
| `SIGN_CLASSES` | `{52: "STOP"}` | Map ONNX class IDs to human-readable sign names. |

### HUD Colour Coding

| Colour | Meaning |
|---|---|
| 🟢 Green box | Tracked vehicle — safe distance |
| 🟡 Yellow box | Caution — vehicle within 25 m or TTC < 5 s |
| 🔴 Red box | **Danger** — ego-lane vehicle with TTC < 2.5 s |
| 🟣 Magenta box | Detected traffic sign |

---

## Training a Custom Traffic Sign Model

```bash
# 1. Prepare a YOLO-format dataset under assets/datasets/ with a data.yaml

# 2. Train
cd app
python tools/train_signs.py

# 3. Export best weights to ONNX
python tools/export_model.py

# 4. Move the .onnx file to models/traffic_signs.onnx
```

---

## Key Dependencies

### Rust (`Cargo.toml`)
| Crate | Purpose |
|---|---|
| `pyo3` 0.21 | Python ↔ Rust bindings |
| `ort` 2.0.0-rc.9 (CUDA) | ONNX Runtime inference |
| `numpy` 0.21 | NumPy ↔ ndarray interop |
| `image` 0.25 | Image decoding & resizing |
| `imageproc` 0.25 | Canny edge detection, Hough transform |
| `ndarray` 0.15 | N-dimensional array operations |

### Python
| Package | Purpose |
|---|---|
| `opencv-python` | Video I/O, image processing, HUD rendering |
| `numpy` | Array operations |
| `torch` | PyTorch backend for Ultralytics YOLO |
| `ultralytics` | YOLOv8 object detection |
| `onnxruntime` | ONNX provider verification |
| `maturin` | Build system for the Rust extension |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: adas_pilot` | Run `maturin develop --release` inside the project root with your venv active. |
| Video not found | Ensure `assets/videos/test_vid.mp4` exists. The app falls back to `test_video.mp4` in `app/`. |
| YOLO fails to load on GPU | Verify CUDA is installed: `python -c "import torch; print(torch.cuda.is_available())"`. |
| ONNX model not found | Ensure `models/traffic_signs.onnx` exists. Train one with `tools/train_signs.py` or supply your own. |
| Low FPS | Increase `AI_SKIP_FRAMES`, use a smaller YOLO model, or ensure GPU acceleration is active. |

---

## License

See repository root for license information.