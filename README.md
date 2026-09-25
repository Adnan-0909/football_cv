# Football Tactical Analysis System ⚽📊

A modular computer-vision and deep-learning pipeline for tactical football (soccer) video analysis.

This project is designed to process match broadcast footage, detect and track players and the ball, classify teams by kit colors, project player coordinates onto a 2D tactical pitch radar, analyze passing lanes, and compute tactical formation metrics (team width, depth, compactness).

---

## 🎯 Project Roadmap & Vision

1. **Object Detection**: Detect outfield players, goalkeepers, referees, and the ball from match video using Ultralytics YOLO.
2. **Multi-Object Tracking**: Track players across frames with persistent IDs using ByteTrack (Ultralytics' built-in implementation, with a dependency-free fallback).
3. **Team Classification**: Cluster player jersey colors using K-Means / color space modeling to identify teams.
4. **2D Pitch Mapping**: Use planar homography and pitch keypoints to map camera coordinates onto a calibrated 2D pitch view.
5. **Formation Reconstruction**: Automatically group player lines to determine formations (e.g., 4-3-3, 4-4-2, 3-5-2).
6. **Player Connections & Structure**: Render Delaunay / distance meshes connecting teammates to show tactical shape.
7. **Passing Lane Analysis**: Detect open passing corridors from the ball carrier to teammates and evaluate defensive pressure.
8. **Tactical Metrics**: Calculate real-time tactical metrics including team width, depth, centroid, and compactness area.

> **Current Status**: **Phase 2 - Detection, Tracking & Annotation (complete)**. The foundation (config, logging, video I/O) plus `FootballDetector.detect()`, ByteTrack tracking with persistent player IDs, per-frame track records with CSV export, and the end-to-end orchestration (`app/pipeline.py`) are implemented: a raw MP4 goes in, an annotated MP4 with stable player IDs comes out, and the tracking data can be exported as CSV. Still to build: team classification, pitch homography, formations, passing lanes.

---

## 📁 Project Architecture & Structure

```
football_computer_vision/
├── app/
│   ├── __init__.py          # Package initialization & versioning
│   ├── config.py            # Dynamic configuration loading & path resolution
│   ├── video.py             # OpenCV VideoReader and VideoWriter abstractions
│   ├── detector.py          # YOLO object detector (Ultralytics) & Detection dataclass
│   ├── matching.py          # Pure-NumPy Hungarian solver for data association
│   ├── tracker.py           # ByteTrack multi-object tracker (Ultralytics + lite fallback)
│   ├── track_log.py         # Per-frame track records ({frame, timestamp, player_id, ...}) & CSV export
│   ├── pipeline.py          # End-to-end orchestration (read -> detect -> track -> draw -> write)
│   ├── team_classifier.py   # Jersey color extraction & clustering interface
│   ├── pitch.py             # 2D pitch transformation & Homography mapping
│   ├── formation.py         # Formation detection & tactical metrics calculation
│   ├── passing.py           # Passing lane detection & corridor openness scoring
│   └── visualization.py     # Overlay drawing (ellipses, trails, minimap radar, HUD)
├── config.yaml              # Central configuration file for all parameters
├── input/                   # Place raw input football match videos here
│   └── .gitkeep
├── output/                  # Annotated videos and tactical analysis exports
│   └── .gitkeep
├── models/                  # YOLO weights (.pt), auto-downloaded on first run
│   └── .gitkeep
├── tests/                   # Unit tests (config, imports, matcher, tracker, pipeline)
│   ├── test_config.py
│   ├── test_imports.py
│   ├── test_matching.py
│   ├── test_tracker.py
│   ├── test_track_log.py
│   ├── test_detector.py
│   ├── test_pipeline.py
│   └── test_visualization.py
├── main.py                  # Project entry point: dependency check + pipeline runner
├── pytest.ini               # pytest configuration (test paths, pythonpath)
├── requirements.txt         # Core dependencies (OpenCV, YOLO, NumPy, PyYAML, lap)
├── .gitignore               # Ignores large weights, media files, and caches
└── README.md                # Project documentation & usage instructions
```

---

## 🚀 Getting Started

### 1. Prerequisites
- **Python**: Python 3.10, 3.11, or 3.12 installed.
- **Git** (optional): For version control.

### 2. Set Up Virtual Environment

To isolate dependencies and avoid package conflicts, use a Python virtual environment:

#### On Windows (PowerShell):
```powershell
# Create virtual environment named .venv
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1
```

*(If PowerShell script execution is restricted, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` or use `.\.venv\Scripts\activate.bat` in Command Prompt).*

#### On Linux / macOS:
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate
```

### 3. Install Dependencies

With the virtual environment activated, install the required packages:

```bash
pip install -r requirements.txt
```

Core libraries installed:
- `opencv-python`: Video capture, frame manipulation, and drawing overlays.
- `ultralytics`: YOLOv8 deep learning models for player and ball detection.
- `numpy`: Coordinate transformations and matrix calculations.
- `pyyaml`: Configuration file parsing.
- `lap`: Linear assignment used by Ultralytics' built-in ByteTrack tracker (`tracker_type: bytetrack`; only needed if you don't use the `bytetrack_lite` fallback).
- `pytest`: Automated testing suite.

---

## ⚙️ Configuration (`config.yaml`)

All parameters are configured in `config.yaml` without hardcoding machine-specific paths:

| Section | Parameter | Default | Description |
| :--- | :--- | :--- | :--- |
| **`model`** | `weights_path` | `models/yolov8s.pt` | Path to YOLO detection weights (auto-downloaded if missing) |
| | `confidence_threshold` | `0.35` | Minimum detection confidence score (0.0 - 1.0) |
| | `iou_threshold` | `0.5` | Non-maximum suppression IoU threshold |
| | `device` | `cpu` | Processing device (`cpu`, `cuda`, or `0`) |
| **`video`** | `input_path` | `input/match_sample.mp4` | Default input match video |
| | `output_path` | `output/tactical_analysis.mp4` | Path for processed output video |
| | `max_frames` | `null` | Maximum frames to analyze (`null` for entire clip) |
| | `frame_stride` | `1` | Stride step (`1` = process every frame) |
| **`tracker`** | `tracker_type` | `bytetrack` | Tracker backend: `bytetrack` (Ultralytics' built-in ByteTrack, needs `lap`) or `bytetrack_lite` (dependency-free fallback) |
| | `player_only` | `true` | Track only players — the ball is still detected and drawn but never gets an ID |
| | `track_high_thresh` | `0.5` | Detections at/above this confidence drive stage-1 matching |
| | `track_low_thresh` | `0.1` | Detections below this confidence are ignored |
| | `new_track_thresh` | `0.6` | Confidence required to start a **new** track (lower it to `0.45` if few players get IDs) |
| | `track_buffer` | `30` | Number of frames to retain lost tracks |
| | `match_thresh` | `0.8` | Maximum assignment cost (1 − IoU × confidence) in stage 1 |
| **`pitch`** | `length_meters` | `105.0` | Standard pitch length in meters |
| | `width_meters` | `68.0` | Standard pitch width in meters |
| **`logging`** | `level` | `INFO` | Console log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

---

## 🏃 Running the Project

### 1. Verify the Environment

Run the main script to check Python, OpenCV, Ultralytics YOLO, PyTorch/TorchVision, and the configuration system:

```bash
python main.py --check-only
```

### 2. Run the Pipeline on a Video

Put your **raw** (unannotated) footage in `input/` — or point at any path — and let the pipeline annotate it:

```bash
# Uses input/match_sample.mp4 -> output/tactical_analysis.mp4 from config.yaml
python main.py

# Or supply your own footage explicitly
python main.py --input input/my_match.mp4 --output output/annotated.mp4

# Quick trial run: first 200 frames only
python main.py --input input/my_match.mp4 --max-frames 200
```

On the first run the configured YOLO weights (`models/yolov8s.pt`, ~21 MB) are downloaded
automatically. The log ends with a summary such as:

```
[INFO] [football_tracker.pipeline]: Pipeline finished: 40 frames | 80 detections |
       2 unique tracks (2 players) | 6.4 fps | 6.3s | -> output/annotated.mp4
```

The annotated video contains, for every tracked player:

* the **bounding box**,
* the **player ID** and **detection confidence** (`#7 0.87`) above the box,
* the foot ellipse and movement **trail**,

plus the ball marker and a HUD line (`frame | detections | players | tracks`).

### 3. Run the Tracker & Export Tracking Data (CSV)

Tracking is on by default — every player keeps the same `#id` across frames:

```bash
# Annotate a clip with persistent player IDs
python main.py --input input/my_match.mp4

# Quick trial run on the first 200 frames
python main.py --input input/my_match.mp4 --max-frames 200
```

To also **save the tracking data as CSV** (columns
`frame,timestamp,player_id,x1,y1,x2,y2,cx,cy`):

```bash
# Default destination: output/tracks.csv
python main.py --input input/my_match.mp4 --track-csv

# Explicit destination
python main.py --input input/my_match.mp4 --track-csv output/match_tracks.csv
```

Log output confirms both artifacts:

```
[INFO] [football_tracker.pipeline]: Pipeline finished: 40 frames | 80 detections | 2 unique tracks ...
[INFO] [football_tracker]: Saved 80 tracking records (2 players) to output/tracks.csv
```

Each row is one tracked player in one frame:

| Column | Meaning |
| :--- | :--- |
| `frame` | Index of the frame in the **source** video (0-based) |
| `timestamp` | Position of that frame in seconds (`frame / source_fps`, stride-independent) |
| `player_id` | Persistent track ID (matches the `#id` drawn on the video) |
| `x1, y1, x2, y2` | Bounding box corners in pixels |
| `cx, cy` | Centre of the bounding box in pixels |

Programmatic access (no CLI needed):

```python
from app.config import load_config
from app.pipeline import TacticalPipeline

stats = TacticalPipeline(load_config()).run()   # -> PipelineStats
records = stats.track_log.records               # [TrackRecord, ...]
stats.track_log.save_csv("tracks.csv")          # same columns as above
```

Notes:

* Only **players** are recorded — `tracker.player_only` keeps non-player objects
  (the ball) out of the tracker so their flickering IDs never pollute `player_id`.
  The ball is still detected and drawn on every frame.
* Records are held in memory for the whole run; for very long clips use
  `--stride 2` / `--max-frames N`.

### Command Line Options

```bash
# View all available CLI flags
python main.py --help

# Dependency check only (no video processing)
python main.py --check-only

# Custom config file, input, output, and frame limits
python main.py --config my_config.yaml --input clip.mp4 --output out.mp4 \
               --max-frames 500 --stride 2 --device cpu
```

| Flag | Effect |
| :--- | :--- |
| `--config PATH` | Use an alternative YAML configuration file |
| `--check-only` | Verify the environment and exit without processing video |
| `--input PATH` | Raw input video (overrides `video.input_path`) |
| `--output PATH` | Annotated output video (overrides `video.output_path`) |
| `--max-frames N` | Process at most N frames (overrides `video.max_frames`) |
| `--stride N` | Process every Nth frame — halves the work at `--stride 2` |
| `--device DEV` | `cpu`, `cuda`, or a GPU index such as `0` |
| `--track-csv [PATH]` | Export per-frame tracking data as CSV (default path: `output/tracks.csv`) |

---

## 🧪 Running Tests

```bash
python -m pytest          # whole suite (75 tests)
python -m pytest tests/test_tracker.py -v
python -m pytest tests/test_track_log.py -v
```

`pytest.ini` sets `testpaths` and `pythonpath`, so tests import `app.*` correctly from the
project root. The suite covers configuration, module imports, the Hungarian matcher
(verified against a brute-force optimum), the tracker (ID stability on both backends,
occlusion recovery, thresholds, backend selection), the track log and its CSV export,
the overlay drawing (box / ID / confidence labels), and an end-to-end pipeline run on a
synthetic clip with an injected stub detector (no weights or network needed).

---

## 🔧 Troubleshooting

| Symptom | Fix |
| :--- | :--- |
| `ModuleNotFoundError: No module named 'sympy'` (or `networkx`/`fsspec`) during inference | torch was installed without its dependencies. Run `pip install sympy networkx fsspec setuptools` (or reinstall torch normally). `python main.py --check-only` reports this. |
| `lap (ByteTrack): NOT INSTALLED` / tracker fails to start | The built-in ByteTrack backend needs `lap`. Run `pip install -r requirements.txt`, or set `tracker.tracker_type: bytetrack_lite` in `config.yaml` to use the dependency-free fallback. |
| A new player ID shows up one frame late | Normal: ByteTrack confirms a track on its second hit, which filters one-off false positives. The first frame of the video is immediate. |
| Very few players get IDs | Lower `tracker.new_track_thresh` in `config.yaml` (e.g. `0.45`); only detections above it start a track. |
| IDs jump between players in crowds | IoU-based association struggles when boxes overlap heavily; try `tracker.tracker_type: bytetrack` (Kalman prediction) and raise `track_buffer` slightly. |
| Slow processing | Use `--stride 2`, a smaller model (`yolov8n.pt`/`yolov8s.pt` in `model.weights_path`), or a GPU (`--device cuda`). |
| `Input video not found` | Put the clip in `input/match_sample.mp4` or pass `--input <path>`. |
| `ModuleNotFoundError: No module named 'app'` | Run commands from the project root (tests are covered by `pytest.ini`). |

---

## 📋 Next Steps

Done: dependency verification, configuration, video I/O, **detection**, **tracking with
persistent IDs**, **per-frame track records + CSV export**, the **end-to-end pipeline**,
and the overlay/HUD. Remaining phases:

1. **Team Color Clustering**: Implement torso color sampling and K-Means in `app/team_classifier.py` — the pipeline already probes for it and switches players to team colours automatically once `predict_team()` works.
2. **Pitch Homography**: Implement 4-point homography and radar projection in `app/pitch.py`, then enable `TacticalVisualizer.draw_radar_minimap()`.
3. **Formation Reconstruction**: Implement `FormationAnalyzer` in `app/formation.py`.
4. **Passing Lanes**: Implement `PassingAnalyzer` in `app/passing.py`.
5. **Tactical Metrics Export**: Per-frame tracks are already exported as CSV (`--track-csv`); add JSON output and computed tactical metrics on top.
