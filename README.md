# Football Tactical Analysis System

A computer-vision pipeline for detecting, tracking, and annotating players in football videos using YOLO and ByteTrack.

## Features

- Player, goalkeeper, referee, and ball detection
- Persistent player tracking IDs
- Annotated video output
- Tracking data CSV export
- Configurable processing options
- Dependency-free lightweight tracker fallback

## Project Structure

- `app/` — Application modules
- `input/` — Source videos
- `models/` — YOLO model weights
- `output/` — Processed videos and tracking exports
- `tests/` — Automated tests
- `config.yaml` — Application configuration
- `main.py` — Command-line entry point

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage

Check the environment:

```powershell
python main.py --check-only
```

Process a video:

```powershell
python main.py --input input\match.mp4 --output output\annotated.mp4
```

Limit processing:

```powershell
python main.py --input input\match.mp4 --max-frames 200 --stride 2
```

Export tracking data:

```powershell
python main.py --input input\match.mp4 --track-csv output\tracks.csv
```

View available options:

```powershell
python main.py --help
```

## Testing

```powershell
python -m pytest
```

## Configuration

Edit `config.yaml` to configure model weights, confidence thresholds, video paths, tracker settings, device selection, and logging.

YOLO weights are downloaded automatically when missing.

## Roadmap

- Team classification
- Pitch homography and radar projection
- Formation analysis
- Passing-lane detection
- Tactical metrics export
