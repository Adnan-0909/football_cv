# Football Tactical Analysis System

A computer-vision pipeline for detecting, tracking, and annotating players in football videos using YOLO and ByteTrack.

## Features

- Player, goalkeeper, referee, and ball detection
- Persistent player tracking IDs
- Two-team classification from jersey colors (TEAM_A / TEAM_B / UNKNOWN)
- Annotated video output with per-team colors and a team legend
- Tracking and team CSV export
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

Export tracking and team data:

```powershell
python main.py --input input\match.mp4 --track-csv output\tracks.csv --team-csv output\teams.csv
```

`tracks.csv` columns: `frame,timestamp,player_id,x1,y1,x2,y2,cx,cy`
`teams.csv` columns: `frame,timestamp,player_id,team,cx,cy`

Teams are discovered from the footage itself (no fixed colors): each player is
labeled `TEAM_A`, `TEAM_B`, or `UNKNOWN` while uncertain (referee, goalkeeper,
heavy occlusion, background-dominated box).

View available options:

```powershell
python main.py --help
```

## Testing

```powershell
python -m pytest
```

## Configuration

Edit `config.yaml` to configure model weights, confidence thresholds, video paths, tracker settings, team classification (`team_classifier` section: torso crop, dominant-color extraction, clustering, uncertainty gates, temporal smoothing), device selection, and logging.

YOLO weights are downloaded automatically when missing.

## Roadmap

- [x] Team classification
- [ ] Pitch homography and radar projection
- [ ] Formation analysis
- [ ] Passing-lane detection
- [ ] Tactical metrics export
