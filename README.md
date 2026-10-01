# Football Tactical Analysis System

A computer-vision pipeline for detecting, tracking, and annotating players in football videos using YOLO and ByteTrack.

## Features

- Player, goalkeeper, referee, and ball detection
- Persistent player tracking IDs
- Two-team classification from jersey colors (TEAM_A / TEAM_B / UNKNOWN)
- Annotated video output with per-team colors and a team legend
- Top-down pitch mapping via manual calibration (radar minimap + pitch CSV)
- Tracking, team, and pitch CSV export
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

### Pitch calibration (once per video)

```powershell
python main.py --input input\match.mp4 --calibrate
```

Phase 1 shows a frame (middle of the clip; `--calibration-frame N` to change);
left-click 4-8 points **on the ground** — penalty-box corners, centre spot,
halfway-line/sideline intersections. Never players, sky, or stands. Press
Enter. Phase 2 shows a drawn pitch diagram: click the same points in the same
order, Enter to build the homography and save `calibration/pitch.json` (a mean
reprojection error well under a metre means a good fit). Keys: `u` undo,
`r` restart, `Esc` cancel. Without a calibration file the pitch stage simply
stays off. The homography matches the camera pose at the chosen frame - if the
broadcast pans or zooms far from it later, re-run with `--calibration-frame`
set inside the segment you actually analyse.

After calibrating, normal runs add a top-down radar in the bottom-left corner
(the original video annotations are unchanged) and can export positions:

```powershell
python main.py --input input\match.mp4 --pitch-csv output\pitch.csv
```

`pitch.csv` columns: `frame,timestamp,player_id,team,pitch_x,pitch_y`

Coordinates are metres on a configurable pitch (default 105 x 68 m): origin
`(0,0)` at one pitch corner, `x` along the length (0 = left goal line,
105 = right goal line), `y` along the width (0 = near sideline, 68 = far one).
Players are projected from their foot position (bottom-center of the box), so
points off the pitch plane come out outside `0..length` / `0..width`.

View available options:

```powershell
python main.py --help
```

## Testing

```powershell
python -m pytest
```

## Configuration

Edit `config.yaml` to configure model weights, confidence thresholds, video paths, tracker settings, team classification (`team_classifier` section: torso crop, dominant-color extraction, clustering, uncertainty gates, temporal smoothing), pitch mapping (`pitch` section: dimensions, calibration path, radar toggle), device selection, and logging.

YOLO weights are downloaded automatically when missing.

## Roadmap

- [x] Team classification
- [x] Pitch homography and radar projection (manual calibration)
- [ ] Formation analysis
- [ ] Passing-lane detection
- [ ] Tactical metrics export
