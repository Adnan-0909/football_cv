# Stage 5 Implementation Summary

## Overview
Implemented Stage 5: Live top-down tactical pitch alongside the original football video. The output video contains two panels:
- **LEFT**: Original football video with player tracking (bounding boxes, team colors, IDs, HUD)
- **RIGHT**: Top-down football pitch diagram with current positions of all tracked players

## Changes Made

### 1. New Module: `app/pitch_renderer.py`
- **`PitchRenderer` class**: Reusable pitch-rendering module that draws a top-down football pitch with all standard markings
  - Boundaries, halfway line, center circle
  - Penalty areas and goal areas (both sides)
  - Penalty spots, center spot, corner arcs
  - Player positions mapped from pitch coordinates (meters)
  - Team A (blueish `(255,60,0)`) and Team B (orange `(0,165,255)`) colors, neutral white for UNKNOWN
  - Player IDs displayed below each dot
  - Orientation: y=0 (near sideline) at the bottom of the panel, matching the camera view
- **`render_side_by_side()` function**: Creates a composite frame combining:
  - Left half: Original annotated video frame (with boxes, legend, HUD)
  - Right half: Pitch diagram with player positions
  - 4px grey separator between panels
  - Letterboxing to handle height differences between video and pitch

### 2. Modified: `app/pipeline.py`
- **`TacticalPipeline.__init__()`**: Accepts optional `stage5: bool` parameter
  - When `stage5=True`, creates a `PitchRenderer` instance
  - Pitch renderer size is fixed (60x39px) to keep the composite close to original video width
- **`TacticalPipeline.run()`**: Stage 5 integration
  - When `stage5=True` and calibration is available:
    - After `_annotate()`, calls `render_side_by_side()` to create composite frame
    - VideoWriter initialized with composite width (video width + pitch width + separator)
    - Composite frame written instead of single annotated frame
  - When `stage5=False` (default): Original single-panel output preserved
  - When calibration is missing: Stage 5 gracefully disabled, single-panel output
- **Imports**: Added `PitchRenderer`, `render_side_by_side` from `app.pitch_renderer`

### 3. Modified: `main.py`
- **`parse_arguments()`**: Added `--stage5` flag
  - `python main.py --stage5` enables Stage 5 output
  - No effect without a calibration file (Stage 4 must be run first)
- **`main()`**: Passes `stage5=args.stage5` to `TacticalPipeline()`

## How It Works

1. **Prerequisite**: Run `python main.py --calibrate` once per video to create `calibration/pitch.json`
   - Interactive mode: click 4-8 points on the ground (penalty-box corners, centre spot, halfway-line/sideline intersections)

2. **Pipeline flow** (with `--stage5`):
   - Detection → Tracking → Team classification (Stage 3) → Pitch projection (Stage 4) → Side-by-side composition (Stage 5)
   - Each frame: original video annotated with boxes/IDs/legend → pitch diagram with player positions → composite written to output

3. **Output format**:
   - Video width: `original_width + 4 + pitch_renderer_width` (e.g., 1280 + 4 + 60 = 1344px)
   - Video height: same as original (720px), pitch letterboxed if needed
   - Codec: same as original (mp4v)
   - Frame rate: same as original

## Testing
- All **158 existing tests pass** without modification
- Stage 5 is opt-in via `--stage5` flag, so default behavior is unchanged
- When run without `--calibrate`, Stage 5 gracefully disables (no crash)
- When run without `--stage5`, original single-panel output preserved

## Known Limitations
- Pitch renderer uses fixed small size (60px wide) to minimize output video width change
- Players must be tracked and have pitch coordinates (from Stage 4 homography)
- Stage 5 requires a valid calibration file; without it, output reverts to single panel
- No formations or passing lanes (as specified in requirements)
- Players move smoothly via per-frame pitch coordinate projection

## Files Modified/Created
- `app/pitch_renderer.py` (new) — Reusable pitch-rendering module
- `app/pipeline.py` — Stage 5 integration, `TacticalPipeline` accepts `stage5` flag
- `main.py` — `--stage5` CLI argument
- `update.md` (this file) — Summary of changes

---
**Current Status**: All 158 tests passing. Stage 5 ready for use with `python main.py --stage5 --input input/match_sample.mp4 --output output/stage5_output.mp4` after calibration.