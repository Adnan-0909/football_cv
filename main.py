"""
Football Tactical Analysis System - Entry Point
===============================================

Main entry script for the project.

Runs an environment/dependency check, loads the configuration, and then executes
the detection -> tracking -> visualization pipeline over a raw video:

    python main.py                       # run with config.yaml defaults
    python main.py --input my_match.mp4  # run on your own footage
    python main.py --track-csv           # also export per-frame tracking data
    python main.py --check-only          # only verify the environment
"""

import argparse
import logging
import os
from pathlib import Path
import sys
from typing import Dict, Optional, Tuple

# Add the project root to sys.path to support execution from any directory
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import AppConfig, load_config, setup_logging


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed CLI options.
    """
    parser = argparse.ArgumentParser(
        description="Football Tactical Analysis System - detection, tracking and tactical overlay"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom configuration YAML file (default: config.yaml in project root)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Run environment and dependency check and exit without processing a video.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Raw input video path (overrides video.input_path from the config).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Annotated output video path (overrides video.output_path from the config).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Process at most N frames (overrides video.max_frames).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Process every Nth frame (overrides video.frame_stride).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Inference device: "cpu", "cuda", or a GPU index such as "0" (overrides model.device).',
    )
    parser.add_argument(
        "--track-csv",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help=(
            "Save per-frame tracking data as CSV with columns "
            "frame,timestamp,player_id,x1,y1,x2,y2,cx,cy. "
            "Optionally pass a path (default: output/tracks.csv)."
        ),
    )
    return parser.parse_args()


def resolve_track_csv_path(config: AppConfig, args: argparse.Namespace) -> Optional[Path]:
    """Return the CSV destination requested by --track-csv, or None when not asked for."""
    if args.track_csv is None:
        return None
    if args.track_csv:
        return Path(args.track_csv).expanduser().resolve()
    return config.project_root / "output" / "tracks.csv"


def apply_overrides(config: AppConfig, args: argparse.Namespace) -> None:
    """Apply command-line overrides on top of the loaded configuration."""
    if args.input is not None:
        config.video.input_path = Path(args.input).expanduser().resolve()
    if args.output is not None:
        config.video.output_path = Path(args.output).expanduser().resolve()
    if args.max_frames is not None:
        config.video.max_frames = max(0, args.max_frames)
    if args.stride is not None:
        config.video.frame_stride = max(1, args.stride)
    if args.device is not None:
        config.model.device = args.device



def verify_dependencies(logger: logging.Logger, tracker_type: str = "bytetrack") -> Tuple[bool, Dict[str, str]]:
    """
    Verify that core dependencies are installed and importable.

    Args:
        logger: Logger used for detailed messages.
        tracker_type: Configured tracker backend - ``lap`` is only required by
            the built-in ``bytetrack`` backend.

    Returns:
        Tuple[bool, Dict[str, str]]: Success flag and dictionary of package versions.
    """
    results: Dict[str, str] = {}
    all_ok = True

    # 1. Check Python version
    py_version = sys.version.split()[0]
    results["Python"] = py_version
    logger.info("Python interpreter: %s", py_version)

    # 2. Check NumPy
    try:
        import numpy as np
        results["NumPy"] = np.__version__
        logger.info("NumPy imported successfully: version %s", np.__version__)
    except ImportError as e:
        results["NumPy"] = "NOT INSTALLED"
        logger.error("Failed to import NumPy: %s", e)
        all_ok = False

    # 3. Check OpenCV
    try:
        import cv2
        results["OpenCV"] = cv2.__version__
        logger.info("OpenCV imported successfully: version %s", cv2.__version__)
    except ImportError as e:
        results["OpenCV"] = "NOT INSTALLED"
        logger.error("Failed to import OpenCV (cv2): %s", e)
        all_ok = False

    # 4. Check Ultralytics YOLO
    try:
        import ultralytics
        from ultralytics import YOLO
        results["Ultralytics YOLO"] = ultralytics.__version__
        logger.info("Ultralytics YOLO imported successfully: version %s", ultralytics.__version__)
    except ImportError as e:
        results["Ultralytics YOLO"] = "NOT INSTALLED"
        logger.error("Failed to import Ultralytics YOLO: %s", e)
        all_ok = False

    # 5. Check PyTorch & Device acceleration
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU only"
        results["PyTorch"] = f"{torch.__version__} (CUDA: {cuda_avail} - {device_name})"
        logger.info("PyTorch imported successfully: %s", results["PyTorch"])
    except ImportError:
        results["PyTorch"] = "Torch not directly imported (will be loaded with Ultralytics)"

    # 5b. TorchVision is imported by Ultralytics during inference warmup; a torch
    # install missing its own dependencies (sympy, networkx, ...) fails there.
    try:
        import torchvision
        results["TorchVision"] = torchvision.__version__
        logger.info("TorchVision imported successfully: version %s", torchvision.__version__)
    except Exception as e:
        results["TorchVision"] = f"NOT USABLE ({type(e).__name__})"
        logger.error(
            "Failed to import torchvision (required for YOLO inference): %s. "
            "Try: pip install --upgrade torch torchvision sympy networkx fsspec",
            e,
        )
        all_ok = False

    # 6. Check PyYAML
    try:
        import yaml
        results["PyYAML"] = getattr(yaml, "__version__", "Available")
        logger.info("PyYAML imported successfully: version %s", results["PyYAML"])
    except ImportError as e:
        results["PyYAML"] = "NOT INSTALLED"
        logger.error("Failed to import PyYAML: %s", e)
        all_ok = False

    # 7. lap - required by Ultralytics' built-in ByteTrack tracker.
    try:
        import lap
        results["lap (ByteTrack)"] = getattr(lap, "__version__", "Installed")
        logger.info("lap imported successfully (ByteTrack backend available)")
    except Exception as e:
        if str(tracker_type).strip().lower() == "bytetrack":
            results["lap (ByteTrack)"] = "NOT INSTALLED"
            logger.error(
                "lap is required by the built-in ByteTrack backend: %s. "
                "Run 'pip install -r requirements.txt', or set "
                "tracker.tracker_type: bytetrack_lite in config.yaml to use the "
                "dependency-free fallback.",
                e,
            )
            all_ok = False
        else:
            results["lap (ByteTrack)"] = "n/a (bytetrack_lite selected)"
            logger.info("lap missing but unused: tracker_type=bytetrack_lite")

    return all_ok, results


def print_summary(
    all_ok: bool,
    versions: Dict[str, str],
    config: AppConfig,
    logger: logging.Logger,
) -> None:
    """
    Print an informative summary banner for beginners.
    """
    border = "=" * 70
    print("\n" + border)
    print("  FOOTBALL TACTICAL ANALYSIS SYSTEM - FOUNDATION VERIFICATION")
    print(border)
    print("Dependencies Status:")
    for package, ver in versions.items():
        status_symbol = "[OK]" if ("NOT INSTALLED" not in ver and "NOT USABLE" not in ver) else "[FAILED]"
        print(f"  {status_symbol:<10} {package:<20} : {ver}")

    print("\nProject Paths:")
    print(f"  Project Root     : {config.project_root}")
    print(f"  Model Weights    : {config.model.weights_path}")
    print(f"  Input Directory  : {config.project_root / 'input'}")
    print(f"  Output Directory : {config.project_root / 'output'}")

    print("\nConfiguration Settings:")
    print(f"  Confidence Thresh: {config.model.confidence_threshold}")
    print(f"  Target Device    : {config.model.device}")
    print(f"  Default Input    : {config.video.input_path}")
    print(f"  Default Output   : {config.video.output_path}")
    print(f"  Tracker Type     : {config.tracker.tracker_type}")
    print(border)

    if all_ok:
        logger.info("SUCCESS: All core libraries and configurations are verified!")
        print("\n[READY] Environment is verified.")
        print("Next: run a video through the pipeline with 'python main.py --input <video.mp4>'.")
        print("Not implemented yet: team classification, pitch homography, formations, passing lanes.")
    else:
        logger.warning("ATTENTION: Some dependencies are missing.")
        print("\nPlease activate your virtual environment and run:")
        print("  pip install -r requirements.txt\n")


def main() -> int:
    """
    Application main function.

    Returns:
        int: Exit code (0 for success, 1 for dependency or pipeline errors).
    """
    args = parse_arguments()

    # 1. Load configuration
    config = load_config(args.config)
    apply_overrides(config, args)

    # 2. Setup logging
    logger = setup_logging(config.logging)
    logger.info("Initializing Football Tactical Analysis System...")
    logger.info("Project Root directory: %s", config.project_root)

    # 3. Verify core dependencies (OpenCV, YOLO, NumPy, PyYAML, lap)
    logger.info("Verifying core dependencies...")
    all_ok, versions = verify_dependencies(logger, config.tracker.tracker_type)

    # 4. Print user-friendly status banner
    print_summary(all_ok, versions, config, logger)

    if not all_ok:
        return 1

    if args.check_only:
        logger.info("--check-only requested: skipping video processing.")
        return 0

    # 5. Run the detection + tracking + annotation pipeline
    from app.pipeline import TacticalPipeline

    track_csv_path = resolve_track_csv_path(config, args)
    logger.info(
        "Starting pipeline: %s -> %s",
        config.video.input_path,
        config.video.output_path,
    )
    if track_csv_path is not None:
        logger.info("Tracking records will also be written to %s", track_csv_path)

    try:
        stats = TacticalPipeline(config).run(
            input_path=config.video.input_path,
            output_path=config.video.output_path,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    except Exception:
        logger.exception("Pipeline failed")
        return 1

    # 6. Optionally export the per-frame tracking records as CSV.
    if track_csv_path is not None:
        saved = stats.track_log.save_csv(track_csv_path)
        logger.info(
            "Saved %d tracking records (%d players) to %s",
            len(stats.track_log),
            len(stats.track_log.unique_ids),
            saved,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
