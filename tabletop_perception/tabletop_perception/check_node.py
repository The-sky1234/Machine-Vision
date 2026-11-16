"""Load a saved frame, project its objects, and print the pixel error."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tabletop_perception.frame_io import default_frames_dir, list_frame_dirs, load_frame
from tabletop_perception.projection import DEFAULT_THRESHOLD_PX, FrameReport, evaluate_frame
from tabletop_perception.topics import COLOR_TOPIC, DEPTH_TOPIC, INFO_TOPIC


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Project tabletop object poses into a saved frame and print the error."
    )
    parser.add_argument(
        "frames",
        nargs="*",
        type=Path,
        help="Frame directories. Default: every sample under frames/.",
    )
    parser.add_argument(
        "--frames-root",
        type=Path,
        default=None,
        help="Directory of frame folders. Used when no frame path is given.",
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD_PX)
    args = parser.parse_args(argv)

    try:
        paths = _frame_paths(args.frames, args.frames_root)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"topics color={COLOR_TOPIC} depth={DEPTH_TOPIC} info={INFO_TOPIC}")
    failed = False
    for path in paths:
        try:
            frame = load_frame(path)
        except (FileNotFoundError, ValueError, KeyError) as exc:
            print(f"error: {path}: {exc}", file=sys.stderr)
            return 1
        report = evaluate_frame(frame, args.threshold)
        print(format_report(report))
        failed = failed or not report.passed
    return 1 if failed else 0


def format_report(report: FrameReport) -> str:
    lines = []
    for item in report.objects:
        depth = "na" if item.depth_error_m is None else f"{item.depth_error_m:.4f}"
        status = "PASS" if item.passed else "FAIL"
        lines.append(
            f"frame={report.frame_name} object={item.name} id={item.object_id} "
            f"center_error_px={item.center_error_px:.3f} "
            f"bbox_error_px={item.bbox_error_px:.3f} "
            f"max_outside_px={item.max_outside_px:.3f} "
            f"depth_error_m={depth} "
            f"projection_error_px={item.projection_error_px:.3f} "
            f"status={status}"
        )
    status = "PASS" if report.passed else "FAIL"
    lines.append(
        f"frame={report.frame_name} "
        f"max_projection_error_px={report.max_projection_error_px:.3f} "
        f"threshold_px={report.threshold_px:.1f} status={status}"
    )
    return "\n".join(lines)


def _frame_paths(frames: list[Path], frames_root: Path | None) -> list[Path]:
    if frames:
        return frames
    root = frames_root if frames_root is not None else default_frames_dir()
    return list_frame_dirs(root)


if __name__ == "__main__":
    sys.exit(main())
