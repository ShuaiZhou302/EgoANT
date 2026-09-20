#!/usr/bin/env python3
"""Remove known EgoANT intermediate artifacts after a successful run."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


FILE_PATTERNS = (
    "hand_recon.json",
    "annotations.json",
    "gold_reference.json",
    "segments.json",
    "speed_plot.png",
    "segment_*_annotation.json",
    "segment_*_error.json",
    "segment_*_payload.json",
    "segment_*_prompt.txt",
    "*_debug.json",
    "*_debug_boundaries.json",
    "api_usage.jsonl",
    "*.log",
)

DIR_PATTERNS = (
    "segment_*_frames_raw",
    "segment_*_frames_overlay",
    "merge_judge_annotations",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Delete EgoANT intermediate output files.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--final-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    final_json = args.final_json.resolve()
    if final_json.parent != output_dir or not final_json.exists():
        raise ValueError("Final JSON must exist directly inside output-dir before cleanup")

    removed = 0
    for pattern in FILE_PATTERNS:
        for path in output_dir.glob(pattern):
            if path.resolve() == final_json or not path.is_file():
                continue
            path.unlink()
            removed += 1
    for pattern in DIR_PATTERNS:
        for path in output_dir.glob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
                removed += 1
    print(f"Removed {removed} intermediate artifacts from {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
