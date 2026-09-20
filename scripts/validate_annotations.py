#!/usr/bin/env python3
"""Validate final annotations and report API-fallback segments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-fallbacks", action="store_true")
    return parser.parse_args()


def is_api_fallback(segment: dict[str, Any]) -> bool:
    return (
        str(segment.get("instruction", "")).strip().upper() == "N/A"
        and float(segment.get("confidence", 0.0)) == 0.0
        and bool(segment.get("needs_review", False))
    )


def main() -> int:
    args = parse_args()
    data = json.loads(args.annotations.read_text(encoding="utf-8"))
    segments = data.get("segments")
    if not isinstance(segments, list):
        raise ValueError(f"Missing segments list: {args.annotations}")

    fallbacks = [
        {
            "seg_id": segment.get("seg_id"),
            "start_frame": segment.get("start_frame"),
            "end_frame": segment.get("end_frame"),
        }
        for segment in segments
        if is_api_fallback(segment)
    ]
    report = {
        "status": "failed" if fallbacks else "passed",
        "segment_count": len(segments),
        "api_fallback_count": len(fallbacks),
        "api_fallback_segments": fallbacks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if not fallbacks or args.allow_fallbacks else 2


if __name__ == "__main__":
    raise SystemExit(main())
