#!/usr/bin/env python3
"""Combine same-boundary label directories into ``candidate_labels`` JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_candidate_specs(values: list[str]) -> list[tuple[str, Path]]:
    result = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"candidate must be NAME=DIR, got {value!r}")
        name, path = value.split("=", 1)
        if not name or not path:
            raise ValueError(f"candidate must be NAME=DIR, got {value!r}")
        result.append((name, Path(path)))
    return result


def segment_label(segment: dict[str, Any]) -> str:
    return str(segment.get("subtask") or segment.get("instruction") or segment.get("label") or "").strip()


def same_boundaries(left: dict[str, Any], right: dict[str, Any], tolerance: float = 1e-3) -> bool:
    for key in ("start_sec", "end_sec"):
        if abs(float(left[key]) - float(right[key])) > tolerance:
            return False
    return True


def assemble_episode(
    boundary: dict[str, Any],
    candidates: list[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    episode_id = str(boundary.get("id") or "")
    boundary_segments = boundary.get("segments") or []
    result = dict(boundary)
    result_segments = []
    for index, base_segment in enumerate(boundary_segments):
        labels = {}
        for name, candidate in candidates:
            candidate_segments = candidate.get("segments") or []
            if len(candidate_segments) != len(boundary_segments):
                raise ValueError(
                    f"{episode_id}: {name} has {len(candidate_segments)} segments; "
                    f"expected {len(boundary_segments)}"
                )
            item = candidate_segments[index]
            if not same_boundaries(base_segment, item):
                raise ValueError(f"{episode_id}: {name} boundary mismatch at segment {index}")
            labels[name] = segment_label(item)
        row = dict(base_segment)
        row["candidate_labels"] = labels
        result_segments.append(row)
    result["segments"] = result_segments
    result["candidate_sources"] = [name for name, _ in candidates]
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundary-dir", type=Path, required=True)
    parser.add_argument("--candidate", nargs="+", required=True, help="NAME=DIR entries")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ids", nargs="*")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    specs = parse_candidate_specs(args.candidate)
    paths = sorted(args.boundary_dir.glob("homer_*.json"))
    if args.ids:
        wanted = set(args.ids)
        paths = [path for path in paths if path.stem in wanted]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for boundary_path in paths:
        candidate_docs = [
            (name, read_json(directory / boundary_path.name))
            for name, directory in specs
        ]
        output = assemble_episode(read_json(boundary_path), candidate_docs)
        output_path = args.output_dir / boundary_path.name
        output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
