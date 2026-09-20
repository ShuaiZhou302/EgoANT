#!/usr/bin/env python3
"""Aggregate per-video token usage and create review pairs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize an EgoANT example-suite run.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--example-root", type=Path, required=True)
    return parser.parse_args()


def add_usage(total: dict[str, Any], usage: dict[str, Any]) -> None:
    for key in (
        "requests",
        "successful_requests",
        "failed_requests",
        "usage_reported_requests",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    ):
        total[key] = total.get(key, 0) + int(usage.get(key, 0) or 0)

    errors = total.setdefault("error_counts", {})
    for key, value in usage.get("error_counts", {}).items():
        errors[key] = errors.get(key, 0) + int(value)


def main() -> int:
    args = parse_args()
    pair_root = args.output_root / "pairs"
    pair_root.mkdir(parents=True, exist_ok=True)
    total: dict[str, Any] = {}
    videos: list[dict[str, Any]] = []

    for output_dir in sorted(path for path in args.output_root.iterdir() if path.is_dir() and path.name != "pairs"):
        final_json = output_dir / "merge_judge_annotations.json"
        usage_path = output_dir / "token_usage.json"
        if not final_json.exists():
            continue
        annotation = json.loads(final_json.read_text(encoding="utf-8"))
        usage = json.loads(usage_path.read_text(encoding="utf-8")) if usage_path.exists() else {}
        add_usage(total, usage)
        pair_dir = pair_root / output_dir.name
        pair_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(final_json, pair_dir / "annotations.json")
        gold_candidates = list((args.example_root / "annotation" / output_dir.name.rsplit("_", 1)[0]).glob("*.json"))
        gold_path = output_dir / "gold_reference.json"
        if gold_path.exists():
            shutil.copy2(gold_path, pair_dir / "gold_reference.json")
        elif gold_candidates:
            shutil.copy2(gold_candidates[0], pair_dir / "gold_reference.json")
        videos.append(
            {
                "sample": output_dir.name,
                "segments": len(annotation.get("segments", [])),
                "usage": usage,
            }
        )

    summary = {"video_count": len(videos), "usage": total, "videos": videos}
    (args.output_root / "suite_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(args.output_root / "suite_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
