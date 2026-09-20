#!/usr/bin/env python3
"""Materialize WGO-Bench rows into mp4 + gold segment JSON files."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def parse_segments(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        value = json.loads(value)
    out = []
    for item in value or []:
        out.append(
            {
                "start_sec": float(item["start_sec"]),
                "end_sec": float(item["end_sec"]),
                "subtask": str(item["subtask"]),
            }
        )
    return out


def write_video(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, (bytes, bytearray)):
        path.write_bytes(bytes(value))
        return
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            path.write_bytes(value["bytes"])
            return
        if value.get("path"):
            shutil.copyfile(value["path"], path)
            return
    if isinstance(value, (str, Path)):
        shutil.copyfile(value, path)
        return
    raise TypeError(f"Unsupported video field type: {type(value)!r}")


def subset_name(sample_id: str) -> str:
    """Infer the WGO subset from IDs such as ``homer_1``."""
    head, separator, tail = sample_id.rpartition("_")
    if separator and tail.isdigit() and head:
        return head.lower()
    return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--subset",
        default=None,
        help="Optional subset name such as 'homer'. Filtering uses the ID prefix.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--dataset", default="macrodata/WGO-Bench")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install datasets first: pip install datasets") from exc

    ds = load_dataset(args.dataset, split=args.split, streaming=False)
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for row in ds:
        sample_id = str(row["id"])
        subset = subset_name(sample_id)
        if args.subset and subset != args.subset.lower():
            continue
        if args.limit is not None and len(manifest) >= args.limit:
            break
        video_path = args.output_root / "videos" / subset / f"{sample_id}.mp4"
        gold_path = args.output_root / "annotations" / subset / f"{sample_id}.json"
        write_video(video_path, row["video"])
        gold = {
            "id": sample_id,
            "instruction": row.get("instruction", ""),
            "segments": parse_segments(row["segments"]),
            "metadata": row.get("metadata", ""),
        }
        gold_path.parent.mkdir(parents=True, exist_ok=True)
        gold_path.write_text(json.dumps(gold, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        manifest.append(
            {
                "id": sample_id,
                "subset": subset,
                "video": video_path.relative_to(args.output_root).as_posix(),
                "gold": gold_path.relative_to(args.output_root).as_posix(),
            }
        )
    if not manifest:
        requested = f" for subset {args.subset!r}" if args.subset else ""
        raise RuntimeError(f"No WGO-Bench rows were materialized{requested}")
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(args.output_root / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
