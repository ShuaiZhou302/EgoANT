#!/usr/bin/env python3
"""Export annotation segments as individual video clips."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


def safe_name(value: str, limit: int = 60) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._")
    return (name or "segment")[:limit]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cut annotation segments into MP4 clips with ffmpeg.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data: dict[str, Any] = json.loads(args.annotations.read_text(encoding="utf-8"))
    segments = data.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError(f"No segments found in {args.annotations}")

    fps = float(data.get("fps") or 30.0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []

    for output_id, segment in enumerate(segments):
        seg_id = int(segment.get("seg_id", output_id))
        start_frame = int(segment["start_frame"])
        end_frame = int(segment["end_frame"])
        if end_frame <= start_frame:
            raise ValueError(f"Invalid frame interval for segment {seg_id}: {start_frame}-{end_frame}")

        instruction = str(segment.get("instruction", "segment"))
        start_sec = start_frame / fps
        end_sec = end_frame / fps
        clip_name = f"segment_{seg_id:03d}_{start_frame:06d}_{end_frame:06d}_{safe_name(instruction)}.mp4"
        clip_path = args.output_dir / clip_name
        command = [
            args.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y" if args.overwrite else "-n",
            "-ss",
            f"{start_sec:.6f}",
            "-i",
            str(args.video),
            "-t",
            f"{end_sec - start_sec:.6f}",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            str(clip_path),
        ]
        subprocess.run(command, check=True)
        manifest.append(
            {
                "seg_id": seg_id,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_sec": round(start_sec, 6),
                "end_sec": round(end_sec, 6),
                "instruction": instruction,
                "clip": clip_name,
            }
        )
        print(f"exported {output_id + 1}/{len(segments)}: {clip_path}", flush=True)

    (args.output_dir / "manifest.json").write_text(
        json.dumps({"fps": fps, "segments": manifest}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
