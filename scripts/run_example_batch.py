#!/usr/bin/env python3
"""Annotate several EgoVerse examples and summarize against gold labels."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/example_batch"


def json_load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def json_dump(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def discover_examples(example_root: Path, limit: int | None) -> list[tuple[str, Path, Path]]:
    videos = sorted((example_root / "video").glob("*/*.mp4"))
    examples: list[tuple[str, Path, Path]] = []
    for video in videos:
        task = video.parent.name
        sample_id = video.stem
        gold = example_root / "annotation" / task / f"{sample_id}.json"
        if gold.exists():
            examples.append((f"{task}_{sample_id}", video, gold))
    return examples[:limit] if limit is not None else examples


def run_command(cmd: list[str], dry_run: bool) -> None:
    print(" ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def frame_overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def text_sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def summarize_one(sample_name: str, output_dir: Path) -> dict[str, Any]:
    pred = json_load(output_dir / "annotations.json")
    gold = json_load(output_dir / "gold_reference.json")
    pred_segments = pred.get("segments", [])
    gold_segments = gold.get("segments", [])

    rows = []
    for g in gold_segments:
        best = None
        best_score = -1.0
        for p in pred_segments:
            overlap = frame_overlap(g["start_frame"], g["end_frame"], p["start_frame"], p["end_frame"])
            if overlap <= 0:
                continue
            union = max(g["end_frame"], p["end_frame"]) - min(g["start_frame"], p["start_frame"])
            iou = overlap / union if union > 0 else 0.0
            sim = text_sim(g.get("instruction", ""), p.get("instruction", ""))
            score = iou + 0.25 * sim
            if score > best_score:
                best_score = score
                best = (p, iou, sim)
        rows.append(
            {
                "gold_segment": {
                    "start_frame": g["start_frame"],
                    "end_frame": g["end_frame"],
                    "instruction": g.get("instruction", ""),
                },
                "best_prediction": None
                if best is None
                else {
                    "seg_id": best[0]["seg_id"],
                    "start_frame": best[0]["start_frame"],
                    "end_frame": best[0]["end_frame"],
                    "instruction": best[0].get("instruction", ""),
                    "object": best[0].get("object", ""),
                    "confidence": best[0].get("confidence", 0.0),
                    "frame_iou": round(best[1], 3),
                    "text_similarity": round(best[2], 3),
                },
            }
        )

    return {
        "sample": sample_name,
        "output_dir": str(output_dir),
        "gold_segments": len(gold_segments),
        "pred_segments": len(pred_segments),
        "non_na_predictions": sum(1 for p in pred_segments if p.get("instruction") != "N/A"),
        "rows": rows,
    }


def write_markdown(path: Path, summaries: list[dict[str, Any]]) -> None:
    lines = ["# EgoANT Batch Annotation Summary", ""]
    for item in summaries:
        lines.append(f"## {item['sample']}")
        lines.append("")
        lines.append(
            f"gold segments: {item['gold_segments']} | pred segments: {item['pred_segments']} | "
            f"non-N/A: {item['non_na_predictions']}"
        )
        lines.append("")
        lines.append("| gold | best prediction | IoU | sim |")
        lines.append("|---|---|---:|---:|")
        for row in item["rows"]:
            gold = row["gold_segment"]["instruction"]
            pred = row["best_prediction"]
            if pred is None:
                lines.append(f"| {gold} | - | 0 | 0 |")
            else:
                lines.append(
                    f"| {gold} | {pred['instruction']} | {pred['frame_iou']} | {pred['text_similarity']} |"
                )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EgoANT annotation on example videos.")
    parser.add_argument("--example-root", type=Path, required=True, help="User-provided video/annotation example tree.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--reconstruction-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--fov-x", type=float, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--api-retries", type=int, default=2)
    parser.add_argument(
        "--caption-input",
        choices=("overlay", "raw", "raw-hand-crop", "temporal-collage"),
        default="raw",
    )
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--force", action="store_true", help="Remove existing sample outputs and rerun every step.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    examples = discover_examples(args.example_root, args.limit)
    if not examples:
        raise FileNotFoundError(f"No examples found under {args.example_root}")

    for sample_name, video, gold in examples:
        out_dir = args.output_root / sample_name
        hand_recon = out_dir / "hand_recon.json"
        annotations = out_dir / "annotations.json"
        if args.force and out_dir.exists():
            print(f"remove existing {sample_name}", flush=True)
            if not args.dry_run:
                shutil.rmtree(out_dir)
        if args.skip_existing and annotations.exists():
            print(f"skip existing {sample_name}", flush=True)
            continue

        export_cmd = [
            sys.executable,
            "scripts/export_hand_reconstruction.py",
            "--reconstruction-root",
            str(args.reconstruction_root),
            "--model-root",
            str(args.model_root),
            "--video",
            str(video),
            "--output",
            str(hand_recon),
            "--device",
            args.device,
        ]
        if args.fov_x is not None:
            export_cmd.extend(["--fov-x", str(args.fov_x)])
        if not hand_recon.exists():
            run_command(export_cmd, args.dry_run)

        annotate_cmd = [
            sys.executable,
            "scripts/annotate_video.py",
            "--video",
            str(video),
            "--gold",
            str(gold),
            "--output-dir",
            str(out_dir),
            "--hand-recon",
            str(hand_recon),
            "--api-retries",
            str(args.api_retries),
            "--caption-input",
            args.caption_input,
        ]
        if args.base_url:
            annotate_cmd.extend(["--base-url", args.base_url])
        if args.model:
            annotate_cmd.extend(["--model", args.model])
        run_command(annotate_cmd, args.dry_run)

    if args.dry_run:
        return 0

    summaries = []
    for sample_name, _, _ in examples:
        out_dir = args.output_root / sample_name
        if (out_dir / "annotations.json").exists():
            summaries.append(summarize_one(sample_name, out_dir))
    json_dump(args.output_root / "batch_summary.json", summaries)
    write_markdown(args.output_root / "batch_summary.md", summaries)
    print(args.output_root / "batch_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
