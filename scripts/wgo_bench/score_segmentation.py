#!/usr/bin/env python3
"""Score a WGO prediction directory at IoU >= 0.75 with outer snap."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wgo_metrics import IOU_THRESHOLD, json_segments, segment_f1  # noqa: E402


def prediction_path(pred_dir: Path, episode_id: str) -> Path | None:
    candidates = [
        pred_dir / f"{episode_id}.json",
        pred_dir / "homer" / f"{episode_id}.json",
        pred_dir / episode_id / "merge_judge_annotations.json",
    ]
    return next((path for path in candidates if path.exists()), None)


def evaluate_directory(
    gold_dir: Path,
    pred_dir: Path,
    threshold: float = IOU_THRESHOLD,
    outer_snap: bool = True,
) -> dict:
    per_episode = []
    total_match = total_pred = total_gold = 0
    for gold_path in sorted(gold_dir.glob("homer_*.json")):
        gold = json.loads(gold_path.read_text(encoding="utf-8"))
        gold_segments = json_segments(gold)
        pred_path = prediction_path(pred_dir, gold_path.stem)
        if pred_path is None:
            metrics = {
                "n_pred": 0,
                "n_gold": len(gold_segments),
                "n_match": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "pairs": [],
                "error": "missing_pred",
            }
        else:
            pred = json.loads(pred_path.read_text(encoding="utf-8"))
            pred_segments = json_segments(pred)
            metrics = segment_f1(pred_segments, gold_segments, threshold, outer_snap)
            for pair in metrics["pairs"]:
                pair["gold_label"] = gold_segments[pair["gold_j"]].label
                pair["pred_label"] = pred_segments[pair["pred_i"]].label
            metrics["pred_file"] = str(pred_path)
        metrics["id"] = gold_path.stem
        metrics["instruction"] = str(gold.get("instruction") or "")
        per_episode.append(metrics)
        total_match += int(metrics["n_match"])
        total_pred += int(metrics["n_pred"])
        total_gold += int(metrics["n_gold"])

    precision = total_match / total_pred if total_pred else 0.0
    recall = total_match / total_gold if total_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "metric": f"segment_f1@{threshold:g}_with_outer_snap" if outer_snap else f"segment_f1@{threshold:g}",
        "n_episodes": len(per_episode),
        "micro": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "n_match": total_match,
            "n_pred": total_pred,
            "n_gold": total_gold,
        },
        "macro_f1": sum(row["f1"] for row in per_episode) / len(per_episode) if per_episode else 0.0,
        "per_episode": per_episode,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-dir", type=Path, required=True)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iou-threshold", type=float, default=IOU_THRESHOLD)
    parser.add_argument("--no-outer-snap", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = evaluate_directory(
        args.gold_dir,
        args.pred_dir,
        threshold=args.iou_threshold,
        outer_snap=not args.no_outer_snap,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["micro"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
