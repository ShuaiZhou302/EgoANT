#!/usr/bin/env python3
"""Evaluate predicted subtask segments against gold segments."""

from __future__ import annotations

import argparse
import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


def load_segments(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    return data.get("segments", [])


def interval_iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    a0, a1 = float(a["start_sec"]), float(a["end_sec"])
    b0, b1 = float(b["start_sec"]), float(b["end_sec"])
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def text_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def boundary_f1(gold: list[dict[str, Any]], pred: list[dict[str, Any]], tolerance_sec: float) -> dict[str, float]:
    gold_bounds = [float(s["end_sec"]) for s in gold[:-1]]
    pred_bounds = [float(s["end_sec"]) for s in pred[:-1]]
    used = set()
    tp = 0
    for p in pred_bounds:
        best_idx = None
        best_dist = tolerance_sec + 1.0
        for idx, g in enumerate(gold_bounds):
            if idx in used:
                continue
            dist = abs(p - g)
            if dist <= tolerance_sec and dist < best_dist:
                best_idx = idx
                best_dist = dist
        if best_idx is not None:
            used.add(best_idx)
            tp += 1
    precision = tp / len(pred_bounds) if pred_bounds else (1.0 if not gold_bounds else 0.0)
    recall = tp / len(gold_bounds) if gold_bounds else (1.0 if not pred_bounds else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"boundary_precision": precision, "boundary_recall": recall, "boundary_f1": f1}


def match_segments(gold: list[dict[str, Any]], pred: list[dict[str, Any]], iou_threshold: float) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows = []
    used_pred = set()
    tp = 0
    text_scores = []
    iou_scores = []
    for gold_idx, g in enumerate(gold):
        best_idx = None
        best_iou = 0.0
        for pred_idx, p in enumerate(pred):
            if pred_idx in used_pred:
                continue
            score = interval_iou(g, p)
            if score > best_iou:
                best_idx = pred_idx
                best_iou = score
        match = None
        if best_idx is not None and best_iou >= iou_threshold:
            used_pred.add(best_idx)
            tp += 1
            p = pred[best_idx]
            sim = text_similarity(str(g.get("subtask", "")), str(p.get("subtask", "")))
            text_scores.append(sim)
            iou_scores.append(best_iou)
            match = {
                "pred_idx": best_idx,
                "pred_start_sec": p.get("start_sec"),
                "pred_end_sec": p.get("end_sec"),
                "pred_subtask": p.get("subtask", ""),
                "iou": round(best_iou, 3),
                "text_similarity": round(sim, 3),
            }
        rows.append(
            {
                "gold_idx": gold_idx,
                "gold_start_sec": g.get("start_sec"),
                "gold_end_sec": g.get("end_sec"),
                "gold_subtask": g.get("subtask", ""),
                "match": match,
            }
        )
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    metrics = {
        "segment_precision": precision,
        "segment_recall": recall,
        "segment_f1": f1,
        "mean_matched_iou": sum(iou_scores) / len(iou_scores) if iou_scores else 0.0,
        "mean_text_similarity": sum(text_scores) / len(text_scores) if text_scores else 0.0,
        "gold_count": len(gold),
        "pred_count": len(pred),
        "matched_count": tp,
    }
    return rows, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--boundary-tolerance-sec", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gold = load_segments(args.gold)
    pred = load_segments(args.pred)
    rows, metrics = match_segments(gold, pred, args.iou_threshold)
    metrics.update(boundary_f1(gold, pred, args.boundary_tolerance_sec))
    result = {"metrics": metrics, "matches": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
