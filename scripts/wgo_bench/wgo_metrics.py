#!/usr/bin/env python3
"""WGO-Bench temporal metrics used by the EgoANT HomER experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


IOU_THRESHOLD = 0.75


@dataclass
class Segment:
    start: float
    end: float
    label: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def interval_iou(a: Segment, b: Segment) -> float:
    overlap = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    union = a.duration + b.duration - overlap
    return overlap / union if union > 1e-9 else 0.0


def snap_outer_boundaries(pred: list[Segment], gold: list[Segment]) -> list[Segment]:
    """Align only the first start and final end to the gold episode extent."""
    if not pred or not gold:
        return list(pred)
    snapped = [Segment(item.start, item.end, item.label) for item in pred]
    snapped[0].start = gold[0].start
    snapped[-1].end = gold[-1].end
    for item in snapped:
        item.end = max(item.start, item.end)
    return snapped


def greedy_match(
    pred: list[Segment],
    gold: list[Segment],
    threshold: float = IOU_THRESHOLD,
) -> list[tuple[int, int, float]]:
    """Return descending-IoU one-to-one matches at or above ``threshold``."""
    candidates = [
        (pred_i, gold_i, interval_iou(pred_seg, gold_seg))
        for pred_i, pred_seg in enumerate(pred)
        for gold_i, gold_seg in enumerate(gold)
    ]
    candidates.sort(key=lambda item: item[2], reverse=True)
    used_pred: set[int] = set()
    used_gold: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for pred_i, gold_i, score in candidates:
        if score < threshold:
            break
        if pred_i in used_pred or gold_i in used_gold:
            continue
        used_pred.add(pred_i)
        used_gold.add(gold_i)
        matches.append((pred_i, gold_i, score))
    return matches


def segment_f1(
    pred: list[Segment],
    gold: list[Segment],
    threshold: float = IOU_THRESHOLD,
    outer_snap: bool = True,
) -> dict[str, Any]:
    evaluated_pred = snap_outer_boundaries(pred, gold) if outer_snap else list(pred)
    matches = greedy_match(evaluated_pred, gold, threshold)
    precision = len(matches) / len(evaluated_pred) if evaluated_pred else 0.0
    recall = len(matches) / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n_pred": len(evaluated_pred),
        "n_gold": len(gold),
        "n_match": len(matches),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pairs": [
            {"pred_i": pred_i, "gold_j": gold_i, "iou": score}
            for pred_i, gold_i, score in matches
        ],
    }


def json_segments(annotation: dict[str, Any]) -> list[Segment]:
    """Read WGO-style seconds or EgoANT frame-indexed segments."""
    fps = float(annotation.get("fps") or 30.0)
    result = []
    for item in annotation.get("segments") or []:
        if "start_frame" in item and "end_frame" in item:
            start = float(item["start_frame"]) / fps
            end = float(item["end_frame"]) / fps
        else:
            start = float(item.get("start_sec", item.get("start", 0.0)))
            end = float(item.get("end_sec", item.get("end", 0.0)))
        if end <= start:
            continue
        label = str(item.get("subtask") or item.get("instruction") or item.get("label") or "")
        result.append(Segment(start, end, label))
    return result
