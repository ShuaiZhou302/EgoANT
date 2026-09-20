#!/usr/bin/env python3
"""Experiment wrapper: WGO labeling with true EgoANT hand crop grids.

This imports scripts/wgo_bench/label_gold_segments.py and overrides only the
raw-hand-crop image builder. Unlike the original WGO experiment path, this reads
work_root/<id>/hand_recon.json and crops around projected hand points.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent))
import label_gold_segments as base  # noqa: E402

_ORIG_BUILD = base.build_images_for_segment


def hand_points_for_frame(hand_recon: dict[str, Any] | None, frame_idx: int) -> np.ndarray:
    if not hand_recon:
        return np.empty((0, 2), dtype=float)
    points: list[np.ndarray] = []
    for hand in ("left", "right"):
        info = (hand_recon.get("hands") or {}).get(hand)
        if not info:
            continue
        valid = np.asarray(info.get("valid", []), dtype=bool)
        if frame_idx >= len(valid) or not valid[frame_idx]:
            continue
        joints_xy = info.get("joints_xy")
        if joints_xy is not None:
            joints = np.asarray(joints_xy, dtype=float)
            if frame_idx < len(joints):
                frame_joints = joints[frame_idx]
                good = np.isfinite(frame_joints).all(axis=1)
                if good.any():
                    points.append(frame_joints[good])
                    continue
        wrist_xy = np.asarray(info.get("wrist_xy", []), dtype=float)
        if frame_idx < len(wrist_xy) and np.isfinite(wrist_xy[frame_idx]).all():
            points.append(wrist_xy[frame_idx][None, :])
    if not points:
        return np.empty((0, 2), dtype=float)
    return np.concatenate(points, axis=0)


def crop_hand_region(frame: np.ndarray, points: np.ndarray) -> np.ndarray | None:
    if points.size == 0:
        return None
    h, w = frame.shape[:2]
    finite = points[np.isfinite(points).all(axis=1)]
    finite = finite[(finite[:, 0] >= 0) & (finite[:, 0] < w) & (finite[:, 1] >= 0) & (finite[:, 1] < h)]
    if len(finite) == 0:
        return None
    x0, y0 = finite.min(axis=0)
    x1, y1 = finite.max(axis=0)
    cx = float((x0 + x1) / 2.0)
    cy = float((y0 + y1) / 2.0)
    bbox_side = max(float(x1 - x0), float(y1 - y0), 1.0)
    short = min(w, h)
    side = max(bbox_side * 2.4, short * 0.35)
    side = min(side, short * 0.75)
    xx0 = max(0, int(round(cx - side / 2.0)))
    yy0 = max(0, int(round(cy - side / 2.0)))
    xx1 = min(w, int(round(cx + side / 2.0)))
    yy1 = min(h, int(round(cy + side / 2.0)))
    if xx1 <= xx0 or yy1 <= yy0:
        return None
    return frame[yy0:yy1, xx0:xx1]


def build_images_for_segment_true_handcrop(cap, fps, nframes, segs, seg_i, caption_input, hand_recon):
    if caption_input != "raw-hand-crop":
        return _ORIG_BUILD(cap, fps, nframes, segs, seg_i, caption_input, hand_recon)

    seg = segs[seg_i]
    start = float(seg["start_sec"])
    end = float(seg["end_sec"])
    frames, idxs = base.load_segment_frames(cap, fps, nframes, start, end, 5)
    imgs = [base.encode_bgr(f) for f in frames]

    crops = []
    used_true = 0
    for fr, idx in zip(frames[:6], idxs[:6]):
        crop = crop_hand_region(fr, hand_points_for_frame(hand_recon, idx))
        if crop is None:
            crop = fr
        else:
            used_true += 1
        crops.append(crop)
    if not crops:
        crops = frames[:1]
    imgs.append(base.encode_bgr(base.make_collage(crops, 6), max_side=960))
    mode = f"true-hand-crop:{used_true}/{len(crops)}"
    return imgs, mode


base.build_images_for_segment = build_images_for_segment_true_handcrop

if __name__ == "__main__":
    base.main()
