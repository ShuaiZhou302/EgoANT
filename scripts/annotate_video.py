#!/usr/bin/env python3
"""Annotate one egocentric video with the EgoANT pipeline.

This script expects either a reconstruction episode file or the normalized
JSON produced by `export_hand_reconstruction.py`. It does not hide missing
reconstruction by generating fake trajectories.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from api_usage import append_usage, summarize_usage, usage_event

os.environ.setdefault("MPLCONFIGDIR", "/tmp/egoant_matplotlib")

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm


DEFAULT_OUTPUT = Path("outputs/annotation/run")
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "qwen3.5-397b-a17b"
FORMAT_RETRY_INSTRUCTION = (
    "Your previous response was invalid. Return only valid JSON matching the schema exactly. "
    "Do not include markdown or extra text."
)
TRANSIENT_API_ERRORS = (
    "connection refused",
    "connection reset",
    "connection aborted",
    "failed to establish a new connection",
)
REQUIRED_ANNOTATION_FIELDS = {
    "instruction",
    "alternatives",
    "actor",
    "verb",
    "main_object",
    "involved_objects",
    "contact",
    "final_state",
    "confidence",
    "needs_review",
}

SYSTEM_PROMPT = "You are a precise video-frame annotator for egocentric human-object interaction clips."

OVERLAY_PROMPT_INTRO = """I will send you 10 ordered frames sampled from one candidate atomic action clip.

The frames come from an egocentric manipulation video. A 2D hand trajectory is overlaid on the frames. The trajectory color shows temporal progress. Respect the frame order."""

RAW_PROMPT_INTRO = """I will send you 10 ordered raw frames sampled uniformly from one candidate atomic action clip.

The frames come from an egocentric manipulation video and are ordered from earliest to latest. Respect the frame order."""

RAW_HAND_CROP_PROMPT_INTRO = """I will send you 10 ordered raw frames sampled uniformly from one candidate atomic action clip, followed by one hand-focused crop grid from the same clip.

The raw frames come from an egocentric manipulation video and are ordered from earliest to latest. The final image is a hand-focused grid from the same current clip. Use it to identify the manipulated object and visible hand-object contact, but write the instruction from the whole clip. Respect the frame order."""

TEMPORAL_COLLAGE_PROMPT_INTRO = """I will send you three ordered frame grids from an egocentric manipulation video: PAST context, CURRENT target, and FUTURE context. Each grid contains 6 raw frames sampled uniformly within that segment and ordered from earliest to latest in row-major order.

Write the instruction for the visible physical action in the CURRENT target grid only. Use the PAST and FUTURE grids only to disambiguate the CURRENT action and its direction. Do not describe an action merely because it appears in the context grids."""

PROMPT_BODY = """

Write one short imperative instruction for the visible physical action.

Focus on:
1. active hand or hands;
2. the main object being acted on or directly manipulated;
3. other involved objects, such as a container, target, tool, or support;
4. concrete action verb;
5. visible contact or grasp;
6. visible final state or final location.

Rules:
- Describe only visible facts from this clip.
- Do not use hidden intent or scene context to guess the action.
- The instruction must be an action command. Do not output prompt or meta text
  such as "Describe the action", "the clip shows", or "in this clip".
- Do not mention the reference image, frames, camera, trajectory overlay, or
  any meta notes.
- Do not address the viewer or use second-person words such as "you" or
  "your". Refer to the left hand, right hand, or both hands directly.
- Do not number items, do not add commentary, and do not use code fences.
- Name an object only when it is visibly involved in the action: touched by the hand, approached by the hand with clear intent, manipulated by a held tool, or visibly affected by a held tool.
- Before naming main_object, check whether its size is plausible for the
  visible container, tool, hand grasp, and motion.
- If an item is moved into or out of a container, main_object is the moved item;
  put the container in involved_objects.
- If a held tool acts on another item, main_object is the affected item; put the
  tool in involved_objects.
- Use a basic object noun for main_object.name.
- Do not include color, size, shape, material, or variety adjectives in main_object.name.
- If unsure of the exact object name, use a generic noun such as 'object', 'block', 'handle', or 'container'.
- If several objects participate, list them in involved_objects with short roles. Keep at most 4 objects.
- Avoid vague verbs. Prefer concrete verbs: approach, align, open the hand, close the hand, grasp, rotate, tilt, release, lift, lower, move forward/backward/left/right/up/down, push, pull, slide, insert, retract, place, hold.
- Do not use "transfer" in instruction, alternatives, or verb. Use the visible
  concrete action instead, such as move, lift, place, insert, remove, or pour.
- Return 3 to 5 alternatives. Alternatives must preserve the same meaning and must not add new facts.

Return only valid JSON:
{
  "instruction": "canonical one-sentence imperative instruction",
  "alternatives": ["3 to 5 equivalent descriptions"],
  "actor": "left hand | right hand | both hands | unknown",
  "verb": "main visible action verb",
  "main_object": {
    "name": "basic noun for the main object being acted on | unknown",
    "name_candidates": [{"name": "possible object name", "confidence": 0.0}],
    "confidence": 0.0
  },
  "involved_objects": [
    {"name": "basic noun", "role": "primary | container | target | tool | support | other"}
  ],
  "contact": "visible contact or grasp | unknown",
  "final_state": "visible final state/location | unknown",
  "confidence": 0.0,
  "needs_review": false
}
"""

CAPTION_PROMPTS = {
    "overlay": OVERLAY_PROMPT_INTRO + PROMPT_BODY,
    "raw": RAW_PROMPT_INTRO + PROMPT_BODY.replace("trajectory overlay, or\n  ", ""),
    "raw-hand-crop": RAW_HAND_CROP_PROMPT_INTRO + PROMPT_BODY.replace("trajectory overlay, or\n  ", ""),
    "temporal-collage": TEMPORAL_COLLAGE_PROMPT_INTRO + PROMPT_BODY.replace("trajectory overlay, or\n  ", ""),
}
USER_PROMPT = CAPTION_PROMPTS["raw"]


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    nframes: int
    width: int
    height: int


def json_dump(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def read_video_info(video_path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return VideoInfo(fps=fps, nframes=nframes, width=width, height=height)


def load_video_frames(video_path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def project_points(points_cam: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    z = np.clip(points_cam[:, 2], 1e-6, None)
    x = intrinsics[0, 0] * (points_cam[:, 0] / z) + intrinsics[0, 2]
    y = intrinsics[1, 1] * (points_cam[:, 1] / z) + intrinsics[1, 2]
    return np.stack([x, y], axis=1)


def load_reconstruction_episode(path: Path, nframes: int) -> dict[str, Any]:
    episode = np.load(path, allow_pickle=True).item()
    intrinsics = episode.get("intrinsics")
    out: dict[str, Any] = {
        "source": str(path),
        "source_type": "reconstruction_episode_npy",
        "camera": {
            "intrinsics": intrinsics.tolist() if isinstance(intrinsics, np.ndarray) else None,
            "extrinsics_shape": list(episode["extrinsics"].shape) if isinstance(episode.get("extrinsics"), np.ndarray) else None,
        },
        "hands": {},
    }
    for hand in ("left", "right"):
        hand_info = episode.get(hand) or {}
        valid = np.asarray(hand_info.get("kept_frames", np.zeros(nframes)), dtype=bool)
        world = hand_info.get("transl_worldspace")
        if world is None:
            joints_world = hand_info.get("joints_worldspace")
            world = joints_world[:, 0, :] if isinstance(joints_world, np.ndarray) else None
        xy = None
        joints_cam = hand_info.get("joints_camspace")
        joints_xy = None
        if isinstance(joints_cam, np.ndarray) and isinstance(intrinsics, np.ndarray):
            xy = project_points(joints_cam[:, 0, :], intrinsics)
            if joints_cam.ndim == 3:
                joints_xy = np.stack([project_points(joints_cam[:, joint_idx, :], intrinsics) for joint_idx in range(joints_cam.shape[1])], axis=1)
        out["hands"][hand] = normalize_hand_track(world, xy, valid, nframes, joints_xy=joints_xy)
    return out


def normalize_hand_track(
    wrist_world: Any,
    wrist_xy: Any,
    valid: Any,
    nframes: int,
    joints_xy: Any = None,
) -> dict[str, Any]:
    def fit_array(value: Any, dims: int) -> np.ndarray:
        arr = np.full((nframes, dims), np.nan, dtype=float)
        if value is None:
            return arr
        raw = np.asarray(value, dtype=float)
        count = min(nframes, raw.shape[0])
        arr[:count] = raw[:count, :dims]
        return arr

    valid_arr = np.zeros(nframes, dtype=bool)
    if valid is not None:
        raw_valid = np.asarray(valid, dtype=bool)
        valid_arr[: min(nframes, raw_valid.shape[0])] = raw_valid[:nframes]

    world_arr = fit_array(wrist_world, 3)
    xy_arr = fit_array(wrist_xy, 2)
    valid_arr &= np.isfinite(world_arr).all(axis=1)
    out = {
        "valid": valid_arr.tolist(),
        "wrist_world": world_arr.tolist(),
        "wrist_xy": xy_arr.tolist(),
    }
    if joints_xy is not None:
        raw_joints = np.asarray(joints_xy, dtype=float)
        if raw_joints.ndim == 3 and raw_joints.shape[2] >= 2:
            count = min(nframes, raw_joints.shape[0])
            joint_count = raw_joints.shape[1]
            joints_arr = np.full((nframes, joint_count, 2), np.nan, dtype=float)
            joints_arr[:count] = raw_joints[:count, :, :2]
            out["joints_xy"] = joints_arr.tolist()
    return out


def load_hand_recon(args: argparse.Namespace, nframes: int) -> dict[str, Any]:
    if args.hand_recon:
        with Path(args.hand_recon).open("r", encoding="utf-8") as f:
            return json.load(f)
    if args.reconstruction_episode:
        return load_reconstruction_episode(Path(args.reconstruction_episode), nframes)
    raise FileNotFoundError(
        "Missing hand reconstruction. Provide --reconstruction-episode PATH_TO_EPISODE.npy "
        "or --hand-recon PATH_TO_hand_recon.json."
    )


def interpolate_track(track: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = track.copy()
    idx = np.arange(len(track))
    for dim in range(track.shape[1]):
        good = valid & np.isfinite(track[:, dim])
        if good.sum() == 0:
            out[:, dim] = np.nan
        elif good.sum() == 1:
            out[:, dim] = track[good, dim][0]
        else:
            out[:, dim] = np.interp(idx, idx[good], track[good, dim])
    return out


def compute_speed(hand: dict[str, Any], fps: float, sigma: float) -> np.ndarray:
    world = np.asarray(hand["wrist_world"], dtype=float)
    valid = np.asarray(hand["valid"], dtype=bool)
    if valid.sum() < 2:
        return np.full(world.shape[0], np.nan)
    interp = interpolate_track(world, valid)
    smooth = gaussian_filter1d(interp, sigma=sigma, axis=0, mode="nearest")
    delta = np.linalg.norm(np.diff(smooth, axis=0), axis=1) * fps
    speed = np.concatenate([[np.nan], delta])
    speed[~valid] = np.nan
    return speed


def find_speed_minima(speed: np.ndarray, fps: float) -> list[int]:
    radius = max(1, int(round(0.5 * fps)) // 2)
    cuts: list[int] = []
    for i in range(radius, len(speed) - radius):
        if not np.isfinite(speed[i]):
            continue
        window = speed[i - radius : i + radius + 1]
        if not np.isfinite(window).any():
            continue
        if speed[i] <= np.nanmin(window):
            cuts.append(i)
    return cuts


def merge_cut_points(cuts: list[int], fps: float, nframes: int) -> list[int]:
    min_gap = max(1, int(round(0.25 * fps)))
    merged: list[int] = []
    for cut in sorted(set(c for c in cuts if 0 < c < nframes)):
        if not merged or cut - merged[-1] >= min_gap:
            merged.append(cut)
        else:
            merged[-1] = int(round((merged[-1] + cut) / 2))
    return merged


def segmentation_diagnostics(
    cuts_by_hand: dict[str, list[int]],
    merged_cuts_by_hand: dict[str, list[int]],
    global_cuts: list[int],
    segments: list[dict[str, int]],
    fps: float,
    sigma_frames: float,
    sigma_sec: float,
    cut_source: str,
    min_segment_sec: float,
    min_segment_frames: int,
    short_segment_policy: str,
) -> dict[str, Any]:
    return {
        "fps": fps,
        "smooth_sigma_frames": sigma_frames,
        "smooth_sigma_sec": sigma_sec,
        "cut_source": cut_source,
        "minima_window_sec": 0.5,
        "merge_gap_sec": 0.25,
        "min_segment_sec": min_segment_sec,
        "min_segment_frames": min_segment_frames,
        "short_segment_policy": short_segment_policy,
        "cuts_by_hand_count": {hand: len(cuts) for hand, cuts in cuts_by_hand.items()},
        "merged_cuts_by_hand_count": {hand: len(cuts) for hand, cuts in merged_cuts_by_hand.items()},
        "global_union_cut_count": len(global_cuts),
        "segment_count": len(segments),
        "note": (
            "The source method describes left/right hand segmentation independently. "
            "This wrapper currently unions left/right cut points into one global timeline, "
            "which can increase over-segmentation when both hands move asynchronously."
        ),
    }


def build_segments(
    cuts: list[int],
    nframes: int,
    fps: float,
    min_sec: float,
    short_policy: str = "coalesce",
) -> list[dict[str, int]]:
    points = [0] + cuts + [nframes]
    min_len = max(1, int(round(min_sec * fps)))
    raw_segments = [
        {"start_frame": int(start), "end_frame": int(end)}
        for start, end in zip(points[:-1], points[1:])
        if end > start
    ]

    if short_policy == "merge-runs":
        segments: list[dict[str, int]] = []
        i = 0
        while i < len(raw_segments):
            current = raw_segments[i]
            current_len = current["end_frame"] - current["start_frame"]
            if current_len >= min_len:
                segments.append(current.copy())
                i += 1
                continue

            run_start = current["start_frame"]
            run_end = current["end_frame"]
            j = i + 1
            while j < len(raw_segments) and run_end - run_start < min_len:
                next_seg = raw_segments[j]
                next_len = next_seg["end_frame"] - next_seg["start_frame"]
                if next_len >= min_len:
                    break
                run_end = next_seg["end_frame"]
                j += 1

            if run_end - run_start >= min_len:
                segments.append({"start_frame": run_start, "end_frame": run_end})
                i = j
            elif segments:
                segments[-1]["end_frame"] = run_end
                i = j
            elif j < len(raw_segments):
                segments.append({"start_frame": run_start, "end_frame": raw_segments[j]["end_frame"]})
                i = j + 1
            else:
                segments.append({"start_frame": run_start, "end_frame": run_end})
                i = j

        for seg_id, segment in enumerate(segments):
            segment["seg_id"] = seg_id
        return segments

    segments: list[dict[str, int]] = []
    start = points[0]
    for end in points[1:]:
        if end - start < min_len and end != nframes:
            continue
        if end == nframes and end - start < min_len and segments:
            segments[-1]["end_frame"] = int(end)
        else:
            segments.append({"seg_id": len(segments), "start_frame": int(start), "end_frame": int(end)})
        start = end
    return segments


def plot_speed(speed_by_hand: dict[str, np.ndarray], cuts: list[int], gold: dict[str, Any] | None, out_path: Path) -> None:
    plt.figure(figsize=(14, 5))
    for hand, speed in speed_by_hand.items():
        plt.plot(speed, label=f"{hand} wrist speed")
    for cut in cuts:
        plt.axvline(cut, color="black", alpha=0.25, linewidth=1)
    if gold:
        for seg in gold.get("segments", []):
            plt.axvline(seg["start_frame"], color="green", alpha=0.18, linestyle="--")
    plt.xlabel("Frame")
    plt.ylabel("Wrist speed")
    plt.title("Wrist speed minima")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def sample_frame_indices(start: int, end: int, count: int) -> list[int]:
    end_exclusive = max(start + 1, end)
    if end_exclusive - start <= count:
        return list(range(start, end_exclusive))
    return np.linspace(start, end_exclusive - 1, count).round().astype(int).tolist()


def trajectory_color(t: float) -> tuple[int, int, int]:
    # BGR: blue -> green -> red.
    t = float(np.clip(t, 0.0, 1.0))
    if t < 0.5:
        a = t / 0.5
        return (int(255 * (1 - a)), int(255 * a), 0)
    a = (t - 0.5) / 0.5
    return (0, int(255 * (1 - a)), int(255 * a))


def draw_overlay(frame: np.ndarray, hand_recon: dict[str, Any], start: int, end: int, current_idx: int) -> np.ndarray:
    img = frame.copy()
    for hand in ("left", "right"):
        hand_info = hand_recon.get("hands", {}).get(hand)
        if not hand_info:
            continue
        xy = np.asarray(hand_info["wrist_xy"], dtype=float)
        valid = np.asarray(hand_info["valid"], dtype=bool)
        pts: list[tuple[int, int, int]] = []
        for idx in range(start, min(end, len(xy))):
            if valid[idx] and np.isfinite(xy[idx]).all():
                pts.append((idx, int(round(xy[idx, 0])), int(round(xy[idx, 1]))))
        for p0, p1 in zip(pts[:-1], pts[1:]):
            t = (p1[0] - start) / max(1, end - start - 1)
            cv2.line(img, (p0[1], p0[2]), (p1[1], p1[2]), trajectory_color(t), 2, cv2.LINE_AA)
        if 0 <= current_idx < len(xy) and valid[current_idx] and np.isfinite(xy[current_idx]).all():
            label = "L" if hand == "left" else "R"
            x, y = int(round(xy[current_idx, 0])), int(round(xy[current_idx, 1]))
            cv2.circle(img, (x, y), 5, (255, 0, 0), -1, cv2.LINE_AA)
            cv2.putText(img, label, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(img, f"Frame {current_idx}", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img


def save_segment_frames(
    frames: list[np.ndarray],
    hand_recon: dict[str, Any],
    segment: dict[str, int],
    out_dir: Path,
    count: int,
    use_overlay: bool = True,
) -> list[Path]:
    seg_id = segment["seg_id"]
    raw_dir = out_dir / f"segment_{seg_id:03d}_frames_raw"
    overlay_dir = out_dir / f"segment_{seg_id:03d}_frames_overlay"
    raw_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    image_paths: list[Path] = []
    for order, frame_idx in enumerate(sample_frame_indices(segment["start_frame"], segment["end_frame"], count), start=1):
        raw_path = raw_dir / f"frame_{order:02d}_{frame_idx:06d}.png"
        overlay_path = overlay_dir / f"frame_{order:02d}_{frame_idx:06d}.png"
        cv2.imwrite(str(raw_path), frames[frame_idx])
        overlay = draw_overlay(frames[frame_idx], hand_recon, segment["start_frame"], segment["end_frame"], frame_idx)
        cv2.imwrite(str(overlay_path), overlay)
        image_paths.append(overlay_path if use_overlay else raw_path)
    return image_paths


def save_segment_collage(
    frames: list[np.ndarray],
    segment: dict[str, int] | None,
    path: Path,
    count: int = 6,
) -> Path:
    tile_width, tile_height = 320, 180
    canvas = np.full((tile_height * 2, tile_width * 3, 3), 32, dtype=np.uint8)
    if segment is None:
        cv2.putText(canvas, "NO CONTEXT", (365, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (210, 210, 210), 2)
    else:
        indices = sample_frame_indices(segment["start_frame"], segment["end_frame"], count)
        for order, frame_idx in enumerate(indices):
            tile = frames[frame_idx]
            height, width = tile.shape[:2]
            scale = min(tile_width / width, tile_height / height)
            resized = cv2.resize(tile, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
            row, col = divmod(order, 3)
            y0 = row * tile_height + (tile_height - resized.shape[0]) // 2
            x0 = col * tile_width + (tile_width - resized.shape[1]) // 2
            canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"Cannot save collage: {path}")
    return path


def hand_points_for_frame(hand_recon: dict[str, Any], frame_idx: int) -> np.ndarray:
    points: list[np.ndarray] = []
    for hand in ("left", "right"):
        hand_info = hand_recon.get("hands", {}).get(hand)
        if not hand_info:
            continue
        valid = np.asarray(hand_info.get("valid", []), dtype=bool)
        if frame_idx >= len(valid) or not valid[frame_idx]:
            continue
        joints_xy = hand_info.get("joints_xy")
        if joints_xy is not None:
            joints = np.asarray(joints_xy, dtype=float)
            if frame_idx < len(joints):
                frame_joints = joints[frame_idx]
                good = np.isfinite(frame_joints).all(axis=1)
                if good.any():
                    points.append(frame_joints[good])
                    continue
        wrist_xy = np.asarray(hand_info.get("wrist_xy", []), dtype=float)
        if frame_idx < len(wrist_xy) and np.isfinite(wrist_xy[frame_idx]).all():
            points.append(wrist_xy[frame_idx][None, :])
    if not points:
        return np.empty((0, 2), dtype=float)
    return np.concatenate(points, axis=0)


def crop_hand_region(frame: np.ndarray, points: np.ndarray) -> np.ndarray | None:
    if points.size == 0:
        return None
    height, width = frame.shape[:2]
    finite = points[np.isfinite(points).all(axis=1)]
    finite = finite[(finite[:, 0] >= 0) & (finite[:, 0] < width) & (finite[:, 1] >= 0) & (finite[:, 1] < height)]
    if len(finite) == 0:
        return None

    x0, y0 = finite.min(axis=0)
    x1, y1 = finite.max(axis=0)
    center_x = float((x0 + x1) / 2.0)
    center_y = float((y0 + y1) / 2.0)
    bbox_side = max(float(x1 - x0), float(y1 - y0), 1.0)
    short_side = min(width, height)
    side = max(bbox_side * 2.4, short_side * 0.35)
    side = min(side, short_side * 0.75)

    x0 = int(round(center_x - side / 2.0))
    y0 = int(round(center_y - side / 2.0))
    x1 = int(round(center_x + side / 2.0))
    y1 = int(round(center_y + side / 2.0))
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(width, x1)
    y1 = min(height, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return frame[y0:y1, x0:x1]


def save_segment_hand_crop_grid(
    frames: list[np.ndarray],
    hand_recon: dict[str, Any],
    segment: dict[str, int],
    path: Path,
    count: int = 6,
) -> Path:
    tile_width, tile_height = 384, 288
    canvas = np.full((tile_height * 2, tile_width * 3, 3), 24, dtype=np.uint8)
    for order, frame_idx in enumerate(sample_frame_indices(segment["start_frame"], segment["end_frame"], count)):
        crop = crop_hand_region(frames[frame_idx], hand_points_for_frame(hand_recon, frame_idx))
        if crop is None:
            crop = frames[frame_idx]
        height, width = crop.shape[:2]
        scale = min(tile_width / width, tile_height / height)
        resized = cv2.resize(crop, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
        row, col = divmod(order, 3)
        y0 = row * tile_height + (tile_height - resized.shape[0]) // 2
        x0 = col * tile_width + (tile_width - resized.shape[1]) // 2
        canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"Cannot save hand crop grid: {path}")
    return path


def image_to_data_url(path: Path, max_side: int | None = None, jpeg_quality: int = 85) -> str:
    if max_side is None or max_side <= 0:
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{data}"

    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Cannot read image for API payload: {path}")
    height, width = image.shape[:2]
    scale = min(1.0, float(max_side) / max(height, width))
    if scale < 1.0:
        image = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    quality = int(np.clip(jpeg_quality, 1, 100))
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError(f"Cannot encode image for API payload: {path}")
    data = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{data}"


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    elif not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    return json.loads(text)


def validate_annotation(data: dict[str, Any]) -> dict[str, Any]:
    def normalize_action_text(value: Any) -> str:
        text = str(value)
        def replace_word(match: re.Match[str], replacement: str) -> str:
            return replacement.capitalize() if match.group(0)[0].isupper() else replacement

        text = re.sub(r"\byour\b", lambda match: replace_word(match, "the"), text, flags=re.IGNORECASE)
        substitutions = (
            (r"\btransferring\b", "moving"),
            (r"\btransferred\b", "moved"),
            (r"\btransfers\b", "moves"),
            (r"\btransfer\b", "move"),
        )
        for pattern, replacement in substitutions:
            text = re.sub(pattern, lambda match, word=replacement: replace_word(match, word), text, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", text).strip()

    def as_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def normalize_candidates(value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        return [
            {
                "name": str(candidate.get("name", "unknown")),
                "confidence": as_float(candidate.get("confidence", 0.0)),
            }
            for candidate in value[:3]
            if isinstance(candidate, dict)
        ]

    def normalize_main_object(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            name = str(value.get("name", "unknown"))
            candidates = normalize_candidates(value.get("name_candidates", []))
            confidence = as_float(value.get("confidence", 0.0))
        else:
            name = str(value or "unknown")
            candidates = []
            confidence = 0.0
        return {"name": name, "name_candidates": candidates, "confidence": confidence}

    def normalize_involved_objects(value: Any) -> list[dict[str, str]]:
        if not isinstance(value, list):
            return []
        out = []
        for item in value[:4]:
            if not isinstance(item, dict):
                continue
            out.append(
                {
                    "name": str(item.get("name", "unknown")),
                    "role": str(item.get("role", "other")),
                }
            )
        return out

    defaults = {
        "instruction": "N/A",
        "alternatives": [],
        "actor": "unknown",
        "verb": "N/A",
        "main_object": {"name": "unknown", "name_candidates": [], "confidence": 0.0},
        "involved_objects": [],
        "contact": "unknown",
        "final_state": "unknown",
        "confidence": 0.0,
        "needs_review": True,
    }
    out = {**defaults, **data}
    out["instruction"] = normalize_action_text(out["instruction"])
    out["verb"] = normalize_action_text(out["verb"])
    if not isinstance(out["alternatives"], list):
        out["alternatives"] = []
    out["alternatives"] = [normalize_action_text(x) for x in out["alternatives"][:5]]
    while len(out["alternatives"]) < 5:
        out["alternatives"].append(out["instruction"])
    out["confidence"] = as_float(out["confidence"])
    out["main_object"] = normalize_main_object(out["main_object"])
    out["involved_objects"] = normalize_involved_objects(out["involved_objects"])
    out["needs_review"] = bool(out["needs_review"])
    return out


def require_valid_annotation(data: dict[str, Any]) -> dict[str, Any]:
    missing = sorted(REQUIRED_ANNOTATION_FIELDS - set(data))
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    alternatives = data.get("alternatives")
    if not isinstance(alternatives, list) or not 3 <= len(alternatives) <= 5:
        raise ValueError("Field `alternatives` must be a list with 3 to 5 items")
    instruction = str(data.get("instruction", "")).strip()
    meta_phrases = (
        "describe the",
        "visible hand-object action",
        "the clip shows",
        "in this clip",
        "provided frames",
        "video frames",
    )
    lowered = instruction.lower()
    if any(phrase in lowered for phrase in meta_phrases):
        raise ValueError(f"Instruction looks like prompt/meta text: {instruction!r}")
    return validate_annotation(data)


def failed_annotation() -> dict[str, Any]:
    return validate_annotation({"instruction": "N/A", "needs_review": True, "confidence": 0.0})


def maybe_restart_api_endpoint(error: Exception, attempts_left: bool) -> None:
    command = os.environ.get("API_RESTART_COMMAND", "").strip()
    if not command or not attempts_left:
        return
    text = str(error).lower()
    if not any(marker in text for marker in TRANSIENT_API_ERRORS):
        return
    timeout = float(os.environ.get("API_RESTART_TIMEOUT", "90"))
    subprocess.run(command, shell=True, check=False, timeout=timeout)
    time.sleep(float(os.environ.get("API_RESTART_SLEEP", "2")))


def call_caption_api(
    image_paths: list[Path],
    prompt: str,
    base_url: str,
    model: str,
    api_key_env: str,
    timeout: float,
    max_tokens: int,
    payload_path: Path,
    error_path: Path,
    retry_count: int,
    api_image_max_side: int | None,
    api_image_jpeg_quality: int,
    disable_thinking: bool,
) -> dict[str, Any]:
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key environment variable: {api_key_env}")

    url = f"{base_url.rstrip('/')}/chat/completions"
    errors: list[dict[str, str]] = []
    max_attempts = retry_count + 1
    for attempt_idx in range(max_attempts):
        attempt_prompt = prompt if attempt_idx == 0 else f"{prompt}\n\n{FORMAT_RETRY_INSTRUCTION}"
        content: list[dict[str, Any]] = [{"type": "text", "text": attempt_prompt}]
        content.extend(
            {
                "type": "image_url",
                "image_url": {"url": image_to_data_url(p, api_image_max_side, api_image_jpeg_quality)},
            }
            for p in image_paths
        )
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            "temperature": 0.2,
            "max_tokens": max_tokens,
        }
        if disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        redacted = {
            **payload,
            "messages": payload["messages"][:1]
            + [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": attempt_prompt},
                        {"type": "image_url", "image_url": {"url": "<redacted>"}},
                    ],
                }
            ],
        }
        json_dump(payload_path, redacted)
        started = time.monotonic()
        response_data: dict[str, Any] | None = None
        try:
            response = requests.post(
                url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=timeout,
            )
            if not response.ok:
                raise requests.HTTPError(
                    f"{response.status_code} Client Error for url: {url}; body={response.text[:1000]}",
                    response=response,
                )
            response_data = response.json()
            raw = response_data["choices"][0]["message"]["content"]
            annotation = require_valid_annotation(extract_json(raw))
            append_usage(
                payload_path.parent / "api_usage.jsonl",
                usage_event(
                    response_data,
                    stage="caption",
                    model=model,
                    latency_sec=time.monotonic() - started,
                    success=True,
                    attempt=attempt_idx + 1,
                ),
            )
            return annotation
        except Exception as exc:
            maybe_restart_api_endpoint(exc, attempt_idx < max_attempts - 1)
            append_usage(
                payload_path.parent / "api_usage.jsonl",
                usage_event(
                    response_data,
                    stage="caption",
                    model=model,
                    latency_sec=time.monotonic() - started,
                    success=False,
                    attempt=attempt_idx + 1,
                    error=str(exc),
                ),
            )
            errors.append({"attempt": str(attempt_idx + 1), "error": str(exc)})

    json_dump(error_path, {"errors": errors})
    raise RuntimeError(f"Caption API failed after {max_attempts} attempts; see {error_path}")


def annotate_prepared_segment(
    segment: dict[str, int],
    image_paths: list[Path],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if args.skip_api:
        annotation = failed_annotation()
    else:
        annotation = call_caption_api(
            image_paths=image_paths,
            prompt=args.caption_prompt,
            base_url=args.base_url,
            model=args.model,
            api_key_env=args.api_key_env,
            timeout=args.timeout,
            max_tokens=args.max_tokens,
            payload_path=args.output_dir / f"segment_{segment['seg_id']:03d}_payload.json",
            error_path=args.output_dir / f"segment_{segment['seg_id']:03d}_error.json",
            retry_count=args.api_retries,
            api_image_max_side=args.api_image_max_side,
            api_image_jpeg_quality=args.api_image_jpeg_quality,
            disable_thinking=args.disable_thinking,
        )
    record = {**segment, **annotation}
    json_dump(args.output_dir / f"segment_{segment['seg_id']:03d}_annotation.json", record)
    return record


def annotate_segments_in_batches(
    prepared: list[tuple[dict[str, int], list[Path]]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if args.skip_api or args.api_batch_size == 1:
        return [annotate_prepared_segment(segment, image_paths, args) for segment, image_paths in tqdm(prepared, desc="caption segments")]

    batch_size = len(prepared) if args.api_batch_size <= 0 else args.api_batch_size
    annotations_by_id: dict[int, dict[str, Any]] = {}
    for batch_start in range(0, len(prepared), batch_size):
        batch = prepared[batch_start : batch_start + batch_size]
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            futures = {
                executor.submit(annotate_prepared_segment, segment, image_paths, args): segment["seg_id"]
                for segment, image_paths in batch
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"caption batch {batch_start // batch_size:03d}"):
                seg_id = futures[future]
                annotations_by_id[seg_id] = future.result()
    return [annotations_by_id[segment["seg_id"]] for segment, _ in prepared]



def load_gold(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate one egocentric video with the EgoANT pipeline.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--gold", type=Path, default=None, help="Optional gold annotation used only for evaluation.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--hand-recon", type=Path, default=None, help="Existing lightweight hand_recon.json.")
    parser.add_argument(
        "--reconstruction-episode",
        type=Path,
        default=None,
        help="Hand-reconstruction episode .npy metadata file.",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key-env", default="ANNOTATION_API_KEY")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--frames-per-segment", type=int, default=10)
    parser.add_argument(
        "--caption-input",
        choices=("overlay", "raw", "raw-hand-crop", "temporal-collage"),
        default="raw",
        help="Visual representation sent to the caption model.",
    )
    parser.add_argument("--min-segment-sec", type=float, default=1.0)
    parser.add_argument(
        "--short-segment-policy",
        choices=("coalesce", "merge-runs"),
        default="merge-runs",
        help="How to handle candidate segments shorter than --min-segment-sec.",
    )
    parser.add_argument("--smooth-sigma-sec", type=float, default=None, help="Optional time-unit smoothing sigma.")
    parser.add_argument("--smooth-sigma", type=float, default=2.0, help="Frame-unit smoothing sigma.")
    parser.add_argument("--cut-source", choices=("union", "left", "right"), default="union")
    parser.add_argument("--max-segments", type=int, default=None)
    parser.add_argument(
        "--api-batch-size",
        type=int,
        default=30,
        help="Caption API concurrency. Use 0 or a negative value to submit all segments at once.",
    )
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--api-retries", type=int, default=2)
    parser.add_argument(
        "--api-image-max-side",
        type=int,
        default=1280,
        help="Resize API payload images to this max side length (keeps aspect ratio). Saved raw/overlay frames are unchanged. Default 1280 keeps hand crops and composed grids readable.",
    )
    parser.add_argument(
        "--api-image-jpeg-quality",
        type=int,
        default=85,
        help="JPEG quality for resized API payload images.",
    )
    parser.add_argument("--skip-api", action="store_true", help="Write visualizations and prompts without calling the VLM.")
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Disable Qwen thinking on compatible local vLLM endpoints.",
    )
    parser.add_argument("--clean-output", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.caption_prompt = CAPTION_PROMPTS[args.caption_input]
    if args.clean_output and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")
    video_info = read_video_info(args.video)
    gold = load_gold(args.gold)
    if gold:
        json_dump(args.output_dir / "gold_reference.json", gold)

    hand_recon = load_hand_recon(args, video_info.nframes)
    json_dump(args.output_dir / "hand_recon.json", hand_recon)

    sigma_frames = float(args.smooth_sigma_sec) * video_info.fps if args.smooth_sigma_sec is not None else float(args.smooth_sigma)
    sigma_sec = sigma_frames / video_info.fps if video_info.fps else 0.0
    speed_by_hand = {
        hand: compute_speed(hand_recon["hands"][hand], video_info.fps, sigma_frames)
        for hand in ("left", "right")
        if hand in hand_recon.get("hands", {})
    }
    cuts_by_hand = {hand: find_speed_minima(speed, video_info.fps) for hand, speed in speed_by_hand.items()}
    merged_cuts_by_hand = {
        hand: merge_cut_points(cuts, video_info.fps, video_info.nframes) for hand, cuts in cuts_by_hand.items()
    }
    if args.cut_source == "union":
        all_cuts = [cut for cuts in cuts_by_hand.values() for cut in cuts]
    else:
        all_cuts = cuts_by_hand.get(args.cut_source, [])
    cuts = merge_cut_points(all_cuts, video_info.fps, video_info.nframes)
    segments = build_segments(
        cuts,
        video_info.nframes,
        video_info.fps,
        args.min_segment_sec,
        short_policy=args.short_segment_policy,
    )
    min_segment_frames = max(1, int(round(args.min_segment_sec * video_info.fps)))
    if args.max_segments is not None:
        segments = segments[: args.max_segments]

    plot_speed(speed_by_hand, cuts, gold, args.output_dir / "speed_plot.png")
    json_dump(
        args.output_dir / "segments.json",
        {
            "cuts": cuts,
            "cuts_by_hand": cuts_by_hand,
            "merged_cuts_by_hand": merged_cuts_by_hand,
            "diagnostics": segmentation_diagnostics(
                cuts_by_hand=cuts_by_hand,
                merged_cuts_by_hand=merged_cuts_by_hand,
                global_cuts=cuts,
                segments=segments,
                fps=video_info.fps,
                sigma_frames=sigma_frames,
                sigma_sec=sigma_sec,
                cut_source=args.cut_source,
                min_segment_sec=args.min_segment_sec,
                min_segment_frames=min_segment_frames,
                short_segment_policy=args.short_segment_policy,
            ),
            "segments": segments,
        },
    )

    frames = load_video_frames(args.video)
    prepared: list[tuple[dict[str, int], list[Path]]] = []
    for segment_idx, segment in enumerate(tqdm(segments, desc="prepare segment frames")):
        if args.caption_input == "temporal-collage":
            # Keep standard frame artifacts for downstream merge/export, but do not send them to captioning.
            save_segment_frames(frames, hand_recon, segment, args.output_dir, args.frames_per_segment)
            collage_dir = args.output_dir / f"segment_{segment['seg_id']:03d}_temporal_collage"
            contexts = (
                segments[segment_idx - 1] if segment_idx > 0 else None,
                segment,
                segments[segment_idx + 1] if segment_idx + 1 < len(segments) else None,
            )
            image_paths = [
                save_segment_collage(frames, context, collage_dir / f"{name}.jpg")
                for name, context in zip(("01_past", "02_current", "03_future"), contexts)
            ]
        else:
            image_paths = save_segment_frames(
                frames,
                hand_recon,
                segment,
                args.output_dir,
                args.frames_per_segment,
                use_overlay=args.caption_input == "overlay",
            )
            if args.caption_input == "raw-hand-crop":
                crop_path = args.output_dir / f"segment_{segment['seg_id']:03d}_hand_crop_grid" / "hand_crop_grid.jpg"
                image_paths.append(save_segment_hand_crop_grid(frames, hand_recon, segment, crop_path))
        prompt_path = args.output_dir / f"segment_{segment['seg_id']:03d}_prompt.txt"
        prompt_path.write_text(args.caption_prompt, encoding="utf-8")
        prepared.append((segment, image_paths))

    annotations = annotate_segments_in_batches(prepared, args)

    result = {
        "fps": video_info.fps,
        "nframes": video_info.nframes,
        "duration_sec": round(video_info.nframes / video_info.fps, 3),
        "model": args.model,
        "caption_input": args.caption_input,
        "segments": annotations,
        # TODO: postprocess adjacent duplicate instructions, N/A spans, and short segments.
        # TODO: add hand-object contact / active-object signals if wrist-speed cuts are weak.
    }
    json_dump(args.output_dir / "annotations.json", result)
    summarize_usage(args.output_dir / "api_usage.jsonl", args.output_dir / "token_usage.json")
    print(args.output_dir / "annotations.json")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
