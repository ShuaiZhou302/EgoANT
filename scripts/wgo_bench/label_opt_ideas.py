#!/usr/bin/env python3
"""Opt-ideas labeling: L1 neighbor contact-sheet, L2 true hand-collage.

Independent of EgoVid defaults. Does not change label_gold_segments.py defaults.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests

SYSTEM = (
    "You label egocentric manipulation subtasks. "
    "Describe the completed manipulation event in one short imperative phrase. "
    "Include the main object and destination/target when visible. "
    "Do not narrate the actor. Reply with ONLY JSON: {\"label\": \"...\"}."
)

PROMPT_L1 = (
    "I will send three contact sheets: PREV segment, CURRENT segment, NEXT segment. "
    "Each sheet has up to 5 frames sampled uniformly from that gold segment, "
    "left-to-right, with a yellow title and yellow per-tile timestamps in seconds. "
    "Label ONLY the completed action in the CURRENT sheet. "
    "Use PREV/NEXT only to disambiguate object state / destination."
)

PROMPT_L2 = (
    "I will send three hand-cropped collages: PAST, CURRENT, FUTURE. "
    "Each collage focuses on the hand/manipulation region (tight crops around wrists). "
    "Label ONLY the completed action visible in the CURRENT collage. "
    "Use PAST/FUTURE only to disambiguate."
)


def message_text(msg: dict) -> str:
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict) and c.get("type") == "text":
                parts.append(str(c.get("text") or ""))
            elif isinstance(c, str):
                parts.append(c)
        return "\n".join(parts)
    for k in ("reasoning", "reasoning_content"):
        if msg.get(k):
            return str(msg[k])
    return ""


def encode_bgr(frame: np.ndarray, max_side: int = 960, quality: int = 85) -> bytes:
    h, w = frame.shape[:2]
    side = max(h, w)
    scale = float(max_side) / side if side > max_side else 1.0
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


def sample_indices(nframes: int, start_f: int, end_f: int, n: int) -> list[int]:
    s = max(0, min(nframes - 1, start_f))
    e = max(s, min(nframes - 1, end_f))
    if n <= 1:
        return [s]
    return [int(round(s + (e - s) * i / (n - 1))) for i in range(n)]


def read_frame(cap: cv2.VideoCapture, idx: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    return frame if ok else None


def load_segment_frames(
    cap: cv2.VideoCapture, fps: float, nframes: int, start_sec: float, end_sec: float, n: int
) -> list[tuple[np.ndarray, float]]:
    """Return list of (frame_bgr, timestamp_sec)."""
    sf = int(round(start_sec * fps))
    ef = int(round(end_sec * fps))
    idxs = sample_indices(nframes, sf, ef, n)
    out: list[tuple[np.ndarray, float]] = []
    for i in idxs:
        fr = read_frame(cap, i)
        if fr is not None:
            out.append((fr, float(i) / float(fps) if fps > 0 else 0.0))
    return out


def make_contact_sheet(
    frames: list,
    title: str,
    n: int = 5,
    tile: int = 224,
) -> np.ndarray:
    """WGO-style neighbor contact sheet with in-tile yellow timestamps (seconds)."""
    cols = n
    header = 36
    canvas = np.full((tile + header, tile * cols, 3), 28, dtype=np.uint8)
    cv2.putText(canvas, title, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    if not frames:
        cv2.putText(
            canvas,
            "NO CONTEXT",
            (tile * cols // 2 - 80, header + tile // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (180, 180, 180),
            2,
        )
        return canvas
    norm: list[tuple[np.ndarray, float | None]] = []
    for item in frames:
        if isinstance(item, tuple) and len(item) == 2:
            norm.append((item[0], float(item[1])))
        else:
            norm.append((item, None))  # type: ignore[arg-type]
    idxs = (
        sample_indices(len(norm), 0, len(norm) - 1, n)
        if len(norm) >= n
        else list(range(len(norm)))
    )
    while len(idxs) < n:
        idxs.append(idxs[-1])
    for col, fi in enumerate(idxs[:n]):
        fr, tsec = norm[fi]
        h, w = fr.shape[:2]
        scale = min(tile / w, tile / h)
        resized = cv2.resize(
            fr,
            (max(1, round(w * scale)), max(1, round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
        y0 = header + (tile - resized.shape[0]) // 2
        x0 = col * tile + (tile - resized.shape[1]) // 2
        canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
        label = f"{tsec:.1f}s" if tsec is not None else str(col)
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)):
            color = (0, 0, 0) if (dx, dy) != (0, 0) else (0, 255, 255)
            cv2.putText(
                canvas,
                label,
                (col * tile + 4 + dx, header + 18 + dy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
            )
    return canvas

_POSE_MODEL = None


def get_pose_model(weights: str):
    global _POSE_MODEL
    if _POSE_MODEL is None:
        from ultralytics import YOLO

        _POSE_MODEL = YOLO(weights)
    return _POSE_MODEL


def wrist_points(frame: np.ndarray, model, conf: float = 0.25) -> np.ndarray:
    """Return Nx2 wrist keypoints from YOLO pose (COCO 9=left_wrist, 10=right_wrist)."""
    res = model.predict(frame, verbose=False, conf=conf)
    pts = []
    if not res:
        return np.empty((0, 2), dtype=float)
    r0 = res[0]
    if r0.keypoints is None or r0.keypoints.xy is None:
        return np.empty((0, 2), dtype=float)
    xy = r0.keypoints.xy.cpu().numpy()
    confs = None
    if r0.keypoints.conf is not None:
        confs = r0.keypoints.conf.cpu().numpy()
    for pi in range(xy.shape[0]):
        for ki in (9, 10):
            if confs is not None and confs[pi, ki] < conf:
                continue
            x, y = xy[pi, ki]
            if np.isfinite(x) and np.isfinite(y) and x > 1 and y > 1:
                pts.append([float(x), float(y)])
    return np.asarray(pts, dtype=float) if pts else np.empty((0, 2), dtype=float)


def crop_hand_from_points(frame: np.ndarray, points: np.ndarray) -> np.ndarray | None:
    if points.size == 0:
        return None
    h, w = frame.shape[:2]
    finite = points[np.isfinite(points).all(axis=1)]
    finite = finite[
        (finite[:, 0] >= 0) & (finite[:, 0] < w) & (finite[:, 1] >= 0) & (finite[:, 1] < h)
    ]
    if len(finite) == 0:
        return None
    x0, y0 = finite.min(axis=0)
    x1, y1 = finite.max(axis=0)
    cx, cy = float((x0 + x1) / 2.0), float((y0 + y1) / 2.0)
    bbox = max(float(x1 - x0), float(y1 - y0), 1.0)
    short = min(w, h)
    side = max(bbox * 3.0, short * 0.22)
    side = min(side, short * 0.55)
    xa = int(round(cx - side / 2))
    ya = int(round(cy - side / 2))
    xb = int(round(cx + side / 2))
    yb = int(round(cy + side / 2))
    xa, ya = max(0, xa), max(0, ya)
    xb, yb = min(w, xb), min(h, yb)
    if xb <= xa or yb <= ya:
        return None
    return frame[ya:yb, xa:xb]


def make_hand_collage(crops: list[np.ndarray], title: str, count: int = 6) -> np.ndarray:
    tile_w, tile_h = 320, 240
    header = 36
    canvas = np.full((tile_h * 2 + header, tile_w * 3, 3), 24, dtype=np.uint8)
    cv2.putText(canvas, title, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    if not crops:
        cv2.putText(
            canvas,
            "NO HAND DETECTED",
            (tile_w - 40, header + tile_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
        return canvas
    idxs = (
        sample_indices(len(crops), 0, len(crops) - 1, count)
        if len(crops) >= count
        else list(range(len(crops)))
    )
    while len(idxs) < count:
        idxs.append(idxs[-1])
    for order, fi in enumerate(idxs[:count]):
        crop = crops[fi]
        h, w = crop.shape[:2]
        scale = min(tile_w / w, tile_h / h)
        resized = cv2.resize(
            crop,
            (max(1, round(w * scale)), max(1, round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
        row, col = divmod(order, 3)
        y0 = header + row * tile_h + (tile_h - resized.shape[0]) // 2
        x0 = col * tile_w + (tile_w - resized.shape[1]) // 2
        canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    return canvas


def build_l1_images(cap, fps, nframes, segs, seg_i, frames_per: int = 5) -> tuple[list[bytes], str, dict]:
    seg = segs[seg_i]
    cur = load_segment_frames(
        cap, fps, nframes, float(seg["start_sec"]), float(seg["end_sec"]), frames_per
    )
    past = segs[seg_i - 1] if seg_i > 0 else None
    fut = segs[seg_i + 1] if seg_i + 1 < len(segs) else None
    past_fr = (
        load_segment_frames(
            cap, fps, nframes, float(past["start_sec"]), float(past["end_sec"]), frames_per
        )
        if past
        else []
    )
    fut_fr = (
        load_segment_frames(
            cap, fps, nframes, float(fut["start_sec"]), float(fut["end_sec"]), frames_per
        )
        if fut
        else []
    )
    sheets = [
        make_contact_sheet(past_fr, "PREV", frames_per),
        make_contact_sheet(cur, "CURRENT", frames_per),
        make_contact_sheet(fut_fr, "NEXT", frames_per),
    ]
    meta = {
        "n_prev": len(past_fr),
        "n_cur": len(cur),
        "n_next": len(fut_fr),
        "kind": "neighbor_contact_sheet",
        "timestamps": True,
        "cur_times": [t for _, t in cur],
    }
    return [encode_bgr(s) for s in sheets], "neighbor_contact_sheet", meta


def build_l2_images(
    cap, fps, nframes, segs, seg_i, pose_model, frames_per: int = 6
) -> tuple[list[bytes], str, dict]:
    seg = segs[seg_i]
    windows = []
    for name, s in (
        ("PAST", segs[seg_i - 1] if seg_i > 0 else None),
        ("CURRENT", seg),
        ("FUTURE", segs[seg_i + 1] if seg_i + 1 < len(segs) else None),
    ):
        if s is None:
            windows.append((name, []))
            continue
        frs = load_segment_frames(
            cap, fps, nframes, float(s["start_sec"]), float(s["end_sec"]), frames_per
        )
        crops = []
        for item in frs:
            fr = item[0] if isinstance(item, tuple) else item
            pts = wrist_points(fr, pose_model)
            crop = crop_hand_from_points(fr, pts)
            if crop is not None:
                crops.append(crop)
        windows.append((name, crops))
    grids = [make_hand_collage(crops, name) for name, crops in windows]
    n_ok = sum(1 for _, c in windows if c)
    cur_crops = windows[1][1]
    meta = {
        "kind": "hand_collage_yolo_pose",
        "n_past_crops": len(windows[0][1]),
        "n_cur_crops": len(cur_crops),
        "n_fut_crops": len(windows[2][1]),
        "windows_with_hands": n_ok,
        "current_has_hand": bool(cur_crops),
    }
    mode = "hand_collage" if cur_crops else "hand_collage_NO_DETECTION"
    return [encode_bgr(g) for g in grids], mode, meta


def call_label(base_url, model, api_key, frames, instruction, intro, timeout, retries) -> str:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"{intro}\nEpisode task (context only): {instruction}\n"
                "Label the CURRENT segment only."
            ),
        }
    ]
    for b in frames:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/jpeg;base64," + base64.b64encode(b).decode("ascii")
                },
            }
        )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": content},
        ],
        "max_tokens": 256,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last = ""
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            r.raise_for_status()
            text = message_text(r.json()["choices"][0]["message"])
            last = text
            m = re.search(r"\{[^{}]*\}", text or "", re.S)
            if m:
                d = json.loads(m.group(0))
                lab = str(d.get("label") or "").strip()
                if lab:
                    return lab
            line = (text or "").strip().splitlines()[0].strip().strip('"') if text else ""
            if line and not line.startswith("[FAILED]"):
                return line
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
            time.sleep(min(2 * attempt, 10))
    return f"[FAILED] {last[:200]}"


def process_one(
    gold_path,
    video_root,
    out_dir,
    mode,
    base_urls,
    model,
    api_key,
    timeout,
    retries,
    workers,
    pose_weights,
    debug_dir,
):
    gold = json.loads(gold_path.read_text())
    rid = gold["id"]
    video = video_root / "homer" / f"{rid}.mp4"
    segs = gold["segments"]
    cap0 = cv2.VideoCapture(str(video))
    if not cap0.isOpened():
        raise RuntimeError(f"cannot open {video}")
    fps = float(cap0.get(cv2.CAP_PROP_FPS) or 30.0)
    nframes = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap0.release()

    pose_model = get_pose_model(pose_weights) if mode == "hand-collage" else None
    intro = PROMPT_L1 if mode == "neighbor-contact-sheet" else PROMPT_L2
    preds: list[dict | None] = [None] * len(segs)
    skip_no_hand = 0

    def work(i_seg):
        i, seg = i_seg
        local = cv2.VideoCapture(str(video))
        try:
            if mode == "neighbor-contact-sheet":
                frames, omode, meta = build_l1_images(local, fps, nframes, segs, i)
            else:
                frames, omode, meta = build_l2_images(local, fps, nframes, segs, i, pose_model)
        finally:
            local.release()
        if mode == "hand-collage" and not meta.get("current_has_hand"):
            return i, {
                "start_sec": float(seg["start_sec"]),
                "end_sec": float(seg["end_sec"]),
                "subtask": "[SKIPPED] no_hand_detection",
                "gold_subtask": seg.get("subtask"),
                "caption_input": mode,
                "overlay_mode": omode,
                "meta": meta,
                "skipped": True,
            }
        if debug_dir and i < 2:
            ddir = Path(debug_dir) / rid
            ddir.mkdir(parents=True, exist_ok=True)
            for j, b in enumerate(frames):
                (ddir / f"seg{i}_{j}.jpg").write_bytes(b)
        url = base_urls[i % len(base_urls)]
        label = call_label(
            url, model, api_key, frames, gold.get("instruction", ""), intro, timeout, retries
        )
        return i, {
            "start_sec": float(seg["start_sec"]),
            "end_sec": float(seg["end_sec"]),
            "subtask": label,
            "gold_subtask": seg.get("subtask"),
            "caption_input": mode,
            "overlay_mode": omode,
            "meta": meta,
            "skipped": False,
        }

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, (i, s)) for i, s in enumerate(segs)]
        for fut in as_completed(futs):
            i, row = fut.result()
            preds[i] = row
            print(f"[{rid}] {i+1}/{len(segs)} ({mode}) -> {row['subtask'][:100]}", flush=True)

    skip_no_hand = sum(1 for r in preds if r and r.get("skipped"))
    out = {
        "id": rid,
        "instruction": gold.get("instruction"),
        "caption_input": mode,
        "n_skipped_no_hand": skip_no_hand,
        "segments": preds,
    }
    (out_dir / f"{rid}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold-dir", type=Path, required=True)
    ap.add_argument("--video-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--mode", choices=("neighbor-contact-sheet", "hand-collage"), required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:18004/v1")
    ap.add_argument("--base-urls", nargs="*", default=None)
    ap.add_argument("--model", default="Qwen/Qwen3.5-397B-A17B-FP8")
    ap.add_argument("--api-key", default=os.environ.get("LOCAL_MODEL_API_KEY", "local-vllm"))
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--segment-workers", type=int, default=4)
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument(
        "--pose-weights",
        default=os.environ.get("POSE_WEIGHTS", "models/yolov8n-pose.pt"),
        help="Pose weights used only by hand-collage mode.",
    )
    ap.add_argument("--debug-dir", type=Path, default=None)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    urls = args.base_urls or [args.base_url]
    golds = sorted(args.gold_dir.glob("homer_*.json"))
    if args.ids:
        want = set(args.ids)
        golds = [g for g in golds if g.stem in want]
    for g in golds:
        outp = args.out_dir / f"{g.stem}.json"
        if args.skip_existing and outp.exists():
            print(f"skip existing {g.stem}", flush=True)
            continue
        print(f"==== {args.mode} {g.stem} ====", flush=True)
        process_one(
            g,
            args.video_root,
            args.out_dir,
            args.mode,
            urls,
            args.model,
            args.api_key,
            args.timeout,
            args.retries,
            args.segment_workers,
            args.pose_weights,
            args.debug_dir,
        )


if __name__ == "__main__":
    main()
