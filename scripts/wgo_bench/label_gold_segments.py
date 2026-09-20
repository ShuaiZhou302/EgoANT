#!/usr/bin/env python3
"""Caption gold WGO segments with EgoANT-style VLM calls (labeling-only eval).

Supports --caption-input: overlay | raw | raw-hand-crop | temporal-collage.
Does not change the default EgoANT pipeline; experiment-only entrypoint.
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

PROMPT_BY_INPUT = {
    "overlay": (
        "I will send ordered frames from one gold segment. "
        "A 2D motion/hand trajectory is overlaid; color shows temporal progress. "
        "Label the CURRENT segment only."
    ),
    "raw": (
        "I will send ordered raw frames sampled uniformly from one gold segment. "
        "Label the CURRENT segment only."
    ),
    "raw-hand-crop": (
        "I will send ordered raw frames plus one hand-focused crop grid from the same gold segment. "
        "Label the CURRENT segment only."
    ),
    "temporal-collage": (
        "I will send three ordered frame grids: PAST context, CURRENT target, and FUTURE context. "
        "Each grid has 6 raw frames in row-major order. "
        "Label ONLY the visible completed action in the CURRENT grid. "
        "Use PAST/FUTURE only to disambiguate."
    ),
}


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


def encode_bgr(frame: np.ndarray, max_side: int = 640, quality: int = 85) -> bytes:
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


def trajectory_color(t: float) -> tuple[int, int, int]:
    a = max(0.0, min(1.0, t))
    return (0, int(255 * (1 - a)), int(255 * a))


def draw_hand_overlay(
    frame: np.ndarray, hand_recon: dict[str, Any], start: int, end: int, current_idx: int
) -> np.ndarray:
    img = frame.copy()
    for hand in ("left", "right"):
        hand_info = (hand_recon.get("hands") or {}).get(hand)
        if not hand_info:
            continue
        xy = np.asarray(hand_info["wrist_xy"], dtype=float)
        valid = np.asarray(hand_info["valid"], dtype=bool)
        pts: list[tuple[int, int, int]] = []
        for idx in range(start, min(end + 1, len(xy))):
            if idx < len(valid) and valid[idx] and np.isfinite(xy[idx]).all():
                pts.append((idx, int(round(xy[idx, 0])), int(round(xy[idx, 1]))))
        for p0, p1 in zip(pts[:-1], pts[1:]):
            t = (p1[0] - start) / max(1, end - start)
            cv2.line(img, (p0[1], p0[2]), (p1[1], p1[2]), trajectory_color(t), 2, cv2.LINE_AA)
        if (
            0 <= current_idx < len(xy)
            and current_idx < len(valid)
            and valid[current_idx]
            and np.isfinite(xy[current_idx]).all()
        ):
            label = "L" if hand == "left" else "R"
            x, y = int(round(xy[current_idx, 0])), int(round(xy[current_idx, 1]))
            cv2.circle(img, (x, y), 5, (255, 0, 0), -1, cv2.LINE_AA)
            cv2.putText(img, label, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(img, f"Frame {current_idx}", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img


def motion_centroids(cap: cv2.VideoCapture, indices: list[int]) -> list[tuple[float, float] | None]:
    """Cheap optical-flow centroid trail when hand_recon is missing."""
    prev_gray = None
    cents: list[tuple[float, float] | None] = []
    for idx in indices:
        frame = read_frame(cap, idx)
        if frame is None:
            cents.append(None)
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            h, w = gray.shape
            cents.append((w * 0.5, h * 0.55))
        else:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mask = mag > np.percentile(mag, 90)
            ys, xs = np.where(mask)
            if len(xs) < 20:
                h, w = gray.shape
                cents.append((w * 0.5, h * 0.55))
            else:
                cents.append((float(xs.mean()), float(ys.mean())))
        prev_gray = gray
    return cents


def draw_motion_overlay(
    frame: np.ndarray, trail: list[tuple[float, float] | None], current: int
) -> np.ndarray:
    img = frame.copy()
    for i in range(1, len(trail)):
        if trail[i - 1] is None or trail[i] is None:
            continue
        t = i / max(1, len(trail) - 1)
        p0 = (int(trail[i - 1][0]), int(trail[i - 1][1]))
        p1 = (int(trail[i][0]), int(trail[i][1]))
        cv2.line(img, p0, p1, trajectory_color(t), 2, cv2.LINE_AA)
    if 0 <= current < len(trail) and trail[current] is not None:
        x, y = int(trail[current][0]), int(trail[current][1])
        cv2.circle(img, (x, y), 6, (255, 0, 0), -1, cv2.LINE_AA)
    cv2.putText(img, f"Frame {current}", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img


def make_collage(frames: list[np.ndarray], count: int = 6) -> np.ndarray:
    tile_w, tile_h = 320, 180
    canvas = np.full((tile_h * 2, tile_w * 3, 3), 32, dtype=np.uint8)
    if not frames:
        cv2.putText(canvas, "NO CONTEXT", (365, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (210, 210, 210), 2)
        return canvas
    idxs = (
        sample_indices(len(frames), 0, len(frames) - 1, count)
        if len(frames) >= count
        else list(range(len(frames)))
    )
    while len(idxs) < count:
        idxs.append(idxs[-1] if idxs else 0)
    for order, fi in enumerate(idxs[:count]):
        tile = frames[fi]
        h, w = tile.shape[:2]
        scale = min(tile_w / w, tile_h / h)
        resized = cv2.resize(
            tile,
            (max(1, round(w * scale)), max(1, round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
        row, col = divmod(order, 3)
        y0 = row * tile_h + (tile_h - resized.shape[0]) // 2
        x0 = col * tile_w + (tile_w - resized.shape[1]) // 2
        canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    return canvas


def load_segment_frames(
    cap: cv2.VideoCapture,
    fps: float,
    nframes: int,
    start_sec: float,
    end_sec: float,
    n: int,
) -> tuple[list[np.ndarray], list[int]]:
    sf = int(round(start_sec * fps))
    ef = int(round(end_sec * fps))
    idxs = sample_indices(nframes, sf, ef, n)
    frames = []
    keep = []
    for i in idxs:
        fr = read_frame(cap, i)
        if fr is not None:
            frames.append(fr)
            keep.append(i)
    return frames, keep


def build_images_for_segment(
    cap: cv2.VideoCapture,
    fps: float,
    nframes: int,
    segs: list[dict],
    seg_i: int,
    caption_input: str,
    hand_recon: dict[str, Any] | None,
) -> tuple[list[bytes], str]:
    seg = segs[seg_i]
    start = float(seg["start_sec"])
    end = float(seg["end_sec"])

    if caption_input == "temporal-collage":
        cur_frames, _ = load_segment_frames(cap, fps, nframes, start, end, 6)
        past = segs[seg_i - 1] if seg_i > 0 else None
        fut = segs[seg_i + 1] if seg_i + 1 < len(segs) else None
        past_frames: list[np.ndarray] = []
        fut_frames: list[np.ndarray] = []
        if past:
            past_frames, _ = load_segment_frames(
                cap, fps, nframes, float(past["start_sec"]), float(past["end_sec"]), 6
            )
        if fut:
            fut_frames, _ = load_segment_frames(
                cap, fps, nframes, float(fut["start_sec"]), float(fut["end_sec"]), 6
            )
        grids = [
            make_collage(past_frames, 6),
            make_collage(cur_frames, 6),
            make_collage(fut_frames, 6),
        ]
        for name, g in zip(("PAST", "CURRENT", "FUTURE"), grids):
            cv2.putText(g, name, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        return [encode_bgr(g, max_side=960) for g in grids], "temporal-collage"

    frames, idxs = load_segment_frames(cap, fps, nframes, start, end, 5)
    if caption_input == "raw":
        return [encode_bgr(f) for f in frames], "raw"

    if caption_input == "overlay":
        sf = int(round(start * fps))
        ef = int(round(end * fps))
        out = []
        if hand_recon is not None:
            for fr, idx in zip(frames, idxs):
                out.append(encode_bgr(draw_hand_overlay(fr, hand_recon, sf, ef, idx)))
            return out, "hand_recon"
        trail_idxs = sample_indices(nframes, sf, ef, max(8, len(idxs)))
        cents = motion_centroids(cap, trail_idxs)
        trail = list(cents)
        for fr, idx in zip(frames, idxs):
            nearest = min(range(len(trail_idxs)), key=lambda j: abs(trail_idxs[j] - idx))
            out.append(encode_bgr(draw_motion_overlay(fr, trail, nearest)))
        return out, "optical_flow_proxy"

    if caption_input == "raw-hand-crop":
        imgs = [encode_bgr(f) for f in frames]
        crops = []
        for fr in frames[:6]:
            h, w = fr.shape[:2]
            side = int(min(h, w) * 0.45)
            cx, cy = w // 2, int(h * 0.6)
            x0, y0 = max(0, cx - side // 2), max(0, cy - side // 2)
            crop = fr[y0 : y0 + side, x0 : x0 + side]
            crops.append(crop if crop.size else fr)
        imgs.append(encode_bgr(make_collage(crops, 6), max_side=960))
        return imgs, "raw-hand-crop-centerproxy"

    raise ValueError(caption_input)


def call_label(
    base_url: str,
    model: str,
    api_key: str,
    frames: list[bytes],
    gold_hint_instruction: str,
    caption_input: str,
    timeout: float,
    retries: int,
) -> str:
    intro = PROMPT_BY_INPUT[caption_input]
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"{intro}\nEpisode task (context only): {gold_hint_instruction}\n"
                "Label the CURRENT segment only."
            ),
        }
    ]
    for b in frames:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(b).decode("ascii")},
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
            msg = r.json()["choices"][0]["message"]
            text = message_text(msg)
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
    gold_path: Path,
    video_root: Path,
    out_dir: Path,
    work_root: Path | None,
    caption_input: str,
    base_urls: list[str],
    model: str,
    api_key: str,
    timeout: float,
    retries: int,
    workers: int,
) -> dict:
    gold = json.loads(gold_path.read_text())
    rid = gold["id"]
    video = video_root / "homer" / f"{rid}.mp4"
    segs = gold["segments"]
    hand_recon = None
    if work_root is not None:
        hp = work_root / rid / "hand_recon.json"
        if hp.exists():
            hand_recon = json.loads(hp.read_text())

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    preds: list[dict | None] = [None] * len(segs)
    overlay_modes: list[str] = []

    def work(i_seg):
        i, seg = i_seg
        local_cap = cv2.VideoCapture(str(video))
        try:
            frames, mode = build_images_for_segment(
                local_cap, fps, nframes, segs, i, caption_input, hand_recon
            )
        finally:
            local_cap.release()
        url = base_urls[i % len(base_urls)]
        label = call_label(
            url,
            model,
            api_key,
            frames,
            gold.get("instruction", ""),
            caption_input,
            timeout,
            retries,
        )
        return i, {
            "start_sec": float(seg["start_sec"]),
            "end_sec": float(seg["end_sec"]),
            "subtask": label,
            "gold_subtask": seg.get("subtask"),
            "caption_input": caption_input,
            "overlay_mode": mode,
        }

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, (i, s)) for i, s in enumerate(segs)]
        for fut in as_completed(futs):
            i, row = fut.result()
            preds[i] = row
            overlay_modes.append(row.get("overlay_mode", ""))
            print(f"[{rid}] {i+1}/{len(segs)} ({caption_input}) -> {row['subtask'][:100]}", flush=True)

    out = {
        "id": rid,
        "instruction": gold.get("instruction"),
        "caption_input": caption_input,
        "overlay_mode_summary": sorted(set(overlay_modes)),
        "segments": preds,
    }
    out_path = out_dir / f"{rid}.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold-dir", type=Path, required=True)
    ap.add_argument("--video-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--work-root", type=Path, default=None, help="optional work dir with hand_recon.json")
    ap.add_argument(
        "--caption-input",
        choices=("overlay", "raw", "raw-hand-crop", "temporal-collage"),
        default="raw",
    )
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--base-urls", nargs="*", default=None, help="round-robin vLLM bases")
    ap.add_argument("--model", default="Qwen/Qwen3.5-397B-A17B-FP8")
    ap.add_argument("--api-key", default=os.environ.get("LOCAL_MODEL_API_KEY", "local-vllm"))
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--segment-workers", type=int, default=4)
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--skip-existing", action="store_true")
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
        print(f"==== labeling {g.stem} caption_input={args.caption_input} ====", flush=True)
        process_one(
            g,
            args.video_root,
            args.out_dir,
            args.work_root,
            args.caption_input,
            urls,
            args.model,
            args.api_key,
            args.timeout,
            args.retries,
            args.segment_workers,
        )


if __name__ == "__main__":
    main()
