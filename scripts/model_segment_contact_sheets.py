#!/usr/bin/env python3
"""Ask a VLM to segment a video from timestamped contact sheets.

This is an experimental branch path inspired by WGO-Bench style fixed visual
summaries. It samples one frame every 0.5 seconds, writes the timestamp on each
frame, groups 20 frames into one 10-second contact sheet, and asks the model to
return subtask boundaries.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests


SYSTEM_PROMPT = "You are a careful egocentric video segmentation annotator."

SEGMENT_PROMPT = """Reconstruct the sequence of manipulation events in this egocentric video from the timestamped contact sheets.

Return only JSON with this shape:
{"overall_instruction":"short task-level instruction","segments":[{"start_sec":0.0,"end_sec":1.0,"subtask":"short action description"}]}

Rules:
- Segment only completed manipulation events, not every visible movement.
- Good boundaries happen when a held object changes, an object is placed or released, a tool starts/stops changing a surface, a container/door/lid opens or closes, or contents move between containers.
- Do not split approach, grasp adjustment, small repositioning, and retreat unless the world state changes.
- Do not merge separate pick/place/open/close/pour/wipe events when they complete different states.
- Most segments should be 2-10 seconds. Shorter segments are okay only for fast pick, place, open, close, or release events.
- Use the visible timestamps for start_sec and end_sec.
- Ignore label wording quality; prioritize temporally correct boundaries.
"""


def json_dump(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


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


def image_to_data_url(path: Path, max_side: int | None, jpeg_quality: int) -> str:
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Cannot read image: {path}")
    if max_side and max_side > 0:
        height, width = image.shape[:2]
        scale = min(1.0, float(max_side) / max(height, width))
        if scale < 1.0:
            image = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        raise RuntimeError(f"Cannot encode image: {path}")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def read_video_info(video: Path) -> tuple[float, int, int, int, float]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration = nframes / fps if fps > 0 else 0.0
    return fps, nframes, width, height, duration


def sample_frames(video: Path, sample_interval_sec: float) -> list[tuple[float, np.ndarray]]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = nframes / fps if fps > 0 else 0.0
    samples: list[tuple[float, np.ndarray]] = []
    t = 0.0
    while t <= duration + 1e-6:
        frame_idx = min(nframes - 1, max(0, int(round(t * fps))))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if ok:
            samples.append((t, frame))
        t += sample_interval_sec
    cap.release()
    return samples


def draw_tile(frame: np.ndarray, timestamp: float, tile_width: int, tile_height: int) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(tile_width / width, tile_height / height)
    resized = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
    tile = np.full((tile_height, tile_width, 3), 18, dtype=np.uint8)
    y0 = (tile_height - resized.shape[0]) // 2
    x0 = (tile_width - resized.shape[1]) // 2
    tile[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    label = f"{timestamp:06.2f}s"
    cv2.rectangle(tile, (0, 0), (92, 28), (0, 0, 0), -1)
    cv2.putText(tile, label, (7, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
    return tile


def write_contact_sheets(
    samples: list[tuple[float, np.ndarray]],
    out_dir: Path,
    frames_per_sheet: int,
    tile_width: int,
    tile_height: int,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = 5
    rows = int(np.ceil(frames_per_sheet / cols))
    sheet_paths: list[Path] = []
    for sheet_idx, offset in enumerate(range(0, len(samples), frames_per_sheet)):
        chunk = samples[offset : offset + frames_per_sheet]
        canvas = np.full((rows * tile_height, cols * tile_width, 3), 18, dtype=np.uint8)
        for idx, (timestamp, frame) in enumerate(chunk):
            row, col = divmod(idx, cols)
            tile = draw_tile(frame, timestamp, tile_width, tile_height)
            y0 = row * tile_height
            x0 = col * tile_width
            canvas[y0 : y0 + tile_height, x0 : x0 + tile_width] = tile
        start_t = chunk[0][0]
        end_t = chunk[-1][0]
        path = out_dir / f"sheet_{sheet_idx:04d}_{start_t:07.1f}_{end_t:07.1f}.jpg"
        if not cv2.imwrite(str(path), canvas):
            raise RuntimeError(f"Cannot write contact sheet: {path}")
        sheet_paths.append(path)
    return sheet_paths


def call_api(args: argparse.Namespace, sheet_paths: list[Path], prompt: str) -> dict[str, Any]:
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key environment variable: {args.api_key_env}")
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for path in sheet_paths:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_to_data_url(path, args.api_image_max_side, args.api_image_jpeg_quality)},
            }
        )
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        "temperature": 0.1,
        "max_tokens": args.max_tokens,
    }
    if args.disable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    redacted = {
        **payload,
        "messages": payload["messages"][:1]
        + [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": "<redacted>"}]}],
    }
    json_dump(args.output_dir / "segmentation_payload.json", redacted)
    started = time.monotonic()
    response = requests.post(
        f"{args.base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=args.timeout,
    )
    if not response.ok:
        raise requests.HTTPError(f"{response.status_code} for {response.url}; body={response.text[:1000]}")
    data = response.json()
    raw = data["choices"][0]["message"]["content"]
    json_dump(args.output_dir / "segmentation_raw_response.json", {"text": raw, "latency_sec": round(time.monotonic() - started, 3)})
    return extract_json(raw)


def normalize_segments(data: dict[str, Any], duration: float) -> dict[str, Any]:
    segments = []
    for item in data.get("segments", []):
        if not isinstance(item, dict):
            continue
        start = max(0.0, min(duration, float(item.get("start_sec", 0.0))))
        end = max(start, min(duration, float(item.get("end_sec", duration))))
        subtask = str(item.get("subtask", "")).strip()
        if end > start and subtask:
            segments.append({"start_sec": round(start, 3), "end_sec": round(end, 3), "subtask": subtask})
    segments.sort(key=lambda x: (x["start_sec"], x["end_sec"]))
    return {
        "overall_instruction": str(data.get("overall_instruction", "")).strip(),
        "duration_sec": round(duration, 3),
        "segments": segments,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3.5-397B-A17B-FP8")
    parser.add_argument("--api-key-env", default="ANNOTATION_API_KEY")
    parser.add_argument("--sample-interval-sec", type=float, default=0.5)
    parser.add_argument("--frames-per-sheet", type=int, default=20)
    parser.add_argument("--tile-width", type=int, default=224)
    parser.add_argument("--tile-height", type=int, default=144)
    parser.add_argument("--api-image-max-side", type=int, default=1600)
    parser.add_argument("--api-image-jpeg-quality", type=int, default=95)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--skip-api", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fps, nframes, width, height, duration = read_video_info(args.video)
    samples = sample_frames(args.video, args.sample_interval_sec)
    sheet_paths = write_contact_sheets(
        samples,
        args.output_dir / "contact_sheets",
        args.frames_per_sheet,
        args.tile_width,
        args.tile_height,
    )
    metadata = {
        "video": str(args.video),
        "fps": fps,
        "nframes": nframes,
        "width": width,
        "height": height,
        "duration_sec": round(duration, 3),
        "sample_interval_sec": args.sample_interval_sec,
        "frames_per_sheet": args.frames_per_sheet,
        "contact_sheets": [str(path) for path in sheet_paths],
    }
    json_dump(args.output_dir / "segmentation_inputs.json", metadata)
    prompt = SEGMENT_PROMPT + f"\n\nVideo duration: {duration:.3f} seconds."
    (args.output_dir / "segmentation_prompt.txt").write_text(prompt, encoding="utf-8")
    if args.skip_api:
        prediction = {"overall_instruction": "", "duration_sec": round(duration, 3), "segments": []}
    else:
        prediction = normalize_segments(call_api(args, sheet_paths, prompt), duration)
    json_dump(args.output_dir / "model_segments.json", prediction)
    print(args.output_dir / "model_segments.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
