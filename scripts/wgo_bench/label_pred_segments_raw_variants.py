#!/usr/bin/env python3
"""Relabel predicted WGO segments with raw-frame variants.

This experiment-only helper keeps the predicted temporal boundaries fixed and
changes only how raw frames are sampled before calling the VLM labeler.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import label_gold_segments as labeler  # noqa: E402


def sample_times(start_sec: float, end_sec: float, n: int, inner_margin_sec: float) -> list[float]:
    start = float(start_sec)
    end = max(start, float(end_sec))
    if inner_margin_sec > 0 and end - start > 2 * inner_margin_sec:
        start += inner_margin_sec
        end -= inner_margin_sec
    if n <= 1 or end <= start:
        return [start]
    return [start + (end - start) * i / (n - 1) for i in range(n)]


def opencv_frames(video: Path, fps: float, times: list[float]) -> list[bytes]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    frames: list[bytes] = []
    try:
        for t in times:
            idx = max(0, int(round(t * fps)))
            frame = labeler.read_frame(cap, idx)
            if frame is not None:
                frames.append(labeler.encode_bgr(frame))
    finally:
        cap.release()
    return frames


def ffmpeg_frames(video: Path, times: list[float], ffmpeg: str) -> list[bytes]:
    frames: list[bytes] = []
    for t in times:
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{max(0.0, t):.6f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "pipe:1",
        ]
        data = subprocess.check_output(cmd, timeout=30)
        if data:
            frames.append(data)
    return frames


def video_meta(video: Path) -> tuple[float, int]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    try:
        return float(cap.get(cv2.CAP_PROP_FPS) or 30.0), int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()


def process_one(
    pred_path: Path,
    video_root: Path,
    out_dir: Path,
    reader: str,
    inner_margin_sec: float,
    frames_per_segment: int,
    ffmpeg: str,
    base_urls: list[str],
    model: str,
    api_key: str,
    timeout: float,
    retries: int,
    workers: int,
) -> None:
    pred = json.loads(pred_path.read_text(encoding="utf-8"))
    rid = str(pred.get("id") or pred_path.stem)
    video = video_root / "homer" / f"{rid}.mp4"
    fps, nframes = video_meta(video)
    segs = pred.get("segments") or []
    out_segments: list[dict[str, Any] | None] = [None] * len(segs)

    def work(item: tuple[int, dict[str, Any]]) -> tuple[int, dict[str, Any]]:
        i, seg = item
        times = sample_times(
            float(seg["start_sec"]),
            float(seg["end_sec"]),
            frames_per_segment,
            inner_margin_sec,
        )
        if reader == "ffmpeg":
            frames = ffmpeg_frames(video, times, ffmpeg)
            mode = f"raw-ffmpeg-{frames_per_segment}frames"
        else:
            frames = opencv_frames(video, fps, times)
            mode = f"raw-opencv-inner{inner_margin_sec:g}"
        label = labeler.call_label(
            base_urls[i % len(base_urls)],
            model,
            api_key,
            frames,
            str(pred.get("instruction") or ""),
            "raw",
            timeout,
            retries,
        )
        row = dict(seg)
        row["source_subtask"] = seg.get("subtask")
        row["subtask"] = label
        row["caption_input"] = mode
        row["sample_times_sec"] = times
        row["video_fps"] = fps
        row["video_frames"] = nframes
        return i, row

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(work, (i, s)) for i, s in enumerate(segs)]
        for done, fut in enumerate(as_completed(futures), 1):
            i, row = fut.result()
            out_segments[i] = row
            print(f"[{rid}] {done}/{len(segs)} ({row['caption_input']}) -> {row['subtask'][:100]}", flush=True)

    out = {
        "id": rid,
        "instruction": pred.get("instruction"),
        "source_pred": str(pred_path),
        "caption_input": f"{reader}_raw",
        "inner_margin_sec": inner_margin_sec,
        "frames_per_segment": frames_per_segment,
        "label_model": model,
        "segments": out_segments,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / pred_path.name).write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-boundary-dir", type=Path, required=True)
    ap.add_argument("--video-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--reader", choices=("opencv", "ffmpeg"), default="opencv")
    ap.add_argument("--inner-margin-sec", type=float, default=0.0)
    ap.add_argument("--frames-per-segment", type=int, default=5)
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--base-url", default="http://127.0.0.1:18001/v1")
    ap.add_argument("--base-urls", nargs="*", default=None)
    ap.add_argument("--model", default="qwen3.6-27b")
    ap.add_argument("--api-key", default=os.environ.get("LOCAL_MODEL_API_KEY", "local-vllm"))
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--segment-workers", type=int, default=4)
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    paths = sorted(args.pred_boundary_dir.glob("homer_*.json"))
    if args.ids:
        wanted = set(args.ids)
        paths = [p for p in paths if p.stem in wanted]
    urls = args.base_urls or [args.base_url]
    for pred_path in paths:
        out_path = args.out_dir / pred_path.name
        if args.skip_existing and out_path.exists():
            print(f"skip existing {pred_path.stem}", flush=True)
            continue
        print(f"==== labeling predicted {pred_path.stem} reader={args.reader} inner={args.inner_margin_sec} ====", flush=True)
        process_one(
            pred_path,
            args.video_root,
            args.out_dir,
            args.reader,
            args.inner_margin_sec,
            args.frames_per_segment,
            args.ffmpeg,
            urls,
            args.model,
            args.api_key,
            args.timeout,
            args.retries,
            args.segment_workers,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
