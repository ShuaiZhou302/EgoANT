#!/usr/bin/env python3
"""Caption predicted WGO segments with the same visual labeler used for Track A.

This is for E2E ablations: the temporal boundaries come from a predicted
segment directory, but each predicted segment is re-labeled from video frames.
The output keeps the same boundaries and replaces only ``subtask``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import label_gold_segments as labeler  # noqa: E402


def process_one(
    pred_path: Path,
    video_root: Path,
    out_dir: Path,
    caption_input: str,
    base_urls: list[str],
    model: str,
    api_key: str,
    timeout: float,
    retries: int,
    workers: int,
) -> dict[str, Any]:
    pred = json.loads(pred_path.read_text(encoding="utf-8"))
    rid = str(pred.get("id") or pred_path.stem)
    video = video_root / "homer" / f"{rid}.mp4"
    segs = pred.get("segments") or []

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    out_segments: list[dict[str, Any] | None] = [None] * len(segs)

    def work(item: tuple[int, dict[str, Any]]) -> tuple[int, dict[str, Any]]:
        i, seg = item
        local_cap = cv2.VideoCapture(str(video))
        try:
            frames, mode = labeler.build_images_for_segment(
                local_cap, fps, nframes, segs, i, caption_input, None
            )
        finally:
            local_cap.release()
        label = labeler.call_label(
            base_urls[i % len(base_urls)],
            model,
            api_key,
            frames,
            str(pred.get("instruction") or ""),
            caption_input,
            timeout,
            retries,
        )
        row = dict(seg)
        row["source_subtask"] = seg.get("subtask")
        row["subtask"] = label
        row["caption_input"] = caption_input
        row["overlay_mode"] = mode
        return i, row

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(work, (i, s)) for i, s in enumerate(segs)]
        for done, fut in enumerate(as_completed(futures), 1):
            i, row = fut.result()
            out_segments[i] = row
            print(f"[{rid}] {done}/{len(segs)} -> {row['subtask'][:100]}", flush=True)

    out = {
        "id": rid,
        "instruction": pred.get("instruction"),
        "source_pred": str(pred_path),
        "caption_input": caption_input,
        "label_model": model,
        "segments": out_segments,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / pred_path.name).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-boundary-dir", type=Path, required=True)
    ap.add_argument("--video-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--caption-input", choices=("raw", "overlay", "raw-hand-crop", "temporal-collage"), default="raw")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
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
        print(f"==== labeling predicted {pred_path.stem} caption_input={args.caption_input} ====", flush=True)
        process_one(
            pred_path,
            args.video_root,
            args.out_dir,
            args.caption_input,
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
