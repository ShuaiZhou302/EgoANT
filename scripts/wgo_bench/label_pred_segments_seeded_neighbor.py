#!/usr/bin/env python3
"""Relabel predicted WGO segments with neighbor sheets and a prior label.

This experiment-only helper keeps predicted temporal boundaries fixed. For
each segment it shows PREV/CURRENT/NEXT contact sheets and asks the VLM to keep
or revise the prior label for the CURRENT segment.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import label_opt_ideas as opt  # noqa: E402

SYSTEM = (
    "You label egocentric manipulation subtasks. "
    "Describe the completed manipulation event in one short imperative phrase. "
    "Include the main object and destination/target when visible. "
    "Do not narrate the actor. Reply with ONLY JSON: {\"label\": \"...\"}."
)

PROMPT = (
    "I will send three contact sheets: PREV segment, CURRENT segment, NEXT segment. "
    "Each sheet has up to 5 timestamped frames sampled from that predicted segment. "
    "A prior label for the CURRENT segment is provided. "
    "Keep the prior label if it matches the CURRENT sheet; otherwise revise it. "
    "Use PREV/NEXT only to disambiguate object state or destination. "
    "Return only the final label for the CURRENT segment."
)


def message_text(msg: dict[str, Any]) -> str:
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(c.get("text") or "") for c in content if isinstance(c, dict) and c.get("type") == "text")
    for k in ("reasoning", "reasoning_content"):
        if msg.get(k):
            return str(msg[k])
    return ""


def call_label(
    base_url: str,
    model: str,
    api_key: str,
    frames: list[bytes],
    instruction: str,
    prior_label: str,
    timeout: float,
    retries: int,
) -> tuple[str, str]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"{PROMPT}\nEpisode task (context only): {instruction}\n"
                f"Prior CURRENT label: {prior_label}\n"
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
            r = requests.post(f"{base_url.rstrip('/')}/chat/completions", headers=headers, json=payload, timeout=timeout)
            r.raise_for_status()
            text = message_text(r.json()["choices"][0]["message"])
            last = text
            m = re.search(r"\{[^{}]*\}", text or "", re.S)
            if m:
                d = json.loads(m.group(0))
                lab = str(d.get("label") or "").strip()
                if lab:
                    return lab, text
            line = (text or "").strip().splitlines()[0].strip().strip('"') if text else ""
            if line and not line.startswith("[FAILED]"):
                return line, text
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
            time.sleep(min(2 * attempt, 10))
    return f"[FAILED] {last[:200]}", last


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
    base_urls: list[str],
    model: str,
    api_key: str,
    timeout: float,
    retries: int,
    workers: int,
    frames_per_sheet: int,
    image_max_side: int,
) -> None:
    pred = json.loads(pred_path.read_text(encoding="utf-8"))
    rid = str(pred.get("id") or pred_path.stem)
    video = video_root / "homer" / f"{rid}.mp4"
    fps, nframes = video_meta(video)
    segs = pred.get("segments") or []
    out_segments: list[dict[str, Any] | None] = [None] * len(segs)

    def work(item: tuple[int, dict[str, Any]]) -> tuple[int, dict[str, Any]]:
        i, seg = item
        cap = cv2.VideoCapture(str(video))
        try:
            frames, mode, meta = opt.build_l1_images(cap, fps, nframes, segs, i, frames_per_sheet)
        finally:
            cap.release()
        if image_max_side != 960:
            # label_opt_ideas encodes at 960; keep a metadata record when callers
            # request a different nominal side for comparison with old runs.
            meta["requested_image_max_side"] = image_max_side
        prior = str(seg.get("subtask") or seg.get("instruction") or "")
        label, raw = call_label(
            base_urls[i % len(base_urls)],
            model,
            api_key,
            frames,
            str(pred.get("instruction") or ""),
            prior,
            timeout,
            retries,
        )
        row = dict(seg)
        row["prior_subtask"] = prior
        row["subtask"] = label
        row["relabel_input"] = "seeded_neighbor_contact_sheet"
        row["relabel_meta"] = meta
        row["raw_response"] = raw
        return i, row

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(work, (i, s)) for i, s in enumerate(segs)]
        for done, fut in enumerate(as_completed(futures), 1):
            i, row = fut.result()
            out_segments[i] = row
            print(f"[{rid}] {done}/{len(segs)} -> {row['subtask'][:100]}", flush=True)

    out = dict(pred)
    out["segments"] = out_segments
    out["relabel_recipe"] = {
        "name": "seeded_neighbor_contact_sheet_27B",
        "source_pred_dir": str(pred_path.parent),
        "frames_per_sheet": frames_per_sheet,
        "image_max_side": image_max_side,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / pred_path.name).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-boundary-dir", type=Path, required=True)
    ap.add_argument("--video-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:18001/v1")
    ap.add_argument("--base-urls", nargs="*", default=None)
    ap.add_argument("--model", default="qwen3.6-27b")
    ap.add_argument("--api-key", default=os.environ.get("LOCAL_MODEL_API_KEY", "local-vllm"))
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--segment-workers", type=int, default=4)
    ap.add_argument("--frames-per-sheet", type=int, default=5)
    ap.add_argument("--image-max-side", type=int, default=1120)
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
        print(f"==== seeded-neighbor relabel {pred_path.stem} ====", flush=True)
        process_one(
            pred_path,
            args.video_root,
            args.out_dir,
            urls,
            args.model,
            args.api_key,
            args.timeout,
            args.retries,
            args.segment_workers,
            args.frames_per_sheet,
            args.image_max_side,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
