#!/usr/bin/env python3
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
import requests

from label_opt_ideas import encode_bgr, load_segment_frames, make_contact_sheet, message_text  # noqa: E402

SYSTEM = (
    "You choose the best subtask label for an egocentric manipulation video segment. "
    "Return one concise imperative phrase. Do not narrate the actor. "
    "Reply with ONLY JSON: {\"label\": \"...\", \"source\": \"A|B|C|D|corrected\"}."
)

PROMPT = """I will send PREV, CURRENT, and NEXT contact sheets for one predicted segment.
Choose the best label for the CURRENT segment only.

Candidate labels:
A raw-frame label: {raw_label}
B raw-prior neighbor label: {rawprior_label}
C segmentation-prior neighbor label: {seed_label}
D ffmpeg raw-frame label: {ffmpeg_label}

Candidate A is usually the most stable. Prefer A unless the CURRENT sheet clearly supports another candidate.
Use PREV and NEXT only to understand what changed at the boundaries.
If all candidates miss an obvious action, output a corrected short label.

Episode task, for context only:
{instruction}"""


def parse_response(text: str) -> tuple[str, str]:
    match = re.search(r"\{[^{}]*\}", text or "", re.S)
    if match:
        try:
            data = json.loads(match.group(0))
            label = str(data.get("label") or "").strip()
            source = str(data.get("source") or "").strip()
            if label:
                return label, source
        except Exception:
            pass
    line = (text or "").strip().splitlines()[0].strip().strip('"')
    return line, "fallback"


def call_model(base_url: str, model: str, api_key: str, images: list[bytes], prompt: str, timeout: float, retries: int) -> tuple[str, str, str]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for image_bytes in images:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("ascii")
                },
            }
        )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": content},
        ],
        "max_tokens": 192,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last = ""
    for attempt in range(1, retries + 1):
        try:
            response = requests.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            text = message_text(response.json()["choices"][0]["message"])
            label, source = parse_response(text)
            if label:
                return label, source, text[:500]
            last = text[:500]
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
            time.sleep(min(2 * attempt, 10))
    return "[FAILED] " + last[:180], "failed", last


def video_path(video_root: Path, rid: str) -> Path:
    for candidate in [video_root / "homer" / f"{rid}.mp4", video_root / f"{rid}.mp4"]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"missing video for {rid}")


def build_images(video: Path, segs: list[dict[str, Any]], seg_i: int, frames_per: int, max_side: int) -> tuple[list[bytes], dict[str, Any]]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        windows = []
        for title, idx in [("PREV", seg_i - 1), ("CURRENT", seg_i), ("NEXT", seg_i + 1)]:
            if idx < 0 or idx >= len(segs):
                windows.append((title, []))
                continue
            seg = segs[idx]
            frames = load_segment_frames(cap, fps, nframes, float(seg["start_sec"]), float(seg["end_sec"]), frames_per)
            windows.append((title, frames))
        sheets = [make_contact_sheet(frames, title, frames_per) for title, frames in windows]
        return [encode_bgr(sheet, max_side=max_side, quality=88) for sheet in sheets], {
            "fps": fps,
            "nframes": nframes,
            "frames_per_sheet": frames_per,
            "image_max_side": max_side,
        }
    finally:
        cap.release()


def load_doc(dir_path: Path, rid: str) -> dict[str, Any]:
    return json.loads((dir_path / f"{rid}.json").read_text())


def seg_label(doc: dict[str, Any], index: int) -> str:
    segs = doc.get("segments") or []
    if 0 <= index < len(segs):
        return str(segs[index].get("subtask") or "")
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boundary-dir", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--rawprior-dir", type=Path, required=True)
    parser.add_argument("--seed-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg-dir", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--base-urls", nargs="+", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-397B-A17B-FP8")
    parser.add_argument("--api-key-env", default="LOCAL_MODEL_API_KEY")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--frames-per-sheet", type=int, default=5)
    parser.add_argument("--image-max-side", type=int, default=1120)
    parser.add_argument("--keep-raw-response", action="store_true")
    args = parser.parse_args()
    api_key = os.environ.get(args.api_key_env, "local-vllm")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    boundary_files = sorted(args.boundary_dir.glob("homer_*.json"))
    docs = {p.stem: json.loads(p.read_text()) for p in boundary_files}
    candidates = {}
    for rid in docs:
        candidates[rid] = {
            "raw": load_doc(args.raw_dir, rid),
            "rawprior": load_doc(args.rawprior_dir, rid),
            "seed": load_doc(args.seed_dir, rid),
            "ffmpeg": load_doc(args.ffmpeg_dir, rid),
        }
    jobs = []
    for rid, doc in docs.items():
        for i, seg in enumerate(doc.get("segments", [])):
            if float(seg.get("end_sec", 0)) > float(seg.get("start_sec", 0)):
                jobs.append((len(jobs), rid, i))

    results: dict[tuple[str, int], dict[str, Any]] = {}

    def work(job: tuple[int, str, int]) -> tuple[str, int, dict[str, Any]]:
        global_i, rid, seg_i = job
        doc = docs[rid]
        cand = candidates[rid]
        raw_label = seg_label(cand["raw"], seg_i)
        rawprior_label = seg_label(cand["rawprior"], seg_i)
        seed_label = seg_label(cand["seed"], seg_i)
        ffmpeg_label = seg_label(cand["ffmpeg"], seg_i)
        images, meta = build_images(video_path(args.video_root, rid), doc["segments"], seg_i, args.frames_per_sheet, args.image_max_side)
        prompt = PROMPT.format(
            raw_label=raw_label or "(empty)",
            rawprior_label=rawprior_label or "(empty)",
            seed_label=seed_label or "(empty)",
            ffmpeg_label=ffmpeg_label or "(empty)",
            instruction=doc.get("instruction") or "",
        )
        url = args.base_urls[global_i % len(args.base_urls)]
        label, source, raw_response = call_model(url, args.model, api_key, images, prompt, args.timeout, args.retries)
        out_seg = dict(doc["segments"][seg_i])
        out_seg["subtask"] = label
        out_seg["candidate_select_source"] = source
        out_seg["candidate_labels"] = {
            "raw": raw_label,
            "rawprior": rawprior_label,
            "seed": seed_label,
            "ffmpeg": ffmpeg_label,
        }
        out_seg["candidate_select_meta"] = meta
        if args.keep_raw_response:
            out_seg["raw_response"] = raw_response
        return rid, seg_i, out_seg

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(work, job) for job in jobs]
        for future in as_completed(futures):
            rid, seg_i, row = future.result()
            results[(rid, seg_i)] = row
            done += 1
            if done % 20 == 0 or done == len(jobs):
                print(f"selected {done}/{len(jobs)}", flush=True)

    for rid, doc in docs.items():
        out_doc = dict(doc)
        out_doc["candidate_select_recipe"] = {
            "name": "raw_rawprior_seed_ffmpeg_neighbor_selector_397B",
            "frames_per_sheet": args.frames_per_sheet,
            "image_max_side": args.image_max_side,
        }
        out_doc["segments"] = [results.get((rid, i), dict(seg)) for i, seg in enumerate(doc.get("segments", []))]
        (args.out_dir / f"{rid}.json").write_text(json.dumps(out_doc, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {len(docs)} files to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()

