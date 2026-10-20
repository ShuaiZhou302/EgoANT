#!/usr/bin/env python3
"""WGO-style contact-sheet VLM segmentation for HomER (Track B).

Samples frames ~0.5s, packs ~20 tiles of 224px with visual timestamps,
asks VLM for completed-event boundaries, writes scorer-compatible JSON.

Recipe-align flags (defaults preserve legacy Track B behavior):
  --max-sheets-per-call 0  -> whole-episode (all sheets in one call)
  --prompt-variant gepa_duration_prior_v1 -> blog GEPA prompt
"""
from __future__ import annotations

import argparse
import base64
import json
import math
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
    "You segment egocentric manipulation videos into completed atomic events. "
    "Reply with ONLY valid JSON."
)

# Legacy Track B prompt (pre-recipe-align)
USER_PROMPT_LEGACY = """These images are contact sheets from one continuous egocentric video.
Each tile is a frame sampled every {sample_sec:.2f}s. The yellow text on each tile is the timestamp in seconds.
Tiles are ordered left-to-right, top-to-bottom, and sheets are in time order.

Episode task (context only): {instruction}

List every COMPLETED manipulation event visible in these sheets.
Rules:
- An event ends when the manipulation is completed (object placed, lid closed, pour finished, etc.).
- Use the yellow timestamps as the time axis.
- start_sec/end_sec must be within [{t0:.2f}, {t1:.2f}] and non-overlapping, sorted by time.
- Prefer concise imperative labels (verb + object + target when visible).
- Do not invent events that are not visible.
- Cover the full span as well as you can; merge tiny flickers.

Return ONLY JSON:
{{"segments":[{{"start_sec":0.0,"end_sec":1.0,"subtask":"..."}}, ...]}}
"""

# Blog GEPA best: completed_events_duration_prior_v1
# Source: https://macrodata.co/blog/annotating-robot-video-subtasks (Prompt: GEPA variants)
# Status: blog_original (extracted from public blog page; not a local repo file)
USER_PROMPT_GEPA = """Reconstruct the sequence of manipulation events in this robot video from the timestamped contact sheets.

Context:
- Each tile is a frame sampled every {sample_sec:.2f}s. Yellow text on each tile is the timestamp in seconds.
- Tiles are ordered left-to-right, top-to-bottom; sheets are in time order.
- Visible time range in this call: [{t0:.2f}, {t1:.2f}] seconds.
- Episode instruction (context only): {instruction}

Return only JSON with this shape:
{{"segments":[{{"start_sec":0.0,"end_sec":1.0,"subtask":"short action description"}}]}}

Rules:
- Segment only completed robot manipulation events, not every visible movement.
- Good boundaries happen when a held object changes, an object is placed or released, a tool starts/stops changing a surface, a container/door/lid opens or closes, or contents move between containers.
- Do not split approach, grasp adjustment, small repositioning, and retreat unless the world state changes.
- Do not merge separate pick/place/open/close/pour/wipe events when they complete different states.
- Most segments should be 2-10 seconds. Shorter segments are okay only for fast pick, place, open, close, or release events.
- Use the visible timestamps for start_sec and end_sec.
- Ignore label wording quality; prioritize temporally correct boundaries.
- start_sec/end_sec must be within [{t0:.2f}, {t1:.2f}], non-overlapping, sorted by time.
"""

PROMPTS = {
    "legacy": USER_PROMPT_LEGACY,
    "gepa_duration_prior_v1": USER_PROMPT_GEPA,
}

PROMPT_SOURCE = {
    "legacy": "local_legacy_track_b",
    "gepa_duration_prior_v1": "blog_original_completed_events_duration_prior_v1",
}


def message_text(msg: dict) -> str:
    candidates = []
    content = msg.get("content")
    if isinstance(content, str) and content.strip():
        candidates.append(content)
    elif isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict) and c.get("type") == "text":
                parts.append(str(c.get("text") or ""))
            elif isinstance(c, str):
                parts.append(c)
        joined = "\n".join(parts).strip()
        if joined:
            candidates.append(joined)
    for k in ("reasoning", "reasoning_content"):
        v = msg.get(k)
        if isinstance(v, str) and v.strip():
            candidates.append(v)
    for c in candidates:
        if "segments" in c and "{" in c:
            return c
    return candidates[0] if candidates else ""


def encode_bgr(frame: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


def sample_times(duration: float, sample_sec: float) -> list[float]:
    if duration <= 0:
        return [0.0]
    n = max(1, int(math.floor(duration / sample_sec)) + 1)
    times = [min(duration, i * sample_sec) for i in range(n)]
    if times[-1] < duration - 1e-3:
        times.append(duration)
    return times


def read_frame_at(cap: cv2.VideoCapture, fps: float, nframes: int, t: float) -> np.ndarray | None:
    idx = min(max(nframes - 1, 0), max(0, int(round(t * fps))))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    return frame if ok else None


def make_tile(frame: np.ndarray, t: float, tile: int = 224) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = tile / max(h, w)
    resized = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    canvas = np.full((tile, tile, 3), 16, dtype=np.uint8)
    y0 = (tile - resized.shape[0]) // 2
    x0 = (tile - resized.shape[1]) // 2
    canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    label = f"{t:.1f}s"
    cv2.rectangle(canvas, (2, 2), (90, 28), (0, 0, 0), -1)
    cv2.putText(canvas, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return canvas


def pack_contact_sheet(tiles: list[np.ndarray], cols: int = 5) -> np.ndarray:
    if not tiles:
        return np.full((224, 224, 3), 32, dtype=np.uint8)
    tile = tiles[0].shape[0]
    rows = int(math.ceil(len(tiles) / cols))
    canvas = np.full((rows * tile, cols * tile, 3), 8, dtype=np.uint8)
    for i, im in enumerate(tiles):
        r, c = divmod(i, cols)
        canvas[r * tile : (r + 1) * tile, c * tile : (c + 1) * tile] = im
    return canvas


def build_sheets(
    video: Path,
    sample_sec: float,
    tiles_per_sheet: int,
    tile_px: int,
    cols: int,
) -> tuple[list[bytes], list[tuple[float, float]], float]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = nframes / fps if fps > 0 else 0.0
    times = sample_times(duration, sample_sec)
    sheets: list[bytes] = []
    spans: list[tuple[float, float]] = []
    for i in range(0, len(times), tiles_per_sheet):
        chunk = times[i : i + tiles_per_sheet]
        tiles = []
        for t in chunk:
            fr = read_frame_at(cap, fps, nframes, t)
            if fr is None:
                continue
            tiles.append(make_tile(fr, t, tile_px))
        if not tiles:
            continue
        sheet = pack_contact_sheet(tiles, cols=cols)
        sheets.append(encode_bgr(sheet, quality=80))
        spans.append((chunk[0], chunk[-1]))
    cap.release()
    return sheets, spans, duration


def parse_segments(text: str) -> list[dict[str, Any]]:
    if not text:
        return []
    # Prefer fenced JSON
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    raw = fence.group(1) if fence else None
    if raw is None:
        m = re.search(r"\{.*\}", text, re.S)
        raw = m.group(0) if m else None
    if not raw:
        return []
    d: dict[str, Any] | None = None
    try:
        d = json.loads(raw)
    except Exception:
        # salvage truncated arrays: take complete objects only
        m2 = re.search(r'"segments"\s*:\s*\[(.*)\]', text, re.S)
        if m2:
            body = m2.group(1)
            objs = re.findall(r"\{[^{}]*\}", body)
            segs_try = []
            for ob in objs:
                try:
                    segs_try.append(json.loads(ob))
                except Exception:
                    continue
            d = {"segments": segs_try}
        else:
            return []
    segs = (d or {}).get("segments") or []
    out = []
    for s in segs:
        try:
            a = float(s.get("start_sec", s.get("start", 0)))
            b = float(s.get("end_sec", s.get("end", 0)))
            # reject obvious millisecond mistakes (e.g. 45000 for 45s) when span is huge
            if a > 10000 or b > 10000:
                a, b = a / 1000.0, b / 1000.0
            lab = str(s.get("subtask") or s.get("label") or s.get("instruction") or "").strip()
        except Exception:
            continue
        if b <= a:
            continue
        out.append({"start_sec": a, "end_sec": b, "subtask": lab or "event"})
    out.sort(key=lambda x: x["start_sec"])
    return out


def is_context_error(err: str) -> bool:
    e = err.lower()
    keys = (
        "context length",
        "maximum context",
        "max_model_len",
        "too many tokens",
        "prompt is too long",
        "exceeds the model",
        "token limit",
        "max tokens",
        "input is too long",
        "context_length_exceeded",
    )
    return any(k in e for k in keys)


def merge_adjacent(segs: list[dict[str, Any]], gap: float = 0.05) -> list[dict[str, Any]]:
    if not segs:
        return []
    segs = sorted(segs, key=lambda x: x["start_sec"])
    out = [dict(segs[0])]
    for s in segs[1:]:
        prev = out[-1]
        if s["start_sec"] <= prev["end_sec"] + gap and s["subtask"] == prev["subtask"]:
            prev["end_sec"] = max(prev["end_sec"], s["end_sec"])
        elif s["start_sec"] < prev["end_sec"]:
            s = dict(s)
            s["start_sec"] = prev["end_sec"]
            if s["end_sec"] > s["start_sec"] + 1e-3:
                out.append(s)
        else:
            out.append(dict(s))
    return out


def postprocess_segments(
    segs: list[dict[str, Any]],
    duration: float,
    min_seg_sec: float,
) -> list[dict[str, Any]]:
    """Clamp, drop FAILED placeholders, enforce min duration by merging tiny gaps into neighbors."""
    cleaned = []
    for s in segs:
        lab = str(s.get("subtask") or "")
        if lab.startswith("[FAILED]"):
            continue
        a = max(0.0, min(duration, float(s["start_sec"])))
        b = max(a, min(duration, float(s["end_sec"])))
        if b - a < 1e-3:
            continue
        cleaned.append({"start_sec": a, "end_sec": b, "subtask": lab or "event"})
    cleaned = merge_adjacent(cleaned)
    if min_seg_sec <= 0:
        return cleaned
    # drop ultra-short segments that are not mergeable (blog: short ok only for fast events;
    # we only drop < min if extremely tiny noise)
    out = []
    for s in cleaned:
        if s["end_sec"] - s["start_sec"] < min_seg_sec and out:
            # absorb into previous
            out[-1]["end_sec"] = max(out[-1]["end_sec"], s["end_sec"])
        else:
            out.append(s)
    return out


def _one_call(
    base_url: str,
    model: str,
    api_key: str,
    sheets: list[bytes],
    spans: list[tuple[float, float]],
    instruction: str,
    sample_sec: float,
    timeout: float,
    retries: int,
    prompt_variant: str,
    max_tokens: int,
    failed_fallback: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Single VLM call over the given sheets. Returns (segs, meta)."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    t0 = spans[0][0]
    t1 = spans[-1][1]
    prompt_tmpl = PROMPTS[prompt_variant]
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": prompt_tmpl.format(
                sample_sec=sample_sec,
                instruction=instruction or "(none)",
                t0=t0,
                t1=t1,
            ),
        }
    ]
    for b in sheets:
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
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    last = ""
    segs: list[dict[str, Any]] = []
    http_ok = False
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            if r.status_code >= 400:
                last = f"HTTP{r.status_code}: {r.text[:500]}"
                if is_context_error(last):
                    return [], {"error": last, "context_overflow": True, "empty": True, "raw_preview": last[:200]}
                time.sleep(min(2 * attempt, 10))
                continue
            http_ok = True
            msg = r.json()["choices"][0]["message"]
            text = message_text(msg)
            last = text
            segs = parse_segments(text)
            if segs:
                break
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
            if is_context_error(last):
                return [], {"error": last, "context_overflow": True, "empty": True, "raw_preview": last[:200]}
            time.sleep(min(2 * attempt, 10))
    meta = {
        "empty": not bool(segs),
        "http_ok": http_ok,
        "raw_preview": (last or "")[:300],
        "n_sheets_in_call": len(sheets),
        "t0": t0,
        "t1": t1,
    }
    if not segs and failed_fallback:
        segs = [{"start_sec": t0, "end_sec": max(t0 + sample_sec, t1), "subtask": f"[FAILED] {last[:80]}"}]
        meta["failed_fallback"] = True
    return segs, meta


def call_segment(
    base_url: str,
    model: str,
    api_key: str,
    sheets: list[bytes],
    spans: list[tuple[float, float]],
    instruction: str,
    sample_sec: float,
    timeout: float,
    retries: int,
    max_sheets_per_call: int,
    prompt_variant: str = "legacy",
    max_tokens: int = 4096,
    failed_fallback: bool = True,
    context_fallback: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Call VLM. max_sheets_per_call<=0 means whole-episode (all sheets)."""
    n = len(sheets)
    if n == 0:
        return [], {"error": "no_sheets"}
    chunk_size = n if max_sheets_per_call <= 0 else max_sheets_per_call
    call_meta: list[dict[str, Any]] = []
    all_segs: list[dict[str, Any]] = []
    used_chunk = chunk_size
    context_fallback_used = False

    # Whole-episode preferred path with optional shrink on context overflow
    if max_sheets_per_call <= 0:
        try_sizes = [n]
        if context_fallback and n > 1:
            # geometric shrink: n, floor(n*2/3), floor(n/2), ... down to 1
            s = n
            while s > 1:
                s = max(1, (s * 2) // 3)
                if s not in try_sizes:
                    try_sizes.append(s)
        best_segs: list[dict[str, Any]] = []
        best_meta: dict[str, Any] = {}
        for sz in try_sizes:
            # if sz < n, tile the episode into non-overlapping windows of sz (max-feasible)
            window_segs: list[dict[str, Any]] = []
            window_metas: list[dict[str, Any]] = []
            overflow = False
            for i in range(0, n, sz):
                segs, meta = _one_call(
                    base_url, model, api_key,
                    sheets[i : i + sz], spans[i : i + sz],
                    instruction, sample_sec, timeout, retries,
                    prompt_variant, max_tokens, failed_fallback=False,
                )
                window_metas.append(meta)
                if meta.get("context_overflow"):
                    overflow = True
                    break
                window_segs.extend(segs)
            if overflow:
                context_fallback_used = True
                continue
            # Empty parse on large whole-episode calls: shrink and retry (same GEPA prompt).
            if (not window_segs) and sz > 1 and context_fallback:
                context_fallback_used = True
                continue
            best_segs = window_segs
            best_meta = {
                "calls": window_metas,
                "sheets_per_call_effective": sz,
                "whole_episode": sz >= n,
                "context_fallback_used": context_fallback_used,
                "n_sheets": n,
            }
            used_chunk = sz
            break
        else:
            # total failure
            if failed_fallback:
                best_segs = [{"start_sec": spans[0][0], "end_sec": spans[-1][1], "subtask": "[FAILED] context/empty"}]
            best_meta = {
                "calls": [],
                "sheets_per_call_effective": 0,
                "whole_episode": False,
                "context_fallback_used": True,
                "n_sheets": n,
                "error": "all_sizes_failed",
            }
        return merge_adjacent(best_segs), best_meta

    # Legacy chunked path
    for i in range(0, n, chunk_size):
        segs, meta = _one_call(
            base_url, model, api_key,
            sheets[i : i + chunk_size], spans[i : i + chunk_size],
            instruction, sample_sec, timeout, retries,
            prompt_variant, max_tokens, failed_fallback,
        )
        call_meta.append(meta)
        all_segs.extend(segs)
    return merge_adjacent(all_segs), {
        "calls": call_meta,
        "sheets_per_call_effective": used_chunk,
        "whole_episode": False,
        "context_fallback_used": False,
        "n_sheets": n,
    }


def process_one(
    gold_path: Path,
    video_root: Path,
    out_dir: Path,
    sheet_dir: Path | None,
    base_url: str,
    model: str,
    api_key: str,
    sample_sec: float,
    tiles_per_sheet: int,
    tile_px: int,
    cols: int,
    max_sheets_per_call: int,
    timeout: float,
    retries: int,
    prompt_variant: str = "legacy",
    max_tokens: int = 4096,
    failed_fallback: bool = True,
    min_seg_sec: float = 0.0,
    context_fallback: bool = True,
) -> dict:
    gold = json.loads(gold_path.read_text())
    rid = gold["id"]
    video = video_root / "homer" / f"{rid}.mp4"
    sheets, spans, duration = build_sheets(video, sample_sec, tiles_per_sheet, tile_px, cols)
    if sheet_dir is not None:
        sd = sheet_dir / rid
        sd.mkdir(parents=True, exist_ok=True)
        for i, b in enumerate(sheets):
            (sd / f"sheet_{i:03d}.jpg").write_bytes(b)
    segs, call_info = call_segment(
        base_url,
        model,
        api_key,
        sheets,
        spans,
        gold.get("instruction", ""),
        sample_sec,
        timeout,
        retries,
        max_sheets_per_call,
        prompt_variant=prompt_variant,
        max_tokens=max_tokens,
        failed_fallback=failed_fallback,
        context_fallback=context_fallback,
    )
    segs = postprocess_segments(segs, duration, min_seg_sec)
    n_empty_calls = sum(1 for c in call_info.get("calls") or [] if c.get("empty"))
    out = {
        "id": rid,
        "instruction": gold.get("instruction"),
        "method": "contact_sheet_vlm",
        "sample_sec": sample_sec,
        "tiles_per_sheet": tiles_per_sheet,
        "tile_px": tile_px,
        "cols": cols,
        "n_sheets": len(sheets),
        "duration_sec": duration,
        "max_sheets_per_call_arg": max_sheets_per_call,
        "sheets_per_call_effective": call_info.get("sheets_per_call_effective"),
        "whole_episode": bool(call_info.get("whole_episode")),
        "context_fallback_used": bool(call_info.get("context_fallback_used")),
        "prompt_variant": prompt_variant,
        "prompt_source": PROMPT_SOURCE.get(prompt_variant, "unknown"),
        "n_empty_calls": n_empty_calls,
        "call_info": {k: v for k, v in call_info.items() if k != "calls"} | {
            "n_calls": len(call_info.get("calls") or []),
            "empty_ratio": (n_empty_calls / max(1, len(call_info.get("calls") or []))),
        },
        "segments": segs,
    }
    (out_dir / f"{rid}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(
        f"[{rid}] sheets={len(sheets)} pred_segs={len(segs)} dur={duration:.1f}s "
        f"whole={out['whole_episode']} eff_chunk={out['sheets_per_call_effective']} "
        f"prompt={prompt_variant}",
        flush=True,
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--episode-dir",
        "--gold-dir",
        dest="episode_dir",
        type=Path,
        required=True,
        help="Episode JSONs; generation reads only id and instruction. --gold-dir is a compatibility alias.",
    )
    ap.add_argument("--video-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--sheet-dir", type=Path, default=None)
    ap.add_argument("--base-url", default="http://127.0.0.1:18004/v1")
    ap.add_argument("--base-urls", nargs="*", default=None)
    ap.add_argument("--model", default="Qwen/Qwen3.5-397B-A17B-FP8")
    ap.add_argument("--api-key", default=os.environ.get("LOCAL_MODEL_API_KEY", "local-vllm"))
    ap.add_argument("--sample-sec", type=float, default=0.5)
    ap.add_argument("--tiles-per-sheet", type=int, default=20)
    ap.add_argument("--tile-px", type=int, default=224)
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument(
        "--max-sheets-per-call",
        type=int,
        default=3,
        help="Sheets per VLM call. <=0 means whole-episode (all sheets once).",
    )
    ap.add_argument(
        "--prompt-variant",
        choices=sorted(PROMPTS.keys()),
        default="legacy",
        help="legacy=Track B; gepa_duration_prior_v1=blog GEPA best",
    )
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--min-seg-sec", type=float, default=0.25, help="Absorb segments shorter than this into previous")
    ap.add_argument("--no-failed-fallback", action="store_true", help="Do not inject [FAILED] coarse segments")
    ap.add_argument("--no-context-fallback", action="store_true", help="Do not shrink sheet count on context overflow")
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.sheet_dir:
        args.sheet_dir.mkdir(parents=True, exist_ok=True)
    urls = args.base_urls or [args.base_url]
    golds = sorted(args.episode_dir.glob("homer_*.json"))
    if args.ids:
        want = set(args.ids)
        golds = [g for g in golds if g.stem in want]

    def run_one(gi, g):
        outp = args.out_dir / f"{g.stem}.json"
        if args.skip_existing and outp.exists():
            print(f"skip existing {g.stem}", flush=True)
            return
        url = urls[gi % len(urls)]
        process_one(
            g,
            args.video_root,
            args.out_dir,
            args.sheet_dir,
            url,
            args.model,
            args.api_key,
            args.sample_sec,
            args.tiles_per_sheet,
            args.tile_px,
            args.cols,
            args.max_sheets_per_call,
            args.timeout,
            args.retries,
            prompt_variant=args.prompt_variant,
            max_tokens=args.max_tokens,
            failed_fallback=not args.no_failed_fallback,
            min_seg_sec=args.min_seg_sec,
            context_fallback=not args.no_context_fallback,
        )

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_one, i, g) for i, g in enumerate(golds)]
        for fut in as_completed(futs):
            fut.result()


if __name__ == "__main__":
    main()
