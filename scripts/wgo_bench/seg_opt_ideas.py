#!/usr/bin/env python3
"""Opt-ideas segmentation: S1 anti-underseg + S2 local refine.

Wraps contact_sheet_segment helpers; does not change its defaults.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
import requests

SCR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCR))
import contact_sheet_segment as css  # noqa: E402

# S1: denser / shorter completed events than GEPA 2-10s prior
USER_PROMPT_ANTI_UNDERSEG = """Reconstruct the sequence of manipulation events in this robot video from the timestamped contact sheets.

Context:
- Each tile is a frame sampled every {sample_sec:.2f}s. Yellow text on each tile is the timestamp in seconds.
- Tiles are ordered left-to-right, top-to-bottom; sheets are in time order.
- Visible time range in this call: [{t0:.2f}, {t1:.2f}] seconds.
- Episode instruction (context only): {instruction}

Return only JSON with this shape:
{{"segments":[{{"start_sec":0.0,"end_sec":1.0,"subtask":"short action description"}}]}}

Rules:
- Segment ONLY completed robot manipulation events, not every visible movement.
- Prefer MORE, SHORTER atomic events over long merged ones. HomER events are often 1-4 seconds.
- Good boundaries: held object changes; object placed/released; tool starts/stops changing a surface; container/door/lid opens/closes; contents move between containers.
- Do NOT merge separate pick / place / open / close / pour / wipe when they complete different states.
- Do not split approach, grasp adjustment, small repositioning, and retreat unless the world state changes.
- Typical segment duration: 1-5 seconds. Segments longer than ~7s are usually under-segmented — split them at the next completed-state change.
- Cover the full visible span; avoid large unlabeled gaps between events.
- Use the visible timestamps for start_sec and end_sec.
- Ignore label wording quality; prioritize temporally correct boundaries.
- start_sec/end_sec must be within [{t0:.2f}, {t1:.2f}], non-overlapping, sorted by time.
"""

USER_PROMPT_REFINE_LOCAL = """These contact sheets cover a SHORT local window of an egocentric manipulation video.
Each tile is sampled every {sample_sec:.2f}s. Yellow text is the timestamp in seconds.
Visible time range: [{t0:.2f}, {t1:.2f}] seconds.
Episode instruction (context only): {instruction}
Coarse hypothesis for this window (may be wrong on boundaries): {coarse_hint}

Re-segment ONLY this local window into completed atomic events.
Prefer shorter events (1-5s). Split if multiple state changes are visible.
Return ONLY JSON:
{{"segments":[{{"start_sec":0.0,"end_sec":1.0,"subtask":"..."}}]}}
"""

# Exact prompt used for the reported HomER V5.1 S2 result (Segment F1 0.2031).
USER_PROMPT_REFINE_FULL_COVER = """These contact sheets cover a SHORT local window of an egocentric manipulation video.
Each tile is sampled every {sample_sec:.2f}s. Yellow text is the timestamp in seconds.
Visible time range: [{t0:.2f}, {t1:.2f}] seconds.
Episode instruction (context only): {instruction}
Coarse hypothesis for this window (may be wrong on boundaries): {coarse_hint}

Re-segment this entire local window into completed atomic manipulation events.
Return a gap-free, non-overlapping timeline that covers the whole visible range from {t0:.2f} to {t1:.2f} seconds.
Prefer completed world-state changes: pick, place, open, close, pour, wipe, cut, fold, tighten, insert, remove.
Do NOT create segments for passive inspection, small repositioning, grasp adjustment, hovering, approach, retreat, or camera motion unless a visible object state is completed.
Do NOT split continuous drawing/cutting/wiping into stroke-level micro-actions unless the object/state target clearly changes.
If a long continuous manipulation is visible, keep it as one segment rather than many micro-segments.
Use timestamp boundaries and keep segments sorted.
Return ONLY JSON:
{{"segments":[{{"start_sec":0.0,"end_sec":1.0,"subtask":"..."}}]}}
"""

# register without mutating css.PROMPTS permanently for other imports if we copy
PROMPTS = {
    "anti_underseg_v1": USER_PROMPT_ANTI_UNDERSEG,
    "refine_local_v1": USER_PROMPT_REFINE_LOCAL,
    "refine_full_cover_v1": USER_PROMPT_REFINE_FULL_COVER,
    **css.PROMPTS,
}


def _one_call_custom(
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
    extra_format: dict | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    t0 = spans[0][0]
    t1 = spans[-1][1]
    prompt_tmpl = PROMPTS[prompt_variant]
    fmt = dict(
        sample_sec=sample_sec,
        instruction=instruction or "(none)",
        t0=t0,
        t1=t1,
    )
    if extra_format:
        fmt.update(extra_format)
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt_tmpl.format(**fmt)}]
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
            {"role": "system", "content": css.SYSTEM},
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
                if css.is_context_error(last):
                    return [], {
                        "error": last,
                        "context_overflow": True,
                        "empty": True,
                        "raw_preview": last[:200],
                    }
                time.sleep(min(2 * attempt, 10))
                continue
            http_ok = True
            text = css.message_text(r.json()["choices"][0]["message"])
            last = text
            segs = css.parse_segments(text)
            if segs:
                break
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
            if css.is_context_error(last):
                return [], {
                    "error": last,
                    "context_overflow": True,
                    "empty": True,
                    "raw_preview": last[:200],
                }
            time.sleep(min(2 * attempt, 10))
    meta = {
        "empty": not bool(segs),
        "http_ok": http_ok,
        "raw_preview": (last or "")[:300],
        "n_sheets_in_call": len(sheets),
        "t0": t0,
        "t1": t1,
        "prompt_variant": prompt_variant,
    }
    if not segs and failed_fallback:
        segs = [
            {
                "start_sec": t0,
                "end_sec": max(t0 + sample_sec, t1),
                "subtask": f"[FAILED] {last[:80]}",
            }
        ]
        meta["failed_fallback"] = True
    return segs, meta


def split_long_segments(
    segs: list[dict[str, Any]],
    sheets: list[bytes],
    spans: list[tuple[float, float]],
    duration: float,
    sample_sec: float,
    base_url: str,
    model: str,
    api_key: str,
    instruction: str,
    timeout: float,
    retries: int,
    max_tokens: int,
    long_sec: float,
) -> tuple[list[dict[str, Any]], int]:
    """Secondary VLM split for segments longer than long_sec."""
    out: list[dict[str, Any]] = []
    n_split = 0
    for s in segs:
        dur = float(s["end_sec"]) - float(s["start_sec"])
        if dur < long_sec:
            out.append(s)
            continue
        # gather sheets overlapping this segment
        idxs = [
            i
            for i, (a, b) in enumerate(spans)
            if not (b < s["start_sec"] - 1e-3 or a > s["end_sec"] + 1e-3)
        ]
        if len(idxs) < 1:
            out.append(s)
            continue
        sub_sheets = [sheets[i] for i in idxs]
        sub_spans = [spans[i] for i in idxs]
        # clamp span ends to segment
        sub_spans = [
            (max(a, float(s["start_sec"])), min(b, float(s["end_sec"]))) for a, b in sub_spans
        ]
        fine, meta = _one_call_custom(
            base_url,
            model,
            api_key,
            sub_sheets,
            sub_spans,
            instruction,
            sample_sec,
            timeout,
            retries,
            "anti_underseg_v1",
            max_tokens,
            failed_fallback=False,
        )
        fine = [
            x
            for x in fine
            if float(x["end_sec"]) - float(x["start_sec"]) > 1e-3
            and float(x["start_sec"]) < float(s["end_sec"])
            and float(x["end_sec"]) > float(s["start_sec"])
        ]
        if len(fine) >= 2:
            n_split += 1
            for x in fine:
                x["start_sec"] = max(float(s["start_sec"]), float(x["start_sec"]))
                x["end_sec"] = min(float(s["end_sec"]), float(x["end_sec"]))
                if x["end_sec"] > x["start_sec"] + 1e-3:
                    out.append(x)
        else:
            out.append(s)
    out = css.merge_adjacent(sorted(out, key=lambda x: x["start_sec"]))
    out = css.postprocess_segments(out, duration, min_seg_sec=0.25)
    return out, n_split


def run_s1_one(
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
    timeout: float,
    retries: int,
    max_tokens: int,
    long_sec: float,
    do_long_split: bool,
):
    gold = json.loads(gold_path.read_text())
    rid = gold["id"]
    video = video_root / "homer" / f"{rid}.mp4"
    sheets, spans, duration = css.build_sheets(
        video, sample_sec, tiles_per_sheet, tile_px, cols
    )
    if sheet_dir:
        sdir = sheet_dir / rid
        sdir.mkdir(parents=True, exist_ok=True)
        for i, b in enumerate(sheets):
            (sdir / f"sheet_{i:03d}.jpg").write_bytes(b)

    segs, meta = _one_call_custom(
        base_url,
        model,
        api_key,
        sheets,
        spans,
        gold.get("instruction", ""),
        sample_sec,
        timeout,
        retries,
        "anti_underseg_v1",
        max_tokens,
        failed_fallback=False,
    )
    # if context overflow, fall back to css.call_segment chunking with custom prompt by
    # temporarily injecting into css.PROMPTS
    if meta.get("context_overflow") or meta.get("empty"):
        css.PROMPTS["anti_underseg_v1"] = USER_PROMPT_ANTI_UNDERSEG
        segs, meta = css.call_segment(
            base_url,
            model,
            api_key,
            sheets,
            spans,
            gold.get("instruction", ""),
            sample_sec,
            timeout,
            retries,
            max_sheets_per_call=0,
            prompt_variant="anti_underseg_v1",
            max_tokens=max_tokens,
            failed_fallback=False,
            context_fallback=True,
        )

    segs = css.postprocess_segments(segs, duration, min_seg_sec=0.25)
    n_split = 0
    if do_long_split:
        segs, n_split = split_long_segments(
            segs,
            sheets,
            spans,
            duration,
            sample_sec,
            base_url,
            model,
            api_key,
            gold.get("instruction", ""),
            timeout,
            retries,
            max_tokens,
            long_sec,
        )

    out = {
        "id": rid,
        "instruction": gold.get("instruction"),
        "duration_sec": duration,
        "n_sheets": len(sheets),
        "prompt_variant": "anti_underseg_v1",
        "whole_episode": True,
        "n_long_splits": n_split,
        "call_meta": meta,
        "segments": segs,
    }
    (out_dir / f"{rid}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(
        f"[{rid}] S1 sheets={len(sheets)} pred={len(segs)} long_splits={n_split} dur={duration:.1f}s",
        flush=True,
    )
    return out


def run_s2_one(
    gold_path: Path,
    video_root: Path,
    coarse_dir: Path,
    out_dir: Path,
    base_url: str,
    model: str,
    api_key: str,
    sample_sec: float,
    tiles_per_sheet: int,
    tile_px: int,
    cols: int,
    timeout: float,
    retries: int,
    max_tokens: int,
    pad_sec: float,
    min_refine_sec: float,
    refine_prompt: str,
):
    """Refine coarse segment boundaries with local contact-sheet calls."""
    gold = json.loads(gold_path.read_text())
    rid = gold["id"]
    coarse_path = coarse_dir / f"{rid}.json"
    if not coarse_path.exists():
        raise FileNotFoundError(coarse_path)
    coarse = json.loads(coarse_path.read_text())
    video = video_root / "homer" / f"{rid}.mp4"
    sheets, spans, duration = css.build_sheets(
        video, sample_sec, tiles_per_sheet, tile_px, cols
    )
    refined: list[dict[str, Any]] = []
    n_refined = 0
    for s in coarse.get("segments") or []:
        a = float(s["start_sec"])
        b = float(s["end_sec"])
        if b - a < min_refine_sec:
            refined.append(dict(s))
            continue
        wa, wb = max(0.0, a - pad_sec), min(duration, b + pad_sec)
        idxs = [i for i, (x0, x1) in enumerate(spans) if not (x1 < wa or x0 > wb)]
        if not idxs:
            refined.append(dict(s))
            continue
        sub_sheets = [sheets[i] for i in idxs]
        sub_spans = [(max(x0, wa), min(x1, wb)) for i in idxs for x0, x1 in [spans[i]]]
        fine, meta = _one_call_custom(
            base_url,
            model,
            api_key,
            sub_sheets,
            sub_spans,
            gold.get("instruction", ""),
            sample_sec,
            timeout,
            retries,
            refine_prompt,
            max_tokens,
            failed_fallback=False,
            extra_format={"coarse_hint": f"{a:.2f}-{b:.2f}: {s.get('subtask','')}"},
        )
        fine = [
            x
            for x in fine
            if float(x["end_sec"]) > float(x["start_sec"]) + 1e-3
        ]
        if fine:
            n_refined += 1
            for x in fine:
                x["start_sec"] = max(wa, min(wb, float(x["start_sec"])))
                x["end_sec"] = max(wa, min(wb, float(x["end_sec"])))
                if x["end_sec"] > x["start_sec"] + 1e-3:
                    refined.append(x)
        else:
            refined.append(dict(s))

    refined = css.merge_adjacent(sorted(refined, key=lambda x: x["start_sec"]))
    refined = css.postprocess_segments(refined, duration, min_seg_sec=0.25)
    out = {
        "id": rid,
        "instruction": gold.get("instruction"),
        "duration_sec": duration,
        "n_sheets": len(sheets),
        "prompt_variant": refine_prompt,
        "coarse_dir": str(coarse_dir),
        "n_windows_refined": n_refined,
        "segments": refined,
    }
    (out_dir / f"{rid}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(
        f"[{rid}] S2 refined_windows={n_refined} pred={len(refined)} (coarse={len(coarse.get('segments') or [])})",
        flush=True,
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("s1", "s2"), required=True)
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
    ap.add_argument("--coarse-dir", type=Path, default=None, help="required for s2")
    ap.add_argument("--sheet-dir", type=Path, default=None)
    ap.add_argument("--base-url", default="http://127.0.0.1:18004/v1")
    ap.add_argument("--base-urls", nargs="*", default=None)
    ap.add_argument("--model", default="Qwen/Qwen3.5-397B-A17B-FP8")
    ap.add_argument("--api-key", default=os.environ.get("LOCAL_MODEL_API_KEY", "local-vllm"))
    ap.add_argument("--sample-sec", type=float, default=0.5)
    ap.add_argument("--tiles-per-sheet", type=int, default=20)
    ap.add_argument("--tile-px", type=int, default=224)
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--long-sec", type=float, default=7.0, help="S1 secondary split threshold")
    ap.add_argument("--no-long-split", action="store_true")
    ap.add_argument("--pad-sec", type=float, default=1.0, help="S2 window padding")
    ap.add_argument("--min-refine-sec", type=float, default=3.0)
    ap.add_argument(
        "--refine-prompt",
        choices=("refine_local_v1", "refine_full_cover_v1"),
        default="refine_local_v1",
        help="S2 local prompt. Use refine_full_cover_v1 for the reported V5.1 recipe.",
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    urls = args.base_urls or [args.base_url]
    golds = sorted(args.episode_dir.glob("homer_*.json"))
    if args.ids:
        want = set(args.ids)
        golds = [g for g in golds if g.stem in want]

    def job(gi, g):
        url = urls[gi % len(urls)]
        outp = args.out_dir / f"{g.stem}.json"
        if args.skip_existing and outp.exists():
            print(f"skip existing {g.stem}", flush=True)
            return
        if args.stage == "s1":
            run_s1_one(
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
                args.timeout,
                args.retries,
                args.max_tokens,
                args.long_sec,
                not args.no_long_split,
            )
        else:
            if args.coarse_dir is None:
                raise SystemExit("--coarse-dir required for s2")
            run_s2_one(
                g,
                args.video_root,
                args.coarse_dir,
                args.out_dir,
                url,
                args.model,
                args.api_key,
                args.sample_sec,
                args.tiles_per_sheet,
                args.tile_px,
                args.cols,
                args.timeout,
                args.retries,
                args.max_tokens,
                args.pad_sec,
                args.min_refine_sec,
                args.refine_prompt,
            )

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(job, i, g) for i, g in enumerate(golds)]
        for fut in as_completed(futs):
            fut.result()


if __name__ == "__main__":
    main()
