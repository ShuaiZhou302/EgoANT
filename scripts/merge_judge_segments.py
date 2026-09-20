#!/usr/bin/env python3
"""Conservative sliding-window merge judge for EgoANT annotations."""

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
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from api_usage import append_usage, summarize_usage, usage_event

import cv2
import requests


DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "qwen3.5-397b-a17b"
TRANSIENT_API_ERRORS = (
    "connection refused",
    "connection reset",
    "connection aborted",
    "failed to establish a new connection",
)

SYSTEM_PROMPT = "You are a precise auditor for egocentric hand-object action annotations."

JUDGE_PROMPT = """You are given a short sliding window of consecutive source segments from one egocentric video. Provided images are middle frames from the same segments.

Each segment may contain:
- seg_id
- instruction
- actor
- verb
- main_object
- involved_objects
- contact
- final_state
- confidence
- needs_review
- motion_hint: `pure_static_candidate` means the caption contains only holding,
  keeping, steadying, or stabilizing, with no visible displacement/state-change
  verb. This is local text evidence, not a forced merge decision.

Judge each adjacent boundary independently.
For boundary [i, j], decide whether segment i and segment j should be merged.

Merge when the evidence shows one continuous visible subtask:
- adjacent in time;
- same main object, or very similar names that visually could refer to the same object;
- the descriptions read as consecutive parts of one short subtask;
- hand/tool contact, motion, or object state changes are temporally coherent;
- no clear reset to a different subtask.
- A `pure_static_candidate` normally should not remain an atomic action by
  itself. Attach it to exactly one neighboring subtask when the images and
  shared objects support continuity:
  - attach left when it is the settling or maintained final state of the left action;
  - attach right when it is the grasp/stabilization preparation for the right action;
  - if both sides are different actions, choose only the better-supported side;
  - never use one static segment to bridge two otherwise distinct actions.

Keep split when:
- either segment has instruction `N/A`;
- the main object clearly changes to a different object;
- the next segment starts an unrelated action or a new repeated trial;
- do not merge across a completed action cycle;
- the left segment already reaches a visible final state and the right segment
  starts a new transfer or repetition;
- the segments describe incompatible outcomes;
- the images or annotations are too ambiguous to support merge.
- Preserve a hold as a separate action when it is visibly task-relevant, such
  as deliberately stabilizing multiple objects or a tool, and is not merely a
  repeated description of the neighboring action.

Object consistency:
- Use images, contact, final_state, main_object, involved_objects, and main_object.name_candidates together.
- Do not split only because one segment uses a generic noun and the next uses a more specific noun, if they visually refer to the same manipulated object.
- Do not merge only because two segments happen near the same container or scene.
- Do not invent object names.

Return only valid JSON:
{
  "boundary_decisions": [
    {
      "left_seg_id": 0,
      "right_seg_id": 1,
      "merge": false,
      "reason": "brief evidence-based reason",
      "confidence": 0.0
    }
  ]
}
"""

REWRITE_PROMPT = """You are given one candidate group of consecutive source segments. A first merge stage proposed that the group may form one visible atomic action.

Use the source annotations and ordered middle-frame images as evidence.

Approve the merge only if the complete group is one continuous visible subtask:
- same main object, or very similar names that visually could refer to the same object;
- the source instructions are consecutive parts of one short subtask;
- hand/tool contact, motion, or object state changes are temporally coherent;
- no clear reset to a different subtask.

If a source segment has motion_hint=`pure_static_candidate`, treat it as a
holding/stabilization phase rather than an independent motion. Include it in
the merged action when it prepares or maintains the neighboring manipulation.
Do not let it bridge two different actions. Preserve it separately only when
stabilization itself is visibly the task-relevant goal.

Reject the merge if the group crosses a completed action cycle, or if an earlier
segment already reaches a visible final state and a later segment starts a new
transfer or repetition.

If the group should not be merged, return merge=false and do not rewrite it.

If the group should be merged, write a new annotation for the whole group.
Do not merge a group that contains any source segment with instruction `N/A`.

Writing style:
- Do not address the viewer or use second-person words such as "you" or
  "your". Refer to the left hand, right hand, or both hands directly.
- Do not use "transfer" in instruction, alternatives, or verb. Use the visible
  concrete action instead, such as move, lift, place, insert, remove, or pour.
- A meaningful hold may remain the main action when the complete group visibly
  stabilizes objects. Do not replace every use of "hold" mechanically.

Object naming:
- Use a basic noun for main_object.name.
- Do not include color, size, shape, material, or variety adjectives in main_object.name.
- If unsure of the exact object name, use a generic noun such as 'object', 'block', 'handle', or 'container'.
- If multiple objects participate, list them in involved_objects with roles.
- Use source main_object.name_candidates only as supporting evidence.
- Keep alternate names in object_aliases. Do not put aliases into main_object.name.
- Do not invent an object name from scene context.

Return 3 to 5 alternative descriptions that preserve the same visible meaning.

Return only valid JSON:
{
  "merge": true,
  "reason": "brief evidence-based reason",
  "annotation": {
    "instruction": "one canonical imperative instruction",
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
    "object_aliases": ["alternate names from source segments"],
    "contact": "visible contact or grasp | unknown",
    "final_state": "visible final state/location | unknown",
    "confidence": 0.0,
    "needs_review": false
  }
}
"""

FORMAT_RETRY = (
    "Your previous response was invalid. Return only valid JSON matching the schema exactly. "
    "Do not include markdown or extra text."
)
SEGMENT_FIELDS = (
    "seg_id",
    "instruction",
    "actor",
    "verb",
    "main_object",
    "involved_objects",
    "contact",
    "final_state",
    "confidence",
    "needs_review",
)


def json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def is_na_segment(segment: dict[str, Any]) -> bool:
    return str(segment.get("instruction", "")).strip().upper() == "N/A"


# Verbs / instruction openings that denote a static (no-progress) hold.
STATIC_VERBS = {
    "hold", "steady", "keep", "rest", "stay", "remain", "stabilize",
    "stabilise", "still", "maintain", "support", "pause", "wait",
}
STATIC_PREFIXES = (
    "hold ", "steady ", "keep ", "rest ", "stay ", "remain ", "stabiliz",
    "maintain ", "pause", "wait", "hold the", "holding",
)

DYNAMIC_ACTION_WORDS = {
    "approach", "align", "close", "grasp", "insert", "lift", "lower",
    "move", "open", "pick", "place", "pour", "pull", "push", "release",
    "remove", "retract", "rotate", "slide", "tilt", "turn",
}


def is_static_segment(segment: dict[str, Any]) -> bool:
    """True when the segment describes a static hold/steady (no visible progress)."""
    if is_na_segment(segment):
        return False
    verb = str(segment.get("verb", "")).strip().lower()
    if verb in STATIC_VERBS:
        return True
    instr = str(segment.get("instruction", "")).strip().lower()
    return any(instr.startswith(p) for p in STATIC_PREFIXES)


def is_pure_static_segment(segment: dict[str, Any]) -> bool:
    """Text-only hint for a hold/steady segment with no displacement action."""
    if not is_static_segment(segment):
        return False
    instruction = str(segment.get("instruction", "")).strip().lower()
    words = set(re.findall(r"[a-z]+", instruction))
    return not bool(words & DYNAMIC_ACTION_WORDS)


def compact_segment(segment: dict[str, Any]) -> dict[str, Any]:
    compact = {field: segment.get(field) for field in SEGMENT_FIELDS if field in segment}
    if is_pure_static_segment(segment):
        compact["motion_hint"] = "pure_static_candidate"
    return compact


def _boundary_side_frame(visual_root: Path, segment: dict[str, Any], frame_kind: str, side: str) -> Path | None:
    """Left segment -> its last sampled frame; right segment -> its first sampled frame."""
    seg_id = int(segment["seg_id"])
    frame_dir = visual_root / f"segment_{seg_id:03d}_frames_{frame_kind}"
    if not frame_dir.exists():
        return None
    frames = sorted(frame_dir.glob("*.png"))
    if not frames:
        return None
    return frames[-1] if side == "end" else frames[0]


def boundary_frame_similarity(
    visual_root: Path, left: dict[str, Any], right: dict[str, Any], frame_kind: str = "raw"
) -> float | None:
    """Visual similarity (0..1) of the frames straddling the boundary: left's last vs right's first.

    1.0 = the frames are nearly identical (scene + hand barely moved across the cut), low = a big
    visual change. For fixed ego cameras a global HSV histogram is useless (background dominates and
    never changes), so we use the fraction of pixels whose grayscale value changed by >30, which
    tracks hand/object motion. Scaled so ~10% changed pixels -> 0. Pure pixels, ignores the
    (possibly wrong) text annotation.
    """
    lp = _boundary_side_frame(visual_root, left, frame_kind, "end")
    rp = _boundary_side_frame(visual_root, right, frame_kind, "start")
    if lp is None or rp is None:
        return None
    a = cv2.imread(str(lp))
    b = cv2.imread(str(rp))
    if a is None or b is None:
        return None
    ga = cv2.cvtColor(cv2.resize(a, (256, 256)), cv2.COLOR_BGR2GRAY).astype("float32")
    gb = cv2.cvtColor(cv2.resize(b, (256, 256)), cv2.COLOR_BGR2GRAY).astype("float32")
    changed = float((cv2.absdiff(ga, gb) > 30).mean())  # fraction of pixels that moved
    return max(0.0, 1.0 - changed * 10.0)


def apply_rule_overrides(
    boundaries: dict[tuple[int, int], dict[str, Any]],
    segments: list[dict[str, Any]],
    visual_root: Path,
    args: argparse.Namespace,
) -> dict[tuple[int, int], dict[str, Any]]:
    """Force-merge boundaries by cheap rules (no model): high frame similarity, or both static.

    Never merges across an N/A boundary. Records frame_similarity + rule_override on the boundary.
    """
    seg_by_id = {int(s["seg_id"]): s for s in segments}
    for (i, j), bd in boundaries.items():
        left, right = seg_by_id.get(i), seg_by_id.get(j)
        if left is None or right is None or is_na_segment(left) or is_na_segment(right):
            continue
        reasons: list[str] = []
        if getattr(args, "static_merge", True) and is_static_segment(left) and is_static_segment(right):
            reasons.append("both segments are static (hold/steady)")
        if getattr(args, "cv_merge_threshold", 0.0) and getattr(args, "cv_merge_threshold") > 0:
            sim = boundary_frame_similarity(visual_root, left, right, args.cv_frame_kind)
            if sim is not None:
                bd["frame_similarity"] = round(sim, 3)
                if sim >= args.cv_merge_threshold:
                    reasons.append(f"boundary frame similarity {sim:.2f} >= {args.cv_merge_threshold}")
        if reasons and not bd.get("merge"):
            bd["merge"] = True
            bd["rule_override"] = reasons
            bd["reason"] = "rule override: " + "; ".join(reasons)
    return boundaries


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
    quality = max(1, min(100, int(jpeg_quality)))
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError(f"Cannot encode image for API payload: {path}")
    data = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{data}"


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


def frame_index_from_path(path: Path) -> int | None:
    match = re.search(r"_(\d+)\.png$", path.name)
    return int(match.group(1)) if match else None


def find_middle_frame(output_root: Path, segment: dict[str, Any], frame_kind: str) -> Path | None:
    frame_dir = output_root / f"segment_{int(segment['seg_id']):03d}_frames_{frame_kind}"
    if not frame_dir.exists():
        return None
    candidates = sorted(frame_dir.glob("*.png"))
    if not candidates:
        return None
    center = 0.5 * (int(segment["start_frame"]) + int(segment["end_frame"]))

    def score(path: Path) -> float:
        frame_idx = frame_index_from_path(path)
        return abs(frame_idx - center) if frame_idx is not None else float("inf")

    return min(candidates, key=score)


def build_prompt(
    segments: list[dict[str, Any]],
    central_boundaries: set[tuple[int, int]],
    cv_hints: dict[tuple[int, int], float] | None = None,
) -> str:
    compact = [compact_segment(segment) for segment in segments]
    expected = [[a, b] for a, b in central_boundaries]
    prompt = (
        JUDGE_PROMPT
        + "\n\nSource segments:\n"
        + json.dumps(compact, indent=2, ensure_ascii=False)
        + "\n\nJudge these adjacent boundaries:\n"
        + json.dumps(expected, indent=2, ensure_ascii=False)
    )
    if cv_hints:
        lines = [
            f"- [{i}, {j}] frame_similarity = {sim:.2f}"
            for (i, j), sim in sorted(cv_hints.items())
            if sim is not None
        ]
        if lines:
            prompt += (
                "\n\nPrecomputed visual cue (pixels, not annotation): similarity of the frames "
                "straddling each boundary, 1.0 = the scene barely changed across the cut, low = a "
                "big visual change. High similarity is strong evidence the two segments are one "
                "continuous action and should merge:\n" + "\n".join(lines)
            )
    return prompt


def validate_judge_result(result: dict[str, Any], expected: set[tuple[int, int]]) -> list[dict[str, Any]]:
    decisions = result.get("boundary_decisions")
    if not isinstance(decisions, list):
        raise ValueError("Missing list field `boundary_decisions`")
    by_boundary: dict[tuple[int, int], dict[str, Any]] = {}
    for item in decisions:
        boundary = (int(item.get("left_seg_id")), int(item.get("right_seg_id")))
        if boundary not in expected:
            continue
        try:
            confidence = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        by_boundary[boundary] = {
            "left_seg_id": boundary[0],
            "right_seg_id": boundary[1],
            "merge": bool(item.get("merge")),
            "reason": str(item.get("reason", "")),
            "confidence": max(0.0, min(1.0, confidence)),
        }
    missing = sorted(expected - set(by_boundary))
    if missing:
        raise ValueError(f"Missing boundary decisions: {missing}")
    return [by_boundary[boundary] for boundary in sorted(expected)]


def call_judge_api(
    prompt: str,
    image_paths: list[Path],
    expected: set[tuple[int, int]],
    base_url: str,
    model: str,
    api_key_env: str,
    timeout: float,
    max_tokens: int,
    retries: int,
    api_image_max_side: int | None,
    api_image_jpeg_quality: int,
    disable_thinking: bool,
    usage_path: Path,
) -> list[dict[str, Any]]:
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key environment variable: {api_key_env}")
    url = f"{base_url.rstrip('/')}/chat/completions"
    errors: list[str] = []
    for attempt in range(retries + 1):
        attempt_prompt = prompt if attempt == 0 else f"{prompt}\n\n{FORMAT_RETRY}"
        if image_paths:
            content: str | list[dict[str, Any]] = [{"type": "text", "text": attempt_prompt}]
            content.extend(
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_data_url(path, api_image_max_side, api_image_jpeg_quality)},
                }
                for path in image_paths
            )
        else:
            content = attempt_prompt
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        if disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
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
            result = extract_json(raw)
            decisions = validate_judge_result(result, expected)
            append_usage(
                usage_path,
                usage_event(
                    response_data,
                    stage="merge_judge",
                    model=model,
                    latency_sec=time.monotonic() - started,
                    success=True,
                    attempt=attempt + 1,
                ),
            )
            return decisions
        except Exception as exc:
            maybe_restart_api_endpoint(exc, attempt < retries)
            append_usage(
                usage_path,
                usage_event(
                    response_data,
                    stage="merge_judge",
                    model=model,
                    latency_sec=time.monotonic() - started,
                    success=False,
                    attempt=attempt + 1,
                    error=str(exc),
                ),
            )
            errors.append(str(exc))
    raise RuntimeError("Merge judge API failed: " + " | ".join(errors))


def windows(n: int, window_size: int, overlap: int) -> list[tuple[int, int]]:
    if n <= 1:
        return []
    stride = max(1, window_size - overlap)
    out: list[tuple[int, int]] = []
    start = 0
    while start < n - 1:
        end = min(n, start + window_size)
        out.append((start, end))
        if end == n:
            break
        start += stride
    return out


def central_boundaries_for_window(start: int, end: int, n: int, overlap: int) -> set[tuple[int, int]]:
    first = start
    last = end - 2
    if start > 0:
        first += overlap // 2
    if end < n:
        last -= max(0, overlap - overlap // 2 - 1)
    boundaries: set[tuple[int, int]] = set()
    for idx in range(first, last + 1):
        boundaries.add((idx, idx + 1))
    return boundaries


def include_images(prompt: str, visual_root: Path, segments: list[dict[str, Any]], frame_kind: str) -> tuple[str, list[Path]]:
    lines = ["\n\nMiddle frame images:"]
    image_paths: list[Path] = []
    for segment in segments:
        image_path = find_middle_frame(visual_root, segment, frame_kind)
        if image_path is None:
            continue
        image_paths.append(image_path)
        lines.append(f"Image {len(image_paths)} corresponds to source segment {segment['seg_id']}.")
    return (prompt + "\n".join(lines), image_paths) if image_paths else (prompt, image_paths)


def save_prompt_images(image_paths: list[Path], image_dir: Path) -> None:
    if not image_paths:
        return
    image_dir.mkdir(parents=True, exist_ok=True)
    for idx, image_path in enumerate(image_paths, start=1):
        shutil.copy2(image_path, image_dir / f"image_{idx:02d}_{image_path.name}")


def decision_is_merge(decision: dict[str, Any], threshold: float) -> bool:
    if not decision.get("merge"):
        return False
    try:
        confidence = float(decision.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return confidence >= threshold


def aggregate_decisions(
    decisions_by_boundary: dict[tuple[int, int], list[dict[str, Any]]],
    threshold: float,
) -> dict[tuple[int, int], dict[str, Any]]:
    aggregated: dict[tuple[int, int], dict[str, Any]] = {}
    for boundary, votes in decisions_by_boundary.items():
        valid_merges = [decision_is_merge(vote, threshold) for vote in votes]
        merge = bool(votes) and all(valid_merges)
        confidences = [float(vote.get("confidence", 0.0) or 0.0) for vote in votes]
        aggregated[boundary] = {
            "left_seg_id": boundary[0],
            "right_seg_id": boundary[1],
            "merge": merge,
            "vote_count": len(votes),
            "votes": votes,
            "confidence": min(confidences) if confidences else 0.0,
            "reason": "all judge votes passed confidence threshold" if merge else "split by judge confidence rule",
        }
    return aggregated


def representative_instruction(group: list[dict[str, Any]]) -> str:
    candidates = [s for s in group if str(s.get("instruction", "")).strip() and s.get("instruction") != "N/A"]
    if not candidates:
        return "N/A"
    return max(candidates, key=lambda s: (float(s.get("confidence", 0.0) or 0.0), s["end_frame"] - s["start_frame"])).get(
        "instruction", "N/A"
    )


def representative_field(group: list[dict[str, Any]], field: str, default: str) -> str:
    values = [str(s.get(field, "")).strip() for s in group if str(s.get(field, "")).strip()]
    values = [v for v in values if v not in {"N/A", "unknown"}]
    if not values:
        return default
    return max(set(values), key=values.count)


def representative_main_object(group: list[dict[str, Any]]) -> dict[str, Any]:
    objects = [s.get("main_object") for s in group if isinstance(s.get("main_object"), dict)]
    names = [
        str(obj.get("name", "")).strip()
        for obj in objects
        if str(obj.get("name", "")).strip() not in {"", "N/A", "unknown"}
    ]
    name = max(set(names), key=names.count) if names else "unknown"
    candidates: list[dict[str, Any]] = []
    for obj in objects:
        raw_candidates = obj.get("name_candidates", [])
        if isinstance(raw_candidates, list):
            candidates.extend(candidate for candidate in raw_candidates if isinstance(candidate, dict))
    return {
        "name": name,
        "name_candidates": candidates[:3],
        "confidence": max((float(obj.get("confidence", 0.0) or 0.0) for obj in objects), default=0.0),
    }


def representative_involved_objects(group: list[dict[str, Any]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, str]] = []
    for segment in group:
        objects = segment.get("involved_objects", [])
        if not isinstance(objects, list):
            continue
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            name = str(obj.get("name", "unknown"))
            role = str(obj.get("role", "other"))
            key = (name, role)
            if key in seen:
                continue
            seen.add(key)
            out.append({"name": name, "role": role})
            if len(out) >= 4:
                return out
    return out


def build_merged_segments(segments: list[dict[str, Any]], boundaries: dict[tuple[int, int], dict[str, Any]]) -> list[dict[str, Any]]:
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = [segments[0]]
    for left, right in zip(segments[:-1], segments[1:]):
        boundary = boundaries.get((int(left["seg_id"]), int(right["seg_id"])))
        if boundary and boundary.get("merge"):
            current.append(right)
        else:
            groups.append(current)
            current = [right]
    groups.append(current)

    merged: list[dict[str, Any]] = []
    for group_id, group in enumerate(groups):
        source_ids = [int(s["seg_id"]) for s in group]
        merged.append(
            {
                "seg_id": group_id,
                "source_seg_ids": source_ids,
                "start_frame": int(group[0]["start_frame"]),
                "end_frame": int(group[-1]["end_frame"]),
                "instruction": representative_instruction(group),
                "actor": representative_field(group, "actor", "unknown"),
                "verb": representative_field(group, "verb", "N/A"),
                "main_object": representative_main_object(group),
                "involved_objects": representative_involved_objects(group),
                "needs_review": len(group) > 1 or any(s.get("needs_review") for s in group),
                "merge_reason": "single segment" if len(group) == 1 else "adjacent boundaries passed merge judge",
            }
        )
    return merged


def build_candidate_groups(
    segments: list[dict[str, Any]], boundaries: dict[tuple[int, int], dict[str, Any]]
) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    current = [segments[0]]
    for left, right in zip(segments[:-1], segments[1:]):
        boundary = boundaries.get((int(left["seg_id"]), int(right["seg_id"])))
        if boundary and boundary.get("merge"):
            current.append(right)
        else:
            groups.append(current)
            current = [right]
    groups.append(current)
    return groups


def boundary_strength(boundary: dict[str, Any] | None) -> tuple[float, float]:
    if not boundary:
        return (-1.0, -1.0)
    confidences = [float(vote.get("confidence", 0.0) or 0.0) for vote in boundary.get("votes", [])]
    if not confidences:
        return (-1.0, -1.0)
    return (min(confidences), sum(confidences) / len(confidences))


def weakest_split_index(group: list[dict[str, Any]], boundaries: dict[tuple[int, int], dict[str, Any]]) -> int:
    candidates: list[tuple[tuple[float, float], int]] = []
    for idx, (left, right) in enumerate(zip(group[:-1], group[1:]), start=1):
        boundary = boundaries.get((int(left["seg_id"]), int(right["seg_id"])))
        candidates.append((boundary_strength(boundary), idx))
    return min(candidates, key=lambda item: item[0])[1]


def group_duration_sec(group: list[dict[str, Any]], fps: float) -> float:
    return (int(group[-1]["end_frame"]) - int(group[0]["start_frame"])) / fps


def split_oversized_group(
    group: list[dict[str, Any]],
    boundaries: dict[tuple[int, int], dict[str, Any]],
    fps: float,
    max_segments: int,
    max_duration_sec: float,
) -> list[list[dict[str, Any]]]:
    if len(group) <= max_segments and group_duration_sec(group, fps) <= max_duration_sec:
        return [group]
    split_idx = weakest_split_index(group, boundaries)
    return split_oversized_group(group[:split_idx], boundaries, fps, max_segments, max_duration_sec) + split_oversized_group(
        group[split_idx:], boundaries, fps, max_segments, max_duration_sec
    )


def validate_rewrite_result(result: dict[str, Any]) -> dict[str, Any]:
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
        if not isinstance(value, dict):
            raise ValueError("Merged group annotation `main_object` must be an object")
        return {
            "name": str(value.get("name", "unknown")),
            "name_candidates": normalize_candidates(value.get("name_candidates", [])),
            "confidence": as_float(value.get("confidence", 0.0)),
        }

    def normalize_involved_objects(value: Any) -> list[dict[str, str]]:
        if not isinstance(value, list):
            return []
        out = []
        for item in value[:4]:
            if not isinstance(item, dict):
                continue
            out.append({"name": str(item.get("name", "unknown")), "role": str(item.get("role", "other"))})
        return out

    merge = bool(result.get("merge"))
    reason = str(result.get("reason", ""))
    if not merge:
        return {"merge": False, "reason": reason}
    annotation = result.get("annotation")
    if not isinstance(annotation, dict):
        raise ValueError("Merged group response is missing `annotation`")
    required = (
        "instruction",
        "alternatives",
        "actor",
        "verb",
        "main_object",
        "involved_objects",
        "object_aliases",
        "contact",
        "final_state",
        "confidence",
        "needs_review",
    )
    for field in required:
        if field not in annotation:
            raise ValueError(f"Merged group annotation is missing `{field}`")
    if not isinstance(annotation["alternatives"], list) or not 3 <= len(annotation["alternatives"]) <= 5:
        raise ValueError("Merged group annotation must contain 3 to 5 alternatives")
    if not isinstance(annotation["object_aliases"], list):
        raise ValueError("Merged group annotation `object_aliases` must be a list")
    annotation["instruction"] = normalize_action_text(annotation["instruction"])
    annotation["alternatives"] = [normalize_action_text(item) for item in annotation["alternatives"][:5]]
    annotation["verb"] = normalize_action_text(annotation["verb"])
    annotation["main_object"] = normalize_main_object(annotation["main_object"])
    annotation["involved_objects"] = normalize_involved_objects(annotation["involved_objects"])
    annotation["object_aliases"] = [str(item) for item in annotation["object_aliases"]]
    annotation["confidence"] = float(annotation["confidence"])
    annotation["needs_review"] = bool(annotation["needs_review"])
    return {"merge": True, "reason": reason, "annotation": annotation}


def build_rewrite_prompt(group: list[dict[str, Any]]) -> str:
    return REWRITE_PROMPT + "\n\nCandidate source segments:\n" + json.dumps(
        [compact_segment(segment) for segment in group], indent=2, ensure_ascii=False
    )


def call_rewrite_api(
    prompt: str,
    image_paths: list[Path],
    base_url: str,
    model: str,
    api_key_env: str,
    timeout: float,
    max_tokens: int,
    retries: int,
    api_image_max_side: int | None,
    api_image_jpeg_quality: int,
    disable_thinking: bool,
    usage_path: Path,
) -> dict[str, Any]:
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key environment variable: {api_key_env}")
    url = f"{base_url.rstrip('/')}/chat/completions"
    errors: list[str] = []
    for attempt in range(retries + 1):
        attempt_prompt = prompt if attempt == 0 else f"{prompt}\n\n{FORMAT_RETRY}"
        content: str | list[dict[str, Any]]
        if image_paths:
            content = [{"type": "text", "text": attempt_prompt}]
            content.extend(
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_data_url(path, api_image_max_side, api_image_jpeg_quality)},
                }
                for path in image_paths
            )
        else:
            content = attempt_prompt
        started = time.monotonic()
        response_data: dict[str, Any] | None = None
        try:
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                "temperature": 0.0,
                "max_tokens": max_tokens,
            }
            if disable_thinking:
                payload["chat_template_kwargs"] = {"enable_thinking": False}
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
            result = validate_rewrite_result(extract_json(raw))
            append_usage(
                usage_path,
                usage_event(
                    response_data,
                    stage="merge_rewrite",
                    model=model,
                    latency_sec=time.monotonic() - started,
                    success=True,
                    attempt=attempt + 1,
                ),
            )
            return result
        except Exception as exc:
            maybe_restart_api_endpoint(exc, attempt < retries)
            append_usage(
                usage_path,
                usage_event(
                    response_data,
                    stage="merge_rewrite",
                    model=model,
                    latency_sec=time.monotonic() - started,
                    success=False,
                    attempt=attempt + 1,
                    error=str(exc),
                ),
            )
            errors.append(str(exc))
    raise RuntimeError("Merge rewrite API failed: " + " | ".join(errors))


def preserve_single_segment(segment: dict[str, Any], reason: str) -> dict[str, Any]:
    preserved = dict(segment)
    preserved["source_seg_ids"] = [int(segment["seg_id"])]
    preserved["merge_reason"] = reason
    return preserved


def formal_segment(segment: dict[str, Any], seg_id: int) -> dict[str, Any]:
    internal_fields = {"seg_id", "source_seg_ids", "merge_reason"}
    formal = {
        "seg_id": seg_id,
        "start_frame": int(segment["start_frame"]),
        "end_frame": int(segment["end_frame"]),
    }
    formal.update(
        {
            key: value
            for key, value in segment.items()
            if key not in internal_fields and key not in {"start_frame", "end_frame"}
        }
    )
    return formal


def formal_annotations(source: dict[str, Any], segments: list[dict[str, Any]], fps: float) -> dict[str, Any]:
    nframes = int(source.get("nframes") or max(int(segment["end_frame"]) for segment in segments))
    return {
        "fps": fps,
        "nframes": nframes,
        "duration_sec": round(nframes / fps, 3),
        "segments": [formal_segment(segment, seg_id) for seg_id, segment in enumerate(segments)],
    }


def finalize_candidate_group(
    group: list[dict[str, Any]],
    boundaries: dict[tuple[int, int], dict[str, Any]],
    visual_root: Path,
    frame_kind: str,
    aux_dir: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if len(group) == 1:
        return [preserve_single_segment(group[0], "single source segment")]
    if any(is_na_segment(segment) for segment in group):
        return [preserve_single_segment(segment, "N/A source segment preserved without merge") for segment in group]

    source_ids = [int(segment["seg_id"]) for segment in group]
    group_name = "_".join(f"{seg_id:03d}" for seg_id in source_ids)
    prompt = build_rewrite_prompt(group)
    prompt, image_paths = include_images(prompt, visual_root, group, frame_kind)
    prompt_path = aux_dir / f"rewrite_group_{group_name}_prompt.txt"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt, encoding="utf-8")
    save_prompt_images(image_paths, aux_dir / f"rewrite_group_{group_name}_images")

    try:
        result = call_rewrite_api(
            prompt=prompt,
            image_paths=image_paths,
            base_url=args.base_url,
            model=args.rewrite_model,
            api_key_env=args.api_key_env,
            timeout=args.timeout,
            max_tokens=args.max_tokens,
            retries=args.api_retries,
            api_image_max_side=args.api_image_max_side,
            api_image_jpeg_quality=args.api_image_jpeg_quality,
            disable_thinking=args.disable_thinking,
            usage_path=aux_dir.parent / "api_usage.jsonl",
        )
        json_dump(aux_dir / f"rewrite_group_{group_name}_result.json", result)
    except Exception as exc:
        json_dump(aux_dir / f"rewrite_group_{group_name}_error.json", {"error": str(exc)})
        result = {"merge": False, "reason": "rewrite call failed; preserve source boundaries"}

    if result["merge"]:
        annotation = dict(result["annotation"])
        annotation.update(
            {
                "source_seg_ids": source_ids,
                "start_frame": int(group[0]["start_frame"]),
                "end_frame": int(group[-1]["end_frame"]),
                "merge_reason": result["reason"],
            }
        )
        return [annotation]

    split_idx = weakest_split_index(group, boundaries)
    return finalize_candidate_group(
        group[:split_idx], boundaries, visual_root, frame_kind, aux_dir, args
    ) + finalize_candidate_group(group[split_idx:], boundaries, visual_root, frame_kind, aux_dir, args)


def judge_window(
    window_start: int,
    window_end: int,
    segments: list[dict[str, Any]],
    visual_root: Path,
    aux_dir: Path,
    args: argparse.Namespace,
) -> tuple[int, int, list[dict[str, Any]]]:
    chunk = segments[window_start:window_end]
    expected = {
        (int(a["seg_id"]), int(b["seg_id"]))
        for a, b in zip(chunk[:-1], chunk[1:])
        if not is_na_segment(a) and not is_na_segment(b)
    }
    if not expected:
        return window_start, window_end, []
    cv_hints: dict[tuple[int, int], float] = {}
    if getattr(args, "cv_frame_hint", True):
        seg_by_id = {int(s["seg_id"]): s for s in chunk}
        for i, j in expected:
            left, right = seg_by_id.get(i), seg_by_id.get(j)
            if left is not None and right is not None:
                sim = boundary_frame_similarity(visual_root, left, right, args.cv_frame_kind)
                if sim is not None:
                    cv_hints[(i, j)] = sim
    prompt = build_prompt(chunk, expected, cv_hints)
    image_paths: list[Path] = []
    if args.include_middle_frame:
        prompt, image_paths = include_images(prompt, visual_root, chunk, args.frame_kind)
    prompt_path = aux_dir / f"judge_window_{window_start:03d}_{window_end:03d}_prompt.txt"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt, encoding="utf-8")
    save_prompt_images(image_paths, aux_dir / f"judge_window_{window_start:03d}_{window_end:03d}_images")

    if args.dry_run:
        decisions = [
            {
                "left_seg_id": left,
                "right_seg_id": right,
                "merge": False,
                "reason": "dry-run split",
                "confidence": 0.0,
            }
            for left, right in sorted(expected)
        ]
    else:
        try:
            decisions = call_judge_api(
                prompt=prompt,
                image_paths=image_paths,
                expected=expected,
                base_url=args.base_url,
                model=args.model,
                api_key_env=args.api_key_env,
                timeout=args.timeout,
                max_tokens=args.max_tokens,
                retries=args.api_retries,
                api_image_max_side=args.api_image_max_side,
                api_image_jpeg_quality=args.api_image_jpeg_quality,
                disable_thinking=args.disable_thinking,
                usage_path=aux_dir.parent / "api_usage.jsonl",
            )
        except Exception as exc:
            json_dump(aux_dir / f"judge_window_{window_start:03d}_{window_end:03d}_error.json", {"error": str(exc)})
            decisions = [
                {
                    "left_seg_id": left,
                    "right_seg_id": right,
                    "merge": False,
                    "reason": "judge call failed; conservative split",
                    "confidence": 0.0,
                }
                for left, right in sorted(expected)
            ]
    json_dump(aux_dir / f"judge_window_{window_start:03d}_{window_end:03d}_result.json", {"boundary_decisions": decisions})
    return window_start, window_end, decisions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge adjacent boundaries and merge EgoANT annotations conservatively.")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--overlap", type=int, default=2)
    parser.add_argument("--score-threshold", type=int, default=4, help="Deprecated V2 score threshold; ignored by V3 confidence voting.")
    parser.add_argument("--merge-confidence-threshold", type=float, default=0.75)
    parser.add_argument(
        "--cv-merge-threshold",
        type=float,
        default=0.90,
        help="If the frames straddling a boundary have HSV-histogram similarity >= this, force-merge "
        "(no model). Set 0 to disable the CV rule.",
    )
    parser.add_argument("--cv-frame-kind", choices=("raw", "overlay"), default="raw",
                        help="Which saved frames to use for the CV similarity check.")
    parser.add_argument("--cv-frame-hint", dest="cv_frame_hint", action="store_true", default=True,
                        help="Also include the boundary frame_similarity number in the judge prompt.")
    parser.add_argument("--no-cv-frame-hint", dest="cv_frame_hint", action="store_false")
    parser.add_argument("--static-merge", dest="static_merge", action="store_true", default=True,
                        help="Force-merge a boundary when both segments are static holds (hold/steady/...).")
    parser.add_argument("--no-static-merge", dest="static_merge", action="store_false")
    parser.add_argument("--visual-root", type=Path, default=None)
    parser.add_argument("--include-middle-frame", action="store_true")
    parser.add_argument("--frame-kind", choices=("overlay", "raw"), default="overlay")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--rewrite-model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key-env", default="ANNOTATION_API_KEY")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-tokens", type=int, default=65536)
    parser.add_argument("--api-retries", type=int, default=2)
    parser.add_argument(
        "--api-image-max-side",
        type=int,
        default=640,
        help="Resize API payload images to this max side length (keeps aspect ratio). Saved prompt images are unchanged. Default 640 unifies input resolution across datasets.",
    )
    parser.add_argument(
        "--api-image-jpeg-quality",
        type=int,
        default=85,
        help="JPEG quality for resized API payload images.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Judge API concurrency. Use 0 or a negative value to submit all judge windows at once.",
    )
    parser.add_argument(
        "--rewrite-workers",
        type=int,
        default=None,
        help="Rewrite API concurrency. Defaults to --workers. Use 0 or a negative value to submit all rewrite groups at once.",
    )
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--max-candidate-segments", type=int, default=8)
    parser.add_argument("--max-candidate-sec", type=float, default=15.0)
    parser.add_argument("--skip-rewrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Disable Qwen thinking on compatible local vLLM endpoints.",
    )
    return parser.parse_args()


def resolve_worker_count(requested: int | None, task_count: int, fallback: int | None = None) -> int:
    if task_count <= 0:
        return 1
    value = fallback if requested is None else requested
    if value is None or value <= 0:
        return task_count
    return max(1, min(value, task_count))


def main() -> int:
    args = parse_args()
    data = json_load(args.annotations)
    segments = data.get("segments", [])
    output = args.output or args.annotations.with_name("merge_judge_annotations.json")
    aux_dir = output.with_suffix("")
    fps = float(args.fps or data.get("fps") or 30.0)

    if len(segments) <= 1:
        merged_segments = [preserve_single_segment(segment, "single segment passthrough") for segment in segments]
        debug_output = output.with_name(output.stem + "_debug.json")
        json_dump(output.with_name(output.stem + "_debug_boundaries.json"), {"boundaries": []})
        json_dump(
            debug_output,
            {
                "source_annotations": str(args.annotations),
                "model": args.model,
                "rewrite_model": args.rewrite_model,
                "window_size": args.window_size,
                "overlap": args.overlap,
                "score_threshold": args.score_threshold,
                "merge_confidence_threshold": args.merge_confidence_threshold,
                "workers": args.workers,
                "resolved_judge_workers": 0,
                "rewrite_workers": args.rewrite_workers if args.rewrite_workers is not None else args.workers,
                "resolved_rewrite_workers": 0,
                "fps": fps,
                "max_candidate_segments": args.max_candidate_segments,
                "max_candidate_sec": args.max_candidate_sec,
                "candidate_groups": [[int(segment["seg_id"]) for segment in merged_segments]] if merged_segments else [],
                "bounded_candidate_groups": [[int(segment["seg_id"]) for segment in merged_segments]] if merged_segments else [],
                "boundaries": [],
                "merged_segments": merged_segments,
                "passthrough_reason": "merge judge skipped because there are fewer than two segments",
            },
        )
        json_dump(output, formal_annotations(data, merged_segments, fps))
        summarize_usage(output.parent / "api_usage.jsonl", output.parent / "token_usage.json")
        print(output)
        return 0

    visual_root = args.visual_root or args.annotations.parent
    all_possible = {(int(a["seg_id"]), int(b["seg_id"])) for a, b in zip(segments[:-1], segments[1:])}
    decisions_by_boundary: dict[tuple[int, int], list[dict[str, Any]]] = {boundary: [] for boundary in all_possible}

    window_ranges = windows(len(segments), args.window_size, args.overlap)
    judge_workers = resolve_worker_count(args.workers, len(window_ranges))
    with ThreadPoolExecutor(max_workers=judge_workers) as executor:
        futures = [
            executor.submit(judge_window, start, end, segments, visual_root, aux_dir, args)
            for start, end in window_ranges
        ]
        completed = 0
        for future in as_completed(futures):
            window_start, window_end, decisions = future.result()
            completed += 1
            print(f"judge windows: {completed}/{len(window_ranges)} ({window_start}-{window_end})", flush=True)
            for decision in decisions:
                boundary = (int(decision["left_seg_id"]), int(decision["right_seg_id"]))
                decisions_by_boundary.setdefault(boundary, []).append(decision)

    boundaries = aggregate_decisions(decisions_by_boundary, args.merge_confidence_threshold)
    boundaries = apply_rule_overrides(boundaries, segments, visual_root, args)
    forced = sum(1 for b in boundaries.values() if b.get("rule_override"))
    print(f"rule overrides forced merge on {forced} boundaries (cv similarity / static)", flush=True)
    candidate_groups = build_candidate_groups(segments, boundaries)
    bounded_groups = [
        bounded
        for group in candidate_groups
        for bounded in split_oversized_group(
            group,
            boundaries,
            fps=fps,
            max_segments=args.max_candidate_segments,
            max_duration_sec=args.max_candidate_sec,
        )
    ]
    if args.skip_rewrite or args.dry_run:
        merged_segments = build_merged_segments(segments, boundaries)
    else:
        rewrite_workers = resolve_worker_count(args.rewrite_workers, len(bounded_groups), fallback=args.workers)
        with ThreadPoolExecutor(max_workers=rewrite_workers) as executor:
            rewritten_groups = list(
                executor.map(
                    lambda group: finalize_candidate_group(group, boundaries, visual_root, args.frame_kind, aux_dir, args),
                    bounded_groups,
                )
            )
        merged_segments = [final for group in rewritten_groups for final in group]
    for seg_id, segment in enumerate(merged_segments):
        segment["seg_id"] = seg_id
    debug_output = output.with_name(output.stem + "_debug.json")
    debug_boundaries = list(boundaries.values())
    json_dump(output.with_name(output.stem + "_debug_boundaries.json"), {"boundaries": debug_boundaries})
    json_dump(
        debug_output,
        {
            "source_annotations": str(args.annotations),
            "model": args.model,
            "rewrite_model": args.rewrite_model,
            "window_size": args.window_size,
            "overlap": args.overlap,
            "score_threshold": args.score_threshold,
            "merge_confidence_threshold": args.merge_confidence_threshold,
            "workers": args.workers,
            "resolved_judge_workers": judge_workers,
            "rewrite_workers": args.rewrite_workers if args.rewrite_workers is not None else args.workers,
            "resolved_rewrite_workers": None if args.skip_rewrite or args.dry_run else rewrite_workers,
            "fps": fps,
            "max_candidate_segments": args.max_candidate_segments,
            "max_candidate_sec": args.max_candidate_sec,
            "candidate_groups": [[int(segment["seg_id"]) for segment in group] for group in candidate_groups],
            "bounded_candidate_groups": [[int(segment["seg_id"]) for segment in group] for group in bounded_groups],
            "boundaries": debug_boundaries,
            "merged_segments": merged_segments,
        },
    )
    json_dump(output, formal_annotations(data, merged_segments, fps))
    summarize_usage(output.parent / "api_usage.jsonl", output.parent / "token_usage.json")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
