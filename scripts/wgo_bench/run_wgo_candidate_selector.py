#!/usr/bin/env python3
"""Select one subtask label from multiple WGO candidate labels.

Input JSON files are episode predictions with ``segments``. Each segment should
contain a ``candidate_labels`` object such as:

{
  "raw": "move the cup onto the tray",
  "rawprior": "...",
  "seed": "...",
  "ffmpeg": "..."
}

The script writes the same episode structure to ``--output-dir`` with
``segments[].subtask`` replaced by the selected label. Gold labels are never
read by this script.
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
import requests


SYSTEM_PROMPT = "You select the best subtask label for a fixed egocentric video segment."

USER_PROMPT_TEMPLATE = """You are selecting the best subtask label for a fixed video segment.

Segment time: [{start_sec}, {end_sec}]
Episode instruction: {instruction}

Candidates:
{candidate_list}

Pick the single candidate that best names the completed manipulation event visible in the segment.
Prefer concrete object + action + destination/state. Prefer candidates that do not invent unseen objects.
If several are equivalent, prefer the more concise imperative phrasing, usually the raw-frame candidate.

Return JSON only:
{{"selected_key": "raw", "label": "the chosen label text", "reason": "one short sentence"}}"""


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def encode_image(path: Path, max_side: int, quality: int) -> str:
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Cannot read image: {path}")
    if max_side > 0:
        h, w = image.shape[:2]
        scale = min(1.0, max_side / max(h, w))
        if scale < 1.0:
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError(f"Cannot encode image: {path}")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


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


def candidate_list(labels: dict[str, Any], preferred_order: list[str]) -> str:
    keys = [k for k in preferred_order if k in labels]
    keys += [k for k in sorted(labels) if k not in keys]
    rows = []
    for key in keys:
        value = str(labels.get(key) or "").strip()
        if value:
            rows.append(f"- {key}: {value}")
    return "\n".join(rows)


def image_paths_for_segment(args: argparse.Namespace, episode_id: str, seg_i: int, segment: dict[str, Any]) -> list[Path]:
    paths = []
    for item in segment.get("selector_images") or []:
        p = Path(item)
        if p.exists():
            paths.append(p)
    if args.image_root:
        pattern = args.image_glob_template.format(episode_id=episode_id, seg_i=seg_i)
        paths.extend(sorted(args.image_root.glob(pattern)))
    return paths[: args.max_images]


def call_selector(args: argparse.Namespace, prompt: str, image_paths: list[Path]) -> tuple[dict[str, Any], dict[str, Any]]:
    key = os.environ.get(args.api_key_env, "")
    if not key:
        raise RuntimeError(f"Set {args.api_key_env} for the OpenAI-compatible endpoint")

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for path in image_paths:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": encode_image(path, args.api_image_max_side, args.api_image_jpeg_quality)},
            }
        )
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        "temperature": 0,
        "max_tokens": args.max_tokens,
    }
    if args.disable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    last_error: Exception | None = None
    url = args.base_url.rstrip("/") + "/chat/completions"
    for attempt in range(args.max_retries + 1):
        try:
            response = requests.post(
                url,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=payload,
                timeout=args.timeout,
            )
            response.raise_for_status()
            raw = response.json()
            text = raw["choices"][0]["message"].get("content") or ""
            return extract_json(text), raw.get("usage") or {}
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= args.max_retries:
                break
            time.sleep(min(2.0 * (attempt + 1), 8.0))
    raise RuntimeError(f"selector request failed: {last_error}")


def process_episode(args: argparse.Namespace, path: Path) -> dict[str, Any]:
    episode = read_json(path)
    episode_id = str(episode.get("id") or episode.get("episode_id") or path.stem)
    instruction = str(
        episode.get("episode_annotation", {}).get("instruction")
        or episode.get("overall_instruction")
        or episode.get("instruction")
        or ""
    )
    out = dict(episode)
    out_segments = []
    for seg_i, segment in enumerate(episode.get("segments") or []):
        labels = segment.get("candidate_labels") or {}
        labels = {str(k): str(v).strip() for k, v in labels.items() if str(v).strip()}
        if not labels:
            out_segments.append(segment)
            continue
        prompt = USER_PROMPT_TEMPLATE.format(
            start_sec=segment.get("start_sec", segment.get("start_time", "")),
            end_sec=segment.get("end_sec", segment.get("end_time", "")),
            instruction=instruction,
            candidate_list=candidate_list(labels, args.preferred_order),
        )
        images = image_paths_for_segment(args, episode_id, seg_i, segment)
        result, usage = call_selector(args, prompt, images)
        selected_key = str(result.get("selected_key") or "").strip()
        selected_label = str(result.get("label") or "").strip()
        if selected_key in labels and not selected_label:
            selected_label = labels[selected_key]
        if not selected_label:
            selected_key = args.preferred_order[0] if args.preferred_order[0] in labels else next(iter(labels))
            selected_label = labels[selected_key]

        new_segment = dict(segment)
        new_segment["subtask"] = selected_label
        new_segment["selector"] = {
            "selected_key": selected_key,
            "label": selected_label,
            "reason": str(result.get("reason") or "").strip(),
            "image_count": len(images),
            "usage": usage,
        }
        out_segments.append(new_segment)
    out["id"] = episode_id
    out["segments"] = out_segments
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument(
        "--image-glob-template",
        default="{episode_id}/segment_{seg_i:03d}*.jpg",
        help="Optional glob under --image-root. Format fields: episode_id, seg_i.",
    )
    parser.add_argument("--max-images", type=int, default=6)
    parser.add_argument("--base-url", default=os.environ.get("MODEL_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME", "Qwen/Qwen3.5-397B-A17B-FP8"))
    parser.add_argument("--api-key-env", default="LOCAL_MODEL_API_KEY")
    parser.add_argument("--preferred-order", nargs="+", default=["raw", "rawprior", "seed", "ffmpeg"])
    parser.add_argument("--api-image-max-side", type=int, default=1280)
    parser.add_argument("--api-image-jpeg-quality", type=int, default=90)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--disable-thinking", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(args.candidate_dir.glob("*.json")):
        out = process_episode(args, path)
        write_json(args.output_dir / path.name, out)
        print(f"wrote {args.output_dir / path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
