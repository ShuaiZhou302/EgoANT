#!/usr/bin/env python3
"""Add a provided or generated episode-level annotation to a final JSON file."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import requests

try:
    from api_usage import append_usage, summarize_usage, usage_event
except ModuleNotFoundError:
    from scripts.api_usage import append_usage, summarize_usage, usage_event


SYSTEM_PROMPT = (
    "You convert ordered atomic actions into a short task-level instruction for an "
    "egocentric manipulation episode."
)
TRANSIENT_API_ERRORS = (
    "connection refused",
    "connection reset",
    "connection aborted",
    "failed to establish a new connection",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--provided", help="Dataset-provided episode annotation. Skips API generation.")
    parser.add_argument("--provided-file", type=Path, help="JSON or text file containing a dataset-provided episode annotation.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="qwen3.5-397b-a17b")
    parser.add_argument("--api-key-env", default="ANNOTATION_API_KEY")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--api-retries", type=int, default=2)
    parser.add_argument("--disable-thinking", action="store_true")
    return parser.parse_args()


def normalized_instruction(value: Any) -> str:
    if isinstance(value, dict):
        return ""
    return re.sub(r"\s+", " ", str(value or "")).strip()


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


def load_provided_file(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return ""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return normalized_instruction(text)

    if isinstance(data, str):
        return normalized_instruction(data)
    if not isinstance(data, dict):
        return normalized_instruction(text)

    candidates: list[Any] = [
        data.get("instruction"),
        data.get("description"),
        data.get("task_description"),
        data.get("task_name"),
    ]
    episode = data.get("episode")
    if isinstance(episode, dict):
        candidates.extend([episode.get("instruction"), episode.get("description"), episode.get("task_name")])
    episode_annotation = data.get("episode_annotation")
    if isinstance(episode_annotation, dict):
        candidates.extend([episode_annotation.get("instruction"), episode_annotation.get("description")])

    for candidate in candidates:
        value = normalized_instruction(candidate)
        if value:
            return value
    return normalized_instruction(text)


def compact_instructions(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for segment in segments:
        instruction = normalized_instruction(segment.get("instruction"))
        if not instruction or instruction.upper() == "N/A":
            continue
        if compact and compact[-1]["instruction"].casefold() == instruction.casefold():
            compact[-1]["repeat_count"] += 1
            compact[-1]["end_frame"] = segment.get("end_frame")
            continue
        compact.append(
            {
                "instruction": instruction,
                "repeat_count": 1,
                "start_frame": segment.get("start_frame"),
                "end_frame": segment.get("end_frame"),
            }
        )
    return compact


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    elif not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    return json.loads(text)


def build_prompt(actions: list[dict[str, Any]]) -> str:
    return (
        "The following ordered annotations describe one complete egocentric video episode.\n"
        "Write one short task-level imperative instruction, usually 4 to 12 words.\n"
        "Describe the dominant overall task, not the person and not the sequence of steps.\n"
        "Start directly with a concrete action verb, such as Pack, Fold, Sort, Scoop, or Arrange.\n"
        "Do not start with 'The person', 'A person', 'Someone', or a personal pronoun.\n"
        "Keep only the main action, main manipulated object category, and essential destination.\n"
        "Omit incidental actions such as grasping, holding, repositioning, opening, or closing "
        "unless one of them is the episode's main task.\n"
        "Treat repeated cycles as one repeated task instead of listing repetitions or object-by-object steps.\n"
        "Do not include color, size, material, shape, or other nonessential object attributes.\n"
        "Do not use the words transfer, transferring, or transferred. Use a concrete visible action instead.\n"
        "Do not invent intent, objects, or outcomes absent from the annotations.\n"
        "Do not mention segments, frames, annotations, video, camera, or the actor as 'you'.\n"
        "Good examples: 'Pack groceries into a bag.' 'Fold clothing on a table.' "
        "'Sort utensils into storage compartments.' 'Scoop granular food into containers.'\n"
        "Return only valid JSON in this form:\n"
        '{"instruction":"short imperative task instruction"}\n\n'
        "Ordered actions:\n"
        + json.dumps(actions, ensure_ascii=False, separators=(",", ":"))
    )


def generate_episode(args: argparse.Namespace, actions: list[dict[str, Any]]) -> dict[str, Any]:
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key environment variable: {args.api_key_env}")
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(actions)},
        ],
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
    }
    if args.disable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    errors: list[str] = []
    for attempt in range(args.api_retries + 1):
        started = time.monotonic()
        response_data: dict[str, Any] | None = None
        try:
            response = requests.post(
                f"{args.base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=args.timeout,
            )
            response.raise_for_status()
            response_data = response.json()
            result = extract_json(response_data["choices"][0]["message"]["content"])
            instruction = normalized_instruction(result.get("instruction"))
            if not instruction:
                raise ValueError("Episode response has an empty instruction")
            append_usage(
                args.annotations.parent / "api_usage.jsonl",
                usage_event(
                    response_data,
                    stage="episode_summary",
                    model=args.model,
                    latency_sec=time.monotonic() - started,
                    success=True,
                    attempt=attempt + 1,
                ),
            )
            return {
                "instruction": instruction,
            }
        except Exception as exc:
            maybe_restart_api_endpoint(exc, attempt < args.api_retries)
            append_usage(
                args.annotations.parent / "api_usage.jsonl",
                usage_event(
                    response_data,
                    stage="episode_summary",
                    model=args.model,
                    latency_sec=time.monotonic() - started,
                    success=False,
                    attempt=attempt + 1,
                    error=str(exc),
                ),
            )
            errors.append(str(exc))
    raise RuntimeError("Episode summary API failed: " + " | ".join(errors))


def with_episode(data: dict[str, Any], episode: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in data.items():
        if key not in {"segments", "episode_annotation"}:
            output[key] = value
    output["episode_annotation"] = episode
    output["segments"] = data.get("segments", [])
    return output


def main() -> int:
    args = parse_args()
    output_path = args.output or args.annotations
    data = json.loads(args.annotations.read_text(encoding="utf-8"))
    segments = data.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError(f"No segments found in {args.annotations}")

    provided = args.provided
    if not provided and args.provided_file:
        provided = load_provided_file(args.provided_file)

    if provided:
        episode = {
            "instruction": normalized_instruction(provided),
        }
    else:
        episode = generate_episode(args, compact_instructions(segments))

    output_path.write_text(json.dumps(with_episode(data, episode), indent=2, ensure_ascii=False) + "\n")
    summarize_usage(args.annotations.parent / "api_usage.jsonl", args.annotations.parent / "token_usage.json")
    print(json.dumps(episode, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
