#!/usr/bin/env python3
"""Thread-safe API usage logging and aggregation."""

from __future__ import annotations

import json
import re
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


_WRITE_LOCK = threading.Lock()


def usage_event(
    response_data: dict[str, Any] | None,
    *,
    stage: str,
    model: str,
    latency_sec: float,
    success: bool,
    attempt: int,
    error: str | None = None,
) -> dict[str, Any]:
    usage = response_data.get("usage", {}) if isinstance(response_data, dict) else {}
    prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
    completion_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
    total_tokens = usage.get("total_tokens")
    if total_tokens is None and isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
        total_tokens = prompt_tokens + completion_tokens
    return {
        "timestamp": time.time(),
        "stage": stage,
        "model": model,
        "success": success,
        "attempt": attempt,
        "latency_sec": round(latency_sec, 3),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "usage_reported": bool(usage),
        "error": error,
    }


def append_usage(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, ensure_ascii=False) + "\n"
    with _WRITE_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)


def summarize_usage(events_path: Path, output_path: Path) -> dict[str, Any]:
    events = []
    if events_path.exists():
        for line in events_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(json.loads(line))

    stages: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "usage_reported_requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "latency_sec": 0.0,
            "error_counts": {},
        }
    )
    for event in events:
        bucket = stages[str(event.get("stage", "unknown"))]
        bucket["requests"] += 1
        bucket["successful_requests" if event.get("success") else "failed_requests"] += 1
        bucket["usage_reported_requests"] += int(bool(event.get("usage_reported")))
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = event.get(key)
            if isinstance(value, int):
                bucket[key] += value
        bucket["latency_sec"] += float(event.get("latency_sec", 0.0) or 0.0)
        error = str(event.get("error") or "")
        if error:
            status_match = re.search(r"\b(4\d\d|5\d\d)\b", error)
            if status_match:
                error_key = f"http_{status_match.group(1)}"
            elif "timeout" in error.lower():
                error_key = "timeout"
            elif "capacity" in error.lower():
                error_key = "capacity"
            elif "reset by peer" in error.lower():
                error_key = "connection_reset"
            elif "too many open files" in error.lower() or "resource busy" in error.lower():
                error_key = "resource_exhausted"
            else:
                error_key = "other"
            bucket["error_counts"][error_key] = bucket["error_counts"].get(error_key, 0) + 1

    for bucket in stages.values():
        requests = bucket["requests"]
        bucket["latency_sec"] = round(bucket["latency_sec"], 3)
        bucket["mean_latency_sec"] = round(bucket["latency_sec"] / requests, 3) if requests else 0.0

    error_counts: dict[str, int] = {}
    for bucket in stages.values():
        for key, value in bucket["error_counts"].items():
            error_counts[key] = error_counts.get(key, 0) + value

    summary = {
        "requests": sum(bucket["requests"] for bucket in stages.values()),
        "successful_requests": sum(bucket["successful_requests"] for bucket in stages.values()),
        "failed_requests": sum(bucket["failed_requests"] for bucket in stages.values()),
        "usage_reported_requests": sum(bucket["usage_reported_requests"] for bucket in stages.values()),
        "prompt_tokens": sum(bucket["prompt_tokens"] for bucket in stages.values()),
        "completion_tokens": sum(bucket["completion_tokens"] for bucket in stages.values()),
        "total_tokens": sum(bucket["total_tokens"] for bucket in stages.values()),
        "error_counts": error_counts,
        "stages": dict(sorted(stages.items())),
    }
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
