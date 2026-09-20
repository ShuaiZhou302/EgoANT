#!/usr/bin/env python3
"""Sample vLLM running and waiting request gauges during a benchmark."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request
from pathlib import Path


GAUGE = re.compile(r'^vllm:num_requests_(running|waiting)\{[^}]*\}\s+([0-9.]+)$', re.MULTILINE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor vLLM request concurrency.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pid", type=int, default=None)
    parser.add_argument("--interval", type=float, default=2.0)
    return parser.parse_args()


def process_exists(pid: int | None) -> bool:
    if pid is None:
        return True
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    samples = []
    while process_exists(args.pid):
        try:
            with urllib.request.urlopen(f"{args.base_url.rstrip('/')}/metrics", timeout=10) as response:
                text = response.read().decode("utf-8")
            values = {name: int(float(value)) for name, value in GAUGE.findall(text)}
            sample = {
                "timestamp": time.time(),
                "running": values.get("running", 0),
                "waiting": values.get("waiting", 0),
            }
            samples.append(sample)
            with args.output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(sample) + "\n")
        except Exception as exc:
            with args.output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"timestamp": time.time(), "error": str(exc)}) + "\n")
        time.sleep(args.interval)

    valid = [sample for sample in samples if "running" in sample]
    summary = {
        "samples": len(valid),
        "peak_running": max((sample["running"] for sample in valid), default=0),
        "peak_waiting": max((sample["waiting"] for sample in valid), default=0),
    }
    args.output.with_name("concurrency_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
