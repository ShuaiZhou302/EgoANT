#!/usr/bin/env python3
"""Create generation inputs containing no gold boundaries or labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def public_episode_context(annotation: dict) -> dict[str, str]:
    return {
        "id": str(annotation["id"]),
        "instruction": str(annotation.get("instruction") or ""),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.gold_dir.glob("homer_*.json"))
    if not paths:
        raise RuntimeError(f"no HomER annotations found in {args.gold_dir}")
    for path in paths:
        annotation = json.loads(path.read_text(encoding="utf-8"))
        context = public_episode_context(annotation)
        output = args.output_dir / path.name
        output.write_text(json.dumps(context, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(paths)} gold-free episode inputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
