#!/usr/bin/env python3
"""Build an E2E score input by replacing labels while keeping IoU matches.

Use this when a relabeling experiment keeps the same predicted temporal
boundaries as an existing score JSON. The output has the WGO score structure
expected by ``scripts/wgo_gemini_rescore.py``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-score-json", type=Path, required=True)
    ap.add_argument("--relabel-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    score = read_json(args.base_score_json)
    out = dict(score)
    out["source_score_json"] = str(args.base_score_json)
    out["pred_dir"] = str(args.relabel_dir)
    out["relabel_dir"] = str(args.relabel_dir)
    out["metric"] = (
        "WGO-style E2E score input: original temporal IoU matches, "
        "pred labels replaced from relabel-dir"
    )

    new_per = []
    for ep in score.get("per_episode", []):
        episode_id = str(ep.get("id") or "")
        pred_path = args.relabel_dir / f"{episode_id}.json"
        pred = read_json(pred_path)
        pred_segments = pred.get("segments") or []
        new_pairs = []
        for pair in ep.get("pairs", []):
            pred_i = int(pair["pred_i"])
            new_pair = dict(pair)
            new_pair["pred_label_original"] = pair.get("pred_label")
            new_pair["pred_label"] = str(pred_segments[pred_i].get("subtask") or "")
            new_pair.pop("semantic_match", None)
            new_pair.pop("judge_raw", None)
            new_pairs.append(new_pair)
        new_ep = dict(ep)
        new_ep["pred_file"] = str(pred_path)
        new_ep["pairs"] = new_pairs
        new_ep.pop("n_semantic_match", None)
        new_ep.pop("e2e_precision", None)
        new_ep.pop("e2e_recall", None)
        new_ep.pop("e2e_f1", None)
        new_per.append(new_ep)
    out["per_episode"] = new_per
    out.pop("judge_model", None)
    out.pop("macro_e2e_f1", None)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(args.out),
        "n_episodes": len(new_per),
        "n_pairs": sum(len(ep.get("pairs", [])) for ep in new_per),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
