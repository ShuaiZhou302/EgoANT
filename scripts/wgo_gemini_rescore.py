#!/usr/bin/env python3
"""Re-score WGO semantic metrics with a Gemini-compatible LLM judge.

This script does not generate new labels. It only re-judges existing
gold/pred label pairs using the WGO-Bench label judge rubric from Macrodata.
It supports:
  1. E2E re-score from an existing WGO score JSON with per_episode[].pairs.
  2. Fixed-boundary label accuracy from a JSONL file of judge pairs.

Environment variables:
  GEMINI_API_KEY       required unless --dry-run
  GEMINI_BASE_URL      OpenAI-compatible base URL for the Gemini judge
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


JUDGE_RUBRIC = """You are judging whether a predicted subtask label matches a gold subtask label.

Gold label:
{gt_label}

Predicted label:
{pred_label}

Episode instruction:
{instruction}

Accept if:
- It describes the same manipulation event or world-state change.
- The main action is correct.
- The main manipulated object is correct.
- Source, destination, direction, or spatial relation is correct when central to the event.
- Wording can differ; synonyms are fine.
- It may be slightly less detailed than the gold label if it is still useful.
Reject if:
- The action is wrong.
- The main object is wrong.
- Source, destination, or direction is flipped or wrong.
- It describes a different event.
- It is too vague to identify the subtask.
- It hallucinates an important object or action.

Ignore:
- Grammar.
- Minor wording differences.
- Timing; timing is evaluated separately.

Return only JSON:
{{"match": true}}"""


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def pair_prompt(pair: dict[str, Any], instruction: str) -> str:
    return JUDGE_RUBRIC.format(
        gt_label=str(pair.get("gold_label", "")).strip(),
        pred_label=str(pair.get("pred_label", "")).strip(),
        instruction=str(instruction or "").strip(),
    )


def cache_key(model: str, prompt: str) -> str:
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    h.update(b"\0")
    h.update(prompt.encode("utf-8"))
    return h.hexdigest()


def load_cache(path: Path) -> dict[str, Any]:
    if path.exists():
        return read_json(path)
    return {}


def save_cache(path: Path, cache: dict[str, Any]) -> None:
    write_json(path, cache)


def extract_match(text: str) -> bool:
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and isinstance(obj.get("match"), bool):
            return bool(obj["match"])
    except json.JSONDecodeError:
        pass
    m = re.search(r'"match"\s*:\s*(true|false)', text, re.I)
    if m:
        return m.group(1).lower() == "true"
    lowered = text.lower()
    if "true" in lowered and "false" not in lowered:
        return True
    if "false" in lowered and "true" not in lowered:
        return False
    raise ValueError(f"Cannot parse judge response as match JSON: {text[:300]}")


def post_chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    timeout: float,
    max_retries: int,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 512,
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_error: str | None = None
    for attempt in range(max_retries + 1):
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            message = data["choices"][0]["message"]
            content = message.get("content") or message.get("reasoning_content") or ""
            return {
                "match": extract_match(content),
                "raw": content,
                "usage": data.get("usage", {}),
                "response_id": data.get("id"),
            }
        except urllib.error.HTTPError as exc:
            try:
                error_body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                error_body = ""
            last_error = f"HTTP {exc.code} {exc.reason}: {error_body[:1000]}"
            if attempt >= max_retries:
                break
            if exc.code in {401, 402}:
                break
            if exc.code in {403, 408, 409, 425, 429, 500, 502, 503, 504}:
                time.sleep(min(10.0 * (attempt + 1), 60.0))
            else:
                time.sleep(min(2.0 * (attempt + 1), 10.0))
        except (
            urllib.error.URLError,
            TimeoutError,
            socket.timeout,
            OSError,
            KeyError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            last_error = repr(exc)
            if attempt >= max_retries:
                break
            time.sleep(min(2.0 * (attempt + 1), 10.0))
    raise RuntimeError(last_error or "unknown chat completion error")


def estimate_tokens_from_text(text: str) -> int:
    # Cheap conservative estimate for English judge prompts.
    return max(1, int(len(text) / 4) + 1)


def judge_pair(
    *,
    prompt: str,
    model: str,
    base_url: str,
    api_key: str,
    cache: dict[str, Any],
    dry_run: bool,
    timeout: float,
    max_retries: int,
) -> dict[str, Any]:
    key = cache_key(model, prompt)
    if key in cache:
        result = dict(cache[key])
        result["cached"] = True
        return result
    if dry_run:
        result = {
            "match": None,
            "raw": None,
            "usage": {
                "prompt_tokens_est": estimate_tokens_from_text(prompt),
                "completion_tokens_est": 16,
            },
            "cached": False,
            "dry_run": True,
        }
        cache[key] = result
        return result
    result = post_chat_completion(
        base_url=base_url,
        api_key=api_key,
        model=model,
        prompt=prompt,
        timeout=timeout,
        max_retries=max_retries,
    )
    result["cached"] = False
    cache[key] = result
    return result


def load_episode_instructions(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    out: dict[str, str] = {}
    for item in sorted(path.glob("*.json")):
        try:
            data = read_json(item)
        except Exception:
            continue
        episode_id = str(data.get("id") or item.stem)
        episode_annotation = data.get("episode_annotation") or {}
        instruction = str(
            data.get("instruction")
            or episode_annotation.get("instruction")
            or episode_annotation.get("activity_name")
            or ""
        ).strip()
        if instruction:
            out[episode_id] = instruction
    return out


def get_episode_instruction(ep: dict[str, Any], instruction_map: dict[str, str]) -> str:
    if ep.get("instruction"):
        return str(ep["instruction"])
    episode_id = str(ep.get("id") or "")
    if episode_id in instruction_map:
        return instruction_map[episode_id]
    pairs = ep.get("pairs", [])
    for p in pairs:
        if p.get("instruction"):
            return str(p["instruction"])
    return ""


def load_candidate_episodes(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for item in sorted(path.glob("*.json")):
        try:
            data = read_json(item)
        except Exception:
            continue
        episode_id = str(data.get("id") or item.stem)
        out[episode_id] = data
    return out


def get_variant_pred_label(
    pair: dict[str, Any],
    episode_id: str,
    candidate_episodes: dict[str, dict[str, Any]],
    candidate_source: str,
) -> str:
    if candidate_source in ("score", "", None):
        return str(pair.get("pred_label", ""))
    episode = candidate_episodes.get(episode_id)
    if not episode:
        raise KeyError(f"Missing candidate episode JSON for {episode_id}")
    pred_i = int(pair["pred_i"])
    segment = episode.get("segments", [])[pred_i]
    if candidate_source == "selected":
        return str(segment.get("subtask", ""))
    labels = segment.get("candidate_labels") or {}
    if candidate_source not in labels:
        raise KeyError(f"Missing candidate source {candidate_source!r} for {episode_id} pred_i={pred_i}")
    return str(labels[candidate_source])


def collect_e2e_jobs(
    score: dict[str, Any],
    instruction_map: dict[str, str],
    candidate_episodes: dict[str, dict[str, Any]],
    candidate_source: str,
) -> list[dict[str, Any]]:
    jobs = []
    for ep in score.get("per_episode", []):
        episode_id = str(ep.get("id") or "")
        instruction = get_episode_instruction(ep, instruction_map)
        for pair in ep.get("pairs", []):
            job_pair = dict(pair)
            if candidate_source not in ("score", "", None):
                job_pair["pred_label"] = get_variant_pred_label(
                    pair=pair,
                    episode_id=episode_id,
                    candidate_episodes=candidate_episodes,
                    candidate_source=candidate_source,
                )
                job_pair["candidate_source"] = candidate_source
            jobs.append({"episode_id": ep.get("id"), "instruction": instruction, "pair": job_pair})
    return jobs


def recompute_e2e(score: dict[str, Any], job_results: list[dict[str, Any]]) -> dict[str, Any]:
    result_by_key = {}
    for row in job_results:
        result_by_key[(row["episode_id"], row["pred_i"], row["gold_j"])] = row

    total_pred = 0
    total_gold = 0
    total_iou_match = 0
    total_sem = 0
    per_episode = []
    for ep in score.get("per_episode", []):
        n_pred = int(ep.get("n_pred", 0))
        n_gold = int(ep.get("n_gold", 0))
        total_pred += n_pred
        total_gold += n_gold
        ep_sem = 0
        new_pairs = []
        for pair in ep.get("pairs", []):
            key = (ep.get("id"), pair.get("pred_i"), pair.get("gold_j"))
            judged = result_by_key[key]
            match = bool(judged["match"])
            ep_sem += int(match)
            new_pair = dict(pair)
            new_pair["semantic_match"] = match
            new_pair["judge_model"] = judged["model"]
            new_pair["judge_raw"] = judged.get("raw")
            new_pair["judge_cached"] = judged.get("cached", False)
            new_pairs.append(new_pair)
        n_iou = len(new_pairs)
        total_iou_match += n_iou
        total_sem += ep_sem
        e2e_p = ep_sem / n_pred if n_pred else 0.0
        e2e_r = ep_sem / n_gold if n_gold else 0.0
        e2e_f1 = 2 * e2e_p * e2e_r / (e2e_p + e2e_r) if e2e_p + e2e_r else 0.0
        per_episode.append({
            **{k: v for k, v in ep.items() if k != "pairs"},
            "pairs": new_pairs,
            "n_semantic_match": ep_sem,
            "e2e_precision": e2e_p,
            "e2e_recall": e2e_r,
            "e2e_f1": e2e_f1,
        })

    seg_p = total_iou_match / total_pred if total_pred else 0.0
    seg_r = total_iou_match / total_gold if total_gold else 0.0
    seg_f1 = 2 * seg_p * seg_r / (seg_p + seg_r) if seg_p + seg_r else 0.0
    e2e_p = total_sem / total_pred if total_pred else 0.0
    e2e_r = total_sem / total_gold if total_gold else 0.0
    e2e_f1 = 2 * e2e_p * e2e_r / (e2e_p + e2e_r) if e2e_p + e2e_r else 0.0

    return {
        "metric": "WGO-style end_to_end_semantic_segment_f1: greedy IoU>=0.75 with outer snap, then Gemini semantic label match on matched pairs",
        "judge_model": job_results[0]["model"] if job_results else None,
        "source_score_json": score.get("source_score_json"),
        "pred_dir": score.get("pred_dir"),
        "n_episodes": score.get("n_episodes"),
        "micro": {
            "n_pred": total_pred,
            "n_gold": total_gold,
            "n_iou_match": total_iou_match,
            "n_semantic_match": total_sem,
            "seg_precision": seg_p,
            "seg_recall": seg_r,
            "seg_f1": seg_f1,
            "e2e_precision": e2e_p,
            "e2e_recall": e2e_r,
            "e2e_f1": e2e_f1,
        },
        "per_episode": per_episode,
    }


def read_label_pairs_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        obj = json.loads(line)
        obj.setdefault("episode_id", obj.get("id", f"row_{line_no}"))
        pair = {
            "gold_label": obj["gold_label"],
            "pred_label": obj["pred_label"],
            "gold_j": obj.get("gold_j", line_no - 1),
            "pred_i": obj.get("pred_i", line_no - 1),
        }
        rows.append({"episode_id": obj["episode_id"], "instruction": obj.get("instruction", ""), "pair": pair})
    return rows


def summarize_label_acc(job_results: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(job_results)
    m = sum(1 for r in job_results if r.get("match") is True)
    return {
        "metric": "WGO-style fixed-boundary label accuracy: Gemini semantic label match on gold boundaries",
        "judge_model": job_results[0]["model"] if job_results else None,
        "n": n,
        "n_match": m,
        "accuracy": m / n if n else 0.0,
        "pairs": job_results,
    }


def usage_summary(results: list[dict[str, Any]], prompts: list[str]) -> dict[str, Any]:
    prompt_tokens = 0
    completion_tokens = 0
    for res, prompt in zip(results, prompts):
        usage = res.get("usage") or {}
        prompt_tokens += int(usage.get("prompt_tokens") or usage.get("prompt_tokens_est") or estimate_tokens_from_text(prompt))
        completion_tokens += int(usage.get("completion_tokens") or usage.get("completion_tokens_est") or 16)
    # User-provided Gemini 3.5 Flash screenshot prices.
    cost_usd = prompt_tokens / 1_000_000 * 1.5 + completion_tokens / 1_000_000 * 9.0
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "estimated_cost_usd": cost_usd,
        "price_note": "Gemini 3.5 Flash estimate: input $1.5/M tokens, output $9/M tokens.",
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--score-json", type=Path, help="Existing E2E score JSON with per_episode[].pairs.")
    ap.add_argument("--label-pairs-jsonl", type=Path, help="JSONL rows with episode_id, instruction, gold_label, pred_label.")
    ap.add_argument("--episode-json-dir", type=Path, help="Directory of episode JSONs used only to recover episode instructions.")
    ap.add_argument("--candidate-json-dir", type=Path, help="Directory of candidate episode JSONs with segments[].candidate_labels.")
    ap.add_argument("--candidate-source", default="score", help="E2E pred label source: score, selected, raw, rawprior, seed, ffmpeg.")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path("outputs/wgo_gemini_judge_cache.json"))
    ap.add_argument("--model", default=os.environ.get("GEMINI_MODEL", "gemini-3.5-flash"))
    ap.add_argument("--base-url", default=os.environ.get("GEMINI_BASE_URL", "https://api.example.com/v1"))
    ap.add_argument("--api-key-env", default="GEMINI_API_KEY")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="Only judge first N pairs, for smoke tests.")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--max-retries", type=int, default=3)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if bool(args.score_json) == bool(args.label_pairs_jsonl):
        print("Provide exactly one of --score-json or --label-pairs-jsonl.", file=sys.stderr)
        return 2

    api_key = os.environ.get(args.api_key_env, "")
    if not args.dry_run and not api_key:
        print(f"Missing API key env: {args.api_key_env}", file=sys.stderr)
        return 2

    source_score = None
    if args.score_json:
        source_score = read_json(args.score_json)
        source_score["source_score_json"] = str(args.score_json)
        jobs = collect_e2e_jobs(
            source_score,
            load_episode_instructions(args.episode_json_dir),
            load_candidate_episodes(args.candidate_json_dir),
            args.candidate_source,
        )
        mode = "e2e"
    else:
        jobs = read_label_pairs_jsonl(args.label_pairs_jsonl)
        mode = "label_acc"

    if args.limit:
        jobs = jobs[: args.limit]

    cache = load_cache(args.cache)
    prompts = [pair_prompt(j["pair"], j.get("instruction", "")) for j in jobs]
    job_results = []
    for i, (job, prompt) in enumerate(zip(jobs, prompts), 1):
        try:
            judged = judge_pair(
                prompt=prompt,
                model=args.model,
                base_url=args.base_url,
                api_key=api_key,
                cache=cache,
                dry_run=args.dry_run,
                timeout=args.timeout,
                max_retries=args.max_retries,
            )
        except Exception as exc:
            save_cache(args.cache, cache)
            pair = job.get("pair", {})
            print(
                "judge failed "
                f"i={i}/{len(jobs)} episode={job.get('episode_id')} "
                f"pred_i={pair.get('pred_i')} gold_j={pair.get('gold_j')} "
                f"error={exc}",
                file=sys.stderr,
            )
            raise
        pair = job["pair"]
        job_results.append({
            "episode_id": job.get("episode_id"),
            "instruction": job.get("instruction", ""),
            "pred_i": pair.get("pred_i"),
            "gold_j": pair.get("gold_j"),
            "gold_label": pair.get("gold_label"),
            "pred_label": pair.get("pred_label"),
            "match": judged.get("match"),
            "raw": judged.get("raw"),
            "usage": judged.get("usage"),
            "cached": judged.get("cached", False),
            "model": args.model,
        })
        if i % 5 == 0 or i == len(jobs):
            print(f"judged {i}/{len(jobs)}", file=sys.stderr)
            save_cache(args.cache, cache)

    save_cache(args.cache, cache)

    if args.dry_run:
        output = {
            "mode": mode,
            "dry_run": True,
            "n_jobs": len(jobs),
            "usage_estimate": usage_summary(job_results, prompts),
            "jobs_preview": job_results[:5],
        }
    elif mode == "e2e":
        output = recompute_e2e(source_score, job_results)
        output["usage_summary"] = usage_summary(job_results, prompts)
        output["judge_pairs"] = job_results
    else:
        output = summarize_label_acc(job_results)
        output["usage_summary"] = usage_summary(job_results, prompts)

    write_json(args.output, output)
    print(json.dumps(output.get("micro", output.get("usage_summary", output)), indent=2, ensure_ascii=False)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
