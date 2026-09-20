#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PY="${PY:-python3}"
EXAMPLE_ROOT="${EXAMPLE_ROOT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO/outputs/v5_1_example_suite}"
MODEL_PROVIDER="${MODEL_PROVIDER:-local_vllm}"
VIDEO_WORKERS="${VIDEO_WORKERS:-10}"
CAPTION_CONCURRENCY="${CAPTION_CONCURRENCY:-5}"
JUDGE_WORKERS="${JUDGE_WORKERS:-5}"
REWRITE_WORKERS="${REWRITE_WORKERS:-5}"
DEBUG="${DEBUG:-0}"
EXPORT_SEGMENT_CLIPS="${EXPORT_SEGMENT_CLIPS:-0}"
EXAMPLE_LIMIT="${EXAMPLE_LIMIT:-10}"
CAPTION_INPUT="${CAPTION_INPUT:-raw}"
RECONSTRUCTION_ROOT="${RECONSTRUCTION_ROOT:-$REPO/vendor/hand_reconstruction}"
MODEL_ROOT="${MODEL_ROOT:-$REPO/models/hand_reconstruction}"
HAND_RECON_CACHE_ROOT="${HAND_RECON_CACHE_ROOT:-}"

if [[ -z "$EXAMPLE_ROOT" ]]; then
  echo "Set EXAMPLE_ROOT to a user-provided video/annotation example tree." >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"
SUITE_START="$(date +%s)"
mapfile -t VIDEOS < <(find "$EXAMPLE_ROOT/video" -mindepth 2 -maxdepth 2 -name '*.mp4' | sort | head -n "$EXAMPLE_LIMIT")
if [[ "${#VIDEOS[@]}" -ne "$EXAMPLE_LIMIT" ]]; then
  echo "Expected $EXAMPLE_LIMIT example videos, found ${#VIDEOS[@]}" >&2
  exit 1
fi

prepare_reconstruction() {
  local video="$1"
  local task sample out
  task="$(basename "$(dirname "$video")")"
  sample="${task}_$(basename "$video" .mp4)"
  out="$OUTPUT_ROOT/$sample"
  mkdir -p "$out"
  if [[ ! -f "$out/hand_recon.json" ]]; then
    if [[ -n "$HAND_RECON_CACHE_ROOT" && -f "$HAND_RECON_CACHE_ROOT/$sample/hand_recon.json" ]]; then
      cp "$HAND_RECON_CACHE_ROOT/$sample/hand_recon.json" "$out/hand_recon.json"
    else
      "$PY" "$REPO/scripts/export_hand_reconstruction.py" \
        --reconstruction-root "$RECONSTRUCTION_ROOT" \
        --model-root "$MODEL_ROOT" \
        --video "$video" \
        --output "$out/hand_recon.json" \
        --device cuda >"$OUTPUT_ROOT/${sample}_reconstruction.log" 2>&1
    fi
  fi
}

for video in "${VIDEOS[@]}"; do
  prepare_reconstruction "$video"
done

run_one() {
  local video="$1"
  local task sample out gold
  task="$(basename "$(dirname "$video")")"
  sample="${task}_$(basename "$video" .mp4)"
  out="$OUTPUT_ROOT/$sample"
  gold="$EXAMPLE_ROOT/annotation/$task/$(basename "$video" .mp4).json"
  VIDEO="$video" OUT="$out" GOLD="$gold" MODEL_PROVIDER="$MODEL_PROVIDER" \
    CAPTION_CONCURRENCY="$CAPTION_CONCURRENCY" JUDGE_WORKERS="$JUDGE_WORKERS" \
    REWRITE_WORKERS="$REWRITE_WORKERS" DEBUG="$DEBUG" EXPORT_SEGMENT_CLIPS="$EXPORT_SEGMENT_CLIPS" \
    CAPTION_INPUT="$CAPTION_INPUT" \
    RUN_HAND_RECON=0 \
    bash "$REPO/scripts/run_lft_pipeline.sh" >"$OUTPUT_ROOT/${sample}.log" 2>&1
}

export -f run_one
export REPO PY EXAMPLE_ROOT OUTPUT_ROOT MODEL_PROVIDER CAPTION_CONCURRENCY JUDGE_WORKERS REWRITE_WORKERS DEBUG EXPORT_SEGMENT_CLIPS CAPTION_INPUT
printf '%s\0' "${VIDEOS[@]}" | xargs -0 -n1 -P "$VIDEO_WORKERS" bash -c 'run_one "$1"' _

"$PY" "$REPO/scripts/summarize_suite.py" \
  --output-root "$OUTPUT_ROOT" \
  --example-root "$EXAMPLE_ROOT"

SUITE_END="$(date +%s)"
printf '{"wall_time_sec": %d, "video_workers": %d}\n' \
  "$((SUITE_END - SUITE_START))" "$VIDEO_WORKERS" >"$OUTPUT_ROOT/suite_timing.json"
