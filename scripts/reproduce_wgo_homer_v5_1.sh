#!/usr/bin/env bash
# Reproduce the EgoANT V5.1 HomER pipeline with OpenAI-compatible VLM endpoints.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 /path/to/WGO_Bench [/path/to/output]" >&2
  exit 2
fi

ROOT=$1
OUT=${2:-outputs/wgo_homer_v5_1}
PYTHON=${PYTHON:-python3}
SEGMENT_BASE_URL=${SEGMENT_BASE_URL:-http://127.0.0.1:8000/v1}
SEGMENT_MODEL=${SEGMENT_MODEL:-Qwen/Qwen3.6-27B}
LABEL_BASE_URL=${LABEL_BASE_URL:-http://127.0.0.1:8001/v1}
LABEL_MODEL=${LABEL_MODEL:-Qwen/Qwen3.5-397B-A17B-FP8}
SEGMENT_WORKERS=${SEGMENT_WORKERS:-2}
LABEL_WORKERS=${LABEL_WORKERS:-8}

GOLD=$ROOT/annotations/homer
VIDEO=$ROOT/videos
EPISODES=$OUT/00_episode_inputs_no_gold
COARSE=$OUT/01_coarse_gepa
S2=$OUT/02_s2_full_cover
RAW=$OUT/03_candidate_raw
FFMPEG=$OUT/03_candidate_ffmpeg
SEED=$OUT/03_candidate_seed_neighbor
RAWPRIOR=$OUT/03_candidate_rawprior_neighbor
SELECTED=$OUT/04_candidate_selector
SCORES=$OUT/scores

mkdir -p "$OUT" "$SCORES"

for directory in "$GOLD" "$VIDEO/homer"; do
  if [[ ! -d "$directory" ]]; then
    echo "missing required directory: $directory" >&2
    exit 2
  fi
done

echo "[setup] create generation inputs without gold boundaries or labels"
$PYTHON scripts/wgo_bench/make_episode_inputs.py \
  --gold-dir "$GOLD" --output-dir "$EPISODES"

echo "[1/8] whole-episode GEPA-style coarse segmentation"
$PYTHON scripts/wgo_bench/contact_sheet_segment.py \
  --episode-dir "$EPISODES" --video-root "$VIDEO" --out-dir "$COARSE" \
  --base-url "$SEGMENT_BASE_URL" --model "$SEGMENT_MODEL" \
  --sample-sec 0.5 --tiles-per-sheet 20 --tile-px 224 --cols 5 \
  --max-sheets-per-call 0 --prompt-variant gepa_duration_prior_v1 \
  --max-tokens 8192 --min-seg-sec 0.25 --no-failed-fallback \
  --timeout 900 --retries 3 --workers "$SEGMENT_WORKERS" --skip-existing

echo "[2/8] S2 pad=0 full-cover local refinement"
$PYTHON scripts/wgo_bench/seg_opt_ideas.py \
  --stage s2 --episode-dir "$EPISODES" --video-root "$VIDEO" \
  --coarse-dir "$COARSE" --out-dir "$S2" \
  --base-url "$SEGMENT_BASE_URL" --model "$SEGMENT_MODEL" \
  --sample-sec 0.5 --tiles-per-sheet 20 --tile-px 224 --cols 5 \
  --max-tokens 8192 --pad-sec 0 --min-refine-sec 3 \
  --refine-prompt refine_full_cover_v1 \
  --timeout 900 --retries 3 --workers "$SEGMENT_WORKERS" --skip-existing

echo "[3/8] candidate A: 397B raw-frame labels"
$PYTHON scripts/wgo_bench/label_pred_segments.py \
  --pred-boundary-dir "$S2" --video-root "$VIDEO" --out-dir "$RAW" \
  --caption-input raw --base-url "$LABEL_BASE_URL" --model "$LABEL_MODEL" \
  --timeout 600 --retries 3 --segment-workers "$LABEL_WORKERS" --skip-existing

echo "[4/8] candidate D: 397B ffmpeg raw-frame labels"
$PYTHON scripts/wgo_bench/label_pred_segments_raw_variants.py \
  --pred-boundary-dir "$S2" --video-root "$VIDEO" --out-dir "$FFMPEG" \
  --reader ffmpeg --frames-per-segment 5 \
  --base-url "$LABEL_BASE_URL" --model "$LABEL_MODEL" \
  --timeout 600 --retries 3 --segment-workers "$LABEL_WORKERS" --skip-existing

echo "[5/8] candidate C: segmentation-prior neighbor relabel"
$PYTHON scripts/wgo_bench/label_pred_segments_seeded_neighbor.py \
  --pred-boundary-dir "$S2" --video-root "$VIDEO" --out-dir "$SEED" \
  --base-url "$LABEL_BASE_URL" --model "$LABEL_MODEL" \
  --frames-per-sheet 5 --image-max-side 1120 \
  --timeout 600 --retries 3 --segment-workers "$LABEL_WORKERS" --skip-existing

echo "[6/8] candidate B: raw-prior neighbor relabel"
$PYTHON scripts/wgo_bench/label_pred_segments_seeded_neighbor.py \
  --pred-boundary-dir "$RAW" --video-root "$VIDEO" --out-dir "$RAWPRIOR" \
  --base-url "$LABEL_BASE_URL" --model "$LABEL_MODEL" \
  --frames-per-sheet 5 --image-max-side 1120 \
  --timeout 600 --retries 3 --segment-workers "$LABEL_WORKERS" --skip-existing

echo "[7/8] 397B four-candidate selector"
$PYTHON scripts/wgo_bench/select_pred_segment_candidates.py \
  --boundary-dir "$S2" --raw-dir "$RAW" --rawprior-dir "$RAWPRIOR" \
  --seed-dir "$SEED" --ffmpeg-dir "$FFMPEG" --video-root "$VIDEO" \
  --out-dir "$SELECTED" --base-urls "$LABEL_BASE_URL" --model "$LABEL_MODEL" \
  --workers "$LABEL_WORKERS" --timeout 240 --retries 3 \
  --frames-per-sheet 5 --image-max-side 1120

echo "[8/8] temporal score and optional Gemini semantic score"
$PYTHON scripts/wgo_bench/score_segmentation.py \
  --gold-dir "$GOLD" --pred-dir "$SELECTED" --out "$SCORES/segment_f1.json"

if [[ -n ${GEMINI_API_KEY:-} ]]; then
  GEMINI_BASE_URL=${GEMINI_BASE_URL:-https://generativelanguage.googleapis.com/v1beta/openai}
  GEMINI_MODEL=${GEMINI_MODEL:-gemini-3.5-flash}
  $PYTHON scripts/wgo_gemini_rescore.py \
    --score-json "$SCORES/segment_f1.json" \
    --episode-json-dir "$GOLD" \
    --output "$SCORES/semantic_e2e_gemini35.json" \
    --cache "$SCORES/gemini_judge_cache.json" \
    --base-url "$GEMINI_BASE_URL" --model "$GEMINI_MODEL"
else
  echo "GEMINI_API_KEY is unset; skipped semantic E2E judging."
fi

echo "outputs: $OUT"
