#!/usr/bin/env bash
set -Eeuo pipefail

# EgoANT single-video pipeline.
# Configure environment variables below, then run this script from any directory.

# -----------------------------------------------------------------------------
# Required paths
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DEFAULT="$(cd "$SCRIPT_DIR/.." && pwd)"
PY="${PY:-python3}"
REPO="${REPO:-$REPO_DEFAULT}"
RECONSTRUCTION_ROOT="${RECONSTRUCTION_ROOT:-$REPO/vendor/hand_reconstruction}"
MODEL_ROOT="${MODEL_ROOT:-$REPO/models/hand_reconstruction}"
CACHE_ROOT="${CACHE_ROOT:-$REPO/.cache}"

mkdir -p \
  "$CACHE_ROOT/huggingface" \
  "$CACHE_ROOT/torch" \
  "$CACHE_ROOT/ultralytics" \
  "$CACHE_ROOT/matplotlib" \
  "$CACHE_ROOT/xdg-cache" \
  "$CACHE_ROOT/xdg-config"

export HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TORCH_HOME="${TORCH_HOME:-$CACHE_ROOT/torch}"
export ULTRALYTICS_SETTINGS_DIR="${ULTRALYTICS_SETTINGS_DIR:-$CACHE_ROOT/ultralytics}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$CACHE_ROOT/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT/xdg-cache}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-$CACHE_ROOT/xdg-config}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export HF_HUB_DISABLE_TELEMETRY=1

VIDEO="${VIDEO:-}"
OUT="${OUT:-$REPO/outputs/final_annotation_run}"

# Optional gold JSON for evaluation only. Leave empty for normal annotation.
# Gold is never sent to the model.
GOLD="${GOLD:-}"

# -----------------------------------------------------------------------------
# API and models. Providers are defined in models/providers.json.
# -----------------------------------------------------------------------------
MODEL_PROVIDER="${MODEL_PROVIDER:-local_vllm}"
MODEL_CONFIG="$REPO/models/providers.json"
mapfile -t MODEL_VALUES < <(
  "$PY" - "$MODEL_CONFIG" "$MODEL_PROVIDER" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))[sys.argv[2]]
for key in ("base_url", "caption_model", "judge_model", "rewrite_model", "api_key_env"):
    print(data[key])
print(int(data.get("disable_thinking", False)))
print(int(data.get("check_endpoint", False)))
PY
)
BASE_URL="${MODEL_VALUES[0]}"
CAPTION_MODEL="${MODEL_VALUES[1]}"
JUDGE_MODEL="${MODEL_VALUES[2]}"
REWRITE_MODEL="${MODEL_VALUES[3]}"
API_KEY_ENV="${MODEL_VALUES[4]}"
DISABLE_THINKING="${MODEL_VALUES[5]}"
CHECK_MODEL_ENDPOINT="${MODEL_VALUES[6]}"
if [[ -n "${BASE_URL_OVERRIDE:-}" ]]; then
  BASE_URL="$BASE_URL_OVERRIDE"
fi
if [[ "$MODEL_PROVIDER" == "local_vllm" ]]; then
  export LOCAL_MODEL_API_KEY="${LOCAL_MODEL_API_KEY:-local-vllm}"
fi

API_TIMEOUT=600
API_RETRIES=2
API_IMAGE_MAX_SIDE=640
API_IMAGE_JPEG_QUALITY=85

# -----------------------------------------------------------------------------
# Stage switches
# Set a stage to 0 when reusing its existing output.
# -----------------------------------------------------------------------------
RUN_HAND_RECON="${RUN_HAND_RECON:-1}"
HAND_SOURCE="${HAND_SOURCE:-recon}"  # recon | asset
HAND_ASSET="${HAND_ASSET:-}"
HAND_ASSET_CAMERA_CONVENTION="${HAND_ASSET_CAMERA_CONVENTION:-auto}"
RUN_CAPTION="${RUN_CAPTION:-1}"
RUN_MERGE="${RUN_MERGE:-1}"
EXPORT_SEGMENT_CLIPS="${EXPORT_SEGMENT_CLIPS:-1}"
DEBUG="${DEBUG:-0}"
ALLOW_API_FALLBACKS="${ALLOW_API_FALLBACKS:-0}"
EPISODE_ANNOTATION="${EPISODE_ANNOTATION:-}"
EPISODE_ANNOTATION_FILE="${EPISODE_ANNOTATION_FILE:-}"

# -----------------------------------------------------------------------------
# Segmentation and caption
# -----------------------------------------------------------------------------
DEVICE="${DEVICE:-cuda}"
FOV_X="${FOV_X:-}"
FRAMES_PER_SEGMENT=10
CAPTION_INPUT="${CAPTION_INPUT:-raw}"
MIN_SEGMENT_SEC=1.1
SHORT_SEGMENT_POLICY="merge-runs"
SMOOTH_SIGMA_FRAMES=3.0
CUT_SOURCE="union"
CAPTION_CONCURRENCY="${CAPTION_CONCURRENCY:-30}"
CAPTION_MAX_TOKENS=16384

# -----------------------------------------------------------------------------
# V3.5 merge judge and rewrite
# Set either worker value to 0 to submit every available task concurrently.
# -----------------------------------------------------------------------------
WINDOW_SIZE=8
WINDOW_OVERLAP=2
MERGE_CONFIDENCE_THRESHOLD=0.75
CV_MERGE_THRESHOLD=0
CV_FRAME_HINT=0
STATIC_MERGE=0
CV_FRAME_KIND="raw"
MAX_CANDIDATE_SEGMENTS=8
MAX_CANDIDATE_SEC=15
JUDGE_WORKERS="${JUDGE_WORKERS:-8}"
REWRITE_WORKERS="${REWRITE_WORKERS:-8}"
MERGE_MAX_TOKENS=65536
EPISODE_MAX_TOKENS=4096

# -----------------------------------------------------------------------------
# Optional final video clips
# Set EXPORT_SEGMENT_CLIPS=1 to cut the final annotations into individual MP4s.
# Change CLIP_ANNOTATIONS to "$CAPTION_JSON" to inspect pre-merge segments.
# -----------------------------------------------------------------------------
CLIP_OUTPUT_DIR="$OUT/segment_clips"

HAND_RECON="$OUT/hand_recon.json"
CAPTION_JSON="$OUT/annotations.json"
FINAL_JSON="$OUT/merge_judge_annotations.json"
QUALITY_REPORT="$OUT/annotation_quality.json"
CLIP_ANNOTATIONS="$FINAL_JSON"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

run_stage() {
  local name="$1"
  shift
  echo "[$(timestamp)] START $name"
  local start=$SECONDS
  set +e
  "$@" 2>&1 | tee "$OUT/${name}.log"
  local rc=${PIPESTATUS[0]}
  set -e
  echo "[$(timestamp)] END $name rc=$rc elapsed_sec=$((SECONDS - start))"
  return "$rc"
}

if ! command -v "$PY" >/dev/null 2>&1 && [[ ! -x "$PY" ]]; then
  echo "Python executable not found: $PY" >&2
  exit 1
fi
if [[ -z "$VIDEO" || ! -f "$VIDEO" ]]; then
  echo "Set VIDEO=/path/to/video.mp4" >&2
  exit 1
fi
if [[ "$HAND_SOURCE" == "recon" && ! -d "$RECONSTRUCTION_ROOT" ]]; then
  echo "Hand-reconstruction source root not found: $RECONSTRUCTION_ROOT" >&2
  echo "Run: git submodule update --init --recursive" >&2
  exit 1
fi
if [[ "$HAND_SOURCE" == "recon" && ! -d "$MODEL_ROOT" ]]; then
  echo "Hand-reconstruction model root not found: $MODEL_ROOT" >&2
  echo "Run: bash scripts/setup_models.sh" >&2
  exit 1
fi
if [[ -z "${!API_KEY_ENV:-}" && ( "$RUN_CAPTION" == 1 || "$RUN_MERGE" == 1 ) ]]; then
  echo "Missing API key environment variable: $API_KEY_ENV" >&2
  echo "Run: export $API_KEY_ENV=..." >&2
  exit 1
fi
if [[ "$CHECK_MODEL_ENDPOINT" == 1 ]] && ! curl --noproxy '*' -fsS --max-time 5 "$BASE_URL/models" >/dev/null; then
  echo "Model endpoint is unavailable: $BASE_URL" >&2
  echo "Start your OpenAI-compatible model server or update models/providers.json." >&2
  exit 1
fi

mkdir -p "$OUT"
cd "$REPO"

cat <<EOF
[$(timestamp)] EgoANT pipeline configuration
video=$VIDEO
output=$OUT
provider=$MODEL_PROVIDER
hand_source=$HAND_SOURCE
hand_asset=$HAND_ASSET
caption_model=$CAPTION_MODEL
judge_model=$JUDGE_MODEL
rewrite_model=$REWRITE_MODEL
caption_concurrency=$CAPTION_CONCURRENCY
judge_workers=$JUDGE_WORKERS
rewrite_workers=$REWRITE_WORKERS
debug=$DEBUG
disable_thinking=$DISABLE_THINKING
cv_merge_threshold=$CV_MERGE_THRESHOLD
static_merge=$STATIC_MERGE
EOF

case "$HAND_SOURCE" in
  recon)
    if [[ "$RUN_HAND_RECON" == 1 ]]; then
      HAND_RECON_ARGS=(
        "$PY" scripts/export_hand_reconstruction.py
        --reconstruction-root "$RECONSTRUCTION_ROOT"
        --model-root "$MODEL_ROOT"
        --video "$VIDEO"
        --output "$HAND_RECON"
        --device "$DEVICE"
      )
      if [[ -n "$FOV_X" ]]; then
        HAND_RECON_ARGS+=(--fov-x "$FOV_X")
      fi
      run_stage hand_recon \
        "${HAND_RECON_ARGS[@]}"
    elif [[ ! -f "$HAND_RECON" ]]; then
      echo "RUN_HAND_RECON=0 but output is missing: $HAND_RECON" >&2
      exit 1
    fi
    ;;
  asset)
    if [[ -z "$HAND_ASSET" || ! -f "$HAND_ASSET" ]]; then
      echo "HAND_SOURCE=asset requires HAND_ASSET=/path/to/hand_asset.npz" >&2
      exit 1
    fi
    run_stage hand_asset \
      "$PY" scripts/convert_hand_asset.py \
        --hand-asset "$HAND_ASSET" \
        --video "$VIDEO" \
        --output "$HAND_RECON" \
        --camera-convention "$HAND_ASSET_CAMERA_CONVENTION"
    ;;
  *)
    echo "Unknown HAND_SOURCE=$HAND_SOURCE; expected recon or asset" >&2
    exit 1
    ;;
esac

if [[ "$RUN_CAPTION" == 1 ]]; then
  caption_cmd=(
    "$PY" scripts/annotate_video.py
    --video "$VIDEO"
    --hand-recon "$HAND_RECON"
    --output-dir "$OUT"
    --base-url "$BASE_URL"
    --model "$CAPTION_MODEL"
    --api-key-env "$API_KEY_ENV"
    --timeout "$API_TIMEOUT"
    --api-retries "$API_RETRIES"
    --frames-per-segment "$FRAMES_PER_SEGMENT"
    --caption-input "$CAPTION_INPUT"
    --min-segment-sec "$MIN_SEGMENT_SEC"
    --short-segment-policy "$SHORT_SEGMENT_POLICY"
    --smooth-sigma "$SMOOTH_SIGMA_FRAMES"
    --cut-source "$CUT_SOURCE"
    --api-batch-size "$CAPTION_CONCURRENCY"
    --max-tokens "$CAPTION_MAX_TOKENS"
    --api-image-max-side "$API_IMAGE_MAX_SIDE"
    --api-image-jpeg-quality "$API_IMAGE_JPEG_QUALITY"
  )
  if [[ -n "$GOLD" ]]; then
    caption_cmd+=(--gold "$GOLD")
  fi
  if [[ "$DISABLE_THINKING" == 1 ]]; then
    caption_cmd+=(--disable-thinking)
  fi
  run_stage caption "${caption_cmd[@]}"
elif [[ ! -f "$CAPTION_JSON" ]]; then
  echo "RUN_CAPTION=0 but output is missing: $CAPTION_JSON" >&2
  exit 1
fi

if [[ "$RUN_MERGE" == 1 ]]; then
  merge_cmd=(
    "$PY" scripts/merge_judge_segments.py
    --annotations "$CAPTION_JSON"
    --output "$FINAL_JSON"
    --visual-root "$OUT"
    --include-middle-frame
    --frame-kind overlay
    --window-size "$WINDOW_SIZE"
    --overlap "$WINDOW_OVERLAP"
    --merge-confidence-threshold "$MERGE_CONFIDENCE_THRESHOLD"
    --cv-merge-threshold "$CV_MERGE_THRESHOLD"
    --cv-frame-kind "$CV_FRAME_KIND"
    --max-candidate-segments "$MAX_CANDIDATE_SEGMENTS"
    --max-candidate-sec "$MAX_CANDIDATE_SEC"
    --base-url "$BASE_URL"
    --model "$JUDGE_MODEL"
    --rewrite-model "$REWRITE_MODEL"
    --api-key-env "$API_KEY_ENV"
    --timeout "$API_TIMEOUT"
    --api-retries "$API_RETRIES"
    --max-tokens "$MERGE_MAX_TOKENS"
    --api-image-max-side "$API_IMAGE_MAX_SIDE"
    --api-image-jpeg-quality "$API_IMAGE_JPEG_QUALITY"
    --workers "$JUDGE_WORKERS"
    --rewrite-workers "$REWRITE_WORKERS"
  )
  [[ "$CV_FRAME_HINT" == 1 ]] && merge_cmd+=(--cv-frame-hint) || merge_cmd+=(--no-cv-frame-hint)
  [[ "$STATIC_MERGE" == 1 ]] && merge_cmd+=(--static-merge) || merge_cmd+=(--no-static-merge)
  [[ "$DISABLE_THINKING" == 1 ]] && merge_cmd+=(--disable-thinking)
  run_stage merge "${merge_cmd[@]}"
else
  FINAL_JSON="$CAPTION_JSON"
  CLIP_ANNOTATIONS="$CAPTION_JSON"
fi

quality_cmd=(
  "$PY" scripts/validate_annotations.py
  --annotations "$FINAL_JSON"
  --output "$QUALITY_REPORT"
)
[[ "$ALLOW_API_FALLBACKS" == 1 ]] && quality_cmd+=(--allow-fallbacks)
run_stage validate_annotations "${quality_cmd[@]}"

episode_cmd=(
  "$PY" scripts/add_episode_annotation.py
  --annotations "$FINAL_JSON"
  --base-url "$BASE_URL"
  --model "$REWRITE_MODEL"
  --api-key-env "$API_KEY_ENV"
  --timeout "$API_TIMEOUT"
  --max-tokens "$EPISODE_MAX_TOKENS"
  --api-retries "$API_RETRIES"
)
[[ -n "$EPISODE_ANNOTATION" ]] && episode_cmd+=(--provided "$EPISODE_ANNOTATION")
[[ -n "$EPISODE_ANNOTATION_FILE" ]] && episode_cmd+=(--provided-file "$EPISODE_ANNOTATION_FILE")
[[ "$DISABLE_THINKING" == 1 ]] && episode_cmd+=(--disable-thinking)
run_stage episode_annotation "${episode_cmd[@]}"

if [[ "$EXPORT_SEGMENT_CLIPS" == 1 ]]; then
  if [[ ! -f "$CLIP_ANNOTATIONS" ]]; then
    echo "Clip annotation file is missing: $CLIP_ANNOTATIONS" >&2
    exit 1
  fi
  run_stage export_clips \
    "$PY" scripts/export_segment_clips.py \
      --video "$VIDEO" \
      --annotations "$CLIP_ANNOTATIONS" \
      --output-dir "$CLIP_OUTPUT_DIR" \
      --overwrite
fi

if [[ "$DEBUG" == 0 ]]; then
  "$PY" scripts/cleanup_output.py --output-dir "$OUT" --final-json "$FINAL_JSON"
fi

echo "[$(timestamp)] DONE"
echo "caption: $CAPTION_JSON"
echo "final:   $FINAL_JSON"
echo "quality: $QUALITY_REPORT"
echo "debug:   ${FINAL_JSON%.json}_debug.json"
echo "mode:    $([[ "$DEBUG" == 1 ]] && echo debug || echo production)"
if [[ "$EXPORT_SEGMENT_CLIPS" == 1 ]]; then
  echo "clips:   $CLIP_OUTPUT_DIR"
fi
