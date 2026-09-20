#!/usr/bin/env bash
set -Eeuo pipefail

# Dataset-level EgoANT annotation runner.
# It consumes a dataset laid out as:
#   DATASET_ROOT/video/<task>/<id>.mp4
# and writes:
#   OUTPUT_ROOT/video/<task>/<id>.mp4
#   OUTPUT_ROOT/annotation/<task>/<id>.json
#
# Heavy API stages can run with many video workers. Hand reconstruction is
# serialized by default because it uses the local GPU.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PY="${PY:-python3}"
REPO="${REPO:-$REPO_ROOT}"
DATASET_ROOT="${DATASET_ROOT:-}"
if [[ -z "$DATASET_ROOT" ]]; then
  echo "ERROR: Set DATASET_ROOT=/path/to/dataset" >&2
  exit 1
fi
DATASET_NAME="${DATASET_NAME:-$(basename "$DATASET_ROOT")}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO/outputs/datasets/${DATASET_NAME}}"
WORK_ROOT="${WORK_ROOT:-$OUTPUT_ROOT/work}"

# Provider for the per-video pipeline.
MODEL_PROVIDER="${MODEL_PROVIDER:-local_vllm}"
BASE_URL_OVERRIDE="${BASE_URL_OVERRIDE:-}"

# Optional multiple local OpenAI-compatible endpoints.
TUNNEL_PORTS="${TUNNEL_PORTS:-8000}"

# Number of videos processed concurrently.
VIDEO_WORKERS="${VIDEO_WORKERS:-64}"
CAPTION_CONCURRENCY="${CAPTION_CONCURRENCY:-0}"
JUDGE_WORKERS="${JUDGE_WORKERS:-0}"
REWRITE_WORKERS="${REWRITE_WORKERS:-0}"

# For a first smoke test, set MAX_VIDEOS=10 or similar. Empty means all videos.
MAX_VIDEOS="${MAX_VIDEOS:-}"
TASK_FILTER="${TASK_FILTER:-}"
ID_FILTER="${ID_FILTER:-}"

# Output behavior.
VIDEO_LINK_MODE="${VIDEO_LINK_MODE:-symlink}"  # symlink | copy | skip
VIDEO_PREPROCESS="${VIDEO_PREPROCESS:-none}"  # none | left_half
EXPORT_SEGMENT_CLIPS="${EXPORT_SEGMENT_CLIPS:-0}"
DEBUG="${DEBUG:-0}"
ALLOW_API_FALLBACKS="${ALLOW_API_FALLBACKS:-0}"
RESUME="${RESUME:-1}"
MANIFEST_ONLY="${MANIFEST_ONLY:-0}"
REFRESH_MANIFEST="${REFRESH_MANIFEST:-0}"

# GPU hand reconstruction.
RUN_HAND_RECON="${RUN_HAND_RECON:-1}"
SERIALIZE_HAND_RECON="${SERIALIZE_HAND_RECON:-1}"
HAND_RECON_LOCK="${HAND_RECON_LOCK:-/tmp/egoant_hand_recon.lock}"
HAND_RECON_SLOTS="${HAND_RECON_SLOTS:-0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4 5 5 5 5 6 6 6 6 7 7 7 7}"

MANIFEST="$OUTPUT_ROOT/manifest.tsv"
STATUS_DIR="$OUTPUT_ROOT/status"
LOG_DIR="$OUTPUT_ROOT/logs"
ABORT_FLAG="$OUTPUT_ROOT/.abort"
RUN_STARTED_AT=""
RUN_ENDED_AT=""

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_path() {
  [[ -e "$1" ]] || die "Missing path: $1"
}

task_from_video() {
  basename "$(dirname "$1")"
}

id_from_video() {
  basename "$1" .mp4
}

sample_name() {
  local task="$1"
  local id="$2"
  printf '%s_%s' "$task" "$id"
}

make_manifest() {
  mkdir -p "$OUTPUT_ROOT" "$STATUS_DIR" "$LOG_DIR"
  local tmp="$MANIFEST.tmp"
  printf 'task\tid\tvideo_path\tfinal_video\tfinal_annotation\twork_dir\n' >"$tmp"

  local count=0
  while IFS= read -r video; do
    local task id sample final_video final_ann work_dir
    task="$(task_from_video "$video")"
    id="$(id_from_video "$video")"
    [[ -z "$TASK_FILTER" || "$task" == "$TASK_FILTER" ]] || continue
    [[ -z "$ID_FILTER" || "$id" == "$ID_FILTER" ]] || continue
    sample="$(sample_name "$task" "$id")"
    final_video="$OUTPUT_ROOT/video/$task/$id.mp4"
    final_ann="$OUTPUT_ROOT/annotation/$task/$id.json"
    work_dir="$WORK_ROOT/$sample"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$task" "$id" "$video" "$final_video" "$final_ann" "$work_dir" >>"$tmp"
    count=$((count + 1))
    [[ -z "$MAX_VIDEOS" || "$count" -lt "$MAX_VIDEOS" ]] || break
  done < <(
    find "$DATASET_ROOT/video" \( -type f -o -type l \) -iname '*.mp4' \
      | sort -t/ -k1,1 -k2,2 -k3,3V
  )

  mv "$tmp" "$MANIFEST"
  echo "[$(timestamp)] manifest=$MANIFEST videos=$count"
}

ensure_video_output() {
  local src="$1"
  local dst="$2"
  mkdir -p "$(dirname "$dst")"
  case "$VIDEO_LINK_MODE" in
    symlink)
      ln -sfn "$src" "$dst"
      ;;
    copy)
      cp -f "$src" "$dst"
      ;;
    skip)
      ;;
    *)
      die "Unknown VIDEO_LINK_MODE=$VIDEO_LINK_MODE"
      ;;
  esac
}

prepare_video_input() {
  local src="$1"
  local dst="$2"
  mkdir -p "$(dirname "$dst")"
  case "$VIDEO_PREPROCESS" in
    none)
      ensure_video_output "$src" "$dst"
      ;;
    left_half)
      "$PY" - "$src" "$dst" <<'PY'
import sys
from pathlib import Path

import cv2

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
cap = cv2.VideoCapture(str(src))
if not cap.isOpened():
    raise RuntimeError(f"Failed to open video: {src}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
ok, frame = cap.read()
if not ok:
    raise RuntimeError(f"Failed to read first frame: {src}")

height, width = frame.shape[:2]
crop_width = max(2, (width // 2) // 2 * 2)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
tmp = dst.with_suffix(dst.suffix + ".tmp.mp4")
writer = cv2.VideoWriter(str(tmp), fourcc, fps, (crop_width, height))
if not writer.isOpened():
    raise RuntimeError(f"Failed to open video writer: {tmp}")

while ok:
    writer.write(frame[:, :crop_width])
    ok, frame = cap.read()

writer.release()
cap.release()
tmp.replace(dst)
PY
      ;;
    *)
      die "Unknown VIDEO_PREPROCESS=$VIDEO_PREPROCESS"
      ;;
  esac
}

run_single_video() {
  local line_no="$1"
  local task="$2"
  local id="$3"
  local video="$4"
  local final_video="$5"
  local final_ann="$6"
  local work_dir="$7"
  local sample port port_array_len pipeline_log status_path
  local recon_slot_count recon_slot_id recon_gpu_id recon_slot_index recon_lock
  local source_hand_asset source_raw_annotation

  sample="$(sample_name "$task" "$id")"
  status_path="$STATUS_DIR/$sample.status"
  pipeline_log="$LOG_DIR/$sample.log"
  source_hand_asset="$DATASET_ROOT/hand_asset/$task/$id.npz"
  source_raw_annotation="$DATASET_ROOT/raw_annotation/$task/$id.json"
  mkdir -p "$work_dir" "$(dirname "$final_ann")"

  if [[ "$RESUME" == 1 && -s "$final_ann" ]]; then
    echo "done_existing" >"$status_path"
    echo "[$(timestamp)] SKIP existing $sample"
    return 0
  fi

  if [[ "$MODEL_PROVIDER" == "collaborator_vllm" ]]; then
    read -r -a ports <<<"$TUNNEL_PORTS"
    port_array_len="${#ports[@]}"
    [[ "$port_array_len" -gt 0 ]] || die "TUNNEL_PORTS is empty"
    port="${ports[$((line_no % port_array_len))]}"
  else
    port="provider"
  fi

  read -r -a recon_slots <<<"$HAND_RECON_SLOTS"
  recon_slot_count="${#recon_slots[@]}"
  if [[ "$recon_slot_count" -gt 0 ]]; then
    recon_slot_index="$((line_no % recon_slot_count))"
    recon_gpu_id="${recon_slots[$recon_slot_index]}"
    recon_slot_id="slot${recon_slot_index}_gpu${recon_gpu_id}"
    recon_lock="/tmp/egoant_hand_recon_${recon_slot_id}.lock"
  else
    recon_slot_index=0
    recon_gpu_id=""
    recon_slot_id="shared"
    recon_lock="$HAND_RECON_LOCK"
  fi

  echo "running port=$port recon_slot=$recon_slot_id started=$(timestamp)" >"$status_path"
  prepare_video_input "$video" "$final_video"

  local rc=0
  (
    set -Eeuo pipefail
    export REPO PY VIDEO="$final_video" OUT="$work_dir"
    export MODEL_PROVIDER
    if [[ "$MODEL_PROVIDER" == "collaborator_vllm" ]]; then
      export BASE_URL_OVERRIDE="http://127.0.0.1:${port}/v1"
      if [[ -z "${API_RESTART_COMMAND:-}" && -n "${COLLAB_TUNNEL_PASSWORD:-}" ]]; then
        export API_RESTART_COMMAND="$REPO/scripts/restart_collaborator_tunnel.sh"
      fi
    elif [[ -n "$BASE_URL_OVERRIDE" ]]; then
      export BASE_URL_OVERRIDE
    else
      unset BASE_URL_OVERRIDE
    fi
    export RUN_HAND_RECON RUN_CAPTION=1 RUN_MERGE=1
    export EXPORT_SEGMENT_CLIPS DEBUG ALLOW_API_FALLBACKS
    export CAPTION_CONCURRENCY JUDGE_WORKERS REWRITE_WORKERS
    if [[ -f "$source_hand_asset" ]]; then
      export HAND_SOURCE=asset
      export HAND_ASSET="$source_hand_asset"
      export RUN_HAND_RECON=0
    else
      export HAND_SOURCE=recon
      unset HAND_ASSET
    fi
    if [[ -f "$source_raw_annotation" ]]; then
      export EPISODE_ANNOTATION_FILE="$source_raw_annotation"
    else
      unset EPISODE_ANNOTATION_FILE
    fi
    if [[ -n "$recon_gpu_id" ]]; then
      export CUDA_VISIBLE_DEVICES="$recon_gpu_id"
      export DEVICE="cuda"
    fi

    if [[ "$SERIALIZE_HAND_RECON" == 1 && "$RUN_HAND_RECON" == 1 ]]; then
      flock "$recon_lock" bash scripts/run_lft_pipeline.sh
    else
      bash scripts/run_lft_pipeline.sh
    fi
  ) >"$pipeline_log" 2>&1 || rc=$?

  if [[ "$rc" -ne 0 ]]; then
    echo "failed port=$port recon_slot=$recon_slot_id rc=$rc finished=$(timestamp)" >"$status_path"
    return "$rc"
  fi

  if [[ ! -s "$work_dir/merge_judge_annotations.json" ]]; then
    echo "failed port=$port recon_slot=$recon_slot_id rc=missing_output finished=$(timestamp)" >"$status_path"
    return 1
  fi

  cp "$work_dir/merge_judge_annotations.json" "$final_ann"
  echo "done port=$port recon_slot=$recon_slot_id finished=$(timestamp)" >"$status_path"
  echo "[$(timestamp)] DONE $sample port=$port recon_slot=$recon_slot_id"
}

worker_loop() {
  local worker_id="$1"
  local line_no=0
  tail -n +2 "$MANIFEST" | while IFS=$'\t' read -r task id video final_video final_ann work_dir; do
    [[ -n "$task" ]] || continue
    [[ ! -f "$ABORT_FLAG" ]] || return 1
    if (( line_no % VIDEO_WORKERS == worker_id )); then
      if ! run_single_video "$line_no" "$task" "$id" "$video" "$final_video" "$final_ann" "$work_dir"; then
        echo "failed line=$line_no task=$task id=$id finished=$(timestamp)" >"$STATUS_DIR/$(sample_name "$task" "$id").status"
        echo "[$(timestamp)] FAIL $task/$id; aborting dataset run" >&2
        touch "$ABORT_FLAG"
        return 1
      fi
    fi
    line_no=$((line_no + 1))
  done || true
  [[ ! -f "$ABORT_FLAG" ]] || return 1
  return 0
}

summarize_run() {
  local total done failed existing duration_sec
  total=$(($(wc -l <"$MANIFEST") - 1))
  done=$( (grep -Rhs '^done ' "$STATUS_DIR" || true) | wc -l )
  existing=$( (grep -Rhs '^done_existing' "$STATUS_DIR" || true) | wc -l )
  failed=$( (grep -Rhs '^failed ' "$STATUS_DIR" || true) | wc -l )
  if [[ -n "$RUN_STARTED_AT" && -n "$RUN_ENDED_AT" ]]; then
    duration_sec=$((RUN_ENDED_AT - RUN_STARTED_AT))
  else
    duration_sec=""
  fi
  cat >"$OUTPUT_ROOT/run_summary.txt" <<EOF
dataset_root=$DATASET_ROOT
output_root=$OUTPUT_ROOT
manifest=$MANIFEST
started_at=$(date -d "@${RUN_STARTED_AT:-0}" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || true)
ended_at=$(date -d "@${RUN_ENDED_AT:-0}" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || true)
duration_sec=$duration_sec
total=$total
done=$done
done_existing=$existing
failed=$failed
video_workers=$VIDEO_WORKERS
model_provider=$MODEL_PROVIDER
tunnel_ports=$TUNNEL_PORTS
video_preprocess=$VIDEO_PREPROCESS
hand_recon_slots=$HAND_RECON_SLOTS
caption_concurrency=$CAPTION_CONCURRENCY
judge_workers=$JUDGE_WORKERS
rewrite_workers=$REWRITE_WORKERS
serialize_hand_recon=$SERIALIZE_HAND_RECON
EOF
  cat "$OUTPUT_ROOT/run_summary.txt"
}

main() {
  require_path "$PY"
  require_path "$REPO/scripts/run_lft_pipeline.sh"
  require_path "$DATASET_ROOT/video"
  mkdir -p "$OUTPUT_ROOT/video" "$OUTPUT_ROOT/annotation" "$WORK_ROOT" "$STATUS_DIR" "$LOG_DIR"

  if [[ "$REFRESH_MANIFEST" == 1 || ! -s "$MANIFEST" ]]; then
    make_manifest
  else
    echo "[$(timestamp)] reusing manifest=$MANIFEST videos=$(($(wc -l <"$MANIFEST") - 1))"
  fi

  if [[ "$MANIFEST_ONLY" == 1 ]]; then
    echo "[$(timestamp)] manifest-only mode, skipping annotation run"
    return 0
  fi

  RUN_STARTED_AT="$(date +%s)"
  echo "[$(timestamp)] START dataset annotation"
  echo "dataset=$DATASET_ROOT"
  echo "output=$OUTPUT_ROOT"
  echo "workers=$VIDEO_WORKERS ports=$TUNNEL_PORTS"
  echo "video_preprocess=$VIDEO_PREPROCESS"
  echo "hand_recon_serialized=$SERIALIZE_HAND_RECON lock=$HAND_RECON_LOCK"
  echo "hand_recon_slots=$HAND_RECON_SLOTS"
  rm -f "$ABORT_FLAG"

  local pids=()
  for worker_id in $(seq 0 $((VIDEO_WORKERS - 1))); do
    worker_loop "$worker_id" &
    pids+=("$!")
  done

  local status=0
  for pid in "${pids[@]}"; do
    wait "$pid" || status=1
  done

  RUN_ENDED_AT="$(date +%s)"
  summarize_run
  echo "[$(timestamp)] END dataset annotation status=$status"
  return "$status"
}

main "$@"
