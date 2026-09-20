#!/usr/bin/env bash
set -Eeuo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ROOT="$REPO/models/hand_reconstruction"
CHECKPOINT="$MODEL_ROOT/hawor/checkpoints/hawor.ckpt"
MODEL_CONFIG="$MODEL_ROOT/hawor/model_config.yaml"
DETECTOR="$MODEL_ROOT/hawor/external/detector.pt"
MANO="$MODEL_ROOT/mano/MANO_RIGHT.pkl"

mkdir -p "$(dirname "$CHECKPOINT")" "$(dirname "$DETECTOR")" "$MODEL_ROOT/mano"

if [[ ! -d "$REPO/vendor/hand_reconstruction/.git" && ! -f "$REPO/vendor/hand_reconstruction/.git" ]]; then
  echo "VITRA submodule is missing. Run: git submodule update --init --recursive" >&2
  exit 1
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  wget -c \
    https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/checkpoints/hawor.ckpt \
    -O "$CHECKPOINT"
fi

if [[ ! -f "$MODEL_CONFIG" ]]; then
  wget -c \
    https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/model_config.yaml \
    -O "$MODEL_CONFIG"
fi

if [[ ! -f "$DETECTOR" ]]; then
  wget -c \
    https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt \
    -O "$DETECTOR"
fi

if [[ ! -f "$MANO" ]]; then
  cat >&2 <<EOF
Missing MANO_RIGHT.pkl.
Download MANO from https://mano.is.tue.mpg.de/ and place the right-hand model at:
$MANO
EOF
  exit 1
fi

echo "Hand-reconstruction models are ready under $MODEL_ROOT"
