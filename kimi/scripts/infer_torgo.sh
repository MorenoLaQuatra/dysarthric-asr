#!/bin/bash
# Zero-shot inference on TORGO (no TORGO fine-tuning).
# Usage: bash kimi/scripts/infer_torgo.sh <model_dir_or_hf_id> <sft|eh|ep|ec> <torgo_dir> <output.jsonl>
#   torgo_dir: output of data/prepare_torgo.py
set -e
source "$(dirname "$0")/env.sh"
MODEL=${1:?model}; VARIANT=${2:?variant}; TORGO=$(realpath "${3:?torgo dir}"); OUT=$(realpath -m "${4:?output}")
EXTRA=""
[ "$VARIANT" = "eh" ] && EXTRA="--torgo_eh_hints"
cd "$KIMI_AUDIO_DIR"
python "$KIMI_DIR/infer.py" --model_path "$MODEL" --manifest "$TORGO/torgo_conversations.jsonl" \
    --base_dir "$TORGO" --output "$OUT" $EXTRA
