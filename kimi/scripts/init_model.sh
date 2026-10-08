#!/bin/bash
# Convert moonshotai/Kimi-Audio-7B-Instruct into the training format (LM + Whisper encoder in one checkpoint).
# Usage: bash kimi/scripts/init_model.sh <output_dir>
set -e
source "$(dirname "$0")/env.sh"
OUT=${1:?output dir}
cd "$KIMI_AUDIO_DIR"
python -m finetune_codes.model --model_name moonshotai/Kimi-Audio-7B-Instruct \
    --action init_from_pretrained --output_dir "$OUT"
