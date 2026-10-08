#!/bin/bash
# Merge the best checkpoint (lowest validation loss) of a training run and export it for inference.
# Usage: bash kimi/scripts/export.sh <run_dir> <base_model_dir> <output_dir>
set -e
source "$(dirname "$0")/env.sh"
RUN=$(realpath "${1:?run dir}"); BASE=$(realpath "${2:?base model dir}"); OUT=$(realpath -m "${3:?output dir}")
TOKENIZER=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('moonshotai/Kimi-Audio-7B-Instruct', allow_patterns=['tokeniz*', 'tiktoken.model', 'special_tokens_map.json']))")
cd "$KIMI_AUDIO_DIR"
python "$KIMI_DIR/export.py" --checkpoint "$RUN" --base_model_path "$BASE" --output_dir "$OUT" --tokenizer_source "$TOKENIZER"
