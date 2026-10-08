#!/bin/bash
# Pre-compute semantic audio tokens for the training manifests (one GPU is enough).
# Usage: bash kimi/scripts/extract_codes.sh <manifest_dir>
#   reads  <manifest_dir>/{sft,eh,ep}/train_valid_merged_conversations.jsonl
#   writes <manifest_dir>/{sft,eh,ep}/processed/train_valid_merged_conversations.jsonl
set -e
source "$(dirname "$0")/env.sh"
MANIFESTS=$(realpath "${1:?manifest dir}")
for variant in sft eh ep; do
    python "$KIMI_DIR/extract_semantic_codes.py" \
        --input_file "$MANIFESTS/$variant/train_valid_merged_conversations.jsonl" \
        --output_file "$MANIFESTS/$variant/processed/train_valid_merged_conversations.jsonl"
done
