#!/bin/bash
# Fetch Kimi-Audio (pinned commit) and apply the small patch used in the paper:
#   - Whisper encoder input cast to the encoder dtype (instead of hard-coded bf16)
#   - training dataset: no per-item lru_cache (memory) and skip utterances without audio tokens
set -e
source "$(dirname "$0")/env.sh"
cd "$KIMI_DIR/../"
git submodule update --init --recursive kimi/third_party/Kimi-Audio
cd "$KIMI_AUDIO_DIR"
if git apply --check "$KIMI_DIR/patches/kimi-audio.patch" 2>/dev/null; then
    git apply "$KIMI_DIR/patches/kimi-audio.patch"
    echo "Patch applied."
else
    echo "Patch already applied (or upstream changed); skipping."
fi
