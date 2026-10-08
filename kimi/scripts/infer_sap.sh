#!/bin/bash
# Inference on the SAP development set.
# Usage: bash kimi/scripts/infer_sap.sh <model_dir_or_hf_id> <sft|eh|ep|ec> <manifest_dir> <output.jsonl>
#   SFT/EP/EC models use the plain prompt; EH uses the oracle etiology hint from the EH manifest.
set -e
source "$(dirname "$0")/env.sh"
MODEL=${1:?model}; VARIANT=${2:?variant}; MANIFESTS=$(realpath "${3:?manifest dir}"); OUT=$(realpath -m "${4:?output}")
case $VARIANT in
    eh) MANIFEST="$MANIFESTS/eh/test_conversations.jsonl" ;;
    *)  MANIFEST="$MANIFESTS/ep/test_conversations.jsonl" ;;   # plain prompt, EP-format reference
esac
cd "$KIMI_AUDIO_DIR"
python "$KIMI_DIR/infer.py" --model_path "$MODEL" --manifest "$MANIFEST" --output "$OUT"
