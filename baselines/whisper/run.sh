#!/bin/bash
# Fine-tune openai/whisper-large-v3 on SAP, then run zero-shot and fine-tuned inference on SAP and TORGO.
# Run from the repository root. Paths default to the layout described in the README.
set -e
DIR=baselines/whisper
CKPT=${CKPT:-checkpoints/whisper-large-v3}
TORGO_TSV=${TORGO_TSV:-data/torgo/torgo.tsv}
OUT=${OUT:-results/predictions}
mkdir -p $OUT/sap $OUT/torgo

# 1. Fine-tuning (4 GPUs, DeepSpeed ZeRO-3)
accelerate launch --config_file baselines/accelerate_bf16_zero3.yaml $DIR/train.py \
    --config $DIR/config.yaml --training.output_dir="$CKPT"

# 2. Inference: zero-shot and fine-tuned, SAP dev set and TORGO
for model in "openai/whisper-large-v3" "$CKPT/best_model"; do
    tag=$([ "$model" = "openai/whisper-large-v3" ] && echo zs || echo ft)
    python $DIR/inference.py --config $DIR/config.yaml --testing.trained_model_path="$model" \
        --testing.output_path=$OUT/sap/$tag-whisper-large-v3.jsonl
    python $DIR/inference.py --config $DIR/config.yaml --testing.trained_model_path="$model" \
        --testing.output_path=$OUT/torgo/$tag-whisper-large-v3.jsonl --data.test_tsv_path="$TORGO_TSV"
done
