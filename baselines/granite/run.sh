#!/bin/bash
# Fine-tune ibm-granite/granite-speech-3.3-8b on SAP, then run zero-shot and fine-tuned inference on SAP and TORGO.
# Run from the repository root. Paths default to the layout described in the README.
set -e
DIR=baselines/granite
CKPT=${CKPT:-checkpoints/granite-speech}
TORGO_TSV=${TORGO_TSV:-data/torgo/torgo.tsv}
OUT=${OUT:-results/predictions}
mkdir -p $OUT/sap $OUT/torgo

# 1. Fine-tuning (4 GPUs, DeepSpeed ZeRO-3)
accelerate launch --config_file baselines/accelerate_bf16_zero3.yaml $DIR/sft.py \
    --config $DIR/config.yaml --training.output_dir="$CKPT"

# 2. Inference: zero-shot and fine-tuned, SAP dev set and TORGO
for model in "ibm-granite/granite-speech-3.3-8b" "$CKPT/best_model"; do
    tag=$([ "$model" = "ibm-granite/granite-speech-3.3-8b" ] && echo zs || echo ft)
    python $DIR/inference.py --config $DIR/config.yaml --testing.trained_model_path="$model" \
        --testing.output_path=$OUT/sap/$tag-granite-speech.jsonl
    python $DIR/inference.py --config $DIR/config.yaml --testing.trained_model_path="$model" \
        --testing.output_path=$OUT/torgo/$tag-granite-speech.jsonl --data.test_tsv_path="$TORGO_TSV"
done
