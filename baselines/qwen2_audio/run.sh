#!/bin/bash
# Fine-tune Qwen/Qwen2-Audio-7B-Instruct on SAP, then run zero-shot and fine-tuned inference on SAP and TORGO.
# Run from the repository root. Paths default to the layout described in the README.
set -e
DIR=baselines/qwen2_audio
CKPT=${CKPT:-checkpoints/qwen2-audio}
TORGO_TSV=${TORGO_TSV:-data/torgo/torgo.tsv}
OUT=${OUT:-results/predictions}
mkdir -p $OUT/sap $OUT/torgo

# 1. Fine-tuning (4 GPUs, DeepSpeed ZeRO-3)
accelerate launch --config_file baselines/accelerate_bf16_zero3.yaml $DIR/sft.py \
    --config $DIR/config.yaml --training.output_dir="$CKPT"

# 2. Inference: zero-shot and fine-tuned, SAP dev set and TORGO
for model in "Qwen/Qwen2-Audio-7B-Instruct" "$CKPT/best_model"; do
    tag=$([ "$model" = "Qwen/Qwen2-Audio-7B-Instruct" ] && echo zs || echo ft)
    python $DIR/inference.py --config $DIR/config.yaml --testing.trained_model_path="$model" \
        --testing.output_path=$OUT/sap/$tag-qwen2-audio.jsonl
    python $DIR/inference.py --config $DIR/config.yaml --testing.trained_model_path="$model" \
        --testing.output_path=$OUT/torgo/$tag-qwen2-audio.jsonl --data.test_tsv_path="$TORGO_TSV"
done
