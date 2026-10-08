#!/bin/bash
# Fine-tune microsoft/Phi-4-multimodal-instruct on SAP, then run zero-shot and fine-tuned inference on SAP and TORGO.
# Run from the repository root. Paths default to the layout described in the README.
set -e
DIR=baselines/phi4
CKPT=${CKPT:-checkpoints/phi4-mm}
TORGO_TSV=${TORGO_TSV:-data/torgo/torgo.tsv}
OUT=${OUT:-results/predictions}
mkdir -p $OUT/sap $OUT/torgo

# 1. Fine-tuning (4 GPUs, DeepSpeed ZeRO-3)
accelerate launch --config_file baselines/accelerate_bf16_zero3.yaml $DIR/sft.py \
    --config $DIR/config.yaml --training.output_dir="$CKPT"

# Make the saved preprocessor config / chat template loadable by the Phi-4 processor
python $DIR/fix_checkpoint.py --ckpt_folder "$CKPT/best_model"

# 2. Inference: zero-shot and fine-tuned, SAP dev set and TORGO
for model in "microsoft/Phi-4-multimodal-instruct" "$CKPT/best_model"; do
    tag=$([ "$model" = "microsoft/Phi-4-multimodal-instruct" ] && echo zs || echo ft)
    python $DIR/inference.py --config $DIR/config.yaml --testing.trained_model_path="$model" \
        --testing.output_path=$OUT/sap/$tag-phi4-mm.jsonl
    python $DIR/inference.py --config $DIR/config.yaml --testing.trained_model_path="$model" \
        --testing.output_path=$OUT/torgo/$tag-phi4-mm.jsonl --data.test_tsv_path="$TORGO_TSV"
done
