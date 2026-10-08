#!/bin/bash
# LoRA fine-tuning of Kimi-Audio on SAP (paper settings).
# Usage: bash kimi/scripts/train.sh <sft|eh|ep|ec> <manifest_dir> <base_model_dir> <output_dir>
#   base_model_dir: output of init_model.sh
#   EC reads the EP manifest and adds the encoder classification head.
# Env: NUM_GPUS (default: all visible), GLOBAL_BATCH (default 64), MASTER_PORT (default 6001)
set -e
source "$(dirname "$0")/env.sh"
VARIANT=${1:?sft|eh|ep|ec}
MANIFESTS=$(realpath "${2:?manifest dir}")
BASE_MODEL=$(realpath "${3:?base model dir}")
OUTPUT=$(realpath -m "${4:?output dir}")

NUM_GPUS=${NUM_GPUS:-$(python -c 'import torch; print(torch.cuda.device_count())')}
GLOBAL_BATCH=${GLOBAL_BATCH:-64}
GRAD_ACC=$((GLOBAL_BATCH / NUM_GPUS))   # micro batch size is 1

EXTRA=""
DATA_VARIANT=$VARIANT
if [ "$VARIANT" = "ec" ]; then
    DATA_VARIANT=ep
    EXTRA="--etiology_classification True --cls_loss_weight 1.0"
fi
DATA="$MANIFESTS/$DATA_VARIANT/processed/train_valid_merged_conversations.jsonl"

cd "$KIMI_AUDIO_DIR"   # DeepSpeed config and relative resources live here
export CUDA_DEVICE_MAX_CONNECTIONS=1
torchrun --nproc_per_node "$NUM_GPUS" --nnodes 1 --master_port "${MASTER_PORT:-6001}" "$KIMI_DIR/finetune.py" \
    --model_name_or_path moonshotai/Kimi-Audio-7B-Instruct \
    --model_path "$BASE_MODEL" \
    --data_path "$DATA" \
    --eval_ratio 0.05 \
    --output_dir "$OUTPUT" \
    --bf16 True \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps "$GRAD_ACC" \
    --learning_rate 5e-5 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine \
    --save_strategy epoch \
    --eval_strategy epoch \
    --save_total_limit 3 \
    --logging_steps 1 \
    --report_to none \
    --model_max_length 512 \
    --gradient_checkpointing False \
    --lazy_preprocess False \
    --use_lora True --lora_r 16 --lora_alpha 32 --lora_dropout 0.1 \
    --deepspeed finetune_codes/ds_config_zero2.json \
    $EXTRA
