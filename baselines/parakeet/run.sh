#!/bin/bash
# Full fine-tuning of nvidia/parakeet-tdt-0.6b-v2 on SAP with NeMo, then inference on SAP and TORGO.
# Run from the repository root.
#   NEMO_DIR: a NeMo checkout at commit 29fc2ec3a (v2.5.0rc0), e.g.
#       git clone https://github.com/NVIDIA/NeMo && git -C NeMo checkout 29fc2ec3a
#   Manifests: python data/make_manifests.py --format nemo ... (see README)
set -e
NEMO_DIR=${NEMO_DIR:?set NEMO_DIR to a NeMo checkout}
MANIFESTS=${MANIFESTS:-data/sap/nemo}
CKPT=${CKPT:-checkpoints/parakeet-tdt-0.6b-v2}
TORGO_DIR=${TORGO_DIR:-data/torgo}
OUT=${OUT:-results/predictions}
export LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 PYTHONIOENCODING=utf-8
mkdir -p $OUT/sap $OUT/torgo

# 1. Fine-tuning (all GPUs, 50 epochs; the run of the paper used 4x A100)
python $NEMO_DIR/examples/asr/speech_to_text_finetune.py \
    --config-path=conf/asr_finetune --config-name=speech_to_text_finetune.yaml \
    model.train_ds.manifest_filepath=$MANIFESTS/train_manifest.json \
    model.validation_ds.manifest_filepath=$MANIFESTS/val_manifest.json \
    model.train_ds.max_duration=45 \
    model.train_ds.batch_size=8 \
    model.validation_ds.batch_size=6 \
    +model.train_ds.channel_selector=average \
    +model.validation_ds.channel_selector=average \
    model.tokenizer.update_tokenizer=False \
    trainer.devices=-1 trainer.accelerator=gpu trainer.max_epochs=50 \
    exp_manager.exp_dir=$CKPT \
    +model.joint.fused_batch_size=1 \
    +init_from_pretrained_model=nvidia/parakeet-tdt-0.6b-v2
NEMO_MODEL=$(ls -t $CKPT/Speech_To_Text_Finetuning/*/checkpoints/Speech_To_Text_Finetuning.nemo | head -1)

# 2. Inference (zero-shot and fine-tuned)
transcribe() {  # $1 model argument, $2 manifest, $3 output
    python $NEMO_DIR/examples/asr/speech_to_text_eval.py $1 \
        dataset_manifest=$2 output_filename=$3 \
        batch_size=64 use_cer=false scores_per_sample=true amp=true channel_selector=average
}
transcribe pretrained_name=nvidia/parakeet-tdt-0.6b-v2 $MANIFESTS/test_manifest.json $OUT/sap/zs-parakeet-tdt-0.6b-v2.jsonl
transcribe model_path=$NEMO_MODEL $MANIFESTS/test_manifest.json $OUT/sap/ft-parakeet-tdt-0.6b-v2.jsonl
python baselines/parakeet/torgo_manifest.py --torgo_tsv $TORGO_DIR/torgo.tsv --output $TORGO_DIR/torgo_nemo_manifest.json
transcribe pretrained_name=nvidia/parakeet-tdt-0.6b-v2 $TORGO_DIR/torgo_nemo_manifest.json $OUT/torgo/zs-parakeet-tdt-0.6b-v2.jsonl
transcribe model_path=$NEMO_MODEL $TORGO_DIR/torgo_nemo_manifest.json $OUT/torgo/ft-parakeet-tdt-0.6b-v2.jsonl
