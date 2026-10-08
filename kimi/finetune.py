"""
LoRA fine-tuning of Kimi-Audio-7B-Instruct on SAP.

The same script trains all four variants of the paper:
  * SFT / EH / EP: plain autoregressive loss; the variant is defined only by the
    manifest (prompt and target format, see data/make_manifests.py).
  * EC: add --etiology_classification True; an MLP head on the Whisper encoder
    predicts the etiology (cross-entropy added to the LM loss) while the decoder
    is trained on the plain transcript. Uses the EP manifest as input.

Adapted from Kimi-Audio's finetune.py (itself based on FastChat / Qwen).
Run through scripts/train.sh (torchrun + DeepSpeed ZeRO-2).
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import transformers
from accelerate.utils import DistributedType
from huggingface_hub import snapshot_download
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer, Trainer, TrainerCallback
from transformers.integrations import deepspeed

from finetune_codes.datasets import LazySupervisedDataset
from finetune_codes.model import KimiAudioModel

from ec_model import CLASSIFIER_FILENAME, NUM_ETIOLOGY_CLASSES, KimiAudioModelEC, LazySupervisedDatasetEC

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"]

local_rank = None


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="moonshotai/Kimi-Audio-7B-Instruct")
    model_path: str = field(default=None, metadata={"help": "Kimi-Audio checkpoint in training format (see scripts/init_model.sh)"})


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Training manifest with pre-extracted audio tokens"})
    eval_ratio: float = field(default=0.05, metadata={"help": "The first eval_ratio of the manifest is held out for validation"})
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    dataloader_pin_memory: bool = field(default=False)
    remove_unused_columns: bool = field(default=False)
    model_max_length: int = field(default=8192)
    # LoRA
    use_lora: bool = field(default=True)
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.1)
    # Etiology classification (EC)
    etiology_classification: bool = field(default=False, metadata={"help": "Train the EC variant"})
    cls_loss_weight: float = field(default=1.0)
    num_etiology_classes: int = field(default=NUM_ETIOLOGY_CLASSES)


def rank0_print(*args):
    if local_rank in (0, -1, None):
        print(*args)


def lm_loss(outputs, labels):
    """Kimi-Audio loss: masked cross-entropy on the audio and text streams."""
    audio_logits, text_logits = outputs.logits
    audio_labels, text_labels, audio_loss_mask, text_loss_mask = labels
    assert audio_labels.shape[0] == 1, "micro batch size must be 1"

    audio_loss = torch.nn.functional.cross_entropy(audio_logits.view(-1, audio_logits.shape[-1]), audio_labels.view(-1), reduction="none")
    text_loss = torch.nn.functional.cross_entropy(text_logits.view(-1, text_logits.shape[-1]), text_labels.view(-1), reduction="none")
    audio_loss = (audio_loss * audio_loss_mask.view(-1)).sum() / (audio_loss_mask.view(-1).sum() + 1e-4)
    text_loss = (text_loss * text_loss_mask.view(-1)).sum() / (text_loss_mask.view(-1).sum() + 1e-4)
    return audio_loss + text_loss


def compute_loss(outputs, labels, num_items_in_batch=None):
    return lm_loss(outputs, labels)


class ECTrainer(Trainer):
    """LM loss + cls_loss_weight * etiology cross-entropy."""

    def __init__(self, cls_loss_weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.cls_loss_weight = cls_loss_weight

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        etiology_label = inputs.pop("etiology_label")
        outputs, etiology_logits = model(**inputs)
        cls_loss = torch.nn.functional.cross_entropy(etiology_logits, etiology_label.unsqueeze(0).to(etiology_logits.device))
        loss = lm_loss(outputs, labels) + self.cls_loss_weight * cls_loss
        return (loss, outputs) if return_outputs else loss


def get_classifier(model):
    for name, module in model.named_modules():
        if name.endswith("etiology_classifier"):
            return module
    raise ValueError("etiology_classifier not found")


class SaveClassifierCallback(TrainerCallback):
    """PEFT checkpoints only contain the adapters: store the EC head next to them."""

    def on_save(self, args, state, control, model=None, **kwargs):
        if state.is_world_process_zero and model is not None:
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            cls_state = {k: v.detach().cpu().clone() for k, v in get_classifier(model).state_dict().items()}
            torch.save(cls_state, os.path.join(ckpt_dir, CLASSIFIER_FILENAME))


def setup_lora(model, training_args):
    lora_config = LoraConfig(
        r=training_args.lora_r,
        lora_alpha=training_args.lora_alpha,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=training_args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    if training_args.etiology_classification:
        # The classification head is trained fully, not through LoRA
        for name, param in model.named_parameters():
            if "etiology_classifier" in name:
                param.requires_grad = True
    model.print_trainable_parameters()
    return model


def make_supervised_data_module(dataset_cls, whisper_model, text_tokenizer, data_args, max_len, kimia_token_offset) -> Dict:
    rank0_print("Loading data...")
    with open(data_args.data_path, "r") as f:
        all_data = [json.loads(line) for line in f]

    n_eval = int(len(all_data) * data_args.eval_ratio)
    eval_data, train_data = (all_data[:n_eval], all_data[n_eval:]) if n_eval > 0 else (None, all_data)

    kwargs = dict(whisper_model=whisper_model, text_tokenizer=text_tokenizer, max_len=max_len, kimia_token_offset=kimia_token_offset)
    return dict(
        train_dataset=dataset_cls(train_data, **kwargs),
        eval_dataset=dataset_cls(eval_data, **kwargs) if eval_data else None,
    )


def train():
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # single-GPU DeepSpeed
    if getattr(training_args, "deepspeed", None) and int(os.environ.get("WORLD_SIZE", 1)) == 1:
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED
    local_rank = training_args.local_rank

    cache_path = model_args.model_name_or_path
    if not os.path.exists(cache_path):
        cache_path = snapshot_download(model_args.model_name_or_path)
    if not os.path.exists(model_args.model_path):
        raise ValueError(f"Model path {model_args.model_path} does not exist")

    load_kwargs = {"low_cpu_mem_usage": not deepspeed.is_deepspeed_zero3_enabled(), "device_map": None}
    if training_args.etiology_classification:
        model = KimiAudioModelEC.from_pretrained(model_args.model_path, num_etiology_classes=training_args.num_etiology_classes, **load_kwargs)
        dataset_cls = LazySupervisedDatasetEC
    else:
        model = KimiAudioModel.from_pretrained(model_args.model_path, **load_kwargs)
        dataset_cls = LazySupervisedDataset

    whisper_model, kimia_token_offset = model.whisper_model, model.config.kimia_token_offset
    if training_args.use_lora:
        model = setup_lora(model, training_args)

    text_tokenizer = AutoTokenizer.from_pretrained(cache_path, trust_remote_code=True)
    data_module = make_supervised_data_module(
        dataset_cls, whisper_model, text_tokenizer, data_args, training_args.model_max_length, kimia_token_offset
    )

    if training_args.etiology_classification:
        trainer = ECTrainer(
            model=model,
            args=training_args,
            cls_loss_weight=training_args.cls_loss_weight,
            data_collator=data_module["train_dataset"].collate_fn,
            callbacks=[SaveClassifierCallback()],
            **data_module,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            compute_loss_func=compute_loss,
            data_collator=data_module["train_dataset"].collate_fn,
            **data_module,
        )

    trainer.train()
    trainer.save_state()
    rank0_print(f"Training done. Checkpoints in {training_args.output_dir}; "
                "export the one with the lowest eval loss with scripts/export.sh")


if __name__ == "__main__":
    train()
