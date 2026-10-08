#!/usr/bin/env python3
"""
Fine-tune Granite Speech model on dysarthric speech ASR
using SAPC dataset with comprehensive evaluation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
import os
import json
from pathlib import Path

import torch
import pandas as pd
from accelerate import Accelerator
from accelerate.utils import gather_object
from tqdm import tqdm
from transformers import (
    BatchFeature,
    Trainer,
    TrainingArguments,
)
from transformers.models.granite_speech import GraniteSpeechForConditionalGeneration, GraniteSpeechProcessor

# Import your custom modules
from yaml_config_manager import load_config
from dataset import GraniteDysarthricDataset
from evaluation.sap_evaluator import DysarthricASREvaluator

# Constants
_IGNORE_INDEX = -100

# Ensure that LossScaler is safe for serialization
import torch.serialization
try:
    from deepspeed.runtime.fp16.loss_scaler import LossScaler
    from deepspeed.runtime.zero.config import ZeroStageEnum
    torch.serialization.add_safe_globals([LossScaler, ZeroStageEnum])
except ImportError:
    print("DeepSpeed not available, skipping safe globals")


class GraniteCollator:
    """Collator for Granite Speech training and inference."""
    
    def __init__(self, processor, inference_mode=False):
        self.processor = processor
        self.inference_mode = inference_mode

    def __call__(self, batch):
        """
        Collate function for Granite Speech data batching
        """
        # Skip empty batches
        if not batch:
            return None
            
        # Filter out any empty samples
        batch = [b for b in batch if b and b['input_ids'].size(0) > 0]
        if not batch:
            return None

        prompts = [example["prompt"] for example in batch]
        audios = [example["audio"] for example in batch]
        
        # Convert audio format if needed
        if isinstance(audios[0], dict):
            audios = [audio["array"] for audio in audios]

        try:
            # Process audio and text
            processed = self.processor(prompts, audios, return_tensors="pt", padding=True, padding_side="left")
            input_ids = processed.input_ids
            attention_mask = processed.attention_mask
            labels = None
            
            # Handle training vs inference mode
            if not self.inference_mode:
                # For training, tokenize targets and concatenate with prompts
                targets = [example["text"] + self.processor.tokenizer.eos_token for example in batch]
                targets = self.processor.tokenizer(targets, return_tensors="pt", padding=True, padding_side="right")
                
                # Combine prompt+targets
                input_ids = torch.cat([input_ids, targets.input_ids], dim=1)
                attention_mask = torch.cat([attention_mask, targets.attention_mask], dim=1)
                labels = targets.input_ids.clone()
                
                # Set non-target tokens to -100 for loss calculation
                labels[~(targets.attention_mask.bool())] = _IGNORE_INDEX
                labels = torch.cat([torch.full_like(processed.input_ids, _IGNORE_INDEX), labels], dim=1)

            return BatchFeature(data={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "input_features": processed.input_features,
                "input_features_mask": processed.input_features_mask
            })
            
        except Exception as e:
            print(f"Error in collate function: {e}")
            return None


def create_model(model_name_or_path, use_flash_attention=False, unfreeze_speech_encoder=False):
    """Create and configure the Granite Speech model."""
    model = GraniteSpeechForConditionalGeneration.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16 if use_flash_attention else torch.float32,
        trust_remote_code=True,
    ).to('cuda')
    
    # Granite Speech parameter freezing strategy
    if unfreeze_speech_encoder:
        print("Unfreezing speech components for Granite...")
        model = unfreeze_granite_speech_components(model)
    else:
        # Default: only train projector and LoRA layers (as in Granite example)
        for n, p in model.named_parameters():
            p.requires_grad = "projector" in n or "lora" in n

    return model


def unfreeze_granite_speech_components(model):
    """Unfreeze speech-related components in Granite Speech model."""
    # Based on the Granite Speech example, we can target specific components
    # This is a more aggressive unfreezing compared to the default LoRA+projector approach
    
    # First, freeze everything
    for param in model.parameters():
        param.requires_grad = False
    
    # Then unfreeze speech-related components
    for name, param in model.named_parameters():
        if any(keyword in name for keyword in ["projector", "lora", "audio", "speech"]):
            param.requires_grad = True
            
    return model


def manage_additional_token_if_needed(processor, model, config):
    """
    If the model uses additional tokens as per config, add them as special tokens to the processor and update the model.
    """
    additional_tokens = []
    normalizer = config.data.normalizer

    # Mapping of actions to tags to be added
    action_tag_map = {
        "disfluency_action": [normalizer.start_disfluency_tag, normalizer.end_disfluency_tag],
        "g_action": [normalizer.start_disfluency_tag, normalizer.end_disfluency_tag],
        "w_action": [normalizer.w_tag],
        "cs_action": [normalizer.start_cs_tag, normalizer.end_cs_tag],
        "ss_action": [normalizer.start_disfluency_tag, normalizer.end_disfluency_tag],
    }

    for action, tags in action_tag_map.items():
        if getattr(normalizer, action) == "tag":
            print(f"{action.replace('_', ' ').upper()} is set to 'tag', adding corresponding tokens to processor")
            additional_tokens.extend(tags)

    # Add tokens to processor if needed
    if additional_tokens:
        original_token_count = len(processor.tokenizer)
        print(f"Before adding, processor has {original_token_count} tokens.")
        print(f"Adding additional tokens: {additional_tokens}")
        # Add unique tokens to avoid duplicates
        additional_tokens = list(set(additional_tokens))
        print(f"After deduplication, {len(additional_tokens)} unique tokens to add.")
        processor.tokenizer.add_special_tokens({'additional_special_tokens': additional_tokens})
        model.resize_token_embeddings(len(processor.tokenizer))
        print(f"Processor now has {len(processor.tokenizer)} tokens after addition.")
    else:
        print("No additional tokens to add to processor.")


@torch.no_grad()
def evaluate_asr(
    model, processor, test_dataset, dysarthric_evaluator, 
    save_path=None, disable_tqdm=False, eval_batch_size=1,
    max_new_tokens=256,
):
    """Evaluate ASR model using DysarthricASREvaluator - adapted for Granite Speech."""
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
    model.eval()
    all_predictions = []
    all_references = []

    # Create evaluation dataloader
    collator = GraniteCollator(processor, inference_mode=True)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        collate_fn=collator,
        shuffle=False,
        drop_last=False,
        num_workers=eval_batch_size,
        prefetch_factor=2,
        pin_memory=True,
    )

    for inputs in tqdm(
        test_dataloader, disable=(rank != 0) or disable_tqdm, desc='Running evaluation'
    ):
        # Skip empty batches
        if inputs is None:
            continue
            
        # Move inputs to device
        inputs = inputs.to(f'cuda:{local_rank}')
        
        # Generate transcriptions using Granite's approach
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                num_beams=4, 
                early_stopping=True
            )
        
        # Extract generated part
        input_length = inputs.input_ids.shape[1]
        outputs = outputs[:, input_length:].cpu()
        
        # Decode generated text
        generated_text = [
            processor.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        
        # Get reference texts from the dataset
        batch_references = []
        for i in range(len(generated_text)):
            # Get the original reference from the dataset
            dataset_idx = len(all_predictions) + i
            if dataset_idx < len(test_dataset):
                ref_text = test_dataset.get_original_transcript(dataset_idx)
                batch_references.append(ref_text)
        
        # Collect results
        all_predictions.extend(generated_text)
        all_references.extend(batch_references)

    # Gather results from all processes
    all_predictions = gather_object(all_predictions)
    all_references = gather_object(all_references)
    
    # Calculate WER and SemScore on main process
    if rank == 0:
        assert len(all_predictions) == len(all_references)
        
        # Use DysarthricASREvaluator for comprehensive evaluation
        results = dysarthric_evaluator.evaluate(
            hypotheses=all_predictions,
            references=all_references,
            dual_reference_mode='auto',
            process_text=True
        )
        
        wer = results['wer'] / 100.0
        semscore = results['semscore'] / 100.0
        
        print(f"Word Error Rate (WER): {wer:.4f} ({results['wer']:.2f}%)")
        print(f"Semantic Score (SemScore): {semscore:.4f} ({results['semscore']:.2f}%)")
        
        # Save evaluation results
        if save_path:
            with open(save_path, 'w') as f:
                save_dict = {
                    'predictions': all_predictions,
                    'references': all_references,
                    'wer': wer,
                    'wer_percent': results['wer'],
                    'semscore': semscore,
                    'semscore_percent': results['semscore'],
                }
                json.dump(save_dict, f, indent=2)

        return wer, semscore
    return None, None


def main():
    # Load configuration
    config = load_config()
    
    # Set environment variables
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    print("🚀 Starting dysarthric speech ASR training with Granite Speech...")
    print(f"Config loaded: {config}")
    
    # Setup accelerator for distributed training
    accelerator = Accelerator()

    # Load model and processor
    with accelerator.local_main_process_first():
        print("Loading Granite Speech processor...")
        processor = GraniteSpeechProcessor.from_pretrained(
            config.model.name_or_path,
            trust_remote_code=True
        )
        
        print("Loading Granite Speech model...")
        model = create_model(
            config.model.name_or_path,
            use_flash_attention=config.model.use_flash_attention,
            unfreeze_speech_encoder=config.model.unfreeze_speech_encoder,
        )
        
        # Manage additional tokens if needed
        manage_additional_token_if_needed(processor, model, config)
        
    # Get rank and world size for distributed training
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    # Create datasets using GraniteDysarthricDataset
    print("Creating datasets...")
    train_dataset = GraniteDysarthricDataset(
        processor=processor,
        tsv_path=config.data.train_tsv_path,
        data_root_path=config.data.data_root_path,
        split='train',
        max_size=config.data.train_size,
        instruction=config.data.instruction,
        use_etiology_in_instruction=config.data.use_etiology_in_instruction,
        etiology_token=config.data.etiology_token,
        normalize_transcripts=config.data.normalizer.normalize_transcripts,
        remove_prompts=config.data.normalizer.remove_prompts,
        disfluency_action=config.data.normalizer.disfluency_action,
        start_disfluency_tag=config.data.normalizer.start_disfluency_tag,
        end_disfluency_tag=config.data.normalizer.end_disfluency_tag,
        g_action=config.data.normalizer.g_action,
        w_action=config.data.normalizer.w_action,
        ss_action=config.data.normalizer.ss_action,
        cs_action=config.data.normalizer.cs_action,
        w_tag=config.data.normalizer.w_tag,
        start_cs_tag=config.data.normalizer.start_cs_tag,
        end_cs_tag=config.data.normalizer.end_cs_tag,
        process_spelled=config.data.normalizer.process_spelled,
        capitalization=config.data.normalizer.capitalization,
    )
    
    validation_dataset = GraniteDysarthricDataset(
        processor=processor,
        tsv_path=config.data.dev_tsv_path,
        data_root_path=config.data.data_root_path,
        split='validation',
        max_size=config.data.eval_size,
        instruction=config.data.instruction,
        use_etiology_in_instruction=config.data.use_etiology_in_instruction,
        etiology_token=config.data.etiology_token,
        normalize_transcripts=False,  # Validation should not normalize transcripts
    )
    
    test_dataset = GraniteDysarthricDataset(
        processor=processor,
        tsv_path=config.data.test_tsv_path,
        data_root_path=config.data.data_root_path,
        split='test',
        max_size=config.data.test_size,
        instruction=config.data.instruction,
        use_etiology_in_instruction=config.data.use_etiology_in_instruction,
        etiology_token=config.data.etiology_token,
        normalize_transcripts=False,  # Test should not normalize transcripts
    )

    # Initialize dysarthric evaluator
    print("Initializing DysarthricASREvaluator...")
    dysarthric_evaluator = DysarthricASREvaluator(
        device='cuda' if torch.cuda.is_available() else 'cpu',
        compute_wer=True,
        compute_semscore=False,  # Set to True if you want to compute SemScore
    )

    # Setup distributed training params
    num_gpus = accelerator.num_processes
    print(f'Training on {num_gpus} GPUs')
    assert (
        config.training.batch_size % (num_gpus * config.training.batch_size_per_gpu) == 0
    ), 'Batch size must be divisible by the number of GPUs'

    gradient_accumulation_steps = config.training.batch_size // (num_gpus * config.training.batch_size_per_gpu)
    print(f'Gradient accumulation steps: {gradient_accumulation_steps}')
    print(f'Effective batch size: {config.training.batch_size} (per GPU: {config.training.batch_size_per_gpu})')

    num_training_samples = len(train_dataset)
    # Fixed: Use global batch size for gradient updates, not micro-batch size
    num_training_steps = (num_training_samples // config.training.batch_size) * config.training.num_train_epochs
    steps_per_epoch = num_training_samples // config.training.batch_size

    print(f'Number of training samples: {num_training_samples}')
    print(f'Number of training steps: {num_training_steps}')
    print(f'Number of training steps = {num_training_steps} (per epoch: {steps_per_epoch})')

    # Configure precision
    if config.model.use_flash_attention:
        fp16 = False
        bf16 = True
    else:
        fp16 = True
        bf16 = False
    
    if accelerator.is_main_process:
        # Print the number of TRAINABLE parameters in the model
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f'-' * 50)
        print(f'# Trainable parameters: {trainable_params / 1e6:.2f}M')
        print(f'# Total parameters: {total_params / 1e6:.2f}M')
        print(f'Trainable parameters percentage: {trainable_params / total_params * 100:.2f}%')
        print(f'-' * 50)

    # Training arguments
    training_args = TrainingArguments(
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.batch_size_per_gpu,
        gradient_checkpointing=False,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim='adamw_torch',
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-7,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        max_grad_norm=1.0,
        lr_scheduler_type='linear',
        warmup_steps=config.training.warmup_steps,
        logging_steps=config.training.logging_steps,
        output_dir=config.training.output_dir,
        save_strategy='epoch',
        eval_strategy='epoch',
        save_total_limit=3,
        bf16=bf16,
        fp16=fp16,
        remove_unused_columns=False,
        report_to=config.training.report_to,
        deepspeed=None,
        disable_tqdm=not config.training.use_tqdm,
        dataloader_num_workers=config.hardware.num_workers,
        ddp_find_unused_parameters=False,  # Usually False for Granite
        load_best_model_at_end=False,
        metric_for_best_model='loss',
        greater_is_better=False,
    )

    # Create output directory
    out_path = Path(training_args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Evaluate before fine-tuning (if configured)
    if config.evaluation.eval_before_training:
        print("🔍 Evaluating model before fine-tuning...")
        wer_before, semscore_before = evaluate_asr(
            model,
            processor,
            test_dataset,
            dysarthric_evaluator,
            save_path=out_path / 'eval_before.json',
            disable_tqdm=not config.training.use_tqdm,
            eval_batch_size=config.evaluation.eval_batch_size,
            max_new_tokens=config.evaluation.max_new_tokens,
        )
        if accelerator.is_main_process:
            print(f'WER before fine-tuning: {wer_before:.4f}')
            print(f'SemScore before fine-tuning: {semscore_before:.4f}')

    # Create trainer and start training
    data_collator = GraniteCollator(processor)
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=processor,
    )

    print("🎯 Starting fine-tuning...")
    trainer.train()
    
    # Save final model
    trainer.save_model()
    if accelerator.is_main_process:
        processor.save_pretrained(training_args.output_dir)
    accelerator.wait_for_everyone()

    # Get best model
    best_ckpt_path = trainer.state.best_model_checkpoint
    
    # Clean up memory
    del model
    del trainer
    __import__('gc').collect()
    torch.cuda.empty_cache()
    
    print(f"Best checkpoint path: {best_ckpt_path}")
    if best_ckpt_path:
        print(f"Loading best checkpoint from {best_ckpt_path}")
        model = GraniteSpeechForConditionalGeneration.from_pretrained(
            best_ckpt_path,
            torch_dtype=torch.bfloat16 if config.model.use_flash_attention else torch.float32,
            trust_remote_code=True,
        ).to('cuda')
        
    # Save final model in output directory + /best_model
    if accelerator.is_main_process:
        final_model_path = out_path / 'best_model'
        final_model_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(final_model_path)
        processor.save_pretrained(final_model_path)
        print(f"Final model saved to {final_model_path}")
        
    wer_after, semscore_after = evaluate_asr(
        model,
        processor,
        test_dataset,
        dysarthric_evaluator,
        save_path=out_path / 'eval_after.json',
        disable_tqdm=not config.training.use_tqdm,
        eval_batch_size=config.evaluation.eval_batch_size,
        max_new_tokens=config.evaluation.max_new_tokens,
    )
    
    if accelerator.is_main_process:
        print(f'🎉 Training complete!')
        print(f'WER after fine-tuning: {wer_after:.4f}')
        print(f'SemScore after fine-tuning: {semscore_after:.4f}')
        
        # Calculate improvements if we have before results
        if config.evaluation.eval_before_training:
            wer_improvement = wer_before - wer_after
            semscore_improvement = semscore_after - semscore_before
            try:
                print(f'WER improvement: {wer_improvement:.4f} ({wer_improvement/wer_before*100:.2f}%)')
            except ZeroDivisionError:
                print('WER improvement: N/A (before WER was zero)')
            
            try:
                print(f'SemScore improvement: {semscore_improvement:.4f} ({semscore_improvement/semscore_before*100:.2f}%)')
            except ZeroDivisionError:
                print('SemScore improvement: N/A (before SemScore was zero)')


if __name__ == '__main__':
    main()