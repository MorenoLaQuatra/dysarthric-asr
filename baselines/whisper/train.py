#!/usr/bin/env python3
"""
Fine-tune Whisper Large V3 on dysarthric speech ASR
using SAPC dataset with comprehensive evaluation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
import pandas as pd
import numpy as np
from accelerate import Accelerator
from accelerate.utils import gather_object
from tqdm import tqdm
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
    WhisperFeatureExtractor,
    Trainer,
    TrainingArguments,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import evaluate

# Import your custom modules
from yaml_config_manager import load_config
from dataset import WhisperDysarthricSpeechDataset
from evaluation.sap_evaluator import DysarthricASREvaluator

# Constants
_IGNORE_INDEX = -100


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator for Whisper that pads inputs and labels.
    """
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels since they have to be of different lengths and need different padding methods
        # First treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # Pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), _IGNORE_INDEX)

        # If bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def create_model_and_processor(model_name_or_path, language="english", task="transcribe"):
    """Create and configure the Whisper model and processor."""
    
    # Load processor components
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)
    tokenizer = WhisperTokenizer.from_pretrained(
        model_name_or_path, 
        language=language, 
        task=task
    )
    processor = WhisperProcessor.from_pretrained(
        model_name_or_path, 
        language=language, 
        task=task
    )
    
    # Load model
    model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path)
    
    # Set language and task tokens
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.language = language
    model.generation_config.task = task
    
    # Set special tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return model, processor


def compute_metrics(eval_pred, processor, metric):
    """Compute WER metric during evaluation."""
    pred_ids = eval_pred.predictions
    label_ids = eval_pred.label_ids

    # Replace -100 with pad token id
    label_ids[label_ids == _IGNORE_INDEX] = processor.tokenizer.pad_token_id

    # Decode predictions and labels
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Calculate WER
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


@torch.no_grad()
def evaluate_whisper(
    model, processor, test_dataset, dysarthric_evaluator, 
    save_path=None, disable_tqdm=False, eval_batch_size=16,
    max_new_tokens=256, language="english", task="transcribe"
):
    """Evaluate Whisper model using DysarthricASREvaluator."""
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    # Ensure model is on the correct device
    device = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_references = []

    # Create evaluation dataloader
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        collate_fn=data_collator,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
    )

    for batch in tqdm(
        test_dataloader, disable=(rank != 0) or disable_tqdm, desc='Running evaluation'
    ):
        # Move inputs to the same device as model
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Generate transcriptions
        generated_tokens = model.generate(
            input_features=batch["input_features"],  # Use input_features instead of inputs
            forced_decoder_ids=processor.get_decoder_prompt_ids(language=language, task=task),
            max_new_tokens=max_new_tokens,
        )

        # Decode predictions
        generated_text = processor.batch_decode(generated_tokens, skip_special_tokens=True)
        
        # Get reference texts
        labels = batch["labels"]
        # Replace -100 with pad token for decoding
        labels[labels == _IGNORE_INDEX] = processor.tokenizer.pad_token_id
        reference_text = processor.batch_decode(labels, skip_special_tokens=True)
        
        # Collect results
        all_predictions.extend(generated_text)
        all_references.extend(reference_text)

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
    
    print("🚀 Starting Whisper Large V3 dysarthric speech ASR training...")
    print(f"Config loaded: {config}")
    
    # Setup accelerator for distributed training
    accelerator = Accelerator()

    # Load model and processor
    with accelerator.local_main_process_first():
        print("Loading Whisper model and processor...")
        model, processor = create_model_and_processor(
            config.model.name_or_path,
            language=config.model.language,
            task=config.model.task
        )
        
    # Get rank and world size for distributed training
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # Create datasets
    print("Creating datasets...")
    train_dataset = WhisperDysarthricSpeechDataset(
        processor=processor,
        tsv_path=config.data.train_tsv_path,
        data_root_path=config.data.data_root_path,
        split='train',
        max_size=config.data.train_size,
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
    
    validation_dataset = WhisperDysarthricSpeechDataset(
        processor=processor,
        tsv_path=config.data.dev_tsv_path,
        data_root_path=config.data.data_root_path,
        split='validation',
        max_size=config.data.eval_size,
        normalize_transcripts=False,  # Validation should not normalize transcripts
    )
    
    test_dataset = WhisperDysarthricSpeechDataset(
        processor=processor,
        tsv_path=config.data.test_tsv_path,
        data_root_path=config.data.data_root_path,
        split='test',
        max_size=config.data.test_size,
        normalize_transcripts=False,  # Test should not normalize transcripts
    )

    # Initialize dysarthric evaluator
    print("Initializing DysarthricASREvaluator...")
    dysarthric_evaluator = DysarthricASREvaluator(
        device='cuda' if torch.cuda.is_available() else 'cpu',
        compute_wer=True,
        compute_semscore=config.evaluation.get('compute_semscore', False),
    )

    # Setup WER metric for training
    wer_metric = evaluate.load("wer")

    # Setup distributed training params
    num_gpus = accelerator.num_processes
    print(f'Training on {num_gpus} GPUs')
    
    gradient_accumulation_steps = config.training.batch_size // (num_gpus * config.training.batch_size_per_gpu)

    if accelerator.is_main_process:
        # Print the number of trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f'-' * 50)
        print(f'# Trainable parameters: {trainable_params / 1e6:.2f}M')
        print(f'# Total parameters: {total_params / 1e6:.2f}M')
        print(f'Trainable parameters percentage: {trainable_params / total_params * 100:.2f}%')
        print(f'-' * 50)

    # Data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.training.output_dir,
        per_device_train_batch_size=config.training.batch_size_per_gpu,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        warmup_steps=config.training.warmup_steps,
        max_steps=config.training.get('max_steps', -1),
        num_train_epochs=config.training.num_train_epochs,
        gradient_checkpointing=True,
        fp16=config.training.get('fp16', True),
        evaluation_strategy="epoch",
        per_device_eval_batch_size=config.training.batch_size_per_gpu,
        predict_with_generate=True,
        generation_max_length=config.evaluation.max_new_tokens,
        save_strategy=config.training.save_strategy, # epoch
        eval_strategy=config.training.eval_strategy, # epoch
        save_steps=config.training.save_steps,
        eval_steps=config.training.eval_steps,
        logging_steps=config.training.logging_steps,
        report_to=config.training.report_to,
        # load_best_model_at_end=False,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        dataloader_num_workers=config.hardware.num_workers,
        remove_unused_columns=False,
        save_total_limit=3,
        disable_tqdm=not config.training.use_tqdm,
    )

    # Create output directory
    out_path = Path(training_args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Evaluate before fine-tuning (if configured)
    if config.evaluation.eval_before_training:
        print("🔍 Evaluating model before fine-tuning...")
        wer_before, semscore_before = evaluate_whisper(
            model,
            processor,
            test_dataset,
            dysarthric_evaluator,
            save_path=out_path / 'eval_before.json',
            disable_tqdm=not config.training.use_tqdm,
            eval_batch_size=config.evaluation.eval_batch_size,
            max_new_tokens=config.evaluation.max_new_tokens,
            language=config.model.language,
            task=config.model.task,
        )
        if accelerator.is_main_process:
            print(f'WER before fine-tuning: {wer_before:.4f}')
            print(f'SemScore before fine-tuning: {semscore_before:.4f}')


    # Create trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, processor, wer_metric),
        tokenizer=processor.feature_extractor,
    )

    # Start training
    print("🎯 Starting fine-tuning...")
    trainer.train()
    
    # Save final model
    trainer.save_model()
    if accelerator.is_main_process:
        processor.save_pretrained(training_args.output_dir)
    accelerator.wait_for_everyone()

    # Get best model path
    best_ckpt_path = trainer.state.best_model_checkpoint
    
    # Clean up memory
    del model
    del trainer
    __import__('gc').collect()
    torch.cuda.empty_cache()
    
    # Load best model for final evaluation
    print(f"Best checkpoint path: {best_ckpt_path}")
    if best_ckpt_path:
        print(f"Loading best checkpoint from {best_ckpt_path}")
        model = WhisperForConditionalGeneration.from_pretrained(best_ckpt_path)
        model = model.to(device)  # Make sure to move to correct device
        
    # save final model to output_dir + '/best_model'
    if accelerator.is_main_process:
        print(f"Saving final model to {out_path / 'best_model'}")
        model.save_pretrained(out_path / 'best_model')
        processor.save_pretrained(out_path / 'best_model')
    else:
        print("Skipping saving model on non-main process.")
        

    # Final evaluation
    wer_after, semscore_after = evaluate_whisper(
        model,
        processor,
        test_dataset,
        dysarthric_evaluator,
        save_path=out_path / 'eval_after.json',
        disable_tqdm=not config.training.use_tqdm,
        eval_batch_size=config.evaluation.eval_batch_size,
        max_new_tokens=config.evaluation.max_new_tokens,
        language=config.model.language,
        task=config.model.task,
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