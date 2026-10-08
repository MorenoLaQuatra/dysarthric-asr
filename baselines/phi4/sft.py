#!/usr/bin/env python3
"""
Fine-tune Phi-4 multimodal model on dysarthric speech ASR
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
    AutoModelForCausalLM,
    AutoProcessor,
    BatchFeature,
    Trainer,
    TrainingArguments,
    StoppingCriteria,
    StoppingCriteriaList,
)

# Import your custom modules
from yaml_config_manager import load_config
from dataset import DysarthricSpeechDataset
from evaluation.sap_evaluator import DysarthricASREvaluator
from unfreezing_manager import UnfreezingManager

# Constants
ANSWER_SUFFIX = "<|end|><|endoftext|>"
_IGNORE_INDEX = -100

# Ensure that LossScaler is safe for serialization
import torch.serialization
try:
    from deepspeed.runtime.fp16.loss_scaler import LossScaler
    from deepspeed.runtime.zero.config import ZeroStageEnum
    torch.serialization.add_safe_globals([LossScaler, ZeroStageEnum])
except ImportError:
    print("DeepSpeed not available, skipping safe globals")
    
    
class MultipleTokenBatchStoppingCriteria(StoppingCriteria):
    """Stopping criteria capable of receiving multiple stop-tokens and handling batched inputs."""

    def __init__(self, stop_tokens: torch.LongTensor, batch_size: int = 1) -> None:
        """Initialize the multiple token batch stopping criteria.

        Args:
            stop_tokens: Stop-tokens.
            batch_size: Batch size.
        """
        self.stop_tokens = stop_tokens
        self.max_stop_tokens = stop_tokens.shape[-1]
        self.stop_tokens_idx = torch.zeros(batch_size, dtype=torch.long, device=stop_tokens.device)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Only gather the maximum number of inputs compatible with stop tokens
        # and checks whether generated inputs are equal to `stop_tokens`
        generated_inputs = torch.eq(input_ids[:, -self.max_stop_tokens:].unsqueeze(1), self.stop_tokens)
        equal_generated_inputs = torch.all(generated_inputs, dim=2)

        # Mark the position where a stop token has been produced for each input in the batch,
        # but only if the corresponding entry is not already set
        sequence_idx = torch.any(equal_generated_inputs, dim=1)
        sequence_set_mask = self.stop_tokens_idx == 0
        self.stop_tokens_idx[sequence_idx & sequence_set_mask] = input_ids.shape[-1]

        return torch.all(self.stop_tokens_idx)


def pad_sequence(sequences, padding_side='right', padding_value=0):
    """
    Pad a list of sequences to the same length.
    sequences: list of tensors in [seq_len, *] shape
    """
    assert padding_side in ['right', 'left']
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == 'right':
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output


def cat_with_pad(tensors, dim, padding_value=0):
    """
    Concatenate along dim, while pad to max for all other dims
    """
    ndim = tensors[0].dim()
    assert all(
        t.dim() == ndim for t in tensors[1:]
    ), 'All tensors must have the same number of dimensions'

    out_size = [max(t.shape[i] for t in tensors) for i in range(ndim)]
    out_size[dim] = sum(t.shape[dim] for t in tensors)
    output = tensors[0].new_full(out_size, padding_value)

    index = 0
    for t in tensors:
        # Create a slice list where every dimension except dim is full slice
        slices = [slice(0, t.shape[d]) for d in range(ndim)]
        # Update only the concat dimension slice
        slices[dim] = slice(index, index + t.shape[dim])

        output[slices] = t
        index += t.shape[dim]

    return output


def asr_collate_fn(batch):
    """
    Collate function for ASR data batching
    """
    # Skip empty batches
    if not batch:
        return None
        
    # Filter out any empty tensors
    batch = [b for b in batch if b['input_ids'].size(1) > 1]
    if not batch:
        return None
    
    input_ids_list = []
    labels_list = []
    input_audio_embeds_list = []
    audio_embed_sizes_list = []
    audio_attention_mask_list = []
    
    for inputs in batch:
        input_ids_list.append(inputs['input_ids'][0])
        labels_list.append(inputs['labels'][0])
        input_audio_embeds_list.append(inputs['input_audio_embeds'])
        audio_embed_sizes_list.append(inputs['audio_embed_sizes'])
        audio_attention_mask_list.append(
            inputs['input_audio_embeds'].new_full((inputs['input_audio_embeds'].size(1),), True, dtype=torch.bool)
        )

    try:
        input_ids = pad_sequence(input_ids_list, padding_side='left', padding_value=0)
        labels = pad_sequence(labels_list, padding_side='left', padding_value=_IGNORE_INDEX)
        audio_attention_mask = (
            pad_sequence(audio_attention_mask_list, padding_side='right', padding_value=False)
            if len(audio_attention_mask_list) > 1
            else None
        )
    except Exception as e:
        print(f"Error in collate function: {e}")
        return None
        
    attention_mask = (input_ids != 0).long()
    input_audio_embeds = cat_with_pad(input_audio_embeds_list, dim=0)
    audio_embed_sizes = torch.cat(audio_embed_sizes_list)

    return BatchFeature(
        {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'input_audio_embeds': input_audio_embeds,
            'audio_embed_sizes': audio_embed_sizes,
            'audio_attention_mask': audio_attention_mask,
            'input_mode': 2,  # speech mode
        }
    )


def create_model(model_name_or_path, use_flash_attention=False, unfreezing_config=None):
    """Create and configure the model."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16 if use_flash_attention else torch.float32,
        _attn_implementation='flash_attention_2' if use_flash_attention else 'sdpa',
        trust_remote_code=True,
    ).to('cuda')
    
    # Enable LoRA adapter for 'speech'
    model.set_lora_adapter('speech')
    
    if unfreezing_config:
        print("Applying unfreezing strategy...")
        unfreezing_manager = UnfreezingManager(unfreezing_config)
        stats = unfreezing_manager.apply_unfreezing_strategy(model)
        unfreezing_manager.print_unfreezing_stats(stats)

    return model

def manage_additional_token_if_needed(processor, model, config):
    '''
    If the model uses additional tokens as per config, add them as special tokens to the processor and update the model.
    '''
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
    
    print(f"\n\n Action tag map: \n{action_tag_map}\n\n")

    for action, tags in action_tag_map.items():
        if getattr(normalizer, action) in ["tag", "unk"]:
            print(f"{action.upper()} is set to 'tag', adding {tags} to processor")
            additional_tokens.extend(tags)

    # Add tokens to processor if needed
    if additional_tokens:
        original_token_count = len(processor.tokenizer)
        print(f"Before adding, processor has {original_token_count} tokens.")
        print(f"Adding additional tokens: {additional_tokens}")
        # Add unique tokens to avoid duplicates
        additional_tokens = list(set(additional_tokens))
        print(f"After deduplication, {len(additional_tokens)} unique tokens to add.")
        print(f"Total tokens after addition will be {original_token_count + len(additional_tokens)}")
        processor.tokenizer.add_special_tokens({'additional_special_tokens': list(set(additional_tokens))})
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
    """Evaluate ASR model using DysarthricASREvaluator."""
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
    model.eval()
    all_predictions = []
    all_references = []

    # Create evaluation dataloader
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        collate_fn=asr_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=eval_batch_size,  # Use batch size as num_workers for simplicity
        prefetch_factor=2,
        pin_memory=True,
    )
    
    # Configure stop tokens for generation
    stop_tokens = ["<|end|>", processor.tokenizer.eos_token]
    stop_tokens_ids = processor.tokenizer(stop_tokens, add_special_tokens=False, padding="longest", return_tensors="pt")["input_ids"]
    stop_tokens_ids = stop_tokens_ids.to(f'cuda:{local_rank}')

    for inputs in tqdm(
        test_dataloader, disable=(rank != 0) or disable_tqdm, desc='Running evaluation'
    ):
        # Skip empty batches
        if inputs is None:
            continue
            
        # Setup stopping criteria
        stopping_criteria = StoppingCriteriaList(
            [MultipleTokenBatchStoppingCriteria(stop_tokens_ids, batch_size=inputs.input_ids.size(0))]
        )
        
        # Move inputs to device
        inputs = inputs.to(f'cuda:{local_rank}')
        
        # Generate transcriptions
        generated_ids = model.generate(
            **inputs, 
            eos_token_id=processor.tokenizer.eos_token_id, 
            max_new_tokens=max_new_tokens,
            stopping_criteria=stopping_criteria,
        )

        # Get stopping indices from criteria
        stop_tokens_idx = stopping_criteria[0].stop_tokens_idx.reshape(inputs.input_ids.size(0), -1)[:, 0]
        stop_tokens_idx = torch.where(
            stop_tokens_idx > 0,
            stop_tokens_idx - stop_tokens_ids.shape[-1],
            generated_ids.shape[-1],
        )
        
        # Extract only the generated part and decode
        generated_text = [
            processor.decode(
                _pred_ids[inputs["input_ids"].shape[1] : _stop_tokens_idx], 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
            for _pred_ids, _stop_tokens_idx in zip(generated_ids, stop_tokens_idx)
        ]
        
        # Get reference texts (ground truth) - filter out both padding and ignored tokens
        reference_text = [
            processor.decode(_label_ids[(_label_ids != 0) & (_label_ids != _IGNORE_INDEX)]).removesuffix(ANSWER_SUFFIX) 
            for _label_ids in inputs["labels"]
        ]
        
        # Collect results
        all_predictions.extend(generated_text)
        all_references.extend(reference_text)

    # Gather results from all processes
    all_predictions = gather_object(all_predictions)
    all_references = gather_object(all_references)
    
    # Calculate WER and SemScore on main process
    if rank == 0:
        assert len(all_predictions) == len(all_references)
        
        # print some examples to see (:5)
        for i in range(min(5, len(all_predictions))):
            print(f"Example {i+1}:")
            print(f"Prediction: {all_predictions[i]}")
            print(f"Reference: {all_references[i]}")
            print("-" * 50)
        
        # Use DysarthricASREvaluator for comprehensive evaluation
        results = dysarthric_evaluator.evaluate(
            hypotheses=all_predictions,
            references=all_references,
            dual_reference_mode='auto',  # Generate with/without parentheses versions
            process_text=True  # Apply full normalization
        )
        
        wer = results['wer'] / 100.0  # Convert back to decimal for consistency
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

# def unfreeze_speech_components(model):
#     """Directly target verified components from your debug logs"""
#     # 1. Audio Embed Module (confirmed exists)
#     audio_embed = model.model.embed_tokens_extend.audio_embed

#     # 2. Entire Audio Encoder (simplified)
#     audio_encoder = audio_embed.encoder  # Direct access

#     # 3. Audio Projection (from debug logs)
#     audio_projection = audio_embed.audio_projection

#     # Unfreeze ONLY these 3 components
#     for component in [audio_embed, audio_encoder, audio_projection]:
#         for param in component.parameters():
#             param.requires_grad = True
#     return model


def main():
    # Load configuration
    config = load_config()
    
    # Set environment variables
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    print("🚀 Starting dysarthric speech ASR training...")
    print(f"Config loaded: {config}")
    
    # Setup accelerator for distributed training
    accelerator = Accelerator()

    # Load model and processor
    with accelerator.local_main_process_first():
        print("Loading processor...")
        processor = AutoProcessor.from_pretrained(
            config.model.name_or_path,
            trust_remote_code=True,
            padding_side='left',
        )
        
        print("Loading model...")
        if config.model.unfreezing == {}: print("[**Warning**] No unfreezing config provided, using default model loading (speech LoRA only)")
        model = create_model(
            config.model.name_or_path,
            use_flash_attention=config.model.use_flash_attention,
            unfreezing_config=config.model.unfreezing
        )
        
        # Manage additional tokens if needed
        manage_additional_token_if_needed(processor, model, config)
        
    # Get rank and world size for distributed training
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    # Create datasets using your DysarthricSpeechDataset
    print("Creating datasets...")
    train_dataset = DysarthricSpeechDataset(
        processor=processor,
        tsv_path=config.data.train_tsv_path,
        data_root_path=config.data.data_root_path,
        split='train',
        max_size=config.data.train_size,
        # Instruction for the model
        instruction=config.data.instruction,
        instruction_hinting=config.data.instruction_hinting,
        use_etiology_in_instruction=config.data.use_etiology_in_instruction,
        percentage_of_etiology_in_instruction=config.data.percentage_of_etiology_in_instruction,    
        etiology_token=config.data.etiology_token,
        # Transcript normalization parameters
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
    
    validation_dataset = DysarthricSpeechDataset(
        processor=processor,
        tsv_path=config.data.dev_tsv_path,
        data_root_path=config.data.data_root_path,
        split='train', # just to have uniform validation inside the Trainer (otherwise throws error of mismatched size)
        max_size=config.data.eval_size,
        # Instruction for the model
        instruction=config.data.instruction,
        instruction_hinting=config.data.instruction_hinting,
        use_etiology_in_instruction=config.data.use_etiology_in_instruction,
        percentage_of_etiology_in_instruction=config.data.percentage_of_etiology_in_instruction,
        etiology_token=config.data.etiology_token,
        # Transcript normalization parameters
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
    
    # Normalization disabled for TEST
    test_dataset = DysarthricSpeechDataset(
        processor=processor,
        tsv_path=config.data.test_tsv_path,
        data_root_path=config.data.data_root_path,
        split='test',
        max_size=config.data.test_size,
        # Instruction for the model
        instruction=config.data.instruction,
        instruction_hinting=config.data.instruction_hinting,
        use_etiology_in_instruction=config.data.use_etiology_in_instruction,
        percentage_of_etiology_in_instruction=config.data.percentage_of_etiology_in_instruction,
        etiology_token=config.data.etiology_token,
        normalize_transcripts=False,  # Test should not normalize transcripts
    )

    # Initialize dysarthric evaluator
    print("Initializing DysarthricASREvaluator...")
    dysarthric_evaluator = DysarthricASREvaluator(
        device='cuda' if torch.cuda.is_available() else 'cpu',
        compute_wer=True,
        compute_semscore=False,  # Set to True if you want to compute SemScore (takes longer)
    )

    # Setup distributed training params
    num_gpus = accelerator.num_processes
    print(f'Training on {num_gpus} GPUs')
    assert (
        config.training.batch_size % (num_gpus * config.training.batch_size_per_gpu) == 0
    ), 'Batch size must be divisible by the number of GPUs'
    gradient_accumulation_steps = config.training.batch_size // (num_gpus * config.training.batch_size_per_gpu)

    # Configure precision
    if config.model.use_flash_attention:
        fp16 = False
        bf16 = True
    else:
        fp16 = True
        bf16 = False
    
    if accelerator.is_main_process:
        # print the number of TRAINABLE parameters in the model
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
        gradient_checkpointing_kwargs={'use_reentrant': False},
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
        # save_only_model=True,
        bf16=bf16,
        fp16=fp16,
        remove_unused_columns=False,
        report_to=config.training.report_to,
        deepspeed=None,
        disable_tqdm=not config.training.use_tqdm,
        dataloader_num_workers=config.hardware.num_workers,
        ddp_find_unused_parameters=True,  # for unused audio model layers
        load_best_model_at_end=False, # we need to do it manually
        metric_for_best_model='loss',
        greater_is_better=False,  # Lower loss is better
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
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=asr_collate_fn,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
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
        model = AutoModelForCausalLM.from_pretrained(
            best_ckpt_path,
            torch_dtype=torch.bfloat16 if config.model.use_flash_attention else torch.float32,
            trust_remote_code=True,
            _attn_implementation='flash_attention_2' if config.model.use_flash_attention else 'sdpa',
        ).to('cuda')
        
        # Set LoRA adapter for speech again
        model.set_lora_adapter('speech')
        
    # save final model in output directory + /best_model
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