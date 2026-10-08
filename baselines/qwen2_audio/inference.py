#!/usr/bin/env python3
"""
Inference script for fine-tuned Qwen2-Audio model on dysarthric speech ASR.
Generates predictions in JSONL format with audio_filepath, text, duration, etiology, pred_text.
"""

import os
import json
from pathlib import Path
from typing import List, Dict

import torch
from tqdm import tqdm
from transformers import (
    Qwen2AudioForConditionalGeneration,
    AutoProcessor,
    StoppingCriteriaList,
)
from peft import PeftModel

# Import from training script
from sft import (
    MultipleTokenBatchStoppingCriteria,
    ANSWER_SUFFIX,
    _IGNORE_INDEX,
)

# Import custom modules
from yaml_config_manager import load_config
from dataset import Qwen2AudioDysarthricDataset


def qwen2audio_inference_collate_fn(batch, processor):
    """
    Collate function for Qwen2-Audio inference data batching
    """
    # Skip empty batches
    if not batch:
        return None
        
    # Filter out any empty samples
    batch = [b for b in batch if b['text'].strip() != '']
    if not batch:
        return None
    
    # Extract texts and audios
    texts = [item['text'] for item in batch]
    audios = [item['audio'] for item in batch]
    
    # Preserve metadata for output
    metadata = []
    for item in batch:
        metadata.append({
            'audio_filepath': item['audio_path'],
            'text': item['target'],  # Ground truth transcript
            'duration': item['duration'],
            'etiology': item['etiology'],
            'contributor_id': item['contributor_id'],
            'category': item['category'],
            'speech_type': item.get('speech_type', ''),  # New field
        })
    
    # Use processor to handle the batching
    try:
        # Process the batch using Qwen2-Audio processor
        inputs = processor(
            text=texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            sampling_rate=16000,
        )
        
        # Add metadata for output
        inputs['metadata'] = metadata
        
        return inputs
        
    except Exception as e:
        print(f"Error in collate function: {e}")
        return None


@torch.no_grad()
def run_inference(
    model, 
    processor, 
    test_dataset, 
    output_path,
    batch_size=16,
    max_new_tokens=256,
    disable_tqdm=False,
    max_samples=None,
):
    """Run inference on test dataset and save results in JSONL format."""
    
    model.eval()
    results = []
    
    # Limit samples if specified
    if max_samples is not None and max_samples < len(test_dataset):
        indices = list(range(max_samples))
    else:
        indices = list(range(len(test_dataset)))
    
    # Create dataloader with proper batch size
    test_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(test_dataset, indices),
        batch_size=batch_size,
        collate_fn=lambda batch: qwen2audio_inference_collate_fn(batch, processor),
        shuffle=False,
        drop_last=False,
        num_workers=min(batch_size, 8),  # Reduced for Qwen2-Audio
        prefetch_factor=2,
        pin_memory=True,
    )
    
    # Configure stop tokens for generation
    stop_tokens = ["<|im_end|>", processor.tokenizer.eos_token]
    stop_tokens_ids = processor.tokenizer(
        stop_tokens, 
        add_special_tokens=False, 
        padding="longest", 
        return_tensors="pt"
    )["input_ids"]
    stop_tokens_ids = stop_tokens_ids.to(model.device)
    
    # Process batches
    with tqdm(
        total=len(indices), 
        disable=disable_tqdm, 
        desc='Running inference'
    ) as pbar:
        
        for inputs in test_dataloader:
            # Skip empty batches
            if inputs is None:
                continue
            
            current_batch_size = inputs['input_ids'].size(0)
            batch_metadata = inputs['metadata']
            
            # Setup stopping criteria
            stopping_criteria = StoppingCriteriaList([
                MultipleTokenBatchStoppingCriteria(
                    stop_tokens_ids, 
                    batch_size=current_batch_size
                )
            ])
            
            # Move inputs to device
            generation_inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                               for k, v in inputs.items() if k != 'metadata'}
            
            # Generate transcriptions
            generated_ids = model.generate(
                **generation_inputs, 
                eos_token_id=processor.tokenizer.eos_token_id, 
                max_new_tokens=max_new_tokens,
                stopping_criteria=stopping_criteria,
                do_sample=True,
                top_p=0.8,
                temperature=1.0,
                pad_token_id=processor.tokenizer.pad_token_id,
            )
            
            # Get stopping indices from criteria
            stop_tokens_idx = stopping_criteria[0].stop_tokens_idx.reshape(
                current_batch_size, -1
            )[:, 0]
            stop_tokens_idx = torch.where(
                stop_tokens_idx > 0,
                stop_tokens_idx - stop_tokens_ids.shape[-1],
                generated_ids.shape[-1],
            )
            
            # Extract only the generated part and decode
            generated_texts = []
            for _pred_ids, _stop_tokens_idx in zip(generated_ids, stop_tokens_idx):
                # Extract generated portion after the input
                generated_portion = _pred_ids[inputs["input_ids"].shape[1]:_stop_tokens_idx]
                
                # Decode the prediction
                pred_text = processor.decode(
                    generated_portion,
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False
                )
                generated_texts.append(pred_text)
            
            # Remove Qwen2-Audio prefix if present
            ANSWER_PREFIX = "The original content of this audio is:"
            generated_texts = [
                pred[len(ANSWER_PREFIX):].strip() if pred.startswith(ANSWER_PREFIX) else pred.strip()
                for pred in generated_texts
            ]
            
            # Combine predictions with metadata
            for i, pred_text in enumerate(generated_texts):
                if i < len(batch_metadata):
                    result = batch_metadata[i].copy()
                    result['pred_text'] = pred_text
                    results.append(result)
            
            # Update progress
            pbar.update(current_batch_size)
    
    # Save results to JSONL file
    print(f"\nSaving {len(results)} results to {output_path}")
    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    return results


def main():
    # Load configuration
    config = load_config()
    
    # Set environment variables
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    print("🚀 Starting Qwen2-Audio dysarthric speech ASR inference...")
    
    # Get paths from config
    model_path = config.testing.trained_model_path
    output_path = config.testing.output_path
    
    # If output path not specified, create one based on model path
    if output_path is None:
        output_path = Path(model_path).parent / 'predictions.jsonl'
    else:
        output_path = Path(output_path)
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model from: {model_path}")
    print(f"Output will be saved to: {output_path}")
    
    # Load processor from trained model
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    
    # Load fine-tuned model
    print("Loading fine-tuned Qwen2-Audio model...")
    
    # Check if this is a PEFT model (has adapter_config.json)
    if (Path(model_path) / "adapter_config.json").exists():
        print("Loading PEFT model...")
        # Load base model first
        base_model = Qwen2AudioForConditionalGeneration.from_pretrained(
            config.model.name_or_path,  # Use original model path
            torch_dtype=torch.bfloat16 if config.model.use_flash_attention else torch.float32,
            trust_remote_code=True,
        )
        # Load PEFT adapter
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        print("Loading full model...")
        # Load full fine-tuned model
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if config.model.use_flash_attention else torch.float32,
            trust_remote_code=True,
        )
    
    # Move to GPU
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Model loaded successfully on {model.device}")
    
    # Create test dataset
    print("Creating test dataset...")
    test_dataset = Qwen2AudioDysarthricDataset(
        processor=processor,
        tsv_path=config.data.test_tsv_path,
        data_root_path=config.data.data_root_path,
        split='test',
        max_size=config.testing.max_samples,
        instruction=config.data.instruction,
        use_etiology_in_instruction=config.data.use_etiology_in_instruction,
        etiology_token=config.data.etiology_token,
        normalize_transcripts=False,  # Never normalize for test
    )
    
    print(f"Test dataset created with {len(test_dataset)} samples")
    
    # Run inference
    print(f"\n🎯 Running inference with batch size {config.testing.batch_size}...")
    results = run_inference(
        model=model,
        processor=processor,
        test_dataset=test_dataset,
        output_path=output_path,
        batch_size=config.testing.batch_size,
        max_new_tokens=config.testing.max_new_tokens,
        disable_tqdm=config.testing.disable_tqdm,
        max_samples=config.testing.max_samples,
    )
    
    print(f"\n✅ Inference complete! {len(results)} samples processed.")
    print(f"Results saved to: {output_path}")
    
    # Print a few sample results
    if results:
        print("\nSample results:")
        for i, result in enumerate(results[:3]):
            print(f"\nSample {i+1}:")
            print(f"  Audio: {Path(result['audio_filepath']).name}")
            print(f"  Reference: {result['text']}")
            print(f"  Prediction: {result['pred_text']}")
            print(f"  Duration: {result['duration']:.2f}s")
            print(f"  Etiology: {result['etiology']}")
            print(f"  Contributor: {result['contributor_id']}")
            print(f"  Category: {result['category']}")


if __name__ == '__main__':
    main()