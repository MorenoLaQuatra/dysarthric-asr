#!/usr/bin/env python3
"""
Inference script for fine-tuned Granite Speech model on dysarthric speech ASR.
Generates predictions in JSONL format with audio_filepath, text, duration, etiology, pred_text.
"""

import os
import json
from pathlib import Path
from typing import List, Dict

import torch
from tqdm import tqdm
from transformers.models.granite_speech import GraniteSpeechForConditionalGeneration, GraniteSpeechProcessor

# Import from training script
from sft import GraniteCollator

# Import custom modules
from yaml_config_manager import load_config
from dataset import GraniteDysarthricDataset


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
    
    # Create dataloader with Granite collator
    collator = GraniteCollator(processor, inference_mode=True)
    test_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(test_dataset, indices),
        batch_size=batch_size,
        collate_fn=collator,
        shuffle=False,
        drop_last=False,
        num_workers=min(batch_size, 16),
        prefetch_factor=2,
        pin_memory=True,
    )
    
    # Process batches
    with tqdm(
        total=len(indices), 
        disable=disable_tqdm, 
        desc='Running inference'
    ) as pbar:
        
        batch_idx = 0
        for inputs in test_dataloader:
            # Skip empty batches
            if inputs is None:
                continue
            
            current_batch_size = inputs.input_ids.size(0)
            
            # Get metadata for current batch from dataset
            batch_metadata = []
            for i in range(current_batch_size):
                dataset_idx = indices[batch_idx * batch_size + i]
                sample_data = test_dataset[dataset_idx]
                try:
                    batch_metadata.append({
                        'audio_filepath': sample_data['audio_path'],
                        'text': sample_data['text'],
                        'duration': sample_data['duration'],
                        'etiology': sample_data['etiology'],
                        'speech_type': sample_data['speech_type'],  # New field
                    })
                except KeyError as e:
                    print(f"Warning: Missing key {e} in sample {dataset_idx}. Skipping this sample.")
                    continue
            
            # Move inputs to device
            inputs = inputs.to(model.device)
            
            # Generate transcriptions using Granite's approach
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                generated_ids = model.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens,
                    num_beams=4,
                    early_stopping=True,
                    pad_token_id=processor.tokenizer.pad_token_id,
                )
            
            # Extract only the generated part and decode
            input_length = inputs.input_ids.shape[1]
            generated_outputs = generated_ids[:, input_length:].cpu()
            
            # Decode the predictions
            generated_texts = []
            for output in generated_outputs:
                pred_text = processor.tokenizer.decode(
                    output,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )
                generated_texts.append(pred_text.strip())
            
            # Combine predictions with metadata
            for i, pred_text in enumerate(generated_texts):
                if i < len(batch_metadata):
                    result = batch_metadata[i].copy()
                    result['pred_text'] = pred_text
                    results.append(result)
            
            # Update progress
            pbar.update(current_batch_size)
            batch_idx += 1
    
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
    
    print("🚀 Starting dysarthric speech ASR inference with Granite Speech...")
    
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
    print("Loading Granite Speech processor...")
    processor = GraniteSpeechProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    
    # Load fine-tuned model
    print(f"Loading fine-tuned Granite Speech model: {model_path}...")
    model = GraniteSpeechForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if config.model.use_flash_attention else torch.float32,
        trust_remote_code=True,
    ).to('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Model loaded successfully on {model.device}")
    
    # Create test dataset
    print("Creating test dataset...")
    test_dataset = GraniteDysarthricDataset(
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


if __name__ == '__main__':
    main()