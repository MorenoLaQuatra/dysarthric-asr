#!/usr/bin/env python3
"""
Inference script for fine-tuned Whisper Large V3 on dysarthric speech ASR.
Generates predictions in JSONL format with audio_filepath, text, duration, etiology, pred_text.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import json
from pathlib import Path
from typing import List, Dict

import torch
from tqdm import tqdm
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

# Import from training script
from train import (
    DataCollatorSpeechSeq2SeqWithPadding,
    _IGNORE_INDEX,
)

# Import custom modules
from yaml_config_manager import load_config
from dataset import WhisperDysarthricSpeechDataset


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
    language="english",
    task="transcribe",
):
    """Run inference on test dataset and save results in JSONL format."""
    
    # Ensure model is on the correct device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    
    results = []
    
    # Limit samples if specified
    if max_samples is not None and max_samples < len(test_dataset):
        indices = list(range(max_samples))
    else:
        indices = list(range(len(test_dataset)))
    
    # Create data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    
    # Create dataloader
    test_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(test_dataset, indices),
        batch_size=batch_size,
        collate_fn=data_collator,
        shuffle=False,
        drop_last=False,
        num_workers=min(batch_size, 4),
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
        for batch in test_dataloader:
            current_batch_size = batch["input_features"].size(0)
            
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
                        'speech_type': sample_data.get('speech_type', '')
                    })
                except KeyError as e:
                    print(f"Warning: Missing key {e} in sample {dataset_idx}. Skipping this sample.")
                    continue
            
            # Move inputs to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Generate transcriptions
            generated_tokens = model.generate(
                input_features=batch["input_features"],
                forced_decoder_ids=processor.get_decoder_prompt_ids(language=language, task=task),
                max_new_tokens=max_new_tokens,
            )
            
            # Decode predictions
            generated_texts = processor.batch_decode(generated_tokens, skip_special_tokens=True)
            
            # Combine predictions with metadata
            for i, pred_text in enumerate(generated_texts):
                if i < len(batch_metadata):
                    result = batch_metadata[i].copy()
                    result['pred_text'] = pred_text.strip()
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
    
    print("🚀 Starting Whisper dysarthric speech ASR inference...")
    
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
    processor = WhisperProcessor.from_pretrained(
        model_path,
        language=config.model.language,
        task=config.model.task,
    )
    
    # Load fine-tuned model
    print("Loading fine-tuned Whisper model...")
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    
    # Move model to appropriate device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    print(f"Model loaded successfully on {device}")
    
    # Create test dataset
    print("Creating test dataset...")
    test_dataset = WhisperDysarthricSpeechDataset(
        processor=processor,
        tsv_path=config.data.test_tsv_path,
        data_root_path=config.data.data_root_path,
        split='test',
        max_size=config.testing.max_samples,
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
        language=config.model.language,
        task=config.model.task,
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