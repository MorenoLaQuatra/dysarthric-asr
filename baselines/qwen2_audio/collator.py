import torch
from dataclasses import dataclass
from typing import Dict, List, Any
import numpy as np


@dataclass
class Qwen2AudioDataCollator:
    """
    Data collator for Qwen2-Audio that handles both audio and text inputs.
    """
    processor: Any
    sampling_rate: int = 16000
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate a batch of samples for Qwen2-Audio training/evaluation.
        
        Args:
            batch: List of samples from dataset
            
        Returns:
            Dictionary with processed inputs ready for model
        """
        # Extract components from batch
        texts = [sample['text'] for sample in batch]
        audios = [sample['audio'] for sample in batch]
        sampling_rates = [sample.get('sampling_rate', self.sampling_rate) for sample in batch]
        
        # Process with Qwen2-Audio processor
        try:
            inputs = self.processor(
                text=texts,
                audio=audios,
                return_tensors="pt",
                padding=True,
                sampling_rate=self.sampling_rate,
            )
        except Exception as e:
            print(f"Error in processor: {e}")
            # Fallback to empty inputs
            inputs = {
                'input_ids': torch.zeros((len(batch), 1), dtype=torch.long),
                'attention_mask': torch.zeros((len(batch), 1), dtype=torch.long),
                'input_features': torch.zeros((len(batch), 1, 80, 3000)),
                'feature_attention_mask': torch.zeros((len(batch), 1, 3000), dtype=torch.long),
            }
        
        # Handle labels for training
        if batch[0].get('labels') is not None:
            # Tokenize labels (targets) for training
            label_texts = [sample['labels'] for sample in batch]
            try:
                label_inputs = self.processor.tokenizer(
                    label_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                inputs['labels'] = label_inputs['input_ids']
                # Replace padding tokens with -100 for loss computation
                inputs['labels'][inputs['labels'] == self.processor.tokenizer.pad_token_id] = -100
            except Exception as e:
                print(f"Error tokenizing labels: {e}")
                inputs['labels'] = None
        
        # Add metadata fields
        metadata_fields = ['target', 'etiology', 'duration', 'transcript', 
                          'audio_path', 'contributor_id', 'category']
        
        for field in metadata_fields:
            if field in batch[0]:
                inputs[field] = [sample[field] for sample in batch]
        
        return inputs