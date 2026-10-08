#!/usr/bin/env python3
"""
Dataset class for Whisper training on dysarthric speech data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
import os
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset
import librosa
import numpy as np
from pathlib import Path
from data.transcript_normalizer import TranscriptNormalizer


class WhisperDysarthricSpeechDataset(TorchDataset):
    def __init__(
            self, 
            processor, 
            tsv_path, 
            data_root_path, 
            split, 
            max_size=None, 
            sampling_rate=16000,
            
            # Transcript normalization parameters
            normalize_transcripts=True,
            remove_prompts=True,
            disfluency_action="keep",
            start_disfluency_tag="<|disfluency|>",
            end_disfluency_tag="<|end_disfluency|>",
            
            g_action="keep",
            w_action="remove",
            ss_action="keep",
            cs_action="keep",
            
            w_tag="<unclear>",
            start_cs_tag="<|external_speech|>",
            end_cs_tag="<|end_external_speech|>",
            
            process_spelled=True,
            capitalization="lower",
        ):
        """
        Dataset for Whisper training on Dysarthric Speech ASR.
        
        Args:
            processor: WhisperProcessor instance
            tsv_path: Path to the TSV metadata file
            data_root_path: Root path to prepend to audio_path
            split: Split name (train/validation/test)
            max_size: Maximum number of samples to use
            sampling_rate: Target sampling rate for audio
            normalize_transcripts: Whether to normalize transcripts
            **kwargs: Transcript normalization parameters
        """
        self.processor = processor
        self.data_root_path = Path(data_root_path)
        self.training = split == "train"
        self.sampling_rate = sampling_rate
        
        # Initialize transcript normalizer
        if normalize_transcripts and self.training:
            # Only normalize transcripts for training split
            self.normalizer = TranscriptNormalizer(
                remove_prompts=remove_prompts,
                disfluency_action=disfluency_action,
                start_disfluency_tag=start_disfluency_tag,
                end_disfluency_tag=end_disfluency_tag,
                g_action=g_action,
                w_action=w_action,
                ss_action=ss_action,
                cs_action=cs_action,
                w_tag=w_tag,
                start_cs_tag=start_cs_tag,
                end_cs_tag=end_cs_tag,
                process_spelled=process_spelled,
                capitalization=capitalization,
                other_action="remove",
            )
        else:
            self.normalizer = None
        
        # Load TSV file
        print(f"Loading {split} data from {tsv_path}")
        try:
            self.df = pd.read_csv(tsv_path, sep='\t')
        except Exception as e:
            raise FileNotFoundError(f"Could not load TSV file {tsv_path}: {e}")
        
        print(f"Loaded {len(self.df)} samples from TSV")
        
        # Filter valid samples
        self.df = self._filter_valid_samples()
        print(f"After filtering: {len(self.df)} valid samples")
        
        # Limit dataset size if specified
        if max_size and max_size < len(self.df):
            self.df = self.df.iloc[:max_size].reset_index(drop=True)
            print(f"Limited dataset to {len(self.df)} samples")
        
        # Reset index for clean access
        self.df = self.df.reset_index(drop=True)
        
        print(f"Initialized Whisper {split} dataset with {len(self.df)} samples")
        
        # Print some statistics
        if len(self.df) > 0:
            print(f"Sample statistics:")
            if 'etiology' in self.df.columns:
                etiologies = self.df['etiology'].dropna().unique()
                print(f"\tEtiologies: {', '.join(sorted(etiologies))}")

    def _filter_valid_samples(self):
        """Filter dataset to include only valid samples with audio files and transcripts."""
        print("Filtering valid samples...")
        
        initial_count = len(self.df)
        
        # Remove samples without audio_path or transcript
        self.df = self.df.dropna(subset=['audio_path', 'transcript'])
        print(f"  • After removing missing audio_path/transcript: {len(self.df)}/{initial_count}")
        
        # Remove samples with empty transcripts
        self.df = self.df[self.df['transcript'].str.strip() != '']
        print(f"  • After removing empty transcripts: {len(self.df)}/{initial_count}")
        
        # Check if audio files exist
        valid_mask = []
        missing_files = 0
        
        for idx, row in self.df.iterrows():
            # Handle both absolute and relative paths
            audio_path = row['audio_path']
            if not os.path.isabs(audio_path):
                # If relative path, prepend data_root_path
                full_audio_path = self.data_root_path / audio_path
            else:
                full_audio_path = Path(audio_path)
            
            if full_audio_path.exists():
                valid_mask.append(True)
                # Update the dataframe with the full path for consistency
                self.df.at[idx, 'audio_path'] = str(full_audio_path)
            else:
                valid_mask.append(False)
                missing_files += 1
                if missing_files <= 5:  # Show first 5 missing files
                    print(f"    Missing audio file: {full_audio_path}")
        
        if missing_files > 5:
            print(f"    ... and {missing_files - 5} more missing files")
        
        self.df = self.df[valid_mask].reset_index(drop=True)
        print(f"  • After checking audio file existence: {len(self.df)}/{initial_count}")
        
        return self.df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        Returns features compatible with Whisper training.
        """
        if idx >= len(self.df):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.df)}")
        
        row = self.df.iloc[idx]
        audio_path = row['audio_path']
        transcript = str(row['transcript']).strip()
        
        # Normalize transcript if normalizer is configured
        if self.normalizer:
            transcript = self.normalizer.normalize(transcript)
        
        # Double-check audio file exists
        if not os.path.isfile(audio_path):
            print(f"Warning: Audio file not found: {audio_path}")
            # Skip invalid samples during training
            if self.training and len(self.df) > 1:
                return self.__getitem__((idx + 1) % len(self.df))
            else:
                # For evaluation, return empty tensors
                return self._get_empty_sample()
        
        # Load audio file
        try:
            audio_array, sr = librosa.load(audio_path, sr=self.sampling_rate)
            
            # Convert stereo to mono if needed
            if audio_array.ndim > 1:
                audio_array = librosa.to_mono(audio_array)
            
            # Ensure we have valid audio
            if len(audio_array) == 0:
                raise ValueError("Empty audio file")
                
        except Exception as e:
            print(f"Error loading audio file {audio_path}: {e}")
            # Skip bad files during training
            if self.training and len(self.df) > 1:
                return self.__getitem__((idx + 1) % len(self.df))
            else:
                # For evaluation, return empty tensors
                return self._get_empty_sample()
        
        # Process audio with Whisper feature extractor
        try:
            # Convert to log-mel spectrogram
            input_features = self.processor.feature_extractor(
                audio_array, 
                sampling_rate=self.sampling_rate, 
                return_tensors="pt"
            ).input_features[0]  # Shape: [80, 3000]
            
        except Exception as e:
            print(f"Error processing audio {audio_path}: {e}")
            if self.training and len(self.df) > 1:
                return self.__getitem__((idx + 1) % len(self.df))
            else:
                return self._get_empty_sample()
        
        # Tokenize transcript
        try:
            # Tokenize the transcript
            labels = self.processor.tokenizer(
                transcript,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=448,  # Whisper's max sequence length
            ).input_ids[0]
            
        except Exception as e:
            print(f"Error tokenizing transcript for {audio_path}: {e}")
            if self.training and len(self.df) > 1:
                return self.__getitem__((idx + 1) % len(self.df))
            else:
                return self._get_empty_sample()
            
        # duration, etiology, text (original transcript), audio_path
        audio_array, sr = librosa.load(audio_path, sr=None)
        duration = librosa.get_duration(y=audio_array, sr=sr)
        
        current_etiology = row.get('etiology', '')
        original_transcript = row.get('transcript', '').strip()
        audio_path = row.get('audio_path', str(Path(audio_path).resolve()))

        return {
            "input_features": input_features,
            "labels": labels,
            "duration": duration,
            "etiology": current_etiology,
            "text": original_transcript,
            "audio_path": audio_path,
            'speech_type': row.get('speech_type', ''),
        }
    
    def _get_empty_sample(self):
        """Return empty tensors for invalid samples during evaluation."""
        # Create dummy mel spectrogram features (80 mel bins, 3000 time steps)
        input_features = torch.zeros((80, 3000))
        
        # Create dummy labels
        labels = torch.tensor([self.processor.tokenizer.eos_token_id])
        
        return {
            "input_features": input_features,
            "labels": labels,
        }
    
    def get_sample_info(self, idx):
        """Get metadata information for a sample."""
        if idx >= len(self.df):
            return None
        
        row = self.df.iloc[idx]
        return {
            'audio_path': row['audio_path'],
            'transcript': row['transcript'],
            'prompt_text': row.get('prompt_text', ''),
            'category': row.get('category', ''),
            'contributor_id': row.get('contributor_id', ''),
            'etiology': row.get('etiology', ''),
        }
    
    def get_statistics(self):
        """Get dataset statistics."""
        if len(self.df) == 0:
            return {}
        
        stats = {
            'total_samples': len(self.df),
            'unique_contributors': self.df['contributor_id'].nunique(),
            'unique_categories': self.df['category'].nunique(),
        }
        
        if 'etiology' in self.df.columns:
            stats['etiologies'] = sorted(self.df['etiology'].dropna().unique().tolist())
        
        if 'category' in self.df.columns:
            stats['categories'] = sorted(self.df['category'].dropna().unique().tolist())
        
        # Transcript length statistics
        transcript_lengths = self.df['transcript'].str.len()
        stats['transcript_length'] = {
            'mean': transcript_lengths.mean(),
            'median': transcript_lengths.median(),
            'min': transcript_lengths.min(),
            'max': transcript_lengths.max(),
        }
        
        return stats


# Helper function to verify dataset loading
def test_dataset_loading(config_path):
    """Test function to verify dataset loading works correctly."""
    from yaml_config_manager import load_config
    from transformers import WhisperProcessor
    
    config = load_config(config_path)
    
    # Load processor
    processor = WhisperProcessor.from_pretrained(
        config.model.name_or_path,
        language=config.model.language,
        task=config.model.task
    )
    
    # Create test dataset (small sample)
    test_dataset = WhisperDysarthricSpeechDataset(
        processor=processor,
        tsv_path=config.data.test_tsv_path,
        data_root_path=config.data.data_root_path,
        split='test',
        max_size=5,  # Just test 5 samples
        normalize_transcripts=False,
    )
    
    print(f"Created test dataset with {len(test_dataset)} samples")
    
    # Test loading a few samples
    for i in range(min(3, len(test_dataset))):
        sample = test_dataset[i]
        print(f"\nSample {i}:")
        print(f"  Input features shape: {sample['input_features'].shape}")
        print(f"  Labels shape: {sample['labels'].shape}")
        print(f"  Labels: {sample['labels']}")
        
        # Decode the labels to see the transcript
        transcript = processor.tokenizer.decode(sample['labels'], skip_special_tokens=True)
        print(f"  Decoded transcript: '{transcript}'")
        
        # Get sample info
        info = test_dataset.get_sample_info(i)
        print(f"  Original transcript: '{info['transcript']}'")
        print(f"  Audio path: {info['audio_path']}")


if __name__ == "__main__":
    # Test the dataset if run directly
    import sys
    if len(sys.argv) > 1:
        test_dataset_loading(sys.argv[1])
    else:
        print("Usage: python whisper_dysarthric_dataset.py <config_path>")