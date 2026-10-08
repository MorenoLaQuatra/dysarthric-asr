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

class GraniteDysarthricDataset(TorchDataset):
    def __init__(
            self, 
            processor, 
            tsv_path, 
            data_root_path, 
            split, 
            max_size=None, 
            instruction="Please transcribe the following audio to text",
            use_etiology_in_instruction=False,
            etiology_token="{etiology}",  # Token to replace with etiology in prompt
            
            # Transcript normalization parameters
            normalize_transcripts=True,
            remove_prompts=True,
            disfluency_action="keep",  # {keep, remove, tag}
            start_disfluency_tag="<|disfluency|>",
            end_disfluency_tag="<|end_disfluency|>",  # Tags for disfluencies
            
            g_action="keep",      # {g: word} - uncertain but pronounced
            w_action="remove",    # {w: ...} - not pronounced
            ss_action="keep",     # {ss: ...} - low voice same person  
            cs_action="keep",   # {cs: ...} - interviewer
            
            w_tag="<unclear>",
            start_cs_tag="<|external_speech|>",
            end_cs_tag="<|end_external_speech|>",  # Tags for external speech
            
            process_spelled=True, # ~A~S~R (spelled letters to A S R with space before each letter)
            capitalization="lower",  # {lower, upper, keep} - how to handle capitalization
        ):
        """
        Dataset for Dysarthric Speech ASR adapted for Granite Speech.
        
        Args:
            processor: The Granite Speech processor
            tsv_path: Path to the TSV metadata file
            data_root_path: Root path to prepend to audio_path
            split: Split name (train/validation/test)
            max_size: Maximum number of samples to use
            instruction: Instruction text for the model
            normalize_transcripts: Whether to normalize transcripts
            **kwargs: Transcript normalization parameters
        """
        self.processor = processor
        self.data_root_path = Path(data_root_path)
        self.training = "train" in split
        self.instruction = instruction
        self.use_etiology_in_instruction = use_etiology_in_instruction
        self.etiology_token = etiology_token
        
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
        
        # Store original transcripts for evaluation
        self.original_transcripts = self.df['transcript'].tolist()
        
        # Filter valid samples
        self.df = self._filter_valid_samples()
        print(f"After filtering: {len(self.df)} valid samples")
        
        # Limit dataset size if specified
        if max_size and max_size < len(self.df):
            self.df = self.df.iloc[:max_size].reset_index(drop=True)
            self.original_transcripts = self.original_transcripts[:max_size]
            print(f"Limited dataset to {len(self.df)} samples")
        
        # Reset index for clean access
        self.df = self.df.reset_index(drop=True)
        
        print(f"Initialized {split} dataset with {len(self.df)} samples")
        
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
        Returns data compatible with Granite Speech model.
        """
        if idx >= len(self.df):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.df)}")
        
        row = self.df.iloc[idx]
        audio_path = row['audio_path']
        transcript = str(row['transcript']).strip()
        
        # Store original transcript for evaluation
        original_transcript = transcript
        
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
                # For evaluation, return empty sample
                return self._get_empty_sample()
        
        # Load audio file
        try:
            # Use processor's expected sampling rate
            sampling_rate = self.processor.audio_processor.sampling_rate
            audio_array, sr = librosa.load(audio_path, sr=sampling_rate)
            
            # If stereo, convert to mono
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
                # For evaluation, return empty sample
                return self._get_empty_sample()
        
        # Create instruction prompt (following Granite format)
        current_instruction = self.instruction
        current_etiology = row.get('etiology', None)
        
        # Replace etiology token if using etiology in instruction
        if self.use_etiology_in_instruction and current_etiology:
            current_instruction = current_instruction.replace(
                self.etiology_token, current_etiology
            )
        else:
            # Remove etiology token if not using it
            current_instruction = current_instruction.replace(self.etiology_token, "").strip()
        
        # Format prompt following Granite Speech format
        user_message = {
            'role': 'user',
            'content': current_instruction + '<|audio|>'
        }
        
        # Apply chat template
        prompt = self.processor.tokenizer.apply_chat_template(
            [user_message], 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Compute duration for statistics
        try:
            duration = librosa.get_duration(y=audio_array, sr=sampling_rate)
        except Exception as e:
            print(f"Error computing duration for {audio_path}: {e}")
            duration = 0.0

        return {
            'prompt': prompt,
            'audio': {'array': audio_array, 'sampling_rate': sampling_rate},
            'text': transcript.strip(),
            'original_text': original_transcript.strip(),
            'duration': duration,
            'etiology': current_etiology,
            'audio_path': str(audio_path),
            'input_ids': torch.tensor([1]),  # Placeholder for collator compatibility
            'speech_type': row.get('speech_type', ''),  # New field
        }
    
    def _get_empty_sample(self):
        """Return empty sample for invalid cases during evaluation."""
        return {
            'prompt': "",
            'audio': {'array': np.array([0.0]), 'sampling_rate': 16000},
            'text': "",
            'original_text': "",
            'duration': 0.0,
            'etiology': None,
            'audio_path': "",
            'input_ids': torch.tensor([0]),
        }
    
    def get_original_transcript(self, idx):
        """Get the original (unnormalized) transcript for evaluation."""
        if idx >= len(self.original_transcripts):
            return ""
        return self.original_transcripts[idx]
    
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