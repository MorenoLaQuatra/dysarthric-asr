"""
WER and SemScore for dysarthric ASR, following the SAP Challenge protocol.

Text normalization, dual-reference WER (with / without parenthesized content) and
SemScore (NLI + BERTScore + phonetic similarity, weights from Phukon et al., 2025)
reproduce the official SAP Challenge scoring.
Adapted from the SAP Challenge evaluation code: https://github.com/xiuwenz2/SAPC-template
"""

import re
import numpy as np
import torch
import jellyfish
import editdistance
from typing import List, Tuple, Dict, Optional, Union
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from bert_score import score as bert_score


class DysarthricASREvaluator:
    """
    All-in-one evaluator for dysarthric speech ASR systems.
    Computes Word Error Rate (WER) and Semantic Score (SemScore).
    """
    
    # Word list for abbreviation separation
    ABBREVIATIONS = [
        'MSNNBC', 'MSNBC', 'MSSNNBC', 'AARP', 'ACDC', 'ADHD', 'BBBC', 'ESPN', 'HGTV', 
        'LBGQ', 'NCIS', 'NPPA', 'PTSD', 'RAMC', 'SPCA', 'TLSC', 'WDRB', 'WDTN', 'WHIO', 
        'WHYY', 'WKRP', 'WNBA', 'WNYC', 'YMCA', 'AAA', 'ABC', 'ACP', 'ADA', 'AHS', 'AJR', 
        'AKA', 'ALS', 'AMC', 'AOL', 'AXL', 'BAU', 'BBC', 'BLT', 'BMW', 'BRB', 'BST', 'BTS', 
        'CBS', 'CCM', 'CEO', 'CFO', 'CIA', 'CNC', 'CNN', 'CPR', 'CSI', 'CVS', 'DCI', 'DDA', 
        'DNA', 'DSW', 'DVD', 'DVR', 'FAC', 'FBI', 'FDR', 'GPA', 'HBO', 'ICS', 'IRS', 'JFK', 
        'LAL', 'LSU', 'MGK', 'MIT', 'MMA', 'MSN', 'NBA', 'NBC', 'NCI', 'NFL', 'NHK', 'NHL', 
        'NPR', 'NRA', 'OAN', 'OCD', 'PBC', 'PBS', 'PDU', 'PGA', 'PLS', 'PSP', 'RBG', 'REM', 
        'REO', 'RSD', 'SNL', 'TBS', 'TNT', 'TSA', 'UPS', 'USA', 'USC', 'VSP', 'WRB', 'AC', 
        'AD', 'AI', 'AM', 'AV', 'BB', 'BJ', 'CD', 'CJ', 'CO', 'CV', 'DC', 'DJ', 'DV', 'ER', 
        'ES', 'FC', 'FO', 'FX', 'GI', 'GP', 'GR', 'GT', 'ID', 'IV', 'JF', 'JJ', 'KC', 'LA', 
        'MP', 'NP', 'OJ', 'OK', 'PA', 'PC', 'PD', 'PJ', 'PM', 'PT', 'QR', 'RC', 'RH', 'RV', 
        'TV', 'UK', 'US', 'WH', 'WO', 'XM', 'PPM', 'TX', 'NYC', 'TTV', 'II', 'AAM', 'IL', 
        'NI', 'SG', 'PB', 'NSYNC', 'YK', 'AJ', 'PBJ'
    ]
    
    def __init__(
        self,
        device: Optional[str] = None,
        nli_weight: float = 0.4012,
        bert_weight: float = 0.2785,
        phonetic_weight: float = 0.3201,
        model_cache_dir: Optional[str] = None,
        use_nemo_normalizer: bool = False,
        compute_wer: bool = True,
        compute_semscore: bool = True
    ):
        """
        Initialize the evaluator.
        
        Args:
            device: Device to use ('cuda' or 'cpu'). If None, auto-detect.
            nli_weight: Weight for NLI score in SemScore calculation
            bert_weight: Weight for BERT score in SemScore calculation
            phonetic_weight: Weight for phonetic similarity in SemScore calculation
            model_cache_dir: Directory to cache downloaded models
            use_nemo_normalizer: Whether to use NEMO text normalizer (requires separate installation)
        """
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Score weights
        self.nli_weight = float(nli_weight)
        self.bert_weight = float(bert_weight)
        self.phonetic_weight = float(phonetic_weight)
        
        self.compute_wer = compute_wer
        self.compute_semscore = compute_semscore
        if not (self.compute_wer or self.compute_semscore):
            raise ValueError("At least one of compute_wer or compute_semscore must be True")

        # The NLI model is only needed for SemScore
        if self.compute_semscore:
            print("Loading NLI model...")
            self.nli_model, self.nli_tokenizer = self._load_nli_model(model_cache_dir)

        # Punctuation processing pattern
        self.valid_chars = '''
            \u0041-\u005a\u0027\u0020
            \u00c0\u00c1\u00c4\u00c5\u00c8\u00c9\u00cd\u00cf
            \u00d1\u00d3\u00d6\u00d8\u00db\u00dc\u0106
            '''
        
        # Initialize NEMO normalizer if requested
        self.normalizer = None
        if use_nemo_normalizer:
            try:
                from nemo_text_processing.text_normalization.normalize import Normalizer
                self.normalizer = Normalizer(input_case='cased', lang='en')
                print("NEMO normalizer loaded successfully")
            except ImportError:
                print("Warning: NEMO text processing not installed. Install with:")
                print("pip install nemo_text_processing")
                print("Continuing without NEMO normalization...")
    
    def evaluate(
        self,
        hypotheses: List[str],
        references: List[str],
        dual_reference_mode: str = 'auto',
        process_text: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate ASR hypotheses against references.
        
        Args:
            hypotheses: List of ASR hypothesis strings
            references: List of reference transcriptions
            dual_reference_mode: How to generate two references:
                - 'auto': Generate two versions (with and without parentheses)
                - 'none': Use same reference twice (single reference mode)
                - 'manual': Assumes references already contains pairs [ref1_1, ref2_1, ref1_2, ref2_2, ...]
            process_text: Whether to apply text processing (uppercase, punctuation removal)
        
        Returns:
            Dictionary with 'wer' and 'semscore' values (as percentages)
        """
        # Handle different reference modes
        if dual_reference_mode == 'manual':
            # Expects references to alternate between ref1 and ref2
            assert len(references) == 2 * len(hypotheses), \
                "In manual mode, references should contain twice as many entries as hypotheses"
            references1 = references[0::2]
            references2 = references[1::2]
        else:
            # Generate two references from single reference list
            references1, references2 = self._generate_dual_references(
                references, 
                mode=dual_reference_mode,
                process_text=process_text
            )
            
        # Process hypotheses if requested
        if process_text:
            hypotheses = [self._normalize_text(h, remove_parentheses=True) for h in hypotheses]
            
        # Validate inputs
        assert len(hypotheses) == len(references1) == len(references2), \
            "All input lists must have the same length"
        
        # Calculate WER
        if self.compute_wer:
            try:
                wer = self._calculate_wer(references1, references2, hypotheses)
            except Exception as e:
                print(f"Error calculating WER: {e} for #samples: {len(hypotheses)}")
                wer = 1.0  # Default to 100% error if calculation fails
        else:
            wer = 0.0
        
        # Calculate SemScore
        if self.compute_semscore:
            semscore = self._calculate_semscore(references1, references2, hypotheses)
        else:
            semscore = 0.0
        
        return {
            'wer': round(wer * 100, 4),  # Convert to percentage
            'semscore': round(semscore * 100, 4)  # Convert to percentage
        }

    def utterance_scores(self, hypotheses: List[str], references: List[str]) -> Dict[str, np.ndarray]:
        """
        Per-utterance contributions to the corpus-level metrics, using the same
        normalization and dual-reference rules as `evaluate`:
          corpus WER      = errors.sum() / lengths.sum()
          corpus SemScore = semscore.mean()
        so any subset of utterances can be scored without recomputing the models.
        """
        references1, references2 = self._generate_dual_references(references, mode='auto', process_text=True)
        hypotheses = [self._normalize_text(h, remove_parentheses=True) for h in hypotheses]

        errors, lengths = [], []
        for ref1, ref2, hyp in zip(references1, references2, hypotheses):
            r1, r2, h = ref1.strip().split(), ref2.strip().split(), hyp.strip().split()
            distance = [min(editdistance.eval(h, r1), len(r1)), min(editdistance.eval(h, r2), len(r2))]
            length = [len(r1), len(r2)]
            wer = [d / l if l > 0 else 0 for d, l in zip(distance, length)]
            if wer[0] == wer[1]:
                errors.append(np.mean(distance))
                lengths.append(np.mean(length))
            elif wer[0] < wer[1]:
                errors.append(distance[0])
                lengths.append(length[0])
            else:
                errors.append(distance[1])
                lengths.append(length[1])

        out = {'errors': np.array(errors, dtype=float), 'lengths': np.array(lengths, dtype=float)}
        if self.compute_semscore:
            s1 = self._score_all(references1, hypotheses)
            s2 = self._score_all(references2, hypotheses)
            out['semscore'] = np.array([max(a, b) for a, b in zip(s1, s2)], dtype=float)
        return out
    
    def _generate_dual_references(
        self, 
        references: List[str], 
        mode: str = 'auto',
        process_text: bool = True
    ) -> Tuple[List[str], List[str]]:
        """
        Generate two reference versions from single reference list.
        
        Args:
            references: Original reference transcripts
            mode: 'auto' to generate with/without parentheses, 'none' for identical refs
            process_text: Whether to apply normalization
            
        Returns:
            Tuple of (references1, references2)
        """
        if mode == 'none':
            # Single reference mode - process identically
            if process_text:
                refs = [self._normalize_text(r, remove_parentheses=True) for r in references]
                return refs, refs
            else:
                return references, references
                
        elif mode == 'auto':
            # Generate two versions: with and without parentheses content
            references1 = []  # Without parentheses
            references2 = []  # With parentheses
            
            for ref in references:
                if process_text:
                    ref1 = self._normalize_text(ref, remove_parentheses=True)
                    ref2 = self._normalize_text(ref, remove_parentheses=False)
                else:
                    ref1 = self._process_parentheses(ref, remove=True)
                    ref2 = self._process_parentheses(ref, remove=False)
                    
                references1.append(ref1)
                references2.append(ref2)
                
            return references1, references2
        else:
            raise ValueError(f"Unknown dual_reference_mode: {mode}")
    
    def _normalize_text(self, text: str, remove_parentheses: bool = True) -> str:
        """
        Full normalization pipeline matching the original preprocessing script.
        
        Args:
            text: Input text
            remove_parentheses: Whether to remove parentheses content
            
        Returns:
            Normalized text
        """
        # Separate abbreviations
        words = text.strip().split()
        new_words = []
        for word in words:
            # Check if word (without punctuation) is in abbreviation list
            clean_word = re.sub(u"([^"+self.valid_chars+"])", "", word.upper())
            if clean_word in self.ABBREVIATIONS:
                new_words.append(" ".join(word))
            else:
                new_words.append(word)
        text = ' '.join(new_words)
        
        # Change curly quotes to straight quotes
        text = re.sub(r"[\'\']", r"'", text)
        
        # Remove [...] content
        text = re.sub(r"\[(.*?)\]", " ", text)
        
        # Process {...} content (uncertain words)
        def process_braces(match):
            content = match.group(1)
            if ':' in content:
                prefix = content.split(':')[0]
                if prefix == 'w':
                    # Extract number of unknown words
                    num_match = re.findall(r'\d+', content)
                    if num_match:
                        num_unk = int(num_match[0])
                        return " ".join(["UNK" for _ in range(num_unk)])
                    return " "
                elif prefix == 'u' or content == ' ':
                    return "UNK"
                else:
                    return re.sub(r"(.+(?=:))", " ", content)
            return content
            
        text = re.sub(r"\{(.*?)\}", process_braces, text)
        
        # Remove * and ~ before normalization
        text = re.sub(r"[\*\~]", " ", text)
        
        # NEMO normalization if available
        if self.normalizer:
            text = ' '.join(text.strip().split())  # Remove extra spaces
            try:
                text = self.normalizer.normalize(text, verbose=False, punct_post_process=True)
            except:
                pass  # Continue without NEMO if it fails
        
        # Process email addresses
        if '@' in text:
            words = []
            for word in text.split():
                if '@' in word:
                    word = re.sub(r"\.", " dot ", word[:-1]) + word[-1] if word.endswith('.') else re.sub(r"\.", " dot ", word)
                    word = re.sub("@", " at ", word)
                words.append(word)
            text = " ".join(words)
        
        # Process parentheses
        text = self._process_parentheses(text, remove=remove_parentheses)
        
        # Final processing
        text = text.upper()
        text = re.sub(u"([^"+self.valid_chars+"])", " ", text)
        
        # Remove extra apostrophes
        text = " ".join([word.strip("'") for word in text.split()])
        
        # Remove extra spaces
        text = ' '.join(text.strip().split())
        
        return text
    
    def _process_parentheses(self, text: str, remove: bool = True) -> str:
        """Process parentheses content."""
        if remove:
            # Remove parentheses content except code-switched content (cs:)
            def process_paren(match):
                content = match.group(1)
                if ':' in content and content.split(':')[0].strip() == 'cs':
                    return re.sub(r"(.+(?=:))", " ", content)
                return ""
            text = re.sub(r"\((.*?)\)", process_paren, text)
        else:
            # Keep parentheses but process code-switched content
            text = re.sub(r"\((.*?)\)", 
                         lambda x: "("+re.sub(r"(.+(?=:))", " ", x.group(1))+")", 
                         text)
        return text
    
    def _process_text(self, text: str) -> str:
        """Simple text processing (for when full normalization isn't needed)."""
        text = text.strip().upper()
        text = re.sub(u"([^"+self.valid_chars+"])", "", text)
        text = ' '.join(text.strip().split())
        return text
    
    def _calculate_wer(
        self,
        references1: List[str],
        references2: List[str],
        hypotheses: List[str]
    ) -> float:
        """Calculate Word Error Rate with dual references."""
        distances = 0
        lengths = 0
        
        for ref1, ref2, hyp in zip(references1, references2, hypotheses):
            r1 = ref1.strip().split()
            r2 = ref2.strip().split()
            h = hyp.strip().split()
            
            # Calculate distances and lengths for both references
            distance = [
                min(editdistance.eval(h, r1), len(r1)),
                min(editdistance.eval(h, r2), len(r2))
            ]
            length = [len(r1), len(r2)]
            
            # Calculate WER for each reference
            wer = [d/l if l > 0 else 0 for d, l in zip(distance, length)]
            
            # Choose the reference with lower WER
            if wer[0] == wer[1]:
                distances += np.mean(distance)
                lengths += np.mean(length)
            elif wer[0] < wer[1]:
                distances += distance[0]
                lengths += length[0]
            else:
                distances += distance[1]
                lengths += length[1]
        
        assert lengths != 0, "Total length cannot be zero"
        return distances / lengths
    
    def _calculate_semscore(
        self,
        references1: List[str],
        references2: List[str],
        hypotheses: List[str]
    ) -> float:
        """Calculate Semantic Score combining NLI, BERT, and phonetic similarity."""
        # Calculate scores for both references
        semscores = {
            'ref1': self._score_all(references1, hypotheses),
            'ref2': self._score_all(references2, hypotheses)
        }
        
        # Take the maximum score for each hypothesis
        semscore = [max(s1, s2) for s1, s2 in zip(semscores['ref1'], semscores['ref2'])]
        
        # Return average
        return sum(semscore) / len(semscore)
    
    def _score_all(self, refs: List[str], hyps: List[str]) -> List[float]:
        """Calculate combined scores for a set of references and hypotheses."""
        # Calculate BERT scores
        bert_scores = self._calculate_bert_scores(refs, hyps)
        bert_scores = self._min_max_normalize(bert_scores, [-0.1180, 1])
        
        # Calculate phonetic scores
        phonetic_scores = []
        for ref, hyp in zip(refs, hyps):
            phonetic_score = self._calculate_phonetic_similarity(ref, hyp)
            phonetic_scores.append(phonetic_score)
        phonetic_scores = self._min_max_normalize(phonetic_scores, [0.5, 1])
        
        # Calculate NLI scores
        nli_scores = self._score_nli(refs, hyps)
        nli_scores = self._min_max_normalize(
            nli_scores,
            [0.0028752246871590614, 0.9661698341369629]
        )
        
        # Combine scores with weights
        combined_scores = [
            self.nli_weight * nli + self.bert_weight * bert + self.phonetic_weight * phon
            for nli, bert, phon in zip(nli_scores, bert_scores, phonetic_scores)
        ]
        
        return combined_scores
    
    def _load_nli_model(self, cache_dir: str):
        """Load the NLI model and tokenizer."""
        model_name = 'ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli'
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,
            cache_dir=cache_dir
        )
        
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3,
            cache_dir=cache_dir
        )
        
        model.eval()
        model = model.to(self.device)
        
        return model, tokenizer
    
    def _score_nli(
        self,
        refs: List[str],
        hyps: List[str],
        direction: str = 'avg'
    ) -> List[float]:
        """Calculate NLI scores."""
        probs_rh, probs_hr = {}, {}
        
        with torch.no_grad():
            # Reference -> Hypothesis direction
            if direction in ['rh', 'avg']:
                probs = []
                for ref, hyp in zip(refs, hyps):
                    input_ids, token_type_ids, attention_mask = \
                        self._prepare_nli_input(ref, hyp)
                    
                    logits = self.nli_model(
                        input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        labels=None
                    )[0]
                    
                    prob = torch.softmax(logits, 1).detach().cpu().numpy()
                    probs.append(prob)
                    
                concatenated = np.concatenate(probs, 0)
                probs_rh['e'] = concatenated[:, 0]  # Entailment
            
            # Hypothesis -> Reference direction
            if direction in ['hr', 'avg']:
                probs = []
                for ref, hyp in zip(refs, hyps):
                    input_ids, token_type_ids, attention_mask = \
                        self._prepare_nli_input(hyp, ref)
                    
                    logits = self.nli_model(
                        input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        labels=None
                    )[0]
                    
                    prob = torch.softmax(logits, 1).detach().cpu().numpy()
                    probs.append(prob)
                    
                concatenated = np.concatenate(probs, 0)
                probs_hr['e'] = concatenated[:, 0]  # Entailment
        
        # Return scores based on direction
        if direction == 'rh':
            return list(probs_rh['e'])
        elif direction == 'hr':
            return list(probs_hr['e'])
        else:  # avg
            return [(s1 + s2) / 2.0 for s1, s2 in zip(probs_rh['e'], probs_hr['e'])]
    
    def _prepare_nli_input(
        self,
        premise: str,
        hypothesis: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare input tensors for NLI model."""
        tokenized = self.nli_tokenizer.encode_plus(
            premise,
            hypothesis,
            max_length=self.nli_tokenizer.model_max_length,
            return_token_type_ids=True,
            truncation=True
        )
        
        input_ids = torch.Tensor(tokenized['input_ids']).long().unsqueeze(0).to(self.device)
        token_type_ids = torch.Tensor(tokenized['token_type_ids']).long().unsqueeze(0).to(self.device)
        attention_mask = torch.Tensor(tokenized['attention_mask']).long().unsqueeze(0).to(self.device)
        
        return input_ids, token_type_ids, attention_mask
    
    def _calculate_bert_scores(self, refs: List[str], hyps: List[str]) -> np.ndarray:
        """Calculate BERT scores."""
        _, _, F1 = bert_score(
            refs,
            hyps,
            lang="en",
            rescale_with_baseline=True,
            device=self.device
        )
        return F1.numpy()
    
    def _calculate_phonetic_similarity(self, reference: str, hypothesis: str) -> float:
        """Calculate phonetic similarity using Soundex and Jaro-Winkler."""
        hypo_soundex = jellyfish.soundex(hypothesis)
        ref_soundex = jellyfish.soundex(reference)
        return jellyfish.jaro_winkler_similarity(hypo_soundex, ref_soundex)
    
    def _min_max_normalize(
        self,
        scores: List[float],
        thresholds: List[float]
    ) -> List[float]:
        """Min-max normalization with thresholds."""
        assert len(scores) != 0, "Scores list cannot be empty"
        
        normalized_scores = [
            max(min((score - thresholds[0]) / (thresholds[1] - thresholds[0]), 1), 0)
            for score in scores
        ]
        
        return normalized_scores
