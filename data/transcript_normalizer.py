"""
Transcript normalization following the SAP Challenge annotation conventions.
Adapted from https://github.com/xiuwenz2/SAPC-template

Defaults reproduce the manifests used for Kimi-Audio and Parakeet. The SLM baselines
(Phi-4, Qwen2-Audio, Granite) were trained with other_action="remove", which their
dataset classes set explicitly.
"""

import re


class TranscriptNormalizer:
    """
    Handles normalization of transcripts following SAPC conventions.
    
    Annotation types:
    - [...] : Prompts given to user
    - (...) : Disfluencies/hesitations  
    - {g: word} : Word pronounced but annotator uncertain
    - {w: ...} : Numbers/content not pronounced (not comprehensible)
    - {ss: ...} or (ss: ...) : Pronounced by person with lower/distant voice
    - {cs: ...} or (cs: ...): Pronounced by interviewer/medical expert
    - ~letter : Spelled letters
    """
    
    def __init__(self,
            # Prompts handling
            remove_prompts=True,
            
            # Disfluencies in parentheses
            disfluency_action="keep",  # "remove", "keep", "tag"
            start_disfluency_tag="<|disfluency|>",  # Tag for start of disfluency
            end_disfluency_tag="<|end_disfluency|>",  # Tag for end of disfluency
            
            # Annotation handling
            g_action="keep",      # {g: word} - uncertain but pronounced ("keep" -> word, "remove" -> remove, "tag" -> tag with disfluency_tag)
            w_action="remove",    # {w: ...} - not pronounced/comprehensible ("remove" -> remove, "keep" -> word, "tag" -> tag with w_tag)
            ss_action="keep",     # {ss: ...} or (ss: ...) - low voice but same person ("keep" -> word, "remove" -> remove, "tag" -> tag with disfluency_tag)
            cs_action="keep",   # {cs: ...} or (cs: ...) - interviewer speech ("remove" -> remove, "keep" -> word, "tag" -> tag with cs_tag)
            other_action="keep",  # Other {...} annotations: "keep" as is or "remove"
            
            # Tags for when action="tag"
            w_tag="<|unclear|>",
            start_cs_tag="<|external_speech|>",
            end_cs_tag="<|end_external_speech|>",
            
            # Spelled letters
            process_spelled=True, # Whether to process spelled letters (~letter -> letter with space before)
            
            # Capialization
            capitalization="lower"  # "lower", "upper", "keep" (default is "lower")
        ):
        
        self.remove_prompts = remove_prompts
        self.disfluency_action = disfluency_action
        self.start_disfluency_tag = start_disfluency_tag
        self.end_disfluency_tag = end_disfluency_tag
        self.g_action = g_action
        self.w_action = w_action
        self.ss_action = ss_action
        self.cs_action = cs_action
        self.other_action = other_action
        self.w_tag = w_tag
        self.start_cs_tag = start_cs_tag
        self.end_cs_tag = end_cs_tag
        self.process_spelled = process_spelled
        self.capitalization = capitalization
    
    def normalize(self, transcript: str) -> str:
        """Normalize transcript according to configuration."""
        text = transcript.strip()
        
        # 1. Handle prompts in square brackets [...]
        if self.remove_prompts:
            text = re.sub(r'\[.*?\]', '', text)
        
        # 2. Handle specific annotation types {type: content}
        text = self._handle_annotations(text)
        
        # 3. Handle disfluencies in parentheses (...)
        text = self._handle_disfluencies(text)
        
        # 4. Handle spelled letters ~letter
        if self.process_spelled:
            text = re.sub(r'~([A-Za-z])', r' \1', text)
        
        # 5. Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 6. To conclude, lowercase/capitalize as needed
        if self.capitalization == "lower":
            text = text.lower()
        elif self.capitalization == "upper":
            text = text.upper()
        # "keep" does nothing
        
        return text
    
    def _handle_annotations(self, text: str) -> str:
        """Handle {type: content} or (type: content) annotations."""
        
        # {g: word} - uncertain but pronounced -> keep word
        if self.g_action == "keep":
            text = re.sub(r'\{g:\s*([^}]*)\}', r'\1', text)
        elif self.g_action == "remove":
            text = re.sub(r'\{g:\s*[^}]*\}', '', text)
        elif self.g_action == "tag":
            text = re.sub(r'\{g:\s*([^}]*)\}', f'{self.start_disfluency_tag}\\1{self.end_disfluency_tag}', text)
        
        # {ss: word} or (ss: word) - low voice but same person -> keep word  
        if self.ss_action == "keep":
            text = re.sub(r'[\{\(]ss:\s*([^}\)]*?)[\}\)]', r'\1', text)
        elif self.ss_action == "remove":
            text = re.sub(r'[\{\(]ss:\s*[^}\)]*?[\}\)]', '', text)
        elif self.ss_action == "tag":
            text = re.sub(r'[\{\(]ss:\s*([^}\)]*?)[\}\)]', f'{self.start_disfluency_tag}\\1{self.end_disfluency_tag}', text)
        
        # {w: ...} - not pronounced/comprehensible
        if self.w_action == "remove":
            text = re.sub(r'\{w:\s*[^}]*\}', '', text)
        elif self.w_action == "keep":
            text = re.sub(r'\{w:\s*([^}]*)\}', r'\1', text)
        elif self.w_action == "tag":
            text = re.sub(r'\{w:\s*([^}]*)\}', f'{self.w_tag}', text) # only tag, no content
        elif self.w_action == "unk":
            # Special case for w, {w: N} are N unknown words
            unk_token = "UNK"
            n_unknown = re.findall(r'\{w:\s*([0-9]+)\}', text)
            if n_unknown:
                n_unknown = int(n_unknown[0])
                text = re.sub(r'\{w:\s*[0-9]+\}', ' '.join([unk_token] * n_unknown), text)
            else: # fallback to removing - it should not happen
                text = re.sub(r'\{w:\s*[^}]*\}', '', text)
        
        # {cs: ...} or (cs: ...) - interviewer speech
        if self.cs_action == "remove":
            text = re.sub(r'[\{\(]cs:\s*[^}\)]*?[\}\)]', '', text)
        elif self.cs_action == "keep":
            text = re.sub(r'[\{\(]cs:\s*([^}\)]*?)[\}\)]', r'\1', text)
        elif self.cs_action == "tag":
            text = re.sub(r'[\{\(]cs:\s*([^}\)]*?)[\}\)]', f'{self.start_cs_tag}\\1{self.end_cs_tag}', text)
            
        # other, means all the {...} annotations not specified for example {u:...}
        if self.other_action == "remove":
            text = re.sub(r'\{[^}]*\}', '', text)
        elif self.other_action == "keep":
            # Keep as is, no changes
            pass
        
        return text
    
    def _handle_disfluencies(self, text: str) -> str:
        """Handle disfluencies in parentheses (...)."""
        if self.disfluency_action == "remove":
            text = re.sub(r'\([^)]*\)', '', text)
        elif self.disfluency_action == "tag":
            text = re.sub(r'\(([^)]*)\)', f'{self.start_disfluency_tag}\\1{self.end_disfluency_tag}', text)
        # "keep" does nothing
        
        return text
