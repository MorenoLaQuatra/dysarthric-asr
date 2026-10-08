"""
Etiology Classification (EC) variant of Kimi-Audio.

An MLP head on the mean-pooled Whisper encoder features predicts the speaker's
etiology. The decoder is trained on the plain transcript only, so the etiology
prediction never enters the generation stream.
"""

import re
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from finetune_codes.datasets import LazySupervisedDataset
from finetune_codes.model import KimiAudioModel
from finetune_codes.modeling_kimia import MoonshotKimiaForCausalLM

ETIOLOGY_TO_IDX = {
    "Parkinson's Disease": 0,
    "ALS": 1,
    "Cerebral Palsy": 2,
    "Down Syndrome": 3,
    "Stroke": 4,
}
IDX_TO_ETIOLOGY = {v: k for k, v in ETIOLOGY_TO_IDX.items()}
NUM_ETIOLOGY_CLASSES = len(ETIOLOGY_TO_IDX)
CLASSIFIER_FILENAME = "etiology_classifier.pt"


def build_etiology_classifier(num_classes: int = NUM_ETIOLOGY_CLASSES) -> nn.Module:
    # Whisper output (1280-d) is stacked 4x by Kimi-Audio -> 5120-d frames
    return nn.Sequential(
        nn.Linear(5120, 1024),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(1024, num_classes),
    )


class KimiAudioModelEC(KimiAudioModel):
    def __init__(self, config, num_etiology_classes: int = NUM_ETIOLOGY_CLASSES):
        super().__init__(config)
        self.num_etiology_classes = num_etiology_classes
        self.etiology_classifier = build_etiology_classifier(num_etiology_classes)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        """Required by PEFT when using TaskType.CAUSAL_LM."""
        model_inputs = {"input_ids": input_ids}
        if past_key_values is not None:
            model_inputs["past_key_values"] = past_key_values
        model_inputs.update(kwargs)
        return model_inputs

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        text_input_ids: torch.LongTensor = None,
        whisper_input_feature: Optional[torch.FloatTensor] = None,
        is_continuous_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        generation_mode: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        whisper_input_feats = torch.from_numpy(whisper_input_feature[0]).unsqueeze(0)[:, :].to(torch.cuda.current_device())
        whisper_feats = self.whisper_model(whisper_input_feats)
        whisper_feats = whisper_feats.reshape(
            whisper_feats.shape[0],
            int(whisper_feats.shape[1] // 4),
            whisper_feats.shape[2] * 4,
        )

        # Etiology classification from temporally mean-pooled encoder features
        etiology_logits = self.etiology_classifier(whisper_feats.mean(dim=1))

        # Skip KimiAudioModel.forward to avoid recomputing the Whisper features
        generation_outputs = MoonshotKimiaForCausalLM.forward(
            self,
            input_ids=input_ids,
            text_input_ids=text_input_ids,
            whisper_input_feature=whisper_feats,
            is_continuous_mask=is_continuous_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            generation_mode=generation_mode,
            return_dict=return_dict,
        )
        return generation_outputs, etiology_logits


class LazySupervisedDatasetEC(LazySupervisedDataset):
    """
    Reads the EP manifest ("Etiology: X\\nTranscript: Y" targets), strips the
    etiology prefix from the decoder target and returns the etiology as a class label.
    """

    def __init__(self, raw_data_list, whisper_model, text_tokenizer, max_len: int, kimia_token_offset: int):
        super().__init__(
            raw_data_list=raw_data_list,
            whisper_model=whisper_model,
            text_tokenizer=text_tokenizer,
            max_len=max_len,
            kimia_token_offset=kimia_token_offset,
        )

    @staticmethod
    def strip_etiology(content: str) -> str:
        match = re.search(r"Transcript:\s*(.*)", content, re.DOTALL)
        return match.group(1).strip() if match else content.strip()

    def _etiology_label(self, entry) -> int:
        etiology = entry.get("etiology")
        if etiology in ETIOLOGY_TO_IDX:
            return ETIOLOGY_TO_IDX[etiology]
        for msg in entry["conversation"]:
            if msg.get("role") == "assistant" and msg.get("message_type") == "text":
                match = re.search(r"Etiology:\s*(.+?)(?:\n|$)", msg["content"])
                if match:
                    return ETIOLOGY_TO_IDX.get(match.group(1).strip(), 0)
        return 0

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        entry = self.raw_data[i]
        conversation = []
        for msg in entry["conversation"]:
            if msg.get("role") == "assistant" and msg.get("message_type") == "text":
                msg = dict(msg, content=self.strip_etiology(msg["content"]))
            conversation.append(msg)

        output_type = "text" if entry["task_type"] == "understanding" else "both"
        tokenized = self.tokenize_conversation(conversation, output_type=output_type, add_assistant_start_msg=False)
        audio_input_ids, text_input_ids, is_continuous_mask, audio_token_loss_mask, text_token_loss_mask = tokenized.to_tensor()

        def shift(x, fill):
            return torch.cat((x[:, 1:], x.new_full((1, 1), fill)), dim=1)

        return dict(
            input_ids=audio_input_ids,
            text_input_ids=text_input_ids,
            whisper_input_feature=tokenized.continuous_feature,
            is_continuous_mask=is_continuous_mask,
            labels=(
                shift(audio_input_ids, self.pad_token),
                shift(text_input_ids, self.pad_token),
                shift(audio_token_loss_mask, False),
                shift(text_token_loss_mask, False),
            ),
            etiology_label=torch.tensor(self._etiology_label(entry), dtype=torch.long),
        )
