---
license: mit
language:
- en
base_model: moonshotai/Kimi-Audio-7B-Instruct
library_name: kimi-audio
pipeline_tag: automatic-speech-recognition
tags:
- dysarthric-speech
- atypical-speech
- speech-recognition
- speech-language-model
- kimi-audio
- interspeech-2026
metrics:
- wer
model-index:
- name: {repo_name}
  results:
  - task:
      type: automatic-speech-recognition
    dataset:
      name: Speech Accessibility Project (v31-05-2025, dev)
      type: speech-accessibility-project
    metrics:
    - type: wer
      value: {sap_wer}
---

# {repo_name}

{summary}

Paper: **Etiology-Aware Speech Language Models for Dysarthric Speech Recognition** (Interspeech 2026)
[Paper](https://www.isca-archive.org/interspeech_2026/laquatra26_interspeech.html) | [Code](https://github.com/MorenoLaQuatra/dysarthric-asr)

## The four models

All four are [Kimi-Audio-7B-Instruct](https://huggingface.co/moonshotai/Kimi-Audio-7B-Instruct) fine-tuned on the same data. They differ only in how the speaker's condition (etiology) is used.

| Model | Strategy | SAP WER | SAP SemScore | TORGO WER |
|---|---|---|---|---|
{variants_table}

WER in %, lower is better. SemScore higher is better. TORGO is zero-shot (no TORGO training).

## Input and output

{format_description}

## Usage

```bash
git clone --recursive https://github.com/MorenoLaQuatra/dysarthric-asr
cd dysarthric-asr
pip install -r requirements/kimi.txt -r kimi/third_party/Kimi-Audio/requirements.txt
bash kimi/scripts/setup.sh
source kimi/scripts/env.sh
```

```python
from kimia_infer.api.kimia import KimiAudio

model = KimiAudio(model_path="{repo_id}", load_detokenizer=False)
messages = [
    {{"role": "user", "message_type": "text", "content": "{prompt}"}},
    {{"role": "user", "message_type": "audio", "content": "speech.wav"}},
]
_, text = model.generate(messages, output_type="text",
                         text_temperature=0.0, text_top_k=5,
                         audio_temperature=0.8, audio_top_k=10)
print(text)
```
{usage_note}
For batch inference and the paper evaluation (SAP and TORGO), see the [code repository](https://github.com/MorenoLaQuatra/dysarthric-asr).

## Results per condition

{per_condition_table}

## Training

- Data: Speech Accessibility Project (SAP), release v31-05-2025. 240,047 utterances (547 h) from speakers with ALS, Parkinson's disease, stroke, Down syndrome and cerebral palsy.
- Method: LoRA (rank 16, alpha 32) on the language model and on the Whisper encoder attention. 3 epochs, learning rate 5e-5, global batch {global_batch}. Checkpoint chosen by validation loss.
- This repository holds the merged weights (fp32) exactly as evaluated in the paper.

## Limitations

- For research on English dysarthric speech recognition.
- The predicted etiology is a helper for transcription, **not a diagnosis**. Do not use it for clinical decisions.
- Trained on five conditions. Other conditions, languages or recording setups are not tested beyond TORGO.
- No audio or transcripts from SAP are included in this repository.

## Citation

If you use this model, please cite the [paper](https://www.isca-archive.org/interspeech_2026/laquatra26_interspeech.html):

```bibtex
@inproceedings{{laquatra26_interspeech,
  title     = {{{{Etiology-Aware Speech Language Models for Dysarthric Speech Recognition}}}},
  author    = {{Moreno {{La Quatra}} and Alkis Koudounas and Valerio Mario Salerno and Sabato Marco Siniscalchi}},
  year      = {{2026}},
  booktitle = {{{{Interspeech 2026}}}},
  pages     = {{5033--5037}},
  doi       = {{10.21437/Interspeech.2026-1773}},
  issn      = {{2958-1796}},
}}
```

## Acknowledgements

Trained on data from the [Speech Accessibility Project](https://speechaccessibilityproject.beckman.illinois.edu/) (Hasegawa-Johnson et al., 2024). Base model: [Kimi-Audio](https://github.com/MoonshotAI/Kimi-Audio) (Ding et al., 2025). License: MIT, as the base model.
