# Etiology-Aware Speech Language Models for Dysarthric Speech Recognition

Code and models for the Interspeech 2026 paper

> Moreno La Quatra, Alkis Koudounas, Valerio Mario Salerno, Sabato Marco Siniscalchi.
> **Etiology-Aware Speech Language Models for Dysarthric Speech Recognition.** Interspeech 2026, pp. 5033-5037.
> [[paper]](https://www.isca-archive.org/interspeech_2026/laquatra26_interspeech.html) | doi:10.21437/Interspeech.2026-1773

Standard ASR ignores the clinical context of speakers with neurological conditions. We compare four ways of
adding the speaker's etiology to a speech language model (Kimi-Audio-7B-Instruct) fine-tuned on the
Speech Accessibility Project (SAP) corpus:

| Strategy | Prompt | Target | Etiology needed at inference |
|---|---|---|---|
| **SFT**: standard fine-tuning | `Please transcribe the spoken content into written text.` | transcript | no |
| **EC**: etiology classification | same as SFT | transcript (+ auxiliary classifier on the audio encoder) | no |
| **EH**: etiology hinting | `... Consider that the speaker has been diagnosed with <etiology>.` | transcript | **yes** |
| **EP**: etiology prediction | same as SFT | `Etiology: <etiology>\nTranscript: <transcript>` | no |

Predicting the etiology *before* transcribing, inside the same autoregressive stream (EP), gives the best
results: **7.77% WER** on the SAP development set (6.4% relative reduction over SFT) and the best zero-shot WER on TORGO.

## Released models

All four Kimi-Audio variants are on the Hugging Face Hub (merged LoRA weights, fp32, exactly the weights evaluated in the paper):

| Model | Strategy | SAP WER | SAP SemScore | TORGO WER |
|---|---|---|---|---|
| [Kimi-Audio-7B-SAP-SFT](https://huggingface.co/morenolq/Kimi-Audio-7B-SAP-SFT) | SFT: standard fine-tuning | 8.30 | 90.48 | 18.0 |
| [Kimi-Audio-7B-SAP-EC](https://huggingface.co/morenolq/Kimi-Audio-7B-SAP-EC) | EC: etiology classification | 8.49 | 90.42 | 16.7 |
| [Kimi-Audio-7B-SAP-EH](https://huggingface.co/morenolq/Kimi-Audio-7B-SAP-EH) | EH: etiology hinting | 8.22 | 90.63 | 16.2 |
| [**Kimi-Audio-7B-SAP-EP**](https://huggingface.co/morenolq/Kimi-Audio-7B-SAP-EP) | **EP: etiology prediction (best)** | **7.77** | **91.05** | **15.8** |

Quick start with the best model (EP):

```bash
git clone --recursive https://github.com/MorenoLaQuatra/dysarthric-asr && cd dysarthric-asr
pip install -r requirements/kimi.txt -r kimi/third_party/Kimi-Audio/requirements.txt
bash kimi/scripts/setup.sh
source kimi/scripts/env.sh
```

```python
from kimia_infer.api.kimia import KimiAudio

model = KimiAudio(model_path="morenolq/Kimi-Audio-7B-SAP-EP", load_detokenizer=False)
messages = [
    {"role": "user", "message_type": "text", "content": "Please transcribe the spoken content into written text."},
    {"role": "user", "message_type": "audio", "content": "speech.wav"},
]
_, text = model.generate(messages, output_type="text", text_temperature=0.0, text_top_k=5,
                         audio_temperature=0.8, audio_top_k=10)
etiology, transcript = text.split("\nTranscript:", 1)   # "Etiology: <condition>" / "<transcript>"
```

## Repository structure

```
data/           SAP and TORGO preparation (metadata, speaker split, manifests, transcript normalization)
  splits/       how to obtain the paper's SAP speaker split (shared on request)
kimi/           Kimi-Audio SFT / EC / EH / EP: training, export, inference
  third_party/Kimi-Audio   upstream Kimi-Audio (git submodule, pinned) + patches/
baselines/      Whisper-large-v3, Parakeet-TDT-0.6B-v2, Phi-4-multimodal, Qwen2-Audio, Granite-Speech
evaluation/     WER / SemScore (SAP Challenge protocol), significance test, paper tables
results/        scores for every row of Tables 1 and 2
hf/             model cards and Hugging Face upload script
requirements/   pinned environments (kimi.txt, phi4.txt, nemo.txt)
```

## Installation

```bash
git clone --recursive https://github.com/MorenoLaQuatra/dysarthric-asr && cd dysarthric-asr
conda create -n dysarthric-asr python=3.12 && conda activate dysarthric-asr
pip install -r requirements/kimi.txt -r kimi/third_party/Kimi-Audio/requirements.txt
bash kimi/scripts/setup.sh        # fetches Kimi-Audio (commit 349251e) and applies kimi/patches/
```

The Phi-4 baseline needs its own environment (`requirements/phi4.txt`, transformers 4.48), and Parakeet
needs NeMo (`requirements/nemo.txt`). All experiments ran on one node with 4× A100 80GB.

## Data

### Speech Accessibility Project (SAP)

SAP is distributed by the [Speech Accessibility Project](https://speechaccessibilityproject.beckman.illinois.edu/)
under a data-use agreement. We use the **v31-05-2025** release; results are reported on the full
development set, as the SAP Challenge test set is not public. Extract the release so that
`<SAP>/train/{audio,metadata}` and `<SAP>/dev/{audio,metadata}` exist, then:

```bash
SAP=/path/to/SAP          # folder with train/ and dev/
# 1. utterance-level metadata (240,047 train / 35,600 dev utterances)
python data/prepare_sap_metadata.py --sap_root $SAP --output_dir data/sap/metadata
# 2. speaker-independent 90/10 train/validation split (see the note below)
python data/split_speakers.py --train_tsv data/sap/metadata/train_asr_metadata.tsv --output_dir data/sap/metadata
# 3. manifests: Kimi-Audio (sft/eh/ep) and NeMo (Parakeet)
python data/make_manifests.py --format kimi --sap_root $SAP --metadata_dir data/sap/metadata --output_dir data/sap/kimi
python data/make_manifests.py --format nemo --sap_root $SAP --metadata_dir data/sap/metadata --output_dir data/sap/nemo
```

The SAP data-use agreement does not allow us to redistribute contributor IDs or file names, so the
speaker lists of the paper's split are not in this repository. We share them on request with anyone who
has SAP access (see `data/splits/README.md`); alternatively, `split_speakers.py --regenerate` draws a new
split with the same procedure. The development set used for all results is the official one and needs no list.

The Phi-4, Qwen2-Audio, Granite and Whisper scripts read the TSVs directly
(`data_root_path` in their `config.yaml` must point to `$SAP`).

### TORGO

```bash
python data/prepare_torgo.py --output_dir data/torgo   # downloads abnerh/TORGO-database from the HF Hub
```

All 16,552 utterances (8 dysarthric, 7 control speakers) are used for zero-shot evaluation.

## Kimi-Audio: SFT, EC, EH, EP

```bash
source kimi/scripts/env.sh
bash kimi/scripts/init_model.sh checkpoints/kimi/base                 # Kimi-Audio-7B-Instruct in training format
bash kimi/scripts/extract_codes.sh data/sap/kimi                       # semantic audio tokens for training

for v in sft eh ep ec; do
  bash kimi/scripts/train.sh  $v data/sap/kimi checkpoints/kimi/base checkpoints/kimi/run-$v
  bash kimi/scripts/export.sh checkpoints/kimi/run-$v checkpoints/kimi/base checkpoints/kimi/kimi-$v
  bash kimi/scripts/infer_sap.sh   checkpoints/kimi/kimi-$v $v data/sap/kimi results/predictions/sap/kimi-$v.jsonl
  bash kimi/scripts/infer_torgo.sh checkpoints/kimi/kimi-$v $v data/torgo     results/predictions/torgo/kimi-$v.jsonl
done
```

`export.sh` merges the LoRA weights of the checkpoint with the lowest validation loss (the merge runs on CPU,
but Kimi-Audio's modeling code imports flash-attn, so one GPU must be visible). To evaluate the released models instead of retraining, pass their Hub ids
(e.g. `morenolq/Kimi-Audio-7B-SAP-EP`) to `infer_sap.sh` / `infer_torgo.sh`.

Zero-shot Kimi-Audio uses the same scripts with `moonshotai/Kimi-Audio-7B-Instruct` and variant `sft`.

## Baselines

Each baseline folder has a `config.yaml` and a `run.sh` that fine-tunes on SAP and writes zero-shot and
fine-tuned predictions on SAP and TORGO to `results/predictions/` (not tracked by git):

```bash
bash baselines/whisper/run.sh
bash baselines/phi4/run.sh          # Phi-4 environment
bash baselines/qwen2_audio/run.sh
bash baselines/granite/run.sh
NEMO_DIR=/path/to/NeMo bash baselines/parakeet/run.sh   # NeMo environment
```

## Evaluation

```bash
# WER + SemScore, per etiology (SAP) or healthy/dysarthric x words/sentences (TORGO)
python evaluation/evaluate.py --predictions results/predictions/sap/kimi-ep.jsonl --dataset sap --semscore \
    --misclassification_analysis --output results/scores/sap/kimi-ep.json
python evaluation/evaluate.py --predictions results/predictions/torgo/kimi-ep.jsonl --dataset torgo --semscore \
    --output results/scores/torgo/kimi-ep.json

# paired significance test (utterances matched by file name)
python evaluation/significance.py --baseline results/predictions/sap/kimi-ep.jsonl \
    --systems results/predictions/sap/ft-kimi-audio.jsonl results/predictions/sap/kimi-ec.jsonl results/predictions/sap/kimi-eh.jsonl

# Tables 1 and 2 (optionally side by side with the published numbers)
python evaluation/paper_tables.py --scores_dir results/scores --compare_paper
```

The metric implementation (`evaluation/sap_evaluator.py`) follows the SAP Challenge: text normalization,
dual references (with / without disfluencies in parentheses) and SemScore (NLI + BERTScore + phonetic
similarity, weights from Phukon et al., 2025). SemScore needs a GPU.

## Results

Overall results from Tables 1 and 2 of the paper (SAP development set; TORGO is zero-shot for every
system). Per-etiology and per-group numbers are in the paper and in `results/scores/`.

| System | Setting | SAP WER | SAP SemScore | TORGO WER | TORGO SemScore |
|---|---|---|---|---|---|
| W-Lv3 | zero-shot | 24.42 | 76.91 | 18.5 | 79.3 |
| Parakeet | zero-shot | 17.77 | 74.30 | 21.2 | 75.6 |
| Q2A | zero-shot | 38.37 | 58.40 | 41.7 | 43.1 |
| Granite | zero-shot | 22.72 | 72.56 | 26.0 | 70.6 |
| Phi-4 | zero-shot | 17.26 | 77.42 | 21.3 | 77.0 |
| Kimi | zero-shot | 16.54 | 77.90 | 19.2 | 79.0 |
| W-Lv3 | adapted | 17.67 | 90.11 | 18.1 | 81.2 |
| Parakeet | adapted | 8.73 | 90.30 | 21.7 | 76.9 |
| Q2A | adapted | 20.75 | 86.52 | 21.0 | 78.7 |
| Granite | adapted | 13.53 | 85.55 | 23.8 | 74.2 |
| Phi-4 | adapted | 8.92 | 90.53 | 18.5 | 81.4 |
| Kimi | adapted | 8.30 | 90.48 | 18.0 | 79.5 |
| Kimi-EC | adapted | 8.49 | 90.42 | 16.7 | 82.6 |
| Kimi-EH | adapted | 8.22 | 90.63 | 16.2 | 82.7 |
| Kimi-EP | adapted | 7.77 | 91.05 | 15.8 | 83.1 |

## Training details

The table lists the settings the released models were actually trained with. Where they differ from the
compact description in the paper (Sec. 4.1), the code is authoritative.

| Model | Trainable parameters | Epochs | Peak LR / schedule | Warmup | Global batch | Model selection |
|---|---|---|---|---|---|---|
| Kimi-Audio SFT / EH / EP | LoRA r=16, α=32, dropout 0.1 on the LM (q,k,v,o,gate,up,down,lm_head) and the Whisper encoder (q,k,v) | 3 | 5e-5, cosine (AdamW, β2=0.95, wd 0.1) | 5% | 64 (4 GPUs × 16 acc.) | lowest validation loss (epoch 1 for EH; not recorded for SFT/EP) |
| Kimi-Audio EC | as above + MLP head (5120→1024→5) on mean-pooled encoder features, CE loss weight 1.0 | 3 | 5e-5, cosine | 5% | **16** (1 GPU × 16 acc.) | lowest validation loss (epoch 1) |
| Phi-4-multimodal | built-in speech LoRA + audio encoder and projector (927.8M params) | 3 | 5e-5, linear | 500 steps | 64 | lowest validation loss |
| Qwen2-Audio-7B-Instruct | LoRA r=128, α=256 on attention/MLP projections | 3 | 1e-5, linear | 500 steps | 64 | lowest validation loss |
| Granite-Speech-3.3-8B | projector + LoRA + audio encoder | 3 | 5e-5, linear | 500 steps | 64 | lowest validation loss |
| Whisper-large-v3 | full fine-tuning | 3 | 1e-5, linear | 500 steps | 64 | lowest validation loss |
| Parakeet-TDT-0.6B-v2 | full fine-tuning (NeMo, `baselines/parakeet/reference_hparams.yaml`) | 50 | 1e-4, cosine | none | 8 per GPU | best validation WER |

Validation data: Kimi-Audio holds out the first 5% of the merged train+validation manifest; all other
models use the speaker-independent validation split (61 speakers). Both are fixed by the split files
described in `data/splits/README.md`.

## Citation

If you use this code or the models, please cite the [paper](https://www.isca-archive.org/interspeech_2026/laquatra26_interspeech.html):

```bibtex
@inproceedings{laquatra26_interspeech,
  title     = {{Etiology-Aware Speech Language Models for Dysarthric Speech Recognition}},
  author    = {Moreno {La Quatra} and Alkis Koudounas and Valerio Mario Salerno and Sabato Marco Siniscalchi},
  year      = {2026},
  booktitle = {{Interspeech 2026}},
  pages     = {5033--5037},
  doi       = {10.21437/Interspeech.2026-1773},
  issn      = {2958-1796},
}
```

## Acknowledgements

This work has been partially supported by the "D.A.R.E. - Digital Lifelong Prevention" project
(code: PNC0000002, CUP: B53C22006450001), co-funded by the Italian Complementary National Plan PNC-I.1.
Kimi-Audio training code is adapted from [MoonshotAI/Kimi-Audio](https://github.com/MoonshotAI/Kimi-Audio) (MIT).
The metric and normalization code follows the SAP Challenge evaluation ([SAPC-template](https://github.com/xiuwenz2/SAPC-template)).
We thank the Speech Accessibility Project and the TORGO authors for the data.

## License

Code: [MIT](LICENSE). Models: MIT, as Kimi-Audio. The SAP data is not included and is subject to the
SAP data-use agreement.
