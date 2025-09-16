# Dysarthric Speech Recognition Research

Research on automatic speech recognition for dysarthric speech using modern speech language models and clinical-informed approaches.

> [!WARNING]  
> This repository contains research currently under peer review. Full implementation details will be released upon acceptance.

## Overview

This work investigates speech language models for dysarthric speech recognition across multiple neurological conditions. We evaluate both encoder-decoder architectures and recent speech language models on large-scale dysarthric speech data, exploring approaches that incorporate clinical knowledge to improve dysarthric speech recognition.

## Models

**Speech Language Models:**
- **Phi-4**: [microsoft/Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct)
- **Qwen2-Audio**: [Qwen/Qwen2-Audio-7B-Instruct](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct)
- **Granite**: [ibm-granite/granite-speech-3.3-8b](https://huggingface.co/ibm-granite/granite-speech-3.3-8b)
- **Kimi**: [moonshotai/Kimi-Audio-7B-Instruct](https://huggingface.co/moonshotai/Kimi-Audio-7B-Instruct)

**Encoder-Decoder Baselines:**
- **Whisper**: [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3)
- **Parakeet**: [nvidia/parakeet-tdt-0.6b-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)

## Datasets

### Speech Accessibility Project (SAP)
- **Scale**: 240k training utterances, 35.6k evaluation utterances
- **Duration**: 547 hours training, 81 hours evaluation
- **Conditions**: Five neurological conditions affecting speech production
  - Amyotrophic Lateral Sclerosis (ALS)
  - Parkinson's Disease
  - Stroke
  - Down Syndrome  
  - Cerebral Palsy
- **Access**: [SAP Website](https://speechaccessibilityproject.beckman.illinois.edu/)
- **License**: Research use with institutional agreement

### TORGO Database
- **Purpose**: Cross-dataset generalization evaluation
- **Content**: Healthy and dysarthric speakers
- **Tasks**: Single words and sentence-level recognition
- **Access**: [TORGO Website](http://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html)
- **License**: Free for research use

## Training Configuration

- **Speech Language Models**: LoRA fine-tuning (rank=16, alpha=32)
- **Encoder-Decoder Models**: Full parameter fine-tuning
- **Evaluation Metrics**: Word Error Rate (WER) and Semantic Score (SemScore)
- **Cross-Dataset Validation**: Zero-shot evaluation on TORGO


## Repository Contents (TBD upon acceptance)

- Fine-tuning scripts for all evaluated models
- Inference pipelines for dysarthric speech recognition
- Adapted model checkpoints 

## Authors

- **Moreno La Quatra** - Kore University of Enna, Italy
- **Alkis Koudounas** - Politecnico di Torino, Italy
- **Valerio Mario Salerno** - Kore University of Enna, Italy
- **Sabato Marco Siniscalchi** - Università degli Studi di Palermo, Italy
