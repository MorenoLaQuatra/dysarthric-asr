# Hugging Face release

The four Kimi-Audio checkpoints of the paper are published as separate model repositories,
each containing the merged fp32 weights exactly as evaluated (LM shards + `whisper-large-v3/`
encoder, plus `etiology_classifier.pt` for EC), the Kimi-Audio text tokenizer and a model card.

| Variant | Exported folder | Hub repository |
|---|---|---|
| SFT | `finetuned_hf_for_inference` | `<namespace>/Kimi-Audio-7B-SAP-SFT` |
| EC  | `finetuned_hf_for_inference_mtl` | `<namespace>/Kimi-Audio-7B-SAP-EC` |
| EH  | `finetuned_hf_for_inference_etiology` | `<namespace>/Kimi-Audio-7B-SAP-EH` |
| EP  | `finetuned_hf_for_inference_etiology_prediction` | `<namespace>/Kimi-Audio-7B-SAP-EP` |

```bash
huggingface-cli login
# 1. staging folders (symlinks to the exported weights + tokenizer + model cards with the paper scores)
python hf/release.py prepare --checkpoints_root /path/to/exported/models \
    --staging_dir hf_staging --namespace <namespace> --scores results/scores
# 2. check the generated hf_staging/*/README.md, then upload (~44 GB per model, resumable)
python hf/release.py upload --staging_dir hf_staging --namespace <namespace> [--private]
```

To update only the model cards once the weights are online:

```bash
python hf/release.py cards --checkpoints_root /path/to/exported/models --staging_dir hf_staging --namespace <namespace>
```

`model_card.md` is the card template; `release.py` fills in the per-variant description and the
published numbers of Tables 1 and 2 (`evaluation/paper_tables.py`).
