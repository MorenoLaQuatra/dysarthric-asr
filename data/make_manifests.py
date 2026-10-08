#!/usr/bin/env python3
"""
Build training / test manifests from the SAP metadata TSVs.

--format kimi  -> conversation JSONL for Kimi-Audio, one folder per variant:
    sft/  prompt "Please transcribe the spoken content into written text."   -> "<transcript>"
    eh/   prompt "... Consider that the speaker has been diagnosed with <etiology>." -> "<transcript>"
    ep/   prompt "Please transcribe the spoken content into written text."   -> "Etiology: <etiology>\\nTranscript: <transcript>"
  each with train_valid_merged_conversations.jsonl (train + validation speakers; the first
  5% is used as the Kimi validation set) and test_conversations.jsonl (SAP dev set).
  EC training uses the ep/ manifest.

--format nemo  -> NeMo manifests for Parakeet (train/val/test_manifest.json).
  Training/validation transcripts are normalized; test references are kept raw since
  normalization happens in the evaluator.

Training transcripts are normalized with data/transcript_normalizer.py (lowercase,
SAP annotations resolved as in the paper).
"""

import argparse
import gzip
import json
import os

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from transcript_normalizer import TranscriptNormalizer

PROMPT = "Please transcribe the spoken content into written text."
EH_PROMPT = "Please transcribe the spoken content into written text. Consider that the speaker has been diagnosed with {etiology}."
EP_TEMPLATE = "Etiology: {etiology}\nTranscript: {transcript}"
ORDER_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "splits", "sap_train_val_order.txt.gz")


def load_tsv(path):
    df = pd.read_csv(path, sep="\t", low_memory=False)
    df = df.dropna(subset=["transcript"])
    return df[df["transcript"].astype(str).str.strip() != ""]


def apply_release_order(df):
    """Reorder the merged train+validation rows as in the paper's runs (affects the Kimi validation split)."""
    if not os.path.exists(ORDER_FILE):
        return df
    with gzip.open(ORDER_FILE, "rt") as f:
        order = {name.strip(): i for i, name in enumerate(f) if name.strip()}
    rank = df["audio_path"].map(lambda p: order.get(os.path.basename(p), len(order)))
    return df.assign(_rank=rank.values).sort_values("_rank", kind="stable").drop(columns="_rank")


def kimi_entries(df, sap_root, variant, normalizer):
    entries = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=variant):
        audio = os.path.join(sap_root, row["audio_path"])
        if not os.path.exists(audio):
            continue
        etiology = str(row["etiology"]).strip()
        transcript = normalizer.normalize(str(row["transcript"]).strip())
        if not transcript.strip():
            continue
        prompt = EH_PROMPT.format(etiology=etiology) if variant == "eh" else PROMPT
        target = EP_TEMPLATE.format(etiology=etiology, transcript=transcript) if variant == "ep" else transcript
        entry = {
            "task_type": "understanding",
            "conversation": [
                {"role": "user", "message_type": "text", "content": prompt},
                {"role": "user", "message_type": "audio", "content": audio},
                {"role": "assistant", "message_type": "text", "content": target},
            ],
            "etiology": etiology,
        }
        if pd.notna(row.get("category")):
            entry["speech_type"] = str(row["category"]).strip()
        entries.append(entry)
    return entries


def valid_audio(path, target_sr=16000):
    try:
        audio, sr = librosa.load(path, sr=None)
        if len(audio) == 0 or not np.isfinite(audio).all() or len(audio) / sr < 0.01:
            return False
        if sr != target_sr and not np.isfinite(librosa.resample(audio, orig_sr=sr, target_sr=target_sr)).all():
            return False
        return True
    except Exception:
        return False


def nemo_entries(df, sap_root, normalizer):
    entries = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        audio = os.path.join(sap_root, row["audio_path"])
        if not os.path.exists(audio) or not valid_audio(audio):
            continue
        text = str(row["transcript"]).strip()
        if normalizer is not None:
            text = normalizer.normalize(text)
            if not text.strip():
                continue
        entries.append({
            "audio_filepath": audio,
            "text": text,
            "duration": librosa.get_duration(path=audio),
            "etiology": row["etiology"] if pd.notna(row["etiology"]) else None,
        })
    return entries


def write_jsonl(entries, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"{path}: {len(entries)} entries")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--format", choices=["kimi", "nemo"], required=True)
    parser.add_argument("--sap_root", required=True, help="SAP folder (audio paths in the TSVs are relative to it)")
    parser.add_argument("--metadata_dir", required=True, help="Folder with sapc_metadata_{train,val}.tsv and dev_asr_metadata.tsv")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--variants", nargs="+", default=["sft", "eh", "ep"], help="Kimi variants to build")
    args = parser.parse_args()

    train = load_tsv(os.path.join(args.metadata_dir, "sapc_metadata_train.tsv"))
    val = load_tsv(os.path.join(args.metadata_dir, "sapc_metadata_val.tsv"))
    test = load_tsv(os.path.join(args.metadata_dir, "dev_asr_metadata.tsv"))
    normalizer = TranscriptNormalizer()

    if args.format == "kimi":
        merged = apply_release_order(pd.concat([train, val], ignore_index=True))
        for variant in args.variants:
            out = os.path.join(args.output_dir, variant)
            write_jsonl(kimi_entries(merged, args.sap_root, variant, normalizer), os.path.join(out, "train_valid_merged_conversations.jsonl"))
            write_jsonl(kimi_entries(test, args.sap_root, variant, normalizer), os.path.join(out, "test_conversations.jsonl"))
    else:
        write_jsonl(nemo_entries(train, args.sap_root, normalizer), os.path.join(args.output_dir, "train_manifest.json"))
        write_jsonl(nemo_entries(val, args.sap_root, normalizer), os.path.join(args.output_dir, "val_manifest.json"))
        write_jsonl(nemo_entries(test, args.sap_root, None), os.path.join(args.output_dir, "test_manifest.json"))


if __name__ == "__main__":
    main()
