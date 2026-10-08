#!/usr/bin/env python3
"""
Build utterance-level metadata TSVs from the SAP release (v31-05-2025).

Expected layout (as distributed by the Speech Accessibility Project, after extraction):
    <sap_root>/train/metadata/*.json   <sap_root>/train/audio/*.wav
    <sap_root>/dev/metadata/*.json     <sap_root>/dev/audio/*.wav

Outputs <output_dir>/train_asr_metadata.tsv and <output_dir>/dev_asr_metadata.tsv with
one row per audio file. Audio paths are stored relative to <sap_root>
(e.g. ./dev/audio/<file>.wav). Only the fields used in the paper are kept
(the full clinical ratings are not needed).
"""

import argparse
import csv
import json
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

COLUMNS = ["contributor_id", "etiology", "audio_path", "filename", "prompt_text", "transcript", "category", "sub_category"]


def read_split(sap_root, split):
    rows = []
    audio_dir = os.path.join(".", split, "audio")
    for json_file in tqdm(sorted(Path(sap_root, split, "metadata").glob("*.json")), desc=split):
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)
        for file_info in data.get("Files", []):
            filename = file_info.get("Filename", "")
            if not filename:
                continue
            audio_path = os.path.join(audio_dir, filename)
            if not os.path.exists(os.path.join(sap_root, audio_path)):
                print(f"Audio file not found: {audio_path}")
                continue
            prompt = file_info.get("Prompt", {})
            rows.append({
                "contributor_id": data.get("Contributor ID", ""),
                "etiology": data.get("Etiology", ""),
                "audio_path": audio_path,
                "filename": filename,
                "prompt_text": prompt.get("Prompt Text", ""),
                "transcript": prompt.get("Transcript", ""),
                "category": prompt.get("Category Description", ""),
                "sub_category": prompt.get("Sub Category Description", ""),
            })
    return pd.DataFrame(rows, columns=COLUMNS)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sap_root", required=True, help="Folder containing train/ and dev/")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    for split, name in (("train", "train_asr_metadata.tsv"), ("dev", "dev_asr_metadata.tsv")):
        df = read_split(args.sap_root, split)
        out = os.path.join(args.output_dir, name)
        df.to_csv(out, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)
        print(f"{out}: {len(df)} utterances, {df['contributor_id'].nunique()} speakers")
        print(df.groupby("etiology")["contributor_id"].agg(["count", "nunique"]).rename(columns={"count": "utterances", "nunique": "speakers"}))


if __name__ == "__main__":
    main()
