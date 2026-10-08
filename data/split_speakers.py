#!/usr/bin/env python3
"""
Speaker-independent 90/10 train/validation split of the SAP training set.

By default the split used in the paper is restored from the speaker lists in data/splits/
(sap_{train,val}_speakers.txt, shared on request, see data/splits/README.md). The original
split was drawn with seed 42, 10% of the speakers of each etiology going to validation (61 speakers).

With --regenerate, a new stratified split is drawn instead (speakers sorted before
shuffling, so it is deterministic but not identical to the paper's).
"""

import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd

SPLITS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "splits")


def stratified_speaker_split(df, val_ratio, seed):
    rng = np.random.RandomState(seed)
    by_etiology = defaultdict(list)
    for speaker, etiology in df.groupby("contributor_id")["etiology"].first().sort_index().items():
        by_etiology[etiology].append(speaker)
    val = set()
    for etiology in sorted(by_etiology):
        speakers = np.array(by_etiology[etiology])
        rng.shuffle(speakers)
        val.update(speakers[:max(1, int(len(speakers) * val_ratio))])
    return set(df["contributor_id"]) - val, val


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train_tsv", required=True, help="train_asr_metadata.tsv")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    args = parser.parse_args()

    df = pd.read_csv(args.train_tsv, sep="\t", low_memory=False)
    if args.regenerate:
        train_spk, val_spk = stratified_speaker_split(df, args.val_ratio, args.seed)
    else:
        def read(name):
            path = os.path.join(SPLITS_DIR, name)
            if not os.path.exists(path):
                raise SystemExit(f"{path} not found. The paper's split is shared on request "
                                 "(see data/splits/README.md); use --regenerate for a new split.")
            with open(path) as f:
                return {line.strip() for line in f if line.strip()}
        train_spk, val_spk = read("sap_train_speakers.txt"), read("sap_val_speakers.txt")
        unknown = set(df["contributor_id"]) - train_spk - val_spk
        if unknown:
            raise ValueError(f"{len(unknown)} speakers are not in the released split lists")

    os.makedirs(args.output_dir, exist_ok=True)
    for name, speakers in (("sapc_metadata_train.tsv", train_spk), ("sapc_metadata_val.tsv", val_spk)):
        part = df[df["contributor_id"].isin(speakers)]
        part.to_csv(os.path.join(args.output_dir, name), sep="\t", index=False)
        print(f"{name}: {len(part)} utterances, {part['contributor_id'].nunique()} speakers")


if __name__ == "__main__":
    main()
