"""Convert data/torgo/torgo.tsv into a NeMo manifest (etiology and speech_type are kept for evaluation)."""
import argparse
import csv
import json

import librosa

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--torgo_tsv", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

with open(args.torgo_tsv) as f, open(args.output, "w") as out:
    for row in csv.DictReader(f, delimiter="\t"):
        out.write(json.dumps({
            "audio_filepath": row["audio_path"],
            "text": row["transcript"],
            "duration": librosa.get_duration(path=row["audio_path"]),
            "etiology": row["etiology"],
            "speech_type": row["speech_type"],
        }) + "\n")
