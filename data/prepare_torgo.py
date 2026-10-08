#!/usr/bin/env python3
"""
Download TORGO from the Hugging Face Hub (abnerh/TORGO-database) and write:
  <output_dir>/audio/*.wav
  <output_dir>/torgo_conversations.jsonl   Kimi-Audio conversation format (EP-style reference)
  <output_dir>/torgo.tsv                   metadata for the other models' inference scripts

Each utterance is labelled healthy / dysarthric (speech_status) and single_word / sentence
(one word vs. more). All 16,552 utterances are used for zero-shot evaluation.
"""

import argparse
import csv
import json
import os
from pathlib import Path

import soundfile as sf
from datasets import load_dataset

PROMPT = "Please transcribe the spoken content into written text."


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    out = Path(args.output_dir).resolve()
    (out / "audio").mkdir(parents=True, exist_ok=True)
    data = load_dataset("abnerh/TORGO-database")["train"]

    with open(out / "torgo_conversations.jsonl", "w", encoding="utf-8") as fj, open(out / "torgo.tsv", "w", newline="") as ft:
        tsv = csv.writer(ft, delimiter="\t")
        tsv.writerow(["audio_path", "transcript", "etiology", "speech_type", "contributor_id", "category", "duration"])
        for idx, sample in enumerate(data):
            audio = sample["audio"]
            stem = Path(audio.get("path") or f"audio_{idx}.wav").stem
            wav_path = out / "audio" / f"{stem}_{idx}.wav"
            if not wav_path.exists():
                sf.write(wav_path, audio["array"], audio["sampling_rate"])

            transcript = sample["transcription"]
            etiology = sample["speech_status"]
            speech_type = "single_word" if len(transcript.strip().split()) == 1 else "sentence"
            fj.write(json.dumps({
                "task_type": "understanding",
                "conversation": [
                    {"role": "user", "message_type": "text", "content": PROMPT},
                    {"role": "user", "message_type": "audio", "content": str(wav_path)},
                    {"role": "assistant", "message_type": "text", "content": f"Etiology: {etiology}\nTranscript: {transcript}"},
                ],
                "etiology": etiology,
                "speech_type": speech_type,
                "transcript": transcript,
            }, ensure_ascii=False) + "\n")
            speaker = stem.split("_")[0]
            tsv.writerow([str(wav_path), transcript, etiology, speech_type, speaker, speech_type, sample.get("duration", 0.0)])

    print(f"Wrote {len(data)} utterances to {out}")


if __name__ == "__main__":
    main()
