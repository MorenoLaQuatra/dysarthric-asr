#!/usr/bin/env python3
"""
Evaluate ASR predictions on SAP or TORGO.

Input: a JSONL file with one utterance per line, as written by the inference scripts:
    {"audio_filepath": ..., "text": <reference>, "pred_text": <prediction>,
     "etiology": <ground-truth etiology>, ["speech_type": single_word|sentence], ...}

Predictions and references may use the etiology-prediction (EP) format
"Etiology: <etiology>\nTranscript: <transcript>"; the etiology prefix is stripped before
computing WER / SemScore and the predicted etiology is scored as a classification task.

Reported numbers:
  * overall WER / SemScore
  * per-group WER / SemScore (SAP: per etiology; TORGO: healthy/non-healthy x words/sentences)
  * etiology classification accuracy (EP predictions, or `pred_etiology` field for EC)
  * optionally, WER stratified by etiology-prediction correctness (--misclassification_analysis)
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.sap_evaluator import DysarthricASREvaluator  # noqa: E402

EP_PATTERN = re.compile(r"Etiology:\s*([^\n]*)\s*\nTranscript:\s*(.+?)(?:\n|$)", re.DOTALL | re.IGNORECASE)


def load_predictions(path):
    """Load a JSONL (or JSON list) prediction file."""
    with open(path, encoding="utf-8") as f:
        content = f.read().strip()
    if content.startswith("["):
        return json.loads(content)
    data = []
    for line_num, line in enumerate(content.split("\n"), 1):
        line = line.strip()
        if not line:
            continue
        try:
            data.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Warning: failed to parse line {line_num}: {e}")
    return data


def parse_ep_format(text):
    """
    Split 'Etiology: <e>\\nTranscript: <t>' into (etiology, transcript).
    Returns (None, text) when the text is a plain transcript.
    """
    if not isinstance(text, str):
        text = str(text)

    match = EP_PATTERN.search(text)
    if match:
        etiology = match.group(1).strip() or "Unknown"
        return etiology, match.group(2).strip()

    # Fallback: line-based parsing, keeping every line after "Transcript:"
    etiology, transcript = None, None
    lines = text.split("\n")
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.lower().startswith("etiology:"):
            etiology = stripped[9:].strip() or "Unknown"
        elif stripped.lower().startswith("transcript:"):
            transcript = stripped[11:].strip()
            rest = "\n".join(lines[i + 1:]).strip()
            if rest:
                transcript = transcript + "\n" + rest if transcript else rest
            break
    if etiology is not None and transcript is not None:
        return etiology, transcript

    if "etiology:" in text.lower():
        # The model started the EP format but did not produce a transcript marker
        return "Unknown", text
    return None, text


def extract_records(data, pred_key, ref_key):
    records = []
    for item in data:
        if pred_key not in item or ref_key not in item:
            continue
        pred_etiology, pred_transcript = parse_ep_format(item[pred_key])
        ref_etiology, ref_transcript = parse_ep_format(item[ref_key])

        # Ground-truth etiology: dedicated field first, EP-formatted reference otherwise
        etiology = item.get("etiology") or ref_etiology or "unknown"
        # Predicted etiology: EP output, or the classification head output (EC)
        if pred_etiology is None:
            pred_etiology = item.get("pred_etiology")

        records.append({
            "id": os.path.basename(item.get("audio_filepath", str(len(records)))),
            "pred": pred_transcript,
            "ref": ref_transcript,
            "etiology": etiology,
            "pred_etiology": pred_etiology,
            "speech_type": item.get("speech_type"),
        })

    # In an etiology-aware run, an utterance without a predicted etiology counts as "Unknown"
    if any(r["pred_etiology"] is not None for r in records):
        for r in records:
            if r["pred_etiology"] is None:
                r["pred_etiology"] = "Unknown"
    return records


def score(records):
    """Corpus-level metrics of a subset of records (per-utterance scores precomputed)."""
    if not records:
        return None
    errors = sum(r["errors"] for r in records)
    lengths = sum(r["length"] for r in records)
    out = {"num_samples": len(records), "wer": round(errors / lengths * 100, 4) if lengths else 0.0}
    if "semscore" in records[0]:
        out["semscore"] = round(float(np.mean([r["semscore"] for r in records])) * 100, 4)
    return out


def group_key(record, dataset):
    if dataset == "torgo":
        health = "healthy" if str(record["etiology"]).lower() == "healthy" else "non-healthy"
        return f"{health}_{record['speech_type']}"
    return record["etiology"]


def classification_metrics(records):
    refs = [r["etiology"] for r in records]
    preds = [r["pred_etiology"] for r in records]
    labels = sorted(set(refs) | set(preds))
    _, _, f1, _ = precision_recall_fscore_support(refs, preds, average="weighted", zero_division=0)
    return {
        "accuracy": accuracy_score(refs, preds) * 100,
        "weighted_f1": f1 * 100,
        "labels": labels,
        "confusion_matrix": confusion_matrix(refs, preds, labels=labels).tolist(),
        "report": classification_report(refs, preds, labels=labels, zero_division=0, output_dict=True),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", required=True, help="JSONL file produced by an inference script")
    parser.add_argument("--dataset", choices=["sap", "torgo"], default="sap",
                        help="Controls the per-group breakdown (default: sap)")
    parser.add_argument("--pred_key", default="pred_text")
    parser.add_argument("--ref_key", default="text")
    parser.add_argument("--semscore", action="store_true", help="Also compute SemScore (slow, needs a GPU)")
    parser.add_argument("--misclassification_analysis", action="store_true",
                        help="Stratify WER by etiology-prediction correctness (EP models)")
    parser.add_argument("--output", default=None, help="Where to write the JSON summary")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    data = load_predictions(args.predictions)
    records = extract_records(data, args.pred_key, args.ref_key)
    print(f"Loaded {len(records)} utterances from {args.predictions}")

    evaluator = DysarthricASREvaluator(device=args.device, compute_wer=True, compute_semscore=args.semscore)
    utt = evaluator.utterance_scores([r["pred"] for r in records], [r["ref"] for r in records])
    for i, r in enumerate(records):
        r["errors"], r["length"] = utt["errors"][i], utt["lengths"][i]
        if args.semscore:
            r["semscore"] = utt["semscore"][i]

    results = {"predictions": args.predictions, "dataset": args.dataset}
    results["overall"] = score(records)

    groups = defaultdict(list)
    for r in records:
        groups[group_key(r, args.dataset)].append(r)
    results["groups"] = {k: score(v) for k, v in sorted(groups.items())}

    has_etiology_pred = all(r["pred_etiology"] is not None for r in records)
    if has_etiology_pred:
        results["etiology_classification"] = classification_metrics(records)

    if args.misclassification_analysis:
        if not has_etiology_pred:
            parser.error("--misclassification_analysis needs etiology predictions")
        for r in records:
            r["correct"] = r["pred_etiology"].lower().strip() == str(r["etiology"]).lower().strip()
        analysis = {
            "correct": score([r for r in records if r["correct"]]),
            "incorrect": score([r for r in records if not r["correct"]]),
            "per_group": {},
        }
        for k, v in sorted(groups.items()):
            analysis["per_group"][k] = {
                "correct": score([r for r in v if r["correct"]]),
                "incorrect": score([r for r in v if not r["correct"]]),
            }
        results["misclassification_analysis"] = analysis

    # Console summary
    o = results["overall"]
    print("=" * 60)
    print(f"Overall  WER {o['wer']:.2f}%" + (f"  SemScore {o['semscore']:.2f}" if "semscore" in o else "") +
          f"  (n={o['num_samples']})")
    for k, g in results["groups"].items():
        print(f"  {k:<28} WER {g['wer']:.2f}%" + (f"  SemScore {g['semscore']:.2f}" if "semscore" in g else "") +
              f"  (n={g['num_samples']})")
    if has_etiology_pred:
        print(f"Etiology accuracy {results['etiology_classification']['accuracy']:.2f}%")
    if args.misclassification_analysis:
        a = results["misclassification_analysis"]
        print(f"Correct etiology WER {a['correct']['wer']:.2f}%  |  incorrect etiology WER {a['incorrect']['wer']:.2f}%")
    print("=" * 60)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x))
        print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
