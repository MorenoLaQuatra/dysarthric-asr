#!/usr/bin/env python3
"""
Paired significance test between two or more systems.

Utterance-level WERs are computed with the same normalization and dual-reference
rule as the corpus-level WER (evaluation/sap_evaluator.py). Utterances are paired
by audio file name, so only utterances present in every file are compared.

Example:
    python evaluation/significance.py \
        --baseline results/sap/kimi-sft.jsonl \
        --systems results/sap/kimi-ec.jsonl results/sap/kimi-eh.jsonl results/sap/kimi-ep.jsonl
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.evaluate import extract_records, load_predictions  # noqa: E402
from evaluation.sap_evaluator import DysarthricASREvaluator  # noqa: E402


def utterance_errors(evaluator, records):
    """Per-utterance (edit distance, reference length), keyed by audio file name."""
    utt = evaluator.utterance_scores([r["pred"] for r in records], [r["ref"] for r in records])
    return {r["id"]: (e, l) for r, e, l in zip(records, utt["errors"], utt["lengths"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", required=True, help="Prediction JSONL of the reference system")
    parser.add_argument("--systems", nargs="+", required=True, help="Prediction JSONL files to compare")
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    evaluator = DysarthricASREvaluator(compute_wer=True, compute_semscore=False)

    def load(path):
        return utterance_errors(evaluator, extract_records(load_predictions(path), "pred_text", "text"))

    base = load(args.baseline)
    results = {}
    for path in args.systems:
        sys_errors = load(path)
        ids = sorted(set(base) & set(sys_errors))
        b = np.array([base[i][0] / base[i][1] if base[i][1] > 0 else 0.0 for i in ids])
        s = np.array([sys_errors[i][0] / sys_errors[i][1] if sys_errors[i][1] > 0 else 0.0 for i in ids])
        b_corpus = sum(base[i][0] for i in ids) / sum(base[i][1] for i in ids)
        s_corpus = sum(sys_errors[i][0] for i in ids) / sum(sys_errors[i][1] for i in ids)

        t_stat, p_t = stats.ttest_rel(b, s)
        try:
            _, p_w = stats.wilcoxon(b, s)
        except ValueError:
            p_w = 1.0
        diffs = b - s
        boot = [diffs[rng.integers(0, len(diffs), len(diffs))].mean() for _ in range(args.n_bootstrap)]

        results[path] = {
            "n_paired": len(ids),
            "baseline_corpus_wer": b_corpus * 100,
            "system_corpus_wer": s_corpus * 100,
            "relative_reduction_percent": (b_corpus - s_corpus) / b_corpus * 100,
            "mean_utt_wer_diff": diffs.mean() * 100,
            "bootstrap_95ci": [float(np.percentile(boot, 2.5) * 100), float(np.percentile(boot, 97.5) * 100)],
            "paired_ttest_p": float(p_t),
            "t_stat": float(t_stat),
            "wilcoxon_p": float(p_w),
        }
        r = results[path]
        print(f"{os.path.basename(path)}: n={r['n_paired']}  WER {r['baseline_corpus_wer']:.2f} -> {r['system_corpus_wer']:.2f} "
              f"(rel. {r['relative_reduction_percent']:+.1f}%)  t-test p={r['paired_ttest_p']:.2e}  "
              f"Wilcoxon p={r['wilcoxon_p']:.2e}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"baseline": args.baseline, "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
