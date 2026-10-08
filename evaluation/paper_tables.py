#!/usr/bin/env python3
"""
Rebuild Tables 1 (SAP) and 2 (TORGO) of the paper from evaluation outputs.

Expects <scores_dir>/{sap,torgo}/<system>.json written by evaluation/evaluate.py
(see results/README.md for the system names). With --compare_paper, every cell is
printed next to the value published in the paper and differences are flagged.
"""

import argparse
import json
import os

SYSTEMS = [
    ("zs-whisper-large-v3", "W-Lv3", "zero-shot"), ("zs-parakeet-tdt-0.6b-v2", "Parakeet", "zero-shot"),
    ("zs-qwen2-audio", "Q2A", "zero-shot"), ("zs-granite-speech", "Granite", "zero-shot"),
    ("zs-phi4-mm", "Phi-4", "zero-shot"), ("zs-kimi-audio", "Kimi", "zero-shot"),
    ("ft-whisper-large-v3", "W-Lv3", "adapted"), ("ft-parakeet-tdt-0.6b-v2", "Parakeet", "adapted"),
    ("ft-qwen2-audio", "Q2A", "adapted"), ("ft-granite-speech", "Granite", "adapted"),
    ("ft-phi4-mm", "Phi-4", "adapted"), ("ft-kimi-audio", "Kimi", "adapted"),
    ("kimi-ec", "Kimi-EC", "adapted"), ("kimi-eh", "Kimi-EH", "adapted"), ("kimi-ep", "Kimi-EP", "adapted"),
]
SAP_COLS = ["ALS", "Parkinson's Disease", "Stroke", "Down Syndrome", "Cerebral Palsy", "overall"]
TORGO_COLS = ["healthy_single_word", "healthy_sentence", "non-healthy_single_word", "non-healthy_sentence", "overall"]

# Published values: (WER, SemScore) per column
PAPER_SAP = {
    "zs-whisper-large-v3": [(23.60, 76.37), (20.37, 85.68), (14.78, 85.26), (37.29, 57.27), (37.94, 61.10), (24.42, 76.91)],
    "zs-parakeet-tdt-0.6b-v2": [(17.36, 71.81), (12.15, 84.57), (16.21, 83.31), (40.64, 53.36), (32.77, 56.94), (17.77, 74.30)],
    "zs-qwen2-audio": [(36.38, 57.95), (32.65, 63.89), (36.97, 63.89), (57.25, 41.07), (56.96, 41.10), (38.37, 58.40)],
    "zs-granite-speech": [(23.41, 67.99), (15.91, 83.68), (17.23, 83.31), (41.59, 54.17), (42.69, 53.90), (22.72, 72.56)],
    "zs-phi4-mm": [(15.76, 76.10), (12.08, 86.49), (14.85, 85.26), (38.05, 58.16), (32.81, 61.54), (17.26, 77.42)],
    "zs-kimi-audio": [(14.70, 76.81), (11.91, 87.04), (14.98, 85.79), (36.21, 59.12), (30.65, 61.39), (16.54, 77.90)],
    "ft-whisper-large-v3": [(15.72, 90.71), (16.88, 93.19), (9.90, 92.77), (17.94, 80.48), (23.67, 84.86), (17.67, 90.11)],
    "ft-parakeet-tdt-0.6b-v2": [(6.20, 91.07), (7.15, 93.14), (11.47, 92.02), (19.18, 80.86), (14.26, 85.56), (8.73, 90.30)],
    "ft-qwen2-audio": [(19.15, 87.43), (19.01, 90.62), (13.40, 88.17), (24.19, 75.66), (28.95, 78.93), (20.75, 86.52)],
    "ft-granite-speech": [(12.41, 84.46), (9.67, 91.61), (13.01, 88.50), (25.26, 73.66), (26.12, 75.08), (13.53, 85.55)],
    "ft-phi4-mm": [(6.76, 91.26), (7.22, 93.64), (9.84, 92.70), (18.69, 80.75), (15.04, 85.21), (8.92, 90.53)],
    "ft-kimi-audio": [(5.98, 91.63), (6.46, 94.07), (10.87, 91.93), (18.29, 79.86), (14.91, 83.89), (8.30, 90.48)],
    "kimi-ec": [(6.02, 91.79), (6.59, 93.99), (9.93, 92.98), (19.27, 79.49), (15.40, 83.61), (8.49, 90.42)],
    "kimi-eh": [(5.97, 91.69), (6.43, 94.08), (11.18, 92.89), (17.97, 80.55), (14.58, 84.20), (8.22, 90.63)],
    "kimi-ep": [(5.75, 92.12), (6.03, 94.32), (8.80, 93.10), (17.90, 80.64), (13.76, 85.21), (7.77, 91.05)],
}
PAPER_TORGO = {
    "zs-whisper-large-v3": [(24.1, 84.2), (2.5, 97.8), (60.0, 60.0), (28.3, 71.5), (18.5, 79.3)],
    "zs-parakeet-tdt-0.6b-v2": [(30.1, 80.5), (3.0, 97.9), (66.9, 52.9), (30.8, 70.3), (21.2, 75.6)],
    "zs-qwen2-audio": [(81.1, 34.3), (15.0, 88.4), (90.6, 23.9), (41.6, 61.6), (41.7, 43.1)],
    "zs-granite-speech": [(46.1, 71.7), (2.6, 98.2), (72.0, 51.1), (36.4, 66.8), (26.0, 70.6)],
    "zs-phi4-mm": [(29.1, 81.5), (2.8, 97.9), (63.6, 56.8), (33.7, 69.6), (21.3, 77.0)],
    "zs-kimi-audio": [(23.4, 85.4), (3.8, 97.8), (62.2, 56.0), (28.3, 73.3), (19.2, 79.0)],
    "ft-whisper-large-v3": [(23.6, 85.3), (3.3, 97.3), (52.2, 65.9), (28.8, 71.1), (18.1, 81.2)],
    "ft-parakeet-tdt-0.6b-v2": [(33.6, 79.5), (4.0, 96.8), (59.3, 60.7), (31.3, 70.3), (21.7, 76.9)],
    "ft-qwen2-audio": [(29.0, 82.7), (3.7, 97.3), (58.8, 61.9), (32.7, 67.5), (21.0, 78.7)],
    "ft-granite-speech": [(36.5, 77.2), (2.9, 97.8), (65.4, 55.5), (37.4, 64.8), (23.8, 74.2)],
    "ft-phi4-mm": [(22.6, 86.0), (3.2, 97.5), (52.7, 65.4), (31.1, 69.5), (18.5, 81.4)],
    "ft-kimi-audio": [(22.9, 84.5), (2.4, 98.3), (57.2, 59.5), (28.6, 72.9), (18.0, 79.5)],
    "kimi-ec": [(18.5, 88.0), (2.5, 98.2), (52.5, 65.1), (28.5, 71.8), (16.7, 82.6)],
    "kimi-eh": [(20.2, 87.1), (2.4, 98.4), (51.4, 66.1), (25.4, 74.9), (16.2, 82.7)],
    "kimi-ep": [(20.3, 87.1), (2.5, 98.4), (50.1, 67.4), (24.2, 76.0), (15.8, 83.1)],
}


def cell(scores, col):
    g = scores["overall"] if col == "overall" else scores["groups"].get(col)
    return (g["wer"], g.get("semscore")) if g else (None, None)


def table(scores_dir, dataset, cols, paper, decimals, compare):
    print(f"\n## {'Table 1 - SAP development set' if dataset == 'sap' else 'Table 2 - TORGO (zero-shot)'}\n")
    print("| System | " + " | ".join(c.replace("_", " ") for c in cols) + " |")
    print("|---|" + "---|" * len(cols))
    n_diff = 0
    for key, label, kind in SYSTEMS:
        path = os.path.join(scores_dir, dataset, f"{key}.json")
        if not os.path.exists(path):
            print(f"| {label} ({kind}) | " + " | ".join("n/a" for _ in cols) + " |")
            continue
        scores = json.load(open(path))
        cells = []
        for i, col in enumerate(cols):
            w, s = cell(scores, col)
            txt = f"{w:.{decimals}f} / {s:.{decimals}f}" if s is not None else f"{w:.{decimals}f}"
            if compare:
                pw, ps = paper[key][i]
                ok = round(w, decimals) == pw and (s is None or round(s, decimals) == ps)
                if not ok:
                    n_diff += 1
                    txt += f" (paper {pw} / {ps}) **≠**"
            cells.append(txt)
        print(f"| {label} ({kind}) | " + " | ".join(cells) + " |")
    if compare:
        print(f"\n{n_diff} cell(s) differ from the paper at the published precision.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores_dir", default="results/scores")
    parser.add_argument("--compare_paper", action="store_true")
    args = parser.parse_args()
    table(args.scores_dir, "sap", SAP_COLS, PAPER_SAP, 2, args.compare_paper)
    table(args.scores_dir, "torgo", TORGO_COLS, PAPER_TORGO, 1, args.compare_paper)


if __name__ == "__main__":
    main()
