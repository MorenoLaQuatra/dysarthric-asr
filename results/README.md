# Scores

`scores/{sap,torgo}/<system>.json` hold the aggregate results behind every row of Tables 1 and 2 of the
paper, as written by `evaluation/evaluate.py --semscore`: overall WER and SemScore, the same per group
(SAP: etiology; TORGO: healthy/dysarthric x single words/sentences) and, for EC and EP, the etiology
classification results.

System names: `zs-*` zero-shot, `ft-*` fine-tuned on SAP, `kimi-{ec,eh,ep}` the etiology-aware variants
(`ft-kimi-audio` is Kimi SFT).

Per-utterance SAP predictions are not released, since they contain SAP transcripts and file names covered
by the SAP data-use agreement. The scripts in this repository write them to `results/predictions/`
(not tracked by git), and you can score them with:

```bash
for ds in sap torgo; do
  for f in results/predictions/$ds/*.jsonl; do
    python evaluation/evaluate.py --predictions $f --dataset $ds --semscore --output results/scores/$ds/$(basename $f .jsonl).json
  done
done
```

Rebuild the tables (optionally side by side with the published numbers):

```bash
python evaluation/paper_tables.py --scores_dir results/scores --compare_paper
```
