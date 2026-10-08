#!/usr/bin/env python3
"""
Prepare and upload the four Kimi-Audio checkpoints of the paper to the Hugging Face Hub.

  python hf/release.py prepare --checkpoints_root <dir with exported models> --staging_dir <dir> \
      --namespace <hf user or org> --scores results/scores
  python hf/release.py upload  --staging_dir <dir> --namespace <hf user or org> [--private]
  python hf/release.py cards   --checkpoints_root <dir> --staging_dir <dir> --namespace <ns> --scores results/scores

`prepare` builds one folder per variant with symlinks to the exported weights (no copy),
the Kimi-Audio text-tokenizer files (so the repo is self-contained) and a model card
filled with the verified scores. `upload` pushes each folder with `upload_large_folder`
(resumable). Run `huggingface-cli login` first.
"""

import argparse
import glob
import json
import os
import shutil
import sys

from huggingface_hub import HfApi, snapshot_download

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "evaluation"))
from paper_tables import PAPER_SAP, PAPER_TORGO, SAP_COLS, TORGO_COLS  # noqa: E402
PLAIN_PROMPT = "Please transcribe the spoken content into written text."

VARIANTS = {
    "sft": dict(
        source="finetuned_hf_for_inference",
        repo="Kimi-Audio-7B-SAP-SFT",
        strategy="SFT: standard fine-tuning",
        summary="Kimi-Audio-7B-Instruct fine-tuned for dysarthric speech recognition with the standard transcription objective. This is the baseline of the paper: it does not use any information about the speaker's condition.",
        format="Input: the prompt `Please transcribe the spoken content into written text.` and the audio.\n\nOutput: the transcript (lowercase).",
        prompt=PLAIN_PROMPT,
        usage="",
        global_batch=64,
    ),
    "ec": dict(
        source="finetuned_hf_for_inference_mtl",
        repo="Kimi-Audio-7B-SAP-EC",
        strategy="EC: etiology classification",
        summary="Kimi-Audio-7B-Instruct fine-tuned for dysarthric speech recognition with an extra classifier that predicts the speaker's condition from the audio encoder. The classifier helps training, but its prediction is not given to the text decoder.",
        format="Input: the prompt `Please transcribe the spoken content into written text.` and the audio.\n\nOutput: the transcript. The file `etiology_classifier.pt` holds the classifier head (classes: Parkinson's Disease, ALS, Cerebral Palsy, Down Syndrome, Stroke).",
        prompt=PLAIN_PROMPT,
        usage="\nThe classifier head is loaded automatically by `kimi/infer.py` in the code repository, which saves its prediction as `pred_etiology`.\n",
        global_batch=16,
    ),
    "eh": dict(
        source="finetuned_hf_for_inference_etiology",
        repo="Kimi-Audio-7B-SAP-EH",
        strategy="EH: etiology hinting",
        summary="Kimi-Audio-7B-Instruct fine-tuned for dysarthric speech recognition with the speaker's diagnosed condition written in the prompt. The condition must be known at inference time.",
        format="Input: the prompt `Please transcribe the spoken content into written text. Consider that the speaker has been diagnosed with <etiology>.` and the audio. `<etiology>` is one of: ALS, Parkinson's Disease, Stroke, Down Syndrome, Cerebral Palsy.\n\nOutput: the transcript.",
        prompt="Please transcribe the spoken content into written text. Consider that the speaker has been diagnosed with Parkinson's Disease.",
        usage="",
        global_batch=64,
    ),
    "ep": dict(
        source="finetuned_hf_for_inference_etiology_prediction",
        repo="Kimi-Audio-7B-SAP-EP",
        strategy="**EP: etiology prediction (best)**",
        summary="**The best model of the paper (7.77% WER on SAP).** Kimi-Audio-7B-Instruct fine-tuned to first name the speaker's condition and then transcribe, in the same output. No information about the speaker is needed at inference time.",
        format="Input: the prompt `Please transcribe the spoken content into written text.` and the audio.\n\nOutput:\n\n```\nEtiology: <ALS | Parkinson's Disease | Stroke | Down Syndrome | Cerebral Palsy>\nTranscript: <transcript>\n```\n\nKeep the text after `Transcript:` as the transcription.",
        prompt=PLAIN_PROMPT,
        usage='\nTo keep only the transcription: `transcript = text.split("Transcript:", 1)[-1].strip()`\n',
        global_batch=64,
    ),
}
TOKENIZER_FILES = ["tokenizer_config.json", "tiktoken.model", "tokenization_kimia.py", "special_tokens_map.json"]
CONDITIONS = ["ALS", "Parkinson's Disease", "Stroke", "Down Syndrome", "Cerebral Palsy"]


def load_scores(scores_dir, variant):
    """Scores shown on the card: the published values of Tables 1 and 2."""
    name = "ft-kimi-audio" if variant == "sft" else f"kimi-{variant}"

    def load(ds):
        with open(os.path.join(scores_dir, ds, f"{name}.json")) as f:
            return json.load(f)
    sap, torgo = load("sap"), load("torgo")
    for scores, cols, paper in ((sap, SAP_COLS, PAPER_SAP), (torgo, TORGO_COLS, PAPER_TORGO)):
        for col, (wer, sem) in zip(cols, paper[name]):
            g = scores["overall"] if col == "overall" else scores["groups"][col]
            g["wer"], g["semscore"] = wer, sem
    return sap, torgo


def per_condition_table(sap, torgo):
    head = "| | " + " | ".join(CONDITIONS) + " | Overall |\n|---|" + "---|" * (len(CONDITIONS) + 1) + "\n"
    wer = "| SAP WER | " + " | ".join(f"{sap['groups'][c]['wer']:.2f}" for c in CONDITIONS) + f" | {sap['overall']['wer']:.2f} |\n"
    sem = "| SAP SemScore | " + " | ".join(f"{sap['groups'][c]['semscore']:.2f}" for c in CONDITIONS) + f" | {sap['overall']['semscore']:.2f} |\n"
    t_groups = ["healthy_single_word", "healthy_sentence", "non-healthy_single_word", "non-healthy_sentence"]
    t_head = "\n| TORGO (zero-shot) | Healthy words | Healthy sentences | Dysarthric words | Dysarthric sentences | Overall |\n|---|---|---|---|---|---|\n"
    t_row = "| WER / SemScore | " + " | ".join(f"{torgo['groups'][g]['wer']:.1f} / {torgo['groups'][g]['semscore']:.1f}" for g in t_groups) + \
            f" | {torgo['overall']['wer']:.1f} / {torgo['overall']['semscore']:.1f} |\n"
    out = head + wer + sem + t_head + t_row
    if "etiology_classification" in sap:
        out += f"\nEtiology prediction accuracy on SAP: {sap['etiology_classification']['accuracy']:.1f}%.\n"
    return out


def prepare(args):
    rows, scores = [], {}
    for v, meta in ([] if args.no_card else VARIANTS.items()):
        sap, torgo = load_scores(args.scores, v)
        scores[v] = (sap, torgo)
        repo_id = f"{args.namespace}/{meta['repo']}"
        rows.append(f"| [{meta['repo']}](https://huggingface.co/{repo_id}) | {meta['strategy']} | "
                    f"{sap['overall']['wer']:.2f} | {sap['overall']['semscore']:.2f} | {torgo['overall']['wer']:.1f} |")

    tok_dir = snapshot_download("moonshotai/Kimi-Audio-7B-Instruct", allow_patterns=TOKENIZER_FILES)
    template = open(os.path.join(HERE, "model_card.md")).read()

    for v, meta in VARIANTS.items():
        src = os.path.join(args.checkpoints_root, meta["source"])
        dst = os.path.join(args.staging_dir, meta["repo"])
        # Real directories + per-file symlinks, so that the uploader walks every file
        for f in sorted(glob.glob(os.path.join(src, "**", "*"), recursive=True)):
            if os.path.isdir(f):
                continue
            link = os.path.join(dst, os.path.relpath(f, src))
            os.makedirs(os.path.dirname(link), exist_ok=True)
            if not os.path.lexists(link):
                os.symlink(os.path.abspath(f), link)
        for f in TOKENIZER_FILES:
            shutil.copy(os.path.join(tok_dir, f), dst)

        if args.no_card:
            print(f"Prepared {dst} (weights and tokenizer only)")
            continue
        sap, torgo = scores[v]
        card = template.format(
            repo_name=meta["repo"], repo_id=f"{args.namespace}/{meta['repo']}",
            summary=meta["summary"], format_description=meta["format"], prompt=meta["prompt"],
            usage_note=meta["usage"], global_batch=meta["global_batch"], sap_wer=round(sap["overall"]["wer"], 2),
            variants_table="\n".join(rows), per_condition_table=per_condition_table(sap, torgo),
        )
        with open(os.path.join(dst, "README.md"), "w") as f:
            f.write(card)
        print(f"Prepared {dst}")


def upload(args):
    api = HfApi()
    for meta in VARIANTS.values():
        repo_id = f"{args.namespace}/{meta['repo']}"
        folder = os.path.join(args.staging_dir, meta["repo"])
        api.create_repo(repo_id, repo_type="model", private=args.private, exist_ok=True)
        api.upload_large_folder(repo_id=repo_id, folder_path=folder, repo_type="model")
        print(f"Uploaded {repo_id}")


def cards(args):
    """Write the model cards and push only README.md (weights already uploaded)."""
    args.no_card = False
    prepare(args)
    api = HfApi()
    for meta in VARIANTS.values():
        repo_id = f"{args.namespace}/{meta['repo']}"
        api.upload_file(path_or_fileobj=os.path.join(args.staging_dir, meta["repo"], "README.md"),
                        path_in_repo="README.md", repo_id=repo_id, repo_type="model",
                        commit_message="Add model card")
        print(f"Updated card of {repo_id}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--checkpoints_root", required=True)
    p.add_argument("--staging_dir", required=True)
    p.add_argument("--namespace", required=True)
    p.add_argument("--scores", default="results/scores")
    p.add_argument("--no_card", action="store_true", help="Skip the model card (weights and tokenizer only)")
    c = sub.add_parser("cards", help="write model cards and upload only README.md")
    c.add_argument("--checkpoints_root", required=True)
    c.add_argument("--staging_dir", required=True)
    c.add_argument("--namespace", required=True)
    c.add_argument("--scores", default="results/scores")
    u = sub.add_parser("upload")
    u.add_argument("--staging_dir", required=True)
    u.add_argument("--namespace", required=True)
    u.add_argument("--private", action="store_true")
    args = parser.parse_args()
    {"prepare": prepare, "cards": cards, "upload": upload}[args.cmd](args)


if __name__ == "__main__":
    main()
