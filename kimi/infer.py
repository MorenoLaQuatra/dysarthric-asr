"""
Multi-GPU inference with an exported Kimi-Audio model (SFT, EC, EH or EP).

The user prompt is taken from the manifest, so the same script serves every variant:
  * SFT / EP manifests carry the plain transcription prompt;
  * EH manifests carry the etiology-hinted prompt.
For TORGO with an EH model, --torgo_eh_hints replaces the prompt with the binary
healthy / neurological-condition hint used in the paper (TORGO has no per-speaker etiology).

If the model folder contains etiology_classifier.pt (EC), the etiology is predicted by
the encoder head and stored in `pred_etiology`; the decoder output is the transcript.

Output: one JSON line per utterance with the reference (`text`), the prediction
(`pred_text`) and the metadata needed by evaluation/evaluate.py.
"""

import argparse
import json
import multiprocessing as mp
import os
import tempfile

import librosa
import torch
from tqdm import tqdm

from ec_model import CLASSIFIER_FILENAME, IDX_TO_ETIOLOGY, build_etiology_classifier

DEFAULT_PROMPT = "Please transcribe the spoken content into written text."
TORGO_HEALTHY_HINT = DEFAULT_PROMPT + " Consider that the speaker is a healthy individual."
TORGO_DYSARTHRIC_HINT = DEFAULT_PROMPT + " Consider that the speaker has a neurological condition causing speech impairment."


def parse_entry(entry, base_dir=None):
    prompt, audio, reference = DEFAULT_PROMPT, None, ""
    for msg in entry["conversation"]:
        if msg.get("message_type") == "audio":
            audio = msg["content"]
        elif msg.get("role") == "user" and msg.get("message_type") == "text":
            prompt = msg["content"]
        elif msg.get("role") == "assistant" and msg.get("message_type") == "text":
            reference = msg["content"]
    if base_dir and not os.path.isabs(audio):
        audio = os.path.join(base_dir, audio)
    return prompt, audio, reference


def worker(gpu_id, entries, args, out_path):
    from kimia_infer.api.kimia import KimiAudio
    from kimia_infer.models.tokenizer.whisper_Lv3.whisper import WhisperEncoder

    torch.cuda.set_device(gpu_id)
    model = KimiAudio(model_path=args.model_path, load_detokenizer=False)
    sampling = dict(
        audio_temperature=0.8, audio_top_k=10, audio_repetition_penalty=1.0, audio_repetition_window_size=64,
        text_temperature=0.0, text_top_k=5, text_repetition_penalty=1.0, text_repetition_window_size=16,
    )

    classifier = None
    cls_path = os.path.join(args.model_path, CLASSIFIER_FILENAME)
    if os.path.exists(cls_path):
        encoder = WhisperEncoder(os.path.join(args.model_path, "whisper-large-v3"), mel_batch_size=20, unfreeze_online_whisper_model=False)
        encoder = encoder.to(torch.cuda.current_device()).to(torch.bfloat16).eval()
        classifier = build_etiology_classifier(len(IDX_TO_ETIOLOGY))
        classifier.load_state_dict(torch.load(cls_path, map_location="cpu"))
        classifier = classifier.to(torch.cuda.current_device()).to(torch.bfloat16).eval()

    with open(out_path, "w", encoding="utf-8") as fout:
        for entry in tqdm(entries, desc=f"GPU {gpu_id}", position=gpu_id):
            prompt, audio, reference = parse_entry(entry, args.base_dir)
            if not audio or not os.path.exists(audio):
                print(f"GPU {gpu_id}: missing audio {audio}")
                continue
            etiology = entry.get("etiology", "unknown")
            if args.torgo_eh_hints:
                prompt = TORGO_HEALTHY_HINT if str(etiology).lower() == "healthy" else TORGO_DYSARTHRIC_HINT

            messages = [
                {"role": "user", "message_type": "text", "content": prompt},
                {"role": "user", "message_type": "audio", "content": audio},
            ]
            try:
                _, text = model.generate(messages, **sampling, output_type="text")
                pred = text.strip() if text else ""
            except Exception as e:  # keep going on single-utterance failures
                print(f"GPU {gpu_id}: generation failed for {audio}: {e}")
                pred = ""

            result = {
                "audio_filepath": audio,
                "text": reference,
                "pred_text": pred,
                "etiology": etiology,
                "duration": round(librosa.get_duration(path=audio), 2),
            }
            for key in ("speech_type", "transcript"):
                if key in entry:
                    result[key] = entry[key]

            if classifier is not None:
                wav = torch.tensor(librosa.load(audio, sr=16000)[0]).unsqueeze(0).to(torch.cuda.current_device())
                with torch.no_grad():
                    feats = encoder(wav)
                    feats = feats.reshape(feats.shape[0], int(feats.shape[1] // 4), feats.shape[2] * 4)
                    logits = classifier(feats.mean(dim=1).to(torch.bfloat16))
                result["pred_etiology"] = IDX_TO_ETIOLOGY[logits.argmax(dim=-1).item()]

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_path", required=True, help="Exported model folder or Hugging Face repo id")
    parser.add_argument("--manifest", required=True, help="Test manifest (conversation JSONL)")
    parser.add_argument("--output", required=True, help="Output JSONL")
    parser.add_argument("--base_dir", default=None, help="Prefix for relative audio paths in the manifest")
    parser.add_argument("--num_gpus", type=int, default=torch.cuda.device_count())
    parser.add_argument("--torgo_eh_hints", action="store_true", help="Binary healthy/dysarthric hints (EH on TORGO)")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        from huggingface_hub import snapshot_download
        args.model_path = snapshot_download(args.model_path)

    with open(args.manifest, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    print(f"{len(entries)} utterances, {args.num_gpus} GPUs")

    chunk = (len(entries) + args.num_gpus - 1) // args.num_gpus
    tmp_dir = tempfile.mkdtemp(prefix="kimi_infer_")
    parts = [os.path.join(tmp_dir, f"part_{i}.jsonl") for i in range(args.num_gpus)]

    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=worker, args=(i, entries[i * chunk:(i + 1) * chunk], args, parts[i])) for i in range(args.num_gpus)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    failed = [i for i, p in enumerate(procs) if p.exitcode != 0]
    if failed:
        raise RuntimeError(f"Workers {failed} failed; partial outputs in {tmp_dir}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    n = 0
    with open(args.output, "w", encoding="utf-8") as fout:
        for part in parts:
            with open(part, encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)
                    n += 1
            os.remove(part)
    os.rmdir(tmp_dir)
    print(f"Wrote {n} predictions to {args.output}")


if __name__ == "__main__":
    main()
