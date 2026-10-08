"""
Pre-compute Kimi-Audio semantic audio tokens (GLM-4-Voice tokenizer) for a training
manifest. Utterances whose audio cannot be tokenized are dropped (4 utterances of the
SAP training set in our runs).

Adapted from Kimi-Audio's finetune_codes/extract_semantic_codes.py.
"""

import argparse
import json
import os

import tqdm
from huggingface_hub import snapshot_download
from transformers import AutoConfig

from kimia_infer.api.prompt_manager import KimiAPromptManager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name_or_path", default="moonshotai/Kimi-Audio-7B-Instruct")
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()

    cache_path = args.model_name_or_path if os.path.exists(args.model_name_or_path) else snapshot_download(args.model_name_or_path)
    config = AutoConfig.from_pretrained(cache_path, trust_remote_code=True)
    prompt_manager = KimiAPromptManager(
        model_path=cache_path,
        kimia_token_offset=config.kimia_token_offset,
        kimia_text_audiodelaytokens=config.kimia_mimo_audiodelaytokens,
    )

    with open(args.input_file) as f:
        lines = f.readlines()

    kept = 0
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w") as f_out:
        for line in tqdm.tqdm(lines):
            data = json.loads(line)
            ok = True
            for msg in data["conversation"]:
                if msg["message_type"] == "audio":
                    try:
                        msg["audio_tokens"] = prompt_manager._tokenize_audio(msg["content"])
                    except Exception as e:
                        print(f"Skipping {msg['content']}: {e}")
                        ok = False
                    if ok and not msg["audio_tokens"]:
                        ok = False
            if ok:
                f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                kept += 1
    print(f"Kept {kept}/{len(lines)} utterances -> {args.output_file}")


if __name__ == "__main__":
    main()
