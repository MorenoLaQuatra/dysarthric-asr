"""
Merge a LoRA checkpoint into Kimi-Audio and export it in the format expected by
kimia_infer.api.kimia.KimiAudio (LM shards + whisper-large-v3/ encoder folder).

If --checkpoint points to a training run directory, the checkpoint with the
lowest validation loss (trainer_state.json) is selected, which is how the
paper's models were chosen.

The EC head (etiology_classifier.pt) is exported as well when present.
"""

import argparse
import glob
import json
import os
import shutil

import finetune_codes
import torch
from finetune_codes.model import KimiAudioModel
from finetune_codes.modeling_kimia import MoonshotKimiaForCausalLM
from kimia_infer.models.tokenizer.whisper_Lv3.whisper import WhisperModel
from peft import PeftModel

from ec_model import CLASSIFIER_FILENAME, NUM_ETIOLOGY_CLASSES, KimiAudioModelEC


def select_best_checkpoint(run_dir):
    """Return the checkpoint-* directory with the lowest eval_loss."""
    states = sorted(glob.glob(os.path.join(run_dir, "checkpoint-*", "trainer_state.json")),
                    key=lambda p: int(p.split("checkpoint-")[-1].split(os.sep)[0]))
    if not states:
        raise FileNotFoundError(f"No checkpoint-*/trainer_state.json in {run_dir}")
    # The last checkpoint holds the complete log history
    with open(states[-1]) as f:
        history = json.load(f)["log_history"]
    evals = [e for e in history if "eval_loss" in e]
    for e in evals:
        print(f"  epoch {e['epoch']:.0f}  step {e['step']}  eval_loss {e['eval_loss']:.4f}")
    best = min(evals, key=lambda e: e["eval_loss"])
    path = os.path.join(run_dir, f"checkpoint-{best['step']}")
    print(f"Selected {path} (eval_loss {best['eval_loss']:.4f})")
    return path


def export(checkpoint, base_model_path, output_dir, tokenizer_source):
    is_ec = os.path.exists(os.path.join(checkpoint, CLASSIFIER_FILENAME))

    print(f"Loading base model from {base_model_path} ({'EC' if is_ec else 'standard'} variant)")
    if is_ec:
        model = KimiAudioModelEC.from_pretrained(base_model_path, num_etiology_classes=NUM_ETIOLOGY_CLASSES)
        model.etiology_classifier.load_state_dict(torch.load(os.path.join(checkpoint, CLASSIFIER_FILENAME), map_location="cpu"))
    else:
        model = KimiAudioModel.from_pretrained(base_model_path)

    print(f"Merging LoRA adapters from {checkpoint}")
    model = PeftModel.from_pretrained(model, checkpoint).merge_and_unload()
    state_dict = model.state_dict()

    # 1. Language model
    os.makedirs(output_dir, exist_ok=True)
    lm = MoonshotKimiaForCausalLM(model.config)
    lm.load_state_dict({k: v for k, v in state_dict.items() if not k.startswith(("whisper_model", "etiology_classifier"))})
    lm.save_pretrained(output_dir)
    code_dir = os.path.dirname(finetune_codes.__file__)
    shutil.copyfile(os.path.join(code_dir, "configuration_moonshot_kimia.py"), os.path.join(output_dir, "configuration_moonshot_kimia.py"))
    shutil.copyfile(os.path.join(code_dir, "modeling_kimia.py"), os.path.join(output_dir, "modeling_moonshot_kimia.py"))

    # 2. Whisper encoder (LoRA also adapts its attention projections)
    whisper = WhisperModel.from_pretrained("openai/whisper-large-v3")
    encoder_sd = {k.replace("speech_encoder.", "encoder."): v for k, v in model.whisper_model.state_dict().items() if k.startswith("speech_encoder")}
    missing, unexpected = whisper.load_state_dict(encoder_sd, strict=False)
    assert not unexpected, f"Unexpected keys: {unexpected}"
    assert all(k.startswith("decoder") for k in missing), f"Missing encoder keys: {missing}"
    whisper.save_pretrained(os.path.join(output_dir, "whisper-large-v3"))

    # 3. EC head
    if is_ec:
        torch.save(model.etiology_classifier.state_dict(), os.path.join(output_dir, CLASSIFIER_FILENAME))

    # 4. Text tokenizer, so that the exported folder is self-contained
    if tokenizer_source:
        for pattern in ("tokenizer_config.json", "tiktoken.model", "tokenization_*.py", "special_tokens_map.json"):
            for f in glob.glob(os.path.join(tokenizer_source, pattern)):
                shutil.copy(f, output_dir)

    print(f"Exported to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="LoRA checkpoint dir, or training run dir (best checkpoint is selected)")
    parser.add_argument("--base_model_path", required=True, help="Kimi-Audio in training format (scripts/init_model.sh)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tokenizer_source", default=None,
                        help="Local snapshot of moonshotai/Kimi-Audio-7B-Instruct to copy the text tokenizer from")
    args = parser.parse_args()

    checkpoint = args.checkpoint
    if not os.path.exists(os.path.join(checkpoint, "adapter_config.json")):
        checkpoint = select_best_checkpoint(checkpoint)
    export(checkpoint, args.base_model_path, args.output_dir, args.tokenizer_source)


if __name__ == "__main__":
    main()
