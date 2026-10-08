"""Patch a fine-tuned Phi-4-multimodal checkpoint so that its processor config loads with recent transformers."""
import json
import os
import argparse
from shutil import move

def update_preprocessor_config(path):
    config_path = os.path.join(path, "preprocessor_config.json")
    if not os.path.isfile(config_path):
        print(f"❌ File not found: {config_path}")
        return

    with open(config_path, "r") as f:
        config = json.load(f)

    # Renaming fields if they exist
    rename_map = {
        "compression_rate": "audio_compression_rate",
        "feat_stride": "audio_feat_stride",
        "qformer_compression_rate": "audio_downsample_rate"
    }

    for old_key, new_key in rename_map.items():
        if old_key in config:
            config[new_key] = config.pop(old_key)

    # Removing specific fields
    for key in ["padding_value", "feature_size", "sampling_rate"]:
        config.pop(key, None)

    # Save the updated file
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Updated {config_path}")

def rename_chat_template(path):
    old = os.path.join(path, "chat_template.json")
    new = os.path.join(path, "old_chat_template.json")
    if os.path.isfile(old):
        move(old, new)
        print(f"✅ Renamed {old} → {new}")
    else:
        print(f"⚠️  No chat_template.json found to rename.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_folder", required=True, help="Path to checkpoint folder")
    args = parser.parse_args()

    update_preprocessor_config(args.ckpt_folder)
    rename_chat_template(args.ckpt_folder)

if __name__ == "__main__":
    main()
