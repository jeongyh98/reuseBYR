"""Convert the INT-ACT pi0-Bridge checkpoint (juexzz/INTACT-pi0-finetune-bridge) to the
lerobot 0.4.4 PI0Policy layout.

The architecture is unchanged; only the PaliGemma sub-module nesting differs between
transformers versions. The config is built from the lerobot pi0 LIBERO config with the Bridge
input/output features, and lerobot's normalization migration then writes the pre/post-processor
files.

Usage (lerobot env):
  python simpler_bridge/convert_intact_checkpoint.py --out data/pi0_bridge
"""
import argparse
import json
import os
import subprocess
import sys

from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

SRC = "juexzz/INTACT-pi0-finetune-bridge"
TEMPLATE = "lerobot/pi0_libero_finetuned_v044"


def remap_key(k: str) -> str:
    k = k.replace("paligemma.language_model.model.", "paligemma.model.language_model.")
    k = k.replace("paligemma.vision_tower.", "paligemma.model.vision_tower.")
    k = k.replace("paligemma.multi_modal_projector.", "paligemma.model.multi_modal_projector.")
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    src, tmpl = snapshot_download(SRC), snapshot_download(TEMPLATE)
    raw = args.out + "_raw"
    os.makedirs(raw, exist_ok=True)

    sd = {remap_key(k): v for k, v in load_file(f"{src}/model.safetensors").items()}
    emb = sd.get("model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight")
    if emb is not None:   # PaliGemma ties lm_head to embed_tokens; the source omits it
        sd.setdefault("model.paligemma_with_expert.paligemma.lm_head.weight", emb.clone())
    tgt = set(load_file(f"{tmpl}/model.safetensors").keys())
    print(f"missing keys: {len(tgt - set(sd))} | extra keys: {len(set(sd) - tgt)}")
    save_file(sd, f"{raw}/model.safetensors", metadata={"format": "pt"})

    cfg = json.load(open(f"{tmpl}/config.json"))
    cfg.update({
        "input_features": {"observation.images.top": {"type": "VISUAL", "shape": [3, 224, 224]},
                           "observation.state": {"type": "STATE", "shape": [7]}},
        "output_features": {"action": {"type": "ACTION", "shape": [7]}},
        "normalization_mapping": {"VISUAL": "IDENTITY", "STATE": "IDENTITY", "ACTION": "IDENTITY"},
        "chunk_size": 4, "n_action_steps": 4, "max_state_dim": 32, "max_action_dim": 32,
        "tokenizer_max_length": 72, "num_inference_steps": 10, "empty_cameras": 0,
        "compile_model": False, "gradient_checkpointing": False, "repo_id": SRC, "pretrained_path": None,
    })
    json.dump(cfg, open(f"{raw}/config.json", "w"), indent=2)

    subprocess.run([sys.executable, "-m", "lerobot.processor.migrate_policy_normalization",
                    "--pretrained-path", raw, "--output-dir", args.out], check=True)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
