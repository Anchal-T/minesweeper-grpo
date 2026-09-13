"""Export a training checkpoint to a Hugging Face upload directory.

Usage:
  .venv/bin/python export_hf.py --ckpt runs/grpo_c/ckpt_600.pt --out hf-out/minesweeper-grpo-gpt2

Creates <out>/ with config.json, model.safetensors (fp32), tokenizer files —
ready for: huggingface-cli upload <repo-id> hf-out/minesweeper-grpo-gpt2 .
"""
import argparse, os

from common import load_tokenizer, load_model, MODEL_NAME


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help=".pt state dict (or 'gpt2' for base)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = load_tokenizer()
    model = load_model("cpu")
    if args.ckpt != "gpt2":
        model.load_state_dict(torch_load(args.ckpt))
    model.eval()

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    card = os.path.join(args.out, "README.md")
    with open(card, "w") as f:
        f.write(f"""---
tags: [reinforcement-learning, grpo, minesweeper, gpt2]
library_name: transformers
base_model: {MODEL_NAME}
---
# GPT-2 fine-tuned for minesweeper with GRPO

GPT-2 124M SFT-warmed then GRPO-trained (group-relative policy optimization,
from-scratch trainer for torch 1.13) to pick safe cells on 6x6 minesweeper
boards with 6 mines. Prompt format: board grid + 'Answer:'; completion 'r,c'.

Checkpoint: `{os.path.basename(args.ckpt)}`
""")
    print(f"exported {args.ckpt} -> {args.out}")


def torch_load(path):
    import torch
    return torch.load(path, map_location="cpu")


if __name__ == "__main__":
    main()
