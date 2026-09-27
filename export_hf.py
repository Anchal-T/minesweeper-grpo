"""Merge a trained LoRA adapter into its base model for Hub upload."""
import argparse
import os

from common import MODEL_NAME, load_model, load_tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--ckpt", required=True, help="LoRA adapter directory, or 'base'")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = load_tokenizer(args.model)
    model = load_model("cpu", args.model,
                       adapter=args.ckpt if args.ckpt != "base" else None)
    model = model.merge_and_unload()
    model.eval()
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    with open(os.path.join(args.out, "README.md"), "w") as card:
        card.write(f"""---
tags: [reinforcement-learning, grpo, minesweeper, qwen, lora]
library_name: transformers
base_model: {args.model}
---
# Qwen Minesweeper GRPO

Qwen adapter trained to select cells on generated Minesweeper boards.
The completion format is `row,column`.

Adapter checkpoint: `{os.path.basename(args.ckpt)}`
""")
    print(f"exported {args.ckpt} -> {args.out}")


if __name__ == "__main__":
    main()
