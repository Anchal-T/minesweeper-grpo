"""Evaluate a checkpoint by playing full minesweeper games greedily."""
import argparse, random

from common import load_tokenizer, load_model, sample_completions
from env import Minesweeper, parse_move, play_episode


def greedy_move_fn(model, tok):
    def fn(prompt):
        texts, _ = sample_completions(model, tok, [prompt],
                                      max_new_tokens=8, greedy=True)
        return texts[0]
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="sft.pt", help="'gpt2' for raw pretrained")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    tok = load_tokenizer()
    model = load_model(args.device)
    if args.ckpt != "gpt2":
        model.load_state_dict(torch_load(args.ckpt))
    model.eval()

    fn = greedy_move_fn(model, tok)
    rng = random.Random(0)
    wins = cleared = 0
    for i in range(args.games):
        b = Minesweeper()
        res = play_episode(b, fn)
        wins += res["solved"]
        cleared += res["cleared"] / res["total_safe"]
    print(f"ckpt={args.ckpt}  games={args.games}  win_rate={wins/args.games:.3f}  "
          f"avg_clear_fraction={cleared/args.games:.3f}")


def torch_load(path):
    import torch
    return torch.load(path, map_location="cpu")


if __name__ == "__main__":
    main()
