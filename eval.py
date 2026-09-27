"""Batched greedy evaluation against a posterior-greedy benchmark on same seeds."""
import argparse

from common import MODEL_NAME, load_model, load_tokenizer, sample_completions
from env import Minesweeper, parse_move, posterior_move


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--ckpt", default="runs/sft/last",
                    help="LoRA adapter directory, or 'base' for the pretrained model")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-moves", type=int, default=None,
                    help="optional policy-game move limit for smoke checks")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    tok = load_tokenizer(args.model)
    model = load_model(args.device, args.model,
                       adapter=args.ckpt if args.ckpt != "base" else None)
    model.eval()
    boards = [Minesweeper(seed=i) for i in range(args.games)]
    moves = [0] * args.games
    active = list(range(args.games))
    while active:
        batch = active[:args.batch_size]
        prompts = [boards[i].prompt() for i in batch]
        texts, _ = sample_completions(
            model, tok, prompts, max_new_tokens=6,
            temperature=max(args.temperature, 1e-3),
            greedy=args.temperature <= 0)
        next_active = active[args.batch_size:]
        for i, text in zip(batch, texts):
            move = parse_move(text)
            if move is None:
                continue
            _, hit_mine = boards[i].click(*move)
            moves[i] += 1
            move_limit = args.max_moves or boards[i].w * boards[i].h
            if not hit_mine and not boards[i].solved() and moves[i] < move_limit:
                next_active.append(i)
        active = next_active

    win = sum(b.solved() for b in boards)
    clear = sum(len(b.revealed) / (b.w * b.h - b.n_mines) for b in boards)
    oracle_win = oracle_clear = 0.0
    for seed in range(args.games):
        board = Minesweeper(seed=seed)
        while not board.solved():
            move = posterior_move(board)
            if move is None:
                break
            _, hit_mine = board.click(*move)
            if hit_mine:
                break
        oracle_win += float(board.solved())
        oracle_clear += len(board.revealed) / (board.w * board.h - board.n_mines)
    mode = "greedy" if args.temperature <= 0 else f"T={args.temperature}"
    print(f"ckpt={args.ckpt} [{mode}] games={args.games} "
          f"win_rate={win/args.games:.3f} avg_clear_fraction={clear/args.games:.3f}")
    print(f"posterior_greedy same seeds: win_rate={oracle_win/args.games:.3f} "
          f"avg_clear_fraction={oracle_clear/args.games:.3f}")


if __name__ == "__main__":
    main()
