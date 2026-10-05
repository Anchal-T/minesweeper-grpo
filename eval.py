"""Evaluation on CAST's held-out boards and turn budgets.

Follows the reference protocol (github.com/Wloner0809/CAST): the same 200
held-out boards per difficulty tier, `rollouts` independent samples per board at
T=0.6 / top-p 0.95, one turn per action with invalid actions still costing a
turn, and a turn budget after which the game scores as a loss. Each tier also
reports the posterior solver's ceiling on the same boards, so a policy number
can be read against the best achievable one. Actions stay our ``row,col``
format (CAST's agent emits ``reveal R C``); boards, budgets, and sampling are
CAST's.
"""
import argparse
import hashlib
import json
import math

import numpy as np

from common import MODEL_NAME, load_model, load_tokenizer, sample_completions
from env import (Minesweeper, generate_layout, mine_posterior, parse_move,
                 posterior_move)

# CAST difficulty tiers: rows, cols, mines, turn budget. CAST's environment
# config allows 36/49 steps; the paper's stricter 30/40 is used here.
TIERS = {"id": (6, 6, 7, 30), "unseen": (7, 7, 10, 40)}
CAST_TEST_SIZE = 200       # held-out boards per tier in the reference protocol
CAST_TRAIN_SIZE = 20000    # train boards the held-out boards are deduplicated against
CAST_WIN_RATE = {"id": 44.7, "unseen": 11.0}  # CAST with Qwen3-4B, for reference


def _puzzle_hash(layout, rows, cols):
    mines, first_click = layout
    canonical = json.dumps({"m": [list(m) for m in mines], "fc": list(first_click),
                            "r": rows, "c": cols}, sort_keys=True)
    return hashlib.md5(canonical.encode()).hexdigest()


def cast_layouts(tier, count=CAST_TEST_SIZE):
    """CAST's held-out board layouts for a tier, in its own order.

    Replays its dataset prep: ``np.random.seed(42)``, seeds drawn without
    replacement from ``[0, 1_000_000)``, the first 20k to train and the next 200
    to test, with any test board whose puzzle (mine layout + first click)
    already appears in train or test replaced by a spare seed. ``count``
    truncates the fixed 200-board set rather than re-drawing it.
    """
    rows, cols, mines = TIERS[tier][:3]
    rng = np.random.RandomState(42)
    total = CAST_TRAIN_SIZE + CAST_TEST_SIZE
    pool = min(1_000_000, max(total * 3, 100_000))
    seeds = rng.choice(1_000_000, size=pool, replace=False)

    def puzzle(seed):
        return _puzzle_hash(generate_layout(int(seed), rows, cols, mines), rows, cols)

    train_hashes, spare = set(), total
    for seed in seeds[:CAST_TRAIN_SIZE]:
        digest = puzzle(seed)
        while digest in train_hashes and spare < len(seeds):
            digest = puzzle(seeds[spare])
            spare += 1
        train_hashes.add(digest)
    layouts, seen = [], set()
    for seed in seeds[CAST_TRAIN_SIZE:total]:
        layout = generate_layout(int(seed), rows, cols, mines)
        digest = _puzzle_hash(layout, rows, cols)
        while (digest in train_hashes or digest in seen) and spare < len(seeds):
            layout = generate_layout(int(seeds[spare]), rows, cols, mines)
            spare += 1
            digest = _puzzle_hash(layout, rows, cols)
        layouts.append(layout)
        seen.add(digest)
    return layouts[:count]


def new_game(tier, layout):
    rows, cols, mines, _ = TIERS[tier]
    board = Minesweeper(rows, cols, mines, mine_positions=layout[0],
                        first_click=layout[1])
    # CAST ends (and wins) a board the first click already clears.
    return {"board": board, "turns": 0, "done": board.solved(),
            "win": board.solved()}


def apply_turn(game, text, turn_limit):
    """Play one completion. Invalid actions still cost the turn, as in CAST.

    Returns (had_provably_safe_cell, move_was_provably_safe) for the state the
    model saw, which is the single-move accuracy a policy is judged on.
    """
    board = game["board"]
    game["turns"] += 1
    posterior = mine_posterior(board)
    move = parse_move(text)
    safe_position = min(posterior.values()) < 1e-12
    safe_action = move in posterior and posterior[move] < 1e-12
    if move in posterior:  # hidden, on-board: the only actions that reach the board
        _, hit_mine = board.click(*move)
        if hit_mine:
            game["done"] = True
        elif board.solved():
            game["done"] = game["win"] = True
    if game["turns"] >= turn_limit:
        game["done"] = True  # out of turns: scored as a loss
    return safe_position, safe_action


def solve(layout, tier):
    """Play the posterior solver greedily; True if it clears the board."""
    board = new_game(tier, layout)["board"]
    while not board.solved():
        move = posterior_move(board)
        if move is None or board.is_mine(*move):
            break
        board.click(*move)
    return board.solved()


def solver_ceiling(layouts, tier):
    return sum(solve(layout, tier) for layout in layouts) / len(layouts)


def eval_tier(model, tok, layouts, tier, args):
    turn_limit = TIERS[tier][3]
    games = [new_game(tier, layout)
             for layout in layouts for _ in range(args.rollouts)]
    safe_positions = safe_actions = 0
    for start in range(0, len(games), args.batch_size):
        active = [g for g in games[start:start + args.batch_size]
                  if not g["done"]]
        while active:
            for offset in range(0, len(active), args.batch_size):
                batch = active[offset:offset + args.batch_size]
                texts, _ = sample_completions(
                    model, tok, [g["board"].prompt() for g in batch],
                    max_new_tokens=6, temperature=args.temperature,
                    top_p=args.top_p)
                for game, text in zip(batch, texts):
                    had_safe_cell, safe_action = apply_turn(game, text, turn_limit)
                    safe_positions += had_safe_cell
                    safe_actions += safe_action
            active = [g for g in active if not g["done"]]
    wins = [g["win"] for g in games]
    win_rate = sum(wins) / len(wins)
    board_means = [sum(wins[i * args.rollouts:(i + 1) * args.rollouts]) / args.rollouts
                   for i in range(len(layouts))]
    stdev = math.sqrt(sum((m - win_rate) ** 2 for m in board_means) / max(1, len(layouts) - 1))
    return {"win_rate": win_rate,
            "ci": 1.96 * stdev / math.sqrt(len(layouts)),
            "safe_positions": safe_positions, "safe_actions": safe_actions}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--ckpt", default="runs/sft/last",
                    help="LoRA adapter directory, or 'base' for the pretrained model")
    ap.add_argument("--tiers", default="id,unseen")
    ap.add_argument("--boards", type=int, default=CAST_TEST_SIZE,
                    help="boards per tier; the first N of CAST's fixed 200")
    ap.add_argument("--rollouts", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--batch-size", type=int, default=64,
                    help="games sampled per forward; lower it on small GPUs")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    tok = load_tokenizer(args.model)
    model = load_model(args.device, args.model,
                       adapter=None if args.ckpt == "base" else args.ckpt)
    model.eval()
    print(f"[EVAL] ckpt={args.ckpt} rollouts={args.rollouts} "
          f"temperature={args.temperature} top_p={args.top_p}", flush=True)
    for tier in [t.strip() for t in args.tiers.split(",") if t.strip()]:
        layouts = cast_layouts(tier, args.boards)
        result = eval_tier(model, tok, layouts, tier, args)
        ceiling = solver_ceiling(layouts, tier)
        accuracy = result["safe_actions"] / max(1, result["safe_positions"])
        print(f"[EVAL:{tier}] boards={len(layouts)} rollouts={args.rollouts} "
              f"turn_limit={TIERS[tier][3]} win_rate={result['win_rate']:.3f} "
              f"+/-{result['ci']:.3f} solver_ceiling={ceiling:.3f} "
              f"safe_move_accuracy={accuracy:.3f} "
              f"({result['safe_actions']}/{result['safe_positions']}) "
              f"cast_published={CAST_WIN_RATE[tier]}", flush=True)


if __name__ == "__main__":
    main()
