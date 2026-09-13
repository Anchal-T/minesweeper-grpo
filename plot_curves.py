"""Plot reward curves from grpo logs into paper/figs/."""
import argparse, glob, os, re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_log(path):
    steps, rewards, bads, mines = [], [], [], []
    pat = re.compile(
        r"step\s+(\d+)\s+reward\s+(-?[\d.]+)\s+mine_or_bad\s+([\d.]+)\s+mines\s+(\d+)")
    for line in open(path):
        m = pat.search(line)
        if m:
            steps.append(int(m.group(1)))
            rewards.append(float(m.group(2)))
            bads.append(float(m.group(3)))
            mines.append(int(m.group(4)))
    return steps, rewards, bads, mines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/figs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    variants = {"a": "full SFT + KL", "b": "full SFT (resumed)", "c": "short SFT + KL",
                "d": "curriculum 2-4-6 mines"}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for v, label in variants.items():
        path = f"logs/grpo_{v}.log"
        if not os.path.exists(path):
            continue
        steps, rewards, bads, mines = parse_log(path)
        if not steps:
            continue
        ax.plot(steps, rewards, marker=".", ms=3, lw=1, label=f"grpo_{v}: {label}")
        # mark curriculum stage boundaries for d
        if v == "d" and mines:
            for i in range(1, len(mines)):
                if mines[i] != mines[i - 1]:
                    ax.axvline(steps[i], color="gray", ls=":", lw=1)
    ax.set_xlabel("GRPO step")
    ax.set_ylabel("mean rollout reward (96 samples/step)")
    ax.set_title("GRPO training reward per variant")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{args.out}/reward_curves.png", dpi=160)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for v, label in variants.items():
        path = f"logs/grpo_{v}.log"
        if not os.path.exists(path):
            continue
        steps, rewards, bads, mines = parse_log(path)
        if not steps:
            continue
        ax.plot(steps, bads, marker=".", ms=3, lw=1, label=f"grpo_{v}: {label}")
    ax.axhline(0.55, color="red", ls="--", lw=1, label="pre-fix plateau (impure reward)")
    ax.set_xlabel("GRPO step")
    ax.set_ylabel("fraction mine/waste/unparseable")
    ax.set_title("Bad-move rate per variant")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{args.out}/bad_move_rate.png", dpi=160)
    print("wrote figures to", args.out)


if __name__ == "__main__":
    main()
