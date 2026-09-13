# minesweeper-grpo

GRPO-tune a small LLM (GPT-2 124M) to specialize in solving minesweeper.
Everything runs locally on the K80 box (torch 1.13.1+cu117, fp16 autocast,
eager attention — no TRL / vLLM / torch 2.x possible on sm_37).

## Setup

```bash
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python torch==1.13.1+cu117 \
    --index-url https://download.pytorch.org/whl/cu117
uv pip install --python .venv/bin/python transformers==4.36.2 numpy==1.26.4
```

First run downloads GPT-2 into `hf-cache/` (~500 MB).

## Task

Board text state (6x6, 6 mines):

```
Minesweeper 6x6 grid, 6 mines. '.' = hidden, digit = adjacent mines.
1 . . 0 0 1
...
Which hidden cell is safe? Answer with row,col (0-indexed). Answer:
```

Model completes `r,c`. Verifiable rewards (env.py `step_reward`):
mine / off-board / unparseable = **-1.0**, wasted click on open cell = **-0.5**,
safe click = **0.5 + 0.5 × (cells revealed by flood fill)** — dense signal that
prefers informative safe moves.

## Pipeline

1. **SFT warmup** (`sft.py`): behavior-clone the ground-truth expert (safe cell
   with max flood-fill reveal) on random mid-game states. Gives the policy the
   output format + basic play so GRPO explores meaningfully instead of from noise.
2. **GRPO** (`grpo.py`): group-relative advantages (G=8 samples per state),
   policy gradient on answer tokens; PPO-style clipping and a KL-to-reference
   option are included for multi-epoch updates.
3. **Eval** (`eval.py`): greedy full-game play, win rate + cleared fraction.

## Run

```bash
.venv/bin/python sft.py --steps 1500          # ~30 min on one K80
.venv/bin/python eval.py --ckpt sft.pt
.venv/bin/python grpo.py --init-from sft.pt --steps 3000
.venv/bin/python eval.py --ckpt runs/grpo/ckpt_3000.pt
```

Pick the freest GPU with `CUDA_VISIBLE_DEVICES=<n>`; box is shared (see nvidia-smi).
