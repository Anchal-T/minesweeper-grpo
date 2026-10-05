# Minesweeper GRPO on Qwen

Train a Qwen2.5-0.5B-Instruct policy to select moves on procedurally
generated Minesweeper boards. Boards, rendering, and evaluation follow
[CAST](https://github.com/Wloner0809/CAST), so a number produced here is
comparable to CAST's published table. The pipeline uses LoRA, posterior-oracle
SFT, and single-move GRPO rewards; training draws positions from
solver-played games instead of only openings, because evaluation plays whole
games. Checkpoints include the adapter, optimizer, gradient scaler, step, and
RNG state, with optional private Hugging Face Hub sync through `HF_TOKEN` and
`HF_REPO_ID`.

## Install

Use a recent PyTorch with CUDA support, then install:

```bash
python -m pip install transformers peft huggingface_hub
```

The base model defaults to `Qwen/Qwen2.5-0.5B-Instruct`. The tokenizer chat
template formats board prompts; completions are `row,column`.

## Train

```bash
python sft.py --target posterior --steps 600 --out runs/sft/last
python grpo.py --init-from runs/sft/last --reward posterior \
  --adv-norm none --kl-coef 0 --steps 1000 --out-dir runs/grpo
python eval.py --ckpt runs/grpo/last
```

Use `--resume` to restore the latest local checkpoint or download it from the
Hub when `--hub-repo` (or `HF_REPO_ID`) is set. `--time-budget-min` saves a
resumable checkpoint before exiting. At every save `sft.py` also prints
T=1 single-move accuracy and mean answer entropy on a fixed held-out set of
game positions (`--eval-states`, 0 disables), which is what checkpoints are
compared on. GRPO supports `--reward truth|posterior|vpr`,
`--adv-norm std|none`, and `--kl-est k3|k2`. For the optional rationale
format, use `sft.py --answer-format cot` followed by GRPO with
`--answer-format cot --prompt-mean-loss --max-new-tokens 160`; set
`--format-reward` to reward the required `<think>...</think> Answer: r,c` structure.

`--kl-coef` is the ablation knob: `kaggle/run.sh` runs 0 and 0.05 side by side
(see below). Over steps 21-40 of a matched 50-step ablation from the same SFT
start, KL 0 reached mean posterior reward -0.23 against -0.31 for KL 0.05, with
lower answer entropy (0.75 against 0.82) and fewer bad or mine moves (0.56
against 0.65), and neither arm produced low-variance groups (0-3% of groups).
So 0 is the default above: on a clean posterior reward the KL term only slowed
learning, and neither arm collapsed the way the paper reports for KL 0.05
under its older, buggy reward.

## Evaluation protocol

`eval.py` replays CAST's protocol rather than a looser local benchmark:

- The same 200 held-out boards per tier, reproduced from CAST's dataset prep:
  `np.random.seed(42)`, seeds drawn without replacement from `[0, 1_000_000)`,
  the first 20k to train and the next 200 to test, with puzzle deduplication
  against the training set. Every run scores the same positions.
- `--rollouts 4` independent samples per board at `--temperature 0.6` and
  `--top-p 0.95` (Avg@4), one turn per action; an unparseable, off-board, or
  already-revealed action still costs a turn.
- Turn budgets of 30 for 6x6/7 mines and 40 for 7x7/10 mines, per the paper;
  running out of turns scores a loss.
- Reported next to the win rate: a board-level 95% CI, the posterior solver's
  ceiling on the same boards, and single-move accuracy on positions with a
  provably safe cell.

Reference values from CAST (Qwen3-4B): 44.7% in-distribution, 11.0% unseen.
Run `python eval.py --ckpt <dir>` for both tiers, or
`--tiers id --boards 40` for a quick check.

## Kaggle

`kaggle/run.sh` runs posterior-target SFT, then the KL ablation: two GRPO arms
that differ only in `--kl-coef` (0 and 0.05), both on `--reward posterior
--adv-norm none`, on one or two GPUs, then evaluates both checkpoints under the
protocol above. Set `SFT_STEPS`, `GRPO_STEPS`, `PROMPTS_PER_STEP`,
`GROUP_SIZE`, `EVAL_BOARDS`, and `SESSION_LIMIT_MIN` before launching.

Defaults run a short end-to-end smoke check (2 steps each, 4 boards per tier);
raise them for a training run. `kaggle/minesweeper_grpo.ipynb` profiles GRPO
phases on a P100, and `scripts/pipeline.sh` runs SFT then GRPO locally with
`PY=/path/to/python` to select the interpreter.
