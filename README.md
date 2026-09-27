# Minesweeper GRPO on Qwen

Train a Qwen2.5-0.5B-Instruct policy to select moves on procedurally
generated Minesweeper boards. The pipeline uses LoRA, posterior-oracle SFT,
and single-move GRPO rewards. Checkpoints include the adapter, optimizer,
gradient scaler, step, and RNG state, with optional private Hugging Face Hub
sync through `HF_TOKEN` and `HF_REPO_ID`.

## Install

Use a recent PyTorch with CUDA support, then install:

```bash
python -m pip install transformers peft huggingface_hub
```

The base model defaults to `Qwen/Qwen2.5-0.5B-Instruct`. The tokenizer chat
template formats board prompts; completions are `row,column`.

## Train

```bash
python sft.py --target posterior --steps 400 --out runs/sft/last
python grpo.py --init-from runs/sft/last --reward posterior \
  --adv-norm none --kl-coef 0.05 --steps 1000 --out-dir runs/grpo
python eval.py --ckpt runs/grpo/last --games 400
```

Use `--resume` to restore the latest local checkpoint or download it from the
Hub when `--hub-repo` (or `HF_REPO_ID`) is set. `--time-budget-min` saves a
resumable checkpoint before exiting. GRPO supports `--reward truth|posterior|vpr`,
`--adv-norm std|none`, and `--kl-est k3|k2`. For the optional rationale
format, use `sft.py --answer-format cot` followed by GRPO with
`--answer-format cot --prompt-mean-loss --max-new-tokens 160`; set
`--format-reward` to reward the required `<think>...</think> Answer: r,c`
structure.

## Kaggle T4x2

Open [kaggle/minesweeper_grpo.ipynb](kaggle/minesweeper_grpo.ipynb), enable two
T4 GPUs, add Kaggle Secrets `HF_TOKEN` and optionally `HF_REPO_ID`, then choose
**Save & Run All**. The launcher runs posterior-target SFT, truth and posterior
GRPO variants on separate GPUs, then reports policy results alongside the
posterior-greedy solver benchmark on the same seeded boards.

Set `SFT_STEPS` and `GRPO_STEPS` in the notebook before launching. The default
20 GRPO steps are for the initial smoke run; raise this for a training run.
Use `scripts/pipeline.sh` locally with `PY=/path/to/python` to select the
Python interpreter.
