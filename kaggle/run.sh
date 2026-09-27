#!/bin/bash
# Kaggle T4x2 phase-1 run: shared posterior SFT, then truth/posterior GRPO A/B.
set -euo pipefail

REPO_URL=${REPO_URL:-https://github.com/Anchal-T/minesweeper-grpo.git}
WORKDIR=${WORKDIR:-/kaggle/working/minesweeper-grpo}
PY=${PY:-python}
SFT_STEPS=${SFT_STEPS:-2}
GRPO_STEPS=${GRPO_STEPS:-2}
EVAL_GAMES=${EVAL_GAMES:-4}
CPU_SMOKE=${CPU_SMOKE:-0}
BASE_REPO=${HF_REPO_ID:-}

if [ "$CPU_SMOKE" = "1" ]; then
  SFT_STEPS=1
  GRPO_STEPS=1
  EVAL_GAMES=1
  SFT_BATCH=1
  PROMPTS_PER_STEP=1
  GROUP_SIZE=2
  MICRO_BATCH=2
  DEVICE=cpu
  MAX_EVAL_MOVES=1
else
  SFT_BATCH=${SFT_BATCH:-32}
  PROMPTS_PER_STEP=${PROMPTS_PER_STEP:-16}
  GROUP_SIZE=${GROUP_SIZE:-8}
  MICRO_BATCH=${MICRO_BATCH:-24}
  DEVICE=cuda
  MAX_EVAL_MOVES=${MAX_EVAL_MOVES:-0}
fi

if [ ! -d "$WORKDIR/.git" ]; then
  git clone "$REPO_URL" "$WORKDIR"
else
  git -C "$WORKDIR" pull --ff-only
fi
cd "$WORKDIR"
"$PY" -m pip uninstall -y -q torchao >/dev/null 2>&1 || true
"$PY" -m pip install -q peft transformers huggingface_hub
mkdir -p logs runs

if [ -z "${HF_TOKEN:-}" ] && [ -f /kaggle/working/.hf-token ]; then
  HF_TOKEN=$(cat /kaggle/working/.hf-token)
  export HF_TOKEN
fi

SFT_ARGS=()
if [ -n "$BASE_REPO" ]; then
  SFT_ARGS+=(--hub-repo "${BASE_REPO}-sft")
fi
if [ "$CPU_SMOKE" = "1" ]; then
  "$PY" sft.py --target posterior --steps "$SFT_STEPS" --batch "$SFT_BATCH" \
    --device "$DEVICE" --out runs/sft/last --resume --time-budget-min 45 "${SFT_ARGS[@]}"
else
  CUDA_VISIBLE_DEVICES=0 "$PY" sft.py --target posterior --steps "$SFT_STEPS" --batch "$SFT_BATCH" \
    --device "$DEVICE" --out runs/sft/last --resume --time-budget-min 690 "${SFT_ARGS[@]}"
fi

TRUTH_ARGS=()
POSTERIOR_ARGS=()
if [ -n "$BASE_REPO" ]; then
  TRUTH_ARGS+=(--hub-repo "${BASE_REPO}-truth")
  POSTERIOR_ARGS+=(--hub-repo "${BASE_REPO}-posterior")
fi

if [ "$CPU_SMOKE" = "1" ]; then
  "$PY" grpo.py --init-from runs/sft/last --reward truth --adv-norm none \
    --kl-coef 0.05 --steps "$GRPO_STEPS" --prompts-per-step "$PROMPTS_PER_STEP" \
    --group "$GROUP_SIZE" --micro-batch "$MICRO_BATCH" --device "$DEVICE" \
    --lr 1e-5 --out-dir runs/truth --resume --time-budget-min 45 \
    "${TRUTH_ARGS[@]}" > logs/truth.log 2>&1
  "$PY" grpo.py --init-from runs/sft/last --reward posterior --adv-norm none \
    --kl-coef 0.05 --steps "$GRPO_STEPS" --prompts-per-step "$PROMPTS_PER_STEP" \
    --group "$GROUP_SIZE" --micro-batch "$MICRO_BATCH" --device "$DEVICE" \
    --lr 1e-5 --out-dir runs/posterior --resume --time-budget-min 45 \
    "${POSTERIOR_ARGS[@]}" > logs/posterior.log 2>&1
else
  CUDA_VISIBLE_DEVICES=0 "$PY" grpo.py --init-from runs/sft/last --reward truth \
    --adv-norm none --kl-coef 0.05 --steps "$GRPO_STEPS" \
    --prompts-per-step "$PROMPTS_PER_STEP" --group "$GROUP_SIZE" --micro-batch "$MICRO_BATCH" \
    --device "$DEVICE" --lr 1e-5 --out-dir runs/truth \
    --resume --time-budget-min 690 "${TRUTH_ARGS[@]}" > logs/truth.log 2>&1 &
  truth_pid=$!
  CUDA_VISIBLE_DEVICES=1 "$PY" grpo.py --init-from runs/sft/last --reward posterior \
    --adv-norm none --kl-coef 0.05 --steps "$GRPO_STEPS" \
    --prompts-per-step "$PROMPTS_PER_STEP" --group "$GROUP_SIZE" --micro-batch "$MICRO_BATCH" \
    --device "$DEVICE" --lr 1e-5 --out-dir runs/posterior \
    --resume --time-budget-min 690 "${POSTERIOR_ARGS[@]}" > logs/posterior.log 2>&1 &
  posterior_pid=$!
  wait "$truth_pid"
  wait "$posterior_pid"
fi

EVAL_ARGS=()
if [ "$MAX_EVAL_MOVES" -gt 0 ]; then
  EVAL_ARGS+=(--max-moves "$MAX_EVAL_MOVES")
fi
"$PY" eval.py --ckpt runs/truth/last --games "$EVAL_GAMES" \
  --device "$DEVICE" "${EVAL_ARGS[@]}"
"$PY" eval.py --ckpt runs/posterior/last --games "$EVAL_GAMES" \
  --device "$DEVICE" "${EVAL_ARGS[@]}"
