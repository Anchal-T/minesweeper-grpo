#!/bin/bash
# Kaggle phase-1 training: posterior SFT, then truth/posterior GRPO on two GPUs.
set -euo pipefail

REPO_URL=${REPO_URL:-https://github.com/Anchal-T/minesweeper-grpo.git}
WORKDIR=${WORKDIR:-/kaggle/working/minesweeper-grpo}
PY=${PY:-python}
CPU_SMOKE=${CPU_SMOKE:-0}
BASE_REPO=${HF_REPO_ID:-}

if [ "$CPU_SMOKE" = "1" ]; then
  SFT_STEPS=1
  SFT_BATCH=1
  DEVICE=cpu
  SMOKE_DIR=runs/cpu-smoke
else
  SFT_STEPS=${SFT_STEPS:-2}
  GRPO_STEPS=${GRPO_STEPS:-2}
  SFT_BATCH=${SFT_BATCH:-32}
  PROMPTS_PER_STEP=${PROMPTS_PER_STEP:-16}
  GROUP_SIZE=${GROUP_SIZE:-8}
  MICRO_BATCH=${MICRO_BATCH:-24}
  EVAL_GAMES=${EVAL_GAMES:-4}
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

stage() {
  printf '\n[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

if [ -z "${HF_TOKEN:-}" ] && [ -f /kaggle/working/.hf-token ]; then
  HF_TOKEN=$(cat /kaggle/working/.hf-token)
  export HF_TOKEN
fi

SFT_ARGS=()
TRUTH_ARGS=()
POSTERIOR_ARGS=()
if [ -n "$BASE_REPO" ]; then
  SFT_ARGS+=(--hub-repo "${BASE_REPO}-sft")
  TRUTH_ARGS+=(--hub-repo "${BASE_REPO}-truth")
  POSTERIOR_ARGS+=(--hub-repo "${BASE_REPO}-posterior")
fi

if [ "$CPU_SMOKE" = "1" ]; then
  stage "SMOKE 1/3 sft target=posterior device=cpu"
  "$PY" sft.py --target posterior --steps "$SFT_STEPS" --batch "$SFT_BATCH" \
    --device "$DEVICE" --out "$SMOKE_DIR/sft" --time-budget-min 8 "${SFT_ARGS[@]}"
  stage "SMOKE 2/3 grpo reward=posterior device=cpu"
  "$PY" grpo.py --init-from "$SMOKE_DIR/sft" --reward posterior --adv-norm none \
    --kl-coef 0.05 --steps 1 --prompts-per-step 1 --group 2 --micro-batch 2 \
    --device "$DEVICE" --lr 1e-5 --out-dir "$SMOKE_DIR/grpo" --time-budget-min 8 \
    "${POSTERIOR_ARGS[@]}"
  stage "SMOKE 3/3 evaluation games=1 max_moves=1"
  "$PY" eval.py --ckpt "$SMOKE_DIR/grpo/last" --games 1 \
    --device "$DEVICE" --max-moves 1
  stage "SMOKE complete"
  exit 0
fi

stage "SFT target=posterior steps=$SFT_STEPS device=$DEVICE"
CUDA_VISIBLE_DEVICES=0 "$PY" sft.py --target posterior --steps "$SFT_STEPS" --batch "$SFT_BATCH" \
  --device "$DEVICE" --out runs/sft/last --resume --time-budget-min 690 "${SFT_ARGS[@]}"

stage "GRPO A/B steps=$GRPO_STEPS prompts=$PROMPTS_PER_STEP group=$GROUP_SIZE"
CUDA_VISIBLE_DEVICES=0 "$PY" grpo.py --init-from runs/sft/last --reward truth \
  --adv-norm none --kl-coef 0.05 --steps "$GRPO_STEPS" \
  --prompts-per-step "$PROMPTS_PER_STEP" --group "$GROUP_SIZE" --micro-batch "$MICRO_BATCH" \
  --device "$DEVICE" --lr 1e-5 --out-dir runs/truth \
  --resume --time-budget-min 690 "${TRUTH_ARGS[@]}" 2>&1 | tee logs/truth.log &
truth_pid=$!
CUDA_VISIBLE_DEVICES=1 "$PY" grpo.py --init-from runs/sft/last --reward posterior \
  --adv-norm none --kl-coef 0.05 --steps "$GRPO_STEPS" \
  --prompts-per-step "$PROMPTS_PER_STEP" --group "$GROUP_SIZE" --micro-batch "$MICRO_BATCH" \
  --device "$DEVICE" --lr 1e-5 --out-dir runs/posterior \
  --resume --time-budget-min 690 "${POSTERIOR_ARGS[@]}" 2>&1 | tee logs/posterior.log &
posterior_pid=$!
wait "$truth_pid"
wait "$posterior_pid"

EVAL_ARGS=()
if [ "$MAX_EVAL_MOVES" -gt 0 ]; then
  EVAL_ARGS+=(--max-moves "$MAX_EVAL_MOVES")
fi
stage "EVALUATION truth games=$EVAL_GAMES"
"$PY" eval.py --ckpt runs/truth/last --games "$EVAL_GAMES" \
  --device "$DEVICE" "${EVAL_ARGS[@]}"
stage "EVALUATION posterior games=$EVAL_GAMES"
"$PY" eval.py --ckpt runs/posterior/last --games "$EVAL_GAMES" \
  --device "$DEVICE" "${EVAL_ARGS[@]}"
