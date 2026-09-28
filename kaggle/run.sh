#!/bin/bash
# Kaggle phase-1 training: posterior SFT, then truth/posterior GRPO.
set -euo pipefail
SECONDS=0

REPO_URL=${REPO_URL:-https://github.com/Anchal-T/minesweeper-grpo.git}
WORKDIR=${WORKDIR:-/kaggle/working/minesweeper-grpo}
PY=${PY:-python}
CPU_SMOKE=${CPU_SMOKE:-0}
BASE_REPO=${HF_REPO_ID:-}
SESSION_LIMIT_MIN=${SESSION_LIMIT_MIN:-720}
EVAL_RESERVE_MIN=${EVAL_RESERVE_MIN:-25}
SESSION_MARGIN_MIN=${SESSION_MARGIN_MIN:-5}
HUB_UPLOAD_EVERY=${HUB_UPLOAD_EVERY:-500}

if [ "$CPU_SMOKE" = "1" ]; then
  DEVICE=cpu
else
  SFT_STEPS=${SFT_STEPS:-2}
  GRPO_STEPS=${GRPO_STEPS:-2}
  SFT_BATCH=${SFT_BATCH:-32}
  PROMPTS_PER_STEP=${PROMPTS_PER_STEP:-16}
  GROUP_SIZE=${GROUP_SIZE:-8}
  MICRO_BATCH=${MICRO_BATCH:-32}
  EVAL_GAMES=${EVAL_GAMES:-4}
  DEVICE=cuda
  MAX_EVAL_MOVES=${MAX_EVAL_MOVES:-0}
  TRAINING_CUTOFF_SECONDS=$(((SESSION_LIMIT_MIN - EVAL_RESERVE_MIN - SESSION_MARGIN_MIN) * 60))
  if ((TRAINING_CUTOFF_SECONDS <= 0)); then
    echo "SESSION_LIMIT_MIN must exceed EVAL_RESERVE_MIN + SESSION_MARGIN_MIN" >&2
    exit 2
  fi
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

if [ "$CPU_SMOKE" = "1" ]; then
  stage "CPU smoke (single model load)"
  "$PY" kaggle/cpu_smoke.py
  exit 0
fi

remaining_training_minutes() {
  local remaining_seconds=$((TRAINING_CUTOFF_SECONDS - SECONDS))
  if ((remaining_seconds < 60)); then
    return 1
  fi
  printf '%d\n' "$((remaining_seconds / 60))"
}

SFT_ARGS=()
TRUTH_ARGS=()
POSTERIOR_ARGS=()
if [ -n "$BASE_REPO" ]; then
  SFT_ARGS+=(--hub-repo "${BASE_REPO}-sft")
  TRUTH_ARGS+=(--hub-repo "${BASE_REPO}-truth")
  POSTERIOR_ARGS+=(--hub-repo "${BASE_REPO}-posterior")
fi

GPU_COUNT=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)
if ((GPU_COUNT < 1)); then
  stage "no CUDA GPU found"
  exit 1
fi

if stage_budget=$(remaining_training_minutes); then
  stage "SFT target=posterior steps=$SFT_STEPS device=$DEVICE budget=${stage_budget}m"
  CUDA_VISIBLE_DEVICES=0 "$PY" sft.py --target posterior --steps "$SFT_STEPS" --batch "$SFT_BATCH" \
    --device "$DEVICE" --out runs/sft/last --resume --hub-every "$HUB_UPLOAD_EVERY" \
    --time-budget-min "$stage_budget" "${SFT_ARGS[@]}"
else
  stage "SFT skipped: training deadline reached"
fi

run_grpo() {
  local reward=$1 gpu=$2 logfile=$3 budget=$4
  shift 4
  stage "GRPO reward=$reward gpu=$gpu steps=$GRPO_STEPS budget=${budget}m"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" grpo.py --init-from runs/sft/last --reward "$reward" \
    --adv-norm none --kl-coef 0.05 --steps "$GRPO_STEPS" \
    --prompts-per-step "$PROMPTS_PER_STEP" --group "$GROUP_SIZE" --micro-batch "$MICRO_BATCH" \
    --device "$DEVICE" --lr 1e-5 --out-dir "runs/$reward" --resume \
    --hub-every "$HUB_UPLOAD_EVERY" --time-budget-min "$budget" "$@" 2>&1 | tee "$logfile"
}

stage "GRPO A/B steps=$GRPO_STEPS prompts=$PROMPTS_PER_STEP group=$GROUP_SIZE gpu_count=$GPU_COUNT"
if ((GPU_COUNT >= 2)); then
  if stage_budget=$(remaining_training_minutes); then
    run_grpo truth 0 logs/truth.log "$stage_budget" "${TRUTH_ARGS[@]}" &
    truth_pid=$!
    run_grpo posterior 1 logs/posterior.log "$stage_budget" "${POSTERIOR_ARGS[@]}" &
    posterior_pid=$!
    wait "$truth_pid"
    wait "$posterior_pid"
  else
    stage "GRPO skipped: training deadline reached"
  fi
else
  if stage_budget=$(remaining_training_minutes); then
    run_grpo truth 0 logs/truth.log "$stage_budget" "${TRUTH_ARGS[@]}"
  else
    stage "truth GRPO skipped: training deadline reached"
  fi
  if stage_budget=$(remaining_training_minutes); then
    run_grpo posterior 0 logs/posterior.log "$stage_budget" "${POSTERIOR_ARGS[@]}"
  else
    stage "posterior GRPO skipped: training deadline reached"
  fi
fi

EVAL_ARGS=()
if [ "$MAX_EVAL_MOVES" -gt 0 ]; then
  EVAL_ARGS+=(--max-moves "$MAX_EVAL_MOVES")
fi

run_eval() {
  local name=$1 checkpoint=$2
  if [ ! -f "$checkpoint/adapter_config.json" ]; then
    stage "EVALUATION $name skipped: checkpoint not found"
    return 0
  fi
  local eval_seconds=$((EVAL_RESERVE_MIN * 30))
  local available_seconds=$(((SESSION_LIMIT_MIN - SESSION_MARGIN_MIN) * 60 - SECONDS))
  if ((eval_seconds > available_seconds)); then
    eval_seconds=$available_seconds
  fi
  if ((eval_seconds <= 0)); then
    stage "EVALUATION $name skipped: session margin reached"
    return 0
  fi
  stage "EVALUATION $name games=$EVAL_GAMES timeout=${eval_seconds}s"
  if CUDA_VISIBLE_DEVICES=0 timeout --signal=INT "${eval_seconds}s" "$PY" eval.py \
      --ckpt "$checkpoint" --games "$EVAL_GAMES" --device "$DEVICE" "${EVAL_ARGS[@]}" \
      2>&1 | tee "logs/eval-$name.log"; then
    :
  else
    local status=$?
    stage "EVALUATION $name ended status=$status"
  fi
}

run_eval truth runs/truth/last
run_eval posterior runs/posterior/last
