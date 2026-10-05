#!/bin/bash
# Kaggle phase-1 training: posterior SFT, then a KL 0 versus 0.05 GRPO ablation.
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
  EVAL_BOARDS=${EVAL_BOARDS:-4}
  DEVICE=cuda
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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
KL0_ARGS=()
KL005_ARGS=()
if [ -n "$BASE_REPO" ]; then
  SFT_ARGS+=(--hub-repo "${BASE_REPO}-sft")
  KL0_ARGS+=(--hub-repo "${BASE_REPO}-kl0")
  KL005_ARGS+=(--hub-repo "${BASE_REPO}-kl005")
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
  local name=$1 gpu=$2 logfile=$3 budget=$4 kl=$5
  shift 5
  stage "GRPO name=$name gpu=$gpu kl_coef=$kl steps=$GRPO_STEPS budget=${budget}m"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" grpo.py --init-from runs/sft/last --reward posterior \
    --adv-norm none --kl-coef "$kl" --steps "$GRPO_STEPS" \
    --prompts-per-step "$PROMPTS_PER_STEP" --group "$GROUP_SIZE" --micro-batch "$MICRO_BATCH" \
    --device "$DEVICE" --lr 1e-5 --out-dir "runs/$name" --resume \
    --hub-every "$HUB_UPLOAD_EVERY" --time-budget-min "$budget" "$@" 2>&1 | tee "$logfile"
}

stage "GRPO KL ablation steps=$GRPO_STEPS prompts=$PROMPTS_PER_STEP group=$GROUP_SIZE gpu_count=$GPU_COUNT"
if ((GPU_COUNT >= 2)); then
  if stage_budget=$(remaining_training_minutes); then
    run_grpo kl0 0 logs/kl0.log "$stage_budget" 0 "${KL0_ARGS[@]}" &
    kl0_pid=$!
    run_grpo kl005 1 logs/kl005.log "$stage_budget" 0.05 "${KL005_ARGS[@]}" &
    kl005_pid=$!
    wait "$kl0_pid"
    wait "$kl005_pid"
  else
    stage "GRPO skipped: training deadline reached"
  fi
else
  if stage_budget=$(remaining_training_minutes); then
    run_grpo kl0 0 logs/kl0.log "$stage_budget" 0 "${KL0_ARGS[@]}"
  else
    stage "kl0 GRPO skipped: training deadline reached"
  fi
  if stage_budget=$(remaining_training_minutes); then
    run_grpo kl005 0 logs/kl005.log "$stage_budget" 0.05 "${KL005_ARGS[@]}"
  else
    stage "kl005 GRPO skipped: training deadline reached"
  fi
fi

EVAL_ARGS=(--boards "$EVAL_BOARDS")

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
  stage "EVALUATION $name boards=$EVAL_BOARDS timeout=${eval_seconds}s"
  if CUDA_VISIBLE_DEVICES=0 timeout --signal=INT "${eval_seconds}s" "$PY" eval.py \
      --ckpt "$checkpoint" --device "$DEVICE" "${EVAL_ARGS[@]}" \
      2>&1 | tee "logs/eval-$name.log"; then
    :
  else
    local status=$?
    stage "EVALUATION $name ended status=$status"
  fi
}

run_eval kl0 runs/kl0/last
run_eval kl005 runs/kl005/last
