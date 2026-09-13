#!/bin/bash
# GPU pipelines: each GPU runs SFT (if any) then GRPO, chained.
# Usage: nohup bash scripts/pipeline.sh <gpu> <sft_steps> <sft_lr> <sft_out> <tag> <grpo_lr> <grpo_temp> [init_ckpt|gpt2] [sft_batch]
set -e
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
GPU=$1; SFT_STEPS=$2; SFT_LR=$3; SFT_OUT=$4; TAG=$5; GRPO_LR=$6; TEMP=$7; INIT=${8:-gpt2}; BATCH=${9:-32}
PY=.venv/bin/python

if [ "$SFT_STEPS" != "0" ]; then
  CUDA_VISIBLE_DEVICES=$GPU $PY sft.py --steps $SFT_STEPS --batch $BATCH --lr $SFT_LR --out $SFT_OUT >> logs/sft_$TAG.log 2>&1
  INIT=$SFT_OUT
fi
CUDA_VISIBLE_DEVICES=$GPU $PY grpo.py --init-from $INIT \
  --steps 1500 --prompts-per-step 12 --group 8 \
  --lr $GRPO_LR --temperature $TEMP --out-dir runs/grpo_$TAG >> logs/grpo_$TAG.log 2>&1
