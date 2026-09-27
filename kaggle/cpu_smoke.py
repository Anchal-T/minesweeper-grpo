"""Run one SFT step, one GRPO step, and capped eval without reloading Qwen."""
import os
import time

import torch

from checkpoint import save_checkpoint
from common import (MODEL_NAME, load_model, load_tokenizer, sample_completions,
                    token_logprobs)
from env import Minesweeper, parse_move, posterior_move
from grpo import compute_advantages, rollout_batch
from sft import build_batch, loss_on_batch


def stage(label):
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"\n[{stamp}] {label}", flush=True)


def main():
    device = "cpu"
    out = "runs/cpu-smoke"
    os.makedirs(out, exist_ok=True)
    tok = load_tokenizer(MODEL_NAME)

    stage("SMOKE 1/4 load_model device=cpu")
    model = load_model(device, MODEL_NAME)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=1e-4,
        weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    stage("SMOKE 2/4 sft target=posterior batch=1")
    model.train()
    ctx_ids, ctx_attn, ans_ids, ans_attn = build_batch(
        tok, 1, device, target="posterior")
    sft_loss = loss_on_batch(model, ctx_ids, ctx_attn, ans_ids, ans_attn,
                             tok.pad_token_id)
    optimizer.zero_grad(set_to_none=True)
    sft_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    print(f"[SFT] step=1 loss={sft_loss.item():.4f}", flush=True)
    model.save_pretrained(os.path.join(out, "sft"))

    stage("SMOKE 3/4 grpo reward=posterior prompts=1 group=2")
    model.eval()
    _, _, _, answer_ids, ctx_ids, ctx_attn, answer_mask, rewards = (
        rollout_batch(model, tok, n_prompts=1, group=2, temperature=1.0,
                      device=device, max_new_tokens=6, reward_mode="posterior"))
    advantages, low_groups = compute_advantages(rewards, group=2, norm="none")
    grpo_optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=1e-5,
        weight_decay=0.0)
    logprobs, mask, entropy = token_logprobs(
        model, ctx_ids, answer_ids, attn_mask=ctx_attn,
        answer_mask=answer_mask, pad_id=tok.pad_token_id)
    total_mask = mask.sum().clamp_min(1)
    loss = -((advantages.to(device).unsqueeze(1) * logprobs) * mask).sum() / total_mask
    with torch.no_grad(), model.disable_adapter():
        ref_logprobs, _, _ = token_logprobs(
            model, ctx_ids, answer_ids, attn_mask=ctx_attn,
            answer_mask=answer_mask, pad_id=tok.pad_token_id)
    log_ratio = (logprobs - ref_logprobs).clamp(-20, 20)
    kl = ((torch.exp(-log_ratio) + log_ratio - 1) * mask).sum() / total_mask
    loss = loss + 0.05 * kl
    grpo_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    grpo_optimizer.step()
    print(f"[GRPO:posterior] step=1 mean_reward={sum(rewards)/len(rewards):.4f} "
          f"entropy={(entropy * mask).sum().item()/total_mask.item():.3f} "
          f"low_variance_groups={low_groups}/1 loss={loss.item():.4f}", flush=True)
    save_checkpoint(os.path.join(out, "grpo"), model, grpo_optimizer,
                    scaler, step=1)
    print(f"[GRPO:posterior] checkpoint_saved path={out}/grpo", flush=True)

    stage("SMOKE 4/4 evaluation games=1 max_moves=1")
    model.eval()
    board = Minesweeper(seed=0)
    texts, _ = sample_completions(model, tok, [board.prompt()],
                                  max_new_tokens=6, greedy=True)
    move = parse_move(texts[0])
    if move is not None:
        board.click(*move)
    oracle = Minesweeper(seed=0)
    while not oracle.solved():
        oracle_move = posterior_move(oracle)
        if oracle_move is None:
            break
        _, hit_mine = oracle.click(*oracle_move)
        if hit_mine:
            break
    print(f"[EVAL] games=1 policy_moves=1 parsed={move is not None} "
          f"clear_fraction={len(board.revealed)/(board.w*board.h-board.n_mines):.3f} "
          f"posterior_solver_win={oracle.solved()}", flush=True)
    stage("SMOKE complete")


if __name__ == "__main__":
    main()
