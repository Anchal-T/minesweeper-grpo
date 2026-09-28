"""SFT warmup on posterior-optimal (or ground-truth expert) moves."""
import argparse
import os
import time

import torch

from checkpoint import (load_training_state, restore_from_hub, save_checkpoint,
                        time_budget_expired)
from common import MODEL_NAME, format_prompt, load_model, load_tokenizer
from env import Minesweeper, expert_move, mine_posterior, posterior_move


def build_batch(tok, n, device, target, answer_format="move"):
    prompts, answers = [], []
    for _ in range(n):
        board = Minesweeper()
        move = posterior_move(board) if target == "posterior" else expert_move(board)
        if move is not None:
            prompts.append(format_prompt(tok, board.prompt()))
            answer = f"{move[0]},{move[1]}"
            if answer_format == "cot":
                p_min = min(mine_posterior(board).values())
                if p_min == 0:
                    rationale = "The posterior proves this move is safe; its mine probability is 0."
                else:
                    rationale = (f"No safe deduction; the lowest risk is p={p_min:.2f} "
                                 f"at {move[0]},{move[1]}.")
                answer = f"<think>{rationale}</think> Answer: {answer}"
            answers.append(answer)
    old_side = tok.padding_side
    tok.padding_side = "left"
    try:
        ctx = tok(prompts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(device)
    finally:
        tok.padding_side = old_side
    ans = tok(answers, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    eos = torch.full((ans.input_ids.shape[0], 1), tok.eos_token_id,
                     dtype=ans.input_ids.dtype, device=device)
    ans_ids = torch.cat([ans.input_ids, eos], dim=1)
    ans_mask = torch.cat([ans.attention_mask,
                          torch.ones_like(eos, dtype=ans.attention_mask.dtype)], dim=1)
    return ctx.input_ids, ctx.attention_mask, ans_ids, ans_mask


def loss_on_batch(model, ctx_ids, ctx_attn, ans_ids, ans_attn, pad_id):
    from common import token_logprobs
    lp, mask, _ = token_logprobs(model, ctx_ids, ans_ids, attn_mask=ctx_attn,
                                 answer_mask=ans_attn, pad_id=pad_id)
    return -(lp * mask).sum() / mask.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--target", choices=["expert", "posterior"], default="posterior")
    ap.add_argument("--answer-format", choices=["move", "cot"], default="move")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="runs/sft/last")
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--hub-repo", default=os.environ.get("HF_REPO_ID"))
    ap.add_argument("--hub-every", type=int, default=500)
    ap.add_argument("--time-budget-min", type=float, default=690)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.answer_format == "cot" and args.target != "posterior":
        ap.error("CoT targets are generated from the posterior oracle")
    if args.hub_every <= 0:
        ap.error("--hub-every must be positive")

    budget_started = time.monotonic()
    if args.resume:
        restore_from_hub(args.out, args.hub_repo)
    tok = load_tokenizer(args.model)
    model = load_model(args.device, args.model,
                       adapter=args.out if args.resume and os.path.exists(
                           os.path.join(args.out, "adapter_config.json")) else None)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                            lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=args.device.startswith("cuda"))
    start_step = load_training_state(args.out, opt, scaler, args.device) if (
        args.resume and os.path.exists(os.path.join(args.out, "training.pt"))) else 0
    started = time.monotonic()
    last_saved_step = None
    model.train()
    for step in range(start_step + 1, args.steps + 1):
        ctx_ids, ctx_attn, ans_ids, ans_attn = build_batch(
            tok, args.batch, args.device, args.target, args.answer_format)
        with torch.autocast("cuda", dtype=torch.float16,
                            enabled=args.device.startswith("cuda")):
            loss = loss_on_batch(model, ctx_ids, ctx_attn, ans_ids, ans_attn,
                                 tok.pad_token_id)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        if step % 25 == 0 or step == start_step + 1:
            elapsed = time.monotonic() - started
            rate = elapsed / max(1, step - start_step)
            print(f"[SFT] step={step} loss={loss.item():.4f} "
                  f"seconds_per_step={rate:.2f}", flush=True)
        budget_hit = time_budget_expired(budget_started, args.time_budget_min)
        should_save = step % args.save_every == 0 or step == args.steps or budget_hit
        if should_save:
            upload_now = (step % args.hub_every == 0 or step == args.steps
                          or budget_hit)
            save_checkpoint(args.out, model, opt, scaler, step,
                            args.hub_repo if upload_now else None)
            last_saved_step = step
        if budget_hit:
            print("[SFT] time_budget_reached saving_checkpoint=true", flush=True)
            break
    if last_saved_step is None:
        save_checkpoint(args.out, model, opt, scaler, start_step, args.hub_repo)


if __name__ == "__main__":
    main()
