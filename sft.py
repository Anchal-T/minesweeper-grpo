"""SFT warmup on posterior-optimal (or ground-truth expert) moves."""
import argparse
import os
import time

import torch

from checkpoint import (load_training_state, restore_from_hub, save_checkpoint,
                        time_budget_expired)
from common import MODEL_NAME, format_prompt, load_model, load_tokenizer
from env import Minesweeper, expert_move, posterior_move


def build_batch(tok, n, device, target):
    prompts, answers = [], []
    for _ in range(n):
        board = Minesweeper()
        move = posterior_move(board) if target == "posterior" else expert_move(board)
        if move is not None:
            prompts.append(format_prompt(tok, board.prompt()))
            answers.append(f"{move[0]},{move[1]}")
    ctx = tok(prompts, return_tensors="pt", padding=True).to(device)
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
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="runs/sft/last")
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--hub-repo", default=os.environ.get("HF_REPO_ID"))
    ap.add_argument("--time-budget-min", type=float, default=690)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

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
    model.train()
    for step in range(start_step + 1, args.steps + 1):
        ctx_ids, ctx_attn, ans_ids, ans_attn = build_batch(
            tok, args.batch, args.device, args.target)
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
            print(f"step {step:5d}  sft_loss {loss.item():.4f}  "
                  f"({(time.monotonic()-started)/max(1,step-start_step):.2f}s/step)", flush=True)
        if step % args.save_every == 0:
            save_checkpoint(args.out, model, opt, scaler, step, args.hub_repo)
        if time_budget_expired(started, args.time_budget_min):
            print("time budget reached; saving resumable checkpoint", flush=True)
            break
    save_checkpoint(args.out, model, opt, scaler,
                    step if 'step' in locals() else start_step, args.hub_repo)


if __name__ == "__main__":
    main()
