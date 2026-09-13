"""SFT warmup: behavior-clone the expert (safe cell with max flood-fill reveal)
on random mid-game states, so GRPO starts from a policy that knows the format
and basic play instead of exploring from noise."""
import argparse, math, random, time

import torch
from torch.utils.data import DataLoader, TensorDataset

from common import MODEL_NAME, load_tokenizer, load_model, PAD_ID
from env import Minesweeper, expert_move


def build_batch(tok, n, device):
    boards = [Minesweeper() for _ in range(n)]
    prompts, answers = [], []
    for b in boards:
        mv = expert_move(b)
        if mv is None:
            continue
        prompts.append(b.prompt())
        answers.append(f"{mv[0]},{mv[1]}")
    enc_p = tok(prompts, return_tensors="pt", padding=True).to(device)
    enc_a = tok(answers, return_tensors="pt", padding=True).to(device)
    return enc_p.input_ids, enc_p.attention_mask, enc_a.input_ids, enc_a.attention_mask


def loss_on_batch(model, ctx_ids, ctx_attn, ans_ids, ans_attn):
    B, Lc = ctx_ids.shape
    full = torch.cat([ctx_ids, ans_ids], dim=1)
    attn = torch.cat([ctx_attn, ans_attn], dim=1)
    logits = model(input_ids=full, attention_mask=attn).logits
    pos = torch.arange(Lc - 1, full.shape[1] - 1, device=full.device)
    pred = logits[:, pos, :]
    tgt = full[:, Lc:]  # answer tokens
    lp = torch.log_softmax(pred.float(), dim=-1)
    nll = torch.nn.functional.nll_loss(
        lp.reshape(-1, lp.size(-1)), tgt.reshape(-1), reduction="none"
    ).view(B, -1)
    loss = (nll * ans_attn).sum() / ans_attn.sum()
    return loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--out", default="sft.pt")
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    tok = load_tokenizer()
    model = load_model(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler()

    model.train()
    t0 = time.time()
    for step in range(1, args.steps + 1):
        with torch.autocast("cuda", dtype=torch.float16):
            ctx_ids, ctx_attn, ans_ids, ans_attn = build_batch(tok, args.batch, args.device)
            loss = loss_on_batch(model, ctx_ids, ctx_attn, ans_ids, ans_attn)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        if step % 50 == 0 or step == 1:
            print(f"step {step:5d}  sft_loss {loss.item():.4f}  "
                  f"({(time.time()-t0)/step:.2f}s/step)", flush=True)

        if step % args.save_every == 0:
            torch.save(model.state_dict(), args.out)
            print(f"  checkpoint saved -> {args.out}", flush=True)

    torch.save(model.state_dict(), args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
