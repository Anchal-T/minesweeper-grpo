"""GRPO (Group Relative Policy Optimization) trainer, from scratch for torch 1.13.

Per step:
  1. sample a batch of fresh random board states (prompts)
  2. roll out G completions per prompt with the current policy
  3. score each completion with the verifiable env reward (safe/mine/parse)
  4. advantage = (r - group mean) / (group std)   <- group-relative, no critic
  5. one (or more) policy-gradient epochs on the answer tokens, with PPO-style
     clipping against the rollout log-probs and an optional KL penalty to a
     frozen reference policy.
"""
import argparse, random, time, os

import torch
import torch.nn.functional as F

from common import load_tokenizer, load_model, sample_completions, PAD_ID
from env import Minesweeper, parse_move, step_reward


def rollout_batch(model, tok, n_prompts, group, temperature, device, max_new_tokens=8):
    """Returns prompts, per-sample completions grouped, rewards."""
    boards = [Minesweeper() for _ in range(n_prompts)]
    prompts = [b.prompt() for b in boards]
    rep_prompts = [p for p in prompts for _ in range(group)]
    texts, gen_ids = sample_completions(
        model, tok, rep_prompts, max_new_tokens=max_new_tokens,
        temperature=temperature, greedy=False)
    rewards = []
    for b, text in zip([bb for bb in boards for _ in range(group)], texts):
        mv = parse_move(text)
        if mv is None:
            rewards.append(-1.0)  # unparseable answer
        else:
            r, _ = step_reward(b, *mv)
            rewards.append(r)
    return prompts, boards, texts, gen_ids, rewards, group


def compute_advantages(rewards, group):
    n_groups = len(rewards) // group
    adv = torch.tensor(rewards, dtype=torch.float32)
    g = adv.view(n_groups, group)
    mean, std = g.mean(dim=1, keepdim=True), g.std(dim=1, keepdim=True)
    return ((g - mean) / (std + 1e-4)).view(-1)


def answer_token_padded(tok, texts, device):
    enc = tok(texts, return_tensors="pt", padding=True).to(device)
    return enc.input_ids, enc.attention_mask


def prompt_token_padded(tok, prompts, device):
    enc = tok(prompts, return_tensors="pt", padding=True).to(device)
    return enc.input_ids, enc.attention_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-from", default="sft.pt", help="checkpoint to start from (gpt2 = raw pretrained)")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--prompts-per-step", type=int, default=16)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--epochs-per-batch", type=int, default=1, help="inner PPO-style epochs (mu>1 needs ratios)")
    ap.add_argument("--kl-coef", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--out-dir", default="runs/grpo")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eval-every", type=int, default=200)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tok = load_tokenizer()
    model = load_model(args.device)
    if args.init_from == "gpt2":
        print("starting from raw pretrained gpt2")
    else:
        sd = torch.load(args.init_from, map_location="cpu")
        model.load_state_dict(sd)
        print(f"loaded init checkpoint {args.init_from}")
    model.gradient_checkpointing_disable()

    ref = None
    if args.kl_coef > 0:
        ref = load_model(args.device)
        ref.load_state_dict(model.state_dict())
        for p in ref.parameters():
            p.requires_grad_(False)
        ref.eval()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    scaler = torch.cuda.amp.GradScaler()
    device = args.device

    log_path = os.path.join(args.out_dir, "log.txt")
    logf = open(log_path, "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    running = {"rew": 0.0, "mine": 0.0, "n": 0}
    t0 = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        # ---------- rollout (no grad) ----------
        model.eval()
        with torch.no_grad():
            prompts, boards, texts, gen_ids, rewards, group = rollout_batch(
                model, tok, args.prompts_per_step, args.group, args.temperature,
                device, args.max_new_tokens)
            ctx_ids, ctx_attn = prompt_token_padded(tok, prompts, device)
            ans_ids, ans_attn = answer_token_padded(tok, texts, device)
            # rollout log-probs for the ratio (also used when mu>1)
            with torch.autocast("cuda", dtype=torch.float16):
                roll_lp, roll_mask = None, None
                if args.epochs_per_batch > 1:
                    lp, m = _logprobs(model, ctx_ids, ans_ids)
                    roll_lp, roll_mask = lp.detach(), m
        adv = compute_advantages(rewards, group).to(device)

        # ---------- update ----------
        model.train()
        for _ in range(args.epochs_per_batch):
            with torch.autocast("cuda", dtype=torch.float16):
                lp, m = _logprobs(model, ctx_ids, ans_ids)
            per_tok = (adv.unsqueeze(1) * lp) * m
            pg_loss = -per_tok.sum() / m.sum()
            loss = pg_loss
            if args.epochs_per_batch > 1:
                ratio = torch.exp((lp - roll_lp).clamp(-20, 20))
                unclipped = ratio * adv.unsqueeze(1)
                clipped = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv.unsqueeze(1)
                pg_loss = -(torch.min(unclipped, clipped) * m).sum() / m.sum()
                loss = pg_loss
            if ref is not None:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    ref_lp, _ = _logprobs(ref, ctx_ids, ans_ids)
                kl = (lp.exp() - ref_lp.exp() - (lp - ref_lp)).masked_select(m.bool()).mean()
                loss = loss + args.kl_coef * kl
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

        # ---------- logging ----------
        r_t = torch.tensor(rewards)
        running["rew"] += r_t.mean().item() * len(rewards)
        running["mine"] += ((r_t <= -0.99) | (r_t == -0.5)).float().sum().item()
        running["n"] += len(rewards)
        if step % 20 == 0 or step == 1:
            n = running["n"]
            log(f"step {step:5d}  reward {running['rew']/n:.4f}  "
                f"mine_or_bad {running['mine']/n:.3f}  "
                f"({(time.time()-t0)/step:.2f}s/step)")
            running = {"rew": 0.0, "mine": 0.0, "n": 0}

        if step % args.eval_every == 0 or step == args.steps:
            ckpt = os.path.join(args.out_dir, f"ckpt_{step}.pt")
            torch.save(model.state_dict(), ckpt)
            log(f"saved {ckpt}")

    logf.close()


def _logprobs(model, ctx_ids, ans_ids):
    """Token log-probs of answers given right-padded contexts."""
    from common import token_logprobs
    lp, mask = token_logprobs(model, ctx_ids, ans_ids)
    return lp, mask


if __name__ == "__main__":
    main()
