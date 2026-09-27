"""Small from-scratch GRPO trainer for Minesweeper answers."""
import argparse
import os
import re
import time

import torch

from checkpoint import (load_training_state, restore_from_hub, save_checkpoint,
                        time_budget_expired)
from common import (MODEL_NAME, load_model, load_tokenizer, sample_completions,
                    token_logprobs)
from env import Minesweeper, parse_move, posterior_reward, step_reward


def rollout_batch(model, tok, n_prompts, group, temperature, device,
                  max_new_tokens=6, n_mines=6, reward_mode="truth",
                  answer_format="move", format_reward=0.0):
    boards = [Minesweeper(n_mines=n_mines) for _ in range(n_prompts)]
    prompts = [b.prompt() for b in boards]
    rep_prompts = [p for p in prompts for _ in range(group)]
    texts, gen_ids, ctx_ids, ctx_attn = sample_completions(
        model, tok, rep_prompts, max_new_tokens=max_new_tokens,
        temperature=temperature, greedy=False, return_inputs=True)
    rewards, masks = [], []
    posteriors = [None] * len(boards)
    for row, (board, text, generated) in enumerate(zip(
            [b for b in boards for _ in range(group)], texts, gen_ids)):
        match = (re.search(r"Answer:\s*\d+\s*,\s*\d+", text)
                 if answer_format == "cot" else re.search(r"\d+\s*,\s*\d+", text))
        move = parse_move(match.group() if match else "")
        if move is None:
            rewards.append(-1.0)
            masks.append(_completion_mask(tok, text, generated, None))
        else:
            if reward_mode == "truth":
                reward, _ = step_reward(board, *move)
            else:
                board_index = row // group
                if posteriors[board_index] is None:
                    from env import mine_posterior
                    posteriors[board_index] = mine_posterior(board)
                reward = posterior_reward(board, *move, mode=reward_mode,
                                          posterior=posteriors[board_index])
            if answer_format == "cot" and re.match(
                    r"\s*<think>.*?</think>\s*Answer:\s*\d+\s*,\s*\d+\s*\Z",
                    text, re.DOTALL):
                reward += format_reward
            rewards.append(reward)
            masks.append(_completion_mask(tok, text, generated, match.end()))
    answer_mask = torch.stack(masks).to(device)
    return prompts, boards, texts, gen_ids, ctx_ids, ctx_attn, answer_mask, rewards


def _completion_mask(tok, text, generated, parsed_end):
    """Keep credit through the parsed move; stop at EOS or generated padding."""
    ids = generated
    eos_positions = (ids == tok.eos_token_id).nonzero(as_tuple=True)[0]
    end = int(eos_positions[0]) + 1 if len(eos_positions) else ids.numel()
    if parsed_end is not None:
        prefix_ids = tok(text[:parsed_end], add_special_tokens=False).input_ids
        end = min(end, len(prefix_ids))
    mask = torch.zeros_like(ids, dtype=torch.float32)
    mask[:end] = 1
    return mask


def compute_advantages(rewards, group, norm="std", min_group_std=1e-3):
    n_groups = len(rewards) // group
    values = torch.tensor(rewards, dtype=torch.float32).view(n_groups, group)
    mean = values.mean(dim=1, keepdim=True)
    std = values.std(dim=1, unbiased=False, keepdim=True)
    low_variance = std.squeeze(1) < min_group_std
    advantage = values - mean
    if norm == "std":
        advantage = advantage / std.clamp_min(min_group_std)
    return advantage.reshape(-1), int(low_variance.sum().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--init-from", default="runs/sft/last",
                    help="LoRA adapter directory, or 'base' for the pretrained model")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--prompts-per-step", type=int, default=16)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--temp-end", type=float, default=None)
    ap.add_argument("--curriculum", default=None)
    ap.add_argument("--epochs-per-batch", type=int, default=1)
    ap.add_argument("--kl-coef", type=float, default=0.0)
    ap.add_argument("--kl-est", choices=["k3", "k2"], default="k3")
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--reward", choices=["truth", "posterior", "vpr"], default="truth")
    ap.add_argument("--answer-format", choices=["move", "cot"], default="move")
    ap.add_argument("--format-reward", type=float, default=0.0)
    ap.add_argument("--prompt-mean-loss", action="store_true")
    ap.add_argument("--adv-norm", choices=["std", "none"], default="std")
    ap.add_argument("--min-group-std", type=float, default=1e-3)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--micro-batch", type=int, default=24)
    ap.add_argument("--out-dir", default="runs/grpo")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--hub-repo", default=os.environ.get("HF_REPO_ID"))
    ap.add_argument("--time-budget-min", type=float, default=690)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eval-every", type=int, default=100)
    args = ap.parse_args()
    if args.answer_format == "cot" and args.reward == "truth":
        ap.error("CoT phase 2 uses posterior or vpr rewards, not truth reward")
    if args.max_new_tokens is None:
        args.max_new_tokens = 160 if args.answer_format == "cot" else 6

    checkpoint_dir = os.path.join(args.out_dir, "last")
    if args.resume:
        restore_from_hub(checkpoint_dir, args.hub_repo)
    tok = load_tokenizer(args.model)
    adapter = checkpoint_dir if args.resume and os.path.exists(
        os.path.join(checkpoint_dir, "adapter_config.json")) else (
            args.init_from if args.init_from != "base" and os.path.isdir(args.init_from) else None)
    model = load_model(args.device, args.model, adapter=adapter)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                            lr=args.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=args.device.startswith("cuda"))
    start_step = load_training_state(checkpoint_dir, opt, scaler, args.device) if (
        args.resume and os.path.exists(os.path.join(checkpoint_dir, "training.pt"))) else 0
    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "log.txt")
    started = time.monotonic()

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    stages = []
    if args.curriculum:
        for part in args.curriculum.split(","):
            mines, until = part.split(":")
            stages.append((int(mines), int(until) if until else args.steps + 1))

    def mines_for(step):
        for mines, until in stages:
            if step <= until:
                return mines
        return stages[-1][0] if stages else 6

    reward_total = mine_total = sample_total = 0.0
    model.eval()  # Keep LoRA dropout disabled so scoring matches rollout policy.
    for step in range(start_step + 1, args.steps + 1):
        temperature = args.temperature
        if args.temp_end is not None:
            fraction = min(1.0, (step - 1) / max(1, args.steps - 1))
            temperature += fraction * (args.temp_end - args.temperature)
        model.eval()
        with torch.no_grad(), torch.autocast(
                "cuda", dtype=torch.float16, enabled=args.device.startswith("cuda")):
            (_, _, texts, answer_ids, ctx_ids, ctx_attn, answer_mask, rewards) = rollout_batch(
                model, tok, args.prompts_per_step, args.group, temperature,
                args.device, args.max_new_tokens, n_mines=mines_for(step),
                reward_mode=args.reward, answer_format=args.answer_format,
                format_reward=args.format_reward)
            roll_lp = None
            if args.epochs_per_batch > 1:
                roll_lp, _, _ = token_logprobs(
                    model, ctx_ids, answer_ids, attn_mask=ctx_attn,
                    answer_mask=answer_mask, pad_id=tok.pad_token_id)
                roll_lp = roll_lp.detach()
        advantages, low_groups = compute_advantages(
            rewards, args.group, args.adv_norm, args.min_group_std)
        advantages = advantages.to(args.device)
        total_mask = answer_mask.sum().clamp_min(1)
        prompt_token_counts = answer_mask.view(
            args.prompts_per_step, args.group, -1).sum(dim=(1, 2)).clamp_min(1)
        entropy_total = 0.0
        model.eval()
        for _ in range(args.epochs_per_batch):
            for offset in range(0, ctx_ids.shape[0], args.micro_batch):
                sl = slice(offset, offset + args.micro_batch)
                with torch.autocast("cuda", dtype=torch.float16,
                                    enabled=args.device.startswith("cuda")):
                    lp, mask, entropy = token_logprobs(
                        model, ctx_ids[sl], answer_ids[sl], attn_mask=ctx_attn[sl],
                        answer_mask=answer_mask[sl], pad_id=tok.pad_token_id)
                    adv = advantages[sl].unsqueeze(1)
                    policy_terms = adv * lp
                    if roll_lp is not None:
                        ratio = torch.exp((lp - roll_lp[sl]).clamp(-20, 20))
                        clipped = ratio.clamp(1 - args.clip, 1 + args.clip)
                        policy_terms = torch.minimum(ratio * adv, clipped * adv)
                    if args.prompt_mean_loss:
                        rows = torch.arange(offset, min(offset + args.micro_batch,
                                                        ctx_ids.shape[0]), device=args.device)
                        row_weights = (1.0 / (args.prompts_per_step *
                                              prompt_token_counts[rows // args.group])).unsqueeze(1)
                        loss = -(policy_terms * mask * row_weights).sum()
                    else:
                        row_weights = None
                        loss = -(policy_terms * mask).sum() / total_mask
                    entropy_total += float((entropy.detach() * mask).sum())
                    if args.kl_coef > 0:
                        with torch.no_grad(), model.disable_adapter(), torch.autocast(
                                "cuda", dtype=torch.float16,
                                enabled=args.device.startswith("cuda")):
                            ref_lp, _, _ = token_logprobs(
                                model, ctx_ids[sl], answer_ids[sl],
                                attn_mask=ctx_attn[sl], answer_mask=answer_mask[sl],
                                pad_id=tok.pad_token_id)
                        log_ratio = (lp - ref_lp).clamp(-20, 20)
                        if args.kl_est == "k2":
                            kl_tokens = 0.5 * log_ratio.square()
                        else:
                            # k3 estimates KL(pi || ref) in value, while its
                            # policy gradient follows KL(ref || pi).
                            kl_tokens = torch.exp(-log_ratio) + log_ratio - 1
                        kl_loss = ((kl_tokens * mask * row_weights).sum()
                                   if row_weights is not None else
                                   (kl_tokens * mask).sum() / total_mask)
                        loss = loss + args.kl_coef * kl_loss
                scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

        reward_tensor = torch.tensor(rewards)
        reward_total += reward_tensor.mean().item() * len(rewards)
        mine_total += ((reward_tensor <= -0.99) | (reward_tensor == -0.5)).sum().item()
        sample_total += len(rewards)
        if step % 20 == 0 or step == start_step + 1:
            elapsed = time.monotonic() - started
            rate = elapsed / max(1, step - start_step)
            log(f"[GRPO:{args.reward}] step={step} "
                f"mean_reward={reward_total/sample_total:.4f} "
                f"mine_or_bad={mine_total/sample_total:.3f} "
                f"entropy={entropy_total/total_mask.item():.3f} "
                f"low_variance_groups={low_groups}/{args.prompts_per_step} "
                f"mines={mines_for(step)} seconds_per_step={rate:.2f}")
            reward_total = mine_total = sample_total = 0.0
        should_save = step % args.eval_every == 0 or step == args.steps
        budget_hit = time_budget_expired(started, args.time_budget_min)
        if should_save or budget_hit:
            save_checkpoint(checkpoint_dir, model, opt, scaler, step, args.hub_repo)
            log(f"[GRPO:{args.reward}] checkpoint_saved step={step} path={checkpoint_dir}")
        if budget_hit:
            break


if __name__ == "__main__":
    main()
