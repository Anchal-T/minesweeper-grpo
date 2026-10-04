"""Small from-scratch GRPO trainer for Minesweeper answers."""
import argparse
import os
import re
import time

import torch

from checkpoint import (load_training_state, restore_from_hub, save_checkpoint,
                        time_budget_expired)
from common import (MODEL_NAME, load_model, load_tokenizer, prompt_token_padded,
                    sample_grouped_completions, token_logprobs)
from env import Minesweeper, parse_move, posterior_reward, step_reward


def rollout_batch(model, tok, n_prompts, group, temperature, device,
                  max_new_tokens=6, n_mines=6, reward_mode="truth",
                  answer_format="move", format_reward=0.0, phase_times=None):
    boards = [Minesweeper(n_mines=n_mines) for _ in range(n_prompts)]
    prompts = [b.prompt() for b in boards]
    generation_started = time.perf_counter() if phase_times is not None else 0.0
    texts, gen_ids, ctx_ids, ctx_attn = sample_grouped_completions(
        model, tok, prompts, group, max_new_tokens=max_new_tokens,
        temperature=temperature)
    if phase_times is not None:
        phase_times["generation"] = time.perf_counter() - generation_started
    reward_started = time.perf_counter() if phase_times is not None else 0.0
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
    if phase_times is not None:
        phase_times["reward"] = time.perf_counter() - reward_started
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
    ap.add_argument("--hub-every", type=int, default=500)
    ap.add_argument("--time-budget-min", type=float, default=690)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--profile-phases", action="store_true",
                    help="report rollout, reward, policy, reference, and save timings")
    args = ap.parse_args()
    if args.group <= 0:
        ap.error("--group must be positive")
    if args.hub_every <= 0:
        ap.error("--hub-every must be positive")
    policy_batch_size = args.micro_batch // args.group * args.group
    if policy_batch_size == 0:
        ap.error("--micro-batch must be at least one group")
    if args.answer_format == "cot" and args.reward == "truth":
        ap.error("CoT phase 2 uses posterior or vpr rewards, not truth reward")
    if args.max_new_tokens is None:
        args.max_new_tokens = 160 if args.answer_format == "cot" else 6
    budget_started = time.monotonic()

    if args.profile_phases and args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            ap.error("phase profiling requested but CUDA is unavailable")
        total_memory = torch.cuda.get_device_properties(args.device).total_memory
        print(f"[PROBE] gpu={torch.cuda.get_device_name(args.device)} "
              f"capability={torch.cuda.get_device_capability(args.device)} "
              f"compiled_arches={torch.cuda.get_arch_list()} "
              f"total_memory_gib={total_memory / (1024 ** 3):.2f}", flush=True)
        product = None
        try:
            matrix = torch.empty((1024, 1024), device=args.device)
            torch.cuda.synchronize(args.device)
            matmul_started = time.perf_counter()
            product = matrix @ matrix
            torch.cuda.synchronize(args.device)
            print(f"[PROBE] float32_matmul_ms="
                  f"{1000 * (time.perf_counter() - matmul_started):.2f}",
                  flush=True)
            del matrix, product
        except RuntimeError as error:
            print(f"[PROBE] float32_matmul=FAILED error={error}", flush=True)
            raise

    checkpoint_dir = os.path.join(args.out_dir, "last")
    if args.resume:
        restore_from_hub(checkpoint_dir, args.hub_repo)
    initialization_started = time.monotonic()
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

    if args.profile_phases:
        _, prompt_mask = prompt_token_padded(
            tok, [Minesweeper().prompt()], args.device)
        log(f"[PROBE] prompt_tokens={prompt_mask.sum().item():.0f} "
            f"tokenizer={args.model}")
        log(f"[PROBE] initialization_seconds="
            f"{time.monotonic() - initialization_started:.1f}")
        if args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(args.device)

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
    phase_totals = {name: 0.0 for name in
                    ("rollout", "generation", "reward", "policy", "ref", "save")}
    profiled_steps = profiled_groups = low_variance_groups = 0
    prompt_token_total = 0.0
    model.eval()  # Keep LoRA dropout disabled so scoring matches rollout policy.
    for step in range(start_step + 1, args.steps + 1):
        temperature = args.temperature
        if args.temp_end is not None:
            fraction = min(1.0, (step - 1) / max(1, args.steps - 1))
            temperature += fraction * (args.temp_end - args.temperature)
        model.eval()
        if args.profile_phases and args.device.startswith("cuda"):
            torch.cuda.synchronize(args.device)
        rollout_started = time.perf_counter()
        rollout_times = {} if args.profile_phases else None
        with torch.no_grad(), torch.autocast(
                "cuda", dtype=torch.float16, enabled=args.device.startswith("cuda")):
            (_, _, _, answer_ids, ctx_ids, ctx_attn, answer_mask, rewards) = rollout_batch(
                model, tok, args.prompts_per_step, args.group, temperature,
                args.device, args.max_new_tokens, n_mines=mines_for(step),
                reward_mode=args.reward, answer_format=args.answer_format,
                format_reward=args.format_reward, phase_times=rollout_times)
            roll_lp = None
            if args.epochs_per_batch > 1:
                roll_lps = []
                for offset in range(0, ctx_ids.shape[0], policy_batch_size):
                    sl = slice(offset, offset + policy_batch_size)
                    roll_lps.append(token_logprobs(
                        model, ctx_ids[sl], answer_ids[sl],
                        attn_mask=ctx_attn[sl], answer_mask=answer_mask[sl],
                        pad_id=tok.pad_token_id, context_repeats=args.group)[0])
                roll_lp = torch.cat(roll_lps, dim=0).detach()
        if args.profile_phases:
            assert rollout_times is not None
            if args.device.startswith("cuda"):
                torch.cuda.synchronize(args.device)
            phase_totals["rollout"] += time.perf_counter() - rollout_started
            phase_totals["generation"] += rollout_times["generation"]
            phase_totals["reward"] += rollout_times["reward"]
            prompt_token_total += ctx_attn.sum().item() / ctx_attn.shape[0]
            profiled_steps += 1
            profiled_groups += args.prompts_per_step

        advantages, low_groups = compute_advantages(
            rewards, args.group, args.adv_norm, args.min_group_std)
        advantages = advantages.to(args.device)
        total_mask = answer_mask.sum().clamp_min(1)
        prompt_token_counts = answer_mask.view(
            args.prompts_per_step, args.group, -1).sum(dim=(1, 2)).clamp_min(1)
        entropy_total = 0.0
        model.eval()
        ref_lp = None
        if args.kl_coef > 0:
            if args.profile_phases and args.device.startswith("cuda"):
                torch.cuda.synchronize(args.device)
            ref_started = time.perf_counter() if args.profile_phases else 0.0
            with torch.no_grad(), model.disable_adapter(), torch.autocast(
                    "cuda", dtype=torch.float16,
                    enabled=args.device.startswith("cuda")):
                ref_lps = []
                for offset in range(0, ctx_ids.shape[0], policy_batch_size):
                    sl = slice(offset, offset + policy_batch_size)
                    ref_lps.append(token_logprobs(
                        model, ctx_ids[sl], answer_ids[sl],
                        attn_mask=ctx_attn[sl], answer_mask=answer_mask[sl],
                        pad_id=tok.pad_token_id, context_repeats=args.group)[0])
                ref_lp = torch.cat(ref_lps, dim=0)
            if args.profile_phases:
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize(args.device)
                phase_totals["ref"] += time.perf_counter() - ref_started
        for _ in range(args.epochs_per_batch):
            if args.profile_phases and args.device.startswith("cuda"):
                torch.cuda.synchronize(args.device)
            policy_started = time.perf_counter() if args.profile_phases else 0.0
            for offset in range(0, ctx_ids.shape[0], policy_batch_size):
                sl = slice(offset, offset + policy_batch_size)
                with torch.autocast("cuda", dtype=torch.float16,
                                    enabled=args.device.startswith("cuda")):
                    lp, mask, entropy = token_logprobs(
                        model, ctx_ids[sl], answer_ids[sl], attn_mask=ctx_attn[sl],
                        answer_mask=answer_mask[sl], pad_id=tok.pad_token_id,
                        context_repeats=args.group)
                    adv = advantages[sl].unsqueeze(1)
                    policy_terms = adv * lp
                    if roll_lp is not None:
                        ratio = torch.exp((lp - roll_lp[sl]).clamp(-20, 20))
                        clipped = ratio.clamp(1 - args.clip, 1 + args.clip)
                        policy_terms = torch.minimum(ratio * adv, clipped * adv)
                    if args.prompt_mean_loss:
                        rows = torch.arange(offset, min(offset + policy_batch_size,
                                                        ctx_ids.shape[0]), device=args.device)
                        row_weights = (1.0 / (args.prompts_per_step *
                                              prompt_token_counts[rows // args.group])).unsqueeze(1)
                        loss = -(policy_terms * mask * row_weights).sum()
                    else:
                        row_weights = None
                        loss = -(policy_terms * mask).sum() / total_mask
                    entropy_total += float((entropy.detach() * mask).sum())
                    if args.kl_coef > 0:
                        assert ref_lp is not None
                        log_ratio = (lp - ref_lp[sl]).clamp(-20, 20)
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
            if args.profile_phases:
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize(args.device)
                phase_totals["policy"] += max(
                    0.0, time.perf_counter() - policy_started)

        if args.profile_phases:
            low_variance_groups += low_groups
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
        budget_hit = time_budget_expired(budget_started, args.time_budget_min)
        if should_save or budget_hit:
            if args.profile_phases and args.device.startswith("cuda"):
                torch.cuda.synchronize(args.device)
            save_started = time.perf_counter()
            upload_now = (step % args.hub_every == 0 or step == args.steps
                          or budget_hit)
            save_checkpoint(checkpoint_dir, model, opt, scaler, step,
                            args.hub_repo if upload_now else None)
            if args.profile_phases:
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize(args.device)
                phase_totals["save"] += time.perf_counter() - save_started
            log(f"[GRPO:{args.reward}] checkpoint_saved step={step} path={checkpoint_dir}")
        if budget_hit:
            break

    if args.profile_phases:
        denominator = max(1, profiled_steps)
        phase_summary = " ".join(
            f"{name}={phase_totals[name] / denominator:.2f}s"
            for name in ("rollout", "generation", "reward", "policy", "ref", "save"))
        log(f"[PROFILE] steps={profiled_steps} mean_prompt_tokens="
            f"{prompt_token_total / denominator:.1f} "
            f"low_variance_groups={low_variance_groups}/{profiled_groups} "
            f"{phase_summary}")
        if args.device.startswith("cuda"):
            peak_allocated = torch.cuda.max_memory_allocated(args.device)
            peak_reserved = torch.cuda.max_memory_reserved(args.device)
            total_memory = torch.cuda.get_device_properties(args.device).total_memory
            log(f"[PROFILE] max_memory_allocated_gib={peak_allocated / (1024 ** 3):.2f} "
                f"max_memory_reserved_gib={peak_reserved / (1024 ** 3):.2f} "
                f"approx_allocated_headroom_gib="
                f"{(total_memory - peak_allocated) / (1024 ** 3):.2f}")


if __name__ == "__main__":
    main()
