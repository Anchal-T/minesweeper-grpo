"""Shared Qwen, tokenizer, sampling, and answer-position scoring helpers."""
import os

os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(__file__), "hf-cache"))
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as transformers_logging
from peft import LoraConfig, get_peft_model, PeftModel

transformers_logging.disable_progress_bar()

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def load_tokenizer(model_name=MODEL_NAME):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def load_model(device="cuda", model_name=MODEL_NAME, adapter=None):
    kwargs = {"torch_dtype": torch.float16 if str(device).startswith("cuda") else torch.float32}
    if "Qwen" in model_name:
        kwargs["attn_implementation"] = "sdpa"
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=True)
    else:
        config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                            "gate_proj", "up_proj", "down_proj"],
                            task_type="CAUSAL_LM")
        model = get_peft_model(model, config)
    return model.to(device)


def format_prompt(tok, prompt):
    return tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True)


def encode_prompt(tok, prompt, device):
    ids = tok(format_prompt(tok, prompt), return_tensors="pt",
              add_special_tokens=False).input_ids
    return ids.to(device)


@torch.no_grad()
def sample_completions(model, tok, prompts, max_new_tokens=8, temperature=1.0,
                       greedy=False, return_inputs=False, top_p=1.0):
    """Sample one completion per prompt and optionally return rollout inputs."""
    device = next(model.parameters()).device
    old_side = tok.padding_side
    tok.padding_side = "left"
    try:
        enc = tok([format_prompt(tok, p) for p in prompts],
                  return_tensors="pt", padding=True,
                  add_special_tokens=False).to(device)
    finally:
        tok.padding_side = old_side
    input_ids, attn = enc.input_ids, enc.attention_mask
    kwargs = dict(input_ids=input_ids, attention_mask=attn,
                  max_new_tokens=max_new_tokens, do_sample=not greedy,
                  pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    if not greedy:
        kwargs.update(temperature=temperature, top_k=0, top_p=top_p)
    out = model.generate(**kwargs)
    gen = out[:, input_ids.shape[1]:]
    texts = tok.batch_decode(gen, skip_special_tokens=True)
    if return_inputs:
        return texts, gen, input_ids, attn
    return texts, gen


def sample_grouped_completions(model, tok, prompts, group, max_new_tokens=6,
                               temperature=1.0):
    """Sample `group` completions per prompt with one shared prefill per prompt.

    Replicates model.generate() for the GRPO rollout call site: fp32 logits,
    the generation config's repetition penalty over prompt+generated tokens,
    temperature, top_k=0/top_p=1.0 (both inactive), eos stop with pad fill.
    Returns texts, generated ids, and the group-replicated padded contexts.
    """
    device = next(model.parameters()).device
    old_side = tok.padding_side
    tok.padding_side = "left"
    try:
        enc = tok([format_prompt(tok, p) for p in prompts],
                  return_tensors="pt", padding=True,
                  add_special_tokens=False).to(device)
    finally:
        tok.padding_side = old_side
    transformer, lm_head = _transformer_and_head(model)
    ctx_ids = enc.input_ids.repeat_interleave(group, dim=0)
    ctx_attn = enc.attention_mask.repeat_interleave(group, dim=0)
    position_ids = enc.attention_mask.cumsum(-1) - 1
    position_ids.clamp_(min=0)
    prefill = transformer(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                          position_ids=position_ids, use_cache=True)
    cache = prefill.past_key_values
    cache.batch_repeat_interleave(group)
    rows = ctx_ids.shape[0]
    penalty = model.generation_config.repetition_penalty
    eos_id, pad_id = tok.eos_token_id, tok.pad_token_id
    generated = ctx_ids.new_zeros((rows, max_new_tokens))
    unfinished = torch.ones(rows, dtype=torch.long, device=device)
    full_ids, full_attn = ctx_ids, ctx_attn
    logits = lm_head(prefill.last_hidden_state[:, -1, :])
    logits = logits.repeat_interleave(group, dim=0).float()
    for step in range(max_new_tokens):
        if penalty is not None and penalty != 1.0:
            score = torch.gather(logits, 1, full_ids)
            score = torch.where(score < 0, score * penalty, score / penalty)
            logits = logits.scatter(1, full_ids, score)
        if temperature != 1.0:
            logits = logits / temperature
        next_tokens = torch.multinomial(
            torch.softmax(logits, dim=-1), 1).squeeze(1)
        next_tokens = next_tokens * unfinished + pad_id * (1 - unfinished)
        generated[:, step] = next_tokens
        unfinished = unfinished * (next_tokens != eos_id).long()
        if step == max_new_tokens - 1 or not unfinished.any():
            break
        full_ids = torch.cat([full_ids, next_tokens.unsqueeze(1)], dim=1)
        full_attn = torch.cat([full_attn, full_attn.new_ones((rows, 1))], dim=1)
        positions = full_attn.sum(-1, keepdim=True) - 1
        hidden = transformer(input_ids=next_tokens.unsqueeze(1),
                             attention_mask=full_attn, position_ids=positions,
                             past_key_values=cache, use_cache=True).last_hidden_state
        logits = lm_head(hidden[:, -1, :]).float()
    texts = tok.batch_decode(generated, skip_special_tokens=True)
    return texts, generated, ctx_ids, ctx_attn


def _transformer_and_head(model):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.model, base.lm_head


def _answer_logits(model, full, attn, answer_start, answer_length):
    """Run the transformer and project only positions predicting answers."""
    transformer, lm_head = _transformer_and_head(model)
    position_ids = attn.cumsum(-1) - 1
    position_ids.clamp_(min=0)
    hidden = transformer(input_ids=full, attention_mask=attn,
                         position_ids=position_ids, use_cache=False).last_hidden_state
    positions = torch.arange(answer_start - 1,
                             answer_start + answer_length - 1,
                             device=full.device)
    return lm_head(hidden[:, positions, :])


def _cached_answer_logits(model, ctx_ids, answer_ids, ctx_attn, ans_attn,
                          context_repeats):
    """Score answers against one shared prefill per group of identical contexts.

    Rows must repeat each context contiguously `context_repeats` times, as the
    GRPO rollout does. Equivalent math to the full-sequence path; in fp16 the
    two paths differ by ordinary fp16 reduction-order noise, which is why the
    fp32 CPU smoke check exists.
    """
    if ctx_ids.shape[0] % context_repeats:
        raise ValueError("batch size must be divisible by context_repeats")
    transformer, lm_head = _transformer_and_head(model)
    unique_ctx = ctx_ids[::context_repeats]
    unique_attn = ctx_attn[::context_repeats]
    position_ids = unique_attn.cumsum(-1) - 1
    position_ids.clamp_(min=0)
    prefill = transformer(input_ids=unique_ctx, attention_mask=unique_attn,
                          position_ids=position_ids, use_cache=True)
    first_logits = lm_head(prefill.last_hidden_state[:, -1:, :])
    if answer_ids.shape[1] == 1:
        return first_logits.repeat_interleave(context_repeats, dim=0)
    cache = prefill.past_key_values
    cache.batch_repeat_interleave(context_repeats)
    answer_inputs = answer_ids[:, :-1]
    attention = torch.cat([ctx_attn, ans_attn[:, :-1]], dim=1)
    answer_positions = attention.cumsum(-1)[:, ctx_ids.shape[1]:] - 1
    answer_positions.clamp_(min=0)
    hidden = transformer(input_ids=answer_inputs, attention_mask=attention,
                         position_ids=answer_positions, past_key_values=cache,
                         use_cache=True).last_hidden_state
    return torch.cat([first_logits.repeat_interleave(context_repeats, dim=0),
                      lm_head(hidden)], dim=1)


def token_logprobs(model, ctx_ids, answer_ids, attn_mask=None, answer_mask=None,
                   pad_id=None, context_repeats=1):
    """Return answer log-probs and entropy, projecting logits only at answer positions.

    context_repeats > 1 shares one prefill across each contiguous group of
    identical contexts instead of forwarding the full sequence per row.
    """
    _, context_length = ctx_ids.shape
    answer_length = answer_ids.shape[1]
    pad_id = 0 if pad_id is None else pad_id
    ctx_attn = (ctx_ids != pad_id).long() if attn_mask is None else attn_mask.long()
    ans_attn = ((answer_ids != pad_id).long() if answer_mask is None
                else answer_mask.long())
    if context_repeats == 1:
        full = torch.cat([ctx_ids, answer_ids], dim=1)
        attn = torch.cat([ctx_attn, ans_attn], dim=1)
        logits = _answer_logits(model, full, attn, context_length, answer_length)
    else:
        logits = _cached_answer_logits(model, ctx_ids, answer_ids, ctx_attn,
                                       ans_attn, context_repeats)
    logits = logits.float()
    log_z = torch.logsumexp(logits, dim=-1)
    token_logits = torch.gather(logits, -1, answer_ids.unsqueeze(-1)).squeeze(-1)
    logprobs = token_logits - log_z
    probs = torch.exp(logits - log_z.unsqueeze(-1))
    entropy = log_z - (probs * logits).sum(dim=-1)
    return logprobs, ans_attn.float(), entropy


def answer_token_padded(tok, texts, device):
    enc = tok(texts, return_tensors="pt", padding=True).to(device)
    return enc.input_ids, enc.attention_mask


def prompt_token_padded(tok, prompts, device):
    enc = tok([format_prompt(tok, p) for p in prompts],
              return_tensors="pt", padding=True,
              add_special_tokens=False).to(device)
    return enc.input_ids, enc.attention_mask
