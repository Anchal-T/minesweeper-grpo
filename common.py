"""Shared model/tokenizer utilities for GPT-2 on torch 1.13 + K80 (fp16 autocast, eager attention)."""
import os

os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(__file__), "hf-cache"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "gpt2"  # 124M params, small enough for K80 RL
PAD_ID = 50256  # eos; we mask pads out of loss/logprob anyway


def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.pad_token = tok.eos_token  # gpt2 has no pad token
    return tok


def load_model(device="cuda"):
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.to(device)
    return model


def encode_prompt(tok, prompt, device):
    ids = tok(prompt, return_tensors="pt").input_ids
    return ids.to(device)


@torch.no_grad()
def sample_completions(model, tok, prompts, max_new_tokens=8, temperature=1.0, greedy=False):
    """Batch-sample one completion per prompt. Returns list of strings + token ids."""
    device = next(model.parameters()).device
    old_side = tok.padding_side
    tok.padding_side = "left"
    try:
        enc = tok(prompts, return_tensors="pt", padding=True).to(device)
    finally:
        tok.padding_side = old_side
    input_ids, attn = enc.input_ids, enc.attention_mask
    gen_kwargs = dict(input_ids=input_ids, attention_mask=attn,
                      max_new_tokens=max_new_tokens, do_sample=not greedy,
                      pad_token_id=PAD_ID)
    if not greedy:
        gen_kwargs.update(temperature=temperature, top_k=0, top_p=1.0)
    out = model.generate(**gen_kwargs)
    gen = out[:, input_ids.shape[1]:]
    texts = tok.batch_decode(gen, skip_special_tokens=True)
    return texts, gen


def token_logprobs(model, ctx_ids, answer_ids, attn_mask=None):
    """Log-prob of answer_ids given ctx_ids (single batched sequence each).

    ctx_ids, answer_ids: [B, Lc], [B, La] (right-padded with PAD_ID)
    Returns [B, La] log-probs and the answer attention mask.
    """
    B, Lc = ctx_ids.shape
    La = answer_ids.shape[1]
    full = torch.cat([ctx_ids, answer_ids], dim=1)
    # right-pad context so all rows align: build attention mask from PAD
    attn = (full != PAD_ID).long()
    # force first token of each row attended
    attn[:, 0] = 1
    # match generate(): position ids must count only real tokens, otherwise
    # answer tokens get shifted positions on shorter-than-max rows
    position_ids = attn.cumsum(-1) - 1
    position_ids.clamp_(min=0)
    logits = model(input_ids=full, attention_mask=attn, position_ids=position_ids).logits
    # predict answer token t from position Lc-1+t
    pos = torch.arange(Lc - 1, Lc + La - 1, device=full.device)
    pred_logits = logits[:, pos, :]  # [B, La, V]
    logprobs = torch.log_softmax(pred_logits.float(), dim=-1)
    ans_lp = torch.gather(logprobs, 2, answer_ids.clamp_max(50256).unsqueeze(-1)).squeeze(-1)
    ans_mask = (answer_ids != PAD_ID).float()
    return ans_lp, ans_mask
