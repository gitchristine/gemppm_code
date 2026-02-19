"""
TinyLLM Backbone
"""

from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_tinyllm(
    model_name: str = "arnir0/Tiny-LLM",
    cache_dir: Optional[str] = None,
    device_map: str = "auto",
    torch_dtype: str = "float16",
    load_in_4bit: bool = False,
):

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(torch_dtype, torch.float16)

    # TODO optional?
    quant_config = None
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    print(f"[TinyLLM] Loading {model_name}  dtype={torch_dtype}  4bit={load_in_4bit}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=cache_dir, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        device_map=device_map,
        torch_dtype=dtype,
        quantization_config=quant_config,
        trust_remote_code=True,
    )

    print(f"[TinyLLM] Loaded — {sum(p.numel() for p in model.parameters()):,} params")
    return model, tokenizer



def apply_lora(
    model,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: list[str] | None = None,
):

    from peft import LoraConfig, get_peft_model, TaskType

    if target_modules is None:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",   # attention
            "gate_proj", "up_proj", "down_proj",        # FFN
        ]

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
    )

    model = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[LoRA] trainable={trainable:,} / total={total:,}  "
          f"({100 * trainable / total:.2f}%)")
    return model



def drop_ffn_deltas(peft_model) -> None:

    ffn_keywords = {"gate_proj", "up_proj", "down_proj"}

    zeroed = 0
    for name, param in peft_model.named_parameters():
        is_lora = ("lora_A" in name) or ("lora_B" in name)
        is_ffn = any(kw in name for kw in ffn_keywords)
        if is_lora and is_ffn:
            param.data.zero_()
            param.requires_grad = False
            zeroed += param.numel()

    # print(f"[TAIA] Dropped {zeroed:,} keeping attention deltas only")



class TinyLLMEncoder(nn.Module):

    def __init__(self, model, tokenizer, pool: str = "last"):

        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.pool = pool

    @torch.no_grad()
    def encode(self, texts: list[str], max_length: int = 1024) -> torch.Tensor:

        tokens = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        device = next(self.model.parameters()).device
        tokens = {k: v.to(device) for k, v in tokens.items()}

        outputs = self.model(**tokens, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]  # (batch, seq, dim)

        if self.pool == "last":
            lengths = tokens["attention_mask"].sum(dim=1) - 1  # (batch,)
            idx = lengths.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, hidden.size(-1))
            pooled = hidden.gather(1, idx).squeeze(1)  # (batch, dim)
        else:
            pooled = hidden[:, -1, :]

        return pooled