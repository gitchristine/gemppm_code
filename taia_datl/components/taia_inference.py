"""
TAIA Selective-Attention Inference
"""

from __future__ import annotations

from typing import Optional

import torch


def taia_selective_predict(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 0.1,
    drop_ffn: bool = True,
) -> str:

    device = next(model.parameters()).device

    # Tokenise
    tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    tokens = {k: v.to(device) for k, v in tokens.items()}

    # Optionally zero FFN LoRA deltas
    ffn_keywords = {"gate_proj", "up_proj", "down_proj"}
    saved_params: dict[str, torch.Tensor] = {}

    if drop_ffn:
        for name, param in model.named_parameters():
            is_lora = ("lora_A" in name) or ("lora_B" in name)
            is_ffn = any(kw in name for kw in ffn_keywords)
            if is_lora and is_ffn:
                saved_params[name] = param.data.clone()
                param.data.zero_()

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else 1.0,
            do_sample=(temperature > 0),
        )

    # Restore FFN weights
    if drop_ffn:
        for name, original in saved_params.items():
            dict(model.named_parameters())[name].data.copy_(original)

    # Decode only the new tokens
    generated = outputs[0][tokens["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text


def build_taia_prompt(
    trace_activities: list[str],
    domain_descriptor: dict,
    few_shot_block: str = "",
    activity_vocab: list[str] | None = None,
) -> str:

    d = domain_descriptor

    # Header
    lines = [
        "You are an expert process-mining analyst.\n",
        f"Domain: {d.get('label', 'Unknown')}",
        f"Time scale: {d.get('t_scale', 'unknown')}",
    ]
    if d.get("notes"):
        lines.append(f"Context: {d['notes']}")
    lines.append("")

    # Activity vocabulary
    if activity_vocab:
        lines.append(f"Possible activities ({len(activity_vocab)}): "
                      f"{', '.join(activity_vocab)}")
        lines.append("")

    # Few-shot examples
    if few_shot_block:
        lines.append(few_shot_block)

    # Current trace
    lines.append("Current trace (chronological):")
    for i, act in enumerate(trace_activities, 1):
        lines.append(f"  {i}. {act}")
    lines.append("")

    lines.append("Predict the single most likely NEXT activity.  "
                 "Reply with the activity name only, nothing else.")

    # TODO append remaining time prediction prompt here.

    return "\n".join(lines)
