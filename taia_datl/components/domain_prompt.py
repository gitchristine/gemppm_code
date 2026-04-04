"""
Domain Prompt Generator
NOTE! Code left for documentation but this code is no longer used, domain_id features are generated in data_functions/1_data_prep

"""

from __future__ import annotations

import json
from typing import Optional

import pandas as pd


class DomainPromptGenerator:

    def __init__(
        self,
        model=None,
        tokenizer=None,
        max_tokens: int = 256,
        temperature: float = 0.3,
    ):

        self.model = model
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.temperature = temperature

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, df: pd.DataFrame, dataset_name: str = "") -> dict:

        if self.model is not None and self.tokenizer is not None:
            return self._generate_with_llm(df, dataset_name)
        return self._generate_heuristic(df, dataset_name)

    # ------------------------------------------------------------------
    # LLM-based generation
    # ------------------------------------------------------------------

    def _generate_with_llm(self, df: pd.DataFrame, dataset_name: str) -> dict:
        """Ask the TinyLLM to produce a domain descriptor JSON."""
        import torch


        activities = sorted(df["activity"].unique().tolist())
        n_cases = df["case_id"].nunique()
        sample_trace = (
            df.groupby("case_id")
            .apply(lambda g: g.sort_values("timestamp")["activity"].tolist())
            .iloc[0][:8]  # first 8 events of first case
        )

        prompt = (
            f"You are a process-mining expert.\n\n"
            f"Dataset: {dataset_name}\n"
            f"Cases: {n_cases}\n"
            f"Activities ({len(activities)}): {', '.join(activities[:20])}\n"
            f"Sample trace: {' → '.join(sample_trace)}\n\n"
            f"Produce a JSON object with exactly these keys:\n"
            f'  "label"   — short human-readable domain label\n'
            f'  "A_dom"   — list of the most important domain activities\n'
            f'  "t_scale" — typical time scale ("seconds","minutes","hours","days")\n'
            f'  "notes"   — 1-2 sentence business-context note\n\n'
            f"Respond ONLY with the JSON object, no markdown.\n"
        )

        tokens = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        tokens = {k: v.to(device) for k, v in tokens.items()}

        with torch.no_grad():
            out = self.model.generate(
                **tokens,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True,
            )

        response = self.tokenizer.decode(out[0][tokens["input_ids"].shape[1]:],
                                          skip_special_tokens=True).strip()

        # Parse JSON from response
        try:
            descriptor = json.loads(response)
        except json.JSONDecodeError:
            # If LLM didn't return clean JSON, fall back to heuristic
            # print("[DomainPrompt] invalid -> using heuristic")
            return self._generate_heuristic(df, dataset_name)


        for key in ("label", "A_dom", "t_scale", "notes"):
            if key not in descriptor:
                descriptor[key] = ""

        print(f"[DomainPrompt] Generated descriptor: {descriptor['label']}")
        return descriptor

    # ------------------------------------------------------------------
    # Heuristic fallback (no LLM needed)
    # ------------------------------------------------------------------

    def _generate_heuristic(self, df: pd.DataFrame, dataset_name: str) -> dict:

        activities = sorted(df["activity"].unique().tolist())

        # Estimate time scale from median inter-event duration
        durations = df.groupby("case_id")["timestamp"].diff().dt.total_seconds().dropna()
        median_secs = durations.median() if len(durations) > 0 else 0
        if median_secs < 60:
            t_scale = "seconds"
        elif median_secs < 3600:
            t_scale = "minutes"
        elif median_secs < 86400:
            t_scale = "hours"
        else:
            t_scale = "days"

        descriptor = {
            "label": dataset_name or "Unknown Process",
            "A_dom": activities,
            "t_scale": t_scale,
            "notes": f"Auto-generated from {df['case_id'].nunique()} cases "
                     f"with {len(activities)} activity types.",
        }
        print(f"[DomainPrompt] Heuristic descriptor: {descriptor['label']}")
        return descriptor
