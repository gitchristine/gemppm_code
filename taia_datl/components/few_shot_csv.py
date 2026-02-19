"""
Few-Shot CSV Upload  [*]
TODO remove from the main pipeline??
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


class FewShotCSVLoader:

    REQUIRED_COLS = {"trace", "next"}

    def __init__(self, csv_path: Optional[str] = None):
        """
        Args:
            csv_path: Path to the CSV file (None = disabled).
        """
        self.csv_path = csv_path
        self.examples: list[dict] = []

        if csv_path is not None:
            self._load(csv_path)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def _load(self, csv_path: str) -> None:
        """Read the CSV and validate columns."""
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Few-shot CSV not found: {csv_path}")

        df = pd.read_csv(path)
        missing = self.REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"Few-shot CSV missing columns: {missing}")

        for _, row in df.iterrows():
            ex = {
                "trace": row["trace"].split("|"),
                "next": row["next"],
                "domain": row.get("domain", ""),
                "time_left": row.get("time_left", None),
            }
            self.examples.append(ex)

        print(f"[FewShot] Loaded {len(self.examples)} exemplars from {csv_path}")

    # ------------------------------------------------------------------
    # Filter by domain
    # TODO optional: check if this improves performance at all?
    # ------------------------------------------------------------------

    def get_examples(
        self,
        domain: Optional[str] = None,
        max_k: int = 5,
    ) -> list[dict]:

        if not self.examples:
            return []

        pool = self.examples
        if domain:
            pool = [e for e in pool if e["domain"] == domain] or pool

        return pool[:max_k]

    # ------------------------------------------------------------------
    # Format for prompt
    # ------------------------------------------------------------------

    def format_for_prompt(
        self,
        domain: Optional[str] = None,
        max_k: int = 5,
    ) -> str:
        examples = self.get_examples(domain, max_k)
        if not examples:
            return ""

        lines = ["Few-shot examples:\n"]
        for i, ex in enumerate(examples, 1):
            trace_str = " -> ".join(ex["trace"])
            lines.append(f"  Example {i}: {trace_str}")
            lines.append(f"    Next activity: {ex['next']}")
            if ex.get("time_left") is not None:
                lines.append(f"    Remaining time: {ex['time_left']}")
            lines.append("")

        return "\n".join(lines)
