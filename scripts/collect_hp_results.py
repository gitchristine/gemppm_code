"""
collect_hp_results.py
======================
Aggregates all per-(dataset, model) JSON files produced by
hyperparameter_tuning.py into a single summary table.

Usage:
    python scripts/collect_hp_results.py --input-dir results/hp_tuning

Outputs:
    results/hp_tuning/hp_summary.csv   — one row per (dataset, model)
    results/hp_tuning/hp_summary.json  — same data as JSON
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default="results/hp_tuning")
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    rows = []

    for jf in sorted(input_dir.glob("*_hp_results.json")):
        with open(jf) as f:
            data = json.load(f)

        row = {
            "dataset":           data["dataset"],
            "model_variant":     data["model_variant"],
            "best_val_accuracy": data["best_val_accuracy"],
            "best_val_mae":      data["best_val_mae"],
        }
        # Flatten best_params into individual columns
        for k, v in (data.get("best_params") or {}).items():
            row[f"best_{k}"] = v
        rows.append(row)

    if not rows:
        print(f"No result files found in {input_dir}")
        return

    df = pd.DataFrame(rows).sort_values(["dataset", "model_variant"])
    csv_path  = input_dir / "hp_summary.csv"
    json_path = input_dir / "hp_summary.json"

    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)

    print(df.to_string(index=False))
    print(f"\nSaved → {csv_path}")
    print(f"Saved → {json_path}")


if __name__ == "__main__":
    main()
