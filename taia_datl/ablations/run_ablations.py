"""
run_ablations.py — Ablation Study Runner
==========================================
Executes a FULL ablation study: runs the pipeline once with all
contributions enabled (baseline), then re-runs it six times — each
time disabling exactly one contribution.

Results are collected into a summary table and saved as both JSON and
a human-readable text report.

Usage:
python -m taia_datl.ablations.run_ablations--dataset bpi2012 --skip-data-prep

Outputs:
    results/{dataset}_ablation_summary.json (TODO CSV -> Viz)
    results/{dataset}_ablation_summary.txt (+ Readable output)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from taia_datl.config import TAIADATLConfig
from taia_datl.pipeline import TAIADATLPipeline



ABLATIONS = {
    "baseline": {
        # All contributions enabled
    },
    "no_taia": {
        "no_taia": True,
    },
    "no_datl": {
        "no_datl": True,
    },
    "no_faiss": {
        "no_faiss": True,
    },
    "no_domain_prompt": {
        "no_domain_prompt": True,
    },
    "no_few_shot": {
        "no_few_shot": True,
    },
    "lstm_backbone": {
        "backbone_lstm": True,
    },
}


def run_all_ablations(
    dataset: str,
    filepath: str | None = None,
    skip_data_prep: bool = True,
) -> dict:
    summary = {}

    for name, overrides in ABLATIONS.items():
        print(f"  ABLATION: {name}")

        cfg = TAIADATLConfig()
        cfg.dataset_name = dataset
        # Apply ablation overrides
        for key, val in overrides.items():
            setattr(cfg, key, val)

        pipeline = TAIADATLPipeline(cfg)
        results = pipeline.run(filepath=filepath, skip_data_prep=skip_data_prep)
        summary[name] = results

    return summary

# Summary of the outputs
def print_summary(summary: dict) -> str:
    lines = []
    lines.append("  ABLATION STUDY SUMMARY")
    lines.append(f"{'Condition':<22} {'Accuracy':>10} {'MAE':>10} {'Δ Acc':>10} {'Δ MAE':>10}")

    baseline_acc = summary.get("baseline", {}).get("accuracy", 0)
    baseline_mae = summary.get("baseline", {}).get("mae", 0)

    for name, res in summary.items():
        acc = res.get("accuracy", 0)
        mae = res.get("mae", 0)
        d_acc = acc - baseline_acc
        d_mae = mae - baseline_mae
        lines.append(f"{name:<22} {acc:>10.4f} {mae:>10.4f} {d_acc:>+10.4f} {d_mae:>+10.4f}")
    lines.append("Δ Acc < 0 means this contribution HELPED accuracy.")
    lines.append("Δ MAE > 0 means this contribution HELPED (lower MAE = better).")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run TAIA-DATL ablations")
    parser.add_argument("--dataset", type=str, default="bpi2012")
    parser.add_argument("--filepath", type=str, default=None,
                        help="Raw XES/CSV path (only if data prep needed)")
    parser.add_argument("--skip-data-prep", action="store_true",
                        help="Use existing preprocessed data")
    args = parser.parse_args()

    start = time.time()
    summary = run_all_ablations(
        dataset=args.dataset,
        filepath=args.filepath,
        skip_data_prep=args.skip_data_prep,
    )
    elapsed = time.time() - start

    # Print and save
    report = print_summary(summary)
    print(report)

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    json_path = out_dir / f"{args.dataset}_ablation_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"JSON results here>>>> {json_path}")

    txt_path = out_dir / f"{args.dataset}_ablation_summary.txt"
    with open(txt_path, "w") as f:
        f.write(report)
    print(f"Text report here>>>> {txt_path}")
    # print(f"\nTotal ablation study time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
