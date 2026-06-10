"""
Experiment 3 — Training-Data Sensitivity
==========================================
Runs DATL-TAIA only across incremental training fractions.
Oyamada / MT-RNN results are added manually afterwards.

Training_Percentage labels reflect fraction of total data
(i.e. 20%–100% of the 65% train split → 13%–65% of total).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from taia_datl.config import TAIADATLConfig
from taia_datl.pipeline import (
    DTAIAPipeline, DTAIADataset, compute_domain_thresholds,
)
from taia_datl.components.domain_triplet_loss import compute_rt_bucket_thresholds

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Fractions of the training split to use
TRAINING_FRACTIONS = [0.20, 0.40, 0.60, 0.80, 1.00]

# Fraction of total data that the train split represents
TRAIN_SPLIT_RATIO = 0.65

ALL_DATASETS = [
    "bpi2012",
    "bpi2015_2",
    "bpi2017",
    "bpi2020_dd",
]

DATASET_DISPLAY: Dict[str, str] = {
    "bpi2012":    "BPI2012",
    "bpi2015_2":  "BPI2015_2",
    "bpi2017":    "BPI2017",
    "bpi2020_dd": "BPI2020_DD",
}

N_BOOTSTRAP = 1000
CI_ALPHA    = 0.95

CSV_COLUMNS = [
    "Dataset", "Model", "Training_Percentage",
    "NA_Acc", "NA_CI", "NA_F1", "F1_CI",
    "RT_MAE", "MAE_CI",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def training_pct_label(fraction: float, train_split_ratio: float = TRAIN_SPLIT_RATIO) -> str:
    """Convert fraction-of-train-split to percentage-of-total-data label."""
    pct = round(fraction * train_split_ratio * 100)
    return f"{pct}%"


def subsample_train(
    train_df: pd.DataFrame,
    fraction: float,
    seed:     int = 42,
) -> pd.DataFrame:
    if fraction >= 1.0:
        return train_df
    case_start   = (train_df.groupby("case_id")["timestamp"].min()
                    if "timestamp" in train_df.columns
                    else train_df.groupby("case_id").ngroup().rename("timestamp"))
    sorted_cases = case_start.sort_values().index
    n_keep       = max(1, int(len(sorted_cases) * fraction))
    kept_cases   = sorted_cases[:n_keep]
    return train_df[train_df["case_id"].isin(kept_cases)].copy()


def bootstrap_ci(
    y_true:      np.ndarray,
    y_pred:      np.ndarray,
    metric:      str,
    n_resamples: int = N_BOOTSTRAP,
    alpha:       float = CI_ALPHA,
    rng:         Optional[np.random.Generator] = None,
) -> float:
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(y_true)
    scores = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]
        if metric == "accuracy":
            scores.append((yt == yp).mean())
        elif metric == "mae":
            scores.append(np.abs(yt - yp).mean())
        elif metric == "f1":
            from sklearn.metrics import f1_score
            scores.append(f1_score(yt, yp, average="macro", zero_division=0))
    lo = np.percentile(scores, (1 - alpha) / 2 * 100)
    hi = np.percentile(scores, (1 + alpha) / 2 * 100)
    return round(float((hi - lo) / 2), 3)


# ---------------------------------------------------------------------------
# DATL-TAIA train + evaluate
# ---------------------------------------------------------------------------

def run_dtaia(
    frac_train_df: pd.DataFrame,
    full_train_df: pd.DataFrame,
    val_df:        pd.DataFrame,
    test_df:       pd.DataFrame,
    cfg:           TAIADATLConfig,
    device:        torch.device,
    seed:          int,
) -> Dict[str, float]:
    """Train DATL-TAIA on frac_train_df, evaluate on test_df.
    Returns dict with acc, f1, mae, and their CIs."""
    from sklearn.metrics import f1_score as sk_f1

    num_activities    = int(full_train_df["activity_encoded"].max()) + 1
    domain_thresholds = compute_domain_thresholds(frac_train_df, cfg.max_sequence_length)
    rt_vals           = (frac_train_df["remaining_time"].values
                         if "remaining_time" in frac_train_df.columns
                         else np.zeros(len(frac_train_df)))
    rt_q33, rt_q66    = compute_rt_bucket_thresholds(rt_vals)

    pipeline = DTAIAPipeline(cfg)
    t0 = time.time()
    model, _ = pipeline.train_dtaia(
        frac_train_df, num_activities,
        domain_thresholds, rt_q33, rt_q66,
        val_df=val_df,
    )
    train_time = time.time() - t0

    test_ds = DTAIADataset(
        test_df,
        max_len=cfg.max_sequence_length,
        domain_thresholds=domain_thresholds,
        rt_q33=rt_q33,
        rt_q66=rt_q66,
    )
    loader = DataLoader(test_ds, batch_size=cfg.batch_size,
                        shuffle=False, num_workers=0)
    model.eval()
    na_preds, na_gts, rt_preds, rt_gts = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            result = model.predict(
                batch["activities"].to(device),
                batch["features"].to(device),
                batch["length"].to(device),
                beta=cfg.ensemble_beta,
                faiss_k=cfg.faiss_rt_k,
            )
            na_preds.extend(result["na_pred"].tolist())
            na_gts.extend(batch["next_activity"].numpy().tolist())
            rt_preds.extend(result["rt_final"].tolist())
            rt_gts.extend(batch["remaining_time"].numpy().tolist())

    na_p = np.array(na_preds)
    na_g = np.array(na_gts)
    rt_p = np.array(rt_preds)
    rt_g = np.array(rt_gts)

    rng    = np.random.default_rng(seed)
    acc    = float((na_p == na_g).mean())
    mae    = float(np.abs(rt_p - rt_g).mean())
    f1     = float(sk_f1(na_g, na_p, average="macro", zero_division=0))
    acc_ci = bootstrap_ci(na_g, na_p, "accuracy", rng=rng)
    mae_ci = bootstrap_ci(rt_g, rt_p, "mae",      rng=rng)
    f1_ci  = bootstrap_ci(na_g, na_p, "f1",       rng=rng)

    return {
        "acc": acc, "acc_ci": acc_ci,
        "f1":  f1,  "f1_ci":  f1_ci,
        "mae": mae, "mae_ci": mae_ci,
        "train_time_s": round(train_time, 2),
    }


# ---------------------------------------------------------------------------
# Per-dataset runner
# ---------------------------------------------------------------------------

def run_dataset(
    dataset:    str,
    fractions:  List[float],
    output_dir: Path,
    seed:       int,
) -> List[Dict[str, Any]]:

    cfg              = TAIADATLConfig()
    cfg.dataset_name = dataset
    cfg.seed         = seed
    device           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    pipeline = DTAIAPipeline(cfg)
    full_train_df, val_df, test_df = pipeline.load_prepared_data()
    if val_df is None:
        print(f"[SKIP] {dataset}: no validation split found.")
        return []

    dataset_display   = DATASET_DISPLAY.get(dataset, dataset.upper())
    total_train_cases = full_train_df["case_id"].nunique()
    print(f"\n{'='*64}")
    print(f"Dataset: {dataset_display}  |  train_cases={total_train_cases}  "
          f"test={len(test_df)}")
    print(f"{'='*64}")

    rows = []
    for frac in fractions:
        frac_train_df = subsample_train(full_train_df, frac, seed)
        n_cases       = frac_train_df["case_id"].nunique()
        pct_label     = training_pct_label(frac)

        print(f"\n  {pct_label} of total data  (cases={n_cases})")

        try:
            metrics = run_dtaia(
                frac_train_df, full_train_df, val_df, test_df, cfg, device, seed
            )
        except Exception as exc:
            print(f"    [ERROR] {exc}")
            continue

        row = {
            "Dataset":              dataset_display,
            "Model":                "DATL-TAIA",
            "Training_Percentage":  pct_label,
            "NA_Acc":               round(metrics["acc"] * 100, 1),
            "NA_CI":                round(metrics["acc_ci"] * 100, 1),
            "NA_F1":                round(metrics["f1"], 3),
            "F1_CI":                round(metrics["f1_ci"], 3),
            "RT_MAE":               round(metrics["mae"], 2),
            "MAE_CI":               round(metrics["mae_ci"], 2),
        }
        rows.append(row)
        print(f"    NA_Acc={row['NA_Acc']}%  F1={row['NA_F1']}  "
              f"MAE={row['RT_MAE']}  t={metrics['train_time_s']:.0f}s")

    out_path = output_dir / f"training_sensitivity_{dataset}.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"  Saved → {out_path}")
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Training-Data Sensitivity Experiment — DATL-TAIA"
    )
    p.add_argument("--datasets",   nargs="+", default=ALL_DATASETS)
    p.add_argument("--fractions",  nargs="+", type=float,
                   default=TRAINING_FRACTIONS)
    p.add_argument("--output-dir", default="results/experiments")
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


def main():
    args       = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    for ds in args.datasets:
        rows = run_dataset(ds, args.fractions, output_dir, args.seed)
        all_rows.extend(rows)

    if all_rows:
        df_summary = pd.DataFrame(all_rows, columns=CSV_COLUMNS)
        csv_path   = output_dir / "training_sensitivity_summary.csv"
        df_summary.to_csv(csv_path, index=False)
        print(f"\nSummary saved → {csv_path}")
        print("\n" + df_summary.to_string(index=False))


if __name__ == "__main__":
    main()