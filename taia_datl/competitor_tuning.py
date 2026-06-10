"""
Competitor Hyperparameter Tuning
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from taia_datl.config import TAIADATLConfig
from taia_datl.pipeline import DTAIAPipeline, DTAIADataset, compute_domain_thresholds
from taia_datl.components.domain_triplet_loss import compute_rt_bucket_thresholds
from taia_datl.competitors.lstm_model    import train_eval_lstm
from taia_datl.competitors.mt_rnn        import train_eval_mt_rnn
from taia_datl.competitors.xgboost_model import (
    extract_prefix_features, xgboost_grid_search,
)
from taia_datl.competitors.ftllm         import train_eval_ftllm

COMPETITORS = ("lstm", "mt_rnn", "xgboost", "ftllm")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _grid_combinations(param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys   = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _make_sequence_datasets(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    cfg:      TAIADATLConfig,
    domain_thresholds: dict,
    rt_q33: float,
    rt_q66: float,
):
    """Build DTAIADatasets (shared by LSTM, MT-RNN, ftLLM)."""
    train_ds = DTAIADataset(
        train_df,
        max_len=cfg.max_sequence_length,
        domain_thresholds=domain_thresholds,
        rt_q33=rt_q33, rt_q66=rt_q66,
    )
    val_ds = DTAIADataset(
        val_df,
        max_len=cfg.max_sequence_length,
        domain_thresholds=domain_thresholds,
        rt_q33=rt_q33, rt_q66=rt_q66,
    )
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Per-competitor grid search
# ---------------------------------------------------------------------------

def _run_lstm(
    param_grid: dict,
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    cfg: TAIADATLConfig,
    device: torch.device,
) -> list:
    num_activities    = int(train_df["activity_encoded"].max()) + 1
    feature_dim       = cfg.feature_dim
    domain_thresholds = compute_domain_thresholds(train_df, cfg.max_sequence_length)
    rt_vals           = train_df["remaining_time"].values \
        if "remaining_time" in train_df.columns else np.zeros(len(train_df))
    rt_q33, rt_q66   = compute_rt_bucket_thresholds(rt_vals)
    train_ds, val_ds  = _make_sequence_datasets(
        train_df, val_df, cfg, domain_thresholds, rt_q33, rt_q66
    )

    results = []
    combos  = _grid_combinations(param_grid)
    print(f"\n[LSTM] {len(combos)} combinations on {cfg.dataset_name}")

    for i, params in enumerate(combos, 1):
        full_cfg = {**params, "num_epochs": cfg.dtaia_epochs,
                    "patience": cfg.dtaia_early_stopping_patience}
        t0 = time.time()
        val_acc, val_mae = train_eval_lstm(
            full_cfg, train_ds, val_ds, num_activities, feature_dim, device
        )
        elapsed = time.time() - t0
        print(f"  [{i:3d}/{len(combos)}] {params} → "
              f"acc={val_acc:.4f}  mae={val_mae:.4f}  ({elapsed:.0f}s)")
        results.append({"params": params, "val_accuracy": val_acc,
                        "val_mae": val_mae, "elapsed_s": round(elapsed, 1)})
    return results


def _run_mt_rnn(
    param_grid: dict,
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    cfg: TAIADATLConfig,
    device: torch.device,
) -> list:
    num_activities    = int(train_df["activity_encoded"].max()) + 1
    feature_dim       = cfg.feature_dim
    domain_thresholds = compute_domain_thresholds(train_df, cfg.max_sequence_length)
    rt_vals           = train_df["remaining_time"].values \
        if "remaining_time" in train_df.columns else np.zeros(len(train_df))
    rt_q33, rt_q66   = compute_rt_bucket_thresholds(rt_vals)
    train_ds, val_ds  = _make_sequence_datasets(
        train_df, val_df, cfg, domain_thresholds, rt_q33, rt_q66
    )

    results = []
    combos  = _grid_combinations(param_grid)
    print(f"\n[MT-RNN] {len(combos)} combinations on {cfg.dataset_name}")

    for i, params in enumerate(combos, 1):
        full_cfg = {**params, "num_epochs": cfg.dtaia_epochs,
                    "patience": cfg.dtaia_early_stopping_patience}
        t0 = time.time()
        val_acc, val_mae = train_eval_mt_rnn(
            full_cfg, train_ds, val_ds, num_activities, feature_dim, device
        )
        elapsed = time.time() - t0
        print(f"  [{i:3d}/{len(combos)}] {params} → "
              f"acc={val_acc:.4f}  mae={val_mae:.4f}  ({elapsed:.0f}s)")
        results.append({"params": params, "val_accuracy": val_acc,
                        "val_mae": val_mae, "elapsed_s": round(elapsed, 1)})
    return results


def _run_xgboost(
    param_grid: dict,
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    cfg: TAIADATLConfig,
) -> list:
    """
    XGBoost uses sklearn's GridSearchCV natively (3-fold CV on train).
    One call covers the entire grid — returns a single-entry results list
    with the best params chosen by CV, validated on val_df.
    """
    print(f"\n[XGBoost] sklearn GridSearchCV on {cfg.dataset_name}")
    X_train, y_na_train, y_rt_train = extract_prefix_features(
        train_df, max_seq_len=cfg.max_sequence_length
    )
    X_val, y_na_val, y_rt_val = extract_prefix_features(
        val_df, max_seq_len=cfg.max_sequence_length
    )

    t0 = time.time()
    best_params, val_acc, val_mae = xgboost_grid_search(
        param_grid,
        X_train, y_na_train, y_rt_train,
        X_val,   y_na_val,   y_rt_val,
        random_state=cfg.seed,
    )
    elapsed = time.time() - t0
    print(f"  Best params: {best_params}")
    print(f"  val_acc={val_acc:.4f}  val_mae={val_mae:.4f}  ({elapsed:.0f}s)")
    return [{"params": best_params, "val_accuracy": val_acc,
             "val_mae": val_mae, "elapsed_s": round(elapsed, 1)}]


def _run_ftllm(
    param_grid: dict,
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    cfg: TAIADATLConfig,
    device: torch.device,
) -> list:
    num_activities = int(train_df["activity_encoded"].max()) + 1
    results = []
    combos  = _grid_combinations(param_grid)
    print(f"\n[ftLLM] {len(combos)} combinations on {cfg.dataset_name}")

    for i, params in enumerate(combos, 1):
        full_cfg = {
            **params,
            "num_activities":  num_activities,
            "finetune_epochs": params.get("finetune_epochs", 3),
            "patience":        3,
            "hf_model_name":   cfg.hf_model_name,
        }
        t0 = time.time()
        val_acc, val_mae = train_eval_ftllm(
            full_cfg, train_df, val_df, num_activities, device
        )
        elapsed = time.time() - t0
        print(f"  [{i:3d}/{len(combos)}] {params} → "
              f"acc={val_acc:.4f}  mae={val_mae:.4f}  ({elapsed:.0f}s)")
        results.append({"params": params, "val_accuracy": val_acc,
                        "val_mae": val_mae, "elapsed_s": round(elapsed, 1)})
    return results


# ---------------------------------------------------------------------------
# Unified runner
# ---------------------------------------------------------------------------

def run_competitor_tuning(
    dataset:     str,
    competitor:  str,
    output_dir:  Path,
    seed:        int = 42,
) -> dict:
    """Run HP tuning for one (dataset, competitor) combination."""

    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg              = TAIADATLConfig()
    cfg.dataset_name = dataset
    cfg.seed         = seed
    device           = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    pipeline = DTAIAPipeline(cfg)
    train_df, val_df, _ = pipeline.load_prepared_data()
    if val_df is None:
        raise RuntimeError(
            f"No val CSV found for '{dataset}'. "
            "Run data prep with val_size=0.15 first."
        )

    # Load dataset + competitor specific grid
    try:
        from scripts.hp_grids import get_competitor_param_grid
        param_grid = get_competitor_param_grid(dataset, competitor)
    except (ImportError, KeyError):
        param_grid = _fallback_grid(competitor)

    # Dispatch
    if competitor == "lstm":
        cv_results = _run_lstm(param_grid, train_df, val_df, cfg, device)
    elif competitor == "mt_rnn":
        cv_results = _run_mt_rnn(param_grid, train_df, val_df, cfg, device)
    elif competitor == "xgboost":
        cv_results = _run_xgboost(param_grid, train_df, val_df, cfg)
    elif competitor == "ftllm":
        cv_results = _run_ftllm(param_grid, train_df, val_df, cfg, device)
    else:
        raise ValueError(f"Unknown competitor '{competitor}'. Choose from {COMPETITORS}")

    # Find best by val_accuracy
    best = max(cv_results, key=lambda r: r["val_accuracy"])

    payload = {
        "dataset":            dataset,
        "competitor":         competitor,
        "best_params":        best["params"],
        "best_val_accuracy":  best["val_accuracy"],
        "best_val_mae":       best["val_mae"],
        "cv_results":         cv_results,
    }

    out_path = output_dir / f"{dataset}_{competitor}_hp_results.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[CompetitorTuning] Results saved → {out_path}")

    return payload


def _fallback_grid(competitor: str) -> dict:
    fallbacks = {
        "lstm":    {"hidden_dim": [64, 128, 256], "num_layers": [1, 2],
                    "dropout": [0.1, 0.3], "learning_rate": [1e-4, 1e-3],
                    "batch_size": [32, 64]},
        "mt_rnn":  {"hidden_dim": [64, 128, 256], "num_layers": [1, 2],
                    "dropout": [0.1, 0.3], "learning_rate": [1e-4, 1e-3],
                    "batch_size": [32, 64], "mt_weight": [0.5, 1.0, 2.0]},
        "xgboost": {"n_estimators": [100, 300], "max_depth": [3, 5],
                    "learning_rate": [0.01, 0.1], "subsample": [0.8, 1.0],
                    "min_child_weight": [1, 3]},
        "ftllm":   {"lora_r": [4, 8, 16], "lora_alpha": [8, 16],
                    "lora_dropout": [0.05, 0.1],
                    "finetune_lr": [1e-5, 5e-5, 1e-4],
                    "finetune_batch_size": [2, 4]},
    }
    return fallbacks.get(competitor, {})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="D-TAIA Competitor Hyperparameter Tuning")
    p.add_argument("--dataset",      type=str, required=True)
    p.add_argument("--competitor",   type=str, required=True, choices=COMPETITORS)
    p.add_argument("--output-dir",   type=str, default="results/hp_tuning")
    p.add_argument("--seed",         type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    run_competitor_tuning(
        dataset=args.dataset,
        competitor=args.competitor,
        output_dir=Path(args.output_dir),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
