"""
D-TAIA Hyperparameter Tuning

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
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from taia_datl.config import TAIADATLConfig
from taia_datl.pipeline import (
    DTAIAPipeline,
    DTAIADataset,
    compute_domain_thresholds,
)
from taia_datl.components.domain_triplet_loss import (
    DomainAwareTripletLoss,
    compute_rt_bucket_thresholds,
)
from taia_datl.model import DTAIAModel


# ---------------------------------------------------------------------------
# Model variant flags  (mirrors ABLATIONS dict in run_ablations.py)
# ---------------------------------------------------------------------------

MODEL_VARIANTS: Dict[str, Dict[str, Any]] = {
    "dtaia":            {},                           # full model
    "no_taia":          {"no_taia": True},
    "no_datl":          {"no_datl": True},
    "no_faiss":         {"no_faiss": True},
    "no_domain_prompt": {"no_domain_prompt": True},
    "no_few_shot":      {"no_few_shot": True},
    "lstm_backbone":    {"backbone_lstm": True},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_params(cfg: TAIADATLConfig, params: Dict[str, Any]) -> TAIADATLConfig:
    """Return a shallow copy of cfg with params applied."""
    cfg2 = copy.copy(cfg)
    for k, v in params.items():
        setattr(cfg2, k, v)
    return cfg2


def _grid_combinations(param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Expand a {name: [v1, v2, …]} grid into a list of flat dicts."""
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


# ---------------------------------------------------------------------------
# Single-run: train + evaluate on val
# ---------------------------------------------------------------------------

def _train_and_eval_val(
    cfg: TAIADATLConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    num_activities: int,
    domain_thresholds: dict,
    rt_q33: float,
    rt_q66: float,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Train the D-TAIA model with `cfg` on train_df and evaluate on val_df.

    Returns:
        (val_accuracy, val_mae)
    """
    model = DTAIAModel(
        num_activities=num_activities,
        feature_dim=cfg.feature_dim,
        hidden_dim=cfg.datl_encoder_dim,
        embedding_dim=cfg.rt_embedding_dim,
        na_num_heads=cfg.na_num_heads,
        encoder_heads=cfg.datl_encoder_heads,
        encoder_layers=cfg.datl_encoder_layers,
        encoder_ff_dim=cfg.datl_encoder_ff_dim,
        dropout=cfg.datl_dropout,
    ).to(device)

    train_ds = DTAIADataset(
        train_df,
        max_len=cfg.max_sequence_length,
        domain_thresholds=domain_thresholds,
        rt_q33=rt_q33,
        rt_q66=rt_q66,
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.dtaia_batch_size,
        shuffle=True, num_workers=0,
    )

    val_ds = DTAIADataset(
        val_df,
        max_len=cfg.max_sequence_length,
        domain_thresholds=domain_thresholds,
        rt_q33=rt_q33,
        rt_q66=rt_q66,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.dtaia_batch_size,
        shuffle=False, num_workers=0,
    )

    triplet_loss_fn = DomainAwareTripletLoss(margin=cfg.dtaia_triplet_margin)
    optimizer = optim.Adam(model.parameters(), lr=cfg.dtaia_lr,
                           weight_decay=cfg.weight_decay)

    best_val_acc = -1.0
    best_val_mae = float("inf")
    patience_counter = 0
    best_state: Optional[dict] = None

    for epoch in range(cfg.dtaia_epochs):
        model.train()
        for batch in train_loader:
            acts       = batch["activities"].to(device)
            feats      = batch["features"].to(device)
            lens       = batch["length"].to(device)
            na_labels  = batch["next_activity"].to(device)
            rt_labels  = batch["remaining_time"].to(device)
            rt_buckets = batch["rt_bucket"].to(device)
            domain_ids = batch["domain_id"].to(device)

            out = model(acts, feats, lens)
            total_loss, _ = model.compute_loss(
                na_logits=out["na_logits"],
                rt_direct=out["rt_direct"],
                rt_embeddings=out["rt_embedding"],
                na_labels=na_labels,
                rt_labels=rt_labels,
                rt_buckets=rt_buckets,
                domain_ids=domain_ids,
                triplet_loss_fn=triplet_loss_fn,
                alpha=cfg.loss_alpha,
                lmbda=cfg.loss_lambda,
            )
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.dtaia_grad_clip)
            optimizer.step()

        # ---- Validation ----
        model.eval()
        na_preds, na_labels_all, rt_preds, rt_labels_all = [], [], [], []
        with torch.no_grad():
            for vbatch in val_loader:
                vacts  = vbatch["activities"].to(device)
                vfeats = vbatch["features"].to(device)
                vlens  = vbatch["length"].to(device)
                result = model.predict(
                    vacts, vfeats, vlens,
                    beta=cfg.ensemble_beta,
                    faiss_k=cfg.faiss_rt_k,
                )
                na_preds.extend(result["na_pred"].tolist())
                na_labels_all.extend(vbatch["next_activity"].numpy().tolist())
                rt_preds.extend(result["rt_final"].tolist())
                rt_labels_all.extend(vbatch["remaining_time"].numpy().tolist())

        val_acc = sum(p == l for p, l in zip(na_preds, na_labels_all)) / max(len(na_labels_all), 1)
        val_mae = float(np.mean(np.abs(np.array(rt_preds) - np.array(rt_labels_all))))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_mae = val_mae
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= cfg.dtaia_early_stopping_patience:
                break

    return best_val_acc, best_val_mae


# ---------------------------------------------------------------------------
# GridSearchCV-style class
# ---------------------------------------------------------------------------

class DTAIAGridSearch:
    """
    Exhaustive hyperparameter search for D-TAIA and ablation models.

    Attributes:
        best_params_  (dict)  – parameter combination with highest val_accuracy
        best_score_   (float) – val_accuracy of the best run
        best_mae_     (float) – val_MAE of the best run
        cv_results_   (list)  – one entry per grid point with params + metrics
    """

    def __init__(
        self,
        base_cfg: TAIADATLConfig,
        param_grid: Dict[str, List[Any]],
        model_variant: str = "dtaia",
        scoring: str = "accuracy",      # "accuracy" | "mae" | "combined"
        verbose: bool = True,
    ):
        if model_variant not in MODEL_VARIANTS:
            raise ValueError(
                f"Unknown model_variant '{model_variant}'. "
                f"Choose from: {list(MODEL_VARIANTS)}"
            )
        self.base_cfg      = base_cfg
        self.param_grid    = param_grid
        self.model_variant = model_variant
        self.scoring       = scoring
        self.verbose       = verbose

        self.best_params_: Optional[Dict[str, Any]] = None
        self.best_score_: float = -float("inf")
        self.best_mae_:   float = float("inf")
        self.cv_results_: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ) -> "DTAIAGridSearch":
        """
        Run the full grid search.
        """
        device = torch.device(
            self.base_cfg.device if torch.cuda.is_available() else "cpu"
        )

        # Frozen preprocessing statistics from training set
        num_activities = int(train_df["activity_encoded"].max()) + 1
        domain_thresholds = compute_domain_thresholds(
            train_df, self.base_cfg.max_sequence_length
        )
        rt_vals = train_df["remaining_time"].values \
            if "remaining_time" in train_df.columns else np.zeros(len(train_df))
        rt_q33, rt_q66 = compute_rt_bucket_thresholds(rt_vals)

        # Apply model variant flags
        variant_flags = MODEL_VARIANTS[self.model_variant]

        combos = _grid_combinations(self.param_grid)
        n = len(combos)
        print(f"\n[GridSearch] {self.model_variant} on {self.base_cfg.dataset_name} "
              f"— {n} combinations")

        for i, params in enumerate(combos, 1):
            cfg = _apply_params(self.base_cfg, {**variant_flags, **params})

            t0 = time.time()
            val_acc, val_mae = _train_and_eval_val(
                cfg, train_df, val_df,
                num_activities, domain_thresholds, rt_q33, rt_q66, device,
            )
            elapsed = time.time() - t0

            score = self._score(val_acc, val_mae)
            entry = {
                "params":      params,
                "val_accuracy": val_acc,
                "val_mae":      val_mae,
                "score":        score,
                "elapsed_s":    round(elapsed, 1),
            }
            self.cv_results_.append(entry)

            if self.verbose:
                param_str = "  ".join(f"{k}={v}" for k, v in params.items())
                print(f"  [{i:3d}/{n}] {param_str} → acc={val_acc:.4f}  "
                      f"mae={val_mae:.4f}  ({elapsed:.0f}s)")

            if score > self.best_score_:
                self.best_score_  = score
                self.best_params_ = params
                self.best_mae_    = val_mae

        print(f"\n[GridSearch] Best params: {self.best_params_}")
        print(f"[GridSearch] Best val accuracy: {self.best_score_:.4f}  "
              f"val MAE: {self.best_mae_:.4f}")
        return self

    def _score(self, acc: float, mae: float) -> float:
        if self.scoring == "accuracy":
            return acc
        if self.scoring == "mae":
            return -mae
        # combined: accuracy weighted higher, mae normalized
        return acc - 0.1 * mae

    def best_config(self) -> TAIADATLConfig:
        if self.best_params_ is None:
            raise RuntimeError("Call fit() before best_config()")
        variant_flags = MODEL_VARIANTS[self.model_variant]
        return _apply_params(self.base_cfg, {**variant_flags, **self.best_params_})

    def save_results(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset":       self.base_cfg.dataset_name,
            "model_variant": self.model_variant,
            "best_params":   self.best_params_,
            "best_val_accuracy": self.best_score_,
            "best_val_mae":  self.best_mae_,
            "cv_results":    self.cv_results_,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[GridSearch] Results saved → {path}")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="D-TAIA Hyperparameter Tuning")
    p.add_argument("--dataset",    type=str, required=True,
                   help="Dataset name, e.g. bpi2012 or bpi2015_1")
    p.add_argument("--model",      type=str, default="dtaia",
                   choices=list(MODEL_VARIANTS),
                   help="Model variant to tune")
    p.add_argument("--output-dir", type=str, default="results/hp_tuning",
                   help="Directory to write JSON results")
    p.add_argument("--scoring",    type=str, default="accuracy",
                   choices=["accuracy", "mae", "combined"])
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    # Base config
    cfg = TAIADATLConfig()
    cfg.dataset_name = args.dataset
    cfg.seed         = args.seed

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load data (expects _train.csv and _val.csv already generated)
    pipeline = DTAIAPipeline(cfg)
    train_df, val_df, _ = pipeline.load_prepared_data()

    if val_df is None:
        raise RuntimeError(
            f"No val CSV found for '{args.dataset}'. "
            "Run data prep with val_size=0.15 first."
        )

    # Load dataset-specific grid
    try:
        from scripts.hp_grids import get_param_grid
        param_grid = get_param_grid(args.dataset)
    except ImportError:
        # Fallback: a compact universal grid
        param_grid = {
            "dtaia_lr":            [1e-4, 1e-3, 1e-2],
            "dtaia_batch_size":    [32, 64, 128],
            "datl_dropout":        [0.1, 0.3, 0.5],
            "loss_lambda":         [0.01, 0.1, 0.5],
        }

    gs = DTAIAGridSearch(
        base_cfg=cfg,
        param_grid=param_grid,
        model_variant=args.model,
        scoring=args.scoring,
    )
    gs.fit(train_df, val_df)

    out_path = Path(args.output_dir) / f"{args.dataset}_{args.model}_hp_results.json"
    gs.save_results(out_path)


if __name__ == "__main__":
    main()
