"""
Experiment 2 — Prefix-Length Sensitivity
==========================================
Runs DATL-TAIA only. Oyamada / MT-RNN results are added manually afterwards.
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
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from taia_datl.config import TAIADATLConfig
from taia_datl.pipeline import (
    DTAIAPipeline, DTAIADataset,
    compute_domain_thresholds, compute_prefix_entropy, assign_domain_id,
)
from taia_datl.components.domain_triplet_loss import compute_rt_bucket_thresholds
from taia_datl.data_functions.data_prep import FEATURE_COLUMNS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PREFIX_BUCKETS: List[Tuple[float, float, str]] = [
    (0.10, 0.20, "10-20%"),
    (0.20, 0.40, "20-40%"),
    (0.40, 0.60, "40-60%"),
    (0.60, 0.80, "60-80%"),
    (0.80, 1.01, "80-100%"),
]

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
    "Dataset", "Model", "Prefix_Percentage",
    "NA_Acc", "NA_CI", "NA_F1", "F1_CI",
    "RT_MAE", "MAE_CI",
]


# ---------------------------------------------------------------------------
# Completion-aware dataset
# ---------------------------------------------------------------------------

class PrefixCompletionDataset(Dataset):

    def __init__(
        self,
        df: pd.DataFrame,
        max_len: int = 20,
        domain_thresholds: Optional[dict] = None,
        rt_q33: Optional[float] = None,
        rt_q66: Optional[float] = None,
    ):
        self.max_len           = max_len
        self.domain_thresholds = domain_thresholds
        self.rt_q33            = rt_q33
        self.rt_q66            = rt_q66
        self.samples           = self._build(df)

    def _build(self, df: pd.DataFrame) -> list:
        feat_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
        samples   = []

        for case_id, cdf in df.groupby("case_id"):
            if "timestamp" in cdf.columns:
                cdf = cdf.sort_values("timestamp")

            acts  = cdf["activity_encoded"].values
            feats = cdf[feat_cols].values.astype(np.float32) \
                    if feat_cols else np.zeros((len(cdf), 0), dtype=np.float32)
            n     = len(cdf)

            for k in range(2, n):
                prefix_acts  = acts[:k]
                prefix_feats = feats[:k]
                next_act     = int(acts[k]) if k < n else int(acts[-1])
                rem_time     = (float(cdf["remaining_time"].iloc[k])
                                if "remaining_time" in cdf.columns else 0.0)

                entropy  = compute_prefix_entropy(prefix_acts)
                length   = min(len(prefix_acts), self.max_len)
                if self.domain_thresholds is not None:
                    domain_id = assign_domain_id(entropy, length,
                                                  self.domain_thresholds)
                else:
                    domain_id = int(cdf["domain_id"].iloc[0]) \
                        if "domain_id" in cdf.columns else 0

                if self.rt_q33 is not None and self.rt_q66 is not None:
                    rt_bucket = (0 if rem_time < self.rt_q33 else
                                 1 if rem_time < self.rt_q66 else 2)
                else:
                    rt_bucket = 0

                seq_len    = min(k, self.max_len)
                feat_width = prefix_feats.shape[1] if prefix_feats.ndim > 1 else 0
                pad_acts   = np.zeros(self.max_len, dtype=np.int64)
                pad_feats  = np.zeros((self.max_len, max(feat_width, 1)),
                                      dtype=np.float32)
                pad_acts[:seq_len]  = prefix_acts[:seq_len]
                if feat_width > 0:
                    pad_feats[:seq_len] = prefix_feats[:seq_len]

                samples.append({
                    "activities":          pad_acts,
                    "features":            pad_feats,
                    "length":              seq_len,
                    "next_activity":       next_act,
                    "remaining_time":      rem_time,
                    "domain_id":           domain_id,
                    "rt_bucket":           rt_bucket,
                    "case_id":             str(case_id),
                    "prefix_k":            k,
                    "case_total_length":   n,
                    "completion_fraction": k / n,
                })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "activities":          torch.tensor(s["activities"],     dtype=torch.long),
            "features":            torch.tensor(s["features"],       dtype=torch.float32),
            "length":              torch.tensor(s["length"],         dtype=torch.long),
            "next_activity":       torch.tensor(s["next_activity"],  dtype=torch.long),
            "remaining_time":      torch.tensor(s["remaining_time"], dtype=torch.float32),
            "domain_id":           torch.tensor(s["domain_id"],      dtype=torch.long),
            "rt_bucket":           torch.tensor(s["rt_bucket"],      dtype=torch.long),
            "completion_fraction": s["completion_fraction"],
            "prefix_k":            s["prefix_k"],
            "case_total_length":   s["case_total_length"],
            "remaining_time_raw":  s["remaining_time"],
        }


# ---------------------------------------------------------------------------
# Bootstrap CI helper
# ---------------------------------------------------------------------------

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
# DATL-TAIA inference
# ---------------------------------------------------------------------------

def evaluate_dtaia(
    model,
    test_ds: PrefixCompletionDataset,
    cfg:     TAIADATLConfig,
    device:  torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns: completions, na_preds, na_labels, rt_preds, rt_labels."""
    loader = DataLoader(test_ds, batch_size=cfg.batch_size,
                        shuffle=False, num_workers=0)
    model.eval()
    completions, na_preds, na_labels, rt_preds, rt_labels = [], [], [], [], []

    with torch.no_grad():
        for batch in loader:
            acts  = batch["activities"].to(device)
            feats = batch["features"].to(device)
            lens  = batch["length"].to(device)

            result  = model.predict(acts, feats, lens,
                                    beta=cfg.ensemble_beta,
                                    faiss_k=cfg.faiss_rt_k)
            completions.extend(batch["completion_fraction"])
            na_preds.extend(result["na_pred"].tolist())
            na_labels.extend(batch["next_activity"].tolist())
            rt_preds.extend(result["rt_final"].tolist())
            rt_labels.extend(batch["remaining_time"].tolist())

    return (
        np.array(completions, dtype=float),
        np.array(na_preds,    dtype=int),
        np.array(na_labels,   dtype=int),
        np.array(rt_preds,    dtype=float),
        np.array(rt_labels,   dtype=float),
    )


# ---------------------------------------------------------------------------
# Bucket metrics
# ---------------------------------------------------------------------------

def compute_bucket_rows(
    dataset_display: str,
    completions:     np.ndarray,
    na_preds:        np.ndarray,
    na_labels:       np.ndarray,
    rt_preds:        np.ndarray,
    rt_labels:       np.ndarray,
    seed:            int,
) -> List[Dict[str, Any]]:
    from sklearn.metrics import f1_score as sk_f1

    rows = []
    rng  = np.random.default_rng(seed)

    for lo, hi, label in PREFIX_BUCKETS:
        mask = (completions >= lo) & (completions < hi)
        n    = int(mask.sum())
        if n == 0:
            continue

        na_p = na_preds[mask]
        na_g = na_labels[mask]
        rt_p = rt_preds[mask]
        rt_g = rt_labels[mask]

        acc    = float((na_p == na_g).mean())
        mae    = float(np.abs(rt_p - rt_g).mean())
        f1     = float(sk_f1(na_g, na_p, average="macro", zero_division=0))
        acc_ci = bootstrap_ci(na_g, na_p, "accuracy", rng=rng)
        mae_ci = bootstrap_ci(rt_g, rt_p, "mae",      rng=rng)
        f1_ci  = bootstrap_ci(na_g, na_p, "f1",       rng=rng)

        rows.append({
            "Dataset":            dataset_display,
            "Model":              "DATL-TAIA",
            "Prefix_Percentage":  label,
            "NA_Acc":             round(acc * 100, 1),
            "NA_CI":              round(acc_ci * 100, 1),
            "NA_F1":              round(f1, 3),
            "F1_CI":              round(f1_ci, 3),
            "RT_MAE":             round(mae, 2),
            "MAE_CI":             round(mae_ci, 2),
        })

    return rows


# ---------------------------------------------------------------------------
# Per-dataset runner
# ---------------------------------------------------------------------------

def run_dataset(
    dataset:    str,
    output_dir: Path,
    model_dir:  Optional[Path],
    seed:       int,
) -> List[Dict[str, Any]]:

    cfg              = TAIADATLConfig()
    cfg.dataset_name = dataset
    cfg.seed         = seed
    device           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    pipeline = DTAIAPipeline(cfg)
    train_df, val_df, test_df = pipeline.load_prepared_data()
    if val_df is None:
        print(f"[SKIP] {dataset}: no validation split found.")
        return []

    num_activities    = int(train_df["activity_encoded"].max()) + 1
    domain_thresholds = compute_domain_thresholds(train_df, cfg.max_sequence_length)
    rt_vals           = (train_df["remaining_time"].values
                         if "remaining_time" in train_df.columns
                         else np.zeros(len(train_df)))
    rt_q33, rt_q66    = compute_rt_bucket_thresholds(rt_vals)
    dataset_display   = DATASET_DISPLAY.get(dataset, dataset.upper())

    print(f"\n{'='*64}")
    print(f"Dataset: {dataset_display}  |  activities={num_activities}  "
          f"train={len(train_df)}  test={len(test_df)}")
    print(f"{'='*64}")

    test_ds = PrefixCompletionDataset(
        test_df, cfg.max_sequence_length,
        domain_thresholds, rt_q33, rt_q66,
    )
    completions_all = np.array([s["completion_fraction"] for s in test_ds.samples])
    for lo, hi, label in PREFIX_BUCKETS:
        cnt = int(((completions_all >= lo) & (completions_all < hi)).sum())
        print(f"  {label:<8}: {cnt:6d} prefixes")

    # Load or train DATL-TAIA
    from taia_datl.model import DTAIAModel
    ckpt = (model_dir / f"{dataset}_dtaia_model.pt"
            if model_dir else
            Path(cfg.model_dir) / f"{dataset}_dtaia_model.pt")

    if ckpt.exists():
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
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"  Loaded checkpoint: {ckpt}")
    else:
        print(f"  No checkpoint found; training now…")
        t0 = time.time()
        model, _ = pipeline.train_dtaia(
            train_df, num_activities,
            domain_thresholds, rt_q33, rt_q66,
            val_df=val_df,
        )
        print(f"  Training time: {time.time()-t0:.0f}s")

    completions, na_preds, na_labels, rt_preds, rt_labels = \
        evaluate_dtaia(model, test_ds, cfg, device)

    overall_acc = float((na_preds == na_labels).mean())
    overall_mae = float(np.abs(rt_preds - rt_labels).mean())
    print(f"  Overall: acc={overall_acc:.4f}  mae={overall_mae:.4f}  n={len(na_preds)}")

    rows = compute_bucket_rows(
        dataset_display, completions, na_preds, na_labels,
        rt_preds, rt_labels, seed,
    )

    for r in rows:
        print(f"  {r['Prefix_Percentage']:<8}  "
              f"NA_Acc={r['NA_Acc']}%  F1={r['NA_F1']}  MAE={r['RT_MAE']}")

    out_path = output_dir / f"prefix_sensitivity_{dataset}.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"  Saved → {out_path}")

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Prefix-Length Sensitivity Experiment — DATL-TAIA"
    )
    p.add_argument("--datasets",   nargs="+", default=ALL_DATASETS)
    p.add_argument("--output-dir", default="results/experiments")
    p.add_argument("--model-dir",  default=None,
                   help="Directory containing pre-trained .pt checkpoints")
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


def main():
    args       = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir  = Path(args.model_dir) if args.model_dir else None

    all_rows: List[Dict[str, Any]] = []
    for ds in args.datasets:
        rows = run_dataset(ds, output_dir, model_dir, args.seed)
        all_rows.extend(rows)

    if all_rows:
        df_summary = pd.DataFrame(all_rows, columns=CSV_COLUMNS)
        csv_path   = output_dir / "prefix_sensitivity_summary.csv"
        df_summary.to_csv(csv_path, index=False)
        print(f"\nSummary saved → {csv_path}")
        print("\n" + df_summary.to_string(index=False))


if __name__ == "__main__":
    main()