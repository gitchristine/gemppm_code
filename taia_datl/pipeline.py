"""
D-TAIA Main Pipeline
====================

Usage:
    # Full pipeline (raw event log)
    python -m taia_datl.pipeline --filepath data/bpi2012.xes --dataset bpi2012

    # Skip data prep (clean CSVs already exist)
    python -m taia_datl.pipeline --dataset bpi2012 --skip-data-prep

    # Ablation: replace NAHead+RTHead with legacy DATL encoder only
    python -m taia_datl.pipeline --dataset bpi2012 --skip-data-prep --no-dtaia-heads
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from taia_datl.config import TAIADATLConfig

# Data
from taia_datl.data_functions.data_prep import (
    EventLogPreprocessor,
    FEATURE_COLUMNS,
)

# Backbone (legacy DATL — kept for ablations)
from taia_datl.backbone.tinyllm import (
    load_tinyllm,
    apply_lora,
    drop_ffn_deltas,
)

# D-TAIA model
from taia_datl.model import DTAIAModel
from taia_datl.components.domain_triplet_loss import (
    DomainAwareTripletLoss,
    compute_rt_bucket_thresholds,
    assign_rt_buckets,
)
from taia_datl.components.faiss_rt_index import FAISSRTIndex

# Legacy components (retained for ablation / domain-prompt path)
from taia_datl.components.domain_prompt import DomainPromptGenerator
from taia_datl.components.few_shot_csv import FewShotCSVLoader
from taia_datl.components.faiss_index import FAISSIndex, RandomFallbackIndex
from taia_datl.components.datl_encoder import DATLEncoder, TripletLoss
from taia_datl.components.triplet_builder import TripletBuilder


# ============================================================================
# Domain label helpers
# ============================================================================

def compute_prefix_entropy(activities: np.ndarray) -> float:
    """Shannon entropy over activity frequencies in a prefix."""
    if len(activities) == 0:
        return 0.0
    counts = np.bincount(activities)
    probs = counts[counts > 0] / len(activities)
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def compute_domain_thresholds(
    train_df: pd.DataFrame,
    max_seq_len: int,
) -> dict:
    """
    Compute entropy and length thresholds from training prefixes (frozen
    after training .
    """
    entropies, lengths = [], []

    for _, cdf in train_df.groupby("case_id"):
        acts = cdf["activity_encoded"].values
        n = len(acts)
        for k in range(2, n):
            entropies.append(compute_prefix_entropy(acts[:k]))
            lengths.append(min(k, max_seq_len))

    entropies = np.array(entropies)
    lengths = np.array(lengths)

    return {
        "entropy_q33": float(np.percentile(entropies, 33)),
        "entropy_q66": float(np.percentile(entropies, 66)),
        "length_q33":  float(np.percentile(lengths, 33)),
        "length_q66":  float(np.percentile(lengths, 66)),
    }


def assign_domain_id(
    entropy: float,
    length: int,
    thresholds: dict,
) -> int:
    """
    Map (entropy_level, length_category) → integer domain id in {0,…,8}.
    """
    eq33, eq66 = thresholds["entropy_q33"], thresholds["entropy_q66"]
    lq33, lq66 = thresholds["length_q33"],  thresholds["length_q66"]

    e_level = 0 if entropy < eq33 else (1 if entropy < eq66 else 2)
    l_cat   = 0 if length  < lq33 else (1 if length  < lq66 else 2)

    return e_level * 3 + l_cat   # 9 possible domains


# ============================================================================
# Dataset with per-prefix domain labels and RT buckets
# ============================================================================

class DTAIADataset(Dataset):

    def __init__(
        self,
        df: pd.DataFrame,
        max_len: int = 20,
        domain_thresholds: Optional[dict] = None,
        rt_q33: Optional[float] = None,
        rt_q66: Optional[float] = None,
    ):
        self.max_len = max_len
        self.domain_thresholds = domain_thresholds
        self.rt_q33 = rt_q33
        self.rt_q66 = rt_q66
        self.samples = self._build_samples(df)

    def _build_samples(self, df: pd.DataFrame) -> list[dict]:
        feat_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
        samples = []

        for case_id, cdf in df.groupby("case_id"):
            if "timestamp" in cdf.columns:
                cdf = cdf.sort_values("timestamp")
            acts = cdf["activity_encoded"].values
            feats = cdf[feat_cols].values.astype(np.float32)

            for k in range(2, len(cdf)):
                prefix_acts = acts[:k]
                prefix_feats = feats[:k]
                next_act = int(acts[k]) if k < len(acts) else int(acts[-1])
                rem_time = (
                    float(cdf["remaining_time"].iloc[k])
                    if "remaining_time" in cdf.columns else 0.0
                )

                # Per-prefix domain ID
                entropy = compute_prefix_entropy(prefix_acts)
                length  = min(len(prefix_acts), self.max_len)
                if self.domain_thresholds is not None:
                    domain_id = assign_domain_id(
                        entropy, length, self.domain_thresholds
                    )
                else:
                    domain_id = int(cdf["domain_id"].iloc[0]) \
                        if "domain_id" in cdf.columns else 0

                # RT bucket — fallback to 0 if thresholds absent
                if self.rt_q33 is not None and self.rt_q66 is not None:
                    if rem_time < self.rt_q33:
                        rt_bucket = 0
                    elif rem_time < self.rt_q66:
                        rt_bucket = 1
                    else:
                        rt_bucket = 2
                else:
                    rt_bucket = 0

                # Pad / truncate prefix to max_len
                seq_len = min(len(prefix_acts), self.max_len)
                pad_acts = np.zeros(self.max_len, dtype=np.int64)
                pad_feats = np.zeros((self.max_len, prefix_feats.shape[1]),
                                     dtype=np.float32)
                pad_acts[:seq_len]  = prefix_acts[:seq_len]
                pad_feats[:seq_len] = prefix_feats[:seq_len]

                samples.append({
                    "activities":    pad_acts,
                    "features":      pad_feats,
                    "length":        seq_len,
                    "next_activity": next_act,
                    "remaining_time": rem_time,
                    "domain_id":     domain_id,
                    "rt_bucket":     rt_bucket,
                    "case_id":       str(case_id),
                })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "activities":    torch.tensor(s["activities"],    dtype=torch.long),
            "features":      torch.tensor(s["features"],      dtype=torch.float32),
            "length":        torch.tensor(s["length"],        dtype=torch.long),
            "next_activity": torch.tensor(s["next_activity"], dtype=torch.long),
            "remaining_time":torch.tensor(s["remaining_time"],dtype=torch.float32),
            "domain_id":     torch.tensor(s["domain_id"],    dtype=torch.long),
            "rt_bucket":     torch.tensor(s["rt_bucket"],    dtype=torch.long),
        }


# ============================================================================
# Pipeline
# ============================================================================

class DTAIAPipeline:

    def __init__(self, cfg: TAIADATLConfig):
        self.cfg = cfg
        cfg.ensure_dirs()
        self.device = torch.device(
            cfg.device if torch.cuda.is_available() else "cpu"
        )
        print(f"[Pipeline] Device: {self.device}")

    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------

    def prepare_data(self, filepath: str):
        prep = EventLogPreprocessor(output_dir=str(self.cfg.clean_data_dir))
        return prep.prepare_data(
            filepath, self.cfg.dataset_name,
            test_size=self.cfg.test_size,
            val_size=self.cfg.val_size,
            min_case_length=self.cfg.min_case_length,
            time_unit=self.cfg.time_unit,
        )

    def load_prepared_data(self):
        d = self.cfg.clean_data_dir
        train_df = pd.read_csv(d / f"{self.cfg.dataset_name}_train.csv")
        test_df  = pd.read_csv(d / f"{self.cfg.dataset_name}_test.csv")

        val_path = d / f"{self.cfg.dataset_name}_val.csv"
        if val_path.exists():
            val_df = pd.read_csv(val_path)
            print(f"[Pipeline] train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
            return train_df, val_df, test_df

        print(f"[Pipeline] train={len(train_df)}  test={len(test_df)}  (no val split found)")
        return train_df, None, test_df

    # ------------------------------------------------------------------
    # 2. Domain prompt (optional, for TAIA backbone path)
    # ------------------------------------------------------------------

    def generate_domain_prompt(self, df, model=None, tokenizer=None) -> dict:
        if self.cfg.no_domain_prompt:
            return {"label": "", "A_dom": [], "t_scale": "", "notes": ""}
        gen = DomainPromptGenerator(
            model=model, tokenizer=tokenizer,
            max_tokens=self.cfg.domain_prompt_max_tokens,
            temperature=self.cfg.domain_prompt_temperature,
        )
        return gen.generate(df, self.cfg.dataset_name)

    # ------------------------------------------------------------------
    # 3. TinyLLM backbone (optional  for TAIA inference)
    # ------------------------------------------------------------------

    def load_backbone(self):
        if self.cfg.backbone_lstm:
            return None, None
        model, tokenizer = load_tinyllm(
            model_name=self.cfg.hf_model_name,
            cache_dir=self.cfg.hf_cache_dir,
            device_map=self.cfg.hf_device_map,
            torch_dtype=self.cfg.hf_torch_dtype,
            load_in_4bit=self.cfg.hf_load_in_4bit,
        )
        if not self.cfg.skip_finetune:
            model = apply_lora(
                model,
                r=self.cfg.lora_r,
                alpha=self.cfg.lora_alpha,
                dropout=self.cfg.lora_dropout,
                target_modules=self.cfg.lora_target_modules,
            )
        return model, tokenizer

    # ------------------------------------------------------------------
    # 4. D-TAIA joint training
    # ------------------------------------------------------------------

    def train_dtaia(
        self,
        train_df: pd.DataFrame,
        num_activities: int,
        domain_thresholds: dict,
        rt_q33: float,
        rt_q66: float,
        val_df: Optional[pd.DataFrame] = None,
    ) -> DTAIAModel:
        """
        Joint training of backbone + NAHead + RTHead with combined loss:
            L_total = L_CE + α·L_MSE + λ·L_D-Triplet

        Early stopping is performed on the validation accuracy when val_df is
        supplied (preferred), otherwise falls back to training loss.
        """
        model = DTAIAModel(
            num_activities=num_activities,
            feature_dim=self.cfg.feature_dim,
            hidden_dim=self.cfg.datl_encoder_dim,
            embedding_dim=self.cfg.rt_embedding_dim,
            na_num_heads=self.cfg.na_num_heads,
            encoder_heads=self.cfg.datl_encoder_heads,
            encoder_layers=self.cfg.datl_encoder_layers,
            encoder_ff_dim=self.cfg.datl_encoder_ff_dim,
            dropout=self.cfg.datl_dropout,
        ).to(self.device)

        ds = DTAIADataset(
            train_df,
            max_len=self.cfg.max_sequence_length,
            domain_thresholds=domain_thresholds,
            rt_q33=rt_q33,
            rt_q66=rt_q66,
        )
        loader = DataLoader(
            ds, batch_size=self.cfg.dtaia_batch_size,
            shuffle=True, num_workers=0,
        )

        val_loader = None
        if val_df is not None:
            val_ds = DTAIADataset(
                val_df,
                max_len=self.cfg.max_sequence_length,
                domain_thresholds=domain_thresholds,
                rt_q33=rt_q33,
                rt_q66=rt_q66,
            )
            val_loader = DataLoader(
                val_ds, batch_size=self.cfg.dtaia_batch_size,
                shuffle=False, num_workers=0,
            )

        triplet_loss_fn = DomainAwareTripletLoss(
            margin=self.cfg.dtaia_triplet_margin
        )
        optimizer = optim.Adam(model.parameters(), lr=self.cfg.dtaia_lr)

        # early stopping tracked on val accuracy (higher = better) or train loss
        best_metric = -float("inf") if val_loader is not None else float("inf")
        patience_counter = 0
        save_path = self.cfg.model_dir / f"{self.cfg.dataset_name}_dtaia_model.pt"

        print(f"[D-TAIA] Joint training for up to {self.cfg.dtaia_epochs} epochs "
              f"(early stopping on {'val accuracy' if val_loader else 'train loss'})…")

        for epoch in range(self.cfg.dtaia_epochs):
            model.train()
            epoch_metrics = {"ce": 0.0, "mse": 0.0, "triplet": 0.0, "total": 0.0}
            n_batches = 0

            for batch in loader:
                acts  = batch["activities"].to(self.device)
                feats = batch["features"].to(self.device)
                lens  = batch["length"].to(self.device)
                na_labels  = batch["next_activity"].to(self.device)
                rt_labels  = batch["remaining_time"].to(self.device)
                rt_buckets = batch["rt_bucket"].to(self.device)
                domain_ids = batch["domain_id"].to(self.device)

                out = model(acts, feats, lens)

                total_loss, metrics = model.compute_loss(
                    na_logits=out["na_logits"],
                    rt_direct=out["rt_direct"],
                    rt_embeddings=out["rt_embedding"],
                    na_labels=na_labels,
                    rt_labels=rt_labels,
                    rt_buckets=rt_buckets,
                    domain_ids=domain_ids,
                    triplet_loss_fn=triplet_loss_fn,
                    alpha=self.cfg.loss_alpha,
                    lmbda=self.cfg.loss_lambda,
                )

                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(), self.cfg.dtaia_grad_clip
                )
                optimizer.step()

                for k, v in metrics.items():
                    epoch_metrics[k] += v
                n_batches += 1

            avg = {k: v / max(n_batches, 1) for k, v in epoch_metrics.items()}

            # ---- Validation pass (for early stopping and logging) ----
            val_acc = None
            if val_loader is not None:
                model.eval()
                correct = total = 0
                with torch.no_grad():
                    for vbatch in val_loader:
                        vacts  = vbatch["activities"].to(self.device)
                        vfeats = vbatch["features"].to(self.device)
                        vlens  = vbatch["length"].to(self.device)
                        result = model.predict(
                            vacts, vfeats, vlens,
                            beta=self.cfg.ensemble_beta,
                            faiss_k=self.cfg.faiss_rt_k,
                        )
                        preds  = result["na_pred"].tolist()
                        labels = vbatch["next_activity"].numpy().tolist()
                        correct += sum(p == l for p, l in zip(preds, labels))
                        total   += len(labels)
                val_acc = correct / total if total > 0 else 0.0
                model.train()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                msg = (f"  Epoch {epoch+1:3d}/{self.cfg.dtaia_epochs}  "
                       f"total={avg['total']:.4f}  ce={avg['ce']:.4f}  "
                       f"mse={avg['mse']:.4f}  triplet={avg['triplet']:.4f}")
                if val_acc is not None:
                    msg += f"  val_acc={val_acc:.4f}"
                print(msg)

            # ---- Early stopping ----
            if val_loader is not None:
                monitor = val_acc          # maximise accuracy
                improved = monitor > best_metric
            else:
                monitor = avg["total"]     # minimise train loss
                improved = monitor < best_metric

            if improved:
                best_metric = monitor
                patience_counter = 0
                torch.save(model.state_dict(), save_path)
            else:
                patience_counter += 1
                if patience_counter >= self.cfg.dtaia_early_stopping_patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        if save_path.exists():
            model.load_state_dict(torch.load(save_path, map_location=self.device))
            print(f"[D-TAIA] Loaded best model ← {save_path}")

        return model, loader

    # ------------------------------------------------------------------
    # 5. Build FAISS RT index (offline, post-training)
    # ------------------------------------------------------------------

    def build_rt_index(
        self,
        model: DTAIAModel,
        train_loader: DataLoader,
    ) -> FAISSRTIndex:
        """Build FAISS RT index from training-set embeddings (Section 4.5.1)."""
        print("[D-TAIA] Building FAISS RT index from training embeddings…")
        index = model.build_faiss_index(
            train_loader,
            device=self.device,
            embedding_dim=self.cfg.rt_embedding_dim,
        )

        # Persist to disk
        index.save(self.cfg.faiss_dir / f"{self.cfg.dataset_name}_rt_index")
        return index

    # ------------------------------------------------------------------
    # 6. Evaluate
    # ------------------------------------------------------------------

    def evaluate(
        self,
        model: DTAIAModel,
        test_df: pd.DataFrame,
        domain_thresholds: dict,
        rt_q33: float,
        rt_q66: float,
    ) -> dict:

        ds = DTAIADataset(
            test_df,
            max_len=self.cfg.max_sequence_length,
            domain_thresholds=domain_thresholds,
            rt_q33=rt_q33,
            rt_q66=rt_q66,
        )
        loader = DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=False,
                            num_workers=0)

        model.eval()

        all_na_preds:   list[int]   = []
        all_na_labels:  list[int]   = []
        all_rt_finals:  list[float] = []
        all_rt_labels:  list[float] = []

        with torch.no_grad():
            for batch in loader:
                acts  = batch["activities"].to(self.device)
                feats = batch["features"].to(self.device)
                lens  = batch["length"].to(self.device)

                result = model.predict(
                    acts, feats, lens,
                    beta=self.cfg.ensemble_beta,
                    faiss_k=self.cfg.faiss_rt_k,
                )

                all_na_preds.extend(result["na_pred"].tolist())
                all_na_labels.extend(
                    batch["next_activity"].numpy().tolist()
                )
                all_rt_finals.extend(result["rt_final"].tolist())
                all_rt_labels.extend(
                    batch["remaining_time"].numpy().tolist()
                )

        correct = sum(p == l for p, l in zip(all_na_preds, all_na_labels))
        total   = len(all_na_labels)
        accuracy = correct / total if total > 0 else 0.0

        mae = float(np.mean(np.abs(
            np.array(all_rt_finals) - np.array(all_rt_labels)
        )))

        results = {
            "accuracy":     accuracy,
            "mae":          mae,
            "total_samples": total,
            "correct":      correct,
        }

        print(
            f"\n[Evaluate] Accuracy={accuracy:.4f}  MAE={mae:.4f}  "
            f"({correct}/{total} correct)"
        )
        return results

    # ------------------------------------------------------------------
    # Full run
    # ------------------------------------------------------------------

    def run(
        self,
        filepath: Optional[str] = None,
        skip_data_prep: bool = False,
    ) -> dict:

        print("\n" + "=" * 72)
        print("  D-TAIA PIPELINE")
        print("=" * 72)
        start = time.time()

        # 1. Data
        if skip_data_prep:
            train_df, val_df, test_df = self.load_prepared_data()
        else:
            assert filepath, "--filepath required when not using --skip-data-prep"
            result = self.prepare_data(filepath)
            if len(result) == 3:
                train_df, val_df, test_df = result
            else:
                train_df, test_df = result
                val_df = None

        num_activities = int(train_df["activity_encoded"].max()) + 1
        print(f"[Pipeline] {num_activities} activity classes")

        # 2. Domain thresholds (frozen from training set)
        print("[Pipeline] Computing domain thresholds from training set…")
        domain_thresholds = compute_domain_thresholds(
            train_df, self.cfg.max_sequence_length
        )
        print(f"  entropy thresholds: {domain_thresholds['entropy_q33']:.3f} / "
              f"{domain_thresholds['entropy_q66']:.3f}")
        print(f"  length thresholds:  {domain_thresholds['length_q33']:.0f} / "
              f"{domain_thresholds['length_q66']:.0f}")

        # 3. RT bucket thresholds (frozen from training set)
        rt_values = train_df["remaining_time"].values \
            if "remaining_time" in train_df.columns else np.zeros(len(train_df))
        rt_q33, rt_q66 = compute_rt_bucket_thresholds(rt_values)
        print(f"[Pipeline] RT bucket thresholds: {rt_q33:.2f} / {rt_q66:.2f}")

        # 4. D-TAIA joint training (early stopping on val_df when available)
        model, train_loader = self.train_dtaia(
            train_df, num_activities, domain_thresholds, rt_q33, rt_q66,
            val_df=val_df,
        )

        # 5. Build FAISS RT index
        rt_index = self.build_rt_index(model, train_loader)
        model.faiss_rt_index = rt_index

        # 6. Evaluate
        results = self.evaluate(
            model, test_df, domain_thresholds, rt_q33, rt_q66
        )

        elapsed = time.time() - start
        results["elapsed_seconds"] = elapsed
        results["domain_thresholds"] = domain_thresholds
        results["rt_q33"] = rt_q33
        results["rt_q66"] = rt_q66
        results["config"] = {
            "loss_alpha":  self.cfg.loss_alpha,
            "loss_lambda": self.cfg.loss_lambda,
            "ensemble_beta": self.cfg.ensemble_beta,
            "faiss_rt_k":  self.cfg.faiss_rt_k,
            "dtaia_triplet_margin": self.cfg.dtaia_triplet_margin,
        }

        res_path = (
            self.cfg.results_dir / f"{self.cfg.dataset_name}_dtaia_results.json"
        )
        with open(res_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[Pipeline] Results saved → {res_path}")
        print(f"[Pipeline] Total time: {elapsed:.1f}s")

        return results


# Keep the legacy TAIADATLPipeline alias so existing ablation scripts import fine
TAIADATLPipeline = DTAIAPipeline


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="D-TAIA Pipeline")
    p.add_argument("--filepath",        type=str, default=None)
    p.add_argument("--dataset",         type=str, default="bpi2012")
    p.add_argument("--skip-data-prep",  action="store_true")

    # Model
    p.add_argument("--hf-model",        type=str, default=None)
    p.add_argument("--load-in-4bit",    action="store_true")
    p.add_argument("--skip-finetune",   action="store_true")

    # D-TAIA hyperparameters
    p.add_argument("--loss-alpha",  type=float, default=None,
                   help="Weight for L_MSE (default from config)")
    p.add_argument("--loss-lambda", type=float, default=None,
                   help="Weight for L_D-Triplet (default from config)")
    p.add_argument("--ensemble-beta", type=float, default=None,
                   help="FAISS ensemble β (default from config)")
    p.add_argument("--rt-embedding-dim", type=int, default=None,
                   help="RT embedding dimension E (default from config)")

    # Ablation flags
    p.add_argument("--no-taia",         action="store_true")
    p.add_argument("--no-datl",         action="store_true")
    p.add_argument("--no-faiss",        action="store_true")
    p.add_argument("--no-domain-prompt",action="store_true")
    p.add_argument("--no-few-shot",     action="store_true")
    p.add_argument("--backbone-lstm",   action="store_true")

    p.add_argument("--few-shot-csv",    type=str, default=None)
    p.add_argument("--seed",            type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    cfg = TAIADATLConfig()
    cfg.dataset_name    = args.dataset
    cfg.seed            = args.seed
    cfg.no_taia         = args.no_taia
    cfg.no_datl         = args.no_datl
    cfg.no_faiss        = args.no_faiss
    cfg.no_domain_prompt = args.no_domain_prompt
    cfg.no_few_shot     = args.no_few_shot
    cfg.backbone_lstm   = args.backbone_lstm
    cfg.skip_finetune   = args.skip_finetune
    cfg.hf_load_in_4bit = args.load_in_4bit

    if args.hf_model:
        cfg.hf_model_name = args.hf_model
    if args.few_shot_csv:
        cfg.few_shot_csv = args.few_shot_csv
    if args.loss_alpha is not None:
        cfg.loss_alpha = args.loss_alpha
    if args.loss_lambda is not None:
        cfg.loss_lambda = args.loss_lambda
    if args.ensemble_beta is not None:
        cfg.ensemble_beta = args.ensemble_beta
    if args.rt_embedding_dim is not None:
        cfg.rt_embedding_dim = args.rt_embedding_dim

    pipeline = DTAIAPipeline(cfg)
    pipeline.run(filepath=args.filepath, skip_data_prep=args.skip_data_prep)


if __name__ == "__main__":
    main()