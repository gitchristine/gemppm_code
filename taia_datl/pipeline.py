"""
TAIA-DATL Main Entry Here
==========================================

Usage:
    # Full pipeline
    python -m taia_datl.pipeline --filepath data/bpi2012.xes --dataset bpi2012

    # Skip data prep (already done)
    python -m taia_datl.pipeline --dataset bpi2012 --skip-data-prep

    # Ablation: remove TAIA
    python -m taia_datl.pipeline --dataset bpi2012 --skip-data-prep --no-taia
"""

from __future__ import annotations

import argparse
import json
import time
import joblib
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

# Backbone
from taia_datl.backbone.tinyllm import (
    load_tinyllm,
    apply_lora,
    drop_ffn_deltas,
    TinyLLMEncoder,
)

# Components
from taia_datl.components.domain_prompt import DomainPromptGenerator
from taia_datl.components.faiss_index import FAISSIndex, RandomFallbackIndex
from taia_datl.components.few_shot_csv import FewShotCSVLoader
from taia_datl.components.taia_inference import taia_selective_predict, build_taia_prompt
from taia_datl.components.datl_encoder import DATLEncoder, TripletLoss
from taia_datl.components.triplet_builder import TripletBuilder
from taia_datl.components.fusion import FusionGate

# Heads
from taia_datl.heads.activity_head import ActivityHead
from taia_datl.heads.time_head import TimeHead


# ============================================================================
# Domain / RT-bucket helpers
# ============================================================================

def _prefix_entropy(activities: np.ndarray) -> float:
    """Shannon entropy of activity frequencies within a prefix."""
    if len(activities) <= 1:
        return 0.0
    counts = np.bincount(activities.astype(int))
    probs = counts[counts > 0] / len(activities)
    return float(-np.sum(probs * np.log2(probs)))


def compute_domain_thresholds(train_df: pd.DataFrame, max_len: int = 20) -> dict:
    """Compute entropy and length quantile thresholds from training prefixes only.

    Iterates over every prefix in the training set (k=2..n-1) and collects
    prefix entropy and prefix length values.  The 33rd and 66th percentiles
    are returned as frozen thresholds to be applied unchanged to all splits.
    """
    entropies, lengths = [], []
    for _, cdf in train_df.groupby("case_id"):
        acts = cdf["activity_encoded"].values
        for k in range(2, len(cdf)):
            entropies.append(_prefix_entropy(acts[:k]))
            lengths.append(k)
    entropies = np.array(entropies, dtype=float)
    lengths = np.array(lengths, dtype=float)
    return {
        "ent_33": float(np.quantile(entropies, 0.33)),
        "ent_66": float(np.quantile(entropies, 0.66)),
        "len_33": float(np.quantile(lengths, 0.33)),
        "len_66": float(np.quantile(lengths, 0.66)),
    }


def compute_rt_thresholds(train_df: pd.DataFrame) -> dict:
    """Compute RT quantile bucket boundaries from training remaining_time only."""
    rt = train_df["remaining_time"].dropna().values
    return {
        "rt_33": float(np.quantile(rt, 0.33)),
        "rt_66": float(np.quantile(rt, 0.66)),
    }


def _assign_domain_id(entropy: float, length: int, thresholds: dict) -> int:
    """Map (entropy_bin, length_bin) to a domain integer in 0..8."""
    e = 0 if entropy < thresholds["ent_33"] else (1 if entropy < thresholds["ent_66"] else 2)
    l = 0 if length < thresholds["len_33"] else (1 if length < thresholds["len_66"] else 2)
    return e * 3 + l


def _assign_rt_bucket(rt: float, thresholds: dict) -> int:
    """Bin remaining time into three quantile buckets (0=short,1=medium,2=long)."""
    if rt < thresholds["rt_33"]:
        return 0
    if rt < thresholds["rt_66"]:
        return 1
    return 2


# ============================================================================
# Dataset Loader
# ============================================================================

class TraceDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        max_len: int = 20,
        domain_thresholds: Optional[dict] = None,
        rt_thresholds: Optional[dict] = None,
    ):
        self.max_len = max_len
        self.domain_thresholds = domain_thresholds
        self.rt_thresholds = rt_thresholds
        self.feat_cols: list[str] = []  # set by _build_samples
        self.samples = self._build_samples(df)

    def _build_samples(self, df: pd.DataFrame) -> list[dict]:
        feat_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
        self.feat_cols = feat_cols  # store for _serialise_prefix
        samples = []

        for case_id, cdf in df.groupby("case_id"):
            cdf = cdf.sort_values("timestamp") if "timestamp" in cdf.columns else cdf
            acts = cdf["activity_encoded"].values
            feats = cdf[feat_cols].values.astype(np.float32)

            for k in range(2, len(cdf)):
                prefix_acts = acts[:k]
                prefix_feats = feats[:k]
                next_act = int(acts[k]) if k < len(acts) else int(acts[-1])
                rem_time = float(cdf["remaining_time"].iloc[k]) if "remaining_time" in cdf.columns else 0.0

                # Domain ID: derived from prefix-observable properties only,
                # using thresholds frozen from the training set.
                if self.domain_thresholds is not None:
                    entropy = _prefix_entropy(prefix_acts)
                    domain_id = _assign_domain_id(entropy, k, self.domain_thresholds)
                else:
                    domain_id = int(cdf["domain_id"].iloc[0]) if "domain_id" in cdf.columns else 0

                # RT bucket: quantile bin using training-set thresholds.
                if self.rt_thresholds is not None:
                    rt_bucket = _assign_rt_bucket(rem_time, self.rt_thresholds)
                else:
                    rt_bucket = 0

                seq_len = min(len(prefix_acts), self.max_len)
                pad_acts = np.zeros(self.max_len, dtype=np.int64)
                pad_feats = np.zeros((self.max_len, len(feat_cols)), dtype=np.float32)
                pad_acts[:seq_len] = prefix_acts[:seq_len]
                pad_feats[:seq_len] = prefix_feats[:seq_len]

                samples.append({
                    "activities": pad_acts,
                    "features": pad_feats,
                    "length": seq_len,
                    "next_activity": next_act,
                    "remaining_time": rem_time,
                    "domain_id": domain_id,
                    "rt_bucket": rt_bucket,
                    "case_id": str(case_id),
                })
        return samples

    def _serialise_prefix(
        self,
        activities: np.ndarray,
        features: np.ndarray,
        feat_cols: list[str],
        domain_id: int,
    ) -> str:
        """Convert a prefix to a text string for TinyLLM input.

        Format:
            [DOM_3] step1: activity=approve elapsed=0.50 dow=2 ...
                    step2: activity=review elapsed=1.20 dow=3 ...
        """
        tokens = [f"[DOM_{domain_id}]"]
        for i in range(len(activities)):
            if activities[i] == 0:  # padding — stop at first pad token
                break
            parts = [f"activity={activities[i]}"]
            for j, col in enumerate(feat_cols):
                parts.append(f"{col}={features[i][j]:.3f}")
            tokens.append(f"step{i + 1}: " + " ".join(parts))
        return " ".join(tokens)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "activities": torch.tensor(s["activities"], dtype=torch.long),
            "features": torch.tensor(s["features"], dtype=torch.float32),
            "length": torch.tensor(s["length"], dtype=torch.long),
            "next_activity": torch.tensor(s["next_activity"], dtype=torch.long),
            "remaining_time": torch.tensor(s["remaining_time"], dtype=torch.float32),
            "domain_id": torch.tensor(s["domain_id"], dtype=torch.long),
            "rt_bucket": torch.tensor(s["rt_bucket"], dtype=torch.long),
            # String serialisation for TinyLLM input
            "text": self._serialise_prefix(
                s["activities"], s["features"], self.feat_cols, s["domain_id"]
            ),
        }


def collate_fn(batch: list[dict]) -> dict:
    """Custom collate: stack tensors, keep text strings as a plain list.

    DataLoader's default collate cannot handle mixed tensor/string batches,
    so we do it manually here.
    """
    return {
        "activities":     torch.stack([b["activities"]     for b in batch]),
        "features":       torch.stack([b["features"]       for b in batch]),
        "length":         torch.stack([b["length"]         for b in batch]),
        "next_activity":  torch.stack([b["next_activity"]  for b in batch]),
        "remaining_time": torch.stack([b["remaining_time"] for b in batch]),
        "domain_id":      torch.stack([b["domain_id"]      for b in batch]),
        "rt_bucket":      torch.stack([b["rt_bucket"]      for b in batch]),
        "text":           [b["text"] for b in batch],  # list of strings
    }


# ============================================================================
# Pipeline
# ============================================================================

class TAIADATLPipeline:

    def __init__(self, cfg: TAIADATLConfig):
        self.cfg = cfg
        cfg.ensure_dirs()

        self.device = torch.device(
            cfg.device if torch.cuda.is_available() else "cpu"
        )
        print(f"[Pipeline] Device: {self.device}")

        # Thresholds computed from training data; frozen and reused for all splits.
        self.domain_thresholds: Optional[dict] = None
        self.rt_thresholds: Optional[dict] = None

        # FAISS index rebuilt after Stage 1; used for RT retrieval at inference.
        self.faiss_index = None

    # ------------------------------------------------------------------
    # 1. Data preparation
    # ------------------------------------------------------------------

    def prepare_data(self, filepath: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run the preprocessing pipeline on a raw event log."""
        prep = EventLogPreprocessor(output_dir=str(self.cfg.clean_data_dir))
        return prep.run(
            filepath, self.cfg.dataset_name,
            test_size=self.cfg.test_size,
            min_case_length=self.cfg.min_case_length,
            time_unit=self.cfg.time_unit,
        )

    def load_prepared_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        d = self.cfg.clean_data_dir
        train_df = pd.read_csv(d / f"{self.cfg.dataset_name}_train.csv")
        test_df = pd.read_csv(d / f"{self.cfg.dataset_name}_test.csv")
        print(f"[Pipeline] Loaded train={len(train_df)} test={len(test_df)}")
        return train_df, test_df

    # ------------------------------------------------------------------
    # 2. Domain prompt
    # ------------------------------------------------------------------

    def generate_domain_prompt(
        self, df: pd.DataFrame, model=None, tokenizer=None,
    ) -> dict:
        if self.cfg.no_domain_prompt:
            print("[Pipeline] Domain prompt DISABLED (ablation)")
            return {"label": "", "A_dom": [], "t_scale": "", "notes": ""}

        gen = DomainPromptGenerator(
            model=model, tokenizer=tokenizer,
            max_tokens=self.cfg.domain_prompt_max_tokens,
            temperature=self.cfg.domain_prompt_temperature,
        )
        return gen.generate(df, self.cfg.dataset_name)

    # ------------------------------------------------------------------
    # 3. TinyLLM backbone
    # ------------------------------------------------------------------

    def load_backbone(self):
        if self.cfg.backbone_lstm:
            print("[Pipeline] Backbone = LSTMBackbone (ablation — "
                  "arnir0/Tiny-LLM replaced by bidirectional LSTM)")
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
    # 4. DATL encoder
    # ------------------------------------------------------------------

    def train_datl(
        self,
        train_df: pd.DataFrame,
        num_activities: int,
    ) -> DATLEncoder:

        if self.cfg.backbone_lstm:
            # TODO optional dependency
            from taia_datl.backbone.lstm_backbone import LSTMBackbone
            print("[Pipeline] Backbone = LSTMBackbone "
                  "(ablation: bidirectional LSTM, no NLP pre-training)")
            encoder = LSTMBackbone(
                num_activities=num_activities,
                feature_dim=self.cfg.feature_dim,
                hidden_dim=self.cfg.datl_encoder_dim // 2,  # each direction
                num_layers=self.cfg.num_layers,
                dropout=self.cfg.datl_dropout,
            ).to(self.device)
        else:
            encoder = DATLEncoder(
                num_activities=num_activities,
                feature_dim=self.cfg.feature_dim,
                d_model=self.cfg.datl_encoder_dim,
                nhead=self.cfg.datl_encoder_heads,
                num_layers=self.cfg.datl_encoder_layers,
                dim_ff=self.cfg.datl_encoder_ff_dim,
                dropout=self.cfg.datl_dropout,
            ).to(self.device)

        if self.cfg.no_datl:
            print("[Pipeline] DATL pre-training DISABLED (ablation)")
            return encoder

        # Build dataset & loader
        ds = TraceDataset(
            train_df, self.cfg.max_sequence_length,
            domain_thresholds=self.domain_thresholds,
            rt_thresholds=self.rt_thresholds,
        )
        loader = DataLoader(ds, batch_size=self.cfg.datl_batch_size, shuffle=True,
                            num_workers=0)

        # Build FAISS / random index for triplet sampling
        index = self._build_index()

        # First pass: compute initial embeddings + collect RT metadata for index
        print("[DATL] Computing initial embeddings for FAISS index...")
        encoder.eval()
        all_embs, all_doms, all_rt_buckets, all_rt_values = [], [], [], []
        with torch.no_grad():
            for batch in loader:
                acts = batch["activities"].to(self.device)
                feats = batch["features"].to(self.device)
                lens = batch["length"].to(self.device)
                h = encoder(acts, feats, lens)
                all_embs.append(h.cpu().numpy())
                all_doms.append(batch["domain_id"].numpy())
                all_rt_buckets.append(batch["rt_bucket"].numpy())
                all_rt_values.append(batch["remaining_time"].numpy())

        all_embs = np.concatenate(all_embs, axis=0)
        all_doms = np.concatenate(all_doms, axis=0)
        all_rt_buckets = np.concatenate(all_rt_buckets, axis=0)
        all_rt_values = np.concatenate(all_rt_values, axis=0)
        index.build(
            all_embs,
            [str(i) for i in range(len(all_embs))],
            all_doms.tolist(),
            rt_buckets=all_rt_buckets.tolist(),
            rt_values=all_rt_values.tolist(),
        )

        # Triplet builder
        tb = TripletBuilder(index, top_k=self.cfg.faiss_top_k, seed=self.cfg.seed)
        triplet_loss_fn = TripletLoss(
            margin=self.cfg.triplet_margin,
            distance=self.cfg.triplet_distance,
        )

        # Training loop
        optimizer = optim.Adam(encoder.parameters(), lr=self.cfg.datl_lr)
        encoder.train()

        print(f"[DATL] Pre-training for {self.cfg.datl_epochs} epochs...")
        for epoch in range(self.cfg.datl_epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch in loader:
                acts = batch["activities"].to(self.device)
                feats = batch["features"].to(self.device)
                lens = batch["length"].to(self.device)

                h = encoder(acts, feats, lens)
                h_np = h.detach().cpu().numpy()
                doms = batch["domain_id"].numpy()
                rt_bucks = batch["rt_bucket"].numpy()

                # D-Triplet: positive = same RT bucket + different domain
                a_idx, p_idx, n_idx = tb.build_triplets(h_np, doms, rt_bucks)
                if len(a_idx) == 0:
                    continue

                loss = triplet_loss_fn(h[a_idx], h[p_idx], h[n_idx])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg = epoch_loss / max(n_batches, 1)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{self.cfg.datl_epochs}  loss={avg:.4f}")

        # Save encoder
        save_path = self.cfg.model_dir / f"{self.cfg.dataset_name}_datl_encoder.pt"
        torch.save(encoder.state_dict(), save_path)
        print(f"[DATL] Saved encoder → {save_path}")

        # Rebuild FAISS index from trained encoder embeddings (one-time offline pass)
        print("[DATL] Rebuilding FAISS index from trained encoder embeddings...")
        encoder.eval()
        final_index = self._build_index()
        trained_embs, trained_rt_buckets, trained_rt_values, trained_doms = [], [], [], []
        with torch.no_grad():
            for batch in loader:
                acts = batch["activities"].to(self.device)
                feats = batch["features"].to(self.device)
                lens = batch["length"].to(self.device)
                h = encoder(acts, feats, lens)
                trained_embs.append(h.cpu().numpy())
                trained_doms.append(batch["domain_id"].numpy())
                trained_rt_buckets.append(batch["rt_bucket"].numpy())
                trained_rt_values.append(batch["remaining_time"].numpy())

        trained_embs = np.concatenate(trained_embs, axis=0)
        trained_doms = np.concatenate(trained_doms, axis=0)
        trained_rt_buckets = np.concatenate(trained_rt_buckets, axis=0)
        trained_rt_values = np.concatenate(trained_rt_values, axis=0)
        final_index.build(
            trained_embs,
            [str(i) for i in range(len(trained_embs))],
            trained_doms.tolist(),
            rt_buckets=trained_rt_buckets.tolist(),
            rt_values=trained_rt_values.tolist(),
        )
        print(f"[DATL] Final FAISS index: {len(trained_embs)} training prefix embeddings")

        # Store index on self so evaluate() can access it for RT retrieval
        self.faiss_index = final_index

        return encoder

    def _build_index(self):
        """Build FAISS or random fallback index."""
        if self.cfg.no_faiss:
            print("[Pipeline] FAISS DISABLED — using random fallback (ablation)")
            return RandomFallbackIndex(seed=self.cfg.seed)
        return FAISSIndex(
            dim=self.cfg.faiss_embedding_dim,
            index_type=self.cfg.faiss_index_type,
            nprobe=self.cfg.faiss_nprobe,
            save_dir=str(self.cfg.faiss_dir),
        )

    # ------------------------------------------------------------------
    # 5. Head training (TinyLLM + heads, supervised CE + MSE)
    # ------------------------------------------------------------------

    def train_heads(
        self,
        train_df: pd.DataFrame,
        num_activities: int,
        model,
        tokenizer,
    ) -> tuple:
        """Stage-2 supervised fine-tuning: LoRA-adapted TinyLLM + prediction heads.

        DATLEncoder is NOT used here.  TinyLLM (with LoRA) produces hidden states
        that feed the activity and time heads, trained jointly with CE + MSE loss.
        """
        llm_dim = model.config.hidden_size

        act_head = ActivityHead(
            llm_dim, num_activities,
            hidden_dim=self.cfg.hidden_dim,
            dropout=self.cfg.dropout,
        ).to(self.device)

        time_head = TimeHead(
            llm_dim,
            hidden_dim=self.cfg.hidden_dim,
            dropout=self.cfg.dropout,
        ).to(self.device)

        ds = TraceDataset(
            train_df, self.cfg.max_sequence_length,
            domain_thresholds=self.domain_thresholds,
            rt_thresholds=self.rt_thresholds,
        )
        loader = DataLoader(
            ds, batch_size=self.cfg.batch_size,
            shuffle=True, num_workers=0,
            collate_fn=collate_fn,
        )

        # Only LoRA params (in model) + head params are trainable
        params = (
            list(p for p in model.parameters() if p.requires_grad)
            + list(act_head.parameters())
            + list(time_head.parameters())
        )
        optimizer = optim.Adam(params, lr=self.cfg.learning_rate)
        ce_loss_fn = nn.CrossEntropyLoss()
        mse_loss_fn = nn.MSELoss()

        model.train()
        act_head.train()
        time_head.train()

        print(f"[HeadTrain] Fine-tuning TinyLLM + heads for "
              f"{self.cfg.finetune_epochs} epochs...")

        for epoch in range(self.cfg.finetune_epochs):
            epoch_ce, epoch_mse = 0.0, 0.0

            for batch in loader:
                # Tokenise text strings and forward through TinyLLM with grads
                tokens = tokenizer(
                    batch["text"],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.cfg.hf_max_length,
                )
                device = next(model.parameters()).device
                tokens = {k: v.to(device) for k, v in tokens.items()}

                outputs = model(**tokens, output_hidden_states=True)
                hidden = outputs.hidden_states[-1]  # (batch, seq, dim)
                lengths = tokens["attention_mask"].sum(dim=1) - 1
                idx = lengths.unsqueeze(-1).unsqueeze(-1).expand(
                    -1, 1, hidden.size(-1)
                )
                h = hidden.gather(1, idx).squeeze(1)  # (batch, llm_dim)

                logits = act_head(h)
                t_pred = time_head(h).squeeze(-1)

                na_labels = batch["next_activity"].to(self.device)
                rt_labels = batch["remaining_time"].to(self.device)

                ce = ce_loss_fn(logits, na_labels)
                mse = mse_loss_fn(t_pred, rt_labels)
                loss = ce + self.cfg.alpha_mse * mse

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_ce += ce.item()
                epoch_mse += mse.item()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                n = max(len(loader), 1)
                print(f"  Epoch {epoch + 1}/{self.cfg.finetune_epochs}  "
                      f"CE={epoch_ce / n:.4f}  MSE={epoch_mse / n:.4f}")

        return act_head, time_head

    # ------------------------------------------------------------------
    # 6-8. Evaluate (TinyLLM backbone + optional DATL fallback)
    # ------------------------------------------------------------------

    def evaluate(
        self,
        test_df: pd.DataFrame,
        encoder: DATLEncoder,
        num_activities: int,
        domain_descriptor: dict,
        model=None,
        tokenizer=None,
        act_head=None,
        time_head=None,
        rt_mu: float = 0.0,
        rt_sigma: float = 1.0,
    ) -> dict:
        """Run inference and compute metrics.

        If *model* and pre-trained *act_head*/*time_head* are supplied, TinyLLM
        hidden states are used for prediction (main path).  Otherwise falls back
        to DATLEncoder embeddings (ablation: --backbone-lstm or no model).
        DATLEncoder is always available for FAISS-based RT retrieval regardless.
        """
        encoder.eval()

        # -- TinyLLM encoder wrapper (prediction backbone) ----------------
        tinyllm_enc = None
        if model is not None:
            tinyllm_enc = TinyLLMEncoder(model, tokenizer, pool="last")
            llm_dim = model.config.hidden_size
        else:
            llm_dim = encoder.embedding_dim()

        # -- Prediction heads (use trained ones or build fresh for ablation) --
        if act_head is None:
            act_head = ActivityHead(
                llm_dim, num_activities,
                hidden_dim=self.cfg.hidden_dim, dropout=self.cfg.dropout,
            ).to(self.device)
        if time_head is None:
            time_head = TimeHead(
                llm_dim,
                hidden_dim=self.cfg.hidden_dim, dropout=self.cfg.dropout,
            ).to(self.device)

        act_head.eval()
        time_head.eval()

        fusion = FusionGate(beta=self.cfg.fusion_beta)

        ds = TraceDataset(
            test_df, self.cfg.max_sequence_length,
            domain_thresholds=self.domain_thresholds,
            rt_thresholds=self.rt_thresholds,
        )
        loader = DataLoader(
            ds, batch_size=self.cfg.batch_size,
            shuffle=False, num_workers=0,
            collate_fn=collate_fn,
        )

        all_preds, all_labels = [], []
        all_time_preds, all_time_labels = [], []

        with torch.no_grad():
            for batch in loader:
                acts = batch["activities"].to(self.device)
                feats = batch["features"].to(self.device)
                lens = batch["length"].to(self.device)

                if tinyllm_enc is not None:
                    # Main path: TinyLLM (post-TAIA) produces hidden states
                    h = tinyllm_enc.encode(
                        batch["text"],
                        max_length=self.cfg.hf_max_length,
                    )
                else:
                    # Ablation fallback: DATLEncoder as prediction backbone
                    h = encoder(acts, feats, lens)

                logits = act_head(h)
                rt_direct = time_head(h).squeeze(-1)  # (batch,)

                # RT retrieval: encode prefix with DATLEncoder, query FAISS,
                # retrieve neighbour RT values, blend with TimeHead via FusionGate.
                if self.faiss_index is not None:
                    h_datl = encoder(acts, feats, lens).cpu().numpy()
                    nbr_dists, _, _, _, _, nbr_rt_vals = self.faiss_index.search(
                        h_datl, top_k=self.cfg.faiss_top_k
                    )
                    # RBF kernel regression: w_i = exp(-d_i^2)
                    # ŷ_retrieved = Σ(w_i * RT_i) / Σ(w_i)
                    rbf_retrieved = []
                    for dists, rt_vals in zip(nbr_dists, nbr_rt_vals):
                        w = np.exp(-(np.array(dists) ** 2))
                        w_sum = w.sum()
                        rt_w = (w * np.array(rt_vals)).sum() / w_sum if w_sum > 0 else np.mean(rt_vals)
                        rbf_retrieved.append(rt_w)
                    rt_retrieved = torch.tensor(
                        rbf_retrieved, dtype=torch.float32, device=self.device,
                    )
                    rt_final = fusion.fuse(rt_direct, rt_retrieved)
                else:
                    rt_final = rt_direct

                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_labels.extend(batch["next_activity"].numpy().tolist())
                all_time_preds.extend(rt_final.cpu().numpy().tolist())
                all_time_labels.extend(batch["remaining_time"].numpy().tolist())

        # Inverse-transform RT predictions and labels back to original time scale:
        #   RT_final = sigma_RT * y_hat + mu_RT
        all_time_preds  = [rt_sigma * p + rt_mu for p in all_time_preds]
        all_time_labels = [rt_sigma * l + rt_mu for l in all_time_labels]

        # Metrics
        correct = sum(p == l for p, l in zip(all_preds, all_labels))
        total = len(all_labels)
        accuracy = correct / total if total > 0 else 0.0
        mae = np.mean(np.abs(np.array(all_time_preds) - np.array(all_time_labels)))

        results = {
            "accuracy": accuracy,
            "mae": mae,
            "total_samples": total,
            "correct": correct,
        }
        print(f"\n[Evaluate] Accuracy={accuracy:.4f}  MAE={mae:.4f}  "
              f"({correct}/{total} correct)")
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
        print("  TAIA-DATL PIPELINE")
        print("=" * 72)
        start = time.time()

        # 1. Data
        if skip_data_prep:
            train_df, test_df = self.load_prepared_data()
        else:
            assert filepath, "filepath required when skip_data_prep=False"
            train_df, test_df = self.prepare_data(filepath)

        num_activities = int(train_df["activity_encoded"].max()) + 1
        print(f"[Pipeline] {num_activities} activity classes")

        # Compute thresholds once from training data and freeze for all splits.
        self.domain_thresholds = compute_domain_thresholds(
            train_df, self.cfg.max_sequence_length
        )
        self.rt_thresholds = compute_rt_thresholds(train_df)
        print(f"[Pipeline] Domain thresholds: {self.domain_thresholds}")
        print(f"[Pipeline] RT thresholds:     {self.rt_thresholds}")

        # 3. Backbone
        model, tokenizer = self.load_backbone()

        # 2. Domain prompt
        domain_desc = self.generate_domain_prompt(train_df, model, tokenizer)
        print(f"[Pipeline] Domain: {domain_desc.get('label', 'N/A')}")

        # 6. Few-shot CSV
        fs_loader = FewShotCSVLoader(self.cfg.few_shot_csv)
        if self.cfg.no_few_shot:
            print("[Pipeline] Few-shot CSV DISABLED (ablation)")
            fs_loader = FewShotCSVLoader(None)

        # 4. Stage-1: DATL pre-training (DATLEncoder with triplet loss)
        #    Purpose: build a well-structured FAISS index for RT retrieval.
        #    DATLEncoder is NOT the prediction backbone.
        encoder = self.train_datl(train_df, num_activities)

        # 5. Stage-2: supervised fine-tuning of TinyLLM + heads (CE + MSE)
        #    TinyLLM is the prediction backbone; heads are trained here.
        act_head, time_head = None, None
        if model is not None:
            act_head, time_head = self.train_heads(
                train_df, num_activities, model, tokenizer
            )

        # 6. TAIA: drop FFN LoRA deltas before inference so that only the
        #    attention deltas remain active (selective-attention inference).
        if model is not None and not self.cfg.no_taia and self.cfg.taia_drop_ffn:
            drop_ffn_deltas(model)
            print("[Pipeline] TAIA applied — FFN LoRA deltas zeroed")
        elif self.cfg.no_taia:
            print("[Pipeline] TAIA DISABLED (ablation)")

        # Load scaler to inverse-transform RT predictions to original time scale.
        rt_mu, rt_sigma = 0.0, 1.0
        scaler_path = self.cfg.clean_data_dir / f"{self.cfg.dataset_name}_scaler.pkl"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            rt_idx = FEATURE_COLUMNS.index("remaining_time")
            rt_mu    = float(scaler.mean_[rt_idx])
            rt_sigma = float(scaler.scale_[rt_idx])
            print(f"[Pipeline] RT scaler loaded — mu={rt_mu:.4f}  sigma={rt_sigma:.4f}")
        else:
            print("[Pipeline] WARNING: scaler not found — RT metrics will be in normalised scale")

        # 7. Evaluate using TinyLLM hidden states for prediction
        results = self.evaluate(
            test_df, encoder, num_activities, domain_desc,
            model, tokenizer,
            act_head=act_head, time_head=time_head,
            rt_mu=rt_mu, rt_sigma=rt_sigma,
        )

        elapsed = time.time() - start
        results["elapsed_seconds"] = elapsed
        results["config"] = {
            "no_taia": self.cfg.no_taia,
            "no_datl": self.cfg.no_datl,
            "no_faiss": self.cfg.no_faiss,
            "no_domain_prompt": self.cfg.no_domain_prompt,
            "no_few_shot": self.cfg.no_few_shot,
            "backbone_lstm": self.cfg.backbone_lstm,
        }

        # Save results
        res_path = self.cfg.results_dir / f"{self.cfg.dataset_name}_results.json"
        with open(res_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[Pipeline] Results saved >>>>>> {res_path}")
        print(f"[Pipeline] time >>>> {elapsed:.1f}s")


        return results


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="TAIA-DATL Pipeline")
    p.add_argument("--filepath", type=str, default=None,
                   help="Path to raw XES or CSV file")
    p.add_argument("--dataset", type=str, default="bpi2012",
                   help="Dataset name")
    p.add_argument("--skip-data-prep", action="store_true",
                   help="Skip data preparation (use existing clean data)")

    # Model
    p.add_argument("--hf-model", type=str, default=None,
                   help="HuggingFace model name (overrides config)")
    p.add_argument("--load-in-4bit", action="store_true",
                   help="Load TinyLLM in 4-bit quantisation")
    p.add_argument("--skip-finetune", action="store_true",
                   help="Skip LoRA fine-tuning")

    # Ablation flags
    p.add_argument("--no-taia", action="store_true",
                   help="Ablation: disable TAIA selective-attention")
    p.add_argument("--no-datl", action="store_true",
                   help="Ablation: disable DATL pre-training")
    p.add_argument("--no-faiss", action="store_true",
                   help="Ablation: use random sampling instead of FAISS")
    p.add_argument("--no-domain-prompt", action="store_true",
                   help="Ablation: disable domain prompt generation")
    p.add_argument("--no-few-shot", action="store_true",
                   help="Ablation: disable few-shot CSV")
    p.add_argument("--backbone-lstm", action="store_true",
                   help="Ablation: replace TinyLLM with plain LSTM backbone")

    # Misc
    p.add_argument("--few-shot-csv", type=str, default=None,
                   help="Path to few-shot exemplar CSV")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    cfg = TAIADATLConfig()
    cfg.dataset_name = args.dataset
    cfg.seed = args.seed
    cfg.no_taia = args.no_taia
    cfg.no_datl = args.no_datl
    cfg.no_faiss = args.no_faiss
    cfg.no_domain_prompt = args.no_domain_prompt
    cfg.no_few_shot = args.no_few_shot
    cfg.backbone_lstm = args.backbone_lstm
    cfg.skip_finetune = args.skip_finetune
    cfg.hf_load_in_4bit = args.load_in_4bit

    if args.hf_model:
        cfg.hf_model_name = args.hf_model
    if args.few_shot_csv:
        cfg.few_shot_csv = args.few_shot_csv

    pipeline = TAIADATLPipeline(cfg)
    pipeline.run(filepath=args.filepath, skip_data_prep=args.skip_data_prep)


if __name__ == "__main__":
    main()