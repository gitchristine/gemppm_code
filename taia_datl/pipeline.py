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
# Dataset Loader
# ============================================================================

class TraceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, max_len: int = 20):
        self.max_len = max_len
        self.samples = self._build_samples(df)

    def _build_samples(self, df: pd.DataFrame) -> list[dict]:
        feat_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
        samples = []

        for case_id, cdf in df.groupby("case_id"):
            cdf = cdf.sort_values("timestamp") if "timestamp" in cdf.columns else cdf
            acts = cdf["activity_encoded"].values
            feats = cdf[feat_cols].values.astype(np.float32)
            dom = cdf["domain_id"].iloc[0] if "domain_id" in cdf.columns else 0

            for k in range(2, len(cdf)):
                prefix_acts = acts[:k]
                prefix_feats = feats[:k]
                next_act = int(acts[k]) if k < len(acts) else int(acts[-1])
                rem_time = float(cdf["remaining_time"].iloc[k]) if "remaining_time" in cdf.columns else 0.0

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
                    "domain_id": int(dom),
                    "case_id": str(case_id),
                })
        return samples

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
        ds = TraceDataset(train_df, self.cfg.max_sequence_length)
        loader = DataLoader(ds, batch_size=self.cfg.datl_batch_size, shuffle=True,
                            num_workers=0)

        # Build FAISS / random index for triplet sampling
        index = self._build_index()

        # First pass: compute initial embeddings for index building
        print("[DATL] Computing initial embeddings for FAISS index...")
        encoder.eval()
        all_embs, all_doms = [], []
        case_ids_list = []
        with torch.no_grad():
            for batch in loader:
                acts = batch["activities"].to(self.device)
                feats = batch["features"].to(self.device)
                lens = batch["length"].to(self.device)
                h = encoder(acts, feats, lens)
                all_embs.append(h.cpu().numpy())
                all_doms.append(batch["domain_id"].numpy())

        all_embs = np.concatenate(all_embs, axis=0)
        all_doms = np.concatenate(all_doms, axis=0)
        index.build(all_embs, [str(i) for i in range(len(all_embs))], all_doms.tolist())

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

                # Build triplets from current embeddings
                a_idx, p_idx, n_idx = tb.build_triplets(h_np, doms)
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
    # 5-8. Evaluate (DATL branch + optional TAIA + fusion)
    # ------------------------------------------------------------------

    def evaluate(
        self,
        test_df: pd.DataFrame,
        encoder: DATLEncoder,
        num_activities: int,
        domain_descriptor: dict,
        model=None,
        tokenizer=None,
    ) -> dict:

        encoder.eval()

        # Build prediction heads
        act_head = ActivityHead(
            encoder.embedding_dim(), num_activities,
            hidden_dim=self.cfg.hidden_dim, dropout=self.cfg.dropout,
        ).to(self.device)
        time_head = TimeHead(
            encoder.embedding_dim(),
            hidden_dim=self.cfg.hidden_dim, dropout=self.cfg.dropout,
        ).to(self.device)

        ds = TraceDataset(test_df, self.cfg.max_sequence_length)
        loader = DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=False,
                            num_workers=0)

        all_preds, all_labels = [], []
        all_time_preds, all_time_labels = [], []

        with torch.no_grad():
            for batch in loader:
                acts = batch["activities"].to(self.device)
                feats = batch["features"].to(self.device)
                lens = batch["length"].to(self.device)

                h = encoder(acts, feats, lens)
                logits = act_head(h)
                t_pred = time_head(h)

                preds = logits.argmax(dim=1).cpu().numpy()
                labels = batch["next_activity"].numpy()

                all_preds.extend(preds.tolist())
                all_labels.extend(labels.tolist())
                all_time_preds.extend(t_pred.squeeze(-1).cpu().numpy().tolist())
                all_time_labels.extend(batch["remaining_time"].numpy().tolist())

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

        # 4. DATL encoder
        encoder = self.train_datl(train_df, num_activities)

        # 7. TAIA
        if model is not None and not self.cfg.no_taia and self.cfg.taia_drop_ffn:
            drop_ffn_deltas(model)
        elif self.cfg.no_taia:
            print("[Pipeline] TAIA DISABLED (ablation)")

        # 8. Evaluate
        results = self.evaluate(
            test_df, encoder, num_activities, domain_desc, model, tokenizer,
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