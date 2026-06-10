"""
Experiment 1 — Backbone Comparison
====================================
Runs DATL-TAIA across all backbones and datasets.
Oyamada / MT-RNN results are added manually to the summary CSV afterwards.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from taia_datl.config import TAIADATLConfig
from taia_datl.pipeline import DTAIAPipeline
from taia_datl.competitors.ftllm import FtLLMDataset, FtLLMModel

# ---------------------------------------------------------------------------
# Backbone registry
# ---------------------------------------------------------------------------

BACKBONES: Dict[str, Dict[str, Any]] = {
    "TinyLLM": {
        "hf_model_name": "arnir0/Tiny-LLM",
        "hf_torch_dtype": "float16",
        "description": "~10M params custom tiny LLM (default D-TAIA backbone)",
        "params_label": "10M",
    },
    "Qwen2.5": {
        "hf_model_name": "Qwen/Qwen2.5-0.5B",
        "hf_torch_dtype": "float16",
        "description": "~500M params Qwen2.5-0.5B instruction model",
        "params_label": "500M",
    },
    "Llama3.2": {
        "hf_model_name": "meta-llama/Llama-3.2-1B",
        "hf_torch_dtype": "float16",
        "description": "~1B params Llama-3.2-1B (Meta licence required)",
        "params_label": "1B",
    },
    "GPT2": {
        "hf_model_name": "openai-community/gpt2-xl",
        "hf_torch_dtype": "float16",
        "description": "~1.5B params GPT-2 XL",
        "params_label": "1.5B",
    },
}

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

DEFAULT_LORA = dict(
    lora_r=8, lora_alpha=16, lora_dropout=0.05,
    finetune_lr=2e-4, finetune_batch_size=4, finetune_epochs=3,
    max_length=512, patience=3,
)

N_BOOTSTRAP = 1000
CI_ALPHA    = 0.95

# CSV columns in desired order
CSV_COLUMNS = [
    "Dataset", "Model", "Backbone", "Params",
    "NA_Acc", "NA_CI", "NA_F1", "F1_CI",
    "RT_MAE", "MAE_CI", "Runtime_Hours", "Runtime_Std",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bootstrap_ci(
    y_true:      np.ndarray,
    y_pred:      np.ndarray,
    metric:      str,
    n_resamples: int = N_BOOTSTRAP,
    alpha:       float = CI_ALPHA,
    rng:         Optional[np.random.Generator] = None,
) -> float:
    """Return half-width of a bootstrap CI for *metric* (accuracy, mae, f1)."""
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


def _load_peft_model(hf_model_name: str, hf_torch_dtype: str, lora_cfg: dict):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                 "float32": torch.float32}
    dtype = dtype_map.get(hf_torch_dtype, torch.float16)

    tokenizer = AutoTokenizer.from_pretrained(hf_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        hf_model_name, torch_dtype=dtype,
        device_map="auto", trust_remote_code=True,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg["lora_r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    peft_model  = get_peft_model(base_model, lora_config)
    n_trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    print(f"  [Backbone] {hf_model_name}  trainable={n_trainable:,}  "
          f"hidden={peft_model.config.hidden_size}")
    return peft_model, tokenizer, n_trainable


# ---------------------------------------------------------------------------
# Single backbone run (multiple seeds)
# ---------------------------------------------------------------------------

def run_backbone(
    backbone_name:  str,
    backbone_cfg:   Dict[str, Any],
    train_df:       pd.DataFrame,
    val_df:         pd.DataFrame,
    test_df:        pd.DataFrame,
    num_activities: int,
    lora_cfg:       dict,
    device:         torch.device,
    seed:           int = 42,
    n_runs:         int = 5,
) -> Dict[str, Any]:
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import f1_score as sk_f1

    print(f"\n  === DATL-TAIA / {backbone_name}: {backbone_cfg['description']} ===")

    try:
        peft_model, tokenizer, n_trainable = _load_peft_model(
            backbone_cfg["hf_model_name"],
            backbone_cfg["hf_torch_dtype"],
            lora_cfg,
        )
    except Exception as exc:
        print(f"  [SKIP] Could not load {backbone_name}: {exc}")
        return {"backbone": backbone_name, "error": str(exc)}

    hidden_dim = peft_model.config.hidden_size
    max_length = lora_cfg.get("max_length", 512)
    bsz        = lora_cfg["finetune_batch_size"]
    run_results = []

    for run_i in range(n_runs):
        run_seed = seed + run_i
        torch.manual_seed(run_seed)
        np.random.seed(run_seed)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        le       = LabelEncoder().fit(train_df["activity"].values)
        ft_model = FtLLMModel(peft_model, num_activities, hidden_dim).to(device)

        train_loader = DataLoader(
            FtLLMDataset(train_df, tokenizer, le, max_length=max_length),
            batch_size=bsz, shuffle=True, num_workers=0)
        val_loader = DataLoader(
            FtLLMDataset(val_df, tokenizer, le, max_length=max_length),
            batch_size=bsz, shuffle=False, num_workers=0)
        test_loader = DataLoader(
            FtLLMDataset(test_df, tokenizer, le, max_length=max_length),
            batch_size=bsz, shuffle=False, num_workers=0)

        opt    = optim.AdamW(
            filter(lambda p: p.requires_grad, ft_model.parameters()),
            lr=lora_cfg["finetune_lr"], weight_decay=1e-2,
        )
        ce_fn  = nn.CrossEntropyLoss()
        mse_fn = nn.MSELoss()

        best_val_acc = -1.0
        best_state   = None
        patience_ctr = 0
        train_start  = time.time()

        for epoch in range(lora_cfg["finetune_epochs"]):
            ft_model.train()
            for batch in train_loader:
                ids   = batch["input_ids"].to(device)
                mask  = batch["attention_mask"].to(device)
                na_gt = batch["next_activity"].to(device)
                rt_gt = batch["remaining_time"].to(device)

                out  = ft_model(ids, mask)
                loss = ce_fn(out["na_logits"], na_gt) + mse_fn(out["rt_pred"], rt_gt)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    [p for p in ft_model.parameters() if p.requires_grad], 1.0)
                opt.step()

            ft_model.eval()
            na_p, na_g, rt_p, rt_g = [], [], [], []
            with torch.no_grad():
                for vb in val_loader:
                    out = ft_model(vb["input_ids"].to(device),
                                   vb["attention_mask"].to(device))
                    na_p.extend(out["na_logits"].argmax(1).tolist())
                    na_g.extend(vb["next_activity"].tolist())
                    rt_p.extend(out["rt_pred"].tolist())
                    rt_g.extend(vb["remaining_time"].tolist())

            val_acc = sum(p == g for p, g in zip(na_p, na_g)) / max(len(na_g), 1)
            val_mae = float(np.mean(np.abs(np.array(rt_p) - np.array(rt_g))))
            print(f"    Run {run_i+1} Epoch {epoch+1}  "
                  f"val_acc={val_acc:.4f}  val_mae={val_mae:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state   = {k: v.cpu().clone()
                                for k, v in ft_model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= lora_cfg["patience"]:
                    print(f"    Early stopping at epoch {epoch+1}")
                    break

        train_time = time.time() - train_start

        if best_state:
            ft_model.load_state_dict(best_state)
        ft_model.eval()

        na_p, na_g, rt_p, rt_g = [], [], [], []
        with torch.no_grad():
            for tb in test_loader:
                out = ft_model(tb["input_ids"].to(device),
                               tb["attention_mask"].to(device))
                na_p.extend(out["na_logits"].argmax(1).tolist())
                na_g.extend(tb["next_activity"].tolist())
                rt_p.extend(out["rt_pred"].tolist())
                rt_g.extend(tb["remaining_time"].tolist())

        na_p_arr = np.array(na_p)
        na_g_arr = np.array(na_g)
        rt_p_arr = np.array(rt_p)
        rt_g_arr = np.array(rt_g)

        run_results.append({
            "test_acc":   float((na_p_arr == na_g_arr).mean()),
            "test_mae":   float(np.abs(rt_p_arr - rt_g_arr).mean()),
            "test_f1":    float(sk_f1(na_g_arr, na_p_arr, average="macro", zero_division=0)),
            "train_time": train_time,
            "na_p": na_p_arr, "na_g": na_g_arr,
            "rt_p": rt_p_arr, "rt_g": rt_g_arr,
        })

    # Aggregate
    rng         = np.random.default_rng(seed)
    accs        = np.array([r["test_acc"]   for r in run_results])
    maes        = np.array([r["test_mae"]   for r in run_results])
    f1s         = np.array([r["test_f1"]    for r in run_results])
    train_times = np.array([r["train_time"] for r in run_results])

    last   = run_results[-1]
    acc_ci = bootstrap_ci(last["na_g"], last["na_p"], "accuracy", rng=rng)
    mae_ci = bootstrap_ci(last["rt_g"], last["rt_p"], "mae",      rng=rng)
    f1_ci  = bootstrap_ci(last["na_g"], last["na_p"], "f1",       rng=rng)

    runtime_hours     = float(train_times.mean()) / 3600
    runtime_hours_std = float(train_times.std(ddof=1)) / 3600 if n_runs > 1 else 0.0
    peak_gpu_mb       = (torch.cuda.max_memory_allocated() / 1e6
                         if torch.cuda.is_available() else 0.0)

    return {
        "Dataset":       "",          # filled by caller
        "Model":         "DATL-TAIA",
        "Backbone":      backbone_name,
        "Params":        backbone_cfg["params_label"],
        "NA_Acc":        round(float(accs.mean()) * 100, 1),
        "NA_CI":         round(acc_ci * 100, 1),
        "NA_F1":         round(float(f1s.mean()), 3),
        "F1_CI":         round(f1_ci, 3),
        "RT_MAE":        round(float(maes.mean()), 2),
        "MAE_CI":        round(mae_ci, 2),
        "Runtime_Hours": round(runtime_hours, 3),
        "Runtime_Std":   round(runtime_hours_std, 3),
        "_n_trainable":  n_trainable,
        "_peak_gpu_mb":  peak_gpu_mb,
    }


# ---------------------------------------------------------------------------
# Per-dataset runner
# ---------------------------------------------------------------------------

def run_dataset(
    dataset:        str,
    backbone_names: List[str],
    output_dir:     Path,
    lora_cfg:       dict,
    seed:           int,
    n_runs:         int,
) -> List[Dict[str, Any]]:

    cfg              = TAIADATLConfig()
    cfg.dataset_name = dataset
    cfg.seed         = seed
    device           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipeline = DTAIAPipeline(cfg)
    train_df, val_df, test_df = pipeline.load_prepared_data()
    if val_df is None:
        print(f"[SKIP] {dataset}: no validation split found.")
        return []

    num_activities  = int(train_df["activity_encoded"].max()) + 1
    dataset_display = DATASET_DISPLAY.get(dataset, dataset.upper())
    print(f"\n{'='*64}")
    print(f"Dataset: {dataset_display}  |  activities={num_activities}  "
          f"train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
    print(f"{'='*64}")

    results = []
    for bname in backbone_names:
        row = run_backbone(
            bname, BACKBONES[bname],
            train_df, val_df, test_df,
            num_activities, lora_cfg, device, seed, n_runs,
        )
        row["Dataset"] = dataset_display
        results.append(row)
        print(f"  → {bname}: NA_Acc={row.get('NA_Acc','ERR')}%  "
              f"F1={row.get('NA_F1','ERR')}  "
              f"MAE={row.get('RT_MAE','ERR')}  "
              f"runtime={row.get('Runtime_Hours','ERR'):.3f}h")

    out_path = output_dir / f"backbone_comparison_{dataset}.json"
    with open(out_path, "w") as f:
        json.dump(
            [{k: v for k, v in r.items() if not k.startswith("_")} for r in results],
            f, indent=2,
        )
    print(f"  Saved → {out_path}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Backbone Comparison — DATL-TAIA")
    p.add_argument("--datasets",        nargs="+", default=ALL_DATASETS)
    p.add_argument("--backbones",       nargs="+", default=list(BACKBONES),
                   choices=list(BACKBONES))
    p.add_argument("--output-dir",      default="results/experiments")
    p.add_argument("--lora-r",          type=int,   default=8)
    p.add_argument("--lora-alpha",      type=int,   default=16)
    p.add_argument("--lora-dropout",    type=float, default=0.05)
    p.add_argument("--finetune-lr",     type=float, default=2e-4)
    p.add_argument("--finetune-batch",  type=int,   default=4)
    p.add_argument("--finetune-epochs", type=int,   default=3)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--n-runs",          type=int,   default=5,
                   help="Independent seeds per backbone for CI estimation")
    return p.parse_args()


def main():
    args       = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lora_cfg = {
        "lora_r":              args.lora_r,
        "lora_alpha":          args.lora_alpha,
        "lora_dropout":        args.lora_dropout,
        "finetune_lr":         args.finetune_lr,
        "finetune_batch_size": args.finetune_batch,
        "finetune_epochs":     args.finetune_epochs,
        "max_length":          512,
        "patience":            3,
    }

    all_rows: List[Dict[str, Any]] = []
    for ds in args.datasets:
        rows = run_dataset(ds, args.backbones, output_dir, lora_cfg, args.seed, args.n_runs)
        all_rows.extend(rows)

    if all_rows:
        public_rows = [{k: r[k] for k in CSV_COLUMNS if k in r} for r in all_rows]
        df_summary  = pd.DataFrame(public_rows, columns=CSV_COLUMNS)
        csv_path    = output_dir / "backbone_comparison_summary.csv"
        df_summary.to_csv(csv_path, index=False)
        print(f"\nSummary saved → {csv_path}")
        print("\n" + df_summary.to_string(index=False))


if __name__ == "__main__":
    main()