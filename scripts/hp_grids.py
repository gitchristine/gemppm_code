
from __future__ import annotations
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Per-dataset grids
# ---------------------------------------------------------------------------

BPI2012_GRID: Dict[str, List[Any]] = {
    "dtaia_lr":             [5e-4, 1e-3, 2e-3],
    "dtaia_batch_size":     [64, 128, 256],
    "datl_dropout":         [0.10, 0.20, 0.30],
    "loss_alpha":           [0.5, 1.0, 2.0],
    "loss_lambda":          [0.05, 0.10, 0.20],
    "dtaia_triplet_margin": [0.5, 1.0, 2.0],
}


BPI2015_GRID: Dict[str, List[Any]] = {
    "dtaia_lr":             [1e-4, 5e-4, 1e-3],
    "dtaia_batch_size":     [16, 32, 64],
    "datl_dropout":         [0.20, 0.30, 0.50],
    "loss_alpha":           [0.5, 1.0, 2.0],
    "loss_lambda":          [0.10, 0.20, 0.30],
    "dtaia_triplet_margin": [1.0, 2.0, 3.0],
}

BPI2017_GRID: Dict[str, List[Any]] = {
    "dtaia_lr":             [1e-3, 2e-3, 5e-3],
    "dtaia_batch_size":     [128, 256, 512],
    "datl_dropout":         [0.10, 0.15, 0.20],
    "loss_alpha":           [0.5, 1.0, 2.0],
    "loss_lambda":          [0.01, 0.05, 0.10],
    "dtaia_triplet_margin": [0.5, 1.0, 2.0],
}


BPI2020_PREPAID_GRID: Dict[str, List[Any]] = {
    "dtaia_lr":             [5e-5, 1e-4, 5e-4],
    "dtaia_batch_size":     [16, 32, 64],
    "datl_dropout":         [0.20, 0.30, 0.40],
    "loss_alpha":           [1.0, 2.0, 3.0],
    "loss_lambda":          [0.10, 0.20, 0.50],
    "dtaia_triplet_margin": [1.0, 1.5, 2.0],
}

BPI2020_TRAVEL_GRID: Dict[str, List[Any]] = {
    "dtaia_lr":             [1e-4, 5e-4, 1e-3],
    "dtaia_batch_size":     [32, 64, 128],
    "datl_dropout":         [0.20, 0.25, 0.30],
    "loss_alpha":           [1.0, 2.0, 3.0],
    "loss_lambda":          [0.10, 0.20, 0.30],
    "dtaia_triplet_margin": [1.0, 1.5, 2.0],
}

BPI2020_PAYMENT_GRID: Dict[str, List[Any]] = {
    "dtaia_lr":             [5e-5, 1e-4, 5e-4],
    "dtaia_batch_size":     [16, 32, 64],
    "datl_dropout":         [0.20, 0.30, 0.40],
    "loss_alpha":           [1.0, 2.0, 3.0],
    "loss_lambda":          [0.10, 0.20, 0.50],
    "dtaia_triplet_margin": [1.0, 1.5, 2.0],
}


# ---------------------------------------------------------------------------
# Fallback universal grid (3 values per HP — used when dataset unknown)
# ---------------------------------------------------------------------------
UNIVERSAL_GRID: Dict[str, List[Any]] = {
    "dtaia_lr":             [1e-4, 1e-3, 1e-2],
    "dtaia_batch_size":     [32, 64, 128],
    "datl_dropout":         [0.10, 0.30, 0.50],
    "loss_alpha":           [0.5, 1.0, 2.0],
    "loss_lambda":          [0.01, 0.10, 0.50],
    "dtaia_triplet_margin": [0.5, 1.0, 2.0],
}


# =============================================================================
# COMPETITOR GRIDS
# =============================================================================

# ── LSTM grids ───────────────────────────────────────────────────────────────

# BPI 2012 LSTM
LSTM_BPI2012: Dict[str, List[Any]] = {
    "hidden_dim":    [128, 256, 512],
    "num_layers":    [1, 2, 3],
    "dropout":       [0.10, 0.20, 0.30],
    "learning_rate": [5e-4, 1e-3, 5e-3],
    "batch_size":    [64, 128, 256],
    "rt_weight":     [0.5, 1.0, 2.0],
}


LSTM_BPI2015: Dict[str, List[Any]] = {
    "hidden_dim":    [64, 128, 256],
    "num_layers":    [1, 2, 3],
    "dropout":       [0.30, 0.40, 0.50],
    "learning_rate": [1e-4, 5e-4, 1e-3],
    "batch_size":    [16, 32, 64],
    "rt_weight":     [1.0, 2.0, 3.0],
}


LSTM_BPI2017: Dict[str, List[Any]] = {
    "hidden_dim":    [256, 512, 1024],
    "num_layers":    [2, 3, 4],
    "dropout":       [0.10, 0.15, 0.20],
    "learning_rate": [1e-3, 2e-3, 5e-3],
    "batch_size":    [128, 256, 512],
    "rt_weight":     [0.5, 1.0, 2.0],
}


LSTM_BPI2020: Dict[str, List[Any]] = {
    "hidden_dim":    [128, 256, 512],
    "num_layers":    [1, 2, 3],
    "dropout":       [0.20, 0.30, 0.40],
    "learning_rate": [1e-4, 5e-4, 1e-3],
    "batch_size":    [32, 64, 128],
    "rt_weight":     [1.0, 2.0, 3.0],
}

_LSTM_REGISTRY: Dict[str, Dict[str, List[Any]]] = {
    "bpi2012":         LSTM_BPI2012,
    "bpi2015_1":       LSTM_BPI2015,
    "bpi2015_2":       LSTM_BPI2015,
    "bpi2015_3":       LSTM_BPI2015,
    "bpi2015_4":       LSTM_BPI2015,
    "bpi2015_5":       LSTM_BPI2015,
    "bpi2017":         LSTM_BPI2017,
    "bpi2020_prepaid": LSTM_BPI2020,
    "bpi2020_travel":  LSTM_BPI2020,
    "bpi2020_payment": LSTM_BPI2020,
}


# BPI 2012 MT-RNN
MTRNN_BPI2012: Dict[str, List[Any]] = {
    "hidden_dim":    [128, 256, 512],
    "num_layers":    [1, 2, 3],
    "dropout":       [0.10, 0.20, 0.30],
    "learning_rate": [5e-4, 1e-3, 5e-3],
    "batch_size":    [64, 128, 256],
    "mt_weight":     [0.5, 1.0, 2.0],
}

# BPI 2015 MT-RNN
MTRNN_BPI2015: Dict[str, List[Any]] = {
    "hidden_dim":    [64, 128, 256],
    "num_layers":    [1, 2, 3],
    "dropout":       [0.30, 0.40, 0.50],
    "learning_rate": [1e-4, 5e-4, 1e-3],
    "batch_size":    [16, 32, 64],
    "mt_weight":     [1.0, 2.0, 3.0],
}

# BPI 2017 MT-RNN
MTRNN_BPI2017: Dict[str, List[Any]] = {
    "hidden_dim":    [256, 512, 1024],
    "num_layers":    [2, 3, 4],
    "dropout":       [0.10, 0.15, 0.20],
    "learning_rate": [1e-3, 2e-3, 5e-3],
    "batch_size":    [128, 256, 512],
    "mt_weight":     [0.5, 1.0, 2.0],
}

# BPI 2020 MT-RNN
MTRNN_BPI2020: Dict[str, List[Any]] = {
    "hidden_dim":    [128, 256, 512],
    "num_layers":    [1, 2, 3],
    "dropout":       [0.20, 0.30, 0.40],
    "learning_rate": [1e-4, 5e-4, 1e-3],
    "batch_size":    [32, 64, 128],
    "mt_weight":     [1.0, 2.0, 3.0],
}

_MTRNN_REGISTRY: Dict[str, Dict[str, List[Any]]] = {
    "bpi2012":         MTRNN_BPI2012,
    "bpi2015_1":       MTRNN_BPI2015,
    "bpi2015_2":       MTRNN_BPI2015,
    "bpi2015_3":       MTRNN_BPI2015,
    "bpi2015_4":       MTRNN_BPI2015,
    "bpi2015_5":       MTRNN_BPI2015,
    "bpi2017":         MTRNN_BPI2017,
    "bpi2020_prepaid": MTRNN_BPI2020,
    "bpi2020_travel":  MTRNN_BPI2020,
    "bpi2020_payment": MTRNN_BPI2020,
}



# BPI 2012 XGBoost
XGBOOST_BPI2012: Dict[str, List[Any]] = {
    "n_estimators":    [200, 300, 500],
    "max_depth":       [3, 5, 7],
    "learning_rate":   [0.01, 0.05, 0.10],
    "subsample":       [0.7, 0.8, 1.0],
    "min_child_weight":[1, 3, 5],
}

# BPI 2015 XGBoost
XGBOOST_BPI2015: Dict[str, List[Any]] = {
    "n_estimators":    [100, 200, 300],
    "max_depth":       [3, 5, 7],
    "learning_rate":   [0.05, 0.10, 0.20],
    "subsample":       [0.6, 0.8, 1.0],
    "min_child_weight":[3, 5, 7],
}

# BPI 2017 XGBoost
XGBOOST_BPI2017: Dict[str, List[Any]] = {
    "n_estimators":    [300, 500, 1000],
    "max_depth":       [5, 7, 9],
    "learning_rate":   [0.01, 0.05, 0.10],
    "subsample":       [0.7, 0.8, 1.0],
    "min_child_weight":[1, 3, 5],
}

# BPI 2020 XGBoost
XGBOOST_BPI2020: Dict[str, List[Any]] = {
    "n_estimators":    [100, 200, 300],
    "max_depth":       [3, 5, 7],
    "learning_rate":   [0.05, 0.10, 0.20],
    "subsample":       [0.7, 0.8, 1.0],
    "min_child_weight":[1, 3, 5],
}

_XGBOOST_REGISTRY: Dict[str, Dict[str, List[Any]]] = {
    "bpi2012":         XGBOOST_BPI2012,
    "bpi2015_1":       XGBOOST_BPI2015,
    "bpi2015_2":       XGBOOST_BPI2015,
    "bpi2015_3":       XGBOOST_BPI2015,
    "bpi2015_4":       XGBOOST_BPI2015,
    "bpi2015_5":       XGBOOST_BPI2015,
    "bpi2017":         XGBOOST_BPI2017,
    "bpi2020_prepaid": XGBOOST_BPI2020,
    "bpi2020_travel":  XGBOOST_BPI2020,
    "bpi2020_payment": XGBOOST_BPI2020,
}



# BPI 2012 ftLLM
FTLLM_BPI2012: Dict[str, List[Any]] = {
    "lora_r":               [8, 16, 32],
    "lora_alpha":           [16, 32, 64],
    "lora_dropout":         [0.05, 0.10, 0.20],
    "finetune_lr":          [5e-5, 1e-4, 2e-4],
    "finetune_batch_size":  [4, 8, 16],
}

# BPI 2015 ftLLM
FTLLM_BPI2015: Dict[str, List[Any]] = {
    "lora_r":               [4, 8, 16],
    "lora_alpha":           [8, 16, 32],
    "lora_dropout":         [0.10, 0.15, 0.20],
    "finetune_lr":          [1e-5, 5e-5, 1e-4],
    "finetune_batch_size":  [2, 4, 8],
}

# BPI 2017 ftLLM
FTLLM_BPI2017: Dict[str, List[Any]] = {
    "lora_r":               [16, 32, 64],
    "lora_alpha":           [32, 64, 128],
    "lora_dropout":         [0.05, 0.10, 0.15],
    "finetune_lr":          [5e-5, 1e-4, 2e-4],
    "finetune_batch_size":  [8, 16, 32],
}

# BPI 2020 ftLLM
FTLLM_BPI2020: Dict[str, List[Any]] = {
    "lora_r":               [4, 8, 16],
    "lora_alpha":           [8, 16, 32],
    "lora_dropout":         [0.05, 0.10, 0.15],
    "finetune_lr":          [1e-5, 5e-5, 1e-4],
    "finetune_batch_size":  [2, 4, 8],
}

_FTLLM_REGISTRY: Dict[str, Dict[str, List[Any]]] = {
    "bpi2012":         FTLLM_BPI2012,
    "bpi2015_1":       FTLLM_BPI2015,
    "bpi2015_2":       FTLLM_BPI2015,
    "bpi2015_3":       FTLLM_BPI2015,
    "bpi2015_4":       FTLLM_BPI2015,
    "bpi2015_5":       FTLLM_BPI2015,
    "bpi2017":         FTLLM_BPI2017,
    "bpi2020_prepaid": FTLLM_BPI2020,
    "bpi2020_travel":  FTLLM_BPI2020,
    "bpi2020_payment": FTLLM_BPI2020,
}

# Master competitor registry
_COMPETITOR_REGISTRY: Dict[str, Dict[str, Dict[str, List[Any]]]] = {
    "lstm":    _LSTM_REGISTRY,
    "mt_rnn":  _MTRNN_REGISTRY,
    "xgboost": _XGBOOST_REGISTRY,
    "ftllm":   _FTLLM_REGISTRY,
}


def get_competitor_param_grid(
    dataset_name: str, competitor: str
) -> Dict[str, List[Any]]:
    """Return the HP grid for a given (dataset, competitor) pair."""
    comp_reg = _COMPETITOR_REGISTRY.get(competitor)
    if comp_reg is None:
        raise ValueError(
            f"Unknown competitor '{competitor}'. "
            f"Choose from {list(_COMPETITOR_REGISTRY)}"
        )
    grid = comp_reg.get(dataset_name)
    if grid is None:
        print(
            f"[hp_grids] Warning: no specific grid for ({dataset_name}, "
            f"{competitor}). Using first available grid."
        )
        grid = next(iter(comp_reg.values()))
    return grid


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_REGISTRY: Dict[str, Dict[str, List[Any]]] = {
    "bpi2012":          BPI2012_GRID,
    # BPI 2015 — all five municipalities share the same grid
    "bpi2015_1":        BPI2015_GRID,
    "bpi2015_2":        BPI2015_GRID,
    "bpi2015_3":        BPI2015_GRID,
    "bpi2015_4":        BPI2015_GRID,
    "bpi2015_5":        BPI2015_GRID,
    "bpi2017":          BPI2017_GRID,
    "bpi2020_prepaid":  BPI2020_PREPAID_GRID,
    "bpi2020_travel":   BPI2020_TRAVEL_GRID,
    "bpi2020_payment":  BPI2020_PAYMENT_GRID,
}


def get_param_grid(dataset_name: str) -> Dict[str, List[Any]]:
    """
    Return the hyperparameter grid for a given dataset (D-TAIA model).

    Falls back to UNIVERSAL_GRID for unknown datasets and prints a warning.
    """
    grid = _REGISTRY.get(dataset_name)
    if grid is None:
        print(
            f"[hp_grids] Warning: no specific grid for '{dataset_name}'. "
            "Using UNIVERSAL_GRID."
        )
        return UNIVERSAL_GRID
    return grid


def n_combinations(dataset_name: str, competitor: Optional[str] = None) -> int:
    """Return the total number of grid combinations for a (dataset, model) pair."""
    import math
    grid = (get_competitor_param_grid(dataset_name, competitor)
            if competitor else get_param_grid(dataset_name))
    return math.prod(len(v) for v in grid.values())


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    header = f"{'Dataset':<22} {'#Combos':>8}  Grid"
    print(header)
    print("-" * 80)
    for ds in sorted(_REGISTRY):
        grid = _REGISTRY[ds]
        combo = n_combinations(ds)
        brief = "  ".join(f"{k}({len(v)})" for k, v in grid.items())
        print(f"{ds:<22} {combo:>8}  {brief}")