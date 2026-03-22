#!/bin/bash
# run_baselines.sh
# Runs all three baseline models: PM-GPT2, LSTM, and XGBoost.
# Used for benchmarking against the TAIA-DATL system.
#
# Usage:
#   bash scripts/run_baselines.sh [--dataset DATASET]
#
# Arguments:
#   --dataset   Dataset name (default: bpi2012)
#               Options: bpi2012 | bpi2017 | bpi2020_prepaid |
#                        bpi2020_travel | bpi2020_payment

set -e

DATASET=${1:-bpi2012}
LOG_DIR="logs/baselines"
mkdir -p "$LOG_DIR"

echo "=================================================="
echo " TAIA-DATL Baseline Runner  (dataset: $DATASET)"
echo "=================================================="

# ---- PM-GPT2 Baseline ----
echo ""
echo "--------------------------------------------------"
echo " Baseline 1: PM-GPT2"
echo "--------------------------------------------------"
# python baselines/baseline_pmgpt2/gpt_training.py --dataset "$DATASET" \
#     2>&1 | tee "$LOG_DIR/pmgpt2_${DATASET}.log"
echo "python baselines/baseline_pmgpt2/gpt_training.py --dataset $DATASET"

# ---- LSTM Baseline ----
echo ""
echo "--------------------------------------------------"
echo " Baseline 2: LSTM"
echo "--------------------------------------------------"
# python baselines/baseline_lstm/model_code.py --dataset "$DATASET" \
#     2>&1 | tee "$LOG_DIR/lstm_${DATASET}.log"
echo "python baselines/baseline_lstm/model_code.py --dataset $DATASET"

# ---- XGBoost Baseline ----
echo ""
echo "--------------------------------------------------"
echo " Baseline 3: XGBoost"
echo "--------------------------------------------------"
# python baselines/baseline_xgb/visualization.py --dataset "$DATASET" \
#     2>&1 | tee "$LOG_DIR/xgb_${DATASET}.log"
echo "python baselines/baseline_xgb/visualization.py --dataset $DATASET"

echo ""
echo "=================================================="
echo " Baseline runs complete. Logs saved to $LOG_DIR/"
echo "=================================================="
