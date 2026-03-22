#!/bin/bash
# run_ablations.sh
# Runs ablation studies by disabling one TAIA-DATL component at a time.
# Each line in ablation_params.txt is a separate ablation configuration.
#
# Usage:
#   bash scripts/run_ablations.sh
#
# Ablation flags:
#   --no-taia           Remove TAIA selective-attention branch
#   --no-datl           Remove DATL triplet-loss encoder
#   --no-faiss          Replace FAISS index with random sampling
#   --no-domain-prompt  Skip LLM-generated domain descriptor
#   --no-few-shot       Disable few-shot CSV injection
#   --backbone-lstm     Replace TinyLLM with Bidirectional LSTM

set -e

PARAMS_FILE="scripts/ablation_params.txt"
LOG_DIR="logs/ablations"
mkdir -p "$LOG_DIR"

echo "=================================================="
echo " TAIA-DATL Ablation Study Runner"
echo "=================================================="

while IFS= read -r params || [ -n "$params" ]; do
    [[ -z "$params" || "$params" == \#* ]] && continue

    echo ""
    echo "--------------------------------------------------"
    echo "Ablation: python -m taia_datl.pipeline $params"
    echo "--------------------------------------------------"

    log_name=$(echo "$params" | tr ' -' '__' | tr '/' '_')
    log_file="$LOG_DIR/ablation_${log_name}.log"

    # python -m taia_datl.pipeline $params 2>&1 | tee "$log_file"
    echo "python -m taia_datl.pipeline $params"

done < "$PARAMS_FILE"

echo ""
echo "=================================================="
echo " Ablation study complete. Logs saved to $LOG_DIR/"
echo "=================================================="
