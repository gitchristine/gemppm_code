#!/bin/bash
# run_taia_datl.sh
# Runs the full TAIA-DATL pipeline across all datasets and configurations.
# Each line in taia_datl_params.txt represents one experiment.
#
# Usage:
#   bash scripts/run_taia_datl.sh
#
# To run a single experiment manually:
#   python -m taia_datl.pipeline --dataset bpi2012 --skip-data-prep

set -e

PARAMS_FILE="scripts/taia_datl_params.txt"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

echo "=================================================="
echo " TAIA-DATL Full Pipeline Runner"
echo "=================================================="

while IFS= read -r params || [ -n "$params" ]; do
    # Skip empty lines and comments
    [[ -z "$params" || "$params" == \#* ]] && continue

    echo ""
    echo "--------------------------------------------------"
    echo "Running: python -m taia_datl.pipeline $params"
    echo "--------------------------------------------------"

    log_name=$(echo "$params" | tr ' -' '__' | tr '/' '_')
    log_file="$LOG_DIR/taia_datl_${log_name}.log"

    # python -m taia_datl.pipeline $params 2>&1 | tee "$log_file"
    echo "python -m taia_datl.pipeline $params"

done < "$PARAMS_FILE"

echo ""
echo "=================================================="
echo " All experiments complete. Logs saved to $LOG_DIR/"
echo "=================================================="
