#!/usr/bin/env bash
# =============================================================================
# run_hp_tuning_competitors.sh — SLURM Array for Competitor HP Tuning
# =============================================================================
#
# Covers four competitor baselines across all 10 datasets:
#   lstm    — Bidirectional LSTM + attention + joint NA/RT heads
#   mt_rnn  — Multi-Task BiGRU (Tax et al. 2017 style)
#   xgboost — XGBoost pair (sklearn GridSearchCV; no GPU needed)
#   ftllm   — TinyLLM + LoRA fine-tuned end-to-end
#
# Job count: 10 datasets × 4 competitors = 40 array jobs (indices 0–39).
#
# GPU allocation:
#   - lstm, mt_rnn, ftllm  → require 1 GPU
#   - xgboost              → CPU-only; SLURM still allocates a GPU node but
#     XGBoost simply ignores it (set --gres=gpu:0 for pure-CPU if supported)
#
# Usage:
#   sbatch scripts/run_hp_tuning_competitors.sh
#
#   # Dry-run job index 7 locally
#   SLURM_ARRAY_TASK_ID=7 bash scripts/run_hp_tuning_competitors.sh --dry-run
#
# After all jobs finish, merge results with D-TAIA results:
#   python scripts/collect_hp_results.py --input-dir results/hp_tuning
#
# =============================================================================
# SLURM directives
# =============================================================================
#SBATCH --job-name=dtaia_competitors_hp
#SBATCH --array=0-39                  # 40 jobs: 10 datasets × 4 competitors
#SBATCH --time=23:59:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1                  # 1 GPU; xgboost jobs will use CPU only
#SBATCH --cpus-per-task=8             # extra CPUs for XGBoost parallelism
#SBATCH --output=logs/comp_hp_%A_%a.out
#SBATCH --error=logs/comp_hp_%A_%a.err
#SBATCH --partition=gpu
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=your@email.com

set -euo pipefail

# =============================================================================
# Environment setup — edit these paths to match your cluster
# =============================================================================
PROJECT_DIR="${PROJECT_DIR:-/path/to/LLM-book-recommender}"
VENV_ACTIVATE="${VENV_ACTIVATE:-${PROJECT_DIR}/venv/bin/activate}"
PARAMS_FILE="${PROJECT_DIR}/scripts/hp_tuning_competitors_params.txt"
OUTPUT_DIR="${PROJECT_DIR}/results/hp_tuning"
LOG_DIR="${PROJECT_DIR}/logs"

# Module loads (uncomment / edit for your HPC cluster)
# module purge
# module load python/3.10
# module load cuda/11.8
# module load cudnn/8.6

if [[ -f "${VENV_ACTIVATE}" ]]; then
    # shellcheck disable=SC1090
    source "${VENV_ACTIVATE}"
else
    echo "Warning: venv not found at ${VENV_ACTIVATE}. Assuming packages available."
fi

# =============================================================================
# Read per-job parameters
# =============================================================================
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

LINE_NUM=$(( SLURM_ARRAY_TASK_ID + 1 ))
PARAMS=$(sed -n "${LINE_NUM}p" "${PARAMS_FILE}")

if [[ -z "${PARAMS}" ]]; then
    echo "ERROR: No parameters found for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

echo "============================================================"
echo "Job array index : ${SLURM_ARRAY_TASK_ID}"
echo "Params line     : ${LINE_NUM}"
echo "Parameters      : ${PARAMS}"
echo "Start time      : $(date)"
echo "Host            : $(hostname)"
echo "GPU(s)          : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "============================================================"

if [[ "${1:-}" == "--dry-run" ]]; then
    echo "[DRY-RUN] Would execute:"
    echo "  python -m taia_datl.competitor_tuning ${PARAMS} --output-dir ${OUTPUT_DIR}"
    exit 0
fi

# =============================================================================
# Run competitor hyperparameter tuning
# =============================================================================
cd "${PROJECT_DIR}"

python -m taia_datl.competitor_tuning \
    ${PARAMS} \
    --output-dir "${OUTPUT_DIR}"

echo "============================================================"
echo "Finished : $(date)"
echo "============================================================"
