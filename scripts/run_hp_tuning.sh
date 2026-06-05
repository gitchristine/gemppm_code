#!/usr/bin/env bash
# =============================================================================
# run_hp_tuning.sh — SLURM Array Job for D-TAIA Hyperparameter Tuning
# =============================================================================
#
# Submits one SLURM job per (dataset, model-variant) combination.
# Each job runs the full GridSearch over that combination's HP grid
# (see scripts/hp_grids.py) and writes results to results/hp_tuning/.
#
# Parallelism:
#   70 combinations (10 datasets × 7 model variants) run simultaneously,
#   each on its own GPU node.  The whole grid search for one combination
#   completes within 24 h on a single A100/V100 GPU.
#
# Usage:
#   # Submit all 70 jobs
#   sbatch scripts/run_hp_tuning.sh
#
#   # Dry-run: print the command for job index 5 without submitting
#   SLURM_ARRAY_TASK_ID=5 bash scripts/run_hp_tuning.sh --dry-run
#
# After all jobs finish, collect best params:
#   python scripts/collect_hp_results.py --input-dir results/hp_tuning
#
# =============================================================================
# SLURM directives
# =============================================================================
#SBATCH --job-name=dtaia_hp
#SBATCH --array=0-69                # 70 jobs: 10 datasets × 7 model variants
#SBATCH --time=23:59:00             # 24-hour wall-clock per job
#SBATCH --mem=32G
#SBATCH --gres=gpu:1                # 1 GPU per job
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/hp_%A_%a.out  # %A = job id, %a = array index
#SBATCH --error=logs/hp_%A_%a.err
#SBATCH --partition=gpu             # adjust to your cluster's GPU partition
# Optional: mail notification on completion/failure
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=your@email.com

set -euo pipefail

# =============================================================================
# Environment setup — edit these paths to match your cluster
# =============================================================================
PROJECT_DIR="${PROJECT_DIR:-/path/to/LLM-book-recommender}"
VENV_ACTIVATE="${VENV_ACTIVATE:-${PROJECT_DIR}/venv/bin/activate}"
PARAMS_FILE="${PROJECT_DIR}/scripts/hp_tuning_params.txt"
OUTPUT_DIR="${PROJECT_DIR}/results/hp_tuning"
LOG_DIR="${PROJECT_DIR}/logs"

# Module loads (uncomment / edit for your HPC cluster)
# module purge
# module load python/3.10
# module load cuda/11.8
# module load cudnn/8.6

# Activate virtual environment
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

# SLURM_ARRAY_TASK_ID is 0-indexed; params file is 1-indexed (sed line numbers)
LINE_NUM=$(( SLURM_ARRAY_TASK_ID + 1 ))
PARAMS=$(sed -n "${LINE_NUM}p" "${PARAMS_FILE}")

if [[ -z "${PARAMS}" ]]; then
    echo "ERROR: No parameters found for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} (line ${LINE_NUM})"
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

# =============================================================================
# Dry-run mode (for local testing without SLURM)
# =============================================================================
if [[ "${1:-}" == "--dry-run" ]]; then
    echo "[DRY-RUN] Would execute:"
    echo "  python -m taia_datl.hyperparameter_tuning ${PARAMS} --output-dir ${OUTPUT_DIR}"
    exit 0
fi

# =============================================================================
# Run hyperparameter tuning
# =============================================================================
cd "${PROJECT_DIR}"

python -m taia_datl.hyperparameter_tuning \
    ${PARAMS} \
    --output-dir "${OUTPUT_DIR}" \
    --scoring accuracy

echo "============================================================"
echo "Finished : $(date)"
echo "============================================================"
