#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --job-name=taia-datl-best
#SBATCH --output=logs/slurm/run_best_%j.out
#SBATCH --error=logs/slurm/run_best_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# TAIA-DATL Best Configuration Runner
# Runs the full pipeline with the best-known hyperparameters.
# Edit the PARAMS variable below after identifying the best config
# from your ablation/tuning runs.
#
# Usage:
#   sbatch scripts/run_best.slurm

set -e

mkdir -p logs/slurm

# ---- Environment Setup ----
module purge
# module load Python/3.12.3
# module load CUDA/12.4.0
# source ~/.bashrc
# source venv/bin/activate

# ---- Best configuration (update after tuning) ----
DATASET="bpi2012"
PARAMS="--dataset $DATASET --skip-data-prep --load-in-4bit"

echo "=================================================="
echo " TAIA-DATL  Best Configuration"
echo " Dataset  : $DATASET"
echo " Params   : $PARAMS"
echo " Node     : $(hostname)"
echo " GPU      : $CUDA_VISIBLE_DEVICES"
echo "=================================================="

python -m taia_datl.pipeline $PARAMS

echo ""
echo "Run complete. Results saved to results/${DATASET}_results.json"
