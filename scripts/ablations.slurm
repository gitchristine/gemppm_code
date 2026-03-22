#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --array=1-18
#SBATCH --job-name=taia-ablation
#SBATCH --output=logs/slurm/ablation_%A_%a.out
#SBATCH --error=logs/slurm/ablation_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL

# TAIA-DATL Ablation Study SLURM Array Job
# Submits one job per line in scripts/ablation_params.txt
#
# Usage:
#   sbatch scripts/ablations.slurm
#
# Adjust --array=1-N where N = number of lines in ablation_params.txt

set -e

mkdir -p logs/slurm

# ---- Environment Setup ----
module purge
# module load Python/3.12.3
# module load CUDA/12.4.0
# source ~/.bashrc
# source venv/bin/activate

# ---- Fetch ablation configuration ----
PARAMS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" scripts/ablation_params.txt)

echo "=================================================="
echo " Ablation Task ID : $SLURM_ARRAY_TASK_ID"
echo " Params           : $PARAMS"
echo " Node             : $(hostname)"
echo " GPU              : $CUDA_VISIBLE_DEVICES"
echo "=================================================="

python -m taia_datl.pipeline $PARAMS
