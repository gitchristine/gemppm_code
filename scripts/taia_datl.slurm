#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --array=1-10
#SBATCH --job-name=taia-datl
#SBATCH --output=logs/slurm/taia_datl_%A_%a.out
#SBATCH --error=logs/slurm/taia_datl_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL

# TAIA-DATL SLURM Array Job
# Submits one job per line in scripts/taia_datl_params.txt
#
# Usage:
#   sbatch scripts/taia_datl.slurm
#
# Adjust --array=1-N where N = number of lines in taia_datl_params.txt

set -e

mkdir -p logs/slurm

# ---- Environment Setup ----
module purge
# module load Python/3.12.3
# module load CUDA/12.4.0
# source ~/.bashrc

# Activate virtual environment
# source venv/bin/activate

# ---- Fetch experiment parameters for this array task ----
PARAMS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" scripts/taia_datl_params.txt)

echo "=================================================="
echo " Task ID  : $SLURM_ARRAY_TASK_ID"
echo " Params   : $PARAMS"
echo " Node     : $(hostname)"
echo " GPU      : $CUDA_VISIBLE_DEVICES"
echo "=================================================="

python -m taia_datl.pipeline $PARAMS
