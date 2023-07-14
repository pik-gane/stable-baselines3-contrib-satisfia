#!/bin/bash
# shellcheck disable=SC2206
#SBATCH --job-name=${JOB_NAME}
${DEPENDENCY}
#SBATCH --output=./logs/slurm/${JOB_NAME}.out
#SBATCH --error=./logs/slurm/${JOB_NAME}.out
#SBATCH --workdir=/p/projects/ou/labs/gane/satisfia/stable-baselines3-contrib-satisfia/experiments/
#SBATCH --array=0-${ARRAY_SIZE}%

# Load modules or your own conda environment here
module load anaconda/2021.11d
export PYTHONPATH=${PYTHONPATH}:/p/projects/ou/labs/gane/satisfia/ai-safety-gridworlds-satisfia:/p/projects/ou/labs/gane/satisfia/stable-baselines3-contrib-satisfia
source activate /p/projects/ou/labs/gane/satisfia/py310-env/

# ===== Call your code below =====
${COMMAND_PLACEHOLDER}
