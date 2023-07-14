#!/bin/bash
# shellcheck disable=SC2206
${PARTITION_OPTION}
#SBATCH --job-name=${JOB_NAME}
${MEMORY_OPTION}

#SBATCH --output=./logs/slurm/${JOB_NAME}.out
#SBATCH --error=./logs/slurm/${JOB_NAME}.out
#SBATCH --workdir=/p/projects/ou/labs/gane/satisfia/stable-baselines3-contrib-satisfia/experiments/
${GIVEN_NODE}
### No need to distribute
#SBATCH --nodes=1
${EXCLUSIVE}
#SBATCH --ntasks-per-node=1

# Load modules or your own conda environment here
${LOAD_ENV}

# ===== Call your code below =====
${COMMAND_PLACEHOLDER}
