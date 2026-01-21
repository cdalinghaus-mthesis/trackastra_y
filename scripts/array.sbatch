#!/bin/bash
#SBATCH --job-name=trackastra_array
#SBATCH --partition=grete:shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=A100:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-7   # Adjust based on number of configs
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# List of YAML configs
CONFIGS=(
    example_config.yaml
    example_config.yaml
    example_config.yaml
    example_config.yaml
    example_config.yaml
    example_config.yaml
    example_config.yaml
    example_config.yaml
)

# Pick the config for this job
CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

echo "Running on config: $CONFIG"
echo "Job ID: $SLURM_JOB_ID | Array Task ID: $SLURM_ARRAY_TASK_ID"

# Activate environment
source ~/.bashrc        # make sure conda is available
conda activate trackastra_scale_fusion


python train.py --config "$CONFIG"


