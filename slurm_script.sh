#!/bin/bash
#SBATCH --job-name=steering_vectors
#SBATCH --output=slurm_runs/steering_%j.out
#SBATCH --error=slurm_runs/steering_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=killable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

cd /home/joberant/NLP_2425b/troyansky1/steering_vectors

# Redirect ALL cache to /tmp to avoid disk quota issues
export HF_HOME=/tmp/hf_cache_${SLURM_JOB_ID}
export TRANSFORMERS_CACHE=/tmp/transformers_cache_${SLURM_JOB_ID}
export HF_DATASETS_CACHE=/tmp/datasets_cache_${SLURM_JOB_ID}
export XDG_CACHE_HOME=/tmp/cache_${SLURM_JOB_ID}

# Create cache directories
mkdir -p $HF_HOME $TRANSFORMERS_CACHE $HF_DATASETS_CACHE $XDG_CACHE_HOME

# Activate conda environment
source /home/joberant/NLP_2425b/troyansky1/miniconda3/etc/profile.d/conda.sh
conda activate /home/joberant/NLP_2425b/troyansky1/nlpenv

python main.py

# Cleanup temp cache files after job completes
rm -rf /tmp/hf_cache_${SLURM_JOB_ID}
rm -rf /tmp/transformers_cache_${SLURM_JOB_ID}
rm -rf /tmp/datasets_cache_${SLURM_JOB_ID}
rm -rf /tmp/cache_${SLURM_JOB_ID}