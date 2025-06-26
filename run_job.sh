#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --partition=componc_gpu
#SBATCH --gres=gpus:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=16:00:00
#SBATCH --output=log/classifier.out
#SBATCH --error=log/classifier.err

# Fallback configuration if first fails
#SBATCH --gres=gpu:a100:1

cd /data1/tanseyw/projects/feiyang/auto_procedure_classification
source ~/.bashrc
mamba activate classifier

python src/procedure_classification.py