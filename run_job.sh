#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --partition=componc_gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Check if folder argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: sbatch run_job.sh <task_folder>"
    echo "Example: sbatch run_job.sh task_radiation_site"
    exit 1
fi

TASK_FOLDER=$1
mkdir -p ${TASK_FOLDER}/log

# Redirect stdout and stderr to task folder
exec > ${TASK_FOLDER}/log/classifier.out 2> ${TASK_FOLDER}/log/classifier.err

cd /data1/tanseyw/projects/feiyang/auto_procedure_classification
source ~/.bashrc
mamba activate classifier

echo "Starting inference for task folder: ${TASK_FOLDER}"
echo "Started at: $(date)"

python run_task.py ${TASK_FOLDER}