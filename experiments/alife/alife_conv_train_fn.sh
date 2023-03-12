#! /bin/bash -l

#SBATCH --job-name=alletf
#SBATCH --chdir=/users/k1502897/workspace/nets/
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --mem=16G
#SBATCH --signal=USR2
#SBATCH --output=/scratch/users/%u/%A_%a.out
#SBATCH --array=1-5

set -euxo pipefail

module load anaconda3/2021.05-gcc-10.3.0

PYTHON_BIN="/scratch/users/k1502897/conda/nets/bin/python"
OUTPUT_DIR="/scratch/users/k1502897/alife/cifar10/train_fn"

mkdir -p $OUTPUT_DIR

# Define parameters for each array job task
SEED=$((42 + $SLURM_ARRAY_TASK_ID))
TRIAL=$SLURM_ARRAY_TASK_ID

# Run the experiment for the current task
$PYTHON_BIN -m experiments.alife.configurable \
    --trial $TRIAL \
    --seed $SEED \
    --out_dir $OUTPUT_DIR \
    --arch conv-6 \
    --dataset cifar10 \
    --optimiser adam \
    --lr 3e-4 \
    --max_iter 50000 \
    --fitness 1epoch

echo "Job $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID finished successfully"
