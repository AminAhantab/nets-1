#! /bin/bash -l

#SBATCH --job-name=lmsfms
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

# Define parameters for each array job task
SEED=$((42 + $SLURM_ARRAY_TASK_ID))
TRIAL=$SLURM_ARRAY_TASK_ID  # Used for filenames
OUTPUT_DIR="/scratch/users/k1502897/alife/sweep"
ARCH=conv-6  # Architecture
DATASET=cifar10  # Target dataset
OPTIMISER=adam  # Optimiser
LR=3e-4  # Learning rate
MAX_ITER=50000  # Maximum (final) training iterations
FITNESS=1epoch  # NeTS fitness function

mkdir -p $OUTPUT_DIR

# max_gen 100
# pop_size (5, 10, 50)
# mr_disable (0, 0.1, 0.2, 0.3)
# mr_noise (0, 0.1, 0.2, 0.3)
# mr_random (0, 0.1, 0.2, 0.3)

# Run the experiment for the current task
$PYTHON_BIN -m experiments.alife.experiment \
    --trial 0 \
    --seed $SEED \
    --out_dir $OUTPUT_DIR \
    --arch $ARCH \
    --dataset $DATASET \
    --optimiser $OPTIMISER \
    --lr $LR \
    --max_iter $MAX_ITER \
    --fitness $FITNESS

echo "Job $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID finished successfully"
