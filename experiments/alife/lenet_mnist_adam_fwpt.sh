#! /bin/bash -l

#SBATCH --job-name=lmaf
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
OUTPUT_DIR="/scratch/users/k1502897/alife"
ARCH=lenet  # Architecture
DATASET=mnist  # Target dataset
OPTIMISER=adam  # Optimiser
LR=1.2e-3  # Learning rate
MAX_ITER=50000  # Maximum (final) training iterations
FITNESS=fwpass_train  # NeTS fitness function

mkdir -p $OUTPUT_DIR

# Run the experiment for the current task
$PYTHON_BIN -m experiments.alife.experiment \
    --trial $TRIAL \
    --seed $SEED \
    --out_dir $OUTPUT_DIR \
    --arch $ARCH \
    --dataset $DATASET \
    --optimiser $OPTIMISER \
    --lr $LR \
    --max_iter $MAX_ITER \
    --fitness $FITNESS

echo "Job $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID finished successfully"
