#! /bin/bash -l

#SBATCH --job-name=lenet_imp
#SBATCH --chdir=/users/k1502897/workspace/nets/
#SBATCH --partition=gpu
#SBATCH --mem=16G
#SBATCH --signal=USR2
#SBATCH --output=/scratch/users/%u/%A-%a.out
#SBATCH --array=0-4
#SBATCH --ntasks=5

set -euxo pipefail

module load anaconda3/2021.05-gcc-10.3.0

PYTHON_BIN="/scratch/users/k1502897/conda/nets/bin/python"
OUT_PATH="/scratch/users/k1502897/nets/lenet_imp_5"
LOG_FILE="/scratch/users/k1502897/nets/lenet_imp_5.log"

# Logging Configuration
LOG_LEVEL=DEBUG
LOG_EVERY=100
LOG_VAL_EVERY=0
LOG_TEST_EVERY=0

# Experiment Configuration
SEEDS=(42 43 44 45 46)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
OPTIMISER=adam
LR=0.001
ITERATIONS=50000
BATCH_SIZE=60
VAL_SIZE=5000
TIMESTAMP=$(date +%s)

# Initialise Network
$PYTHON_BIN -m nets_cli init \
    --data=mnist \
    --arch=lenet \
    --density=1.0 \
    --seed=$SEED \
    --out_path=$OUT_PATH/lenet/mnist/init-$SEED.pt \
    --log_level=$LOG_LEVEL \
    --log_file=$LOG_FILE \

# Train Network
$PYTHON_BIN -m nets_cli train \
    --data=mnist \
    --model=$OUT_PATH/lenet/mnist/init-$SEED.pt \
    --out_path=$OUT_PATH/lenet/mnist/trained-$SEED.pt \
    --csv_path=$OUT_PATH/lenet/mnist/training-$SEED.csv \
    --seed=$SEED \
    --log_level=$LOG_LEVEL \
    --log_file=$LOG_FILE \
    --batch=$BATCH_SIZE \
    --opt=$OPTIMISER \
    --lr=$LR \
    --val_size=$VAL_SIZE \
    --log_every=$LOG_EVERY \
    --log_val_every=$LOG_VAL_EVERY \
    --log_test_every=$LOG_TEST_EVERY \
    --iterations=$ITERATIONS
