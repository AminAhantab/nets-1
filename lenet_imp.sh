#! /bin/bash

set -euxo pipefail

PYTHON_BIN=/home/$USER/.conda/envs/nets/bin/python
OUT_PATH="results/lenet_imp_5"
LOG_FILE="lenet_imp_5.log"

# Logging Configuration
LOG_LEVEL=DEBUG
LOG_EVERY=100
LOG_VAL_EVERY=0
LOG_TEST_EVERY=0

# Experiment Configuration
SEED=$1
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
    --out_path=$OUT_PATH/lenet/mnist/init.pt \
    --log_level=$LOG_LEVEL \
    --log_file=$LOG_FILE

# Train Network
$PYTHON_BIN -m nets_cli train \
    --data=mnist \
    --model=$OUT_PATH/lenet/mnist/init.pt \
    --out_path=$OUT_PATH/lenet/mnist/trained.pt \
    --csv_path=$OUT_PATH/lenet/mnist/training.csv \
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
