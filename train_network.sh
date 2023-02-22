#! /bin/bash


if [ $# -ne 1 ]; then
    echo "Usage: $0 <seed>"
    exit 1
fi

set -euxo pipefail

# Path to python binary (conda environment)
PYTHON_BIN=/home/$USER/.conda/envs/nets/bin/python

# Output path for results
OUT_PATH="results/train_network"

# Filenames
INIT_FILENAME="init.pt"
TRAINED_FILENAME="trained.pt"
RESULTS_FILENAME="results.csv"

# Logging Configuration
LOG_LEVEL=DEBUG
LOG_EVERY=100
LOG_VAL_EVERY=0 # 0 means no validation other than end of epoch
LOG_TEST_EVERY=0 # 0 means no testing other than end of epoch

# Experiment Configuration
SEED=$1
EPOCHS=1
# ITERATIONS=10000
VAL_SIZE=5000

# Network Configuration
ARCH=conv-6
DATASET=cifar10
INIT_DENSITY=1.0
BATCH_SIZE=60

# Training Configuration
OPTIMISER=adam
LR=0.001

# Make unique output path
TIMESTAMP=$(date +%s)
OUTPUT_PATH="$OUT_PATH/$ARCH/$DATASET/$TIMESTAMP"
LOG_PATH="$OUTPUT_PATH/${ARCH}_train.log"
mkdir -p $OUTPUT_PATH

# Timestamped Paths
INIT_MODEL_PATH="$OUTPUT_PATH/$INIT_FILENAME"
TRAINED_MODEL_PATH="$OUTPUT_PATH/$TRAINED_FILENAME"
RESULTS_MODEL_PATH="$OUTPUT_PATH/$RESULTS_FILENAME"

# Initialise Network
$PYTHON_BIN -m nets_cli init \
    --data=$DATASET \
    --arch=$ARCH \
    --density=$INIT_DENSITY \
    --out_path=$INIT_MODEL_PATH \
    --log_level=$LOG_LEVEL \
    --log_file=$LOG_PATH

# Train Network
$PYTHON_BIN -m nets_cli train \
    --data=$DATASET \
    --model=$INIT_MODEL_PATH \
    --out_path=$TRAINED_MODEL_PATH \
    --csv_path=$RESULTS_MODEL_PATH \
    --seed=$SEED \
    --log_level=$LOG_LEVEL \
    --log_file=$LOG_PATH \
    --batch=$BATCH_SIZE \
    --opt=$OPTIMISER \
    --lr=$LR \
    --val_size=$VAL_SIZE \
    --log_every=$LOG_EVERY \
    --log_val_every=$LOG_VAL_EVERY \
    --log_test_every=$LOG_TEST_EVERY \
    --epochs $EPOCHS

echo "Training complete. Results saved to $RESULTS_MODEL_PATH"