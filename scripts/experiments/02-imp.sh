#! /bin/bash


if [ $# -ne 7 ]; then
    echo "Usage: $0 <seed> <arch> <dataset> <opt> <lr> <cycles> <pr>"
    exit 1
fi

set -euxo pipefail

PYTHON_BIN="/home/$USER/.conda/envs/nets/bin/python"
OUT_PATH="results/experiments/01-train"
INIT_FILENAME="init.pt"
LOG_SUFFIX="imp.log"

# Experiment Configuration
SEED=$1
LOG_LEVEL=DEBUG
LOG_EVERY=100
LOG_VAL_EVERY=500 # 0 means no validation other than end of epoch
LOG_TEST_EVERY=0 # 0 means no testing other than end of epoch

# Network Configuration
ARCH=$2
DATASET=$3
INIT_DENSITY=1.0

# Training Configuration
ITERATIONS=2000
VAL_SIZE=5000
BATCH_SIZE=60
OPTIMISER=$4
LR=$5

# Pruning Configuration
PRUNING_CRITERION="magnitude"
CYCLES=$6
PRUNING_RATE=$7

# Make unique output path
TIMESTAMP=$(date +%s)
OUTPUT_PATH="$OUT_PATH/$ARCH/$DATASET/$TIMESTAMP"
LOG_PATH="$OUTPUT_PATH/${ARCH}_${LOG_SUFFIX}"
mkdir -p $OUTPUT_PATH

# Timestamped Paths
INIT_MODEL_PATH="$OUTPUT_PATH/$INIT_FILENAME"

# Initialise Network
$PYTHON_BIN -m nets_cli init \
    --data=$DATASET \
    --arch=$ARCH \
    --density=$INIT_DENSITY \
    --out_path=$INIT_MODEL_PATH \
    --log_level=$LOG_LEVEL \
    --log_file=$LOG_PATH

# Train Network
$PYTHON_BIN -m nets_cli imp \
    --data=$DATASET \
    --model=$INIT_MODEL_PATH \
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
    --iterations=$ITERATIONS \
    --cycles=$CYCLES \
    --reinit \
    --criterion=$PRUNING_CRITERION \
    --fraction=$PRUNING_RATE \
    --out_path=$OUTPUT_PATH \
    --csv_path=$OUTPUT_PATH

echo "Training complete. Results saved to $OUTPUT_PATH"