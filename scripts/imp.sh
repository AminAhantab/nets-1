#! /bin/bash


if [ $# -ne 7 ]; then
    echo "Usage: $0 <seed> <arch> <dataset> <opt> <lr> <pr> <cycles>"
    exit 1
fi

set -euxo pipefail

# Path to python binary (conda environment)
PYTHON_BIN=/home/$USER/.conda/envs/nets/bin/python

# Output path for results
OUT_PATH="results/imp"

# Logging Configuration
LOG_LEVEL=DEBUG
LOG_EVERY=100
LOG_VAL_EVERY=500 # 0 means no validation other than end of epoch
LOG_TEST_EVERY=0 # 0 means no testing other than end of epoch

# Experiment Configuration
SEED=$1
EPOCHS=1
VAL_SIZE=5000
CYCLES=$7

# Network Configuration
ARCH=$2
DATASET=$3
INIT_DENSITY=1.0
BATCH_SIZE=60
PRUNING_RATE=$6

# Training Configuration
OPTIMISER=$4
LR=$5

# Make unique output path
TIMESTAMP=$(date +%s)
OUTPUT_PATH="$OUT_PATH/$ARCH/$DATASET/$TIMESTAMP"
LOG_PATH="$OUTPUT_PATH/${ARCH}_imp.log"
mkdir -p $OUTPUT_PATH

# Timestamped Paths
INIT_MODEL_PATH="$OUTPUT_PATH/init.pt"
RESULTS_PATH_PREFIX="$OUTPUT_PATH/results_"
TRAINED_MODEL_PATH_PREFIX="$OUTPUT_PATH/trained_"
PRUNED_MODEL_PATH_PREFIX="$OUTPUT_PATH/pruned_"
WINNING_TICKET_PATH="$OUTPUT_PATH/winning_ticket.pt"
TRAINED_TICKET_PATH="$OUTPUT_PATH/trained_ticket.pt"

# Initialise Network
$PYTHON_BIN -m nets_cli init \
    --data=$DATASET \
    --arch=$ARCH \
    --density=$INIT_DENSITY \
    --out_path=$INIT_MODEL_PATH \
    --log_level=$LOG_LEVEL \
    --log_file=$LOG_PATH

# Train Initial Overparametrised Network
$PYTHON_BIN -m nets_cli train \
    --data=$DATASET \
    --model=$INIT_MODEL_PATH \
    --out_path="${TRAINED_MODEL_PATH_PREFIX}0.pt" \
    --csv_path="${RESULTS_PATH_PREFIX}0.csv" \
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
    --epochs=$EPOCHS

# Begin cycles

for ((i=0; i<$CYCLES; i++)); do
    echo "Beginning cycle $i"

    # if last cycle set paths differently
    if [ $i -eq $((CYCLES-1)) ]; then
        PRUNED_MODEL_PATH=$WINNING_TICKET_PATH
        TRAINED_MODEL_PATH=$TRAINED_TICKET_PATH
    else
        PRUNED_MODEL_PATH=$PRUNED_MODEL_PATH_PREFIX$i.pt
        TRAINED_MODEL_PATH=$TRAINED_MODEL_PATH_PREFIX$i.pt
    fi

    # Prune
    $PYTHON_BIN -m nets_cli prune \
        --model=$TRAINED_MODEL_PATH \
        --criterion=magnitude \
        --fraction=$PRUNING_RATE \
        --out_path=$PRUNED) \
        --log_level=$LOG_LEVEL \
        --log_file=$LOG_PATH

    # Train
    $PYTHON_BIN -m nets_cli train \
        --data=$DATASET \
        --model=$PRUNED_MODEL_PATH \
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
done

echo "Iterative Magnitude Pruning complete! Results saved to $OUTPUT_PATH"