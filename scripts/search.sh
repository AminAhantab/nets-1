#! /bin/bash


if [ $# -ne 5 ]; then
    echo "Usage: $0 <seed> <arch> <dataset> <opt> <lr>"
    exit 1
fi

set -euxo pipefail

# Path to python binary (conda environment)
PYTHON_BIN=/home/$USER/.conda/envs/nets/bin/python

OUT_PATH="results/train_network"
FOUND_FILENAME="found.pt"
TRAINED_FILENAME="trained.pt"
EVO_RESULTS_FILENAME="evolution.csv"
TRAIN_RESULTS_FILENAME="training.csv"
LOG_SUFFIX="_search.log"

# Logging Configuration
LOG_LEVEL=DEBUG
LOG_EVERY=100
LOG_VAL_EVERY=500 # 0 means no validation other than end of epoch
LOG_TEST_EVERY=0 # 0 means no testing other than end of epoch

# Experiment Configuration
SEED=$1
ARCH=$2
DATASET=$3
DISABLE_CUDA=False

# Genetic Algorithm Configuration
INIT_DENSITY=0.1
TARGET_DENSITY=0.3
POP_SIZE=3
GENERATIONS=2
MR_DISABLE=0.1
MR_RANDOM=0.1
MR_NOISE=0.1
MR_NOISE_SCALE=0.1
P_CROSSOVER=0.5
ELITES=1
MIN_FITNESS=0.0
EVO_MAX_EPOCHS=1

# Training Configuration
OPTIMISER=$4
LR=$5
BATCH_SIZE=64
VAL_SIZE=5000
ITERATIONS=17500

# Make unique output path
TIMESTAMP=$(date +%s)
OUTPUT_PATH="$OUT_PATH/$ARCH/$DATASET/$TIMESTAMP"
mkdir -p $OUTPUT_PATH

# Timestamped Paths
LOG_PATH="$OUTPUT_PATH/${ARCH}_search.log"
FOUND_MODEL_PATH="$OUTPUT_PATH/$FOUND_FILENAME"
TRAINED_MODEL_PATH="$OUTPUT_PATH/$TRAINED_FILENAME"
EVO_RESULTS_PATH="$OUTPUT_PATH/$EVO_RESULTS_FILENAME"
TRAIN_RESULTS_PATH="$OUTPUT_PATH/$TRAIN_RESULTS_FILENAME"

# Initialise Network
$PYTHON_BIN -m nets_cli search \
    --seed=$SEED \
    --log_level=$LOG_LEVEL \
    --log_file=$LOG_PATH \
    --out_path=$FOUND_MODEL_PATH \
    --arch=$ARCH \
    --data=$DATASET \
    --opt=$OPTIMISER \
    --lr=$LR \
    --epochs=$EVO_MAX_EPOCHS \
    --batch=$BATCH_SIZE \
    --val_size=$VAL_SIZE \
    --log_every=$LOG_EVERY \
    --log_val_every=$LOG_VAL_EVERY \
    --log_test_every=$LOG_TEST_EVERY \
    --init_density=$INIT_DENSITY \
    --target_density=$TARGET_DENSITY \
    --pop=$POP_SIZE \
    --gens=$GENERATIONS \
    --mr_disable=$MR_DISABLE \
    --mr_random=$MR_RANDOM \
    --mr_noise=$MR_NOISE \
    --mr_noise_scale=$MR_NOISE_SCALE \
    --p_crossover=$P_CROSSOVER \
    --elites=$ELITES \
    --min_fitness=$MIN_FITNESS \
    --csv_path=$EVO_RESULTS_PATH

# Train Network
$PYTHON_BIN -m nets_cli train \
    --data=$DATASET \
    --model=$FOUND_MODEL_PATH \
    --out_path=$TRAINED_MODEL_PATH \
    --csv_path=$TRAIN_RESULTS_PATH \
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
    --iterations $ITERATIONS

echo "Training complete. Results saved to $OUTPUT_PATH"