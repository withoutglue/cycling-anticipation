#!/bin/bash
#SBATCH --job-name=cyclist_eval
#SBATCH --account=project_XXXXXXX
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:gh200:1
#SBATCH --output=/scratch/project_XXXXXXX/cycling/logs/%x_%j.out
#SBATCH --error=/scratch/project_XXXXXXX/cycling/logs/%x_%j.err

set -e

BASE=${BASE:-/scratch/project_XXXXXXX}
DATA=$BASE/cycling
VJEPA2=$DATA/vjepa2

PROBE_TYPE=${PROBE_TYPE:-attentive}
DATA_FRACTION=${DATA_FRACTION:-1.0}
USE_PREDICTOR=${USE_PREDICTOR:-0}

FRAC_SUFFIX=$(python3 -c "f=${DATA_FRACTION}; print('' if f==1.0 else f'_f{f:.2f}')")
if [ "$USE_PREDICTOR" = "1" ]; then
    DIR_PREFIX=probe_pred_
    RUN_TAG=pred_${PROBE_TYPE}${FRAC_SUFFIX}
    FEATURES=$DATA/features_predictor.h5
else
    DIR_PREFIX=probe_
    RUN_TAG=${PROBE_TYPE}${FRAC_SUFFIX}
    FEATURES=$DATA/features.h5
fi

mkdir -p $DATA/logs $DATA/plots/$RUN_TAG

module purge
module load python-pytorch/2.10

source $BASE/vjepa_venv/bin/activate

CHECKPOINT=$DATA/checkpoints/${DIR_PREFIX}${PROBE_TYPE}${FRAC_SUFFIX}/best_probe.pt
PREDICTIONS=$DATA/predictions_${RUN_TAG}.csv
METRICS=$DATA/checkpoints/${DIR_PREFIX}${PROBE_TYPE}${FRAC_SUFFIX}/metrics.csv

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: no checkpoint at $CHECKPOINT — train first."
    exit 1
fi

echo "Evaluating probe_type=${PROBE_TYPE} data_fraction=${DATA_FRACTION}"

# Step 1: evaluate probe on val set
python $BASE/eval_probe.py \
    --features     $FEATURES \
    --checkpoint   $CHECKPOINT \
    --val_sessions $DATA/val_sessions.txt \
    --vjepa2_dir   $VJEPA2 \
    --output       $PREDICTIONS \
    --probe_type   $PROBE_TYPE \
    --batch_size   64

echo "Predictions saved to $PREDICTIONS"

# Step 2: produce analysis plots (CPU-only, runs in same job)
METRICS_ARG=""
if [ -f "$METRICS" ]; then
    METRICS_ARG="--metrics $METRICS"
fi

python $BASE/analyze_results.py \
    $METRICS_ARG \
    --predictions $PREDICTIONS \
    --output_dir  $DATA/plots/$RUN_TAG

echo "Done. Plots written to $DATA/plots/${RUN_TAG}/"
