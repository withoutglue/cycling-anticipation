#!/bin/bash
#SBATCH --job-name=cyclist_probe
#SBATCH --account=project_XXXXXXX
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:gh200:1
#SBATCH --output=/scratch/project_XXXXXXX/cycling/logs/%x_%j.out
#SBATCH --error=/scratch/project_XXXXXXX/cycling/logs/%x_%j.err

set -e

BASE=${BASE:-/scratch/project_XXXXXXX}
VJEPA2=$BASE/cycling/vjepa2
DATA=$BASE/cycling

# --probe_type: attentive (paper default) | linear (low-capacity baseline) | mlp
PROBE_TYPE=${PROBE_TYPE:-attentive}
DATA_FRACTION=${DATA_FRACTION:-1.0}

# Output dir encodes both probe type and data fraction (e.g. probe_linear_f0.50)
FRAC_SUFFIX=$(python3 -c "f=${DATA_FRACTION}; print('' if f==1.0 else f'_f{f:.2f}')")

mkdir -p $DATA/logs

module purge
module load python-pytorch/2.10

source $BASE/vjepa_venv/bin/activate

FEATURES=$DATA/features.h5
OUT_DIR=$DATA/checkpoints/probe_${PROBE_TYPE}${FRAC_SUFFIX}

mkdir -p $OUT_DIR

# Step 2: train probe on pre-encoded features (~10 sec/epoch)
python $BASE/train_probe.py \
    --features        $FEATURES \
    --train_sessions  $DATA/train_sessions.txt \
    --val_sessions    $DATA/val_sessions.txt \
    --vjepa2_dir      $VJEPA2 \
    --output_dir      $OUT_DIR \
    --probe_type      $PROBE_TYPE \
    --data_fraction   $DATA_FRACTION \
    --epochs          50 \
    --batch_size      64 \
    --lr              1e-4 \
    --wd              1e-2 \
    --num_workers     8

echo "Done: ${PROBE_TYPE} probe encoder (data_fraction=${DATA_FRACTION})"
