#!/bin/bash
#SBATCH --job-name=cyclist_encode
#SBATCH --account=project_XXXXXXX
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:gh200:1
#SBATCH --output=/scratch/project_XXXXXXX/cycling/logs/%x_%j.out
#SBATCH --error=/scratch/project_XXXXXXX/cycling/logs/%x_%j.err

set -e

BASE=${BASE:-/scratch/project_XXXXXXX}
DATA=$BASE/cycling
VJEPA2=$DATA/vjepa2

mkdir -p $DATA/logs

module purge
module load python-pytorch/2.10

source $BASE/vjepa_venv/bin/activate

FEATURES_PRED=$DATA/features_predictor.h5
echo "Encoding encoder+predictor features → $FEATURES_PRED"

python $BASE/encode_features.py \
    --clips_h5    $DATA/clips.h5 \
    --manifest    $DATA/manifest.csv \
    --vjepa2_dir  $VJEPA2 \
    --checkpoint  $DATA/checkpoints/vitg.pt \
    --output      $FEATURES_PRED \
    --use_predictor \
    --batch_size  8

echo "Done. features_predictor.h5 written."
