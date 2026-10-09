# Cyclist Maneuver Anticipation with V-JEPA 2

Master's thesis project (Data Science, University of Helsinki, 2026).

Can a frozen video world model anticipate what a cyclist will do at an intersection *before* they do it?
This repo probes frozen [V-JEPA 2](https://github.com/facebookresearch/vjepa2) ViT-g representations of first-person
cycling video to classify the upcoming maneuver — **left, right, straight, or stop** — seconds before the cyclist
reaches the intersection.

<!-- TODO: one headline result, e.g. "Best probe reaches X% macro accuracy at a Y s anticipation gap (chance 25%)." -->

![Pipeline](results/figures/pipeline.png)

## Approach

1. **Labeling from GPS.** Ride GPS tracks are matched to OpenStreetMap intersections; the heading change through
   each intersection gives the maneuver label (`detect_intersections.py`).
2. **Clip extraction.** For each event, a clip is cut from the video ending a fixed anticipation gap before the
   cyclist crosses the intersection (`extract_clips.py`).
3. **Frozen features.** Clips are encoded once with the frozen V-JEPA 2 ViT-g encoder, optionally concatenated with
   the frozen predictor's "imagined" future tokens (`encode_features.py`).
4. **Probing.** Lightweight probes (linear, MLP, attentive pooler) are trained on the cached features
   (`train_probe.py`), evaluated on held-out sessions (`eval_probe.py`).

Evaluation uses **macro-averaged accuracy** (mean per-class recall): the validation set is ~80% `straight`, so plain
accuracy would reward a constant predictor.

## Results

| Experiment | Figure |
|---|---|
| Probe architecture comparison | [`results/figures/comparison/`](results/figures/comparison) |
| Encoder vs. encoder + predictor | [`results/figures/probes/`](results/figures/probes) (`pred_*` = with predictor) |
| Data scaling (25 / 50 / 75 / 100 %) | `results/figures/probes/*_f0.*` |
| Accuracy vs. anticipation gap | `results/figures/probes/*/gap_analysis.png` |
| Seen vs. unseen intersections | [`results/figures/geo/`](results/figures/geo), [`results/tables/seen_unseen_accuracy.csv`](results/tables/seen_unseen_accuracy.csv) |

<!-- TODO: results table with macro accuracy per probe -->

## Repository layout

```
src/                  dataset, feature encoding, probe training and evaluation
scripts/data_prep/    GPS/IMU extraction, intersection detection, clip extraction, HDF5 packing
scripts/analysis/     result analysis and plotting
slurm/                SLURM jobs used on the CSC Roihu supercomputer
patches/              local modification to V-JEPA 2 (load encoder from a local checkpoint)
results/              aggregate figures and tables
```

## Setup

```bash
git clone https://github.com/facebookresearch/vjepa2
cd vjepa2 && git checkout 204698b45b3712590f06245fbfba32d3be539812
git apply ../cycling-anticipation/patches/vjepa2-checkpoint-path.patch
cd ../cycling-anticipation
pip install -r requirements.txt
```

Training example (on pre-encoded features):

```bash
python src/train_probe.py \
    --features features.h5 \
    --train_sessions train_sessions.txt \
    --val_sessions val_sessions.txt \
    --vjepa2_dir ../vjepa2 \
    --output_dir checkpoints/probe_attentive \
    --probe_type attentive
```

## Data

The dataset (first-person cycling video, GPS and IMU from rides in Helsinki) is **not included**: I do not hold
redistribution rights to all of it, and the GPS tracks are personal location data. The code documents the full
pipeline from raw recordings to features, so it can be applied to comparable data
(360°/action-camera video + GPS track per session).

## Acknowledgements

Built on [V-JEPA 2](https://github.com/facebookresearch/vjepa2) by Meta FAIR (MIT License); V-JEPA 2 code and weights
are not redistributed here. Computations run on CSC's Roihu supercomputer.

If you use this work, please also cite V-JEPA 2:

```bibtex
@article{assran2025vjepa2,
  title={V-JEPA~2: Self-Supervised Video Models Enable Understanding, Prediction and Planning},
  author={Assran, Mahmoud and Bardes, Adrien and Fan, David and Garrido, Quentin and Howes, Russell and
Komeili, Mojtaba and Muckley, Matthew and Rizvi, Ammar and Roberts, Claire and Sinha, Koustuv and Zholus, Artem and
Arnaud, Sergio and Gejji, Abha and Martin, Ada and Robert Hogan, Francois and Dugas, Daniel and
Bojanowski, Piotr and Khalidov, Vasil and Labatut, Patrick and Massa, Francisco and Szafraniec, Marc and
Krishnakumar, Kapil and Li, Yong and Ma, Xiaodong and Chandar, Sarath and Meier, Franziska and LeCun, Yann and
Rabbat, Michael and Ballas, Nicolas},
  journal={arXiv preprint arXiv:2506.09985},
  year={2025}
}
```
