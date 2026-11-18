# EpOf-QCNDT 5G handover prediction

MATLAB simulation of handover prediction on the 5G production dataset (Raca et al., MMSys 2020). This stage prepares the traces. The quantum model is not included yet.

## Label

Each sample is labeled from the serving cell:

- **0, maintain**, when RSRP ≥ −100 dBm, SNR ≥ −5 dB and delta RSRP > −2 dB
- **1, macro handover**, when that test fails and speed ≥ 30 km/h
- **2, small-cell handover**, when that test fails and speed < 30 km/h

A decision is held for 2 s. Details are in `documentation/labeling_convention.md`.

The model features are `RSRP`, `delta_RSRP`, `SNR` and `Speed`.

## Run

From this folder, in MATLAB:

```matlab
main_preprocess
```

The labeled tables already written under `dataset/processed/` were produced with `preprocessing/label_dataset.py` because MATLAB is not installed on the machine used to build them. `main_preprocess.m` implements the same rules.

## Outputs

- `dataset/processed/labeled_data/labeled_traces.csv`
- `dataset/processed/labeled_data/labeled_traces.mat`
- `dataset/processed/labeled_data/session_split.csv`
- `dataset/processed/labeled_data/cell_layout.csv`
- `dataset/processed/normalized_data/selected_features_scaled.csv`
- `dataset/processed/normalized_data/feature_scaling.csv`
- `dataset/dataset_analysis/dataset_analysis.txt`
- `dataset/dataset_analysis/feature_ranking.csv`
- `dataset/dataset_analysis/dataset_summary.csv`

The files in `dataset/input/5G-production-dataset` are the original traces and are not modified.

## Phase 4

`main_phase4` trains the 4-qubit EpOf-QCNDT model. The search uses the training sessions and keeps the setting with the best validation accuracy. Test sessions are left untouched.

```matlab
main_phase4
```

The run that produced `results/metrics` used `proposed_method/run_phase4.py`, because MATLAB is not installed on the machine that trained the model. The MATLAB functions implement the same circuit and the same optimizer.

## Full comparison

`main.m` trains EpOf-QCNDT and the comparison models, then writes the metric table and figures. Phases 5 to 7 are present as source. They have not been executed.

```matlab
main
```
