# OCD‑EEG Classification Pipeline

Deep‑learning **and** classical‑ML framework for detecting Obsessive‑Compulsive
Disorder from EEG recordings in **BrainVision** format (`*.vhdr/*.eeg/*.vmrk`).

---
## Highlights

* **Flexible loader:** recordings split into class folders (`CONTROL/`, `OCD/`).
* **Stratified subject split** with leakage‑free train / val / test.
* End‑to‑end **notebooks**: `ocd_pipeline_nb.ipynb` (train → eval).
* Feature extraction: epoching + multitaper **PSD** (1–40 Hz).
* Choice of models
  * 3‑D / 2‑D / 1‑D **CNN** (`EEGCNN`, `EEG3DCNN`)
  * **MLP** for flat features
  * Logistic Reg., SVM, RandomForest + **Optuna‑tuned LightGBM**.
* Metrics: accuracy, F1, ROC‑AUC, confusion matrix; JSON‑export via `save_results`.

---
## Installation
```bash
# clone
$ git clone https://github.com/<you>/ocd_classification.git
$ cd ocd_classification

# install deps (Python ≥3.9) – Poetry recommended
$ poetry install --with dev
```

Main external deps: `mne`, `torch`, `lightgbm`, `optuna`, `scikit‑learn`.

---
## Quick start

### 1  Prepare data structure
```
data/
├── CONTROL/
│   ├── sub01.vhdr
│   └── …
└── OCD/
    ├── sub11.vhdr
    └── …
```
*BrainVision sidecars (`.eeg/.vmrk`) must sit next to `.vhdr`.*

### 2  Train + evaluate via CLI
```bash
poetry run python -m ocd_classification.main \
    data           \
    --output out   \
    --mode train

poetry run python -m ocd_classification.main \
    data           \
    --output out   \
    --mode evaluate
```
Config overrides live in `config.yaml`; see `config-example.yaml` for
all knobs (model sizes, Optuna trials, etc.).

### 3  Predict on new subjects
```bash
poetry run python -m ocd_classification.main \
    new_data --output out --mode predict
# → out/predictions.csv
```

### 4  Interactive notebook
Open **`notebooks/ocd_pipeline_nb.ipynb`** – runs full pipeline in Jupyter.

---
## Repository layout
```
ocd_classification/
├── data_loader.py       # load_dataset + split_dataset
├── preprocess.py        # PSD features, no splitting
├── trainers/
│   ├── nn_trainer.py    # CNN / MLP training
│   └── ml_trainer.py    # Grid/Random + Optuna‑LGBM
├── evaluate.py          # metrics + JSON helpers
├── main.py              # CLI entry‑point
└── models/              # EEGCNN, EEG3DCNN, EEGMLP
notebooks/
└── ocd_pipeline_nb.ipynb
```

---
## Pre‑processing details
* **Drop channels:** `FT9`, `empty`, blank‑named.
* **Epoching:** default −0.2 s → 0.8 s around events `S10/S20/S30` (configurable).
* **Power spectra:** multitaper PSD, log‑power (dB).
* Optional z‑score normalisation via `normalize_data`.

---
## License
[MIT](LICENSE)
