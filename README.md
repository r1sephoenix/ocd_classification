# OCD Classification from EEG Data

This project implements a deep learning-based approach for classifying Obsessive-Compulsive Disorder (OCD) from electroencephalogram (EEG) data. It uses convolutional neural networks (CNNs) to analyze EEG signals and identify patterns associated with OCD.

## Features

- EEG data loading and preprocessing pipeline
- Wavelet transform for feature extraction
- Multiple CNN architectures (1D, 2D, and 3D) for different data representations
- Support for both subject-based and standard data splitting
- Comprehensive evaluation metrics
- Prediction capabilities for new data

## Installation

This project uses Poetry for dependency management. To install:

```bash
# Clone the repository
git clone <repository-url>
cd ocd_classification

# Install dependencies with Poetry
poetry install
```

### Requirements

- Python 3.13 or higher
- Dependencies (automatically installed by Poetry):
  - mne (EEG data processing)
  - torch (deep learning)
  - matplotlib and seaborn (visualization)
  - scikit-learn (machine learning utilities)
  - pandas (data manipulation)
  - numpy (numerical operations)

## Usage

### Training a Model

```bash
python -m ocd_classification.main --config config.yaml --data_dir /path/to/data --output_dir ./output --mode train
```

### Evaluating a Model

```bash
python -m ocd_classification.main --config config.yaml --data_dir /path/to/test_data --output_dir ./output --mode evaluate
```

### Making Predictions

```bash
python -m ocd_classification.main --config config.yaml --data_dir /path/to/new_data --output_dir ./output --mode predict
```

### Using the Evaluation Script

The project includes a dedicated script for evaluating OCD classification by providing paths to control and OCD group data:

```bash
python notebooks/ocd_evaluation.py --control /path/to/control/group/data --ocd /path/to/ocd/group/data --output ./output
```

See the [notebooks/README.md](notebooks/README.md) for more details on using this script.

## Data Format

The project expects EEG data in EDF format. For training and evaluation, a `labels.csv` file should be provided in the data directory with columns for `subject_id` and `label`.

Example data directory structure:
```
data/
├── subject1.edf
├── subject2.edf
├── subject3.edf
└── labels.csv
```

## Project Structure

```
ocd_classification/
├── notebooks/
│   ├── ocd_evaluation.py            # Script for evaluating OCD classification
│   └── README.md                    # Documentation for notebooks
├── src/
│   └── ocd_classification/
│       ├── data/
│       │   └── data_loader.py       # EEG data loading functionality
│       ├── models/
│       │   ├── cnn_model.py         # CNN model architectures
│       │   ├── evaluate.py          # Model evaluation utilities
│       │   └── train.py             # Model training utilities
│       ├── preprocessing/
│       │   ├── preprocess.py        # EEG preprocessing pipeline
│       │   └── wavelet.py           # Wavelet transform utilities
│       ├── utils/                   # Utility functions
│       ├── visualization/           # Visualization utilities
│       └── main.py                  # Main entry point
├── pyproject.toml                   # Project configuration
└── README.md                        # This file
```

## Model Architecture

The project implements multiple CNN architectures:

1. **EEGCNN**: A flexible CNN that can handle both:
   - 1D data (time series)
   - 2D data (time-frequency representations)

2. **EEG3DCNN**: A 3D CNN for more complex EEG data representations

Both models use convolutional layers followed by max pooling, and then fully connected layers with dropout for regularization. The final output is a sigmoid activation for binary classification (OCD vs. non-OCD).

## Preprocessing Pipeline

The preprocessing pipeline includes:
- Filtering (bandpass, notch)
- Artifact removal
- Independent Component Analysis (ICA)
- Wavelet transform for feature extraction
- Standardization

## License

[MIT]
