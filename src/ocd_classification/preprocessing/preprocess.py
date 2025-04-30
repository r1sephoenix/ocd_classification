#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for preprocessing EEG data for OCD classification.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import mne
from mne.preprocessing import ICA
from sklearn.preprocessing import StandardScaler
from ocd_classification.preprocessing.wavelet import apply_wavelet_transform, extract_wavelet_features

logger = logging.getLogger(__name__)


def preprocess_data(
    data: Dict[str, Any],
    config: Dict[str, Any],
    mode: str = "train"
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Preprocess EEG data for model training or inference.

    Parameters
    ----------
    data : Dict
        Dictionary containing EEG data and metadata.
    config : Dict
        Configuration parameters for preprocessing.
    mode : str, optional
        Mode to preprocess data for: 'train', 'test', or 'predict'.
        Default is 'train'.

    Returns
    -------
    Union[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], np.ndarray]
        Preprocessed data. For 'train' mode, returns (X_train, X_val, y_train, y_val).
        For 'test' mode, returns (X_test, y_test). For 'predict' mode, returns X.
    """
    logger.info(f"Preprocessing data in {mode} mode")

    # Extract raw EEG data
    raw_data = data["eeg_data"]

    # Apply preprocessing steps to each subject's data
    processed_data = []
    for i, raw in enumerate(raw_data):
        subject_id = data["subject_ids"][i]
        logger.debug(f"Preprocessing data for subject {subject_id}")

        try:
            # Apply preprocessing pipeline
            processed = _apply_preprocessing_pipeline(raw, config)
            processed_data.append(processed)
        except Exception as e:
            logger.error(f"Error preprocessing data for subject {subject_id}: {str(e)}")
            continue

    # Convert to numpy arrays
    X = np.array(processed_data)

    if mode == "predict":
        return X

    # Get labels
    y = np.array(data["labels"])

    if mode == "test":
        return X, y

    # For train mode, split data into train and validation sets
    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=config.get("val_size", 0.2),
        random_state=config.get("random_state", 42),
        stratify=y
    )

    return X_train, X_val, y_train, y_val


def _apply_preprocessing_pipeline(raw: mne.io.Raw, config: Dict[str, Any]) -> np.ndarray:
    """
    Apply preprocessing pipeline to a single subject's EEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data.
    config : Dict
        Configuration parameters for preprocessing.

    Returns
    -------
    np.ndarray
        Preprocessed EEG data.
    """
    # Make a copy to avoid modifying the original data
    raw = raw.copy()

    # 1. Filter the data
    if config.get("apply_filter", True):
        l_freq = config.get("l_freq", 1.0)  # High-pass filter cutoff frequency
        h_freq = config.get("h_freq", 40.0)  # Low-pass filter cutoff frequency
        raw.filter(l_freq=l_freq, h_freq=h_freq)
        logger.debug(f"Applied bandpass filter: {l_freq}-{h_freq} Hz")

    # 2. Remove power line noise
    if config.get("remove_line_noise", True):
        line_freq = config.get("line_freq", 50.0)  # Power line frequency (50 Hz in Europe, 60 Hz in US)
        raw.notch_filter(freqs=line_freq)
        logger.debug(f"Removed power line noise at {line_freq} Hz")

    # 3. Apply ICA for artifact removal if configured
    if config.get("apply_ica", False):
        ica = ICA(
            n_components=config.get("ica_components", 15),
            random_state=config.get("random_state", 42)
        )
        ica.fit(raw)

        # Automatically detect and remove EOG artifacts
        if config.get("auto_reject_eog", True):
            eog_indices, eog_scores = ica.find_bads_eog(raw)
            ica.exclude = eog_indices
            logger.debug(f"ICA: Excluded {len(eog_indices)} components related to EOG artifacts")

        # Apply ICA to remove artifacts
        raw = ica.apply(raw)
        logger.debug("Applied ICA for artifact removal")

    # 4. Extract epochs if configured
    if config.get("extract_epochs", True):
        epoch_duration = config.get("epoch_duration", 2.0)  # Duration of each epoch in seconds
        overlap = config.get("epoch_overlap", 0.5)  # Overlap between epochs (0-1)

        # Calculate epoch parameters
        sfreq = raw.info["sfreq"]
        epoch_samples = int(epoch_duration * sfreq)
        step_samples = int(epoch_samples * (1 - overlap))

        # Extract epochs
        data = raw.get_data()
        n_channels = data.shape[0]
        n_samples = data.shape[1]

        # Calculate number of epochs
        n_epochs = max(1, (n_samples - epoch_samples) // step_samples + 1)

        # Initialize epochs array
        epochs_data = np.zeros((n_epochs, n_channels, epoch_samples))

        # Extract epochs
        for i in range(n_epochs):
            start = i * step_samples
            end = start + epoch_samples
            if end <= n_samples:
                epochs_data[i] = data[:, start:end]

        # Reshape to (n_epochs, n_channels * epoch_samples) if flatten_epochs is True
        if config.get("flatten_epochs", True):
            epochs_data = epochs_data.reshape(n_epochs, -1)

        logger.debug(f"Extracted {n_epochs} epochs of duration {epoch_duration}s with {overlap*100}% overlap")

        return epochs_data

    # If not extracting epochs, return the continuous data
    data = raw.get_data()

    # 5. Apply feature extraction if configured
    if config.get("extract_features", False):
        feature_method = config.get("feature_method", "wavelet")

        if feature_method == "wavelet":
            # Apply wavelet transform
            if config.get("wavelet_transform", True):
                wavelet = config.get("wavelet", "cmor")
                scales = config.get("wavelet_scales", [2, 4, 8, 12, 30])

                if config.get("extract_epochs", True):
                    # If we have epochs data (shape: n_epochs, n_channels, n_samples)
                    if not config.get("flatten_epochs", True):
                        data = apply_wavelet_transform(data, scales, wavelet)
                        logger.debug(f"Applied wavelet transform to epochs data with {wavelet} wavelet and scales {scales}")
                else:
                    # If we have continuous data (shape: n_channels, n_samples)
                    data = apply_wavelet_transform(data, scales, wavelet)
                    logger.debug(f"Applied wavelet transform to continuous data with {wavelet} wavelet and scales {scales}")

            # Extract features from wavelet transform
            if config.get("extract_wavelet_features", False):
                feature_type = config.get("wavelet_feature_type", "all")
                wavelet = config.get("wavelet", "cmor")
                scales = config.get("wavelet_scales", [2, 4, 8, 12, 30])

                if config.get("extract_epochs", True):
                    # If we have epochs data
                    if not config.get("flatten_epochs", True):
                        data = extract_wavelet_features(data, scales, wavelet, feature_type)
                        logger.debug(f"Extracted {feature_type} features from wavelet transformed epochs data")
                else:
                    # If we have continuous data
                    data = extract_wavelet_features(data, scales, wavelet, feature_type)
                    logger.debug(f"Extracted {feature_type} features from wavelet transformed continuous data")

        elif feature_method == "spectral":
            # Placeholder for spectral feature extraction
            logger.debug("Spectral feature extraction not yet implemented")
            pass

        elif feature_method == "connectivity":
            # Placeholder for connectivity feature extraction
            logger.debug("Connectivity feature extraction not yet implemented")
            pass

        else:
            logger.warning(f"Unknown feature extraction method: {feature_method}")

    # Flatten the data if needed
    if config.get("flatten", True):
        data = data.reshape(1, -1)

    return data


def normalize_data(
    X_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    X_test: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, ...]:
    """
    Normalize EEG data using StandardScaler.

    Parameters
    ----------
    X_train : np.ndarray
        Training data.
    X_val : np.ndarray, optional
        Validation data.
    X_test : np.ndarray, optional
        Test data.

    Returns
    -------
    Tuple[np.ndarray, ...]
        Normalized data arrays in the same order as input.
    """
    # Initialize scaler
    scaler = StandardScaler()

    # Reshape data if needed
    orig_shape = X_train.shape
    if len(orig_shape) > 2:
        X_train_2d = X_train.reshape(orig_shape[0], -1)
    else:
        X_train_2d = X_train

    # Fit scaler on training data and transform
    X_train_scaled = scaler.fit_transform(X_train_2d)

    # Reshape back if needed
    if len(orig_shape) > 2:
        X_train_scaled = X_train_scaled.reshape(orig_shape)

    result = [X_train_scaled]

    # Transform validation data if provided
    if X_val is not None:
        orig_shape_val = X_val.shape
        if len(orig_shape_val) > 2:
            X_val_2d = X_val.reshape(orig_shape_val[0], -1)
        else:
            X_val_2d = X_val

        X_val_scaled = scaler.transform(X_val_2d)

        if len(orig_shape_val) > 2:
            X_val_scaled = X_val_scaled.reshape(orig_shape_val)

        result.append(X_val_scaled)

    # Transform test data if provided
    if X_test is not None:
        orig_shape_test = X_test.shape
        if len(orig_shape_test) > 2:
            X_test_2d = X_test.reshape(orig_shape_test[0], -1)
        else:
            X_test_2d = X_test

        X_test_scaled = scaler.transform(X_test_2d)

        if len(orig_shape_test) > 2:
            X_test_scaled = X_test_scaled.reshape(orig_shape_test)

        result.append(X_test_scaled)

    return tuple(result)
