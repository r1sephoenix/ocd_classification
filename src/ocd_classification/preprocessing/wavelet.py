#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for wavelet transform preprocessing of EEG data.
"""

import logging
import numpy as np
import pywt
from typing import List, Union, Optional

logger = logging.getLogger(__name__)


def apply_wavelet_transform(
    data: np.ndarray,
    scales: List[int] = [2, 4, 8, 12, 30],
    wavelet: str = 'cmor'
) -> np.ndarray:
    """
    Apply Continuous Wavelet Transform (CWT) on EEG data.
    
    Parameters
    ----------
    data : np.ndarray
        EEG data of shape (epochs, channels, timepoints) or (channels, timepoints)
    scales : List[int], optional
        Scales for the wavelet transform. Default is [2, 4, 8, 12, 30].
    wavelet : str, optional
        Wavelet to use for the transform. Default is 'cmor'.
    
    Returns
    -------
    np.ndarray
        Wavelet transformed data. If input is (epochs, channels, timepoints),
        output is (epochs, channels, scales, timepoints).
        If input is (channels, timepoints), output is (channels, scales, timepoints).
    """
    logger.debug(f"Applying wavelet transform with {wavelet} wavelet and scales {scales}")
    
    # Check if data is 2D (channels, timepoints) or 3D (epochs, channels, timepoints)
    is_3d = len(data.shape) == 3
    
    if is_3d:
        transformed_data = []
        for epoch in data:
            transformed_epoch = []
            for channel in epoch:
                coeffs, _ = pywt.cwt(channel, scales, wavelet)  # CWT on single channel
                transformed_epoch.append(coeffs)
            transformed_data.append(np.array(transformed_epoch))
        return np.array(transformed_data)
    else:
        transformed_data = []
        for channel in data:
            coeffs, _ = pywt.cwt(channel, scales, wavelet)  # CWT on single channel
            transformed_data.append(coeffs)
        return np.array(transformed_data)


def extract_wavelet_features(
    data: np.ndarray,
    scales: List[int] = [2, 4, 8, 12, 30],
    wavelet: str = 'cmor',
    feature_type: str = 'all'
) -> np.ndarray:
    """
    Extract features from wavelet transformed EEG data.
    
    Parameters
    ----------
    data : np.ndarray
        EEG data of shape (epochs, channels, timepoints) or (channels, timepoints)
    scales : List[int], optional
        Scales for the wavelet transform. Default is [2, 4, 8, 12, 30].
    wavelet : str, optional
        Wavelet to use for the transform. Default is 'cmor'.
    feature_type : str, optional
        Type of features to extract. Options are 'all', 'mean', 'std', 'energy'.
        Default is 'all'.
    
    Returns
    -------
    np.ndarray
        Extracted features from wavelet transformed data.
    """
    # Apply wavelet transform
    transformed_data = apply_wavelet_transform(data, scales, wavelet)
    
    # Extract features based on feature_type
    if feature_type == 'mean':
        # Calculate mean across time for each scale and channel
        if len(transformed_data.shape) == 4:  # (epochs, channels, scales, timepoints)
            features = np.mean(transformed_data, axis=3)
        else:  # (channels, scales, timepoints)
            features = np.mean(transformed_data, axis=2)
    elif feature_type == 'std':
        # Calculate standard deviation across time for each scale and channel
        if len(transformed_data.shape) == 4:
            features = np.std(transformed_data, axis=3)
        else:
            features = np.std(transformed_data, axis=2)
    elif feature_type == 'energy':
        # Calculate energy (sum of squares) across time for each scale and channel
        if len(transformed_data.shape) == 4:
            features = np.sum(transformed_data**2, axis=3)
        else:
            features = np.sum(transformed_data**2, axis=2)
    elif feature_type == 'all':
        # Combine mean, std, and energy features
        if len(transformed_data.shape) == 4:
            mean_features = np.mean(transformed_data, axis=3)
            std_features = np.std(transformed_data, axis=3)
            energy_features = np.sum(transformed_data**2, axis=3)
            features = np.concatenate([mean_features, std_features, energy_features], axis=2)
        else:
            mean_features = np.mean(transformed_data, axis=2)
            std_features = np.std(transformed_data, axis=2)
            energy_features = np.sum(transformed_data**2, axis=2)
            features = np.concatenate([mean_features, std_features, energy_features], axis=1)
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")
    
    return features