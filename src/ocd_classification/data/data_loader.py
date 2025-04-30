#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for loading EEG data for OCD classification.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import mne

logger = logging.getLogger(__name__)


def load_data(
    data_dir: Union[str, Path], 
    mode: str = "train"
) -> Dict:
    """
    Load EEG data from the specified directory.
    
    Parameters
    ----------
    data_dir : str or Path
        Directory containing the EEG data files.
    mode : str, optional
        Mode to load data for: 'train', 'test', or 'predict'.
        Default is 'train'.
    
    Returns
    -------
    Dict
        Dictionary containing loaded data and metadata.
    """
    data_dir = Path(data_dir)
    logger.info(f"Loading {mode} data from {data_dir}")
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir} does not exist")
    
    # Initialize data dictionary
    data = {
        "eeg_data": [],
        "labels": [] if mode != "predict" else None,
        "subject_ids": [],
        "metadata": {}
    }
    
    # Find all EEG files in the directory
    file_pattern = "*.edf"  # Assuming EDF format, adjust as needed
    eeg_files = list(data_dir.glob(file_pattern))
    
    if not eeg_files:
        raise ValueError(f"No EEG files found in {data_dir} with pattern {file_pattern}")
    
    logger.info(f"Found {len(eeg_files)} EEG files")
    
    # Load labels if in train or test mode
    if mode in ["train", "test"]:
        labels_file = data_dir / "labels.csv"
        if not labels_file.exists():
            raise FileNotFoundError(f"Labels file {labels_file} not found")
        
        labels_df = pd.read_csv(labels_file)
        logger.info(f"Loaded labels for {len(labels_df)} subjects")
    
    # Process each EEG file
    for eeg_file in eeg_files:
        try:
            # Extract subject ID from filename
            subject_id = eeg_file.stem
            
            # Load EEG data using MNE
            raw = mne.io.read_raw_edf(eeg_file, preload=True)
            
            # Store data
            data["eeg_data"].append(raw)
            data["subject_ids"].append(subject_id)
            
            # Add labels if in train or test mode
            if mode in ["train", "test"]:
                subject_label = labels_df.loc[labels_df["subject_id"] == subject_id, "label"].values
                if len(subject_label) == 0:
                    logger.warning(f"No label found for subject {subject_id}")
                    subject_label = np.nan
                else:
                    subject_label = subject_label[0]
                data["labels"].append(subject_label)
            
            logger.debug(f"Loaded data for subject {subject_id}")
            
        except Exception as e:
            logger.error(f"Error loading file {eeg_file}: {str(e)}")
            continue
    
    # Add metadata
    data["metadata"]["num_subjects"] = len(data["subject_ids"])
    data["metadata"]["num_channels"] = len(data["eeg_data"][0].ch_names) if data["eeg_data"] else 0
    data["metadata"]["sampling_rate"] = data["eeg_data"][0].info["sfreq"] if data["eeg_data"] else 0
    
    logger.info(f"Successfully loaded data for {data['metadata']['num_subjects']} subjects")
    return data


def split_data(
    data: Dict, 
    test_size: float = 0.2, 
    val_size: float = 0.1, 
    random_state: int = 42
) -> Tuple[Dict, Dict, Dict]:
    """
    Split data into training, validation, and test sets.
    
    Parameters
    ----------
    data : Dict
        Dictionary containing loaded data.
    test_size : float, optional
        Proportion of data to use for testing. Default is 0.2.
    val_size : float, optional
        Proportion of data to use for validation. Default is 0.1.
    random_state : int, optional
        Random seed for reproducibility. Default is 42.
    
    Returns
    -------
    Tuple[Dict, Dict, Dict]
        Training, validation, and test data dictionaries.
    """
    from sklearn.model_selection import train_test_split
    
    # Extract data
    eeg_data = data["eeg_data"]
    labels = data["labels"]
    subject_ids = data["subject_ids"]
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test, ids_temp, ids_test = train_test_split(
        eeg_data, labels, subject_ids, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=labels
    )
    
    # Second split: separate validation set from remaining data
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
        X_temp, y_temp, ids_temp, 
        test_size=val_ratio, 
        random_state=random_state, 
        stratify=y_temp
    )
    
    # Create data dictionaries
    train_data = {
        "eeg_data": X_train,
        "labels": y_train,
        "subject_ids": ids_train,
        "metadata": data["metadata"].copy()
    }
    
    val_data = {
        "eeg_data": X_val,
        "labels": y_val,
        "subject_ids": ids_val,
        "metadata": data["metadata"].copy()
    }
    
    test_data = {
        "eeg_data": X_test,
        "labels": y_test,
        "subject_ids": ids_test,
        "metadata": data["metadata"].copy()
    }
    
    # Update metadata
    train_data["metadata"]["num_subjects"] = len(ids_train)
    val_data["metadata"]["num_subjects"] = len(ids_val)
    test_data["metadata"]["num_subjects"] = len(ids_test)
    
    logger.info(f"Data split: {len(ids_train)} training, {len(ids_val)} validation, {len(ids_test)} test subjects")
    
    return train_data, val_data, test_data