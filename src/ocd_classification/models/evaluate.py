#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for evaluating PyTorch models for OCD classification.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from ocd_classification.models.cnn_model import EEGDataset

logger = logging.getLogger(__name__)


def evaluate_model(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: Dict[str, Any] = None
) -> Dict[str, float]:
    """
    Evaluate a PyTorch model on test data.
    
    Parameters
    ----------
    model : nn.Module
        PyTorch model to evaluate.
    X_test : np.ndarray
        Test data.
    y_test : np.ndarray
        Test labels.
    config : Dict[str, Any], optional
        Configuration parameters for evaluation. Default is None.
    
    Returns
    -------
    Dict[str, float]
        Dictionary of evaluation metrics.
    """
    if config is None:
        config = {}
    
    # Set device (GPU if available, otherwise CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()  # Set model to evaluation mode
    
    # Create dataset and data loader
    test_dataset = EEGDataset(X_test, y_test)
    batch_size = config.get("batch_size", 32)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize variables for metrics
    all_preds = []
    all_probs = []
    all_labels = []
    
    # Evaluate model
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Convert outputs to predictions and probabilities
            preds = (outputs > 0.5).float().cpu().numpy()
            probs = outputs.cpu().numpy()
            labels = labels.cpu().numpy()
            
            # Store predictions and labels
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels)
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds).flatten()
    all_probs = np.array(all_probs).flatten()
    all_labels = np.array(all_labels).flatten()
    
    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1_score": f1_score(all_labels, all_preds, zero_division=0),
        "roc_auc": roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5
    }
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    metrics["true_negatives"] = cm[0, 0] if cm.shape == (2, 2) else 0
    metrics["false_positives"] = cm[0, 1] if cm.shape == (2, 2) else 0
    metrics["false_negatives"] = cm[1, 0] if cm.shape == (2, 2) else 0
    metrics["true_positives"] = cm[1, 1] if cm.shape == (2, 2) else 0
    
    # Log metrics
    logger.info(f"Evaluation metrics: {metrics}")
    
    return metrics


def predict(
    model: nn.Module,
    X: np.ndarray,
    config: Dict[str, Any] = None
) -> np.ndarray:
    """
    Make predictions using a PyTorch model.
    
    Parameters
    ----------
    model : nn.Module
        PyTorch model to use for predictions.
    X : np.ndarray
        Input data.
    config : Dict[str, Any], optional
        Configuration parameters for prediction. Default is None.
    
    Returns
    -------
    np.ndarray
        Predicted probabilities.
    """
    if config is None:
        config = {}
    
    # Set device (GPU if available, otherwise CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()  # Set model to evaluation mode
    
    # Create dataset and data loader
    dataset = EEGDataset(X)
    batch_size = config.get("batch_size", 32)
    data_loader = DataLoader(dataset, batch_size=batch_size)
    
    # Initialize array for predictions
    all_probs = []
    
    # Make predictions
    with torch.no_grad():
        for inputs in data_loader:
            if isinstance(inputs, list) and len(inputs) == 2:
                # If dataset returns (inputs, labels)
                inputs = inputs[0]
            
            inputs = inputs.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Store predictions
            all_probs.extend(outputs.cpu().numpy())
    
    # Convert to numpy array
    all_probs = np.array(all_probs).flatten()
    
    return all_probs


def save_results(
    metrics: Dict[str, float],
    path: Union[str, Path]
) -> None:
    """
    Save evaluation results to a JSON file.
    
    Parameters
    ----------
    metrics : Dict[str, float]
        Dictionary of evaluation metrics.
    path : str or Path
        Path to save the results to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python types for JSON serialization
    metrics_json = {k: float(v) if isinstance(v, (np.float32, np.float64)) else int(v) if isinstance(v, (np.int32, np.int64)) else v
                   for k, v in metrics.items()}
    
    # Save to JSON file
    with open(path, 'w') as f:
        json.dump(metrics_json, f, indent=4)
    
    logger.info(f"Evaluation results saved to {path}")


def save_predictions(
    predictions: np.ndarray,
    path: Union[str, Path],
    subject_ids: Optional[List[str]] = None
) -> None:
    """
    Save predictions to a CSV file.
    
    Parameters
    ----------
    predictions : np.ndarray
        Predicted probabilities.
    path : str or Path
        Path to save the predictions to.
    subject_ids : List[str], optional
        List of subject IDs corresponding to the predictions. Default is None.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create DataFrame for predictions
    if subject_ids is not None:
        df = pd.DataFrame({
            'subject_id': subject_ids,
            'prediction': predictions
        })
    else:
        df = pd.DataFrame({
            'prediction': predictions
        })
    
    # Save to CSV file
    df.to_csv(path, index=False)
    
    logger.info(f"Predictions saved to {path}")


def evaluate_model_with_subject_split(
    model: nn.Module,
    data: Dict[str, Any],
    config: Dict[str, Any] = None
) -> Dict[str, float]:
    """
    Evaluate a model with subject-based splitting to ensure that epochs from the same subject
    don't appear in both training and test sets.
    
    Parameters
    ----------
    model : nn.Module
        PyTorch model to evaluate.
    data : Dict[str, Any]
        Dictionary containing EEG data, labels, and subject IDs.
    config : Dict[str, Any], optional
        Configuration parameters for evaluation. Default is None.
    
    Returns
    -------
    Dict[str, float]
        Dictionary of evaluation metrics.
    """
    if config is None:
        config = {}
    
    # Extract data
    X = data["eeg_data"]
    y = data["labels"]
    subject_ids = data["subject_ids"]
    
    # Split subjects into train and test sets
    unique_subjects = np.unique(subject_ids)
    test_size = config.get("test_size", 0.2)
    random_state = config.get("random_state", 42)
    
    # Split subjects
    from sklearn.model_selection import train_test_split
    _, test_subjects = train_test_split(
        unique_subjects,
        test_size=test_size,
        random_state=random_state
    )
    
    # Create mask for test set
    test_mask = np.isin(subject_ids, test_subjects)
    
    # Split data using mask
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    logger.info(f"Evaluating on {len(test_subjects)} test subjects with {X_test.shape[0]} samples")
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test, config)
    
    return metrics