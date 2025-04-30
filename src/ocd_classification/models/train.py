#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for training PyTorch models for OCD classification.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from ocd_classification.models.cnn_model import EEGDataset, EEGCNN, EEG3DCNN

logger = logging.getLogger(__name__)


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    config: Dict[str, Any] = None
) -> nn.Module:
    """
    Train a PyTorch model for OCD classification.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Training labels.
    X_val : np.ndarray, optional
        Validation data. Default is None.
    y_val : np.ndarray, optional
        Validation labels. Default is None.
    config : Dict[str, Any], optional
        Configuration parameters for training. Default is None.
    
    Returns
    -------
    nn.Module
        Trained PyTorch model.
    """
    if config is None:
        config = {}
    
    # Set device (GPU if available, otherwise CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create datasets
    train_dataset = EEGDataset(X_train, y_train)
    if X_val is not None and y_val is not None:
        val_dataset = EEGDataset(X_val, y_val)
    
    # Create data loaders
    batch_size = config.get("batch_size", 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    if X_val is not None and y_val is not None:
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Determine input shape
    input_shape = X_train.shape[1:]
    
    # Create model
    model_type = config.get("model_type", "cnn")
    if model_type == "cnn":
        if len(input_shape) <= 3:  # 1D or 2D data
            model = EEGCNN(input_shape, config)
        else:  # 3D data
            model = EEG3DCNN(input_shape, config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Move model to device
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.get("learning_rate", 0.001),
        weight_decay=config.get("weight_decay", 0.0)
    )
    
    # Training loop
    num_epochs = config.get("num_epochs", 10)
    logger.info(f"Starting training for {num_epochs} epochs")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            train_correct += (predicted == labels).sum().item()
        
        train_loss = train_loss / len(train_dataset)
        train_acc = train_correct / len(train_dataset)
        
        # Validation phase
        if X_val is not None and y_val is not None:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    # Forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    # Statistics
                    val_loss += loss.item() * inputs.size(0)
                    predicted = (outputs > 0.5).float()
                    val_correct += (predicted == labels).sum().item()
            
            val_loss = val_loss / len(val_dataset)
            val_acc = val_correct / len(val_dataset)
            
            logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                       f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        else:
            logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                       f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    
    logger.info("Training completed")
    return model


def save_model(model: nn.Module, path: Union[str, Path]) -> None:
    """
    Save a PyTorch model to disk.
    
    Parameters
    ----------
    model : nn.Module
        PyTorch model to save.
    path : str or Path
        Path to save the model to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model state dictionary
    torch.save(model.state_dict(), path)
    logger.info(f"Model saved to {path}")


def load_model(path: Union[str, Path], config: Dict[str, Any] = None) -> nn.Module:
    """
    Load a PyTorch model from disk.
    
    Parameters
    ----------
    path : str or Path
        Path to load the model from.
    config : Dict[str, Any], optional
        Configuration parameters for the model. Default is None.
    
    Returns
    -------
    nn.Module
        Loaded PyTorch model.
    """
    if config is None:
        config = {}
    
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file {path} not found")
    
    # Determine model type and input shape from config
    model_type = config.get("model_type", "cnn")
    input_shape = config.get("input_shape", (1, 64, 100))  # Default shape
    
    # Create model
    if model_type == "cnn":
        if len(input_shape) <= 3:  # 1D or 2D data
            model = EEGCNN(input_shape, config)
        else:  # 3D data
            model = EEG3DCNN(input_shape, config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load model state dictionary
    model.load_state_dict(torch.load(path))
    model.eval()  # Set model to evaluation mode
    
    logger.info(f"Model loaded from {path}")
    return model


def train_model_with_subject_split(
    data: Dict[str, Any],
    config: Dict[str, Any] = None
) -> nn.Module:
    """
    Train a model with subject-based splitting to ensure that epochs from the same subject
    don't appear in both training and test sets.
    
    Parameters
    ----------
    data : Dict[str, Any]
        Dictionary containing EEG data, labels, and subject IDs.
    config : Dict[str, Any], optional
        Configuration parameters for training. Default is None.
    
    Returns
    -------
    nn.Module
        Trained PyTorch model.
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
    val_size = config.get("val_size", 0.1)
    random_state = config.get("random_state", 42)
    
    # First split: separate test subjects
    train_val_subjects, test_subjects = train_test_split(
        unique_subjects,
        test_size=test_size,
        random_state=random_state
    )
    
    # Second split: separate validation subjects from training subjects
    val_ratio = val_size / (1 - test_size)
    train_subjects, val_subjects = train_test_split(
        train_val_subjects,
        test_size=val_ratio,
        random_state=random_state
    )
    
    # Create masks for each set
    train_mask = np.isin(subject_ids, train_subjects)
    val_mask = np.isin(subject_ids, val_subjects)
    test_mask = np.isin(subject_ids, test_subjects)
    
    # Split data using masks
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    logger.info(f"Data split by subjects: {len(train_subjects)} training subjects, "
               f"{len(val_subjects)} validation subjects, {len(test_subjects)} test subjects")
    logger.info(f"Data shapes - X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
    
    # Train model
    model = train_model(X_train, y_train, X_val, y_val, config)
    
    return model