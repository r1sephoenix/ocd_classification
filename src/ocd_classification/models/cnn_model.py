"""
PyTorch CNN model for OCD classification from EEG data.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class EEGDataset(Dataset):
    """
    Dataset class for EEG data.
    """
    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None, subject_ids: Optional[np.ndarray] = None, transform=None):
        """
        Initialize the dataset.

        Parameters
        ----------
        X : np.ndarray
            EEG data.
        y : np.ndarray, optional
            Labels. Default is None.
        subject_ids : np.ndarray, optional
            Subject IDs for each sample. Default is None.
        transform : callable, optional transform to be applied on a sample.
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None
        self.subject_ids = subject_ids
        self.transform = transform

        # Group samples by subject if subject_ids are provided
        self.subject_indices = {}
        if subject_ids is not None:
            for i, subject_id in enumerate(subject_ids):
                if subject_id not in self.subject_indices:
                    self.subject_indices[subject_id] = []
                self.subject_indices[subject_id].append(i)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.X[idx]

        if self.transform:
            sample = self.transform(sample)

        if self.y is not None:
            return sample, self.y[idx]
        else:
            return sample

    def get_subject_data(self, subject_id):
        """
        Get all samples for a specific subject.

        Parameters
        ----------
        subject_id : str
            Subject ID.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple of (X, y) for the specified subject.
        """
        if self.subject_ids is None:
            raise ValueError("Subject IDs not provided during initialization")

        if subject_id not in self.subject_indices:
            raise ValueError(f"Subject ID {subject_id} not found in dataset")

        indices = self.subject_indices[subject_id]
        X_subject = self.X[indices]

        if self.y is not None:
            y_subject = self.y[indices]
            return X_subject, y_subject
        else:
            return X_subject

    def get_unique_subjects(self):
        """
        Get the list of unique subject IDs in the dataset.

        Returns
        -------
        List[str]
            List of unique subject IDs.
        """
        if self.subject_ids is None:
            raise ValueError("Subject IDs not provided during initialization")

        return list(self.subject_indices.keys())


class EEGCNN(nn.Module):
    """
    CNN model for OCD classification from EEG data.
    """
    def __init__(self, input_shape: Tuple[int, ...], config: Dict[str, Any]):
        """
        Initialize the CNN model.

        Parameters
        ----------
        input_shape : Tuple[int, ...]
            Shape of the input data (channels, height, width) or (channels, time).
        config : Dict[str, Any]
            Configuration parameters for the model.
        """
        super(EEGCNN, self).__init__()

        self.input_shape = input_shape
        self.config = config

        # Determine if input is 1D or 2D
        self.is_1d = len(input_shape) == 2

        # Get configuration parameters
        n_filters_1 = config.get("n_filters_1", 32)
        n_filters_2 = config.get("n_filters_2", 64)
        kernel_size_1 = config.get("kernel_size_1", 3)
        kernel_size_2 = config.get("kernel_size_2", 3)
        pool_size = config.get("pool_size", 2)
        dropout_rate = config.get("dropout_rate", 0.5)
        n_fc = config.get("n_fc", 128)

        # Define the model architecture
        if self.is_1d:
            # 1D CNN for time series data
            self.conv1 = nn.Conv1d(input_shape[0], n_filters_1, kernel_size_1, padding='same')
            self.pool1 = nn.MaxPool1d(pool_size)
            self.conv2 = nn.Conv1d(n_filters_1, n_filters_2, kernel_size_2, padding='same')
            self.pool2 = nn.MaxPool1d(pool_size)

            # Calculate the size of the flattened features
            self.flat_size = n_filters_2 * (input_shape[1] // (pool_size * pool_size))

        else:
            # 2D CNN for image-like data (e.g., time-frequency representations)
            self.conv1 = nn.Conv2d(input_shape[0], n_filters_1, kernel_size_1, padding='same')
            self.pool1 = nn.MaxPool2d(pool_size)
            self.conv2 = nn.Conv2d(n_filters_1, n_filters_2, kernel_size_2, padding='same')
            self.pool2 = nn.MaxPool2d(pool_size)

            # Calculate the size of the flattened features
            self.flat_size = n_filters_2 * (input_shape[1] // (pool_size * pool_size)) * (input_shape[2] // (pool_size * pool_size))

        # Fully connected layers
        self.fc1 = nn.Linear(self.flat_size, n_fc)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(n_fc, 1)

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input data.

        Returns
        -------
        torch.Tensor
            Output predictions.
        """
        # Apply convolutional layers
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))

        return x


class EEG3DCNN(nn.Module):
    """
    3D CNN model for OCD classification from EEG data.
    This model is designed for data with shape (batch_size, channels, depth, height, width).
    """
    def __init__(self, input_shape: Tuple[int, ...], config: Dict[str, Any]):
        """
        Initialize the 3D CNN model.

        Parameters
        ----------
        input_shape : Tuple[int, ...]
            Shape of the input data (channels, depth, height, width).
        config : Dict[str, Any]
            Configuration parameters for the model.
        """
        super(EEG3DCNN, self).__init__()

        self.input_shape = input_shape
        self.config = config

        # Get configuration parameters
        n_filters_1 = config.get("n_filters_1", 32)
        n_filters_2 = config.get("n_filters_2", 64)
        kernel_size = config.get("kernel_size", (3, 3, 3))
        pool_size = config.get("pool_size", (2, 2, 2))
        dropout_rate = config.get("dropout_rate", 0.5)
        n_fc = config.get("n_fc", 128)

        # Define the model architecture
        self.conv1 = nn.Conv3d(input_shape[0], n_filters_1, kernel_size, padding='same')
        self.pool1 = nn.MaxPool3d(pool_size)
        self.conv2 = nn.Conv3d(n_filters_1, n_filters_2, kernel_size, padding='same')
        self.pool2 = nn.MaxPool3d(pool_size)

        # Calculate the size of the flattened features
        self.flat_size = n_filters_2 * (input_shape[1] // (pool_size[0] * pool_size[0])) * \
                         (input_shape[2] // (pool_size[1] * pool_size[1])) * \
                         (input_shape[3] // (pool_size[2] * pool_size[2]))

        # Fully connected layers
        self.fc1 = nn.Linear(self.flat_size, n_fc)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(n_fc, 1)

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input data.

        Returns
        -------
        torch.Tensor
            Output predictions.
        """
        # Apply convolutional layers
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))

        return x
