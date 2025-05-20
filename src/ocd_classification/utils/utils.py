"""
Utility helpers for OCD EEG classification.

This module is intentionally framework‑agnostic: no code here depends on
`nn_trainer.py` or the model definitions.  Everything should be import‑able from
scripts, notebooks or unit tests without pulling heavy deps.

Key parts
--------------

Ensure_same_shape
    Broadcast / reshape two tensors so that their shapes match.  Helps to avoid
    repetitive `view()` logic in the training loop.

AverageMeter
    Tiny class for running (weighted) average of a metric.

Compute_binary_accuracy
    Vectorized accuracy for logits or probabilities in [0,1].

SubjectBatchSampler
    Sampler that groups all examples from the same subject together while still
    randomizing subject order every epoch.

Split_by_subjects
    Convenience wrapper around ``GroupShuffleSplit`` to obtain boolean masks for
    train / val / test so that no subject leaks across splits.

Flatten_data
    Reshape an N‑dimensional feature tensor to 2‑D ``(n_samples, n_features)``.

You are welcome to extend the module, but please keep it dependency‑light
(sk‑learn and torch are already heavy enough!).
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import BatchSampler, Sampler, SubsetRandomSampler

def ensure_same_shape(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Make ``a`` and ``b`` broadcast‑compatible (identical ``shape``) by reshaping
    whichever tensor is “thinner.”

    Examples
    --------
    >>> y   = torch.randint(0, 2, (64,))          # (N,)
    >>> out = torch.rand(64, 1)                   # (N, 1)
    >>> y2, out2 = ensure_same_shape(y, out)
    >>> y2.shape, out2.shape
    (torch.Size([64, 1]), torch.Size([64, 1]))
    """
    if a.shape == b.shape:
        return a, b

    if a.ndim < b.ndim:  # e.g. (N,) vs (N,1)
        a = a.view(*b.shape)
    elif b.ndim < a.ndim:
        b = b.view(*a.shape)
    else:
        # Same #dims but different trailing size: try to unsqueeze the singleton
        if a.shape[-1] == 1:
            a = a.squeeze(-1)
        if b.shape[-1] == 1:
            b = b.squeeze(-1)
        if a.shape != b.shape:
            raise RuntimeError(f"Cannot reconcile shapes {a.shape} and {b.shape}")

    return a, b


# --------------------------------------------------------------------------- #
# metrics                                                                     #
# --------------------------------------------------------------------------- #

class AverageMeter:
    """Keeps running sum / count so that you can print epoch metrics succinctly."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.count = 0
        self.sum = 0.0
        self.avg = math.nan

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n
        self.avg = self.sum / self.count


@torch.no_grad()
def compute_binary_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Fast accuracy for binary classification where *outputs* are probabilities
    (after sigmoid) or logits (any real numbers)."""
    targets, outputs = ensure_same_shape(targets, outputs)
    preds = (outputs > 0.5).float()
    return (preds == targets).float().mean().item()


# --------------------------------------------------------------------------- #
# subject‑aware utilities                                                     #
# --------------------------------------------------------------------------- #

class SubjectBatchSampler(Sampler[List[int]]):
    """Yield batches that never mix subjects."""

    def __init__(self, subject_ids: np.ndarray, batch_size: int) -> None:
        super().__init__(data_source=None)  # type: ignore[arg-type]
        self.subject_ids = np.asarray(subject_ids)
        self.batch_size = int(batch_size)
        self._indices_by_subject = {
            sid: np.where(self.subject_ids == sid)[0]
            for sid in np.unique(self.subject_ids)
        }

    def __iter__(self):
        for sid in np.random.permutation(list(self._indices_by_subject.keys())):
            idx = self._indices_by_subject[sid]
            yield from BatchSampler(
                SubsetRandomSampler(idx), batch_size=self.batch_size, drop_last=False
            )

    def __len__(self) -> int:
        return len(self.subject_ids)


def split_by_subjects(
    subject_ids: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Boolean masks for train / val / test such that no subject leaks across splits."""
    unique_subjects = np.unique(subject_ids)
    train_val_subjects, test_subjects = train_test_split(
        unique_subjects, test_size=test_size, random_state=random_state
    )

    val_ratio = val_size / (1 - test_size)
    train_subjects, val_subjects = train_test_split(
        train_val_subjects, test_size=val_ratio, random_state=random_state
    )

    train_mask = np.isin(subject_ids, train_subjects)
    val_mask = np.isin(subject_ids, val_subjects)
    test_mask = np.isin(subject_ids, test_subjects)
    return train_mask, val_mask, test_mask


# --------------------------------------------------------------------------- #
# misc                                                                        #
# --------------------------------------------------------------------------- #

def flatten_data(x: np.ndarray) -> np.ndarray:
    """Reshape ``x`` to ``(n_samples, -1)`` for classical ML algorithms."""
    if x.ndim > 2:
        return x.reshape(x.shape[0], -1)
    return x