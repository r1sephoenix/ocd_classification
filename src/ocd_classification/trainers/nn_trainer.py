from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from ocd_classification.models.cnn_model import EEGCNN, EEG3DCNN
from ocd_classification.utils.utils import (
    AverageMeter,
    SubjectBatchSampler,
    compute_binary_accuracy,
    ensure_same_shape,
)

log = logging.getLogger(__name__)


def _build_model(input_shape: Tuple[int, ...], cfg: Mapping[str, Any]) -> nn.Module:
    cls_map = {
        "cnn": EEGCNN,
        "eegcnn": EEGCNN,
        "eeg3dcnn": EEG3DCNN,
    }
    model_key = cfg.get("model_type", "cnn").lower()
    try:
        return cls_map[model_key](input_shape, cfg)
    except KeyError:
        raise ValueError(
            f"Unknown model_type '{model_key}'. Choose from {list(cls_map)}"
        )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None = None,
    device: torch.device | str = "cpu",
) -> Tuple[float, float]:
    """Returns (loss_avg, acc_avg) for the epoch."""
    train_mode = optimizer is not None
    model.train(train_mode)

    meter_loss, meter_acc = AverageMeter(), AverageMeter()

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float()

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        out = model(x).float()
        y_, out_ = ensure_same_shape(y, out)
        loss = criterion(out_, y_)

        if train_mode:
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            acc = compute_binary_accuracy(out_, y_)
            meter_loss.update(loss.item(), n=x.size(0))
            meter_acc.update(acc, n=x.size(0))

    return meter_loss.avg, meter_acc.avg


def train_nn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    *,
    subject_ids_train: np.ndarray | None = None,
    subject_ids_val: np.ndarray | None = None,
    cfg: Mapping[str, Any] | None = None,
) -> nn.Module:
    """High‑level convenience wrapper.

    Parameters
    ----------
    x_train, y_train : ndarray
        Training data / labels.
    x_val, y_val : ndarray, optional
        Validation split. If *None*, metrics for val are skipped.
    subject_ids_*: ndarray, optional
        Subject identifiers for grouping.
    cfg : dict, optional
        Hyper‑parameters, e.g. ``epochs``, ``batch_size``, ``lr``.
    """
    cfg = dict(cfg or {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bs = cfg.get("batch_size", 32)
    group_by_subject = (
        cfg.get("group_by_subject", True) and subject_ids_train is not None
    )

    # Datasets / loaders --------------------------------------------------- #
    ds_train = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    if group_by_subject:
        sampler = SubjectBatchSampler(subject_ids_train, bs)
        dl_train = DataLoader(
            ds_train, batch_sampler=sampler, pin_memory=(device.type == "cuda")
        )
    else:
        dl_train = DataLoader(
            ds_train, batch_size=bs, shuffle=True, pin_memory=(device.type == "cuda")
        )

    if x_val is not None and y_val is not None:
        ds_val = TensorDataset(
            torch.tensor(x_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
        )
        dl_val = DataLoader(
            ds_val, batch_size=bs, shuffle=False, pin_memory=(device.type == "cuda")
        )
    else:
        dl_val = None

    # Model / optim / loss -------------------------------------------------- #
    model = _build_model(x_train.shape[1:], cfg).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.get("lr", 1e-3),
        weight_decay=cfg.get("weight_decay", 0.0),
    )

    # Training loop --------------------------------------------------------- #
    epochs = cfg.get("epochs", 10)
    for epoch in range(epochs):
        train_loss, train_acc = _run_epoch(
            model, dl_train, criterion, optimizer, device
        )
        if dl_val is not None:
            val_loss, val_acc = _run_epoch(model, dl_val, criterion, None, device)
            log.info(
                "E%02d  loss %.4f/%.4f  acc %.3f/%.3f",
                epoch,
                train_loss,
                val_loss,
                train_acc,
                val_acc,
            )
        else:
            log.info("E%02d  loss %.4f  acc %.3f", epoch, train_loss, train_acc)

    return model


def save_model(model: nn.Module, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    log.info("Saved model → %s", path)


def load_model(
    path: str | Path, input_shape: Tuple[int, ...], cfg: Mapping[str, Any] | None = None
) -> nn.Module:
    cfg = dict(cfg or {})
    model = _build_model(input_shape, cfg)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    log.info("Loaded model ← %s", path)
    return model
