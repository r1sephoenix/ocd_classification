from __future__ import annotations

from typing import Any, Dict, List, Sequence, Optional
from pathlib import Path

import numpy as np
import json
import sklearn.metrics as skm
import torch
from torch.utils.data import DataLoader, TensorDataset


def _is_torch(model: Any) -> bool:
    return isinstance(model, torch.nn.Module)


def predict_proba(
    model: Any,
    x: np.ndarray,
    *,
    batch_size: int = 64,
) -> np.ndarray:
    """Return positive‑class probabilities."""

    if _is_torch(model):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()

        loader = DataLoader(
            TensorDataset(torch.tensor(x, dtype=torch.float32)), batch_size=batch_size
        )
        probs: List[float] = []
        with torch.no_grad():
            for (xx,) in loader:
                xx = xx.to(device)
                out = (
                    model(xx).float().sigmoid()
                    if model.training is False
                    else model(xx)
                )
                probs.extend(out.squeeze().cpu().numpy())
        return np.asarray(probs, dtype=np.float32)

    # scikit‑learn like interface
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x.reshape(x.shape[0], -1))[:, 1].astype(np.float32)
    preds = model.predict(x.reshape(x.shape[0], -1)).astype(np.float32)
    return preds  # best effort


def _aggregate_by_subject(
    probs: Sequence[float],
    labels: Sequence[int],
    subject_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Average probabilities per subject and derive hard label by majority."""

    import pandas as pd

    df = pd.DataFrame({"prob": probs, "label": labels, "subj": subject_ids})
    grouped = df.groupby("subj").mean()  # mean prob and mean label
    return grouped["prob"].values, (grouped["label"].values > 0.5).astype(int)


def evaluate(
    model: Any,
    x: np.ndarray,
    y: np.ndarray,
    *,
    subject_ids: Sequence[str] | None = None,
    group_by_subject: bool = True,
    batch_size: int = 64,
) -> Dict[str, Any]:
    """Compute standard metrics. Works for both torch and sklearn models."""

    probs = predict_proba(model, x, batch_size=batch_size)
    labels = y.astype(int)

    if group_by_subject and subject_ids is not None:
        probs, labels = _aggregate_by_subject(probs, labels, subject_ids)

    preds = (probs > 0.5).astype(int)

    metrics: Dict[str, Any] = {
        "accuracy": skm.accuracy_score(labels, preds),
        "precision": skm.precision_score(labels, preds, zero_division=0),
        "recall": skm.recall_score(labels, preds, zero_division=0),
        "f1": skm.f1_score(labels, preds, zero_division=0),
        "roc_auc": skm.roc_auc_score(labels, probs)
        if len(np.unique(labels)) > 1
        else 0.5,
    }
    cm = skm.confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    metrics.update(
        {
            "confusion_matrix": cm,
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        }
    )
    return metrics


def _to_builtin(obj: Any):
    """Convert NumPy scalars/arrays to builtin types for JSON."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"{obj!r} is not JSON‑serialisable")


def save_results(metrics: Dict[str, Any], path: Path, **json_kw) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, default=_to_builtin, indent=2, **json_kw)


def save_predictions(
    preds: np.ndarray, path: Path, subject_ids: Optional[Sequence[str]] = None
):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "subject_id": subject_ids
            if subject_ids is not None
            else np.arange(len(preds)),
            "prob": preds,
        }
    )
    df.to_csv(path, index=False)
