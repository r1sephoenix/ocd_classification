from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union

import mne
import numpy as np
from sklearn.model_selection import train_test_split

from ocd_classification.utils.utils import split_by_subjects

log = logging.getLogger(__name__)


def load_dataset(
    root: Union[str, Path],
    *,
    mode: str = "train",  # {"train", "test", "predict"}
    class_map: Dict[str, int] | None = None,
    ext: str = "*.vhdr",
) -> Dict[str, object]:
    """Load BrainVision files arranged in class subfolders.

    If the mode is "predict," labels will be None.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)

    class_map = class_map or {"control": 0, "ocd": 1}

    raws: List[mne.io.BaseRaw] = []
    subj_ids: List[str] = []
    labels: List[int] | None = [] if mode != "predict" else None

    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        cls_name = class_dir.name.lower()
        if cls_name not in class_map:
            log.warning("Skip folder %s — unknown class", class_dir.name)
            continue
        label = class_map[cls_name]
        for vhdr in sorted(class_dir.glob(ext)):
            sid = vhdr.stem
            try:
                raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose=False)
            except Exception as exc:  # pragma: no cover
                log.error("Skip %s (%s)", vhdr.name, exc)
                continue
            raws.append(raw)
            subj_ids.append(sid)
            if labels is not None:
                labels.append(label)

    if not raws:
        raise RuntimeError("No BrainVision files found under given subfolders")

    meta = {
        "n_subjects": len(subj_ids),
        "n_channels": len(raws[0].ch_names),
        "sfreq": raws[0].info["sfreq"],
    }
    return {
        "eeg_data": raws,
        "labels": labels,
        "subject_ids": subj_ids,
        "metadata": meta,
    }


def _mask_from_subjects(all_ids: np.ndarray, subset: np.ndarray) -> np.ndarray:
    return np.isin(all_ids, subset)


def _ensure_each_class(
    train_m: np.ndarray, val_m: np.ndarray, test_m: np.ndarray, labels: np.ndarray
):
    """Move one subject if the val or test lacks a class present in train."""
    classes = np.unique(labels)
    for cls in classes:
        for src, dst in ((train_m, val_m), (train_m, test_m)):
            if np.any(labels[dst] == cls) or not np.any(labels[src] == cls):
                continue
            idx = np.where(src & (labels == cls))[0][0]
            src[idx] = False
            dst[idx] = True
    return train_m, val_m, test_m


def split_dataset(
    data: Dict[str, object],
    *,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[Dict[str, object], Dict[str, object], Dict[str, object]]:
    """Stratified subject-level split.

    * If `n_subjects` ≤ 4 → fallback 2‑1‑1 (train/val/test).
    * Else → два вызова `train_test_split` c `stratify` на уровне субъектов.
    """
    subj_ids = np.asarray(data["subject_ids"], dtype=str)
    labels_arr = np.asarray(
        data["labels"] if data["labels"] is not None else [0] * len(subj_ids)
    )

    unique_subj, first_idx = np.unique(subj_ids, return_index=True)
    subj_labels = labels_arr[first_idx]
    n_subj = len(unique_subj)
    rng_state = random_state

    if n_subj <= 4:
        # deterministic small split -------------------------------------------------
        rng = np.random.RandomState(rng_state)
        perm = rng.permutation(unique_subj)
        train_subj = perm[: max(2, n_subj - 2)]
        remain = perm[len(train_subj) :]
        val_subj = remain[:1] if len(remain) > 1 else []
        test_subj = remain[1:] if len(remain) > 1 else remain
    else:
        # stratified subject-level split -------------------------------------------
        try:
            train_val_subj, test_subj, y_train_val, y_test = train_test_split(
                unique_subj,
                subj_labels,
                test_size=test_size,
                random_state=rng_state,
                stratify=subj_labels,
            )
            val_ratio = val_size / (1 - test_size)
            train_subj, val_subj, *_ = train_test_split(
                train_val_subj,
                y_train_val,
                test_size=val_ratio,
                random_state=rng_state,
                stratify=y_train_val,
            )
        except ValueError:
            # Fallback to group split w/out stratifying
            train_m, val_m, test_m = split_by_subjects(
                subj_ids,
                test_size=test_size,
                val_size=val_size,
                random_state=rng_state,
            )
            return _assemble(data, subj_ids, labels_arr, train_m, val_m, test_m)

    train_m = _mask_from_subjects(subj_ids, np.asarray(train_subj))
    val_m = _mask_from_subjects(subj_ids, np.asarray(val_subj))
    test_m = _mask_from_subjects(subj_ids, np.asarray(test_subj))

    train_m, val_m, test_m = _ensure_each_class(train_m, val_m, test_m, labels_arr)

    return _assemble(data, subj_ids, labels_arr, train_m, val_m, test_m)


def _assemble(
    data: Dict[str, object],
    subj_ids: np.ndarray,
    labels_arr: np.ndarray,
    train_m: np.ndarray,
    val_m: np.ndarray,
    test_m: np.ndarray,
):
    def subset(mask: np.ndarray) -> Dict[str, object]:
        idx = np.where(mask)[0]
        return {
            "eeg_data": [data["eeg_data"][i] for i in idx],
            "labels": (
                [int(labels_arr[i]) for i in idx]
                if data["labels"] is not None
                else None
            ),
            "subject_ids": subj_ids[mask].tolist(),
            "metadata": {**data["metadata"], "n_subjects": int(mask.sum())},
        }

    return subset(train_m), subset(val_m), subset(test_m)
