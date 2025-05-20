import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import mne
import numpy as np
from sklearn.preprocessing import StandardScaler

from ocd_classification.utils.utils import split_by_subjects

log = logging.getLogger(__name__)

def preprocess_data(
    data: Dict[str, Any],
    cfg: Dict[str, Any],
    mode: str = "train",
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],  # train
    Tuple[np.ndarray, np.ndarray, np.ndarray],  # test
    np.ndarray,  # predict
]:
    """Preprocess raw recordings → epoch‑wise PSD features.

    Parameters
    ----------
    data : dict
        Must contain keys ``eeg_data`` (List[mne.io.Raw]), ``labels``, ``subject_ids``.
    cfg : dict
        Preprocessing options (see README or docstring of `_process_single`).
    mode : {"train", "test", "predict"}

    Returns
    -------
    Depending on *mode*.
    """

    raws: List[mne.io.BaseRaw] = data["eeg_data"]
    subj_ids: List[str] = data["subject_ids"]
    labels_all: Optional[np.ndarray] = np.asarray(data.get("labels", []), dtype=np.int64) if mode != "predict" else None

    X_list: List[np.ndarray] = []
    subj_epoch_ids: List[str] = []

    for raw, sid in zip(raws, subj_ids):
        try:
            feats = _process_single(raw, cfg)  # (n_epochs, n_features)
            if feats is None:
                continue
            X_list.append(feats)
            subj_epoch_ids.extend([sid] * feats.shape[0])
        except Exception as exc:  # pragma: no cover
            log.error("%s skipped: %s", sid, exc)

    if not X_list:
        raise RuntimeError("No epochs after preprocessing — check input files / cfg")

    X = np.vstack(X_list).astype(np.float32)
    subj_epoch_ids = np.asarray(subj_epoch_ids)

    # ------- predict mode ------------------------------------------------- #
    if mode == "predict":
        return X

    # Align labels with retained epochs
    labels_epoch: List[int] = []
    ptr = 0
    for arr, lab in zip(X_list, data["labels"]):
        labels_epoch.extend([lab] * arr.shape[0])
        ptr += 1
    y = np.asarray(labels_epoch, dtype=np.int64)

    # ------- test mode ---------------------------------------------------- #
    if mode == "test":
        return X, y, subj_epoch_ids

    # ------- train/val split (unique subjects) ---------------------------- #
    val_size = cfg.get("val_size", 0.2)
    train_m, val_m, _ = split_by_subjects(
        subj_epoch_ids,
        test_size=0.0,
        val_size=val_size,
        random_state=cfg.get("random_state", 42),
    )

    return (
        X[train_m], X[val_m],
        y[train_m], y[val_m],
        subj_epoch_ids[train_m], subj_epoch_ids[val_m],
    )


# --------------------------------------------------------------------------- #
# Single‑record helper                                                        #
# --------------------------------------------------------------------------- #

def _process_single(raw: mne.io.BaseRaw, cfg: Dict[str, Any]) -> Optional[np.ndarray]:
    """Return array (n_epochs, n_features) of PSD power values or *None* if no epochs."""

    raw = raw.copy()

    # -- drop channels ---------------------------------------------------- #
    to_drop = [ch for ch in cfg.get("drop_channels", ["FT9", "empty"]) if ch in raw.ch_names]
    to_drop += [ch for ch in raw.ch_names if ch.strip() == ""]
    if to_drop:
        raw.drop_channels(to_drop)
        log.debug("Dropped %s", to_drop)

    # -- epoching --------------------------------------------------------- #
    events, _ = mne.events_from_annotations(raw)
    event_ids = cfg.get("event_ids", {"Stimulus/S 10": 10, "Stimulus/S 20": 20, "Stimulus/S 30": 30})
    if not events.size:
        log.warning("No events found in %s", raw.filenames[0] if raw.filenames else "<raw>")
        return None

    tmin, tmax = cfg.get("tmin", -0.2), cfg.get("tmax", 0.8)
    epochs = mne.Epochs(raw, events, event_id=event_ids, tmin=tmin, tmax=tmax,
                        baseline=None, preload=True, verbose=False)
    if len(epochs) == 0:
        return None

    # -- multitaper power ------------------------------------------------- #
    fmin, fmax = cfg.get("fmin", 1.0), cfg.get("fmax", 40.0)
    psds, freqs = mne.time_frequency.psd_multitaper(
        epochs, fmin=fmin, fmax=fmax,
        verbose=False, n_jobs=cfg.get("n_jobs", 1),
    )  # shape (n_epochs, n_channels, n_freqs)

    if cfg.get("log_power", True):
        psds = 10 * np.log10(psds + np.finfo(np.float32).eps)

    # flatten channel × freq
    feats = psds.reshape(psds.shape[0], -1)
    return feats


# --------------------------------------------------------------------------- #
# Normalisation                                                               #
# --------------------------------------------------------------------------- #

def normalize_data(
    X_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    X_test: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, ...]:
    scaler = StandardScaler()

    def _trans(arr: np.ndarray, fit: bool) -> np.ndarray:
        orig = arr.shape
        arr2d = arr.reshape(arr.shape[0], -1)
        arr2d = scaler.fit_transform(arr2d) if fit else scaler.transform(arr2d)
        return arr2d.reshape(orig)

    out: List[np.ndarray] = [_trans(X_train, True)]
    if X_val is not None:
        out.append(_trans(X_val, False))
    if X_test is not None:
        out.append(_trans(X_test, False))
    return tuple(out)

