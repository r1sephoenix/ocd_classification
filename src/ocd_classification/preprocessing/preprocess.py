import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import mne
import numpy as np
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


def preprocess_data(
    data: Dict[str, Any],
    cfg: Dict[str, Any],
    mode: str = "train",  # train | test | predict
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray],  # (X, y, subj_ids) – train/test
    np.ndarray,  # X – predict
]:
    """Convert Raw → epoch‑wise PSD features (no splitting).

    * Each epoch becomes one feature vector (channels × freqs).
    * Labels/subject_ids are replicated per‑epoch.
    """

    raws: List[mne.io.BaseRaw] = data["eeg_data"]
    subj_ids: List[str] = data["subject_ids"]

    X_list: List[np.ndarray] = []
    subj_epoch_ids: List[str] = []

    for raw, sid in zip(raws, subj_ids):
        feats = _process_single(raw, cfg)
        if feats is None:
            continue
        X_list.append(feats)
        subj_epoch_ids.extend([sid] * feats.shape[0])

    if not X_list:
        raise RuntimeError("No epochs after preprocessing — check config / events")

    X = np.vstack(X_list).astype(np.float32)
    subj_epoch_ids_arr = np.asarray(subj_epoch_ids, dtype=str)

    if mode == "predict":
        return X

    # replicate labels per epoch
    y_epoch: List[int] = []
    for feats_arr, label in zip(X_list, data["labels"]):
        y_epoch.extend([label] * feats_arr.shape[0])
    y = np.asarray(y_epoch, dtype=np.int64)

    return X, y, subj_epoch_ids_arr


def _process_single(raw: mne.io.BaseRaw, cfg: Dict[str, Any]) -> Optional[np.ndarray]:
    """Return an array (n_epochs, n_ch, n_freqs) of PSD power values or None if no epochs."""

    raw = raw.copy()

    to_drop = [
        ch for ch in cfg.get("drop_channels", ["FT9", "empty"]) if ch in raw.ch_names
    ]
    to_drop += [ch for ch in raw.ch_names if ch.strip() == ""]
    if to_drop:
        raw.drop_channels(to_drop)
        log.debug("Dropped %s", to_drop)

    events, _ = mne.events_from_annotations(raw)
    event_ids = cfg.get(
        "event_ids", {"Stimulus/S 10": 10, "Stimulus/S 20": 20, "Stimulus/S 30": 30}
    )
    if not events.size:
        log.warning(
            "No events found in %s", raw.filenames[0] if raw.filenames else "<raw>"
        )
        return None

    tmin, tmax = cfg.get("tmin", -0.2), cfg.get("tmax", 0.8)
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_ids,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose=False,
    )
    if len(epochs) == 0:
        return None

    fmin, fmax = cfg.get("fmin", 1.0), cfg.get("fmax", 40.0)
    psd = epochs.compute_psd(
        method="multitaper",
        fmin=fmin,
        fmax=fmax,
        n_jobs=cfg.get("n_jobs", 1),
    )

    psds = psd.get_data()  # array (epochs, channels, freqs)
    if cfg.get("log_power", True):
        psds = 10 * np.log10(psds + float(np.finfo(np.float32).eps))

    feats = psds.astype(np.float32)
    return feats  # shape (epochs, n_ch, n_freqs)


def normalize_data(
    x_train: np.ndarray,
    x_val: Optional[np.ndarray] = None,
    x_test: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, ...]:
    scaler = StandardScaler()

    def _trans(arr: np.ndarray, fit: bool) -> np.ndarray:
        orig = arr.shape
        arr2d = arr.reshape(arr.shape[0], -1)
        arr2d = scaler.fit_transform(arr2d) if fit else scaler.transform(arr2d)
        return arr2d.reshape(orig)

    out: List[np.ndarray] = [_trans(x_train, True)]
    if x_val is not None:
        out.append(_trans(x_val, False))
    if x_test is not None:
        out.append(_trans(x_test, False))
    return tuple(out)
