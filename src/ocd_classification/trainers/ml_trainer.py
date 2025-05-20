from __future__ import annotations

import logging
from typing import Any, Dict, Mapping

import lightgbm as lgb
import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import (
    StratifiedKFold,
)
from sklearn.svm import SVC

from ocd_classification.utils.utils import flatten_data

log = logging.getLogger(__name__)

_GRID_LOGREG = {
    "C": np.logspace(-3, 2, 6),
}
_GRID_RF = {
    "n_estimators": [100, 200, 400],
    "max_depth": [None, 5, 10, 20],
}
_GRID_SVM = {
    "C": np.logspace(-2, 2, 5),
    "kernel": ["rbf", "linear"],
    "gamma": ["scale", "auto"],
}
_GRID_LGB = {
    "n_estimators": [100, 300],
    "learning_rate": [0.05, 0.1],
    "max_depth": [-1, 5, 10],
}


def _tune_lgbm_optuna(
    x: np.ndarray,
    y: np.ndarray,
    x_val: np.ndarray | None,
    y_val: np.ndarray | None,
    n_trials: int,
    timeout: int | None,
    random_state: int,
) -> "lgb.LGBMClassifier":
    if optuna is None:
        raise ImportError("optuna is required for LightGBM tuning")

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", -1, 16),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": random_state,
        }
        clf = lgb.LGBMClassifier(**params)  # type: ignore[arg-type]
        clf.fit(
            x,
            y,
            eval_set=[(x_val, y_val)] if x_val is not None else None,
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(30)] if x_val is not None else None,  # type: ignore[arg-type]
        )
        if x_val is not None and y_val is not None:
            preds = clf.predict(x_val)
            return f1_score(y_val, preds, zero_division=0)
        # CV fallback
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
        scores = []
        for tr_idx, vl_idx in cv.split(x, y):
            clf.fit(x[tr_idx], y[tr_idx])
            scores.append(f1_score(y[vl_idx], clf.predict(x[vl_idx]), zero_division=0))
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state)
    )
    study.optimize(
        objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False
    )
    log.info("[Optuna] best LGBM f1: %.4f", study.best_value)
    best_params = study.best_params | {"random_state": random_state}
    return lgb.LGBMClassifier(**best_params).fit(x, y)


def train_ml(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    cfg: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Trains 'classic' ML models and (optionally) validates them.

    Parameters
    ----------
    x_train, y_train : ndarray
        Training features / labels
    x_val, y_val : ndarray, optional
        Validation split for computing metrics
    cfg : dict, optional
        Hyper-parameters, e.g. ``cfg["rf_n_estimators"]``.

    Returns
    -------
    models : dict
        ``{"logreg": model, "random_forest": model, ..., "metrics": {...}}``
    """
    cfg = dict(cfg or {})  # copy to avoid side‑effects

    X_tr = flatten_data(x_train)
    X_va = flatten_data(x_val) if x_val is not None else None

    models: Dict[str, Any] = {
        "logreg": LogisticRegression(
            C=cfg.get("lr_C", 1.0),
            max_iter=cfg.get("lr_max_iter", 1000),
            random_state=cfg.get("random_state", 42),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=cfg.get("rf_n_estimators", 100),
            max_depth=cfg.get("rf_max_depth"),
            random_state=cfg.get("random_state", 42),
        ),
        "svm": SVC(
            C=cfg.get("svm_C", 1.0),
            kernel=cfg.get("svm_kernel", "rbf"),
            probability=True,
            random_state=cfg.get("random_state", 42),
        ),
    }

    if lgb is not None:
        models["lightgbm"] = lgb.LGBMClassifier(
            n_estimators=cfg.get("lgb_n_estimators", 100),
            learning_rate=cfg.get("lgb_learning_rate", 0.1),
            max_depth=cfg.get("lgb_max_depth", -1),
            random_state=cfg.get("random_state", 42),
        )
    else:
        log.warning("LightGBM not installed – skipping lgb model")

    for name, model in models.items():
        log.info("Training %s", name)
        model.fit(X_tr, y_train)

    if X_va is not None and y_val is not None:
        metrics: Dict[str, Dict[str, float]] = {}
        for name, model in models.items():
            acc = float(np.mean(model.predict(X_va) == y_val))
            metrics[name] = {"accuracy": acc}
            log.info("%s val acc: %.4f", name, acc)
        models["metrics"] = metrics

    log.info("ML training complete")
    return models
