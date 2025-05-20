from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict

import joblib
import yaml

from data.data_loader import load_dataset, split_dataset
from evaluate.evaluate import evaluate, predict_proba
from preprocessing.preprocess import preprocess_data, normalize_data
from trainers.ml_trainer import train_ml
from trainers.nn_trainer import (
    train_nn,
    save_model as save_torch,
    load_model as load_torch,
)


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def _save_ml_models(models: Dict[str, object], out_dir: Path) -> None:
    for name, model in models.items():
        if name == "metrics":
            continue
        if hasattr(model, "predict_proba") or name == "lightgbm":
            joblib.dump(model, out_dir / f"{name}.joblib")


def _load_ml_models(out_dir: Path) -> Dict[str, object]:
    models = {}
    for p in out_dir.glob("*.joblib"):
        models[p.stem] = joblib.load(p)
    return models


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("OCD‑EEG pipeline")
    ap.add_argument("data_dir", type=Path, help="root folder with class subfolders")
    ap.add_argument("--config", type=Path, default="config.yaml")
    ap.add_argument("--output", type=Path, default=Path("output"))
    ap.add_argument("--mode", choices=["train", "evaluate", "predict"], default="train")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    _setup_logging()
    log = logging.getLogger("cli")

    cfg = yaml.safe_load(Path(args.config).read_text()) if args.config.exists() else {}
    args.output.mkdir(parents=True, exist_ok=True)

    if args.mode == "train":
        raw_data = load_dataset(args.data_dir, mode="train")
        train_d, val_d, test_d = split_dataset(raw_data, **cfg.get("split", {}))

        X_tr, y_tr = preprocess_data(train_d, cfg.get("preprocess", {}), mode="train")[
            :2
        ]
        X_val, y_val = preprocess_data(val_d, cfg.get("preprocess", {}), mode="test")[
            :2
        ]

        X_tr, X_val = normalize_data(X_tr, X_val)

        nn_cfg = cfg.get("nn", {})
        nn_model = train_nn(X_tr, y_tr, X_val, y_val, cfg=nn_cfg)
        save_torch(nn_model, args.output / "nn_model.pt")

        ml_cfg = cfg.get("ml", {})
        ml_models = train_ml(X_tr, y_tr, X_val, y_val, cfg=ml_cfg)
        _save_ml_models(ml_models, args.output)
        with open(args.output / "val_metrics.json", "w") as f:
            json.dump(ml_models.get("metrics", {}), f, indent=2)

        log.info("Training complete; models saved → %s", args.output)

    elif args.mode == "evaluate":
        raw_test = load_dataset(args.data_dir, mode="test")
        _, _, test_d = split_dataset(raw_test, **cfg.get("split", {}))
        X_test, y_test = preprocess_data(
            test_d, cfg.get("preprocess", {}), mode="test"
        )[:2]
        (X_test,) = normalize_data(X_test)

        nn_model = load_torch(args.output / "nn_model.pt", input_shape=X_test.shape[1:])
        ml_models = _load_ml_models(args.output)
        models = {"nn": nn_model, **ml_models}

        results = {}
        for name, m in models.items():
            results[name] = evaluate(m, X_test, y_test)
        with open(args.output / "eval.json", "w") as f:
            json.dump(results, f, indent=2)
        log.info("Evaluation saved → eval.json")

    else:
        raw_pred = load_dataset(args.data_dir, mode="predict")
        X = preprocess_data(raw_pred, cfg.get("preprocess", {}), mode="predict")
        (X,) = normalize_data(X)

        nn_model = load_torch(args.output / "nn_model.pt", input_shape=X.shape[1:])
        probs = predict_proba(nn_model, X)
        out_csv = args.output / "predictions.csv"
        with open(out_csv, "w") as f:
            f.write("subject_id,prob\n")
            for sid, pr in zip(raw_pred["subject_ids"], probs):
                f.write(f"{sid},{pr:.6f}\n")
        log.info("Predictions → %s", out_csv)
