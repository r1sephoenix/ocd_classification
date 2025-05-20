import argparse
import logging
import sys
from pathlib import Path
from typing import Dict

import joblib
import yaml

from data.data_loader import load_dataset, split_dataset
from preprocessing.preprocess import preprocess_data, normalize_data
from trainers.nn_trainer import (
    train_nn,
    save_model as save_torch,
    load_model as load_torch,
)
from trainers.ml_trainer import train_ml
from evaluate.evaluate import evaluate, predict_proba, save_results


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
        joblib.dump(model, out_dir / f"{name}.joblib")


def _load_ml_models(out_dir: Path) -> Dict[str, object]:
    return {p.stem: joblib.load(p) for p in out_dir.glob("*.joblib")}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("OCD‑EEG pipeline")
    ap.add_argument("data_dir", type=Path, help="root folder with class subfolders")
    ap.add_argument("--config", type=Path, default=Path("config.yaml"))
    ap.add_argument("--output", type=Path, default=Path("output"))
    ap.add_argument("--mode", choices=["train", "evaluate", "predict"], default="train")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    _setup_logging()
    log = logging.getLogger("cli")

    cfg = yaml.safe_load(args.config.read_text()) if args.config.exists() else {}
    args.output.mkdir(parents=True, exist_ok=True)

    if args.mode == "train":
        raw = load_dataset(args.data_dir, mode="train")
        train_d, val_d, _ = split_dataset(raw, **cfg.get("split", {}))

        X_tr, y_tr, _ = preprocess_data(
            train_d, cfg.get("preprocess", {}), mode="train"
        )
        X_val, y_val, _ = preprocess_data(val_d, cfg.get("preprocess", {}), mode="test")
        X_tr, X_val = normalize_data(X_tr, X_val)

        # NN
        nn_model = train_nn(X_tr, y_tr, X_val, y_val, cfg=cfg.get("nn", {}))
        save_torch(nn_model, args.output / "nn_model.pt")

        # ML
        ml_models = train_ml(X_tr, y_tr, X_val, y_val, cfg=cfg.get("ml", {}))
        _save_ml_models(ml_models, args.output)
        save_results(ml_models.get("metrics", {}), args.output / "val_metrics.json")
        log.info("Training complete; models saved → %s", args.output)

    elif args.mode == "evaluate":
        raw = load_dataset(args.data_dir, mode="test")
        _, _, test_d = split_dataset(raw, **cfg.get("split", {}))
        X_test, y_test, _ = preprocess_data(
            test_d, cfg.get("preprocess", {}), mode="test"
        )
        (X_test,) = normalize_data(X_test)

        nn_model = load_torch(args.output / "nn_model.pt", input_shape=X_test.shape[1:])
        models = {"nn": nn_model, **_load_ml_models(args.output)}

        results = {name: evaluate(m, X_test, y_test) for name, m in models.items()}
        save_results(results, args.output / "eval.json")
        log.info("Evaluation saved → eval.json")

    else:  # predict
        raw = load_dataset(args.data_dir, mode="predict")
        X = preprocess_data(raw, cfg.get("preprocess", {}), mode="predict")
        (X,) = normalize_data(X)

        nn_model = load_torch(args.output / "nn_model.pt", input_shape=X.shape[1:])
        probs = predict_proba(nn_model, X)

        out_csv = args.output / "predictions.csv"
        out_csv.write_text(
            "subject_id,prob\n"
            + "\n".join(f"{sid},{pr:.6f}" for sid, pr in zip(raw["subject_ids"], probs))
        )
        log.info("Predictions → %s", out_csv)


if __name__ == "__main__":
    main()
