#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main module for OCD classification from EEG data.
This module serves as the entry point for the application.
"""

import argparse
import logging
import sys
from pathlib import Path

from ocd_classification.data import data_loader
from ocd_classification.preprocessing import preprocess
from ocd_classification.models import train, evaluate
from ocd_classification.utils import config


def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="OCD classification from EEG data")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml", 
        help="Path to configuration file"
    )
    parser.add_argument(
        "--data_dir", 
        type=str, 
        help="Directory containing EEG data files"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./output", 
        help="Directory to save results"
    )
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["train", "evaluate", "predict"], 
        default="train", 
        help="Operation mode"
    )
    return parser.parse_args()


def main():
    """Main function to run the OCD classification pipeline."""
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)

    # Parse arguments
    args = parse_arguments()

    # Load configuration
    cfg = config.load_config(args.config)

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting OCD classification pipeline")

    # Execute based on mode
    if args.mode == "train":
        # Load and preprocess data
        logger.info("Loading and preprocessing data")
        data = data_loader.load_data(args.data_dir or cfg.data_dir)

        # Check if we should use subject-based splitting
        use_subject_split = cfg.model.get("use_subject_split", True) if hasattr(cfg, "model") and isinstance(cfg.model, dict) else True

        if use_subject_split:
            # Train model with subject-based splitting
            logger.info("Training model with subject-based splitting")
            model_config = cfg.model if hasattr(cfg, "model") else {}
            model = train.train_model_with_subject_split(data, model_config)
        else:
            # Preprocess data and train model with standard splitting
            preprocess_config = cfg.preprocessing if hasattr(cfg, "preprocessing") else {}
            X_train, X_val, y_train, y_val = preprocess.preprocess_data(data, preprocess_config)
            logger.info("Training model with standard splitting")
            model_config = cfg.model if hasattr(cfg, "model") else {}
            model = train.train_model(X_train, y_train, X_val, y_val, model_config)

        # Save models
        if isinstance(model, dict):
            # Save each model in the dictionary
            for model_name, model_obj in model.items():
                if model_name != "metrics":  # Skip metrics entry
                    if hasattr(model_obj, 'state_dict'):  # PyTorch model
                        model_path = output_dir / f"{model_name}_model.h5"
                        train.save_model(model_obj, model_path)
                    else:  # scikit-learn or LightGBM model
                        import joblib
                        model_path = output_dir / f"{model_name}_model.joblib"
                        joblib.dump(model_obj, model_path)
                    logger.info(f"{model_name} model saved to {model_path}")
        else:
            # Save a single model (backward compatibility)
            model_path = output_dir / "model.h5"
            train.save_model(model, model_path)
            logger.info(f"Model saved to {model_path}")

    elif args.mode == "evaluate":
        # Load data
        logger.info("Loading test data")
        data = data_loader.load_data(args.data_dir or cfg.data_dir, mode="test")

        # Load models
        model_config = cfg.model if hasattr(cfg, "model") else {}

        # Check if we have multiple models
        neural_network_path = output_dir / "neural_network_model.h5"
        if neural_network_path.exists():
            # Load multiple models
            import joblib
            import glob

            models = {}

            # Load PyTorch neural network model if it exists
            if neural_network_path.exists():
                models["neural_network"] = train.load_model(neural_network_path, model_config)
                logger.info(f"Neural network model loaded from {neural_network_path}")

            # Load scikit-learn and LightGBM models
            for model_path in glob.glob(str(output_dir / "*_model.joblib")):
                model_name = Path(model_path).stem.replace("_model", "")
                models[model_name] = joblib.load(model_path)
                logger.info(f"{model_name} model loaded from {model_path}")

            model = models
        else:
            # Load single model (backward compatibility)
            model_path = output_dir / "model.h5"
            model = train.load_model(model_path, model_config)
            logger.info(f"Model loaded from {model_path}")

        # Check if we should use subject-based splitting
        use_subject_split = cfg.model.get("use_subject_split", True) if hasattr(cfg, "model") and isinstance(cfg.model, dict) else True
        model_config = cfg.model if hasattr(cfg, "model") else {}

        if use_subject_split:
            # Evaluate model with subject-based splitting
            logger.info("Evaluating model with subject-based splitting")
            metrics = evaluate.evaluate_model_with_subject_split(model, data, model_config)
        else:
            # Preprocess data and evaluate model with standard splitting
            preprocess_config = cfg.preprocessing if hasattr(cfg, "preprocessing") else {}
            X_test, y_test = preprocess.preprocess_data(data, preprocess_config, mode="test")
            logger.info("Evaluating model with standard splitting")

            # Check if we have multiple models
            if isinstance(model, dict):
                # Evaluate each model
                logger.info("Evaluating multiple models")
                all_metrics = {}

                for model_name, model_obj in model.items():
                    if model_name != "metrics":  # Skip metrics entry
                        logger.info(f"Evaluating {model_name} model")
                        model_metrics = evaluate.evaluate_model(model_obj, X_test, y_test, model_config)
                        all_metrics[model_name] = model_metrics

                # Combine metrics from all models
                metrics = {
                    "models": all_metrics,
                    # Add summary metrics (e.g., best model by accuracy)
                    "best_model": max(all_metrics.items(), key=lambda x: x[1]["accuracy"])[0]
                }
            else:
                # Evaluate single model
                metrics = evaluate.evaluate_model(model, X_test, y_test, model_config)

        # Save evaluation results
        results_path = output_dir / "evaluation_results.json"
        evaluate.save_results(metrics, results_path)
        logger.info(f"Evaluation results saved to {results_path}")

        # If we have multiple models, log the best model
        if isinstance(metrics, dict) and "best_model" in metrics:
            logger.info(f"Best model by accuracy: {metrics['best_model']}")

    elif args.mode == "predict":
        # Load data
        logger.info("Loading data for prediction")
        data = data_loader.load_data(args.data_dir or cfg.data_dir, mode="predict")

        # Preprocess data
        preprocess_config = cfg.preprocessing if hasattr(cfg, "preprocessing") else {}
        X = preprocess.preprocess_data(data, preprocess_config, mode="predict")

        # Load models
        model_config = cfg.model if hasattr(cfg, "model") else {}

        # Check if we have multiple models
        neural_network_path = output_dir / "neural_network_model.h5"
        if neural_network_path.exists():
            # Load multiple models
            import joblib
            import glob

            models = {}

            # Load PyTorch neural network model if it exists
            if neural_network_path.exists():
                models["neural_network"] = train.load_model(neural_network_path, model_config)
                logger.info(f"Neural network model loaded from {neural_network_path}")

            # Load scikit-learn and LightGBM models
            for model_path in glob.glob(str(output_dir / "*_model.joblib")):
                model_name = Path(model_path).stem.replace("_model", "")
                models[model_name] = joblib.load(model_path)
                logger.info(f"{model_name} model loaded from {model_path}")

            model = models
        else:
            # Load single model (backward compatibility)
            model_path = output_dir / "model.h5"
            model = train.load_model(model_path, model_config)
            logger.info(f"Model loaded from {model_path}")

        # Make predictions
        logger.info("Making predictions")

        # If we have multiple models, use the one specified in config or default to neural_network
        if isinstance(model, dict):
            model_to_use = model_config.get("predict_model", "neural_network")
            if model_to_use in model:
                logger.info(f"Using {model_to_use} model for predictions")
                predictions = evaluate.predict(model[model_to_use], X, model_config)
            else:
                # Use the first available model
                model_to_use = next(iter(model))
                logger.info(f"Specified model not found, using {model_to_use} model for predictions")
                predictions = evaluate.predict(model[model_to_use], X, model_config)
        else:
            # Single model
            predictions = evaluate.predict(model, X, model_config)

        # Save predictions with subject IDs if available
        predictions_path = output_dir / "predictions.csv"
        if "subject_ids" in data:
            evaluate.save_predictions(predictions, predictions_path, data["subject_ids"])
        else:
            evaluate.save_predictions(predictions, predictions_path)
        logger.info(f"Predictions saved to {predictions_path}")

    logger.info("OCD classification pipeline completed successfully")


if __name__ == "__main__":
    main()
