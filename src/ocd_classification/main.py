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
from ocd_classification.models.cnn_model import EEGCNN, EEG3DCNN
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

        # Save model
        model_path = output_dir / "model.h5"
        train.save_model(model, model_path)
        logger.info(f"Model saved to {model_path}")

    elif args.mode == "evaluate":
        # Load data
        logger.info("Loading test data")
        data = data_loader.load_data(args.data_dir or cfg.data_dir, mode="test")

        # Load model
        model_path = output_dir / "model.h5"
        model_config = cfg.model if hasattr(cfg, "model") else {}
        model = train.load_model(model_path, model_config)

        # Check if we should use subject-based splitting
        use_subject_split = cfg.model.get("use_subject_split", True) if hasattr(cfg, "model") and isinstance(cfg.model, dict) else True

        if use_subject_split:
            # Evaluate model with subject-based splitting
            logger.info("Evaluating model with subject-based splitting")
            model_config = cfg.model if hasattr(cfg, "model") else {}
            metrics = evaluate.evaluate_model_with_subject_split(model, data, model_config)
        else:
            # Preprocess data and evaluate model with standard splitting
            preprocess_config = cfg.preprocessing if hasattr(cfg, "preprocessing") else {}
            X_test, y_test = preprocess.preprocess_data(data, preprocess_config, mode="test")
            logger.info("Evaluating model with standard splitting")
            model_config = cfg.model if hasattr(cfg, "model") else {}
            metrics = evaluate.evaluate_model(model, X_test, y_test, model_config)

        # Save evaluation results
        results_path = output_dir / "evaluation_results.json"
        evaluate.save_results(metrics, results_path)
        logger.info(f"Evaluation results saved to {results_path}")

    elif args.mode == "predict":
        # Load data
        logger.info("Loading data for prediction")
        data = data_loader.load_data(args.data_dir or cfg.data_dir, mode="predict")

        # Preprocess data
        preprocess_config = cfg.preprocessing if hasattr(cfg, "preprocessing") else {}
        X = preprocess.preprocess_data(data, preprocess_config, mode="predict")

        # Load model
        model_path = output_dir / "model.h5"
        model_config = cfg.model if hasattr(cfg, "model") else {}
        model = train.load_model(model_path, model_config)

        # Make predictions
        logger.info("Making predictions")
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
