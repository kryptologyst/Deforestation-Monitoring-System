#!/usr/bin/env python3
"""Main training script for deforestation monitoring models."""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import yaml
import torch
import random
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data.processor import DeforestationDataProcessor
from models.deforestation_models import (
    XGBoostModel, LightGBMModel, NeuralNetworkModel, SpatialMLModel
)
from eval.evaluator import DeforestationEvaluator
from viz.visualizer import DeforestationVisualizer


def setup_logging(config: Dict[str, Any]) -> None:
    """Set up logging configuration.
    
    Args:
        config: Configuration dictionary
    """
    log_config = config.get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    
    logging.basicConfig(
        level=log_level,
        format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=[
            logging.FileHandler(log_config.get('file', 'logs/training.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )


def set_deterministic_seeds(seed: int = 42) -> None:
    """Set deterministic seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_models(config: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Train all configured models.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (trained_models, evaluation_results)
    """
    logger = logging.getLogger(__name__)
    
    # Initialize data processor
    processor = DeforestationDataProcessor(config)
    
    # Generate synthetic data
    logger.info("Generating synthetic deforestation data")
    X, y, metadata = processor.generate_synthetic_data()
    
    # Add spatial features
    metadata = processor.add_spatial_features(metadata)
    
    # Prepare train-test split
    X_train, X_test, y_train, y_test, metadata_train, metadata_test = processor.prepare_train_test_split(
        X, y, metadata
    )
    
    # Scale features
    X_train_scaled, X_test_scaled = processor.scale_features(X_train, X_test)
    
    # Save processed data
    processor.save_data(X, y, metadata)
    
    # Initialize models
    models = {}
    model_configs = config.get('model', {})
    
    if 'xgboost' in model_configs.get('models', []):
        models['XGBoost'] = XGBoostModel(model_configs)
    
    if 'lightgbm' in model_configs.get('models', []):
        models['LightGBM'] = LightGBMModel(model_configs)
    
    if 'neural_network' in model_configs.get('models', []):
        models['Neural Network'] = NeuralNetworkModel(model_configs)
    
    if 'spatial_ml' in model_configs.get('models', []):
        models['Spatial ML'] = SpatialMLModel(model_configs)
    
    # Train models
    trained_models = {}
    for name, model in models.items():
        logger.info(f"Training {name} model")
        
        try:
            # Train model
            train_metrics = model.train(X_train_scaled, y_train)
            trained_models[name] = model
            
            logger.info(f"{name} training completed. Metrics: {train_metrics}")
            
        except Exception as e:
            logger.error(f"Error training {name}: {str(e)}")
            continue
    
    # Evaluate models
    evaluator = DeforestationEvaluator(config)
    evaluation_results = {}
    
    for name, model in trained_models.items():
        logger.info(f"Evaluating {name} model")
        
        try:
            results = evaluator.evaluate_model(
                model, X_test_scaled, y_test, name, metadata_test
            )
            evaluation_results[name] = results
            
            logger.info(f"{name} evaluation completed. Accuracy: {results['accuracy']:.4f}")
            
        except Exception as e:
            logger.error(f"Error evaluating {name}: {str(e)}")
            continue
    
    return trained_models, evaluation_results


def create_visualizations(config: Dict[str, Any], metadata: pd.DataFrame, 
                         evaluation_results: Dict[str, Any],
                         trained_models: Dict[str, Any]) -> None:
    """Create and save visualizations.
    
    Args:
        config: Configuration dictionary
        metadata: Metadata DataFrame
        evaluation_results: Model evaluation results
        trained_models: Trained models
    """
    logger = logging.getLogger(__name__)
    
    # Initialize visualizer
    visualizer = DeforestationVisualizer(config)
    
    # Create visualizations
    logger.info("Creating visualizations")
    
    # Load test data for predictions
    processor = DeforestationDataProcessor(config)
    X, y, metadata_full = processor.load_data()
    
    # Get test split
    X_train, X_test, y_train, y_test, metadata_train, metadata_test = processor.prepare_train_test_split(
        X, y, metadata_full
    )
    X_train_scaled, X_test_scaled = processor.scale_features(X_train, X_test)
    
    # Get predictions from best model
    best_model_name = max(evaluation_results.keys(), 
                         key=lambda x: evaluation_results[x]['f1'])
    best_model = trained_models[best_model_name]
    
    predictions = best_model.predict(X_test_scaled)
    probabilities = best_model.predict_proba(X_test_scaled)
    
    # Create and save all plots
    visualizer.save_all_plots(
        metadata_test, evaluation_results, predictions, probabilities
    )
    
    # Create interactive map
    interactive_map = visualizer.create_interactive_map(
        metadata_test, predictions, "assets/maps/deforestation_map.html"
    )
    
    logger.info("Visualizations created successfully")


def generate_reports(config: Dict[str, Any], evaluation_results: Dict[str, Any]) -> None:
    """Generate evaluation reports.
    
    Args:
        config: Configuration dictionary
        evaluation_results: Model evaluation results
    """
    logger = logging.getLogger(__name__)
    
    # Initialize evaluator
    evaluator = DeforestationEvaluator(config)
    evaluator.results = evaluation_results
    
    # Generate reports
    logger.info("Generating evaluation reports")
    
    # Create leaderboard
    leaderboard = evaluator.create_leaderboard()
    leaderboard.to_csv("assets/leaderboard.csv", index=False)
    logger.info("Leaderboard saved to assets/leaderboard.csv")
    
    # Generate comprehensive report
    report = evaluator.generate_report("assets/evaluation_report.md")
    logger.info("Evaluation report saved to assets/evaluation_report.md")
    
    # Save results as JSON
    evaluator.save_results("assets/evaluation_results.json")
    logger.info("Evaluation results saved to assets/evaluation_results.json")
    
    # Print summary
    print("\n" + "="*60)
    print("DEFORESTATION MONITORING MODEL EVALUATION SUMMARY")
    print("="*60)
    print(leaderboard.to_string(index=False))
    print("="*60)


def main():
    """Main training pipeline."""
    # Load configuration
    config = load_config()
    
    # Set up logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    # Set deterministic seeds
    set_deterministic_seeds(config.get('data', {}).get('random_seed', 42))
    
    logger.info("Starting deforestation monitoring model training")
    
    try:
        # Train models
        trained_models, evaluation_results = train_models(config)
        
        if not trained_models:
            logger.error("No models were successfully trained")
            return
        
        # Create visualizations
        processor = DeforestationDataProcessor(config)
        _, _, metadata = processor.load_data()
        create_visualizations(config, metadata, evaluation_results, trained_models)
        
        # Generate reports
        generate_reports(config, evaluation_results)
        
        logger.info("Training pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
