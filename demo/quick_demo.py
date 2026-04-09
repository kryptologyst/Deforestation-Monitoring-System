#!/usr/bin/env python3
"""Quick demo script for deforestation monitoring system."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data.processor import DeforestationDataProcessor
from models.deforestation_models import XGBoostModel, LightGBMModel
from eval.evaluator import DeforestationEvaluator


def quick_demo():
    """Run a quick demonstration of the deforestation monitoring system."""
    print("🌲 Deforestation Monitoring System - Quick Demo")
    print("=" * 60)
    
    # Load configuration
    config = {
        'data': {
            'n_samples': 200,
            'random_seed': 42,
            'features': {
                'ndvi_change': {'mean': -0.1, 'std': 0.2},
                'surface_temp_rise': {'mean': 1.5, 'std': 0.5},
                'logging_index': {'mean': 0.4, 'std': 0.3},
                'days_since_rain': {'mean': 15, 'std': 5},
                'distance_to_road': {'mean': 5, 'std': 2}
            },
            'deforestation_thresholds': {
                'ndvi_change': -0.2,
                'logging_index': 0.5,
                'days_since_rain': 10,
                'distance_to_road': 6
            }
        },
        'evaluation': {
            'test_size': 0.2,
            'random_seed': 42
        },
        'xgboost': {
            'n_estimators': 50,
            'max_depth': 4,
            'learning_rate': 0.1
        },
        'lightgbm': {
            'n_estimators': 50,
            'max_depth': 4,
            'learning_rate': 0.1
        }
    }
    
    print("1. Generating synthetic deforestation data...")
    
    # Initialize data processor
    processor = DeforestationDataProcessor()
    processor.config = config
    
    # Generate synthetic data
    X, y, metadata = processor.generate_synthetic_data()
    
    print(f"   Generated {len(metadata)} samples")
    print(f"   Deforestation rate: {metadata['deforested'].mean()*100:.1f}%")
    
    # Add spatial features
    metadata = processor.add_spatial_features(metadata)
    
    # Prepare train-test split
    X_train, X_test, y_train, y_test, metadata_train, metadata_test = processor.prepare_train_test_split(
        X, y, metadata
    )
    
    # Scale features
    X_train_scaled, X_test_scaled = processor.scale_features(X_train, X_test)
    
    print(f"   Train set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")
    
    print("\n2. Training models...")
    
    # Train XGBoost model
    print("   Training XGBoost model...")
    xgb_model = XGBoostModel(config)
    xgb_train_metrics = xgb_model.train(X_train_scaled, y_train)
    
    # Train LightGBM model
    print("   Training LightGBM model...")
    lgb_model = LightGBMModel(config)
    lgb_train_metrics = lgb_model.train(X_train_scaled, y_train)
    
    print("\n3. Evaluating models...")
    
    # Initialize evaluator
    evaluator = DeforestationEvaluator(config)
    
    # Evaluate XGBoost
    xgb_results = evaluator.evaluate_model(
        xgb_model, X_test_scaled, y_test, "XGBoost", metadata_test
    )
    
    # Evaluate LightGBM
    lgb_results = evaluator.evaluate_model(
        lgb_model, X_test_scaled, y_test, "LightGBM", metadata_test
    )
    
    print("\n4. Results Summary:")
    print("-" * 40)
    
    # Create results table
    results_data = [
        ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
        ["XGBoost", f"{xgb_results['accuracy']:.3f}", f"{xgb_results['precision']:.3f}", 
         f"{xgb_results['recall']:.3f}", f"{xgb_results['f1']:.3f}", f"{xgb_results['roc_auc']:.3f}"],
        ["LightGBM", f"{lgb_results['accuracy']:.3f}", f"{lgb_results['precision']:.3f}", 
         f"{lgb_results['recall']:.3f}", f"{lgb_results['f1']:.3f}", f"{lgb_results['roc_auc']:.3f}"]
    ]
    
    # Print formatted table
    for row in results_data:
        print(f"{row[0]:<12} {row[1]:<10} {row[2]:<10} {row[3]:<10} {row[4]:<10} {row[5]:<10}")
    
    print("\n5. Sample Predictions:")
    print("-" * 40)
    
    # Get sample predictions
    sample_indices = np.random.choice(len(X_test), 5, replace=False)
    sample_X = X_test_scaled[sample_indices]
    sample_y = y_test[sample_indices]
    sample_metadata = metadata_test.iloc[sample_indices]
    
    xgb_preds = xgb_model.predict(sample_X)
    xgb_proba = xgb_model.predict_proba(sample_X)
    
    for i, idx in enumerate(sample_indices):
        actual = "Deforested" if sample_y[i] == 1 else "Forest Intact"
        predicted = "Deforested" if xgb_preds[i] == 1 else "Forest Intact"
        confidence = xgb_proba[i, 1]
        
        print(f"Sample {i+1}:")
        print(f"  Location: ({sample_metadata.iloc[i]['latitude']:.2f}, {sample_metadata.iloc[i]['longitude']:.2f})")
        print(f"  NDVI Change: {sample_metadata.iloc[i]['ndvi_change']:.3f}")
        print(f"  Actual: {actual}")
        print(f"  Predicted: {predicted} (confidence: {confidence:.3f})")
        print()
    
    print("6. Feature Analysis:")
    print("-" * 40)
    
    # Feature statistics by deforestation status
    intact_stats = metadata[metadata['deforested'] == 0][['ndvi_change', 'logging_index', 'days_since_rain']].mean()
    deforested_stats = metadata[metadata['deforested'] == 1][['ndvi_change', 'logging_index', 'days_since_rain']].mean()
    
    print("Average values by deforestation status:")
    print(f"{'Feature':<15} {'Forest Intact':<15} {'Deforested':<15}")
    print("-" * 45)
    print(f"{'NDVI Change':<15} {intact_stats['ndvi_change']:<15.3f} {deforested_stats['ndvi_change']:<15.3f}")
    print(f"{'Logging Index':<15} {intact_stats['logging_index']:<15.3f} {deforested_stats['logging_index']:<15.3f}")
    print(f"{'Days Since Rain':<15} {intact_stats['days_since_rain']:<15.1f} {deforested_stats['days_since_rain']:<15.1f}")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("For interactive visualization, run: streamlit run demo/streamlit_app.py")
    print("Author: kryptologyst - https://github.com/kryptologyst")


if __name__ == "__main__":
    quick_demo()
