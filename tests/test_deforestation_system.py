"""Tests for deforestation monitoring system."""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.processor import DeforestationDataProcessor
from models.deforestation_models import XGBoostModel, LightGBMModel, NeuralNetworkModel, SpatialMLModel
from eval.evaluator import DeforestationEvaluator
from viz.visualizer import DeforestationVisualizer


class TestDataProcessor:
    """Test data processing functionality."""
    
    def test_processor_initialization(self):
        """Test processor initialization."""
        config = {
            'data': {
                'n_samples': 100,
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
            }
        }
        
        processor = DeforestationDataProcessor()
        processor.config = config
        
        assert processor.config == config
        assert processor.feature_names == [
            "ndvi_change", "surface_temp_rise", "logging_index", 
            "days_since_rain", "distance_to_road"
        ]
    
    def test_synthetic_data_generation(self):
        """Test synthetic data generation."""
        config = {
            'data': {
                'n_samples': 100,
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
            }
        }
        
        processor = DeforestationDataProcessor()
        processor.config = config
        
        X, y, metadata = processor.generate_synthetic_data(50)
        
        assert X.shape == (50, 5)
        assert y.shape == (50,)
        assert len(metadata) == 50
        assert 'latitude' in metadata.columns
        assert 'longitude' in metadata.columns
        assert 'deforested' in metadata.columns


class TestModels:
    """Test model functionality."""
    
    def test_xgboost_model(self):
        """Test XGBoost model."""
        config = {
            'xgboost': {
                'n_estimators': 10,
                'max_depth': 3,
                'learning_rate': 0.1
            }
        }
        
        model = XGBoostModel(config)
        
        # Generate test data
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = np.random.randint(0, 2, 100)
        
        # Train model
        train_metrics = model.train(X_train, y_train)
        
        assert model.is_trained
        assert 'train_accuracy' in train_metrics
        
        # Test predictions
        X_test = np.random.randn(20, 5)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        
        assert len(predictions) == 20
        assert probabilities.shape == (20, 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)
    
    def test_lightgbm_model(self):
        """Test LightGBM model."""
        config = {
            'lightgbm': {
                'n_estimators': 10,
                'max_depth': 3,
                'learning_rate': 0.1
            }
        }
        
        model = LightGBMModel(config)
        
        # Generate test data
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = np.random.randint(0, 2, 100)
        
        # Train model
        train_metrics = model.train(X_train, y_train)
        
        assert model.is_trained
        assert 'train_accuracy' in train_metrics
        
        # Test predictions
        X_test = np.random.randn(20, 5)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        
        assert len(predictions) == 20
        assert probabilities.shape == (20, 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)
    
    def test_neural_network_model(self):
        """Test neural network model."""
        config = {
            'neural_network': {
                'hidden_layers': [32, 16],
                'epochs': 5,
                'batch_size': 32,
                'learning_rate': 0.001
            }
        }
        
        model = NeuralNetworkModel(config)
        
        # Generate test data
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = np.random.randint(0, 2, 100)
        
        # Train model
        train_metrics = model.train(X_train, y_train)
        
        assert model.is_trained
        assert 'train_accuracy' in train_metrics
        
        # Test predictions
        X_test = np.random.randn(20, 5)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        
        assert len(predictions) == 20
        assert probabilities.shape == (20, 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)
    
    def test_spatial_ml_model(self):
        """Test spatial ML model."""
        config = {}
        
        model = SpatialMLModel(config)
        
        # Generate test data
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = np.random.randint(0, 2, 100)
        
        # Train model
        train_metrics = model.train(X_train, y_train)
        
        assert model.is_trained
        assert 'train_accuracy' in train_metrics
        
        # Test predictions
        X_test = np.random.randn(20, 5)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        
        assert len(predictions) == 20
        assert probabilities.shape == (20, 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)


class TestEvaluator:
    """Test evaluation functionality."""
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        config = {'evaluation': {'test_size': 0.2}}
        
        evaluator = DeforestationEvaluator(config)
        
        assert evaluator.config == config
        assert evaluator.results == {}
    
    def test_model_evaluation(self):
        """Test model evaluation."""
        config = {'evaluation': {'test_size': 0.2}}
        
        evaluator = DeforestationEvaluator(config)
        
        # Create mock model
        class MockModel:
            def predict(self, X):
                return np.random.randint(0, 2, len(X))
            
            def predict_proba(self, X):
                proba = np.random.rand(len(X), 2)
                return proba / proba.sum(axis=1, keepdims=True)
        
        # Generate test data
        np.random.seed(42)
        X_test = np.random.randn(50, 5)
        y_test = np.random.randint(0, 2, 50)
        
        model = MockModel()
        results = evaluator.evaluate_model(model, X_test, y_test, "TestModel")
        
        assert 'accuracy' in results
        assert 'precision' in results
        assert 'recall' in results
        assert 'f1' in results
        assert 'roc_auc' in results
        assert results['model_name'] == "TestModel"
    
    def test_leaderboard_creation(self):
        """Test leaderboard creation."""
        config = {'evaluation': {'test_size': 0.2}}
        
        evaluator = DeforestationEvaluator(config)
        
        # Add mock results
        evaluator.results = {
            'Model1': {'accuracy': 0.8, 'precision': 0.75, 'recall': 0.8, 'f1': 0.77, 'roc_auc': 0.85, 'pr_auc': 0.8},
            'Model2': {'accuracy': 0.85, 'precision': 0.8, 'recall': 0.85, 'f1': 0.82, 'roc_auc': 0.9, 'pr_auc': 0.85}
        }
        
        leaderboard = evaluator.create_leaderboard()
        
        assert len(leaderboard) == 2
        assert 'Model' in leaderboard.columns
        assert 'Accuracy' in leaderboard.columns
        assert 'F1-Score' in leaderboard.columns
        assert leaderboard.iloc[0]['Model'] == 'Model2'  # Should be ranked first


class TestVisualizer:
    """Test visualization functionality."""
    
    def test_visualizer_initialization(self):
        """Test visualizer initialization."""
        config = {
            'visualization': {
                'figure_size': [12, 8],
                'colors': {
                    'forest_intact': '#228B22',
                    'forest_deforested': '#8B0000'
                }
            }
        }
        
        visualizer = DeforestationVisualizer(config)
        
        assert visualizer.config == config
        assert visualizer.colors['forest_intact'] == '#228B22'
        assert visualizer.colors['forest_deforested'] == '#8B0000'
    
    def test_feature_distribution_plot(self):
        """Test feature distribution plotting."""
        config = {
            'visualization': {
                'colors': {
                    'forest_intact': '#228B22',
                    'forest_deforested': '#8B0000'
                }
            }
        }
        
        visualizer = DeforestationVisualizer(config)
        
        # Generate test metadata
        np.random.seed(42)
        metadata = pd.DataFrame({
            'ndvi_change': np.random.randn(100),
            'surface_temp_rise': np.random.randn(100),
            'logging_index': np.random.randn(100),
            'days_since_rain': np.random.randn(100),
            'distance_to_road': np.random.randn(100),
            'deforested': np.random.randint(0, 2, 100)
        })
        
        # This should not raise an exception
        visualizer.plot_feature_distributions(metadata)


if __name__ == "__main__":
    pytest.main([__file__])
