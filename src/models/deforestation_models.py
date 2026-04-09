"""Model implementations for deforestation monitoring."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for deforestation monitoring models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the model with configuration.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.model = None
        self.is_trained = False
        
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary of training metrics
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted labels
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted probabilities
        """
        pass
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate the model.
        
        Args:
            X: Test features
            y: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
            
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y, y_proba[:, 1]) if len(np.unique(y)) > 1 else 0.0
        }
        
        return metrics
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
            
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model.
        
        Args:
            filepath: Path to the saved model
        """
        self.model = joblib.load(filepath)
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")


class XGBoostModel(BaseModel):
    """XGBoost model for deforestation monitoring."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize XGBoost model.
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__(config)
        self.xgb_config = config.get('xgboost', {})
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Train XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary of training metrics
        """
        logger.info("Training XGBoost model")
        
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        # Set up parameters
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': self.xgb_config.get('max_depth', 6),
            'learning_rate': self.xgb_config.get('learning_rate', 0.1),
            'subsample': self.xgb_config.get('subsample', 0.8),
            'colsample_bytree': self.xgb_config.get('colsample_bytree', 0.8),
            'random_state': 42
        }
        
        # Train model
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            self.model = xgb.train(
                params, 
                dtrain, 
                num_boost_round=self.xgb_config.get('n_estimators', 100),
                evals=[(dtrain, 'train'), (dval, 'val')],
                early_stopping_rounds=10,
                verbose_eval=False
            )
        else:
            self.model = xgb.train(
                params, 
                dtrain, 
                num_boost_round=self.xgb_config.get('n_estimators', 100),
                verbose_eval=False
            )
        
        self.is_trained = True
        
        # Calculate training metrics
        train_pred = self.predict_proba(X_train)
        train_metrics = {
            'train_accuracy': accuracy_score(y_train, (train_pred[:, 1] > 0.5).astype(int)),
            'train_roc_auc': roc_auc_score(y_train, train_pred[:, 1]) if len(np.unique(y_train)) > 1 else 0.0
        }
        
        return train_metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
            
        dtest = xgb.DMatrix(X)
        proba = self.model.predict(dtest)
        return (proba > 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
            
        dtest = xgb.DMatrix(X)
        proba = self.model.predict(dtest)
        return np.column_stack([1 - proba, proba])


class LightGBMModel(BaseModel):
    """LightGBM model for deforestation monitoring."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize LightGBM model.
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__(config)
        self.lgb_config = config.get('lightgbm', {})
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Train LightGBM model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary of training metrics
        """
        logger.info("Training LightGBM model")
        
        # Set up parameters
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'max_depth': self.lgb_config.get('max_depth', 6),
            'learning_rate': self.lgb_config.get('learning_rate', 0.1),
            'subsample': self.lgb_config.get('subsample', 0.8),
            'colsample_bytree': self.lgb_config.get('colsample_bytree', 0.8),
            'random_state': 42,
            'verbose': -1
        }
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            self.model = lgb.train(
                params,
                train_data,
                num_boost_round=self.lgb_config.get('n_estimators', 100),
                valid_sets=[train_data, val_data],
                valid_names=['train', 'val'],
                callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
            )
        else:
            self.model = lgb.train(
                params,
                train_data,
                num_boost_round=self.lgb_config.get('n_estimators', 100),
                callbacks=[lgb.log_evaluation(0)]
            )
        
        self.is_trained = True
        
        # Calculate training metrics
        train_pred = self.predict_proba(X_train)
        train_metrics = {
            'train_accuracy': accuracy_score(y_train, (train_pred[:, 1] > 0.5).astype(int)),
            'train_roc_auc': roc_auc_score(y_train, train_pred[:, 1]) if len(np.unique(y_train)) > 1 else 0.0
        }
        
        return train_metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
            
        proba = self.model.predict(X)
        return (proba > 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
            
        proba = self.model.predict(X)
        return np.column_stack([1 - proba, proba])


class NeuralNetworkModel(BaseModel):
    """Neural network model for deforestation monitoring."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize neural network model.
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__(config)
        self.nn_config = config.get('neural_network', {})
        self.device = self._get_device()
        
    def _get_device(self) -> torch.device:
        """Get the best available device.
        
        Returns:
            PyTorch device
        """
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    def _create_model(self, input_size: int) -> nn.Module:
        """Create neural network model.
        
        Args:
            input_size: Number of input features
            
        Returns:
            PyTorch model
        """
        hidden_layers = self.nn_config.get('hidden_layers', [64, 32])
        dropout = self.nn_config.get('dropout', 0.2)
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        return nn.Sequential(*layers)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Train neural network model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary of training metrics
        """
        logger.info(f"Training neural network model on {self.device}")
        
        # Create model
        self.model = self._create_model(X_train.shape[1]).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.nn_config.get('batch_size', 32),
            shuffle=True
        )
        
        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val),
                torch.FloatTensor(y_val)
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.nn_config.get('batch_size', 32),
                shuffle=False
            )
        
        # Set up optimizer and loss
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.nn_config.get('learning_rate', 0.001)
        )
        criterion = nn.BCELoss()
        
        # Training loop
        epochs = self.nn_config.get('epochs', 50)
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                        outputs = self.model(batch_X).squeeze()
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                val_losses.append(val_loss)
                
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            else:
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")
        
        self.is_trained = True
        
        # Calculate training metrics
        train_pred = self.predict_proba(X_train)
        train_metrics = {
            'train_accuracy': accuracy_score(y_train, (train_pred[:, 1] > 0.5).astype(int)),
            'train_roc_auc': roc_auc_score(y_train, train_pred[:, 1]) if len(np.unique(y_train)) > 1 else 0.0,
            'final_train_loss': train_losses[-1]
        }
        
        if val_losses:
            train_metrics['final_val_loss'] = val_losses[-1]
        
        return train_metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
            
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
            
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            proba = self.model(X_tensor).cpu().numpy()
        
        return np.column_stack([1 - proba, proba])


class SpatialMLModel(BaseModel):
    """Spatial machine learning model with spatial features."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize spatial ML model.
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__(config)
        self.base_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Train spatial ML model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary of training metrics
        """
        logger.info("Training spatial ML model")
        
        self.model = self.base_model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Calculate training metrics
        train_pred = self.predict_proba(X_train)
        train_metrics = {
            'train_accuracy': accuracy_score(y_train, (train_pred[:, 1] > 0.5).astype(int)),
            'train_roc_auc': roc_auc_score(y_train, train_pred[:, 1]) if len(np.unique(y_train)) > 1 else 0.0
        }
        
        return train_metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
            
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
            
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores.
        
        Returns:
            Feature importance array
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
            
        return self.model.feature_importances_
