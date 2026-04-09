"""Evaluation module for deforestation monitoring models."""

import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class DeforestationEvaluator:
    """Evaluator for deforestation monitoring models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the evaluator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.results = {}
        
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray, 
                      model_name: str, metadata_test: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Evaluate a single model comprehensively.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
            metadata_test: Optional test metadata
            
        Returns:
            Dictionary of evaluation results
        """
        logger.info(f"Evaluating {model_name}")
        
        # Basic predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        # Classification metrics
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba[:, 1]) if len(np.unique(y_test)) > 1 else 0.0,
            'pr_auc': average_precision_score(y_test, y_proba[:, 1]) if len(np.unique(y_test)) > 1 else 0.0
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Additional metrics
        tn, fp, fn, tp = cm.ravel()
        metrics.update({
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0.0,
            'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0.0
        })
        
        # Spatial metrics if metadata is available
        if metadata_test is not None:
            spatial_metrics = self._calculate_spatial_metrics(y_test, y_pred, metadata_test)
            metrics.update(spatial_metrics)
        
        # Store results
        self.results[model_name] = metrics
        
        return metrics
    
    def _calculate_spatial_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 metadata: pd.DataFrame) -> Dict[str, float]:
        """Calculate spatial-specific metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            metadata: Spatial metadata
            
        Returns:
            Dictionary of spatial metrics
        """
        spatial_metrics = {}
        
        # Regional performance
        if 'latitude' in metadata.columns:
            # Split by latitude regions
            lat_bins = pd.cut(metadata['latitude'], bins=3, labels=['South', 'Center', 'North'])
            
            for region in ['South', 'Center', 'North']:
                region_mask = lat_bins == region
                if np.sum(region_mask) > 0:
                    region_y_true = y_true[region_mask]
                    region_y_pred = y_pred[region_mask]
                    
                    if len(np.unique(region_y_true)) > 1:
                        spatial_metrics[f'accuracy_{region.lower()}'] = accuracy_score(region_y_true, region_y_pred)
                        spatial_metrics[f'f1_{region.lower()}'] = f1_score(region_y_true, region_y_pred, zero_division=0)
        
        # Distance-based analysis
        if 'distance_to_road' in metadata.columns:
            # Performance by distance to roads
            road_bins = pd.cut(metadata['distance_to_road'], bins=3, labels=['Close', 'Medium', 'Far'])
            
            for distance in ['Close', 'Medium', 'Far']:
                distance_mask = road_bins == distance
                if np.sum(distance_mask) > 0:
                    distance_y_true = y_true[distance_mask]
                    distance_y_pred = y_pred[distance_mask]
                    
                    if len(np.unique(distance_y_true)) > 1:
                        spatial_metrics[f'accuracy_{distance.lower()}_to_road'] = accuracy_score(distance_y_true, distance_y_pred)
        
        return spatial_metrics
    
    def cross_validate_model(self, model, X: np.ndarray, y: np.ndarray, 
                           cv_folds: int = 5) -> Dict[str, float]:
        """Perform cross-validation on a model.
        
        Args:
            model: Model to evaluate
            X: Features
            y: Labels
            cv_folds: Number of CV folds
            
        Returns:
            Dictionary of CV results
        """
        logger.info(f"Performing {cv_folds}-fold cross-validation")
        
        # Use StratifiedKFold for balanced splits
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Cross-validation scores
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
        
        cv_results = {
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'cv_scores': cv_scores.tolist()
        }
        
        return cv_results
    
    def create_leaderboard(self) -> pd.DataFrame:
        """Create a model leaderboard.
        
        Returns:
            DataFrame with model rankings
        """
        if not self.results:
            logger.warning("No results available for leaderboard")
            return pd.DataFrame()
        
        # Extract key metrics for leaderboard
        leaderboard_data = []
        
        for model_name, metrics in self.results.items():
            leaderboard_data.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1'],
                'ROC-AUC': metrics['roc_auc'],
                'PR-AUC': metrics['pr_auc']
            })
        
        leaderboard = pd.DataFrame(leaderboard_data)
        
        # Sort by F1-score (primary) and ROC-AUC (secondary)
        leaderboard = leaderboard.sort_values(['F1-Score', 'ROC-AUC'], ascending=False)
        leaderboard['Rank'] = range(1, len(leaderboard) + 1)
        
        return leaderboard
    
    def plot_confusion_matrices(self, save_path: Optional[str] = None) -> None:
        """Plot confusion matrices for all models.
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.results:
            logger.warning("No results available for plotting")
            return
        
        n_models = len(self.results)
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, metrics) in enumerate(self.results.items()):
            cm = np.array(metrics['confusion_matrix'])
            
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                ax=axes[i],
                xticklabels=['Forest Intact', 'Deforested'],
                yticklabels=['Forest Intact', 'Deforested']
            )
            
            axes[i].set_title(f'{model_name}\nAccuracy: {metrics["accuracy"]:.3f}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrices saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curves(self, save_path: Optional[str] = None) -> None:
        """Plot ROC curves for all models.
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.results:
            logger.warning("No results available for plotting")
            return
        
        plt.figure(figsize=(8, 6))
        
        for model_name, metrics in self.results.items():
            # Note: This is a simplified version. In practice, you'd need to store
            # the actual probabilities and true labels to plot proper ROC curves
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            plt.text(0.5, 0.5, f'{model_name}: AUC = {metrics["roc_auc"]:.3f}', 
                    transform=plt.gca().transAxes, fontsize=10)
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curves saved to {save_path}")
        
        plt.show()
    
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """Generate a comprehensive evaluation report.
        
        Args:
            save_path: Optional path to save the report
            
        Returns:
            Report string
        """
        if not self.results:
            return "No evaluation results available."
        
        report_lines = [
            "# Deforestation Monitoring Model Evaluation Report",
            "",
            "## Model Performance Summary",
            ""
        ]
        
        # Leaderboard
        leaderboard = self.create_leaderboard()
        report_lines.append("### Model Leaderboard")
        report_lines.append(leaderboard.to_string(index=False))
        report_lines.append("")
        
        # Detailed results for each model
        for model_name, metrics in self.results.items():
            report_lines.extend([
                f"## {model_name} Detailed Results",
                "",
                f"- **Accuracy**: {metrics['accuracy']:.4f}",
                f"- **Precision**: {metrics['precision']:.4f}",
                f"- **Recall**: {metrics['recall']:.4f}",
                f"- **F1-Score**: {metrics['f1']:.4f}",
                f"- **ROC-AUC**: {metrics['roc_auc']:.4f}",
                f"- **PR-AUC**: {metrics['pr_auc']:.4f}",
                "",
                f"- **True Positives**: {metrics['true_positives']}",
                f"- **False Positives**: {metrics['false_positives']}",
                f"- **True Negatives**: {metrics['true_negatives']}",
                f"- **False Negatives**: {metrics['false_negatives']}",
                "",
                f"- **Specificity**: {metrics['specificity']:.4f}",
                f"- **Sensitivity**: {metrics['sensitivity']:.4f}",
                f"- **False Positive Rate**: {metrics['false_positive_rate']:.4f}",
                f"- **False Negative Rate**: {metrics['false_negative_rate']:.4f}",
                ""
            ])
            
            # Spatial metrics if available
            spatial_keys = [k for k in metrics.keys() if any(x in k for x in ['accuracy_', 'f1_'])]
            if spatial_keys:
                report_lines.append("### Spatial Analysis")
                for key in spatial_keys:
                    report_lines.append(f"- **{key.replace('_', ' ').title()}**: {metrics[key]:.4f}")
                report_lines.append("")
        
        report = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Evaluation report saved to {save_path}")
        
        return report
    
    def save_results(self, filepath: str) -> None:
        """Save evaluation results to JSON file.
        
        Args:
            filepath: Path to save results
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {filepath}")
    
    def load_results(self, filepath: str) -> None:
        """Load evaluation results from JSON file.
        
        Args:
            filepath: Path to load results from
        """
        with open(filepath, 'r') as f:
            self.results = json.load(f)
        
        logger.info(f"Evaluation results loaded from {filepath}")
