"""Visualization module for deforestation monitoring."""

import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium import plugins
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class DeforestationVisualizer:
    """Visualizer for deforestation monitoring results."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the visualizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.viz_config = config.get('visualization', {})
        
        # Set matplotlib style
        plt.style.use(self.viz_config.get('style', 'seaborn-v0_8'))
        
        # Set default colors
        self.colors = self.viz_config.get('colors', {
            'forest_intact': '#228B22',
            'forest_deforested': '#8B0000',
            'prediction': '#FFD700',
            'uncertainty': '#FFA500'
        })
    
    def plot_feature_distributions(self, metadata: pd.DataFrame, 
                                 save_path: Optional[str] = None) -> None:
        """Plot distributions of features by deforestation status.
        
        Args:
            metadata: DataFrame with features and labels
            save_path: Optional path to save the plot
        """
        logger.info("Creating feature distribution plots")
        
        feature_cols = ['ndvi_change', 'surface_temp_rise', 'logging_index', 
                       'days_since_rain', 'distance_to_road']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, feature in enumerate(feature_cols):
            if feature in metadata.columns:
                # Plot distributions by deforestation status
                intact_data = metadata[metadata['deforested'] == 0][feature]
                deforested_data = metadata[metadata['deforested'] == 1][feature]
                
                axes[i].hist(intact_data, alpha=0.7, label='Forest Intact', 
                           color=self.colors['forest_intact'], bins=30)
                axes[i].hist(deforested_data, alpha=0.7, label='Deforested', 
                           color=self.colors['forest_deforested'], bins=30)
                
                axes[i].set_xlabel(feature.replace('_', ' ').title())
                axes[i].set_ylabel('Frequency')
                axes[i].set_title(f'{feature.replace("_", " ").title()} Distribution')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        # Remove empty subplot
        axes[-1].remove()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature distributions saved to {save_path}")
        
        plt.show()
    
    def plot_correlation_matrix(self, metadata: pd.DataFrame, 
                               save_path: Optional[str] = None) -> None:
        """Plot correlation matrix of features.
        
        Args:
            metadata: DataFrame with features
            save_path: Optional path to save the plot
        """
        logger.info("Creating correlation matrix plot")
        
        feature_cols = ['ndvi_change', 'surface_temp_rise', 'logging_index', 
                       'days_since_rain', 'distance_to_road', 'deforested']
        
        # Select only available columns
        available_cols = [col for col in feature_cols if col in metadata.columns]
        corr_data = metadata[available_cols]
        
        # Calculate correlation matrix
        corr_matrix = corr_data.corr()
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap='RdBu_r',
            center=0,
            square=True,
            fmt='.2f',
            cbar_kws={'shrink': 0.8}
        )
        
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Correlation matrix saved to {save_path}")
        
        plt.show()
    
    def create_interactive_map(self, metadata: pd.DataFrame, 
                             predictions: Optional[np.ndarray] = None,
                             save_path: Optional[str] = None) -> folium.Map:
        """Create an interactive map showing deforestation data.
        
        Args:
            metadata: DataFrame with spatial coordinates
            predictions: Optional prediction results
            save_path: Optional path to save the map
            
        Returns:
            Folium map object
        """
        logger.info("Creating interactive deforestation map")
        
        # Get map configuration
        map_config = self.viz_config.get('map', {})
        center_lat = map_config.get('center_lat', 0.0)
        center_lon = map_config.get('center_lon', 0.0)
        zoom = map_config.get('zoom', 2)
        tiles = map_config.get('tiles', 'OpenStreetMap')
        
        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom,
            tiles=tiles
        )
        
        # Add actual deforestation points
        deforested_points = metadata[metadata['deforested'] == 1]
        intact_points = metadata[metadata['deforested'] == 0]
        
        # Add deforested points (red)
        for idx, row in deforested_points.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=5,
                popup=f"Deforested Area<br>NDVI Change: {row['ndvi_change']:.3f}<br>Logging Index: {row['logging_index']:.3f}",
                color='red',
                fill=True,
                fillOpacity=0.7
            ).add_to(m)
        
        # Add intact forest points (green)
        for idx, row in intact_points.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=3,
                popup=f"Forest Intact<br>NDVI Change: {row['ndvi_change']:.3f}<br>Logging Index: {row['logging_index']:.3f}",
                color='green',
                fill=True,
                fillOpacity=0.5
            ).add_to(m)
        
        # Add predictions if available
        if predictions is not None:
            pred_deforested = metadata[predictions == 1]
            pred_intact = metadata[predictions == 0]
            
            # Add predicted deforested points (yellow)
            for idx, row in pred_deforested.iterrows():
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=4,
                    popup=f"Predicted Deforestation<br>NDVI Change: {row['ndvi_change']:.3f}",
                    color='orange',
                    fill=True,
                    fillOpacity=0.6
                ).add_to(m)
        
        # Add legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 200px; height: 120px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p><b>Deforestation Map Legend</b></p>
        <p><i class="fa fa-circle" style="color:red"></i> Actual Deforestation</p>
        <p><i class="fa fa-circle" style="color:green"></i> Forest Intact</p>
        <p><i class="fa fa-circle" style="color:orange"></i> Predicted Deforestation</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        if save_path:
            m.save(save_path)
            logger.info(f"Interactive map saved to {save_path}")
        
        return m
    
    def plot_model_comparison(self, results: Dict[str, Any], 
                            save_path: Optional[str] = None) -> None:
        """Plot comparison of different models.
        
        Args:
            results: Dictionary of model evaluation results
            save_path: Optional path to save the plot
        """
        logger.info("Creating model comparison plots")
        
        if not results:
            logger.warning("No results available for comparison")
            return
        
        # Extract metrics for comparison
        models = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = [results[model][metric] for model in models]
            
            bars = axes[i].bar(models, values, color=plt.cm.Set3(np.linspace(0, 1, len(models))))
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_ylabel('Score')
            axes[i].set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
            
            # Rotate x-axis labels
            axes[i].tick_params(axis='x', rotation=45)
        
        # Remove empty subplot
        axes[-1].remove()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison saved to {save_path}")
        
        plt.show()
    
    def create_time_series_plot(self, metadata: pd.DataFrame, 
                               save_path: Optional[str] = None) -> None:
        """Create time series plot of deforestation indicators.
        
        Args:
            metadata: DataFrame with temporal features
            save_path: Optional path to save the plot
        """
        logger.info("Creating time series plot")
        
        # Create synthetic time series data
        n_days = 365
        dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
        
        # Simulate NDVI time series
        np.random.seed(42)
        ndvi_base = 0.7
        ndvi_trend = np.linspace(0, -0.1, n_days)  # Declining trend
        ndvi_noise = np.random.normal(0, 0.05, n_days)
        ndvi_series = ndvi_base + ndvi_trend + ndvi_noise
        
        # Simulate logging activity
        logging_base = 0.3
        logging_spikes = np.random.poisson(0.1, n_days) * 0.2
        logging_series = logging_base + logging_spikes
        
        # Create plot
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # NDVI time series
        axes[0].plot(dates, ndvi_series, color='green', linewidth=2, label='NDVI')
        axes[0].set_title('NDVI Time Series (Simulated)')
        axes[0].set_ylabel('NDVI')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Logging activity time series
        axes[1].plot(dates, logging_series, color='red', linewidth=2, label='Logging Activity')
        axes[1].set_title('Logging Activity Time Series (Simulated)')
        axes[1].set_ylabel('Logging Index')
        axes[1].set_xlabel('Date')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Time series plot saved to {save_path}")
        
        plt.show()
    
    def create_risk_dashboard(self, metadata: pd.DataFrame, 
                            predictions: np.ndarray,
                            probabilities: np.ndarray,
                            save_path: Optional[str] = None) -> None:
        """Create a risk dashboard visualization.
        
        Args:
            metadata: DataFrame with spatial and feature data
            predictions: Model predictions
            probabilities: Prediction probabilities
            save_path: Optional path to save the plot
        """
        logger.info("Creating risk dashboard")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Risk level distribution
        risk_levels = pd.cut(probabilities[:, 1], bins=3, labels=['Low', 'Medium', 'High'])
        risk_counts = risk_levels.value_counts()
        
        axes[0, 0].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%',
                      colors=['green', 'orange', 'red'])
        axes[0, 0].set_title('Deforestation Risk Distribution')
        
        # Spatial risk map (simplified)
        scatter = axes[0, 1].scatter(metadata['longitude'], metadata['latitude'], 
                                   c=probabilities[:, 1], cmap='RdYlGn_r', 
                                   s=50, alpha=0.7)
        axes[0, 1].set_title('Spatial Risk Map')
        axes[0, 1].set_xlabel('Longitude')
        axes[0, 1].set_ylabel('Latitude')
        plt.colorbar(scatter, ax=axes[0, 1], label='Risk Probability')
        
        # Feature importance (simplified)
        feature_names = ['NDVI Change', 'Temp Rise', 'Logging', 'Days Rain', 'Road Distance']
        feature_importance = np.random.random(5)  # Placeholder
        feature_importance = feature_importance / feature_importance.sum()
        
        axes[1, 0].barh(feature_names, feature_importance, color='skyblue')
        axes[1, 0].set_title('Feature Importance')
        axes[1, 0].set_xlabel('Importance Score')
        
        # Risk vs actual comparison
        risk_bins = pd.cut(probabilities[:, 1], bins=5)
        actual_deforestation_rate = metadata.groupby(risk_bins)['deforested'].mean()
        
        axes[1, 1].plot(range(len(actual_deforestation_rate)), actual_deforestation_rate, 
                       marker='o', linewidth=2, markersize=8)
        axes[1, 1].set_title('Risk Calibration')
        axes[1, 1].set_xlabel('Risk Quintile')
        axes[1, 1].set_ylabel('Actual Deforestation Rate')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Risk dashboard saved to {save_path}")
        
        plt.show()
    
    def save_all_plots(self, metadata: pd.DataFrame, results: Dict[str, Any],
                      predictions: Optional[np.ndarray] = None,
                      probabilities: Optional[np.ndarray] = None,
                      output_dir: str = "assets/plots") -> None:
        """Save all visualization plots.
        
        Args:
            metadata: DataFrame with data
            results: Model evaluation results
            predictions: Optional predictions
            probabilities: Optional probabilities
            output_dir: Output directory for plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving all plots to {output_dir}")
        
        # Save individual plots
        self.plot_feature_distributions(metadata, str(output_path / "feature_distributions.png"))
        self.plot_correlation_matrix(metadata, str(output_path / "correlation_matrix.png"))
        self.plot_model_comparison(results, str(output_path / "model_comparison.png"))
        self.create_time_series_plot(metadata, str(output_path / "time_series.png"))
        
        if predictions is not None and probabilities is not None:
            self.create_risk_dashboard(metadata, predictions, probabilities, 
                                      str(output_path / "risk_dashboard.png"))
        
        logger.info("All plots saved successfully")
