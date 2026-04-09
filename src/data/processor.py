"""Data processing module for deforestation monitoring."""

import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import geopandas as gpd
from shapely.geometry import Point
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


class DeforestationDataProcessor:
    """Process deforestation monitoring data including satellite imagery and spatial features."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize the data processor with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.scaler = StandardScaler()
        self.feature_names = [
            "ndvi_change",
            "surface_temp_rise", 
            "logging_index",
            "days_since_rain",
            "distance_to_road"
        ]
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def generate_synthetic_data(self, n_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Generate synthetic deforestation monitoring data.
        
        Args:
            n_samples: Number of samples to generate. If None, uses config value.
            
        Returns:
            Tuple of (features, labels, metadata)
        """
        if n_samples is None:
            n_samples = self.config['data']['n_samples']
            
        # Set random seed for reproducibility
        np.random.seed(self.config['data']['random_seed'])
        
        logger.info(f"Generating {n_samples} synthetic deforestation samples")
        
        # Generate features based on configuration
        features_config = self.config['data']['features']
        
        ndvi_change = np.random.normal(
            features_config['ndvi_change']['mean'],
            features_config['ndvi_change']['std'],
            n_samples
        )
        
        surface_temp_rise = np.random.normal(
            features_config['surface_temp_rise']['mean'],
            features_config['surface_temp_rise']['std'],
            n_samples
        )
        
        logging_index = np.random.normal(
            features_config['logging_index']['mean'],
            features_config['logging_index']['std'],
            n_samples
        )
        
        days_since_rain = np.random.normal(
            features_config['days_since_rain']['mean'],
            features_config['days_since_rain']['std'],
            n_samples
        )
        
        distance_to_road = np.random.normal(
            features_config['distance_to_road']['mean'],
            features_config['distance_to_road']['std'],
            n_samples
        )
        
        # Generate labels based on deforestation criteria
        thresholds = self.config['data']['deforestation_thresholds']
        deforested = (
            (ndvi_change < thresholds['ndvi_change']) &
            (logging_index > thresholds['logging_index']) &
            (days_since_rain > thresholds['days_since_rain']) &
            (distance_to_road < thresholds['distance_to_road'])
        ).astype(int)
        
        # Create feature matrix
        X = np.stack([
            ndvi_change,
            surface_temp_rise,
            logging_index,
            days_since_rain,
            distance_to_road
        ], axis=1)
        
        # Create metadata DataFrame with spatial information
        metadata = pd.DataFrame({
            'sample_id': range(n_samples),
            'latitude': np.random.uniform(-20, 10, n_samples),  # Amazon region
            'longitude': np.random.uniform(-80, -40, n_samples),
            'ndvi_change': ndvi_change,
            'surface_temp_rise': surface_temp_rise,
            'logging_index': logging_index,
            'days_since_rain': days_since_rain,
            'distance_to_road': distance_to_road,
            'deforested': deforested
        })
        
        logger.info(f"Generated data: {np.sum(deforested)}/{n_samples} deforested samples ({np.mean(deforested)*100:.1f}%)")
        
        return X, deforested, metadata
    
    def add_spatial_features(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """Add spatial features to the metadata.
        
        Args:
            metadata: DataFrame with latitude and longitude columns
            
        Returns:
            DataFrame with additional spatial features
        """
        logger.info("Adding spatial features")
        
        # Create GeoDataFrame
        geometry = [Point(xy) for xy in zip(metadata['longitude'], metadata['latitude'])]
        gdf = gpd.GeoDataFrame(metadata, geometry=geometry, crs=self.config['spatial']['crs'])
        
        # Add spatial features
        metadata['distance_to_equator'] = np.abs(metadata['latitude'])
        metadata['distance_to_meridian'] = np.abs(metadata['longitude'] + 60)  # Approximate center
        
        # Add spatial lag features (simplified)
        metadata['spatial_lag_ndvi'] = metadata['ndvi_change'].rolling(window=5, center=True).mean().fillna(metadata['ndvi_change'])
        metadata['spatial_lag_logging'] = metadata['logging_index'].rolling(window=5, center=True).mean().fillna(metadata['logging_index'])
        
        return metadata
    
    def prepare_train_test_split(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        metadata: Optional[pd.DataFrame] = None,
        test_size: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Prepare train-test split with optional spatial stratification.
        
        Args:
            X: Feature matrix
            y: Target labels
            metadata: Optional metadata DataFrame
            test_size: Test set size. If None, uses config value.
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, metadata_train, metadata_test)
        """
        if test_size is None:
            test_size = self.config['evaluation']['test_size']
            
        logger.info(f"Creating train-test split with test_size={test_size}")
        
        # Use spatial stratification if metadata is provided
        if metadata is not None and 'latitude' in metadata.columns:
            # Create spatial bins for stratification
            lat_bins = pd.cut(metadata['latitude'], bins=5, labels=False)
            lon_bins = pd.cut(metadata['longitude'], bins=5, labels=False)
            spatial_strata = lat_bins * 5 + lon_bins
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=test_size,
                stratify=spatial_strata,
                random_state=self.config['evaluation']['random_seed']
            )
            
            metadata_train, metadata_test = train_test_split(
                metadata,
                test_size=test_size,
                stratify=spatial_strata,
                random_state=self.config['evaluation']['random_seed']
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                stratify=y,
                random_state=self.config['evaluation']['random_seed']
            )
            
            if metadata is not None:
                metadata_train, metadata_test = train_test_split(
                    metadata,
                    test_size=test_size,
                    stratify=y,
                    random_state=self.config['evaluation']['random_seed']
                )
            else:
                metadata_train, metadata_test = None, None
        
        logger.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test, metadata_train, metadata_test
    
    def scale_features(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Scale features using StandardScaler.
        
        Args:
            X_train: Training features
            X_test: Test features
            
        Returns:
            Tuple of scaled (X_train, X_test)
        """
        logger.info("Scaling features")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled
    
    def save_data(self, X: np.ndarray, y: np.ndarray, metadata: pd.DataFrame, 
                  file_prefix: str = "deforestation_data") -> None:
        """Save processed data to files.
        
        Args:
            X: Feature matrix
            y: Target labels
            metadata: Metadata DataFrame
            file_prefix: Prefix for saved files
        """
        data_dir = Path(self.config['data']['processed_dir'])
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Save features and labels
        np.save(data_dir / f"{file_prefix}_features.npy", X)
        np.save(data_dir / f"{file_prefix}_labels.npy", y)
        
        # Save metadata
        metadata.to_csv(data_dir / f"{file_prefix}_metadata.csv", index=False)
        
        logger.info(f"Saved data to {data_dir} with prefix {file_prefix}")
    
    def load_data(self, file_prefix: str = "deforestation_data") -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Load processed data from files.
        
        Args:
            file_prefix: Prefix for saved files
            
        Returns:
            Tuple of (features, labels, metadata)
        """
        data_dir = Path(self.config['data']['processed_dir'])
        
        X = np.load(data_dir / f"{file_prefix}_features.npy")
        y = np.load(data_dir / f"{file_prefix}_labels.npy")
        metadata = pd.read_csv(data_dir / f"{file_prefix}_metadata.csv")
        
        logger.info(f"Loaded data from {data_dir} with prefix {file_prefix}")
        
        return X, y, metadata
