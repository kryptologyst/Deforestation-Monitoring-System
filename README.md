# Deforestation Monitoring System

A comprehensive machine learning system for monitoring deforestation using satellite imagery and vegetation indices. This project demonstrates advanced techniques for environmental monitoring with multiple model architectures, spatial analysis, and interactive visualization.

## Features

- **Multiple ML Models**: XGBoost, LightGBM, Neural Networks, and Spatial ML
- **Spatial Analysis**: Geographic visualization and regional performance analysis
- **Interactive Dashboard**: Streamlit-based web interface for real-time monitoring
- **Comprehensive Evaluation**: Detailed metrics and model comparison
- **Modern Architecture**: Clean, typed code with proper configuration management

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Deforestation-Monitoring-System.git
cd Deforestation-Monitoring-System
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the training pipeline:
```bash
python scripts/train_models.py
```

4. Launch the interactive demo:
```bash
streamlit run demo/streamlit_app.py
```

## Project Structure

```
deforestation-monitoring/
├── src/                    # Source code
│   ├── data/              # Data processing modules
│   ├── models/            # Model implementations
│   ├── eval/              # Evaluation modules
│   └── viz/               # Visualization modules
├── configs/               # Configuration files
├── data/                  # Data directories
│   ├── raw/               # Raw data
│   ├── processed/         # Processed data
│   └── external/          # External data sources
├── scripts/               # Training and utility scripts
├── demo/                  # Interactive demo
├── tests/                 # Unit tests
├── assets/                # Generated outputs
│   ├── models/            # Trained models
│   ├── plots/             # Visualization plots
│   └── maps/               # Interactive maps
└── notebooks/              # Jupyter notebooks
```

## Data Schema

The system processes the following features:

- **NDVI Change**: Normalized Difference Vegetation Index change over time
- **Surface Temperature Rise**: Temperature increase in degrees Celsius
- **Logging Index**: Activity level indicator (0-1 scale)
- **Days Since Rain**: Temporal drought indicator
- **Distance to Road**: Proximity to infrastructure (km)

### Spatial Features

- Latitude and longitude coordinates
- Distance to equator and meridian
- Spatial lag features for neighboring areas
- Regional classification (North/Center/South)

## Model Architecture

### Baseline Models
- **XGBoost**: Gradient boosting with spatial features
- **LightGBM**: Fast gradient boosting implementation
- **Random Forest**: Ensemble method for spatial ML

### Advanced Models
- **Neural Network**: Deep learning with dropout regularization
- **Spatial ML**: Random Forest with spatial feature engineering

### Model Selection
Models are evaluated using:
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC and PR-AUC
- Spatial performance metrics
- Cross-validation scores

## Training and Evaluation

### Training Pipeline
```bash
# Train all models
python scripts/train_models.py

# Train specific model
python scripts/train_models.py --model xgboost
```

### Evaluation Metrics
- **Classification**: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
- **Spatial**: IoU, Dice coefficient, Moran's I
- **Regional**: Performance by geographic region
- **Calibration**: Risk score calibration analysis

### Model Comparison
The system generates:
- Performance leaderboard
- Confusion matrices
- ROC curves
- Feature importance analysis

## Interactive Demo

The Streamlit dashboard provides:

### Overview Tab
- Data summary statistics
- Feature distribution plots
- Correlation analysis

### Spatial Analysis Tab
- Interactive deforestation map
- Regional performance analysis
- Geographic clustering visualization

### Risk Assessment Tab
- Risk score distribution
- Risk calibration analysis
- High-risk area identification

### Model Performance Tab
- Performance comparison charts
- Detailed metrics table
- Model recommendations

## Configuration

### Main Configuration (`configs/config.yaml`)
- Data parameters and feature engineering
- Model hyperparameters
- Evaluation settings
- Visualization preferences

### Spatial Configuration (`configs/spatial.yaml`)
- Coordinate reference systems
- Region of interest settings
- Spatial feature parameters
- Remote sensing configuration

## Usage Examples

### Basic Usage
```python
from src.data.processor import DeforestationDataProcessor
from src.models.deforestation_models import XGBoostModel

# Initialize processor
processor = DeforestationDataProcessor("configs/config.yaml")

# Generate data
X, y, metadata = processor.generate_synthetic_data(1000)

# Train model
model = XGBoostModel(config)
model.train(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
```

### Advanced Usage
```python
from src.eval.evaluator import DeforestationEvaluator
from src.viz.visualizer import DeforestationVisualizer

# Evaluate model
evaluator = DeforestationEvaluator(config)
results = evaluator.evaluate_model(model, X_test, y_test, "XGBoost")

# Create visualizations
visualizer = DeforestationVisualizer(config)
visualizer.create_interactive_map(metadata, predictions)
```

## Development

### Code Quality
- Type hints throughout the codebase
- Google/NumPy docstring format
- Black code formatting
- Ruff linting
- Comprehensive error handling

### Testing
```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

### Pre-commit Hooks
```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install
```

## Outputs

The system generates several outputs:

### Models
- Trained model files (`.pkl`)
- Model performance metrics
- Cross-validation results

### Visualizations
- Feature distribution plots
- Correlation matrices
- Model comparison charts
- Interactive maps
- Risk dashboards

### Reports
- Evaluation report (Markdown)
- Performance leaderboard (CSV)
- Results summary (JSON)

## Limitations and Disclaimers

⚠️ **Important**: This system is for **research and educational purposes only**. See [DISCLAIMER.md](DISCLAIMER.md) for important limitations and ethical considerations.

### Key Limitations
- Uses synthetic data for demonstration
- Not validated on real-world deforestation data
- Not suitable for operational environmental monitoring
- Performance metrics are simulated

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Author

**kryptologyst**  
GitHub: https://github.com/kryptologyst

## Acknowledgments

- Remote sensing and environmental monitoring research community
- Open source machine learning libraries
- Satellite data providers (Landsat, Sentinel, MODIS)

## References

1. Hansen, M. C., et al. (2013). High-resolution global maps of 21st-century forest cover change. Science, 342(6160), 850-853.
2. Gorelick, N., et al. (2017). Google Earth Engine: Planetary-scale geospatial analysis for everyone. Remote Sensing of Environment, 202, 18-27.
3. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.

---

For questions or support, please open an issue on GitHub or contact the author.
# Deforestation-Monitoring-System
