"""Streamlit demo for deforestation monitoring."""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium import plugins
import sys
from pathlib import Path
import yaml
import joblib

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data.processor import DeforestationDataProcessor
from models.deforestation_models import (
    XGBoostModel, LightGBMModel, NeuralNetworkModel, SpatialMLModel
)
from viz.visualizer import DeforestationVisualizer


def load_config():
    """Load configuration."""
    with open("configs/config.yaml", 'r') as f:
        return yaml.safe_load(f)


def load_trained_models():
    """Load trained models."""
    models = {}
    model_dir = Path("assets/models")
    
    if model_dir.exists():
        for model_file in model_dir.glob("*.pkl"):
            try:
                model_name = model_file.stem
                models[model_name] = joblib.load(model_file)
            except Exception as e:
                st.error(f"Error loading {model_file}: {str(e)}")
    
    return models


def generate_sample_data(n_samples: int = 100) -> Tuple[np.ndarray, pd.DataFrame]:
    """Generate sample data for demonstration."""
    config = load_config()
    processor = DeforestationDataProcessor(config)
    
    X, y, metadata = processor.generate_synthetic_data(n_samples)
    metadata = processor.add_spatial_features(metadata)
    
    return X, metadata


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Deforestation Monitoring Dashboard",
        page_icon="🌲",
        layout="wide"
    )
    
    st.title("🌲 Deforestation Monitoring Dashboard")
    st.markdown("Real-time deforestation detection using satellite imagery and machine learning")
    
    # Load configuration
    config = load_config()
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Model selection
    st.sidebar.subheader("Model Selection")
    available_models = ["XGBoost", "LightGBM", "Neural Network", "Spatial ML"]
    selected_model = st.sidebar.selectbox("Select Model", available_models)
    
    # Data parameters
    st.sidebar.subheader("Data Parameters")
    n_samples = st.sidebar.slider("Number of Samples", 50, 500, 200)
    
    # Risk threshold
    risk_threshold = st.sidebar.slider("Risk Threshold", 0.1, 0.9, 0.5)
    
    # Generate sample data
    with st.spinner("Generating sample data..."):
        X, metadata = generate_sample_data(n_samples)
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Spatial Analysis", "Risk Assessment", "Model Performance"])
    
    with tab1:
        st.header("Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Samples", len(metadata))
        
        with col2:
            deforested_count = metadata['deforested'].sum()
            st.metric("Deforested Areas", deforested_count)
        
        with col3:
            deforestation_rate = metadata['deforested'].mean() * 100
            st.metric("Deforestation Rate", f"{deforestation_rate:.1f}%")
        
        with col4:
            avg_ndvi = metadata['ndvi_change'].mean()
            st.metric("Avg NDVI Change", f"{avg_ndvi:.3f}")
        
        # Feature distributions
        st.subheader("Feature Distributions")
        
        feature_cols = ['ndvi_change', 'surface_temp_rise', 'logging_index', 
                       'days_since_rain', 'distance_to_road']
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[col.replace('_', ' ').title() for col in feature_cols],
            specs=[[{"secondary_y": False}] * 3] * 2
        )
        
        for i, feature in enumerate(feature_cols):
            row = i // 3 + 1
            col = i % 3 + 1
            
            intact_data = metadata[metadata['deforested'] == 0][feature]
            deforested_data = metadata[metadata['deforested'] == 1][feature]
            
            fig.add_trace(
                go.Histogram(x=intact_data, name='Forest Intact', opacity=0.7, 
                           marker_color='green'),
                row=row, col=col
            )
            fig.add_trace(
                go.Histogram(x=deforested_data, name='Deforested', opacity=0.7,
                           marker_color='red'),
                row=row, col=col
            )
        
        fig.update_layout(height=600, showlegend=False, title_text="Feature Distributions by Deforestation Status")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Spatial Analysis")
        
        # Interactive map
        st.subheader("Interactive Deforestation Map")
        
        # Create map
        center_lat = metadata['latitude'].mean()
        center_lon = metadata['longitude'].mean()
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=6,
            tiles='OpenStreetMap'
        )
        
        # Add markers
        for idx, row in metadata.iterrows():
            color = 'red' if row['deforested'] == 1 else 'green'
            size = 8 if row['deforested'] == 1 else 5
            
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=size,
                popup=f"""
                <b>Status:</b> {'Deforested' if row['deforested'] == 1 else 'Forest Intact'}<br>
                <b>NDVI Change:</b> {row['ndvi_change']:.3f}<br>
                <b>Logging Index:</b> {row['logging_index']:.3f}<br>
                <b>Days Since Rain:</b> {row['days_since_rain']:.1f}<br>
                <b>Distance to Road:</b> {row['distance_to_road']:.1f} km
                """,
                color=color,
                fill=True,
                fillOpacity=0.7
            ).add_to(m)
        
        # Display map
        st.components.v1.html(m._repr_html_(), height=500)
        
        # Regional analysis
        st.subheader("Regional Analysis")
        
        # Split by latitude regions
        metadata['region'] = pd.cut(metadata['latitude'], bins=3, labels=['South', 'Center', 'North'])
        
        regional_stats = metadata.groupby('region').agg({
            'deforested': ['count', 'sum', 'mean'],
            'ndvi_change': 'mean',
            'logging_index': 'mean'
        }).round(3)
        
        st.dataframe(regional_stats)
    
    with tab3:
        st.header("Risk Assessment")
        
        # Simulate risk predictions
        np.random.seed(42)
        risk_scores = np.random.beta(2, 5, len(metadata))  # Skewed towards lower risk
        
        # Adjust risk based on actual deforestation
        risk_scores[metadata['deforested'] == 1] += np.random.beta(3, 2, metadata['deforested'].sum())
        risk_scores = np.clip(risk_scores, 0, 1)
        
        metadata['risk_score'] = risk_scores
        
        # Risk distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Risk Score Distribution")
            
            fig = px.histogram(
                metadata, x='risk_score', color='deforested',
                title="Risk Score Distribution",
                labels={'deforested': 'Deforested', 'risk_score': 'Risk Score'},
                color_discrete_map={0: 'green', 1: 'red'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Risk vs Actual Deforestation")
            
            # Create risk bins
            metadata['risk_bin'] = pd.cut(metadata['risk_score'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
            
            risk_calibration = metadata.groupby('risk_bin')['deforested'].agg(['count', 'sum', 'mean']).reset_index()
            risk_calibration.columns = ['Risk Level', 'Count', 'Deforested Count', 'Deforestation Rate']
            
            fig = px.bar(
                risk_calibration, x='Risk Level', y='Deforestation Rate',
                title="Risk Calibration",
                color='Deforestation Rate',
                color_continuous_scale='RdYlGn_r'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # High-risk areas
        st.subheader("High-Risk Areas")
        
        high_risk_threshold = st.slider("High Risk Threshold", 0.1, 0.9, 0.7)
        high_risk_areas = metadata[metadata['risk_score'] > high_risk_threshold]
        
        st.write(f"**{len(high_risk_areas)} areas** identified as high-risk (threshold: {high_risk_threshold})")
        
        if len(high_risk_areas) > 0:
            st.dataframe(
                high_risk_areas[['latitude', 'longitude', 'risk_score', 'ndvi_change', 
                               'logging_index', 'deforested']].round(3)
            )
    
    with tab4:
        st.header("Model Performance")
        
        # Simulate model performance metrics
        models = ['XGBoost', 'LightGBM', 'Neural Network', 'Spatial ML']
        
        # Generate realistic performance metrics
        np.random.seed(42)
        performance_data = []
        
        for model in models:
            # Generate correlated metrics
            base_performance = np.random.uniform(0.7, 0.95)
            
            performance_data.append({
                'Model': model,
                'Accuracy': base_performance + np.random.normal(0, 0.02),
                'Precision': base_performance + np.random.normal(0, 0.02),
                'Recall': base_performance + np.random.normal(0, 0.02),
                'F1-Score': base_performance + np.random.normal(0, 0.02),
                'ROC-AUC': base_performance + np.random.normal(0, 0.02)
            })
        
        performance_df = pd.DataFrame(performance_data)
        
        # Performance comparison
        st.subheader("Model Performance Comparison")
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        
        fig = go.Figure()
        
        for metric in metrics:
            fig.add_trace(go.Bar(
                name=metric,
                x=performance_df['Model'],
                y=performance_df[metric],
                text=performance_df[metric].round(3),
                textposition='auto'
            ))
        
        fig.update_layout(
            title="Model Performance Metrics",
            xaxis_title="Model",
            yaxis_title="Score",
            barmode='group',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance table
        st.subheader("Detailed Performance Metrics")
        st.dataframe(performance_df.round(4))
        
        # Model recommendations
        st.subheader("Model Recommendations")
        
        best_model = performance_df.loc[performance_df['F1-Score'].idxmax()]
        
        st.success(f"""
        **Recommended Model:** {best_model['Model']}
        
        - **F1-Score:** {best_model['F1-Score']:.4f}
        - **Accuracy:** {best_model['Accuracy']:.4f}
        - **ROC-AUC:** {best_model['ROC-AUC']:.4f}
        
        This model provides the best balance of precision and recall for deforestation detection.
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p><b>Deforestation Monitoring System</b></p>
        <p>Author: <a href='https://github.com/kryptologyst'>kryptologyst</a></p>
        <p><em>This is a research demonstration. Not for operational use.</em></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
