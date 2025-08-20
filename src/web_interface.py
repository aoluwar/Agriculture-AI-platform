#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Web Interface for Agricultural AI Platform

This module provides a Streamlit-based web interface for the Agricultural AI Platform,
allowing users to interact with the platform's models and services through a user-friendly UI.
"""

import os
import yaml
import numpy as np
import pandas as pd
import streamlit as st
import cv2
from PIL import Image
import io
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import platform modules
from models.crop_disease_model import CropDiseaseModel
from src.data_preprocessing import preprocess_image
from src.visualization import (
    visualize_crop_disease_prediction,
    plot_soil_analysis_results,
    plot_weather_forecast,
    plot_yield_prediction
)

# Load configuration
def load_config(config_path='config.yaml'):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        st.error(f"Error loading configuration: {e}")
        # Return default configuration
        return {
            'models': {
                'crop_disease': {
                    'num_classes': 38,
                    'input_shape': [224, 224, 3],
                    'saved_model_path': 'models/saved/crop_disease_model.h5'
                }
            }
        }

# Initialize models
@st.cache_resource
def load_crop_disease_model(config):
    """
    Load the crop disease detection model.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        CropDiseaseModel: Loaded model
    """
    try:
        # Get model parameters from config
        model_path = config['models']['crop_disease']['saved_model_path']
        num_classes = config['models']['crop_disease']['num_classes']
        input_shape = tuple(config['models']['crop_disease']['input_shape'])
        
        # Initialize model
        model = CropDiseaseModel(num_classes=num_classes, input_shape=input_shape)
        
        # Try to load saved model
        try:
            model.load(model_path)
            st.success("Crop disease model loaded successfully")
        except Exception as e:
            st.warning(f"Could not load saved model: {e}. Initializing new model.")
            model.build_model()
        
        return model
    except Exception as e:
        st.error(f"Error initializing model: {e}")
        # Return a basic model as fallback
        model = CropDiseaseModel()
        model.build_model()
        return model

# Main app
def main():
    """
    Main function for the Streamlit web interface.
    """
    # Set page config
    st.set_page_config(
        page_title="Agricultural AI Platform",
        page_icon="🌱",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load configuration
    config = load_config()
    
    # Sidebar
    st.sidebar.title("Agricultural AI Platform")
    app_mode = st.sidebar.selectbox(
        "Choose a mode",
        ["Home", "Crop Disease Detection", "Soil Analysis", "Weather Prediction", "Yield Prediction"]
    )
    
    # Home page
    if app_mode == "Home":
        st.title("Welcome to the Agricultural AI Platform")
        
        st.markdown("""
        ## About the Platform
        
        The Agricultural AI Platform is a comprehensive solution for applying artificial intelligence 
        to agricultural challenges. Our platform offers tools for crop disease detection, soil analysis, 
        weather prediction, and yield forecasting.
        
        ## Features
        
        - **Crop Disease Detection**: Upload images of crop leaves to identify diseases
        - **Soil Analysis**: Analyze soil samples for nutrient content and recommendations
        - **Weather Prediction**: Get weather forecasts tailored for agricultural planning
        - **Yield Prediction**: Predict crop yields based on various parameters
        
        ## Getting Started
        
        Select a mode from the sidebar to get started with the platform.
        """)
        
        # Display sample images
        st.subheader("Sample Applications")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.image("https://via.placeholder.com/300x200.png?text=Crop+Disease+Detection", 
                    caption="Crop Disease Detection")
        
        with col2:
            st.image("https://via.placeholder.com/300x200.png?text=Soil+Analysis", 
                    caption="Soil Analysis")
        
        with col3:
            st.image("https://via.placeholder.com/300x200.png?text=Weather+Prediction", 
                    caption="Weather Prediction")
        
        with col4:
            st.image("https://via.placeholder.com/300x200.png?text=Yield+Prediction", 
                    caption="Yield Prediction")
    
    # Crop Disease Detection
    elif app_mode == "Crop Disease Detection":
        st.title("Crop Disease Detection")
        
        st.markdown("""
        Upload an image of a crop leaf to detect diseases. The model will analyze the image 
        and provide a diagnosis along with confidence scores.
        """)
        
        # Load model
        model = load_crop_disease_model(config)
        
        # File uploader
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Convert PIL Image to numpy array
            img_array = np.array(image)
            
            # Preprocess the image
            input_shape = tuple(config['models']['crop_disease']['input_shape'][:2])
            img_resized = cv2.resize(img_array, input_shape)
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB) if len(img_array.shape) == 3 else cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
            img_normalized = img_rgb / 255.0
            
            # Make prediction
            with st.spinner("Analyzing image..."):
                predictions = model.predict(img_normalized)
            
            # Get top predictions
            top_indices = np.argsort(predictions[0])[-3:][::-1]
            
            # Placeholder class names (in a real app, you'd load these from somewhere)
            class_names = [f"Disease_{i}" for i in range(config['models']['crop_disease']['num_classes'])]
            
            # Display results
            st.subheader("Detection Results")
            
            # Create columns for results
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Top Predictions")
                for i, idx in enumerate(top_indices):
                    confidence = predictions[0][idx] * 100
                    st.markdown(f"**{i+1}. {class_names[idx]}**: {confidence:.2f}%")
            
            with col2:
                # Create a bar chart for visualization
                fig, ax = plt.subplots(figsize=(8, 5))
                y_pos = np.arange(len(top_indices))
                confidences = [predictions[0][idx] * 100 for idx in top_indices]
                labels = [class_names[idx] for idx in top_indices]
                
                bars = ax.barh(y_pos, confidences, align='center')
                ax.set_yticks(y_pos)
                ax.set_yticklabels(labels)
                ax.invert_yaxis()  # Labels read top-to-bottom
                ax.set_xlabel('Confidence (%)')
                ax.set_title('Disease Prediction Confidence')
                
                # Add confidence values to the end of each bar
                for i, bar in enumerate(bars):
                    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                            f'{confidences[i]:.1f}%', va='center')
                
                st.pyplot(fig)
            
            # Recommendations based on detected disease (placeholder)
            st.subheader("Recommendations")
            st.markdown(f"Based on the detected disease ({class_names[top_indices[0]]}), we recommend:")
            st.markdown("""
            1. **Isolate affected plants** to prevent spread of the disease
            2. **Apply appropriate fungicide or treatment** specific to this disease
            3. **Adjust watering practices** to reduce leaf wetness
            4. **Improve air circulation** around plants
            5. **Monitor other plants** for early signs of infection
            """)
    
    # Soil Analysis
    elif app_mode == "Soil Analysis":
        st.title("Soil Analysis")
        
        st.markdown("""
        Enter soil parameters to get analysis results and recommendations for improving soil health.
        """)
        
        # Create form for soil parameters
        with st.form("soil_analysis_form"):
            # Create columns for form fields
            col1, col2 = st.columns(2)
            
            with col1:
                ph = st.slider("pH", 3.0, 10.0, 6.5, 0.1)
                nitrogen = st.slider("Nitrogen (N) mg/kg", 0, 200, 50)
                phosphorus = st.slider("Phosphorus (P) mg/kg", 0, 200, 30)
                potassium = st.slider("Potassium (K) mg/kg", 0, 200, 40)
            
            with col2:
                organic_matter = st.slider("Organic Matter %", 0.0, 10.0, 2.5, 0.1)
                cec = st.slider("CEC (meq/100g)", 0, 50, 15)
                texture = st.selectbox("Soil Texture", ["Sandy", "Loamy", "Clay", "Silt", "Sandy Loam", "Clay Loam", "Silt Loam"])
                moisture = st.slider("Moisture %", 0, 100, 30)
            
            submitted = st.form_submit_button("Analyze Soil")
        
        if submitted:
            # Create soil data dictionary
            soil_data = {
                'pH': ph,
                'N': nitrogen,
                'P': phosphorus,
                'K': potassium,
                'Organic Matter': organic_matter,
                'CEC': cec,
                'Moisture': moisture
            }
            
            # Display results
            st.subheader("Soil Analysis Results")
            
            # Create columns for results
            col1, col2 = st.columns(2)
            
            with col1:
                # Display soil properties table
                st.markdown("### Soil Properties")
                soil_df = pd.DataFrame({
                    'Property': list(soil_data.keys()),
                    'Value': list(soil_data.values()),
                    'Status': ['Optimal', 'Low', 'Medium', 'Medium', 'Medium', 'High', 'Medium']
                })
                st.dataframe(soil_df)
                
                # Soil type classification (simplified)
                st.markdown("### Soil Classification")
                st.markdown(f"**Soil Texture**: {texture}")
                st.markdown(f"**Soil Type**: {texture} soil with {soil_df.loc[soil_df['Property'] == 'pH', 'Status'].values[0]} pH")
            
            with col2:
                # Visualize soil analysis results
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # Define optimal ranges for visualization
                optimal_ranges = {
                    'pH': (6.0, 7.5),
                    'N': (50, 100),
                    'P': (25, 50),
                    'K': (40, 80),
                    'Organic Matter': (3.0, 5.0),
                    'CEC': (10, 20),
                    'Moisture': (20, 40)
                }
                
                # Create normalized values for visualization
                normalized_values = []
                colors = []
                properties = []
                
                for prop, val in soil_data.items():
                    if prop in optimal_ranges:
                        min_val, max_val = optimal_ranges[prop]
                        if val < min_val:
                            normalized_values.append(val / min_val)
                            colors.append('orange')
                        elif val > max_val:
                            normalized_values.append(1 + (val - max_val) / max_val)
                            colors.append('orange')
                        else:
                            normalized_values.append(1.0)
                            colors.append('green')
                        properties.append(prop)
                
                # Create bar chart
                bars = ax.bar(properties, normalized_values, color=colors)
                
                # Add a horizontal line at y=1.0 to indicate optimal level
                ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.7)
                
                # Add labels
                ax.set_title('Soil Properties Relative to Optimal Levels')
                ax.set_ylabel('Relative Level (1.0 = Optimal)')
                ax.set_ylim(0, 2.0)
                
                # Rotate x-axis labels
                plt.xticks(rotation=45, ha='right')
                
                st.pyplot(fig)
            
            # Recommendations
            st.subheader("Recommendations")
            
            # Generate recommendations based on soil parameters
            recommendations = []
            
            if ph < 6.0:
                recommendations.append("Apply lime to increase soil pH")
            elif ph > 7.5:
                recommendations.append("Apply sulfur to decrease soil pH")
            
            if nitrogen < 50:
                recommendations.append("Increase nitrogen with nitrogen-rich fertilizers or legume cover crops")
            
            if phosphorus < 25:
                recommendations.append("Add phosphorus with bone meal or rock phosphate")
            
            if potassium < 40:
                recommendations.append("Increase potassium with potassium sulfate or wood ash")
            
            if organic_matter < 3.0:
                recommendations.append("Add compost or other organic matter to improve soil structure")
            
            if moisture < 20:
                recommendations.append("Improve water retention with mulch or organic matter")
            elif moisture > 40:
                recommendations.append("Improve drainage to reduce excess moisture")
            
            # Display recommendations
            for i, rec in enumerate(recommendations):
                st.markdown(f"{i+1}. {rec}")
            
            if not recommendations:
                st.markdown("Your soil parameters are within optimal ranges. Continue with current practices.")
    
    # Weather Prediction
    elif app_mode == "Weather Prediction":
        st.title("Weather Prediction")
        
        st.markdown("""
        Get weather predictions for agricultural planning. Enter your location and time period 
        to receive forecasts tailored for farming activities.
        """)
        
        # Create form for weather prediction
        with st.form("weather_prediction_form"):
            # Create columns for form fields
            col1, col2 = st.columns(2)
            
            with col1:
                location = st.text_input("Location", "New York, NY")
                start_date = st.date_input("Start Date")
            
            with col2:
                radius = st.slider("Radius (km)", 1, 50, 10)
                days = st.slider("Forecast Days", 1, 14, 7)
            
            submitted = st.form_submit_button("Get Forecast")
        
        if submitted:
            # Generate mock weather forecast data
            forecast_data = []
            start_temp = np.random.randint(15, 25)  # Starting temperature
            
            for i in range(days):
                date = start_date + pd.Timedelta(days=i)
                temp = start_temp + np.random.randint(-3, 4)  # Random temperature variation
                precip = max(0, np.random.normal(0.5, 1.0))  # Random precipitation
                humidity = np.random.randint(50, 80)  # Random humidity
                wind = np.random.randint(0, 20)  # Random wind speed
                
                forecast_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'temperature': temp,
                    'precipitation': round(precip, 1),
                    'humidity': humidity,
                    'wind_speed': wind
                })
            
            # Display results
            st.subheader(f"Weather Forecast for {location}")
            
            # Create columns for results
            col1, col2 = st.columns([2, 3])
            
            with col1:
                # Display forecast table
                forecast_df = pd.DataFrame(forecast_data)
                st.dataframe(forecast_df)
                
                # Calculate statistics
                avg_temp = np.mean([d['temperature'] for d in forecast_data])
                total_precip = sum([d['precipitation'] for d in forecast_data])
                avg_humidity = np.mean([d['humidity'] for d in forecast_data])
                
                st.markdown("### Summary Statistics")
                st.markdown(f"**Average Temperature**: {avg_temp:.1f}°C")
                st.markdown(f"**Total Precipitation**: {total_precip:.1f} mm")
                st.markdown(f"**Average Humidity**: {avg_humidity:.1f}%")
            
            with col2:
                # Visualize weather forecast
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
                
                # Temperature plot
                dates = [d['date'] for d in forecast_data]
                temps = [d['temperature'] for d in forecast_data]
                ax1.plot(dates, temps, 'o-', color='red', linewidth=2, markersize=8)
                ax1.set_title('Temperature Forecast')
                ax1.set_ylabel('Temperature (°C)')
                ax1.grid(True, alpha=0.3)
                
                # Precipitation plot
                precips = [d['precipitation'] for d in forecast_data]
                ax2.bar(dates, precips, color='blue', alpha=0.7)
                ax2.set_title('Precipitation Forecast')
                ax2.set_xlabel('Date')
                ax2.set_ylabel('Precipitation (mm)')
                ax2.grid(True, axis='y', alpha=0.3)
                
                # Rotate x-axis labels
                plt.xticks(rotation=45)
                
                plt.tight_layout()
                st.pyplot(fig)
            
            # Agricultural recommendations
            st.subheader("Agricultural Recommendations")
            
            # Generate recommendations based on weather forecast
            recommendations = []
            
            # Check for rain
            rainy_days = sum(1 for d in forecast_data if d['precipitation'] > 1.0)
            if rainy_days > days / 2:
                recommendations.append("Delay planting or harvesting due to frequent rain")
                recommendations.append("Ensure proper drainage in fields")
            elif rainy_days == 0:
                recommendations.append("Plan for irrigation as no significant rainfall is expected")
            
            # Check temperature trends
            if all(d['temperature'] > 25 for d in forecast_data):
                recommendations.append("Monitor crops for heat stress")
                recommendations.append("Increase irrigation frequency")
            elif any(d['temperature'] < 10 for d in forecast_data):
                recommendations.append("Protect sensitive crops from cold temperatures")
            
            # Display recommendations
            for i, rec in enumerate(recommendations):
                st.markdown(f"{i+1}. {rec}")
            
            if not recommendations:
                st.markdown("Weather conditions appear favorable for most agricultural activities.")
    
    # Yield Prediction
    elif app_mode == "Yield Prediction":
        st.title("Yield Prediction")
        
        st.markdown("""
        Predict crop yields based on various parameters including crop type, soil conditions, 
        weather data, and farming practices.
        """)
        
        # Create form for yield prediction
        with st.form("yield_prediction_form"):
            # Create columns for form fields
            col1, col2, col3 = st.columns(3)
            
            with col1:
                crop_type = st.selectbox("Crop Type", ["Wheat", "Rice", "Corn", "Soybeans", "Potatoes"])
                planting_date = st.date_input("Planting Date")
                field_size = st.number_input("Field Size (hectares)", min_value=0.1, max_value=1000.0, value=10.0)
            
            with col2:
                soil_type = st.selectbox("Soil Type", ["Sandy", "Loamy", "Clay", "Silt", "Sandy Loam"])
                irrigation = st.selectbox("Irrigation Method", ["Drip", "Sprinkler", "Flood", "None"])
                fertilizer = st.selectbox("Fertilizer Type", ["Nitrogen", "Phosphorus", "Potassium", "NPK Mix", "Organic"])
            
            with col3:
                avg_temp = st.slider("Average Temperature (°C)", 0, 40, 20)
                rainfall = st.slider("Seasonal Rainfall (mm)", 0, 2000, 500)
                pest_pressure = st.slider("Pest Pressure", 0, 10, 3)
            
            submitted = st.form_submit_button("Predict Yield")
        
        if submitted:
            # Generate mock yield prediction
            # In a real app, this would use a trained model
            
            # Base yield by crop type (tons per hectare)
            base_yields = {
                "Wheat": 3.5,
                "Rice": 4.2,
                "Corn": 5.5,
                "Soybeans": 2.8,
                "Potatoes": 25.0
            }
            
            # Soil type factors
            soil_factors = {
                "Sandy": 0.8,
                "Loamy": 1.2,
                "Clay": 0.9,
                "Silt": 1.0,
                "Sandy Loam": 1.1
            }
            
            # Irrigation factors
            irrigation_factors = {
                "Drip": 1.3,
                "Sprinkler": 1.2,
                "Flood": 1.1,
                "None": 0.7
            }
            
            # Calculate predicted yield
            base_yield = base_yields.get(crop_type, 4.0)
            soil_factor = soil_factors.get(soil_type, 1.0)
            irrigation_factor = irrigation_factors.get(irrigation, 1.0)
            
            # Temperature effect (simplified)
            temp_factor = 1.0
            if avg_temp < 10 or avg_temp > 30:
                temp_factor = 0.8
            
            # Rainfall effect (simplified)
            rainfall_factor = 1.0
            if rainfall < 300:
                rainfall_factor = 0.7
            elif rainfall > 1500:
                rainfall_factor = 0.9
            
            # Pest pressure effect
            pest_factor = 1.0 - (pest_pressure / 20)  # Reduce yield by up to 50% for max pest pressure
            
            # Calculate final yield
            predicted_yield = base_yield * soil_factor * irrigation_factor * temp_factor * rainfall_factor * pest_factor
            
            # Add some randomness for variation
            predicted_yield = predicted_yield * np.random.uniform(0.9, 1.1)
            
            # Calculate confidence interval (simplified)
            ci_low = predicted_yield * 0.9
            ci_high = predicted_yield * 1.1
            
            # Display results
            st.subheader("Yield Prediction Results")
            
            # Create columns for results
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Predicted Yield")
                st.markdown(f"**Crop**: {crop_type}")
                st.markdown(f"**Field Size**: {field_size} hectares")
                st.markdown(f"**Predicted Yield**: {predicted_yield:.2f} tons/hectare")
                st.markdown(f"**Total Yield**: {predicted_yield * field_size:.2f} tons")
                st.markdown(f"**Confidence Interval**: {ci_low:.2f} - {ci_high:.2f} tons/hectare")
                
                # Impact factors
                st.markdown("### Impact Factors")
                impact_factors = [
                    {"name": "Soil Quality", "impact": "High" if soil_factor > 1.0 else "Low"},
                    {"name": "Irrigation", "impact": "High" if irrigation_factor > 1.0 else "Low"},
                    {"name": "Temperature", "impact": "Medium" if temp_factor == 1.0 else "Low"},
                    {"name": "Rainfall", "impact": "Medium" if rainfall_factor == 1.0 else "Low"},
                    {"name": "Pest Pressure", "impact": "High" if pest_factor < 0.9 else "Low"}
                ]
                
                for factor in impact_factors:
                    st.markdown(f"**{factor['name']}**: {factor['impact']} impact")
            
            with col2:
                # Visualize yield prediction
                crop_data = {"crop": crop_type, "field_size": field_size}
                predictions = {
                    "predicted_yield": predicted_yield,
                    "confidence_interval": [ci_low, ci_high]
                }
                
                # Create bar chart for yield prediction
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # Plot predicted yield
                ax.bar(["Predicted Yield"], [predicted_yield], color='green', alpha=0.7)
                
                # Add confidence interval
                ax.errorbar(["Predicted Yield"], [predicted_yield], 
                           yerr=[[predicted_yield - ci_low], [ci_high - predicted_yield]],
                           fmt='o', color='black', capsize=10)
                
                # Add average yield line
                ax.axhline(y=base_yields[crop_type], color='red', linestyle='--', alpha=0.7,
                          label=f'Average {crop_type} Yield')
                
                ax.set_title(f"Yield Prediction for {crop_type}")
                ax.set_ylabel('Yield (tons/hectare)')
                ax.legend()
                ax.grid(True, axis='y', alpha=0.3)
                
                st.pyplot(fig)
                
                # Create impact factor visualization
                factor_names = [f["name"] for f in impact_factors]
                impact_values = []
                colors = []
                
                for factor in impact_factors:
                    if factor["impact"] == "High":
                        impact_values.append(3)
                        colors.append('red' if "Pest" in factor["name"] else 'green')
                    elif factor["impact"] == "Medium":
                        impact_values.append(2)
                        colors.append('orange')
                    else:  # Low
                        impact_values.append(1)
                        colors.append('blue')
                
                fig, ax = plt.subplots(figsize=(8, 6))
                bars = ax.barh(factor_names, impact_values, color=colors, alpha=0.7)
                
                # Add impact labels
                for i, bar in enumerate(bars):
                    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                           impact_factors[i]['impact'], ha='left', va='center')
                
                ax.set_title('Impact Factors')
                ax.set_xlabel('Impact Level')
                ax.set_xlim(0, 4)
                ax.set_xticks([1, 2, 3])
                ax.set_xticklabels(['Low', 'Medium', 'High'])
                ax.grid(True, axis='x', alpha=0.3)
                
                st.pyplot(fig)
            
            # Recommendations
            st.subheader("Recommendations to Improve Yield")
            
            # Generate recommendations based on factors
            recommendations = []
            
            if soil_factor < 1.0:
                recommendations.append("Improve soil quality with appropriate amendments based on soil test results")
            
            if irrigation_factor < 1.0:
                recommendations.append("Upgrade irrigation system to improve water use efficiency")
            
            if temp_factor < 1.0:
                if avg_temp < 10:
                    recommendations.append("Consider planting later in the season to avoid cold stress")
                else:
                    recommendations.append("Implement shade or cooling measures to reduce heat stress")
            
            if rainfall_factor < 1.0:
                if rainfall < 300:
                    recommendations.append("Increase irrigation to compensate for low rainfall")
                else:
                    recommendations.append("Improve drainage to manage excess rainfall")
            
            if pest_factor < 0.9:
                recommendations.append("Implement integrated pest management strategies to reduce pest pressure")
            
            # Display recommendations
            for i, rec in enumerate(recommendations):
                st.markdown(f"{i+1}. {rec}")
            
            if not recommendations:
                st.markdown("Your current practices appear optimal for maximizing yield.")

if __name__ == "__main__":
    main()