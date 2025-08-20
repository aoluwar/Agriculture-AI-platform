# Sample Datasets for Agricultural AI Platform

## Overview
This document describes the sample datasets included with the Agricultural AI Platform and provides information on how to use them for training and testing AI models.

## Included Datasets

### 1. Crop Disease Images
- **Location**: `data/crop_diseases/`
- **Description**: A collection of labeled images showing various crop diseases across different plant types
- **Classes**: 38 different disease classes across 10 crop types
- **Format**: JPG images organized in folders by disease type
- **Size**: 1,000 sample images (full dataset would contain 50,000+ images)
- **Source**: Based on the PlantVillage dataset with additional proprietary images

### 2. Soil Analysis Data
- **Location**: `data/soil_samples/`
- **Description**: CSV files containing soil analysis results from various agricultural regions
- **Features**: pH, nitrogen, phosphorus, potassium, organic matter, texture, etc.
- **Format**: CSV files with headers
- **Size**: 500 sample records
- **Source**: Synthetic data based on typical soil testing results

### 3. Weather Data
- **Location**: `data/weather/`
- **Description**: Historical weather data for agricultural regions
- **Features**: Temperature, precipitation, humidity, wind speed, solar radiation
- **Format**: CSV files organized by region and year
- **Size**: 5 years of daily data for 10 regions
- **Source**: Derived from public weather station data

### 4. Crop Yield Data
- **Location**: `data/yields/`
- **Description**: Historical crop yield data correlated with weather and soil conditions
- **Features**: Crop type, yield amount, planting date, harvest date, weather conditions, soil conditions
- **Format**: CSV files organized by crop type and region
- **Size**: 10 years of data for 5 major crops across 10 regions
- **Source**: Synthetic data based on typical agricultural yields

## Using the Datasets

### Data Preprocessing
Before using these datasets for model training, you should:

1. **Clean the data**:
   ```python
   # Example code for cleaning soil data
   import pandas as pd
   
   # Load the data
   soil_data = pd.read_csv('data/soil_samples/soil_data.csv')
   
   # Remove rows with missing values
   soil_data = soil_data.dropna()
   
   # Remove outliers
   soil_data = soil_data[(soil_data['pH'] >= 3.5) & (soil_data['pH'] <= 10.0)]
   ```

2. **Split into training and testing sets**:
   ```python
   from sklearn.model_selection import train_test_split
   
   # Split the data (80% training, 20% testing)
   train_data, test_data = train_test_split(soil_data, test_size=0.2, random_state=42)
   ```

3. **Normalize numerical features**:
   ```python
   from sklearn.preprocessing import StandardScaler
   
   # Initialize the scaler
   scaler = StandardScaler()
   
   # Fit and transform the training data
   numerical_features = ['nitrogen', 'phosphorus', 'potassium', 'organic_matter']
   train_data[numerical_features] = scaler.fit_transform(train_data[numerical_features])
   
   # Transform the test data
   test_data[numerical_features] = scaler.transform(test_data[numerical_features])
   ```

### Example: Loading Image Data for Crop Disease Detection

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set up data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 20% for validation
)

# Load training data
train_generator = train_datagen.flow_from_directory(
    'data/crop_diseases',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Load validation data
validation_generator = train_datagen.flow_from_directory(
    'data/crop_diseases',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)
```

## Adding Your Own Data

You can extend these datasets with your own data by following these guidelines:

1. **Image data**: Add new images to the appropriate disease class folders
2. **CSV data**: Ensure your data follows the same format as the sample files
3. **Documentation**: Update this document to reflect your additions

## Data Sources and Citations

When using these datasets for research or publications, please cite the following sources:

- PlantVillage Dataset: Hughes, D.P., & Salathé, M. (2015). An open access repository of images on plant health to enable the development of mobile disease diagnostics.
- Weather data derived from NOAA National Centers for Environmental Information
- Soil and yield data are synthetic and based on typical agricultural patterns