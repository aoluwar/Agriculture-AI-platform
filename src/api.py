#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
API Module for Agricultural AI Platform

This module provides a Flask-based REST API for the Agricultural AI Platform,
allowing users to interact with the platform's models and services.
"""

import os
import yaml
import logging
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import uuid

# Import platform modules
from models.crop_disease_model import CropDiseaseModel
from src.data_preprocessing import preprocess_image, preprocess_soil_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
        logger.error(f"Error loading configuration: {e}")
        # Return default configuration
        return {
            'service': {
                'host': '0.0.0.0',
                'port': 5000,
                'debug': False,
                'upload_folder': 'uploads',
                'allowed_extensions': ['jpg', 'jpeg', 'png'],
                'max_content_length': 16777216  # 16MB
            }
        }

# Initialize Flask app
app = Flask(__name__)
config = load_config()

# Configure app
app.config['UPLOAD_FOLDER'] = config['service']['upload_folder']
app.config['MAX_CONTENT_LENGTH'] = config['service']['max_content_length']
ALLOWED_EXTENSIONS = set(config['service']['allowed_extensions'])

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize models
crop_disease_model = None

def allowed_file(filename):
    """
    Check if the file extension is allowed.
    
    Args:
        filename (str): Name of the file
        
    Returns:
        bool: True if the file extension is allowed, False otherwise
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_models():
    """
    Load AI models for the API.
    """
    global crop_disease_model
    
    try:
        # Load crop disease model
        model_path = config['models']['crop_disease']['saved_model_path']
        num_classes = config['models']['crop_disease']['num_classes']
        
        crop_disease_model = CropDiseaseModel(num_classes=num_classes)
        crop_disease_model.load(model_path)
        logger.info("Crop disease model loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load crop disease model: {e}")
        logger.info("Initializing new crop disease model")
        
        # Initialize a new model if loading fails
        input_shape = tuple(config['models']['crop_disease']['input_shape'])
        crop_disease_model = CropDiseaseModel(
            num_classes=num_classes,
            input_shape=input_shape
        )
        crop_disease_model.build_model()

@app.route('/')
def index():
    """
    API root endpoint.
    """
    return jsonify({
        'name': 'Agricultural AI Platform API',
        'version': config.get('api', {}).get('version', 'v1'),
        'endpoints': [
            '/api/v1/crop-disease',
            '/api/v1/soil-analysis',
            '/api/v1/weather-prediction',
            '/api/v1/crop-yield'
        ]
    })

@app.route('/api/v1/crop-disease', methods=['POST'])
def predict_crop_disease():
    """
    Endpoint for crop disease detection.
    
    Expects an image file in the request.
    """
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    # If user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Generate a unique filename
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save the file
        file.save(file_path)
        
        try:
            # Preprocess the image
            input_shape = tuple(config['models']['crop_disease']['input_shape'][:2])
            img = preprocess_image(file_path, target_size=input_shape)
            
            # Make prediction
            predictions = crop_disease_model.predict(img[0])
            
            # Get the top prediction
            top_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][top_idx])
            
            # Get class names (placeholder - in a real app, you'd load these from somewhere)
            class_names = [f"class_{i}" for i in range(config['models']['crop_disease']['num_classes'])]
            
            # Return the result
            return jsonify({
                'prediction': class_names[top_idx],
                'confidence': confidence,
                'top_predictions': [
                    {'class': class_names[i], 'confidence': float(predictions[0][i])}
                    for i in np.argsort(predictions[0])[-3:][::-1]
                ]
            })
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return jsonify({'error': f"Error processing image: {str(e)}"}), 500
        finally:
            # Clean up the uploaded file
            try:
                os.remove(file_path)
            except:
                pass
    
    return jsonify({'error': 'Invalid file format'}), 400

@app.route('/api/v1/soil-analysis', methods=['POST'])
def analyze_soil():
    """
    Endpoint for soil analysis.
    
    Expects JSON data with soil parameters.
    """
    # Placeholder for soil analysis endpoint
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Here you would process the soil data and make predictions
        # For now, we'll just return a placeholder response
        return jsonify({
            'soil_type': 'Clay Loam',
            'fertility': 'Medium',
            'recommendations': [
                'Add organic matter to improve soil structure',
                'Consider adding nitrogen fertilizer for better crop growth'
            ]
        })
    except Exception as e:
        logger.error(f"Error analyzing soil data: {e}")
        return jsonify({'error': f"Error analyzing soil data: {str(e)}"}), 500

@app.route('/api/v1/weather-prediction', methods=['POST'])
def predict_weather():
    """
    Endpoint for weather prediction.
    
    Expects JSON data with location and time parameters.
    """
    # Placeholder for weather prediction endpoint
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Here you would process the weather data and make predictions
        # For now, we'll just return a placeholder response
        return jsonify({
            'location': data.get('location', 'Unknown'),
            'forecast': [
                {'date': '2023-06-01', 'temperature': 25, 'precipitation': 0.2, 'humidity': 65},
                {'date': '2023-06-02', 'temperature': 27, 'precipitation': 0.0, 'humidity': 60},
                {'date': '2023-06-03', 'temperature': 26, 'precipitation': 0.1, 'humidity': 70}
            ]
        })
    except Exception as e:
        logger.error(f"Error predicting weather: {e}")
        return jsonify({'error': f"Error predicting weather: {str(e)}"}), 500

@app.route('/api/v1/crop-yield', methods=['POST'])
def predict_crop_yield():
    """
    Endpoint for crop yield prediction.
    
    Expects JSON data with crop and field parameters.
    """
    # Placeholder for crop yield prediction endpoint
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Here you would process the crop data and make predictions
        # For now, we'll just return a placeholder response
        return jsonify({
            'crop': data.get('crop', 'Unknown'),
            'field_size': data.get('field_size', 0),
            'predicted_yield': 4.5,  # tons per hectare
            'confidence_interval': [4.2, 4.8],
            'factors': [
                {'name': 'Soil quality', 'impact': 'high'},
                {'name': 'Rainfall', 'impact': 'medium'},
                {'name': 'Temperature', 'impact': 'low'}
            ]
        })
    except Exception as e:
        logger.error(f"Error predicting crop yield: {e}")
        return jsonify({'error': f"Error predicting crop yield: {str(e)}"}), 500

def start_api_server():
    """
    Start the API server.
    """
    # Load models
    load_models()
    
    # Start the server
    host = config['service']['host']
    port = config['service']['port']
    debug = config['service']['debug']
    
    logger.info(f"Starting API server on {host}:{port}")
    app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    start_api_server()