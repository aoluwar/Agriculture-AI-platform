#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualization Utilities for Agricultural AI Platform

This module provides functions for visualizing various types of agricultural data,
including crop disease detection results, soil analysis, weather data, and yield predictions.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import cv2
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import Model


def plot_training_history(history, save_path=None):
    """
    Plot training and validation metrics from model training history.
    
    Args:
        history: Keras history object or dictionary containing training metrics
        save_path (str, optional): Path to save the plot image
    """
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='lower right')
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='upper right')
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None):
    """
    Plot confusion matrix for classification results.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names (list, optional): List of class names
        save_path (str, optional): Path to save the plot image
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize the confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot confusion matrix
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_roc_curve(y_true, y_score, n_classes=None, save_path=None):
    """
    Plot ROC curve for multi-class classification.
    
    Args:
        y_true: True labels (one-hot encoded for multi-class)
        y_score: Predicted probabilities
        n_classes (int, optional): Number of classes
        save_path (str, optional): Path to save the plot image
    """
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    if n_classes is None:
        n_classes = y_score.shape[1]
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot micro-average ROC curve
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'Micro-average ROC curve (area = {roc_auc["micro"]:.2f})',
             color='deeppink', linestyle=':', linewidth=4)
    
    # Plot ROC curve for each class
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, n_classes))
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {i} (area = {roc_auc[i]:.2f})')
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_crop_disease_prediction(image_path, predictions, class_names, save_path=None):
    """
    Visualize crop disease prediction results.
    
    Args:
        image_path (str): Path to the input image
        predictions (numpy.ndarray): Model predictions (probabilities)
        class_names (list): List of class names
        save_path (str, optional): Path to save the visualization
    """
    # Load and resize image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Get top predictions
    top_indices = np.argsort(predictions)[-3:][::-1]
    top_predictions = [(class_names[i], predictions[i] * 100) for i in top_indices]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Display image
    ax1.imshow(img)
    ax1.set_title('Input Image')
    ax1.axis('off')
    
    # Display prediction results
    bars = ax2.barh([class_names[i] for i in top_indices], [predictions[i] * 100 for i in top_indices])
    
    # Add percentage labels to bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax2.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%',
                 ha='left', va='center', fontweight='bold')
    
    ax2.set_title('Top Predictions')
    ax2.set_xlabel('Confidence (%)')
    ax2.set_xlim(0, 100)
    ax2.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_feature_importance(feature_names, importances, title='Feature Importance', save_path=None):
    """
    Visualize feature importance for machine learning models.
    
    Args:
        feature_names (list): Names of the features
        importances (numpy.ndarray): Importance scores for each feature
        title (str): Plot title
        save_path (str, optional): Path to save the visualization
    """
    # Sort features by importance
    indices = np.argsort(importances)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot feature importance
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importance')
    plt.title(title)
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_soil_analysis_results(soil_data, save_path=None):
    """
    Visualize soil analysis results.
    
    Args:
        soil_data (dict): Dictionary containing soil analysis results
        save_path (str, optional): Path to save the visualization
    """
    # Extract soil properties and values
    properties = list(soil_data.keys())
    values = list(soil_data.values())
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define colors based on optimal ranges (simplified example)
    colors = []
    for prop, val in soil_data.items():
        if prop == 'pH':
            if 6.0 <= val <= 7.5:
                colors.append('green')
            else:
                colors.append('orange')
        elif prop in ['N', 'P', 'K']:
            if val >= 50:
                colors.append('green')
            else:
                colors.append('orange')
        else:
            colors.append('blue')
    
    # Plot soil properties
    bars = ax.bar(properties, values, color=colors)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}', ha='center', va='bottom')
    
    ax.set_title('Soil Analysis Results')
    ax.set_ylabel('Value')
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_weather_forecast(forecast_data, save_path=None):
    """
    Visualize weather forecast data.
    
    Args:
        forecast_data (list): List of dictionaries containing forecast data
        save_path (str, optional): Path to save the visualization
    """
    # Convert forecast data to DataFrame
    df = pd.DataFrame(forecast_data)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot temperature
    ax1.plot(df['date'], df['temperature'], 'o-', color='red', linewidth=2, markersize=8)
    ax1.set_title('Temperature Forecast')
    ax1.set_ylabel('Temperature (°C)')
    ax1.grid(True, alpha=0.3)
    
    # Plot precipitation
    bars = ax2.bar(df['date'], df['precipitation'], color='blue', alpha=0.7)
    ax2.set_title('Precipitation Forecast')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Precipitation (mm)')
    ax2.grid(True, axis='y', alpha=0.3)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_yield_prediction(crop_data, predictions, factors=None, save_path=None):
    """
    Visualize crop yield prediction results.
    
    Args:
        crop_data (dict): Dictionary containing crop information
        predictions (dict): Dictionary containing yield predictions
        factors (list, optional): List of dictionaries containing impact factors
        save_path (str, optional): Path to save the visualization
    """
    # Create figure with subplots
    if factors:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    else:
        fig, ax1 = plt.subplots(figsize=(8, 6))
    
    # Plot yield prediction
    ax1.bar(['Predicted Yield'], [predictions['predicted_yield']], color='green', alpha=0.7)
    
    # Add confidence interval
    if 'confidence_interval' in predictions:
        ci_low, ci_high = predictions['confidence_interval']
        ax1.errorbar(['Predicted Yield'], [predictions['predicted_yield']], 
                    yerr=[[predictions['predicted_yield'] - ci_low], [ci_high - predictions['predicted_yield']]],
                    fmt='o', color='black', capsize=10)
    
    ax1.set_title(f"Yield Prediction for {crop_data.get('crop', 'Crop')}")
    ax1.set_ylabel('Yield (tons/hectare)')
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Plot impact factors if provided
    if factors:
        # Extract factor names and impact values
        factor_names = [factor['name'] for factor in factors]
        impact_values = []
        colors = []
        
        for factor in factors:
            if factor['impact'].lower() == 'high':
                impact_values.append(3)
                colors.append('red')
            elif factor['impact'].lower() == 'medium':
                impact_values.append(2)
                colors.append('orange')
            else:  # low
                impact_values.append(1)
                colors.append('green')
        
        # Plot impact factors
        bars = ax2.barh(factor_names, impact_values, color=colors, alpha=0.7)
        
        # Add impact labels
        for i, bar in enumerate(bars):
            ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                    factors[i]['impact'].capitalize(), ha='left', va='center')
        
        ax2.set_title('Impact Factors')
        ax2.set_xlabel('Impact Level')
        ax2.set_xlim(0, 4)
        ax2.set_xticks([1, 2, 3])
        ax2.set_xticklabels(['Low', 'Medium', 'High'])
        ax2.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_model_activations(model, image_path, layer_name, save_path=None):
    """
    Visualize activations of a specific layer in a neural network model.
    
    Args:
        model: Keras model
        image_path (str): Path to the input image
        layer_name (str): Name of the layer to visualize
        save_path (str, optional): Path to save the visualization
    """
    # Load and preprocess image
    img = load_img(image_path, target_size=model.input_shape[1:3])
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Create a model that outputs the activations of the specified layer
    layer_outputs = [layer.output for layer in model.layers if layer.name == layer_name]
    if not layer_outputs:
        raise ValueError(f"Layer '{layer_name}' not found in the model")
    
    activation_model = Model(inputs=model.input, outputs=layer_outputs[0])
    activations = activation_model.predict(img_array)
    
    # Number of features to display
    n_features = min(16, activations.shape[-1])
    
    # Create figure
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()
    
    # Plot original image
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Plot activations
    for i in range(1, n_features):
        axes[i].imshow(activations[0, :, :, i-1], cmap='viridis')
        axes[i].set_title(f'Feature {i}')
        axes[i].axis('off')
    
    # Hide empty subplots
    for i in range(n_features, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Activations of layer: {layer_name}')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_soil_fertility_map(lat_coords, lon_coords, fertility_values, save_path=None):
    """
    Create a soil fertility map using coordinates and fertility values.
    
    Args:
        lat_coords (numpy.ndarray): Latitude coordinates
        lon_coords (numpy.ndarray): Longitude coordinates
        fertility_values (numpy.ndarray): Soil fertility values
        save_path (str, optional): Path to save the visualization
    """
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create custom colormap for fertility (red to green)
    cmap = LinearSegmentedColormap.from_list('fertility', ['red', 'yellow', 'green'])
    
    # Create scatter plot
    scatter = plt.scatter(lon_coords, lat_coords, c=fertility_values, 
                         cmap=cmap, s=100, alpha=0.7, edgecolors='black')
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Soil Fertility')
    
    plt.title('Soil Fertility Map')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()