#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data Preprocessing Utilities for Agricultural AI Platform

This module provides functions for preprocessing various types of agricultural data,
including image data for crop disease detection, soil data, weather data, and more.
"""

import os
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess a single image for model input.
    
    Args:
        image_path (str): Path to the image file
        target_size (tuple): Target size for resizing (height, width)
        
    Returns:
        numpy.ndarray: Preprocessed image ready for model input
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize the image
    img = cv2.resize(img, target_size)
    
    # Normalize pixel values to [0, 1]
    img = img / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img


def create_image_data_generators(augmentation=True):
    """
    Create image data generators for training and validation.
    
    Args:
        augmentation (bool): Whether to apply data augmentation to training data
        
    Returns:
        tuple: (train_generator, validation_generator)
    """
    if augmentation:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
    else:
        train_datagen = ImageDataGenerator(rescale=1./255)
    
    validation_datagen = ImageDataGenerator(rescale=1./255)
    
    return train_datagen, validation_datagen


def prepare_image_data_generators(train_dir, validation_dir, batch_size=32, target_size=(224, 224), augmentation=True):
    """
    Prepare image data generators from directory structure.
    
    Args:
        train_dir (str): Directory containing training data
        validation_dir (str): Directory containing validation data
        batch_size (int): Batch size for training
        target_size (tuple): Target size for images (height, width)
        augmentation (bool): Whether to apply data augmentation
        
    Returns:
        tuple: (train_generator, validation_generator, class_names)
    """
    train_datagen, validation_datagen = create_image_data_generators(augmentation)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    class_names = list(train_generator.class_indices.keys())
    
    return train_generator, validation_generator, class_names


def preprocess_soil_data(data_path, test_size=0.2, random_state=42, normalize=True):
    """
    Preprocess soil data from CSV file.
    
    Args:
        data_path (str): Path to the CSV file containing soil data
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        normalize (bool): Whether to normalize the features
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names, scaler)
    """
    # Read the data
    df = pd.read_csv(data_path)
    
    # Separate features and target
    if 'soil_type' in df.columns:
        target_col = 'soil_type'
    elif 'fertility' in df.columns:
        target_col = 'fertility'
    else:
        # Assume the last column is the target
        target_col = df.columns[-1]
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    feature_names = X.columns.tolist()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Normalize the features if requested
    scaler = None
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, feature_names, scaler


def preprocess_weather_data(data_path, target_col=None, sequence_length=7, test_size=0.2, random_state=42):
    """
    Preprocess weather data for time series analysis.
    
    Args:
        data_path (str): Path to the CSV file containing weather data
        target_col (str): Name of the target column to predict
        sequence_length (int): Number of time steps to use for each sample
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    # Read the data
    df = pd.read_csv(data_path)
    
    # Convert date column to datetime if it exists
    date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    if date_cols:
        df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])
        df = df.sort_values(by=date_cols[0])
    
    # If target column is not specified, use temperature as default
    if target_col is None:
        temp_cols = [col for col in df.columns if 'temp' in col.lower()]
        if temp_cols:
            target_col = temp_cols[0]
        else:
            # Use the last numeric column as target
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            target_col = numeric_cols[-1]
    
    # Select only numeric columns for features
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Normalize the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(numeric_df)
    
    # Create sequences
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i+sequence_length])
        # Get the target column index
        target_idx = numeric_df.columns.get_loc(target_col)
        y.append(scaled_data[i+sequence_length, target_idx])
    
    X = np.array(X)
    y = np.array(y)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test, scaler


def preprocess_yield_data(data_path, test_size=0.2, random_state=42):
    """
    Preprocess crop yield data.
    
    Args:
        data_path (str): Path to the CSV file containing yield data
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names, scaler)
    """
    # Read the data
    df = pd.read_csv(data_path)
    
    # Identify the target column (yield)
    yield_cols = [col for col in df.columns if 'yield' in col.lower()]
    if yield_cols:
        target_col = yield_cols[0]
    else:
        # Assume the last column is the yield
        target_col = df.columns[-1]
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Handle categorical features
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    feature_names = X.columns.tolist()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Normalize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, feature_names, scaler


def load_and_split_dataset(data_path, target_col=None, test_size=0.2, val_size=0.1, random_state=42):
    """
    Load a dataset from a CSV file and split it into train, validation, and test sets.
    
    Args:
        data_path (str): Path to the CSV file
        target_col (str): Name of the target column
        test_size (float): Proportion of data to use for testing
        val_size (float): Proportion of training data to use for validation
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test, feature_names, scaler)
    """
    # Read the data
    df = pd.read_csv(data_path)
    
    # If target column is not specified, use the last column
    if target_col is None:
        target_col = df.columns[-1]
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Handle categorical features
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    feature_names = X.columns.tolist()
    
    # First split: training + validation vs test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Second split: training vs validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size/(1-test_size), random_state=random_state
    )
    
    # Normalize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names, scaler