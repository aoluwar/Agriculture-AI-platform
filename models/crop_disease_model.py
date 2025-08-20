#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Crop Disease Detection Model

This module implements a convolutional neural network for detecting
diseases in crop images.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class CropDiseaseModel:
    """A model for detecting diseases in crop images."""
    
    def __init__(self, num_classes=38, input_shape=(224, 224, 3)):
        """Initialize the model.
        
        Args:
            num_classes (int): Number of disease classes to predict
            input_shape (tuple): Shape of input images (height, width, channels)
        """
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None
    
    def build_model(self, dropout_rate=0.2):
        """Build the CNN model architecture.
        
        Args:
            dropout_rate (float): Dropout rate for regularization
            
        Returns:
            The compiled Keras model
        """
        # Use MobileNetV2 as base model (efficient for mobile deployment)
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze the base model layers
        base_model.trainable = False
        
        # Create the model
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(1024, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(512, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, train_dir, validation_dir, epochs=20, batch_size=32):
        """Train the model on the provided dataset.
        
        Args:
            train_dir (str): Directory containing training data
            validation_dir (str): Directory containing validation data
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
        
        # Data augmentation for training
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
        
        # Only rescaling for validation
        validation_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create data generators
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        # Train the model
        history = self.model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // batch_size
        )
        
        return history
    
    def fine_tune(self, train_dir, validation_dir, epochs=10, batch_size=32):
        """Fine-tune the model by unfreezing some layers of the base model.
        
        Args:
            train_dir (str): Directory containing training data
            validation_dir (str): Directory containing validation data
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model must be trained before fine-tuning")
        
        # Unfreeze the top layers of the base model
        base_model = self.model.layers[0]
        base_model.trainable = True
        
        # Freeze all the layers except the last 4
        for layer in base_model.layers[:-4]:
            layer.trainable = False
        
        # Recompile the model with a lower learning rate
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Use the same data generators as in train()
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
        
        validation_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        # Fine-tune the model
        history = self.model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // batch_size
        )
        
        return history
    
    def predict(self, image):
        """Make a prediction for a single image.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            Predicted class probabilities
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Preprocess the image
        image = tf.image.resize(image, self.input_shape[:2])
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        
        # Make prediction
        predictions = self.model.predict(image)
        return predictions
    
    def save(self, model_path):
        """Save the model to disk.
        
        Args:
            model_path (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save(model_path)
    
    def load(self, model_path):
        """Load a model from disk.
        
        Args:
            model_path (str): Path to the saved model
        """
        self.model = models.load_model(model_path)
        return self.model


# Example usage
if __name__ == "__main__":
    print("Crop Disease Detection Model")
    print("This module provides a CNN model for crop disease detection.")
    print("Import and use the CropDiseaseModel class in your application.")