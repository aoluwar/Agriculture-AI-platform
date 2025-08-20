#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test suite for the Crop Disease Detection Model

This module contains unit tests for the CropDiseaseModel class.
"""

import unittest
import numpy as np
import os
import sys

# Add the parent directory to the path so we can import the model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the model to test
from models.crop_disease_model import CropDiseaseModel


class TestCropDiseaseModel(unittest.TestCase):
    """Test cases for the CropDiseaseModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = CropDiseaseModel(num_classes=5, input_shape=(224, 224, 3))
    
    def test_model_initialization(self):
        """Test that the model initializes correctly."""
        self.assertEqual(self.model.num_classes, 5)
        self.assertEqual(self.model.input_shape, (224, 224, 3))
        self.assertIsNone(self.model.model)
    
    def test_build_model(self):
        """Test that the model builds correctly."""
        model = self.model.build_model()
        
        # Check that the model is not None
        self.assertIsNotNone(model)
        
        # Check the output shape
        self.assertEqual(model.output_shape, (None, 5))
        
        # Check that the model is compiled
        self.assertIsNotNone(model.optimizer)
        self.assertIsNotNone(model.loss)
    
    def test_predict_without_training(self):
        """Test that prediction without training raises an error."""
        # Create a dummy image
        image = np.random.rand(224, 224, 3)
        
        # Prediction should raise ValueError because model is not trained
        with self.assertRaises(ValueError):
            self.model.predict(image)
    
    def test_save_without_training(self):
        """Test that saving without training raises an error."""
        with self.assertRaises(ValueError):
            self.model.save('dummy_path.h5')
    
    def test_predict_after_building(self):
        """Test prediction after building the model."""
        # Build the model
        self.model.build_model()
        
        # Create a dummy image
        image = np.random.rand(224, 224, 3)
        
        # Make a prediction
        predictions = self.model.predict(image)
        
        # Check the shape of the predictions
        self.assertEqual(predictions.shape, (1, 5))
        
        # Check that the predictions sum to approximately 1
        self.assertAlmostEqual(np.sum(predictions), 1.0, places=5)
    
    def test_model_save_and_load(self):
        """Test saving and loading the model."""
        # Skip this test if we're not in a writable environment
        if not os.access('.', os.W_OK):
            self.skipTest("No write access to current directory")
        
        # Build the model
        self.model.build_model()
        
        # Save the model to a temporary file
        temp_model_path = 'temp_model.h5'
        self.model.save(temp_model_path)
        
        # Check that the file exists
        self.assertTrue(os.path.exists(temp_model_path))
        
        # Create a new model instance
        new_model = CropDiseaseModel(num_classes=5)
        
        # Load the model
        new_model.load(temp_model_path)
        
        # Check that the model is loaded
        self.assertIsNotNone(new_model.model)
        
        # Clean up
        os.remove(temp_model_path)


if __name__ == '__main__':
    unittest.main()