# Getting Started with Agricultural AI Platform

## Introduction

Welcome to the Agricultural AI Platform! This guide will help you get started with using the platform for various agricultural AI applications such as crop disease detection, yield prediction, and soil analysis.

## Installation

### Prerequisites

Before installing the platform, ensure you have the following prerequisites:

- Python 3.8 or higher
- pip (Python package manager)
- Git (for version control)
- 8GB RAM or higher (for model training)
- NVIDIA GPU (optional, but recommended for faster model training)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/aoluwar/agricultural-ai-platform.git
   cd agricultural-ai-platform
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   # Using venv
   python -m venv env
   
   # Activate the environment
   # On Windows
   env\Scripts\activate
   # On macOS/Linux
   source env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python src/main.py --debug
   ```
   You should see output indicating that the platform has initialized successfully.

## Basic Usage

### Running the Platform

The platform can be run in different modes:

- **Training mode**: For training AI models on your data
  ```bash
  python src/main.py --mode train --config config/train_config.yaml
  ```

- **Prediction mode**: For making predictions using trained models
  ```bash
  python src/main.py --mode predict --config config/predict_config.yaml
  ```

- **Service mode**: For running the web service (default)
  ```bash
  python src/main.py --mode serve --config config/serve_config.yaml
  ```

### Using the Web Interface

When running in service mode, the platform provides a web interface accessible at `http://localhost:5000`. The interface allows you to:

1. Upload images for crop disease detection
2. Input soil parameters for analysis
3. View and download prediction results
4. Manage and monitor AI models

## Working with Models

### Crop Disease Detection

The platform includes a pre-trained model for detecting diseases in crop images:

```python
from models.crop_disease_model import CropDiseaseModel
import cv2

# Load the model
model = CropDiseaseModel()
model.load('models/saved/crop_disease_model.h5')

# Load an image
image = cv2.imread('path/to/crop_image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Make a prediction
predictions = model.predict(image)

# Get the top prediction
top_prediction = predictions.argmax()
print(f"Predicted disease class: {top_prediction}")
```

### Training Your Own Model

You can train the crop disease model on your own dataset:

```python
from models.crop_disease_model import CropDiseaseModel

# Initialize the model
model = CropDiseaseModel(num_classes=10)  # Adjust based on your dataset
model.build_model()

# Train the model
history = model.train(
    train_dir='data/crop_diseases/train',
    validation_dir='data/crop_diseases/validation',
    epochs=20,
    batch_size=32
)

# Fine-tune the model for better performance
history_ft = model.fine_tune(
    train_dir='data/crop_diseases/train',
    validation_dir='data/crop_diseases/validation',
    epochs=10,
    batch_size=32
)

# Save the model
model.save('models/saved/my_crop_disease_model.h5')
```

## Next Steps

- Explore the sample datasets in the `data/` directory
- Check out the documentation in the `docs/` directory for more detailed information
- Join our community forum to ask questions and share your experiences

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'tensorflow'**
   - Ensure you've activated your virtual environment
   - Try reinstalling the dependencies: `pip install -r requirements.txt`

2. **CUDA/GPU issues**
   - Check your NVIDIA drivers are up to date
   - Ensure you have the correct version of CUDA installed for your TensorFlow version

3. **Memory errors during model training**
   - Reduce the batch size in the training configuration
   - Use a smaller model or reduce input image dimensions

### Getting Help

If you encounter issues not covered in this guide, please:

1. Check the full documentation in the `docs/` directory
2. Open an issue on our GitHub repository
3. Contact us at support@agricultural-ai-platform.example.com

## Contributing

We welcome contributions to the Agricultural AI Platform! Please see our contributing guidelines in `CONTRIBUTING.md` for more information on how to get involved.
