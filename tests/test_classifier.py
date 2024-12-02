import pytest
import torch
import numpy as np
from PIL import Image

def test_model_initialization(model):
    """Test model initialization"""
    assert model is not None
    assert isinstance(model.base_model, torch.nn.Module)
    assert model.num_classes == 5  # NILM, LSIL, HSIL, SCC, Other

def test_model_forward_pass(model, sample_image):
    """Test model forward pass"""
    # Load and preprocess image
    image = Image.open(sample_image)
    image_tensor = model.preprocess_image(np.array(image))
    
    # Get prediction
    with torch.no_grad():
        output = model(image_tensor)
    
    assert output.shape == (1, 5)  # Batch size 1, 5 classes
    assert torch.is_tensor(output)

def test_model_prediction(model, sample_image):
    """Test model prediction pipeline"""
    # Load image
    image = Image.open(sample_image)
    
    # Get prediction
    prediction = model.predict(np.array(image))
    
    assert isinstance(prediction, dict)
    assert 'class_id' in prediction
    assert 'class_name' in prediction
    assert 'confidence' in prediction
    assert 0 <= prediction['confidence'] <= 1

def test_model_preprocessing(model, sample_image):
    """Test image preprocessing"""
    # Load image
    image = Image.open(sample_image)
    image_array = np.array(image)
    
    # Preprocess
    processed = model.preprocess_image(image_array)
    
    assert torch.is_tensor(processed)
    assert processed.shape[0] == 1  # Batch dimension
    assert processed.shape[1] == 3  # RGB channels
    assert len(processed.shape) == 4  # B, C, H, W

def test_model_device_handling(model):
    """Test model device handling"""
    # Test CPU
    model.to('cpu')
    assert next(model.parameters()).device.type == 'cpu'
    
    # Test GPU if available
    if torch.cuda.is_available():
        model.to('cuda')
        assert next(model.parameters()).device.type == 'cuda'
        model.to('cpu')  # Move back to CPU for other tests
