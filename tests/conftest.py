import pytest
import os
import tempfile
import shutil
from pathlib import Path
import torch
import numpy as np
from PIL import Image

from src.model.classifier import CervicalLesionClassifier
from src.services.analysis_service import AnalysisService
from src.validation.validation_service import ValidationService
from src.export.export_service import DataExporter
from src.model.interpretability import ModelInterpreter

@pytest.fixture(scope="session")
def test_data_dir():
    """Create temporary directory for test data"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture(scope="session")
def sample_image(test_data_dir):
    """Create a sample test image"""
    image_path = Path(test_data_dir) / "test_image.png"
    # Create a blank image for testing
    img = Image.new('RGB', (224, 224), color='white')
    img.save(str(image_path))
    return str(image_path)

@pytest.fixture(scope="session")
def model():
    """Initialize model for testing"""
    return CervicalLesionClassifier(num_classes=5)

@pytest.fixture(scope="session")
def analysis_service(model):
    """Initialize analysis service"""
    return AnalysisService(model=model)

@pytest.fixture(scope="session")
def validation_service():
    """Initialize validation service"""
    return ValidationService()

@pytest.fixture(scope="session")
def export_service(test_data_dir):
    """Initialize export service"""
    return DataExporter(output_dir=test_data_dir)

@pytest.fixture(scope="session")
def model_interpreter(model):
    """Initialize model interpreter"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return ModelInterpreter(model=model, device=device)
