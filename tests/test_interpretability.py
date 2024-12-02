import pytest
import torch
import numpy as np
from PIL import Image

def test_interpreter_initialization(model_interpreter):
    """Test model interpreter initialization"""
    assert model_interpreter is not None
    assert model_interpreter.model is not None
    assert model_interpreter.integrated_gradients is not None
    assert model_interpreter.guided_gradcam is not None

def test_saliency_map_generation(model_interpreter, sample_image):
    """Test saliency map generation"""
    # Load and preprocess image
    image = Image.open(sample_image)
    image_tensor = model_interpreter.model.preprocess_image(np.array(image))
    
    # Generate saliency maps
    saliency_maps = model_interpreter.generate_saliency_map(
        image_tensor,
        target_class=0
    )
    
    assert isinstance(saliency_maps, dict)
    assert 'integrated_gradients' in saliency_maps
    assert 'guided_gradcam' in saliency_maps
    assert 'occlusion' in saliency_maps
    
    # Check map properties
    for method, smap in saliency_maps.items():
        assert isinstance(smap, np.ndarray)
        assert smap.shape == (512, 512)  # Assuming input size
        assert smap.min() >= 0
        assert smap.max() <= 1

def test_cell_region_analysis(model_interpreter, sample_image):
    """Test cell region analysis"""
    # Create sample cell regions
    cell_regions = [
        {
            'region_id': '1',
            'coordinates': {'x': 0, 'y': 0, 'w': 128, 'h': 128}
        }
    ]
    
    # Load and preprocess image
    image = Image.open(sample_image)
    image_tensor = model_interpreter.model.preprocess_image(np.array(image))
    
    # Analyze regions
    results = model_interpreter.analyze_cell_regions(image_tensor, cell_regions)
    
    assert isinstance(results, list)
    assert len(results) == len(cell_regions)
    for result in results:
        assert 'region_id' in result
        assert 'prediction' in result
        assert 'saliency_maps' in result

def test_visualization(model_interpreter, sample_image, test_data_dir):
    """Test interpretation visualization"""
    # Load image
    image = np.array(Image.open(sample_image))
    
    # Create sample saliency maps
    saliency_maps = {
        'test_method': np.random.rand(512, 512)
    }
    
    # Generate visualization
    output_path = f"{test_data_dir}/test_vis.png"
    result = model_interpreter.visualize_interpretations(
        image,
        saliency_maps,
        output_path
    )
    
    assert result == output_path
    assert Path(output_path).exists()

def test_feature_importance(model_interpreter):
    """Test feature importance analysis"""
    # Create sample cell features
    cell_features = {
        'area': 1000,
        'perimeter': 100,
        'circularity': 0.8,
        'intensity_mean': 150
    }
    
    # Get feature importance
    importance = model_interpreter.get_feature_importance(cell_features)
    
    assert isinstance(importance, dict)
    assert len(importance) == len(cell_features)
    assert sum(importance.values()) == pytest.approx(1.0)
