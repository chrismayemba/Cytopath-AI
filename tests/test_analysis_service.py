import pytest
from pathlib import Path
import numpy as np
from PIL import Image

def test_analysis_service_initialization(analysis_service):
    """Test analysis service initialization"""
    assert analysis_service is not None
    assert analysis_service.classifier is not None

def test_analyze_image(analysis_service, sample_image):
    """Test image analysis"""
    # Analyze image
    result = analysis_service.analyze_image(sample_image)
    
    # Check result structure
    assert isinstance(result, dict)
    assert 'analysis_id' in result
    assert 'image_id' in result
    assert 'classification' in result
    assert 'confidence' in result
    assert 'cell_regions' in result
    assert isinstance(result['cell_regions'], list)

def test_cell_detection(analysis_service, sample_image):
    """Test cell detection"""
    # Load image
    image = np.array(Image.open(sample_image))
    
    # Detect cells
    cells = analysis_service.detect_cells(image)
    
    assert isinstance(cells, list)
    if len(cells) > 0:
        assert 'coordinates' in cells[0]
        assert 'confidence' in cells[0]
        assert 'region_id' in cells[0]

def test_batch_analysis(analysis_service, test_data_dir):
    """Test batch analysis"""
    # Create multiple test images
    image_paths = []
    for i in range(3):
        img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        path = Path(test_data_dir) / f"test_image_{i}.png"
        img.save(str(path))
        image_paths.append(str(path))
    
    # Run batch analysis
    results = analysis_service.analyze_batch(image_paths)
    
    assert len(results) == len(image_paths)
    for result in results:
        assert isinstance(result, dict)
        assert 'analysis_id' in result
        assert 'status' in result

def test_error_handling(analysis_service):
    """Test error handling"""
    # Test with non-existent image
    with pytest.raises(FileNotFoundError):
        analysis_service.analyze_image("nonexistent.png")
    
    # Test with invalid image path
    with pytest.raises(Exception):
        analysis_service.analyze_image(None)

def test_result_storage(analysis_service, sample_image):
    """Test analysis result storage"""
    # Analyze image
    result = analysis_service.analyze_image(sample_image)
    
    # Retrieve stored result
    stored_result = analysis_service.get_analysis_result(result['analysis_id'])
    
    assert stored_result is not None
    assert stored_result['analysis_id'] == result['analysis_id']
    assert stored_result['image_id'] == result['image_id']
