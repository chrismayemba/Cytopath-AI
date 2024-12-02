import logging
import torch
import pathlib
from PIL import Image
import random
from src.model.classifier import CervicalLesionClassifier
from src.model.interpretability import ModelInterpreter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_test_image():
    """Load a random test image"""
    base_dir = pathlib.Path(__file__).parent / "test_data"
    categories = ["NILM", "LSIL", "HSIL", "SCC", "OTHER"]
    
    # Select random category and image
    category = random.choice(categories)
    category_dir = base_dir / category
    image_files = list(category_dir.glob("*.jpg"))
    
    if not image_files:
        raise FileNotFoundError(f"No test images found in {category_dir}")
    
    image_path = random.choice(image_files)
    logger.info(f"Loading test image: {image_path}")
    logger.info(f"True category: {category}")
    
    return Image.open(image_path), category

def test_model_loading():
    """Test model loading"""
    try:
        model = CervicalLesionClassifier()
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return None

def test_preprocessing(model):
    """Test image preprocessing"""
    try:
        image, _ = load_test_image()
        image_tensor = model.preprocess_image(image)
        logger.info(f"Image preprocessed successfully, shape: {image_tensor.shape}")
        return image_tensor
    except Exception as e:
        logger.error(f"Failed to preprocess image: {str(e)}")
        return None

def test_prediction(model, image_tensor):
    """Test model prediction"""
    try:
        prediction = model.predict(image_tensor)
        logger.info("Prediction successful:")
        logger.info(f"Class: {prediction['class_name']}")
        logger.info(f"Confidence: {prediction['confidence']:.2f}")
        logger.info(f"Class ID: {prediction['class_id']}")
        return prediction
    except Exception as e:
        logger.error(f"Failed to make prediction: {str(e)}")
        return None

def test_interpretability(model, image_tensor):
    """Test model interpretability"""
    try:
        interpreter = ModelInterpreter(model, device=model.device)
        saliency_maps = interpreter.generate_saliency_map(
            image_tensor,
            target_class=0  # NILM class
        )
        logger.info("Generated saliency maps successfully")
        return saliency_maps
    except Exception as e:
        logger.error(f"Failed to generate saliency maps: {str(e)}")
        return None

def main():
    """Main test function"""
    logger.info("Starting system test...")
    
    # Test model loading
    model = test_model_loading()
    if model is None:
        return
    
    # Test preprocessing
    image_tensor = test_preprocessing(model)
    if image_tensor is None:
        return
    
    # Test prediction
    prediction = test_prediction(model, image_tensor)
    if prediction is None:
        return
    
    # Test interpretability
    saliency_maps = test_interpretability(model, image_tensor)
    if saliency_maps is None:
        return
    
    logger.info("All tests completed successfully!")

if __name__ == "__main__":
    main()
