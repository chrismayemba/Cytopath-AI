# AI-Assisted Cervical Lesion Detection

An application for automated detection and classification of cervical lesions from cytological smears using the Bethesda System (TBS).

## Features

- Deep learning-based cell classification using EfficientNet-B0
- Automated detection and segmentation of individual cells
- Multi-class classification according to Bethesda categories:
  - NILM (Negative for Intraepithelial Lesion or Malignancy)
  - LSIL (Low-grade Squamous Intraepithelial Lesion)
  - HSIL (High-grade Squamous Intraepithelial Lesion)
  - Squamous Cell Carcinoma
  - Other abnormalities (ASC-US, ASC-H, etc.)
- Model interpretability with:
  - Integrated Gradients
  - Guided GradCAM
  - Occlusion-based attribution
- Interactive visualization of results
- Batch processing capabilities
- Comprehensive error handling and logging
- Automated test suite
- Report generation
- Manual validation interface for cytopathologists

## Technical Stack

### Backend
- Python 3.8.10
- PyTorch 2.1.1 (Deep Learning)
- torchvision (Image Processing)
- FastAPI (Web API)
- OpenCV (Image Processing)
- scikit-learn (Machine Learning)
- Captum (Model Interpretability)
- NumPy & Pandas (Data Processing)

### Storage
- PostgreSQL (metadata and analysis results)
- MongoDB (cell regions and features)

### Development
- pytest (Testing)
- tqdm (Progress Bars)
- logging (Error Tracking)
- black & isort (Code Formatting)

## Project Structure

```
cervical_lesion_detection/
├── src/
│   ├── model/
│   │   ├── classifier.py         # Deep learning model
│   │   └── interpretability.py   # Model interpretation
│   ├── services/
│   │   └── analysis_service.py   # Image analysis service
│   ├── preprocessing/
│   │   └── cell_segmentation.py  # Cell detection
│   ├── database/
│   │   └── config.py            # Database configuration
│   └── main.py                  # FastAPI application
├── tests/
│   ├── conftest.py              # Test configuration
│   ├── test_classifier.py       # Model tests
│   ├── test_analysis_service.py # Service tests
│   └── test_interpretability.py # Interpretation tests
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Setup

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
cp .env.example .env
```

4. Configure the databases:
```bash
# PostgreSQL
createdb cervical_lesion_detection
psql -d cervical_lesion_detection -f schema.sql

# MongoDB
mongod --dbpath /path/to/data
```

5. Run the tests:
```bash
pytest --cov=src
```

6. Start the service:
```bash
python src/main.py
```

## Model Architecture

The system uses EfficientNet-B0 as the base model, with custom modifications:
- Additional dropout layers for regularization
- Custom classifier head with 512 -> 256 -> num_classes architecture
- Softmax activation for probability outputs

## API Endpoints

### POST /analyze
Analyze a single cervical smear image
```json
{
  "image_path": "path/to/image.jpg",
  "metadata": {
    "patient_id": "string",
    "sample_date": "2024-01-20"
  }
}
```

### POST /batch
Process multiple images
```json
{
  "image_paths": ["path1.jpg", "path2.jpg"],
  "batch_size": 32
}
```

### GET /result/{analysis_id}
Retrieve analysis results
```json
{
  "analysis_id": "uuid",
  "classification": "NILM",
  "confidence": 0.95,
  "num_cells": 150
}
```

## Development

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/test_classifier.py
```

### Code Style
```bash
# Format code
black src tests

# Sort imports
isort src tests
```

## Contributing

**Note: This is a private repository. Access is restricted to authorized collaborators only.**

1. Fork the repository (if you have access)
2. Create a feature branch
3. Make your changes
4. Run the tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
