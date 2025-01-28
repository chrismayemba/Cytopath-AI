# Cytopath-AI: Advanced Cervical Cancer Screening Assistant

[![Live Demo](https://img.shields.io/badge/demo-live-success)](https://cytopath-ai.lovable.app/)
[![Python Version](https://img.shields.io/badge/python-3.8.10-blue)](https://www.python.org/downloads/release/python-3810/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.1-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Overview

Cytopath-AI is a state-of-the-art AI-powered system for automated cervical cancer screening through cytological smear analysis. Built on advanced deep learning techniques, it assists cytopathologists in detecting and classifying cervical lesions according to the Bethesda System (TBS), significantly improving screening efficiency and accuracy.

🔗 **Live Demo**: [https://cytopath-ai.lovable.app/](https://cytopath-ai.lovable.app/)

### Why Cytopath-AI?

- **Early Detection**: Assists in early identification of cervical abnormalities
- **Improved Accuracy**: Reduces false negatives through AI-assisted screening
- **Efficiency**: Automates routine screening tasks, allowing pathologists to focus on complex cases
- **Standardization**: Provides consistent classification according to the Bethesda System
- **Accessibility**: Web-based platform accessible from any modern browser

## Features

### Core Capabilities
- Deep learning-based cell classification using EfficientNet-B4 with attention mechanisms
- Advanced cell segmentation with SIPaKMeD mask integration
- Multi-class classification according to Bethesda categories:
  - NILM (Negative for Intraepithelial Lesion or Malignancy)
  - LSIL (Low-grade Squamous Intraepithelial Lesion)
  - HSIL (High-grade Squamous Intraepithelial Lesion)
  - Squamous Cell Carcinoma
  - Other abnormalities (ASC-US, ASC-H, etc.)

### Advanced Features
- **Model Interpretability**:
  - Integrated Gradients for feature attribution
  - Guided GradCAM for visual explanations
  - Occlusion-based attribution maps
- **Quality Assurance**:
  - Automated image quality assessment
  - Cell detection confidence scores
  - Uncertainty estimation
- **Workflow Integration**:
  - Batch processing capabilities
  - Automated reporting
  - Manual validation interface
  - Integration with laboratory systems

## Technical Stack

### Backend
- Python 3.8.10
- PyTorch 2.1.1 with EfficientNet-B4
- FastAPI for high-performance API
- OpenCV & Albumentations for image processing
- scikit-learn for ML operations
- Captum for model interpretability

### Storage & Database
- PostgreSQL for metadata and analysis results
- MongoDB for cell regions and features
- Efficient caching system

### Development & Testing
- pytest for comprehensive testing
- Weights & Biases for experiment tracking
- Black & isort for code formatting
- Comprehensive logging system

## Project Structure

```
cervical_lesion_detection/
├── src/
│   ├── model/
│   │   ├── classifier.py         # EfficientNet-B4 with attention
│   │   └── interpretability.py   # Model interpretation
│   ├── services/
│   │   └── analysis_service.py   # Image analysis service
│   ├── preprocessing/
│   │   └── cell_segmentation.py  # Advanced cell detection
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

## Deployment

The application is deployed and accessible at [https://cytopath-ai.lovable.app/](https://cytopath-ai.lovable.app/)

### Deployment Features
- Secure HTTPS encryption
- Load balancing for high availability
- Automated backups
- Monitoring and alerting
- Regular security updates

### Performance Metrics
- Average inference time: <500ms
- 99.9% uptime guarantee
- Supports concurrent analysis of multiple samples
- Automatic scaling based on demand

## Setup for Local Development

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

6. Start the development server:
```bash
python src/main.py
```

## API Documentation

The API documentation is available at:
- Swagger UI: [https://cytopath-ai.lovable.app/docs](https://cytopath-ai.lovable.app/docs)
- ReDoc: [https://cytopath-ai.lovable.app/redoc](https://cytopath-ai.lovable.app/redoc)

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use Cytopath-AI in your research, please cite:

```bibtex
@software{cytopath_ai_2024,
  author = {Mayemba, Chris},
  title = {Cytopath-AI: Advanced Cervical Cancer Screening Assistant},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/chrismayemba/Cytopath-AI}
}
```

## Contact

For questions and support, please [open an issue](https://github.com/chrismayemba/Cytopath-AI/issues) or contact the maintainers.
