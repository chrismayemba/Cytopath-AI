import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from PIL import Image
from typing import Optional, Callable, Dict, List
import json
import logging

class CervicalDataset(Dataset):
    """Dataset for cervical cell images"""
    
    def __init__(self,
                 data_dir: str,
                 transform: Optional[Callable] = None,
                 class_map: Optional[Dict[str, int]] = None):
        """
        Initialize dataset
        
        Args:
            data_dir: Directory containing images and annotations
            transform: Optional transform to apply to images
            class_map: Optional mapping from class names to indices
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Default class mapping if not provided
        self.class_map = class_map or {
            "NILM": 0,
            "LSIL": 1,
            "HSIL": 2,
            "SCC": 3,
            "Other": 4
        }
        
        # Load dataset
        self.samples = self._load_dataset()
        
        # Log dataset statistics
        self._log_statistics()
    
    def _load_dataset(self) -> List[Dict]:
        """Load dataset samples and annotations"""
        samples = []
        annotation_file = self.data_dir / "annotations.json"
        
        if not annotation_file.exists():
            raise FileNotFoundError(
                f"Annotations file not found at {annotation_file}"
            )
        
        # Load annotations
        with open(annotation_file) as f:
            annotations = json.load(f)
        
        # Process each annotation
        for image_name, anno in annotations.items():
            image_path = self.data_dir / "images" / image_name
            
            if not image_path.exists():
                logging.warning(f"Image not found: {image_path}")
                continue
            
            # Verify class label
            if anno['class'] not in self.class_map:
                logging.warning(
                    f"Unknown class {anno['class']} for {image_name}"
                )
                continue
            
            samples.append({
                'image_path': str(image_path),
                'class_name': anno['class'],
                'class_idx': self.class_map[anno['class']],
                'metadata': anno.get('metadata', {})
            })
        
        return samples
    
    def _log_statistics(self):
        """Log dataset statistics"""
        # Count samples per class
        class_counts = {}
        for sample in self.samples:
            class_name = sample['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Log statistics
        logging.info(f"Loaded dataset from {self.data_dir}")
        logging.info(f"Total samples: {len(self.samples)}")
        for class_name, count in class_counts.items():
            logging.info(f"{class_name}: {count} samples")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> tuple:
        """Get a sample from the dataset"""
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        
        # Apply transform if available
        if self.transform:
            image = self.transform(image)
        
        return image, sample['class_idx']
    
    @property
    def class_weights(self) -> torch.Tensor:
        """Calculate class weights for balanced training"""
        class_counts = torch.zeros(len(self.class_map))
        
        for sample in self.samples:
            class_counts[sample['class_idx']] += 1
        
        # Calculate inverse frequency weights
        weights = 1.0 / class_counts
        weights = weights / weights.sum()  # Normalize
        
        return weights
    
    def get_sampler(self) -> torch.utils.data.Sampler:
        """Get weighted sampler for balanced training"""
        sample_weights = torch.zeros(len(self))
        
        for idx, sample in enumerate(self.samples):
            class_idx = sample['class_idx']
            sample_weights[idx] = self.class_weights[class_idx]
        
        return torch.utils.data.WeightedRandomSampler(
            sample_weights,
            len(sample_weights),
            replacement=True
        )
