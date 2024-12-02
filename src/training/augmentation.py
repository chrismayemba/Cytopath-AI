import torchvision.transforms as T
from typing import Dict, List, Tuple
import random
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter

class CervicalAugmentation:
    """Custom augmentation pipeline for cervical cell images"""
    
    def __init__(self, config: Dict):
        """
        Initialize augmentation pipeline
        
        Args:
            config: Augmentation configuration from training config
        """
        self.config = config
        self.base_transform = self._create_base_transform()
        self.aug_transform = self._create_aug_transform()
    
    def _create_base_transform(self) -> T.Compose:
        """Create base transformation pipeline"""
        return T.Compose([
            T.Resize(self.config['image_size']),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _create_aug_transform(self) -> T.Compose:
        """Create augmentation pipeline"""
        aug_list = [
            # Geometric transformations
            T.RandomHorizontalFlip(p=self.config['random_horizontal_flip']),
            T.RandomVerticalFlip(p=self.config['random_vertical_flip']),
            T.RandomRotation(self.config['random_rotation']),
            
            # Color augmentations
            T.ColorJitter(
                brightness=self.config['color_jitter']['brightness'],
                contrast=self.config['color_jitter']['contrast'],
                saturation=self.config['color_jitter']['saturation'],
                hue=self.config['color_jitter']['hue']
            ),
            
            # Custom augmentations
            self.RandomGaussianBlur(p=0.2),
            self.RandomGaussianNoise(p=0.2),
            self.RandomStaining(p=0.3)
        ]
        
        return T.Compose([
            T.Resize(self.config['image_size']),
            *aug_list,
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    class RandomGaussianBlur:
        """Random Gaussian blur augmentation"""
        def __init__(self, p: float = 0.5):
            self.p = p
        
        def __call__(self, img: Image.Image) -> Image.Image:
            if random.random() < self.p:
                radius = random.uniform(0.5, 1.5)
                return img.filter(ImageFilter.GaussianBlur(radius))
            return img
    
    class RandomGaussianNoise:
        """Random Gaussian noise augmentation"""
        def __init__(self, p: float = 0.5, mean: float = 0., std: float = 0.1):
            self.p = p
            self.mean = mean
            self.std = std
        
        def __call__(self, img: Image.Image) -> Image.Image:
            if random.random() < self.p:
                img_np = np.array(img).astype(np.float32)
                noise = np.random.normal(self.mean, self.std, img_np.shape)
                noisy_img = np.clip(img_np + noise * 255, 0, 255).astype(np.uint8)
                return Image.fromarray(noisy_img)
            return img
    
    class RandomStaining:
        """Random staining variation augmentation"""
        def __init__(self, p: float = 0.5):
            self.p = p
        
        def __call__(self, img: Image.Image) -> Image.Image:
            if random.random() < self.p:
                # Randomly adjust H&E staining appearance
                enhancer = ImageEnhance.Color(img)
                img = enhancer.enhance(random.uniform(0.8, 1.2))
                
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(random.uniform(0.8, 1.2))
                
                return img
            return img
    
    def __call__(self, img: Image.Image, training: bool = True) -> torch.Tensor:
        """Apply transformation pipeline"""
        if training:
            return self.aug_transform(img)
        return self.base_transform(img)
    
    @staticmethod
    def get_validation_transform(image_size: Tuple[int, int]) -> T.Compose:
        """Get validation/test transformation pipeline"""
        return T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
