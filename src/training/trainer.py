import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path
import json
import logging
from tqdm import tqdm
import wandb

from ..model.classifier import CervicalLesionClassifier

class CervicalCytologyDataset(Dataset):
    def __init__(self, data_dir: str, split: str = 'train'):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Load annotations
        annotations_file = self.data_dir / f'{split}_annotations.json'
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
            
        self.image_paths = list(self.annotations.keys())
        
        # Class mapping
        self.class_to_idx = {
            "NILM": 0,
            "LSIL": 1,
            "HSIL": 2,
            "Squamous Cell Carcinoma": 3,
            "Other Abnormalities": 4
        }
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        image = Image.open(self.data_dir / img_path).convert('RGB')
        label = self.class_to_idx[self.annotations[img_path]]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class ModelTrainer:
    def __init__(self, 
                 data_dir: str,
                 model_save_dir: str,
                 batch_size: int = 32,
                 num_epochs: int = 100,
                 learning_rate: float = 0.001,
                 use_wandb: bool = True):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CervicalLesionClassifier().to(self.device)
        self.data_dir = Path(data_dir)
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Training parameters
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        
        # Initialize wandb
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project="cervical-lesion-detection")
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize datasets and dataloaders
        self.train_dataset = CervicalCytologyDataset(data_dir, 'train')
        self.val_dataset = CervicalCytologyDataset(data_dir, 'val')
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
        
        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.1,
            patience=5,
            verbose=True
        )
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (images, targets) in enumerate(tqdm(self.train_loader)):
            images, targets = images.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, targets in self.val_loader:
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self):
        """Full training loop"""
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            self.logger.info(f"Epoch {epoch+1}/{self.num_epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch()
            self.logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            
            # Validate
            val_loss, val_acc = self.validate()
            self.logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Log metrics
            if self.use_wandb:
                wandb.log({
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_path = self.model_save_dir / f'best_model.pth'
                torch.save(self.model.state_dict(), model_path)
                self.logger.info(f"Saved best model to {model_path}")
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                checkpoint_path = self.model_save_dir / f'checkpoint_epoch_{epoch+1}.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                }, checkpoint_path)
                self.logger.info(f"Saved checkpoint to {checkpoint_path}")
