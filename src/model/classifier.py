import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from typing import Dict, List, Union, Any
import os
import numpy as np
from PIL import Image
import cv2

class CervicalLesionClassifier(nn.Module):
    def __init__(self, num_classes: int = 5):
        """Initialize the model"""
        super().__init__()
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained EfficientNet
        self.base_model = models.efficientnet_b0(pretrained=True)
        
        # Modify classifier
        num_ftrs = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        
        # Move model to device
        self.to(self.device)
        
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.softmax = nn.Softmax(dim=1)
        
        self._num_classes = num_classes
        
    @property
    def num_classes(self) -> int:
        return self._num_classes
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.base_model(x)
        return self.softmax(x)
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input"""
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            elif image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        
        # Add batch dimension if needed
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        return image
    
    def predict(self, image: Union[np.ndarray, torch.Tensor, Image.Image]) -> Dict[str, Any]:
        """Predict the class of an image"""
        self.eval()
        with torch.no_grad():
            # Handle different input types
            if isinstance(image, (np.ndarray, Image.Image)):
                image = self.preprocess_image(image)
            elif isinstance(image, torch.Tensor):
                if len(image.shape) == 3:
                    image = image.unsqueeze(0)
            
            # Get model prediction
            output = self(image)
            probs = torch.softmax(output, dim=1)[0].cpu().numpy()
            
            # Map to class names
            classes = {
                0: "NILM",
                1: "LSIL",
                2: "HSIL",
                3: "Squamous Cell Carcinoma",
                4: "Other Abnormalities"
            }
            
            predictions = {classes[i]: float(prob) 
                         for i, prob in enumerate(probs)}
            
            # Get predicted class and confidence
            pred_class_idx = int(np.argmax(probs))
            pred_class = classes[pred_class_idx]
            
            return {
                'class_name': pred_class,
                'class_id': pred_class_idx,
                'confidence': float(probs[pred_class_idx]),
                'probabilities': predictions
            }

class BethesdaClassifier:
    def __init__(self, model_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CervicalLesionClassifier().to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        self.classes = {
            0: "NILM",
            1: "LSIL",
            2: "HSIL",
            3: "Squamous Cell Carcinoma",
            4: "Other Abnormalities"
        }
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load a trained model"""
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input"""
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            elif image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        
        # Add batch dimension if needed
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        return image.to(self.device)

    def predict(self, image: Union[np.ndarray, torch.Tensor, Image.Image]) -> Dict[str, Any]:
        """Predict the class of an image"""
        self.model.eval()
        with torch.no_grad():
            # Handle different input types
            if isinstance(image, (np.ndarray, Image.Image)):
                image = self.preprocess_image(image)
            elif isinstance(image, torch.Tensor):
                if len(image.shape) == 3:
                    image = image.unsqueeze(0)
                image = image.to(self.device)
            
            # Get model prediction
            output = self.model(image)
            probs = torch.softmax(output, dim=1)[0].cpu().numpy()
            
            # Map to class names
            classes = {
                0: "NILM",
                1: "LSIL",
                2: "HSIL",
                3: "Squamous Cell Carcinoma",
                4: "Other Abnormalities"
            }
            
            predictions = {classes[i]: float(prob) 
                         for i, prob in enumerate(probs)}
            
            # Get predicted class and confidence
            pred_class_idx = int(np.argmax(probs))
            pred_class = classes[pred_class_idx]
            
            return {
                'class_name': pred_class,
                'class_id': pred_class_idx,
                'confidence': float(probs[pred_class_idx]),
                'probabilities': predictions
            }
    
    def predict_batch(self, images: List[Union[np.ndarray, torch.Tensor]],
                     batch_size: int = 32) -> List[Dict[str, Any]]:
        """Predict classes for a batch of images"""
        self.model.eval()
        predictions = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_tensors = []
            
            for image in batch:
                if isinstance(image, np.ndarray):
                    image = self.preprocess_image(image)
                batch_tensors.append(image)
            
            batch_input = torch.cat(batch_tensors, dim=0)
            
            with torch.no_grad():
                output = self.model(batch_input)
                probs = torch.softmax(output, dim=1).cpu().numpy()
                
                for prob in probs:
                    pred_dict = {self.classes[i]: float(p) 
                               for i, p in enumerate(prob)}
                    pred_class_idx = int(np.argmax(prob))
                    pred_class = self.classes[pred_class_idx]
                    
                    predictions.append({
                        'class_name': pred_class,
                        'class_id': pred_class_idx,
                        'confidence': float(prob[pred_class_idx]),
                        'probabilities': pred_dict
                    })
        
        return predictions
    
    def monte_carlo_dropout(self, image: Union[np.ndarray, torch.Tensor],
                          n_iterations: int = 10) -> Dict:
        """Estimate prediction uncertainty using MC Dropout"""
        if isinstance(image, np.ndarray):
            image = self.preprocess_image(image)
        
        self.model.train()  # Enable dropout
        predictions = []
        
        for _ in range(n_iterations):
            with torch.no_grad():
                output = self.model(image)
                probs = torch.softmax(output, dim=1)[0].cpu().numpy()
                predictions.append(probs)
        
        # Calculate mean and standard deviation
        mean_probs = np.mean(predictions, axis=0)
        std_probs = np.std(predictions, axis=0)
        
        # Get prediction and uncertainty for each class
        uncertainty = {
            self.classes[i]: {
                'mean_probability': float(mean),
                'uncertainty': float(std)
            }
            for i, (mean, std) in enumerate(zip(mean_probs, std_probs))
        }
        
        # Get overall prediction
        pred_class_idx = int(np.argmax(mean_probs))
        pred_class = self.classes[pred_class_idx]
        
        return {
            'class_name': pred_class,
            'class_id': pred_class_idx,
            'confidence': float(mean_probs[pred_class_idx]),
            'uncertainty': uncertainty[pred_class]['uncertainty'],
            'class_uncertainties': uncertainty
        }
    
    def save_model(self, path: str):
        """Save model weights"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
    
    def train_step(self, images: torch.Tensor, labels: torch.Tensor,
                  optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
        
        return {
            'loss': loss.item(),
            'accuracy': correct / total
        }
    
    def evaluate(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Evaluate model on a dataset"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        class_correct = {class_name: 0 for class_name in self.classes.values()}
        class_total = {class_name: 0 for class_name in self.classes.values()}
        
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                total_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Per-class accuracy
                for i in range(len(labels)):
                    label = labels[i].item()
                    class_name = self.classes[label]
                    class_total[class_name] += 1
                    if predicted[i] == label:
                        class_correct[class_name] += 1
        
        # Calculate metrics
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        class_accuracies = {
            class_name: class_correct[class_name] / class_total[class_name]
            if class_total[class_name] > 0 else 0
            for class_name in self.classes.values()
        }
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'class_accuracies': class_accuracies
        }
