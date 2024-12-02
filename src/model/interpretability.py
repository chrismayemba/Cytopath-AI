import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Union
import cv2
from captum.attr import (
    IntegratedGradients,
    GuidedGradCam,
    Occlusion,
    LayerGradCam,
    LayerAttribution
)
from PIL import Image
import matplotlib.pyplot as plt
import io

class ModelInterpreter:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.integrated_gradients = IntegratedGradients(model)
        self.guided_gradcam = GuidedGradCam(model, model.base_model.features[-1])
        self.occlusion = Occlusion(model)
        
    def _normalize_attribution(self, attribution: np.ndarray) -> np.ndarray:
        """Normalize attribution scores to [0, 1] range"""
        if attribution.min() == attribution.max():
            return np.zeros_like(attribution)
            
        return (attribution - attribution.min()) / (attribution.max() - attribution.min())
    
    def generate_saliency_map(self,
                            image: torch.Tensor,
                            target_class: int) -> Dict[str, np.ndarray]:
        """Generate saliency maps using multiple interpretation methods"""
        self.model.eval()
        image = image.to(self.device)
        image.requires_grad = True

        # Integrated Gradients
        ig_attr = self.integrated_gradients.attribute(
            image,
            target=target_class,
            n_steps=50
        )
        ig_attr = ig_attr.detach().cpu().numpy().squeeze()
        ig_attr = np.transpose(ig_attr, (1, 2, 0))
        ig_attr = np.mean(np.abs(ig_attr), axis=2)
        ig_attr = self._normalize_attribution(ig_attr)

        # Guided GradCAM
        gradcam_attr = self.guided_gradcam.attribute(
            image,
            target=target_class
        )
        gradcam_attr = gradcam_attr.detach().cpu().numpy().squeeze()
        gradcam_attr = np.transpose(gradcam_attr, (1, 2, 0))
        gradcam_attr = np.mean(np.abs(gradcam_attr), axis=2)
        gradcam_attr = self._normalize_attribution(gradcam_attr)

        # Occlusion
        occlusion_attr = self.occlusion.attribute(
            image,
            target=target_class,
            sliding_window_shapes=(3, 15, 15)
        )
        occlusion_attr = occlusion_attr.detach().cpu().numpy().squeeze()
        occlusion_attr = np.transpose(occlusion_attr, (1, 2, 0))
        occlusion_attr = np.mean(np.abs(occlusion_attr), axis=2)
        occlusion_attr = self._normalize_attribution(occlusion_attr)

        return {
            'integrated_gradients': ig_attr,
            'guided_gradcam': gradcam_attr,
            'occlusion': occlusion_attr
        }
    
    def analyze_cell_regions(self,
                           image: torch.Tensor,
                           cell_regions: List[Dict]) -> List[Dict]:
        """Analyze individual cell regions"""
        results = []
        
        for region in cell_regions:
            # Extract cell region
            x, y, w, h = region['coordinates'].values()
            cell_img = image[:, :, y:y+h, x:x+w]
            
            # Get model prediction
            with torch.no_grad():
                output = self.model(cell_img)
                probs = F.softmax(output, dim=1)
                pred_class = torch.argmax(probs, dim=1).item()
                confidence = probs[0, pred_class].item()
            
            # Generate saliency maps
            saliency_maps = self.generate_saliency_map(cell_img, pred_class)
            
            results.append({
                'region_id': region['region_id'],
                'prediction': {
                    'class': pred_class,
                    'confidence': confidence
                },
                'saliency_maps': saliency_maps
            })
        
        return results
    
    def visualize_interpretations(self,
                                image: np.ndarray,
                                saliency_maps: Dict[str, np.ndarray],
                                output_path: str = None) -> Union[str, None]:
        """Create visualization of different interpretation methods"""
        # Ensure image is RGB and in correct size
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        # Resize image if needed
        target_size = (224, 224)
        if image.shape[:2] != target_size:
            image = cv2.resize(image, target_size)
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Saliency maps
        for idx, (method, smap) in enumerate(saliency_maps.items()):
            if idx >= 3:  # Only show first 3 methods
                break
                
            row = (idx + 1) // 2
            col = (idx + 1) % 2
            
            # Resize saliency map if needed
            if smap.shape != target_size:
                smap = cv2.resize(smap, target_size)
                
            # Create heatmap overlay
            heatmap = cv2.applyColorMap(
                (smap * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            # Ensure image and heatmap have same dimensions
            if image.shape != heatmap.shape:
                heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
                
            # Overlay on original image
            overlay = cv2.addWeighted(
                image,
                0.7,
                heatmap,
                0.3,
                0
            )
            
            axes[row, col].imshow(overlay)
            axes[row, col].set_title(method.replace('_', ' ').title())
            axes[row, col].axis('off')
            
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            return output_path
        else:
            return None

    def get_feature_importance(self,
                             cell_features: Dict[str, float]) -> Dict[str, float]:
        """Analyze importance of extracted cell features"""
        # Use SHAP values or similar methods to analyze feature importance
        # This is a placeholder implementation
        feature_importance = {}
        total = sum(cell_features.values())
        
        if total > 0:
            for feature, value in cell_features.items():
                feature_importance[feature] = abs(value) / total
        
        return feature_importance
