import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from sklearn.cluster import KMeans

@dataclass
class CellRegion:
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    contour: np.ndarray
    area: float
    circularity: float
    mean_intensity: float
    mask: Optional[np.ndarray] = None

class CellSegmenter:
    def __init__(self):
        self.min_cell_size = 100  # minimum cell area in pixels
        self.max_cell_size = 5000  # maximum cell area in pixels
        
    def preprocess_image(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Preprocess image for cell segmentation with optional mask"""
        # Apply mask if provided
        if mask is not None:
            if len(mask.shape) == 2:
                mask = mask[:, :, np.newaxis]
            image = cv2.bitwise_and(image, image, mask=mask)
        
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Split channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # Merge channels back
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        return enhanced

    def color_segmentation(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Segment cells based on color using K-means clustering"""
        # Apply mask if provided
        if mask is not None:
            image = cv2.bitwise_and(image, image, mask=mask)
        
        # Reshape image for K-means
        pixels = image.reshape((-1, 3))
        pixels = np.float32(pixels)
        
        # Apply K-means
        k = 3  # number of clusters
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert back to uint8
        centers = np.uint8(centers)
        segmented = centers[labels.flatten()]
        segmented = segmented.reshape(image.shape)
        
        return segmented

    def detect_cells(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> List[CellRegion]:
        """Detect and segment individual cells with optional mask"""
        # Preprocess image
        preprocessed = self.preprocess_image(image, mask)
        
        # If mask is provided, use it directly for cell detection
        if mask is not None:
            binary = mask
        else:
            # Color segmentation
            segmented = self.color_segmentation(preprocessed)
            
            # Convert to grayscale
            gray = cv2.cvtColor(segmented, cv2.COLOR_RGB2GRAY)
            
            # Threshold
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Morphological operations
            kernel = np.ones((3,3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and analyze cell regions
        cell_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by size
            if self.min_cell_size <= area <= self.max_cell_size:
                # Calculate circularity
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Create cell mask
                cell_mask = np.zeros(binary.shape, np.uint8)
                cv2.drawContours(cell_mask, [contour], -1, 255, -1)
                
                # Calculate mean intensity in the original image
                mean_intensity = cv2.mean(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), mask=cell_mask)[0]
                
                cell_regions.append(CellRegion(
                    bbox=(x, y, w, h),
                    contour=contour,
                    area=area,
                    circularity=circularity,
                    mean_intensity=mean_intensity,
                    mask=cell_mask
                ))
        
        return cell_regions

    def extract_cell_features(self, image: np.ndarray, region: CellRegion) -> np.ndarray:
        """Extract features from a cell region"""
        # Extract cell using mask
        if region.mask is not None:
            cell = cv2.bitwise_and(image, image, mask=region.mask)
        else:
            x, y, w, h = region.bbox
            cell = image[y:y+h, x:x+w]
        
        # Resize to standard size
        cell = cv2.resize(cell, (64, 64))
        
        return cell
