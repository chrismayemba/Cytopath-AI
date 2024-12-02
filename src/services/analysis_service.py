import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from tqdm import tqdm
from pathlib import Path

from ..database.config import get_postgres_connection, get_mongo_client
from ..preprocessing.cell_segmentation import CellSegmenter
from ..model.classifier import BethesdaClassifier

class AnalysisService:
    def __init__(self, model=None):
        self.cell_segmenter = CellSegmenter()
        self.classifier = model if model else BethesdaClassifier()
        self.mongo_db = get_mongo_client()
        self.max_batch_size = 32
        self.max_workers = 4

    def analyze_image(self, image_path: str, metadata_id: Optional[str] = None) -> Dict[str, Any]:
        """Perform complete analysis of a cervical smear image"""
        # Generate metadata_id if not provided
        if metadata_id is None:
            metadata_id = str(uuid.uuid4())
            
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect and segment cells
        cell_regions = self.cell_segmenter.detect_cells(image)
        
        # Extract features and classify each cell
        cell_classifications = []
        for cell in cell_regions:
            features = self.cell_segmenter.extract_features(image, cell)
            
            # Store cell data in MongoDB
            cell_data = {
                'image_id': metadata_id,
                'region_id': str(uuid.uuid4()),
                'coordinates': {
                    'x': cell.bbox[0],
                    'y': cell.bbox[1],
                    'width': cell.bbox[2],
                    'height': cell.bbox[3]
                },
                'features': features,
                'analysis_date': datetime.utcnow()
            }
            self.mongo_db.cell_regions.insert_one(cell_data)
            
            # Classify the cell
            cell_img = image[cell.bbox[1]:cell.bbox[1]+cell.bbox[3],
                           cell.bbox[0]:cell.bbox[0]+cell.bbox[2]]
            classification = self.classifier.predict(cell_img)
            cell_classifications.append(classification)
        
        # Aggregate results
        final_classification = self._aggregate_classifications(cell_classifications)
        
        # Store analysis results in PostgreSQL
        try:
            with get_postgres_connection() as conn:
                with conn.cursor() as cur:
                    analysis_id = str(uuid.uuid4())
                    cur.execute("""
                        INSERT INTO analysis_results 
                        (id, image_id, classification, confidence, bethesda_category,
                         num_cells_detected, processing_time)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                    """, (
                        analysis_id,
                        metadata_id,
                        final_classification['classification'],
                        final_classification['confidence'],
                        final_classification['bethesda_category'],
                        len(cell_regions),
                        final_classification['processing_time']
                    ))
                    conn.commit()
        except Exception as e:
            logging.error(f"Database error: {str(e)}")
            analysis_id = str(uuid.uuid4())
        
        return {
            'analysis_id': analysis_id,
            'image_id': metadata_id,
            'classification': final_classification['classification'],
            'confidence': final_classification['confidence'],
            'num_cells': len(cell_regions),
            'bethesda_category': final_classification['bethesda_category']
        }
    
    def detect_cells(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect and analyze cells in an image"""
        # Detect cells
        cell_regions = self.cell_segmenter.detect_cells(image)
        
        # Extract features for each cell
        cells = []
        for cell in cell_regions:
            features = self.cell_segmenter.extract_features(image, cell)
            cells.append({
                'bbox': cell.bbox,
                'features': features
            })
            
        return cells

    def _aggregate_classifications(self, cell_classifications: List[Dict]) -> Dict[str, Any]:
        """Aggregate individual cell classifications into final result"""
        # Count occurrences of each class
        class_counts = {}
        total_confidence = 0
        
        for clf in cell_classifications:
            class_name = clf['classification']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            total_confidence += clf['confidence']
        
        # Get most frequent class
        if not class_counts:
            return {
                'classification': 'NILM',
                'confidence': 0.0,
                'bethesda_category': 'NILM',
                'processing_time': 0.0
            }
        
        most_frequent = max(class_counts.items(), key=lambda x: x[1])
        avg_confidence = total_confidence / len(cell_classifications)
        
        return {
            'classification': most_frequent[0],
            'confidence': avg_confidence,
            'bethesda_category': most_frequent[0],
            'processing_time': 0.0  # TODO: Implement actual timing
        }

    def analyze_batch(self,
                     image_paths: List[str],
                     batch_size: Optional[int] = None,
                     progress_callback: Optional[Callable] = None) -> List[Dict]:
        """
        Analyze a batch of images in parallel
        
        Args:
            image_paths: List of paths to images
            batch_size: Optional batch size (default: self.max_batch_size)
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of analysis results
        """
        if batch_size is None:
            batch_size = self.max_batch_size
            
        results = []
        failed = []
        
        # Process images in batches
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i + batch_size]
            batch_results = []
            
            # Process batch in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_path = {
                    executor.submit(self.analyze_image, path, str(uuid.uuid4())): path
                    for path in batch
                }
                
                # Process completed analyses
                for future in tqdm(
                    as_completed(future_to_path),
                    total=len(batch),
                    desc=f"Processing batch {i//batch_size + 1}"
                ):
                    path = future_to_path[future]
                    try:
                        result = future.result()
                        batch_results.append({
                            'status': 'success',
                            'path': path,
                            **result
                        })
                    except Exception as e:
                        logging.error(f"Failed to process {path}: {str(e)}")
                        failed.append({
                            'status': 'failed',
                            'path': path,
                            'error': str(e)
                        })
                        
            # Update progress
            if progress_callback:
                progress_callback(i + len(batch), len(image_paths))
            
            results.extend(batch_results)
        
        # Add failed analyses to results
        results.extend(failed)
        
        return results
    
    def get_batch_statistics(self, batch_results: List[Dict]) -> Dict:
        """
        Calculate statistics for a batch of analyses
        
        Args:
            batch_results: List of analysis results
            
        Returns:
            Dictionary containing batch statistics
        """
        total = len(batch_results)
        successful = sum(1 for r in batch_results if r['status'] == 'success')
        failed = total - successful
        
        # Calculate class distribution
        class_distribution = {}
        confidence_sum = 0
        confidence_count = 0
        
        for result in batch_results:
            if result['status'] == 'success':
                class_name = result['classification']
                class_distribution[class_name] = class_distribution.get(class_name, 0) + 1
                confidence_sum += result['confidence']
                confidence_count += 1
        
        return {
            'total_images': total,
            'successful': successful,
            'failed': failed,
            'success_rate': successful / total if total > 0 else 0,
            'class_distribution': class_distribution,
            'average_confidence': confidence_sum / confidence_count if confidence_count > 0 else 0
        }
    
    def analyze_directory(self,
                         directory_path: str,
                         recursive: bool = False,
                         file_pattern: str = "*.{jpg,jpeg,png}",
                         progress_callback: Optional[Callable] = None) -> Dict:
        """
        Analyze all images in a directory
        
        Args:
            directory_path: Path to directory
            recursive: Whether to search recursively
            file_pattern: Glob pattern for image files
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary containing results and statistics
        """
        # Get all image paths
        directory = Path(directory_path)
        if recursive:
            image_paths = []
            for pattern in file_pattern.split(','):
                image_paths.extend(directory.rglob(pattern))
        else:
            image_paths = []
            for pattern in file_pattern.split(','):
                image_paths.extend(directory.glob(pattern))
        
        image_paths = [str(p) for p in image_paths]
        
        if not image_paths:
            raise ValueError(f"No images found in {directory_path}")
        
        # Analyze images
        results = self.analyze_batch(
            image_paths,
            progress_callback=progress_callback
        )
        
        # Calculate statistics
        statistics = self.get_batch_statistics(results)
        
        return {
            'results': results,
            'statistics': statistics
        }

    def get_analysis_result(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve analysis result by ID"""
        try:
            with get_postgres_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT id, image_id, classification, confidence, bethesda_category,
                               num_cells_detected, processing_time
                        FROM analysis_results
                        WHERE id = %s
                    """, (analysis_id,))
                    
                    result = cur.fetchone()
                    if result is None:
                        return None
                        
                    return {
                        'analysis_id': result[0],
                        'image_id': result[1],
                        'classification': result[2],
                        'confidence': result[3],
                        'bethesda_category': result[4],
                        'num_cells': result[5],
                        'processing_time': result[6]
                    }
        except Exception as e:
            logging.error(f"Failed to retrieve analysis result: {str(e)}")
            return None
