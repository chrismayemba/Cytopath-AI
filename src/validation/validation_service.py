from datetime import datetime
from typing import Dict, List, Optional
import uuid
from ..database.config import get_postgres_connection, get_mongo_client
from ..services.analysis_service import AnalysisService

class ValidationService:
    def __init__(self):
        self.mongo_db = get_mongo_client()
        self.analysis_service = AnalysisService()
    
    def get_pending_validations(self, pathologist_id: str) -> List[Dict]:
        """Get list of analyses pending validation"""
        with get_postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT ar.id, ar.image_id, ar.classification, ar.confidence,
                           ar.bethesda_category, ar.analysis_date,
                           im.filename, im.upload_date
                    FROM analysis_results ar
                    JOIN image_metadata im ON ar.image_id = im.id
                    WHERE ar.validated_by_pathologist = FALSE
                    ORDER BY ar.analysis_date DESC
                """)
                results = cur.fetchall()
                
        return [dict(result) for result in results]
    
    def get_validation_details(self, analysis_id: str) -> Dict:
        """Get detailed information for validation"""
        # Get analysis results
        with get_postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT ar.*, im.filename, im.upload_date,
                           im.staining_method, im.microscope_info
                    FROM analysis_results ar
                    JOIN image_metadata im ON ar.image_id = im.id
                    WHERE ar.id = %s
                """, (analysis_id,))
                analysis = dict(cur.fetchone())
        
        # Get cell regions from MongoDB
        cell_regions = list(self.mongo_db.cell_regions.find(
            {"analysis_id": analysis_id}
        ))
        
        return {
            "analysis": analysis,
            "cell_regions": cell_regions
        }
    
    def submit_validation(self, 
                         analysis_id: str,
                         pathologist_id: str,
                         validation_data: Dict) -> str:
        """Submit pathologist's validation"""
        validation_id = str(uuid.uuid4())
        
        with get_postgres_connection() as conn:
            with conn.cursor() as cur:
                # Update analysis results
                cur.execute("""
                    UPDATE analysis_results
                    SET validated_by_pathologist = TRUE,
                        pathologist_notes = %s,
                        validation_date = %s,
                        validated_classification = %s,
                        validated_bethesda_category = %s
                    WHERE id = %s
                """, (
                    validation_data.get('notes'),
                    datetime.utcnow(),
                    validation_data.get('classification'),
                    validation_data.get('bethesda_category'),
                    analysis_id
                ))
                
                # Insert validation record
                cur.execute("""
                    INSERT INTO pathologist_validations
                    (id, analysis_id, pathologist_id, validation_date,
                     agreement_status, correction_notes)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    validation_id,
                    analysis_id,
                    pathologist_id,
                    datetime.utcnow(),
                    validation_data.get('agreement_status'),
                    validation_data.get('correction_notes')
                ))
                
                conn.commit()
        
        # Update cell regions if provided
        if 'cell_annotations' in validation_data:
            for cell_id, annotation in validation_data['cell_annotations'].items():
                self.mongo_db.cell_regions.update_one(
                    {"region_id": cell_id},
                    {"$set": {
                        "pathologist_annotation": annotation,
                        "validated_by": pathologist_id,
                        "validation_date": datetime.utcnow()
                    }}
                )
        
        return validation_id
    
    def get_validation_statistics(self, pathologist_id: Optional[str] = None) -> Dict:
        """Get validation statistics"""
        with get_postgres_connection() as conn:
            with conn.cursor() as cur:
                query = """
                    SELECT 
                        COUNT(*) as total_validations,
                        SUM(CASE WHEN agreement_status = 'agree' THEN 1 ELSE 0 END) as agreements,
                        AVG(CASE WHEN agreement_status = 'agree' THEN 1 ELSE 0 END) as agreement_rate
                    FROM pathologist_validations
                """
                if pathologist_id:
                    query += " WHERE pathologist_id = %s"
                    cur.execute(query, (pathologist_id,))
                else:
                    cur.execute(query)
                    
                stats = dict(cur.fetchone())
        
        return stats
