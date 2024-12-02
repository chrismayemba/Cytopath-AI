import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import csv
from typing import List, Dict, Optional
import zipfile
import io
import shutil

from ..database.config import get_postgres_connection, get_mongo_client

class DataExporter:
    def __init__(self, export_dir: str):
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.mongo_db = get_mongo_client()
    
    def export_analysis_results(self,
                              start_date: Optional[datetime] = None,
                              end_date: Optional[datetime] = None,
                              format: str = 'csv') -> str:
        """Export analysis results within date range"""
        # Build query
        query = """
            SELECT 
                ar.id as analysis_id,
                ar.image_id,
                im.filename,
                ar.analysis_date,
                ar.classification,
                ar.confidence,
                ar.bethesda_category,
                ar.num_cells_detected,
                ar.processing_time,
                ar.validated_by_pathologist,
                ar.validated_classification,
                ar.validation_date,
                ar.pathologist_notes,
                im.staining_method,
                im.microscope_info
            FROM analysis_results ar
            JOIN image_metadata im ON ar.image_id = im.id
            WHERE 1=1
        """
        params = []
        
        if start_date:
            query += " AND ar.analysis_date >= %s"
            params.append(start_date)
        if end_date:
            query += " AND ar.analysis_date <= %s"
            params.append(end_date)
            
        query += " ORDER BY ar.analysis_date DESC"
        
        # Execute query
        with get_postgres_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        
        # Export based on format
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if format == 'csv':
            output_path = self.export_dir / f'analysis_results_{timestamp}.csv'
            df.to_csv(output_path, index=False)
        elif format == 'excel':
            output_path = self.export_dir / f'analysis_results_{timestamp}.xlsx'
            df.to_excel(output_path, index=False)
        
        return str(output_path)
    
    def export_cell_data(self, analysis_id: str, format: str = 'csv') -> str:
        """Export cell-level data for a specific analysis"""
        # Get cell regions from MongoDB
        cell_regions = list(self.mongo_db.cell_regions.find(
            {"analysis_id": analysis_id}
        ))
        
        # Convert to DataFrame
        df = pd.json_normalize(cell_regions)
        
        # Export based on format
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if format == 'csv':
            output_path = self.export_dir / f'cell_data_{analysis_id}_{timestamp}.csv'
            df.to_csv(output_path, index=False)
        elif format == 'excel':
            output_path = self.export_dir / f'cell_data_{analysis_id}_{timestamp}.xlsx'
            df.to_excel(output_path, index=False)
        
        return str(output_path)
    
    def export_validation_statistics(self,
                                   start_date: Optional[datetime] = None,
                                   end_date: Optional[datetime] = None) -> str:
        """Export validation statistics"""
        query = """
            SELECT 
                pv.pathologist_id,
                COUNT(*) as total_validations,
                SUM(CASE WHEN pv.agreement_status = 'agree' THEN 1 ELSE 0 END) as agreements,
                AVG(CASE WHEN pv.agreement_status = 'agree' THEN 1 ELSE 0 END) as agreement_rate,
                ar.bethesda_category,
                COUNT(*) as category_count
            FROM pathologist_validations pv
            JOIN analysis_results ar ON pv.analysis_id = ar.id
            WHERE 1=1
        """
        params = []
        
        if start_date:
            query += " AND pv.validation_date >= %s"
            params.append(start_date)
        if end_date:
            query += " AND pv.validation_date <= %s"
            params.append(end_date)
            
        query += " GROUP BY pv.pathologist_id, ar.bethesda_category"
        
        # Execute query
        with get_postgres_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        
        # Export to Excel with multiple sheets
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = self.export_dir / f'validation_statistics_{timestamp}.xlsx'
        
        with pd.ExcelWriter(output_path) as writer:
            # Overall statistics
            df_overall = df.groupby('pathologist_id').agg({
                'total_validations': 'sum',
                'agreements': 'sum',
                'agreement_rate': 'mean'
            }).reset_index()
            df_overall.to_excel(writer, sheet_name='Overall Statistics', index=False)
            
            # Category-wise statistics
            df_category = df.pivot_table(
                index='pathologist_id',
                columns='bethesda_category',
                values=['category_count', 'agreement_rate'],
                aggfunc={'category_count': 'sum', 'agreement_rate': 'mean'}
            ).reset_index()
            df_category.to_excel(writer, sheet_name='Category Statistics')
        
        return str(output_path)
    
    def create_research_dataset(self,
                              output_path: str,
                              include_images: bool = True) -> str:
        """Create a complete dataset for research purposes"""
        # Create temporary directory for dataset
        temp_dir = self.export_dir / 'temp_dataset'
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # Export all data tables
            self.export_analysis_results(
                format='csv',
                output_path=temp_dir / 'analysis_results.csv'
            )
            
            # Export validation statistics
            self.export_validation_statistics(
                output_path=temp_dir / 'validation_statistics.xlsx'
            )
            
            # Export cell data
            with get_postgres_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT id FROM analysis_results")
                    analysis_ids = [r[0] for r in cur.fetchall()]
            
            for analysis_id in analysis_ids:
                self.export_cell_data(
                    analysis_id,
                    format='csv',
                    output_path=temp_dir / f'cell_data_{analysis_id}.csv'
                )
            
            # Include images if requested
            if include_images:
                image_dir = temp_dir / 'images'
                image_dir.mkdir(exist_ok=True)
                
                with get_postgres_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT filename FROM image_metadata")
                        filenames = [r[0] for r in cur.fetchall()]
                
                for filename in filenames:
                    src_path = Path(filename)
                    if src_path.exists():
                        shutil.copy2(src_path, image_dir / src_path.name)
            
            # Create README
            with open(temp_dir / 'README.md', 'w') as f:
                f.write("""# Cervical Lesion Detection Dataset

This dataset contains cervical cytology images and their analysis results.

## Contents
1. analysis_results.csv: All analysis results with metadata
2. validation_statistics.xlsx: Pathologist validation statistics
3. cell_data_*.csv: Individual cell detection results
4. images/: Original cytology images (if included)

## Data Format
[Detailed description of data formats and fields]
""")
            
            # Create zip archive
            shutil.make_archive(output_path, 'zip', temp_dir)
            
        finally:
            # Cleanup
            shutil.rmtree(temp_dir)
        
        return output_path + '.zip'
