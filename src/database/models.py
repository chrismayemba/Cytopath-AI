from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field

class ImageMetadata(BaseModel):
    """Metadata for uploaded images stored in PostgreSQL"""
    id: str
    filename: str
    upload_date: datetime = Field(default_factory=datetime.utcnow)
    file_size: int
    mime_type: str
    resolution: tuple[int, int]
    staining_method: Optional[str] = None
    microscope_info: Optional[str] = None

class AnalysisResult(BaseModel):
    """Analysis results stored in PostgreSQL"""
    id: str
    image_id: str
    analysis_date: datetime = Field(default_factory=datetime.utcnow)
    classification: str
    confidence: float
    bethesda_category: str
    num_cells_detected: int
    processing_time: float
    validated_by_pathologist: bool = False
    pathologist_notes: Optional[str] = None

class CellRegion(BaseModel):
    """Individual cell region information stored in MongoDB"""
    image_id: str
    analysis_id: str
    region_id: str
    coordinates: dict  # x, y, width, height
    features: dict    # extracted features
    classification: str
    confidence: float

# SQL queries for table creation
CREATE_TABLES_QUERIES = [
    """
    CREATE TABLE IF NOT EXISTS image_metadata (
        id VARCHAR(36) PRIMARY KEY,
        filename VARCHAR(255) NOT NULL,
        upload_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        file_size INTEGER NOT NULL,
        mime_type VARCHAR(100) NOT NULL,
        resolution_width INTEGER NOT NULL,
        resolution_height INTEGER NOT NULL,
        staining_method VARCHAR(100),
        microscope_info TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS analysis_results (
        id VARCHAR(36) PRIMARY KEY,
        image_id VARCHAR(36) REFERENCES image_metadata(id),
        analysis_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        classification VARCHAR(50) NOT NULL,
        confidence FLOAT NOT NULL,
        bethesda_category VARCHAR(50) NOT NULL,
        num_cells_detected INTEGER NOT NULL,
        processing_time FLOAT NOT NULL,
        validated_by_pathologist BOOLEAN DEFAULT FALSE,
        pathologist_notes TEXT,
        CONSTRAINT fk_image
            FOREIGN KEY(image_id)
            REFERENCES image_metadata(id)
            ON DELETE CASCADE
    );
    """
]
