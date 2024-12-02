-- Initialize database schema for cervical lesion detection

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Analysis Results Table
CREATE TABLE IF NOT EXISTS analysis_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    image_id UUID NOT NULL,
    classification VARCHAR(50) NOT NULL,
    confidence FLOAT NOT NULL,
    bethesda_category VARCHAR(50) NOT NULL,
    num_cells_detected INTEGER,
    processing_time FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Cell Detection Results
CREATE TABLE IF NOT EXISTS cell_detections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    analysis_id UUID REFERENCES analysis_results(id),
    cell_count INTEGER NOT NULL,
    cell_locations JSONB,
    detection_confidence FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_analysis_results_image_id ON analysis_results(image_id);
CREATE INDEX IF NOT EXISTS idx_analysis_results_classification ON analysis_results(classification);
CREATE INDEX IF NOT EXISTS idx_cell_detections_analysis_id ON cell_detections(analysis_id);

-- Create view for analysis statistics
CREATE OR REPLACE VIEW analysis_statistics AS
SELECT 
    classification,
    COUNT(*) as total_analyses,
    AVG(confidence) as avg_confidence,
    AVG(num_cells_detected) as avg_cells_detected,
    AVG(processing_time) as avg_processing_time
FROM analysis_results
GROUP BY classification;
