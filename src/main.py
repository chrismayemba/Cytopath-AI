from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from typing import List, Optional
import uvicorn
from pathlib import Path
from datetime import datetime
import uuid
import shutil
import cv2
import logging
from pydantic import BaseModel
from typing import Dict

from .services.analysis_service import AnalysisService
from .validation.validation_service import ValidationService
from .reporting.report_generator import ReportGenerator
from .export.export_service import DataExporter
from .model.interpretability import ModelInterpreter

app = FastAPI(title="Cervical Lesion Detection API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
UPLOAD_DIR = Path("uploads")
REPORT_DIR = Path("reports")
EXPORT_DIR = Path("exports")
UPLOAD_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)
EXPORT_DIR.mkdir(exist_ok=True)

# Services
analysis_service = AnalysisService()
validation_service = ValidationService()
report_generator = ReportGenerator(str(REPORT_DIR))
export_service = DataExporter(str(EXPORT_DIR))
model_interpreter = ModelInterpreter(analysis_service.classifier.model, analysis_service.classifier.device)

class BatchAnalysisRequest(BaseModel):
    directory_path: str
    recursive: bool = False
    file_pattern: str = "*.{jpg,jpeg,png}"

class BatchStatus(BaseModel):
    batch_id: str
    total: int
    processed: int
    status: str
    results: Optional[Dict] = None
    statistics: Optional[Dict] = None

# Store batch processing status
batch_status = {}

@app.get("/")
async def root():
    return {"message": "Cervical Lesion Detection API"}

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    """Upload a cytological image for analysis"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, detail="File must be an image")
    
    try:
        # Generate unique filename
        file_id = str(uuid.uuid4())
        file_extension = Path(file.filename).suffix
        file_path = UPLOAD_DIR / f"{file_id}{file_extension}"
        
        # Save file
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        return {"filename": file_path.name, "status": "uploaded"}
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.post("/analyze/{image_id}")
async def analyze_image(image_id: str):
    """Analyze an uploaded image"""
    try:
        result = analysis_service.analyze_image(str(UPLOAD_DIR / image_id))
        return result
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.get("/validations/pending")
async def get_pending_validations(pathologist_id: str):
    """Get list of analyses pending validation"""
    try:
        return validation_service.get_pending_validations(pathologist_id)
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.get("/validation/{analysis_id}")
async def get_validation_details(analysis_id: str):
    """Get detailed information for validation"""
    try:
        return validation_service.get_validation_details(analysis_id)
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.post("/validation/{analysis_id}")
async def submit_validation(
    analysis_id: str,
    pathologist_id: str,
    validation_data: dict
):
    """Submit pathologist's validation"""
    try:
        validation_id = validation_service.submit_validation(
            analysis_id,
            pathologist_id,
            validation_data
        )
        return {"validation_id": validation_id}
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.get("/validation/statistics")
async def get_validation_statistics(pathologist_id: Optional[str] = None):
    """Get validation statistics"""
    try:
        return validation_service.get_validation_statistics(pathologist_id)
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.get("/report/{analysis_id}")
async def generate_report(analysis_id: str):
    """Generate and return PDF report"""
    try:
        report_path = report_generator.generate_report(analysis_id)
        return FileResponse(
            report_path,
            media_type="application/pdf",
            filename=f"report_{analysis_id}.pdf"
        )
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.get("/export/analysis")
async def export_analysis_results(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    format: str = 'csv'
):
    """Export analysis results"""
    try:
        output_path = export_service.export_analysis_results(
            start_date=start_date,
            end_date=end_date,
            format=format
        )
        return FileResponse(
            output_path,
            media_type='application/octet-stream',
            filename=Path(output_path).name
        )
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.get("/export/cell-data/{analysis_id}")
async def export_cell_data(analysis_id: str, format: str = 'csv'):
    """Export cell-level data for an analysis"""
    try:
        output_path = export_service.export_cell_data(
            analysis_id=analysis_id,
            format=format
        )
        return FileResponse(
            output_path,
            media_type='application/octet-stream',
            filename=Path(output_path).name
        )
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.get("/export/validation-stats")
async def export_validation_statistics(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
):
    """Export validation statistics"""
    try:
        output_path = export_service.export_validation_statistics(
            start_date=start_date,
            end_date=end_date
        )
        return FileResponse(
            output_path,
            media_type='application/octet-stream',
            filename=Path(output_path).name
        )
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.post("/export/research-dataset")
async def create_research_dataset(include_images: bool = True):
    """Create complete dataset for research"""
    try:
        output_path = Path("exports") / f"research_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        zip_path = export_service.create_research_dataset(
            str(output_path),
            include_images=include_images
        )
        return FileResponse(
            zip_path,
            media_type='application/zip',
            filename=Path(zip_path).name
        )
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.get("/interpret/{analysis_id}")
async def get_interpretation(analysis_id: str):
    """Get model interpretation for an analysis"""
    try:
        # Get analysis details
        with get_postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT ar.*, im.filename
                    FROM analysis_results ar
                    JOIN image_metadata im ON ar.image_id = im.id
                    WHERE ar.id = %s
                """, (analysis_id,))
                analysis = dict(cur.fetchone())
        
        # Get cell regions
        cell_regions = list(model_interpreter.mongo_db.cell_regions.find(
            {"analysis_id": analysis_id}
        ))
        
        # Load image
        image_path = analysis['filename']
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Prepare image tensor
        image_tensor = analysis_service.classifier.preprocess_image(image)
        
        # Get interpretations
        interpretations = model_interpreter.analyze_cell_regions(
            image_tensor,
            cell_regions
        )
        
        # Generate visualization
        vis_path = Path("exports") / f"interpretation_{analysis_id}.png"
        model_interpreter.visualize_interpretations(
            image,
            interpretations[0]['saliency_maps'],  # Use first cell's saliency maps
            str(vis_path)
        )
        
        return {
            "interpretations": interpretations,
            "visualization_url": f"/static/interpretations/{vis_path.name}"
        }
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.post("/analyze/batch")
async def analyze_batch(request: BatchAnalysisRequest, background_tasks: BackgroundTasks):
    """Start batch analysis of images in directory"""
    try:
        # Validate directory
        directory = Path(request.directory_path)
        if not directory.exists():
            raise HTTPException(400, "Directory does not exist")
        
        # Generate batch ID
        batch_id = str(uuid.uuid4())
        
        # Initialize status
        batch_status[batch_id] = {
            'batch_id': batch_id,
            'total': 0,
            'processed': 0,
            'status': 'initializing',
            'results': None,
            'statistics': None
        }
        
        # Start background task
        background_tasks.add_task(
            process_batch,
            batch_id,
            str(directory),
            request.recursive,
            request.file_pattern
        )
        
        return JSONResponse({
            'batch_id': batch_id,
            'message': 'Batch processing started'
        })
        
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/analyze/batch/{batch_id}")
async def get_batch_status(batch_id: str):
    """Get status of batch processing"""
    if batch_id not in batch_status:
        raise HTTPException(404, "Batch ID not found")
    
    return BatchStatus(**batch_status[batch_id])

async def process_batch(batch_id: str,
                       directory: str,
                       recursive: bool,
                       file_pattern: str):
    """Background task for batch processing"""
    try:
        def progress_callback(processed, total):
            batch_status[batch_id]['total'] = total
            batch_status[batch_id]['processed'] = processed
        
        # Update status
        batch_status[batch_id]['status'] = 'processing'
        
        # Process directory
        result = analysis_service.analyze_directory(
            directory,
            recursive=recursive,
            file_pattern=file_pattern,
            progress_callback=progress_callback
        )
        
        # Update status with results
        batch_status[batch_id].update({
            'status': 'completed',
            'results': result['results'],
            'statistics': result['statistics']
        })
        
    except Exception as e:
        # Update status with error
        batch_status[batch_id].update({
            'status': 'failed',
            'error': str(e)
        })
        
        logging.error(f"Batch processing failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
