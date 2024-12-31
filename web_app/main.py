from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from datetime import datetime
import json
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to sys.path
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
sys.path.append(str(root_dir))

try:
    # Import your model
    from src.model.classifier import BethesdaClassifier
    logger.info("Successfully imported BethesdaClassifier")
except Exception as e:
    logger.error(f"Error importing BethesdaClassifier: {str(e)}")
    BethesdaClassifier = None

# Initialize FastAPI app
app = FastAPI(
    title="Cytopath AI",
    description="API pour la détection des lésions cervicales",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure templates and static files
templates = Jinja2Templates(directory=str(current_dir / "templates"))
static_dir = current_dir / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Create directories if they don't exist
UPLOAD_DIR = root_dir / "uploads"
FEEDBACK_DIR = root_dir / "expert_feedback"
UPLOAD_DIR.mkdir(exist_ok=True)
FEEDBACK_DIR.mkdir(exist_ok=True)

# Initialize classifier
try:
    classifier = BethesdaClassifier() if BethesdaClassifier else None
    logger.info("Classifier initialized successfully")
except Exception as e:
    logger.error(f"Error initializing classifier: {str(e)}")
    classifier = None

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        if not classifier:
            raise ValueError("Classifier not initialized")

        # Create a temporary file path
        temp_path = UPLOAD_DIR / f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        
        # Save the uploaded file
        with temp_path.open("wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Make prediction
        result = classifier.predict(str(temp_path))
        
        # Clean up
        if temp_path.exists():
            temp_path.unlink()
        
        return result
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/feedback")
async def save_feedback(request: Request):
    try:
        feedback = await request.json()
        
        # Add timestamp to feedback
        feedback["timestamp"] = datetime.now().isoformat()
        
        # Generate unique filename
        filename = f"feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = FEEDBACK_DIR / filename
        
        # Save feedback to file
        with filepath.open("w") as f:
            json.dump(feedback, f, indent=4)
        
        logger.info(f"Expert feedback saved to {filepath}")
        return {"status": "success", "message": "Feedback saved successfully"}
    except Exception as e:
        logger.error(f"Error saving feedback: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "classifier_loaded": classifier is not None
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port) 