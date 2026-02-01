"""
FastAPI main application
Entry point for the AQI prediction API
"""

import os
import sys
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Load environment variables from project root
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))
load_dotenv(dotenv_path=env_path)

from backend.routes import router as api_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AQI Predictor API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(api_router, prefix="/api", tags=["predictions"])


@app.get("/", tags=["root"])
async def root():
    """Root endpoint"""
    return {
        "name": "AQI Predictor API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint"""
    import os.path
    from backend.routes import model_artifacts
    
    hopsworks_api_key = os.getenv("HOPSWORKS_API_KEY")
    
    # Check if model artifacts were successfully loaded
    model_loaded = model_artifacts is not None
    
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "model_loaded": model_loaded,
        "feature_store_available": bool(hopsworks_api_key),
        "version": "1.0.0",
        "location": "Hyderabad, Sindh"
    }


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    logger.error(f"HTTP Exception: {exc.detail}")
    return {
        "status": "error",
        "message": exc.detail,
        "timestamp": datetime.now()
    }


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled Exception: {str(exc)}", exc_info=True)
    return {
        "status": "error",
        "message": "Internal server error",
        "timestamp": datetime.now()
    }


if __name__ == "__main__":
    import uvicorn
    
    # Get configuration from environment
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    reload = os.getenv("API_RELOAD", "true").lower() == "true"
    
    logger.info(f"Starting API server on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )
