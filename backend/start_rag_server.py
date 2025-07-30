#!/usr/bin/env python3
"""
Startup script for the RAG-enhanced candidate search server.
Initializes RAG system and starts FastAPI server.
"""

import os
import sys
import logging
import uvicorn
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_environment():
    """Check if required environment variables are set."""
    required_vars = [
        'PROJECT_ID',
        'GOOGLE_API_KEY'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        logger.error("Please set the following environment variables:")
        for var in missing_vars:
            logger.error(f"  export {var}=your_value_here")
        return False
    
    return True

def setup_gcp_credentials():
    """Setup GCP credentials if not already configured."""
    if not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
        logger.warning("GOOGLE_APPLICATION_CREDENTIALS not set")
        logger.warning("Using default credentials from gcloud CLI")
    return True

def start_server():
    """Start the FastAPI server with RAG system."""
    try:
        logger.info("Starting RAG server...")
        
        # Check environment
        if not check_environment():
            return False
        
        # Setup GCP credentials
        if not setup_gcp_credentials():
            return False
        
        # Import and start the app
        logger.info("Loading FastAPI application...")
        
        try:
            # Import the app
            from gcp_main_rag import app
            logger.info("FastAPI app loaded successfully")
        except ImportError as e:
            logger.error(f"Failed to import FastAPI app: {e}")
            return False
        
        # Initialize RAG system
        logger.info("Initializing RAG system...")
        try:
            from rag_enhanced import initialize_rag_system
            initialize_rag_system()
            logger.info("RAG system initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing RAG system: {e}")
            logger.warning("Continuing without RAG initialization...")
        
        # Start the server
        logger.info("Starting FastAPI server on http://0.0.0.0:8000")
        logger.info("API documentation available at http://0.0.0.0:8000/docs")
        
        uvicorn.run(
            "gcp_main_rag:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info",
            access_log=True
        )
        
        return True
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        return True
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        return False

if __name__ == "__main__":
    logger.info("RAG-Enhanced Candidate Search Server")
    logger.info("="*50)
    
    success = start_server()
    
    if not success:
        logger.error("Failed to start server")
        sys.exit(1)
    else:
        logger.info("Server shutdown complete")
        sys.exit(0) 