#!/usr/bin/env python3
"""
CV Processing and Candidate Search API
FastAPI backend for candidate search with vector embeddings and GCP integration.
"""

import os
import logging
import uuid
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import google.generativeai as genai
from google.cloud import spanner
from google.cloud import storage

# logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GCP project configuration
PROJECT_ID = "niky-search-demo"
INSTANCE_ID = "niky-search-spanner"
DATABASE_ID = "candidates-db"
BUCKET_NAME = "candidate-cv-bucket-niky-search-demo"

# Initialize GCP clients
spanner_client = spanner.Client(project=PROJECT_ID)
instance = spanner_client.instance(INSTANCE_ID)
database = instance.database(DATABASE_ID)
storage_client = storage.Client(project=PROJECT_ID)
bucket = storage_client.bucket(BUCKET_NAME)

# Setup Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logger.warning("GOOGLE_API_KEY not found. Please set it in your environment.")
    GOOGLE_API_KEY = "your-api-key-here"

genai.configure(api_key=GOOGLE_API_KEY)

# Placeholder for other functions and models if they existed in the original file
# For the purpose of this edit, we'll assume they are defined elsewhere or will be added.
# Since the original file was empty, we'll just define the functions that were present.

def upload_to_gcs(file_content: bytes, filename: str) -> str:
    try:
        blob = bucket.blob(f"cvs/{filename}")
        blob.upload_from_string(file_content, content_type='application/pdf')
        gcs_url = f"gs://{BUCKET_NAME}/cvs/{filename}"
        logger.info(f"File uploaded to GCS: {gcs_url}")
        return gcs_url
    except Exception as e:
        logger.error(f"Error uploading to GCS: {e}")
        raise

def insert_candidate(candidate_data: Dict[str, Any]) -> None:
    try:
        with database.batch() as batch:
            batch.insert(
                table='candidates',
                columns=(
                    'candidate_id', 'name', 'email', 'phone', 'summary', 
                    'gcs_url', 'created_at', 'skills'
                ),
                values=[(
                    candidate_data['candidate_id'],
                    candidate_data['name'],
                    candidate_data['email'],
                    candidate_data['phone'],
                    candidate_data['summary'],
                    candidate_data['gcs_url'],
                    candidate_data['created_at'],
                    candidate_data['skills']
                )]
            )
            
        logger.info(f"Candidate inserted into Spanner: {candidate_data['candidate_id']}")
        
        # Update skills table separately
        update_skills_table(candidate_data['skills'])
        
    except Exception as e:
        logger.error(f"Error inserting candidate: {e}")
        raise

def update_skills_table(skills: List[str]) -> None:
    try:
        for skill in skills:
            # Check if skill already exists
            query = "SELECT skill_id, candidate_count FROM skills WHERE name = @skill_name"
            
            with database.snapshot() as snapshot:
                results = snapshot.execute_sql(
                    query,
                    params={'skill_name': skill},
                    param_types={'skill_name': spanner.param_types.STRING}
                )
                
                if results:
                    # Skill exists, update count
                    skill_id, current_count = results[0]
                    new_count = current_count + 1
                    
                    with database.batch() as batch:
                        batch.update(
                            table='skills',
                            columns=('candidate_count', 'updated_at'),
                            values=[(new_count, datetime.now())],
                            keyset=spanner.KeySet(keys=[[skill_id]])
                        )
                else:
                    # Skill doesn't exist, insert new skill
                    skill_id = f"skill-{skill.lower().replace(' ', '-')}-{uuid.uuid4().hex[:8]}"
                    
                    with database.batch() as batch:
                        batch.insert(
                            table='skills',
                            columns=('skill_id', 'name', 'candidate_count', 'created_at'),
                            values=[(skill_id, skill, 1, datetime.now())]
                        )
        
        logger.info(f"Skills table updated with: {skills}")
    except Exception as e:
        logger.error(f"Error updating skills table: {e}")

def get_candidates(skip: int = 0, limit: int = 50) -> List[Dict[str, Any]]:
    try:
        query = """
        SELECT candidate_id, name, email, phone, summary, gcs_url, created_at, skills
        FROM candidates
        ORDER BY created_at DESC
        LIMIT @limit OFFSET @skip
        """
        
        with database.snapshot() as snapshot:
            results = snapshot.execute_sql(
                query,
                params={'limit': limit, 'skip': skip},
                param_types={'limit': spanner.param_types.INT64, 'skip': spanner.param_types.INT64}
            )
            
            candidates = []
            for row in results:
                candidates.append({
                    'candidate_id': str(row[0]),
                    'name': str(row[1]),
                    'email': str(row[2]) if row[2] else None,
                    'phone': str(row[3]) if row[3] else None,
                    'summary': str(row[4]) if row[4] else None,
                    'gcs_url': str(row[5]) if row[5] else None,
                    'created_at': row[6],
                    'skills': list(row[7]) if row[7] else []
                })
            
            return candidates
    except Exception as e:
        logger.error(f"Error getting candidates: {e}")
        return []

def get_candidate_by_id(candidate_id: str) -> Optional[Dict[str, Any]]:
    try:
        query = """
        SELECT candidate_id, name, email, phone, summary, gcs_url, created_at, skills
        FROM candidates
        WHERE candidate_id = @candidate_id
        """
        
        with database.snapshot() as snapshot:
            results = snapshot.execute_sql(
                query,
                params={'candidate_id': candidate_id},
                param_types={'candidate_id': spanner.param_types.STRING}
            )
            
            candidates = list(results)
            if candidates:
                row = candidates[0]
                return {
                    'candidate_id': str(row[0]),
                    'name': str(row[1]),
                    'email': str(row[2]) if row[2] else None,
                    'phone': str(row[3]) if row[3] else None,
                    'summary': str(row[4]) if row[4] else None,
                    'gcs_url': str(row[5]) if row[5] else None,
                    'created_at': row[6],
                    'skills': list(row[7]) if row[7] else []
                }
            
            return None
    except Exception as e:
        logger.error(f"Error getting candidate: {e}")
        return None

# Assuming these models and functions exist elsewhere or are placeholders
class UploadResponse(BaseModel):
    success: bool
    message: str
    candidate_id: str
    candidate_name: str
    filename: str
    skills_extracted: List[str]
    skills_found: int
    chunks_processed: int

# Assuming these constants exist elsewhere or are placeholders
ALLOWED_EXTENSIONS = ['.pdf']
MAX_FILE_SIZE = 10 * 1024 * 1024 # 10MB

# Assuming these functions exist elsewhere or are placeholders
def extract_text(file_path: str) -> Optional[str]:
    """Placeholder for text extraction logic."""
    logger.warning("extract_text is a placeholder. Implement actual text extraction.")
    return "Placeholder text for " + os.path.basename(file_path)

def extract_candidate_name(text: str) -> str:
    """Placeholder for candidate name extraction logic."""
    logger.warning("extract_candidate_name is a placeholder. Implement actual name extraction.")
    return "Placeholder Candidate Name"

def extract_summary(text: Optional[str], name: str) -> str:
    """Placeholder for summary extraction logic."""
    logger.warning("extract_summary is a placeholder. Implement actual summary extraction.")
    return f"Placeholder summary for {name}. Extracted from {os.path.basename(file_path)}"

def extract_skills_from_text(text: Optional[str]) -> List[Dict[str, Any]]:
    """Placeholder for skill extraction logic."""
    logger.warning("extract_skills_from_text is a placeholder. Implement actual skill extraction.")
    return [{"name": "Placeholder Skill", "confidence": 0.9}]

# Assuming app is defined elsewhere or is a placeholder
app = FastAPI()

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
@app.post("/upload", response_model=UploadResponse, tags=["Upload"])
async def upload_cv(file: UploadFile = File(...)):
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400, 
                detail=f"File type not allowed. Allowed types: {ALLOWED_EXTENSIONS}"
            )
        
        # Read file content
        file_content = await file.read()
        if len(file_content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File too large")
        
        logger.info(f"Processing CV: {file.filename}")
        logger.info(f"File size: {len(file_content)} bytes")
        
        # Save file temporarily
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(file_content)
        
        # Extract text
        text = extract_text(temp_path)
        if text:
            logger.info(f"Extracted {len(text)} characters of text")
            candidate_name = extract_candidate_name(text)
            logger.info(f"Extracted candidate name: {candidate_name}")
        else:
            logger.warning("Could not extract text, falling back to filename")
            candidate_name = file.filename.split('.')[0].replace('CV - ', '').replace('_', ' ')
        
        # If name extraction failed, try filename
        if candidate_name == "Unknown Candidate" or len(candidate_name.split()) < 2:
            filename_name = file.filename.replace('_detailed_resume.pdf', '').replace('_', ' ').title()
            if len(filename_name.split()) >= 2:
                candidate_name = filename_name
                logger.info(f"Using filename for candidate name: {candidate_name}")
        
        # Generate unique candidate ID
        candidate_id = f"candidate-{uuid.uuid4().hex[:8]}-{datetime.now().strftime('%Y%m%d%H%M')}"
        
        # Extract summary
        summary = extract_summary(text, candidate_name)
        logger.info(f"Extracted summary: {summary[:100]}...")
        
        # Extract skills
        skills_data = extract_skills_from_text(text)
        logger.info(f"Skills with confidence: {skills_data}")
        skills = [skill['name'] for skill in skills_data]
        
        # Upload to GCS
        gcs_url = upload_to_gcs(file_content, file.filename)
        
        # Prepare candidate data
        candidate_data = {
            'candidate_id': candidate_id,
            'name': candidate_name,
            'email': None,
            'phone': None,
            'summary': summary,
            'gcs_url': gcs_url,
            'created_at': datetime.now(),
            'skills': skills
        }
        
        # Insert into database
        insert_candidate(candidate_data)
        
        return UploadResponse(
            success=True,
            message=f"Successfully uploaded and processed: {file.filename}",
            candidate_id=candidate_id,
            candidate_name=candidate_name,
            filename=file.filename,
            skills_extracted=skills,
            skills_found=len(skills),
            chunks_processed=1
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing uploaded file: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    finally:
        # Clean up temp file
        if 'temp_path' in locals():
            try:
                os.remove(temp_path)
            except:
                pass

@app.delete("/candidates/{candidate_id}", tags=["Candidates"])
async def delete_candidate(candidate_id: str):
    try:
        # Get candidate info
        candidate = get_candidate_by_id(candidate_id)
        if not candidate:
            raise HTTPException(status_code=404, detail="Candidate not found")
        
        gcs_url = candidate.get('gcs_url')
        
        # Delete file from GCS if it exists
        if gcs_url and gcs_url.startswith('gs://'):
            try:
                # Extract bucket and blob name from GCS URL
                parts = gcs_url.replace('gs://', '').split('/', 1)
                if len(parts) == 2:
                    bucket_name, blob_name = parts
                    bucket = storage_client.bucket(bucket_name)
                    blob = bucket.blob(blob_name)
                    
                    if blob.exists():
                        blob.delete()
                        logger.info(f"Deleted file from GCS: {gcs_url}")
                    else:
                        logger.warning(f"File not found in GCS: {gcs_url}")
            except Exception as e:
                logger.error(f"Error deleting file from GCS: {e}")
        
        # Delete from Cloud Spanner
        try:
            with database.batch() as batch:
                batch.delete(
                    table='candidates',
                    keyset=spanner.KeySet(keys=[[candidate_id]])
                )
            logger.info(f"Deleted candidate from Spanner: {candidate_id}")
        except Exception as e:
            logger.error(f"Error deleting from Spanner: {e}")
            raise HTTPException(status_code=500, detail="Error deleting from database")
        
        return {
            "success": True,
            "message": f"Successfully deleted candidate: {candidate.get('name', candidate_id)}",
            "candidate_id": candidate_id,
            "file_deleted": gcs_url is not None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting candidate: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting candidate: {str(e)}")

@app.get("/skills", tags=["Skills"])
async def list_skills():
    try:
        query = """
        SELECT name, candidate_count
        FROM skills
        ORDER BY candidate_count DESC
        """
        
        with database.snapshot() as snapshot:
            results = snapshot.execute_sql(query)
            
            skills = []
            for row in results:
                skills.append({
                    'name': str(row[0]),
                    'candidate_count': int(row[1])
                })
            
            return {"skills": skills, "total": len(skills)}
    except Exception as e:
        logger.error(f"Error getting skills: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True) 