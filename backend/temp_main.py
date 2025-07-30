#!/usr/bin/env python3
"""
Temporary main file for testing CV processing functionality.
This is a simplified version for development and testing purposes.
"""

import os
import logging
import uuid
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Models
class UploadResponse(BaseModel):
    success: bool
    message: str
    candidate_id: Optional[str] = None
    candidate_name: Optional[str] = None
    filename: Optional[str] = None
    skills_extracted: List[str] = []
    skills_found: int = 0
    chunks_processed: int = 0

class HealthCheck(BaseModel):
    status: str
    timestamp: datetime

# File processing functions
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF file."""
    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return ""

def extract_text_from_docx(docx_path: str) -> str:
    """Extract text from DOCX file."""
    try:
        from docx import Document
        doc = Document(docx_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from DOCX: {e}")
        return ""

def extract_text(file_path: str) -> str:
    """Extract text from supported file formats."""
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.pdf':
        return extract_text_from_pdf(file_path)
    elif file_extension == '.docx':
        return extract_text_from_docx(file_path)
    else:
        logger.warning(f"Unsupported file format: {file_extension}")
        return ""

def extract_candidate_name(text: str) -> str:
    """Extract candidate name from CV text."""
    if not text:
        return "Unknown Candidate"
    
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    if not lines:
        return "Unknown Candidate"
    
    # Look for name in first few lines
    for line in lines[:5]:
        # Skip lines with common keywords
        if any(keyword in line.lower() for keyword in ['resume', 'cv', 'email', 'phone']):
            continue
        
        # Simple name pattern matching
        words = line.split()
        if 2 <= len(words) <= 4 and all(word.isalpha() and word[0].isupper() for word in words):
            return line.strip()
    
    return "Unknown Candidate"

def extract_summary(text: str, candidate_name: str) -> str:
    """Extract basic summary from CV text."""
    if not text:
        return f"Professional with experience in software development."
    
    # Look for summary/objective sections
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if any(keyword in line.lower() for keyword in ['summary', 'objective', 'profile']):
            # Take next few lines
            summary_lines = []
            for j in range(i + 1, min(i + 3, len(lines))):
                if lines[j].strip() and len(lines[j]) > 20:
                    summary_lines.append(lines[j].strip())
            if summary_lines:
                return ' '.join(summary_lines)[:300]
    
    # Fallback to first meaningful paragraph
    for line in lines:
        if len(line.strip()) > 50:
            return line.strip()[:300]
    
    return f"Professional with experience in software development."

def extract_skills_from_text(text: str) -> List[Dict[str, Any]]:
    """Extract skills using simple keyword matching."""
    if not text:
        return []
    
    # Common technical skills
    skills_list = [
        'Python', 'JavaScript', 'Java', 'C++', 'C#', 'Go', 'PHP', 'Ruby', 'Swift',
        'React', 'Angular', 'Vue', 'Node.js', 'Django', 'Flask', 'Spring',
        'AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes', 'Git', 'Jenkins',
        'MongoDB', 'PostgreSQL', 'MySQL', 'Redis', 'Elasticsearch',
        'Machine Learning', 'AI', 'Data Science', 'TensorFlow', 'PyTorch',
        'Linux', 'Agile', 'Scrum', 'DevOps', 'CI/CD'
    ]
    
    found_skills = []
    text_lower = text.lower()
    
    for skill in skills_list:
        if skill.lower() in text_lower:
            found_skills.append({
                'name': skill,
                'confidence': 0.8
            })
    
    return found_skills[:10]  # Limit to top 10

# FastAPI app
app = FastAPI(
    title="Temporary CV Processing API",
    description="Simple CV processing for testing",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {".pdf", ".docx"}

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "Temporary CV Processing API",
        "version": "1.0.0",
        "status": "active"
    }

@app.get("/health", response_model=HealthCheck, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthCheck(
        status="healthy",
        timestamp=datetime.now()
    )

@app.post("/upload", response_model=UploadResponse, tags=["Upload"])
async def upload_cv(file: UploadFile = File(...)):
    """Upload and process CV file."""
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        # Check file size
        file_content = await file.read()
        if len(file_content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File too large")
        
        logger.info(f"Processing CV: {file.filename}")
        logger.info(f"File size: {len(file_content)} bytes")
        logger.info(f"File extension: {file_ext}")
        
        # Save file temporarily
        logger.info("Extracting text from document...")
        temp_file_path = f"/tmp/{file.filename}"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(file_content)
        
        # Extract text
        text = extract_text(temp_file_path)
        if text:
            logger.info(f"Extracted {len(text)} characters of text")
            candidate_name = extract_candidate_name(text)
        else:
            logger.warning("Could not extract text, falling back to filename")
            candidate_name = file.filename.split('.')[0].replace('_', ' ')
        
        # Generate candidate ID
        candidate_id = f"temp-{uuid.uuid4().hex[:8]}"
        
        # Extract summary and skills
        summary = extract_summary(text, candidate_name)
        skills_data = extract_skills_from_text(text)
        skills = [skill['name'] for skill in skills_data]
        
        logger.info(f"Extracted summary: {summary[:100]}...")
        
        # Clean up temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
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
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True) 