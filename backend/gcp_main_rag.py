#!/usr/bin/env python3
"""
Candidate Search API with RAG Integration
"""

import os
import logging
import uuid
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import google.generativeai as genai
from google.cloud import spanner
from google.cloud import storage

from rag_enhanced import RAGEnhancer, initialize_rag_system

# logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GCP config
PROJECT_ID = "niky-search-demo"
INSTANCE_ID = "niky-search-spanner"
DATABASE_ID = "candidates-db"
BUCKET_NAME = "candidate-cv-bucket-niky-search-demo"

# init GCP clients
spanner_client = spanner.Client(project=PROJECT_ID)
instance = spanner_client.instance(INSTANCE_ID)
database = instance.database(DATABASE_ID)
storage_client = storage.Client(project=PROJECT_ID)
bucket = storage_client.bucket(BUCKET_NAME)

# gemini setup
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logger.warning("GOOGLE_API_KEY not found")
    GOOGLE_API_KEY = "your-api-key-here"

genai.configure(api_key=GOOGLE_API_KEY)

# global rag instance
rag_enhancer = None

# models
class UploadResponse(BaseModel):
    success: bool
    message: str
    candidate_id: str = None
    candidate_name: str = None
    filename: str = None
    skills_extracted: List[str] = []
    skills_found: int = 0
    chunks_processed: int = 0

class HealthCheck(BaseModel):
    status: str
    timestamp: datetime
    services: Dict[str, bool]

class Candidate(BaseModel):
    candidate_id: str
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    summary: Optional[str] = None
    created_at: datetime
    gcs_url: Optional[str] = None
    skills: List[str] = []

class CandidateSkill(BaseModel):
    name: str
    confidence: float

class SearchQuery(BaseModel):
    query: str
    top_k: int = 10
    similarity_threshold: float = 0.7

class SearchResult(BaseModel):
    candidate_id: str
    candidate_name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    summary: Optional[str] = None
    similarity_score: float
    skills: List[str]
    relevant_text: Optional[str] = None
    gcs_url: Optional[str] = None

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int
    rag_answer: Optional[str] = None
    search_method: str = "hybrid"

def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
            return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return ""

def extract_text_from_docx(docx_path: str) -> str:
    try:
        from docx import Document
        doc = Document(docx_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting text from DOCX: {e}")
        return ""

def extract_text(file_path: str) -> str:
    file_path = Path(file_path)
    if file_path.suffix.lower() == '.pdf':
        return extract_text_from_pdf(str(file_path))
    elif file_path.suffix.lower() == '.docx':
        return extract_text_from_docx(str(file_path))
    else:
        logger.error(f"Unsupported file type: {file_path.suffix}")
        return ""

def extract_candidate_name(text: str) -> str:
    """extract name from CV text"""
    if not text:
        return "Unknown Candidate"
    
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if not lines:
        return "Unknown Candidate"
    
    # try to find name in first few lines
    for i, line in enumerate(lines[:10]):
        # skip obvious non-names
        if any(keyword in line.lower() for keyword in ['resume', 'curriculum', 'vitae', 'cv', 'phone', 'email', 'address', 'linkedin', 'education', 'experience', 'skills']):
            continue
        
        # clean up line
        cleaned_line = re.sub(r'([a-z])([A-Z])', r'\1 \2', line)
        cleaned_line = re.sub(r'(\d)([A-Za-z])', r'\1 \2', cleaned_line)
        cleaned_line = re.sub(r'([A-Za-z])(\d)', r'\1 \2', cleaned_line)
        cleaned_line = re.sub(r'\s+', ' ', cleaned_line).strip()
        
        # name patterns
        name_patterns = [
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})(?:\s|$)',
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})(?:\s*[-–—])',
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})(?:\s*[A-Z])',
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, cleaned_line)
            if match:
                name = match.group(1).strip()
                words = name.split()
                # basic validation
                if 2 <= len(words) <= 4 and 4 <= len(name) <= 50:
                    valid_name_words = all(
                        len(word) >= 2 and 
                        word[0].isupper() and 
                        word[1:].islower() 
                        for word in words
                    )
                    if valid_name_words:
                        return name
    
    # fallback to first line
    first_line = lines[0] if lines else ""
    if first_line:
        name_match = re.match(r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})', first_line)
        if name_match:
            name = name_match.group(1).strip()
            words = name.split()
            if 2 <= len(words) <= 4:
                return name
    
    return "Unknown Candidate"

def extract_summary(text: str, candidate_name: str) -> str:
    try:
        # use gemini for summary
        prompt = f"""
        Extract a professional summary from this CV text for {candidate_name}.
        
        CV Text:
        {text[:2000]}
        
        Provide a concise summary (2-3 sentences) including:
        1. Key skills and expertise
        2. Years of experience  
        3. Notable achievements
        
        Keep it professional and focused.
        """
        
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        response = gemini_model.generate_content(prompt)
        
        if response and response.text:
            return response.text.strip()
        else:
            return extract_summary_fallback(text, candidate_name)
            
    except Exception as e:
        logger.error(f"Error extracting summary with Gemini: {e}")
        return extract_summary_fallback(text, candidate_name)

def extract_summary_fallback(text: str, candidate_name: str) -> str:
    # remove name from beginning
    text_without_name = re.sub(f'^{re.escape(candidate_name)}', '', text, flags=re.IGNORECASE)
    
    # find first meaningful paragraph
    paragraphs = [p.strip() for p in text_without_name.split('\n\n') if p.strip()]
    
    for paragraph in paragraphs[:3]:
        if len(paragraph) > 50 and len(paragraph) < 500:
            cleaned = re.sub(r'\s+', ' ', paragraph).strip()
            if cleaned and not cleaned.startswith(('Email:', 'Phone:', 'Address:', 'LinkedIn:')):
                return cleaned
    
    # fallback to first 200 chars
    return text_without_name[:200].strip()

def extract_skills_from_text(text: str) -> List[Dict[str, Any]]:
    try:
        prompt = f"""
        Extract technical skills from this CV text.
        
        CV Text:
        {text[:1500]}
        
        Identify:
        1. Programming languages
        2. Frameworks and technologies
        3. Tools and platforms
        4. Soft skills
        5. Certifications
        
        Return only skill names, one per line, no numbering.
        Focus on technical and professional skills.
        """
        
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        response = gemini_model.generate_content(prompt)
        
        if response and response.text:
            skills = []
            for line in response.text.strip().split('\n'):
                skill = line.strip().strip('- ').strip()
                if skill and len(skill) > 1:
                    skills.append({
                        'name': skill,
                        'confidence': 0.9
                    })
            return skills
        else:
            return extract_skills_fallback(text)
            
    except Exception as e:
        logger.error(f"Error extracting skills with Gemini: {e}")
        return extract_skills_fallback(text)

def extract_skills_fallback(text: str) -> List[Dict[str, Any]]:
    # common skills
    technical_skills = [
        'Python', 'JavaScript', 'Java', 'C++', 'C#', 'Go', 'Rust', 'PHP', 'Ruby', 'Swift',
        'React', 'Angular', 'Vue', 'Node.js', 'Django', 'Flask', 'Spring', 'Laravel',
        'AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes', 'Jenkins', 'Git', 'GitHub',
        'MySQL', 'PostgreSQL', 'MongoDB', 'Redis', 'Elasticsearch', 'Kafka',
        'Machine Learning', 'AI', 'Data Science', 'NLP', 'Computer Vision',
        'DevOps', 'CI/CD', 'Agile', 'Scrum', 'Project Management'
    ]
    
    skills = []
    text_lower = text.lower()
    
    for skill in technical_skills:
        if skill.lower() in text_lower:
            skills.append({
                'name': skill,
                'confidence': 0.8
            })
    
    return skills

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
            
        logger.info(f"Candidate inserted: {candidate_data['candidate_id']}")
        
        # update skills table
        update_skills_table(candidate_data['skills'])
        
        # rebuild vector index
        rag_enhancer.build_vector_index()
        
    except Exception as e:
        logger.error(f"Error inserting candidate: {e}")
        raise

def update_skills_table(skills: List[str]) -> None:
    try:
        for skill in skills:
            # check if skill exists
            query = "SELECT skill_id, candidate_count FROM skills WHERE name = @skill_name"
            
            with database.snapshot() as snapshot:
                results = snapshot.execute_sql(
                    query,
                    params={'skill_name': skill},
                    param_types={'skill_name': spanner.param_types.STRING}
                )
                
                if results:
                    # update count
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
                    # new skill
                    skill_id = f"skill-{skill.lower().replace(' ', '-')}-{uuid.uuid4().hex[:8]}"
                    
                    with database.batch() as batch:
                        batch.insert(
                            table='skills',
                            columns=('skill_id', 'name', 'candidate_count', 'created_at'),
                            values=[(skill_id, skill, 1, datetime.now())]
                        )
        
        logger.info(f"Skills updated: {skills}")
    except Exception as e:
        logger.error(f"Error updating skills: {e}")

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
        logger.error(f"Error getting candidate by ID: {e}")
        return None

# FastAPI app
app = FastAPI(
    title="RAG Candidate Search API",
    description="GCP integration with RAG using Gemini",
    version="3.0.0"
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

@app.on_event("startup")
async def startup_event():
    global rag_enhancer
    try:
        logger.info("Starting RAG system...")
        initialize_rag_system()
        from rag_enhanced import rag_enhancer as rag_enhancer_instance
        rag_enhancer = rag_enhancer_instance
        logger.info("RAG system ready")
    except Exception as e:
        logger.error(f"Error initializing RAG: {e}")

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "RAG Candidate Search API",
        "version": "3.0.0",
        "docs": "/docs",
        "gcp_services": {
            "cloud_storage": True,
            "cloud_spanner": True,
            "vertex_ai": True
        },
        "rag_features": {
            "gemini_integration": True,
            "vector_search": True,
            "semantic_search": True,
            "llm_generation": True
        }
    }

@app.get("/health", response_model=HealthCheck, tags=["Health"])
async def health_check():
    # check GCP services
    gcs_healthy = False
    spanner_healthy = False
    rag_healthy = False
    
    try:
        bucket.reload()
        gcs_healthy = True
    except Exception as e:
        logger.error(f"GCS health check failed: {e}")
    
    try:
        with database.snapshot() as snapshot:
            snapshot.execute_sql("SELECT 1")
        spanner_healthy = True
    except Exception as e:
        logger.error(f"Spanner health check failed: {e}")
    
    try:
        if len(rag_enhancer.candidate_vectors) > 0:
            rag_healthy = True
    except Exception as e:
        logger.error(f"RAG health check failed: {e}")
    
    return HealthCheck(
        status="healthy" if all([gcs_healthy, spanner_healthy, rag_healthy]) else "degraded",
        timestamp=datetime.now(),
        services={
            "gcs": gcs_healthy,
            "spanner": spanner_healthy,
            "rag": rag_healthy
        }
    )

@app.post("/upload", response_model=UploadResponse, tags=["Upload"])
async def upload_cv(file: UploadFile = File(...)):
    try:
        # validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400, 
                detail=f"File type not allowed. Allowed: {ALLOWED_EXTENSIONS}"
            )
        
        # read file
        file_content = await file.read()
        if len(file_content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File too large")
        
        # save temp
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(file_content)
        
        # extract text
        text = extract_text(temp_path)
        if text:
            logger.info(f"Extracted {len(text)} characters")
            candidate_name = extract_candidate_name(text)
            logger.info(f"Candidate name: {candidate_name}")
        else:
            logger.warning("Could not extract text, using filename")
            candidate_name = file.filename.split('.')[0].replace('CV - ', '').replace('_', ' ')
        
        # fallback name extraction from filename
        if candidate_name == "Unknown Candidate" or len(candidate_name.split()) < 2:
            filename_name = file.filename.replace('_detailed_resume.pdf', '').replace('_', ' ').title()
            if len(filename_name.split()) >= 2:
                candidate_name = filename_name
                logger.info(f"Using filename for name: {candidate_name}")
        
        # generate ID
        candidate_id = f"candidate-{uuid.uuid4().hex[:8]}-{datetime.now().strftime('%Y%m%d%H%M')}"
        
        # extract summary
        summary = extract_summary(text, candidate_name)
        logger.info(f"Generated summary: {len(summary)} chars")
        
        # extract skills
        skills_data = extract_skills_from_text(text)
        skills = [skill['name'] for skill in skills_data]
        logger.info(f"Extracted {len(skills)} skills")
        
        # upload to GCS
        gcs_url = upload_to_gcs(file_content, file.filename)
        
        # prepare data
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
        
        # insert to db
        insert_candidate(candidate_data)
        
        return UploadResponse(
            success=True,
            message=f"Successfully processed CV for {candidate_name}",
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
        logger.error(f"Error processing CV: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing CV: {str(e)}")
    finally:
        # cleanup
        if 'temp_path' in locals():
            try:
                os.remove(temp_path)
            except:
                pass

@app.get("/candidates", tags=["Candidates"])
async def list_candidates(skip: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=100)):
    candidates = get_candidates(skip, limit)
    return {"candidates": candidates, "total": len(candidates)}

@app.get("/candidates/{candidate_id}", response_model=Candidate, tags=["Candidates"])
async def get_candidate(candidate_id: str):
    candidate = get_candidate_by_id(candidate_id)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return candidate

@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search_candidates_endpoint(query: SearchQuery):
    try:
        logger.info(f"Search query: {query.query}")
        
        # semantic search
        candidates = rag_enhancer.semantic_search_with_similarity(
            query.query, 
            query.top_k, 
            query.similarity_threshold
        )
        search_method = "semantic_with_similarity"
        
        # generate RAG response
        try:
            if candidates:
                rag_answer = rag_enhancer.generate_rag_response(query.query, candidates)
            else:
                rag_answer = f"No candidates found matching '{query.query}'. Try broader search terms."
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            rag_answer = f"Found {len(candidates)} candidates matching '{query.query}'."
        
        # extract relevant text
        for candidate in candidates:
            if candidate.get('summary'):
                relevant_text = rag_enhancer._extract_relevant_text_fallback(candidate['summary'], query.query)
                candidate['relevant_text'] = relevant_text
        
        logger.info(f"Found {len(candidates)} candidates using {search_method}")
        
        results = []
        for i, candidate in enumerate(candidates):
            try:
                result = SearchResult(
                    candidate_id=str(candidate['candidate_id']),
                    candidate_name=str(candidate['name']),
                    email=str(candidate.get('email', '')) if candidate.get('email') else None,
                    phone=str(candidate.get('phone', '')) if candidate.get('phone') else None,
                    summary=str(candidate.get('summary', '')) if candidate.get('summary') else None,
                    similarity_score=candidate.get('similarity_score', 0.9 - (i * 0.1)),
                    skills=list(candidate.get('skills', [])) if candidate.get('skills') else [],
                    relevant_text=candidate.get('relevant_text', f"Relevant text from {candidate['name']}'s CV..."),
                    gcs_url=candidate.get('gcs_url')
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing candidate {i}: {e}")
                continue
        
        return SearchResponse(
            query=query.query,
            results=results,
            total_results=len(results),
            rag_answer=rag_answer,
            search_method=search_method
        )
    except Exception as e:
        logger.error(f"Error in search: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/fuzzy", response_model=SearchResponse, tags=["Search"])
async def fuzzy_search_candidates_endpoint(query: SearchQuery):
    try:
        logger.info(f"Fuzzy search: {query.query}")
        
        # fuzzy search
        candidates = rag_enhancer.fuzzy_search(
            query.query, 
            query.top_k, 
            query.similarity_threshold
        )
        search_method = "fuzzy_search"
        
        rag_answer = None
        
        # extract relevant text
        for candidate in candidates:
            if candidate.get('summary'):
                relevant_text = rag_enhancer._extract_relevant_text_fallback(candidate['summary'], query.query)
                candidate['relevant_text'] = relevant_text
        
        logger.info(f"Found {len(candidates)} candidates using {search_method}")
        
        results = []
        for i, candidate in enumerate(candidates):
            try:
                result = SearchResult(
                    candidate_id=str(candidate['candidate_id']),
                    candidate_name=str(candidate['name']),
                    email=str(candidate.get('email', '')) if candidate.get('email') else None,
                    phone=str(candidate.get('phone', '')) if candidate.get('phone') else None,
                    summary=str(candidate.get('summary', '')) if candidate.get('summary') else None,
                    similarity_score=candidate.get('similarity_score', 0.9 - (i * 0.1)),
                    skills=list(candidate.get('skills', [])) if candidate.get('skills') else [],
                    relevant_text=candidate.get('relevant_text', f"Relevant text from {candidate['name']}'s CV..."),
                    gcs_url=candidate.get('gcs_url')
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing candidate {i}: {e}")
                continue
        
        return SearchResponse(
            query=query.query,
            results=results,
            total_results=len(results),
            rag_answer=rag_answer,
            search_method=search_method
        )
    except Exception as e:
        logger.error(f"Error in fuzzy search: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/candidates/{candidate_id}", tags=["Candidates"])
async def delete_candidate(candidate_id: str):
    try:
        # get candidate
        candidate = get_candidate_by_id(candidate_id)
        if not candidate:
            raise HTTPException(status_code=404, detail="Candidate not found")
        
        gcs_url = candidate.get('gcs_url')
        
        # delete from GCS
        if gcs_url and gcs_url.startswith('gs://'):
            try:
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
        
        # delete from spanner
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
        
        # rebuild index
        try:
            rag_enhancer.build_vector_index()
            logger.info("Rebuilt vector index")
        except Exception as e:
            logger.error(f"Error rebuilding vector index: {e}")
        
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

@app.get("/download/{candidate_id}", tags=["Download"])
async def download_cv(candidate_id: str):
    try:
        # get candidate
        candidate = get_candidate_by_id(candidate_id)
        if not candidate:
            raise HTTPException(status_code=404, detail="Candidate not found")
        
        gcs_url = candidate.get('gcs_url')
        if not gcs_url:
            raise HTTPException(status_code=404, detail="CV file not found")
        
        # parse GCS URL
        if not gcs_url.startswith('gs://'):
            raise HTTPException(status_code=400, detail="Invalid GCS URL")
        
        parts = gcs_url.replace('gs://', '').split('/', 1)
        if len(parts) != 2:
            raise HTTPException(status_code=400, detail="Invalid GCS URL")
        
        bucket_name, blob_name = parts
        
        # get file
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        if not blob.exists():
            raise HTTPException(status_code=404, detail="CV file not found in GCS")
        
        # download
        file_content = blob.download_as_bytes()
        filename = blob_name.split('/')[-1]
        
        # return file
        from fastapi.responses import Response
        return Response(
            content=file_content,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Type": "application/pdf"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading CV: {e}")
        raise HTTPException(status_code=500, detail=f"Error downloading CV: {str(e)}")

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
        logger.error(f"Error listing skills: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True) 