# RAG-Enhanced Candidate Search System

## Current Active Files

### Core Application Files:
- **`gcp_main_rag.py`** - Main FastAPI server with RAG integration
- **`rag_enhanced.py`** - RAG system implementation with vector search and LLM
- **`start_rag_server.py`** - Startup script for the RAG server

### Utility Files:
- **`create_gcp_resumes.py`** - Script to generate test resumes with GCP skills
- **`requirements.txt`** - Python dependencies
- **`schema.sql`** - Database schema
- **`config.py`** - Configuration settings
- **`check_db.py`** - Database connectivity checker

### Infrastructure:
- **`Dockerfile`** - Docker container configuration
- **`.dockerignore`** - Docker ignore file
- **`.env`** - Environment variables

### Data:
- **`Resumes/`** - Directory containing generated test resumes

## Archived Files

All old/unused files have been moved to `old_files/` directory:
- Old server files (`gcp_main.py`, `main.py`, etc.)
- Old CV processors and database utilities
- Test files and debug scripts
- Old resume generation scripts
- Documentation files

## How to Run

### Start the RAG Server:
```bash
cd backend
GOOGLE_API_KEY=your-api-key USE_LLM=true source venv/bin/activate
uvicorn gcp_main_rag:app --host 0.0.0.0 --port 8000
```

### Or use the startup script:
```bash
cd backend
GOOGLE_API_KEY=your-api-key USE_LLM=true source venv/bin/activate
python start_rag_server.py
```

### Generate Test Resumes:
```bash
cd backend
source venv/bin/activate
python create_gcp_resumes.py
```

## Features

- **RAG-Enhanced Search**: Combines vector similarity and keyword matching
- **LLM Integration**: Uses Google Gemini for query understanding and response generation
- **CV Download**: Direct file download from Google Cloud Storage
- **Multi-Skill Search**: Find candidates with multiple skills (AND logic)
- **Fuzzy Search**: Levenshtein distance-based text similarity
- **Real-time Processing**: Upload and process CVs with skill extraction

## Frontend

The React frontend is located in `../frontend/` and includes:
- Search interface with threshold controls
- Candidate management
- CV download functionality
- Real-time search results 