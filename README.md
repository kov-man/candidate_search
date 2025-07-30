# Candidate Search Application

A candidate search system that allows you to upload resumes, extract skills, and search for candidates using natural language queries. Built with FastAPI, React, and Google Cloud Platform.

## Features

### What's Actually Working
- **Resume Upload**: Upload PDF and DOCX resumes with automatic text extraction
- **Skill Extraction**: Extract technical skills from resumes using keyword matching
- **Basic Search**: Search candidates using text queries
- **Candidate Management**: View, list, and delete candidate profiles
- **File Storage**: Store resume files in Google Cloud Storage
- **Database**: Store candidate data in Cloud Spanner

### Experimental Features (May Not Work Reliably)
- **Semantic Search**: Vector-based similarity search (requires proper setup)
- **RAG Integration**: AI-powered search responses (requires Gemini API key)
- **Similarity Filtering**: Filter results by similarity threshold

## Architecture

```
Frontend (React + TypeScript)
    ↓
Backend (FastAPI)
    ↓
Google Cloud Platform
├── Cloud Storage (Resume files)
├── Cloud Spanner (Database)
└── Optional: Vertex AI (if configured)
```

## Prerequisites

### Required Software
- Python 3.10+
- Node.js 18+
- npm

### Google Cloud Setup (Basic)
1. Create a GCP project
2. Enable Cloud Storage API and Cloud Spanner API
3. Create a service account with Storage and Spanner permissions
4. Download service account key file

### Environment Variables
Create a `.env` file in the `backend/` directory:

```env
PROJECT_ID=your-gcp-project-id
GOOGLE_APPLICATION_CREDENTIALS=path/to/your/service-account-key.json
GOOGLE_API_KEY=your-google-api-key (optional, for AI features)
```

**Note**: The system will work with basic text search even without the GOOGLE_API_KEY.

## Installation

### Backend Setup

1. **Navigate to backend directory**:
   ```bash
   cd backend
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv venvir
   source venvir/bin/activate  # On Windows: venvir\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up database** (manual step):
   - Create a Cloud Spanner instance and database in GCP Console
   - Run the SQL schema from `schema.sql`

### Frontend Setup

1. **Navigate to frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

## Running the Application

### Start Backend Server

1. **Activate virtual environment**:
   ```bash
   cd backend
   source venvir/bin/activate
   ```

2. **Start the FastAPI server**:
   ```bash
   python gcp_main_rag.py
   ```

   The backend will be available at: `http://localhost:8000`

### Start Frontend Application

1. **In a new terminal, navigate to frontend**:
   ```bash
   cd frontend
   ```

2. **Start the development server**:
   ```bash
   npm run dev
   ```

   The frontend will be available at: `http://localhost:3000`

## Usage Guide

### 1. Upload Resumes
- Click "Upload CV" in the navigation
- Drag and drop PDF or DOCX files
- The system extracts text and identifies technical skills
- View the extracted information

### 2. Search Candidates
- Use the search page to find candidates
- Try queries like:
  - "Python developer"
  - "React experience"
  - "Machine learning"
- View results with basic relevance scoring

### 3. Manage Candidates
- Browse all candidates in the candidates page
- View individual candidate profiles
- Delete candidates when needed
- See extracted skills lists

## API Documentation

Once the backend is running, visit:
- **Interactive API Docs**: `http://localhost:8000/docs`

### Available Endpoints
- `POST /upload` - Upload resume files
- `POST /search` - Search candidates with query
- `GET /candidates` - List all candidates
- `GET /candidates/{id}` - Get specific candidate
- `DELETE /candidates/{id}` - Delete candidate
- `GET /skills` - List all extracted skills
- `GET /download/{id}` - Download resume file

## Development

### Project Structure
```
├── backend/
│   ├── gcp_main_rag.py      # Main FastAPI application
│   ├── rag_enhanced.py      # Search and AI logic (experimental)
│   ├── config.py            # Configuration settings
│   ├── requirements.txt     # Python dependencies
│   └── schema.sql           # Database schema
├── frontend/
│   ├── src/
│   │   ├── components/      # React components
│   │   ├── pages/          # Page components
│   │   ├── services/       # API services
│   │   └── types/          # TypeScript types
│   └── package.json        # Node.js dependencies
└── Resumes/                # Sample resume files
```

### Building for Production

**Frontend**:
```bash
cd frontend
npm run build
```

The built files will be in the `dist/` directory.

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure virtual environment is activated
2. **GCP Authentication**: Verify service account key path
3. **Database Connection**: Check Spanner instance exists
4. **Port Conflicts**: Backend uses 8000, frontend uses 3000
5. **Search Not Working**: Check if GOOGLE_API_KEY is set for AI features

### What To Expect

**Working Features:**
- File upload and text extraction
- Basic keyword-based search
- Candidate management (add/view/delete)
- Skills extraction from resumes

**Experimental Features:**
- Semantic search may require additional setup
- AI responses depend on valid Gemini API key
- Vector search needs proper initialization

## Known Limitations

- Search is primarily keyword-based, not true semantic search
- AI features require proper API keys and may not work consistently
- No user authentication or multi-tenancy
- Limited file format support (PDF, DOCX only)
- No data export functionality

## Contributing

1. Fork the repository
2. Create a feature branch
3. Test your changes with the basic upload/search flow
4. Submit a pull request

## License

This project is licensed under the MIT License. 