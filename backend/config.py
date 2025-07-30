"""Config settings for the app."""

import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class Settings:
    """App settings from environment."""
    
    # GCP stuff
    PROJECT_ID: str = os.getenv("PROJECT_ID", "")
    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    VERTEX_AI_LOCATION: str = os.getenv("VERTEX_AI_LOCATION", "us-central1")
    
    # Storage
    GCS_BUCKET: str = os.getenv("GCS_BUCKET", "candidate-cv-bucket")
    
    # Spanner
    SPANNER_INSTANCE: str = os.getenv("SPANNER_INSTANCE", "")
    SPANNER_DATABASE: str = os.getenv("SPANNER_DATABASE", "")
    
    # AI models
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-004")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-1.5-flash")
    MATCHING_ENGINE_INDEX_ENDPOINT: str = os.getenv("MATCHING_ENGINE_INDEX_ENDPOINT", "")
    
    # App limits
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: set = {".pdf", ".docx"}
    
    def validate(self) -> bool:
        """Check if required settings are present."""
        required = [
            self.PROJECT_ID,
            self.SPANNER_INSTANCE,
            self.SPANNER_DATABASE,
            self.GCS_BUCKET
        ]
        return all(required)

settings = Settings() 