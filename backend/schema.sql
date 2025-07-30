-- Cloud Spanner Schema for Candidate Search System

-- Candidates table
CREATE TABLE candidates (
    candidate_id STRING(100) NOT NULL,
    name STRING(200) NOT NULL,
    email STRING(200),
    phone STRING(50),
    summary STRING(MAX),
    gcs_url STRING(500),
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP,
    skills ARRAY<STRING(100)>,
    embedding_id STRING(100),
    similarity_score FLOAT64,
) PRIMARY KEY (candidate_id);

-- Skills table for tracking all skills
CREATE TABLE skills (
    skill_id STRING(100) NOT NULL,
    name STRING(100) NOT NULL,
    candidate_count INT64 NOT NULL,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP,
) PRIMARY KEY (skill_id);

-- Create indexes for better query performance
CREATE INDEX idx_candidates_name ON candidates(name);
CREATE INDEX idx_candidates_created_at ON candidates(created_at DESC);
CREATE INDEX idx_candidates_skills ON candidates(skills);
CREATE INDEX idx_skills_name ON skills(name);
CREATE INDEX idx_skills_count ON skills(candidate_count DESC); 