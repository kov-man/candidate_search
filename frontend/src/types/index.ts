export interface Candidate {
  candidate_id: string
  name: string
  email?: string
  phone?: string
  summary?: string
  created_at: string
  gcs_url?: string
  skills?: string[]
}

export interface Skill {
  skill_id: string
  name: string
}

export interface CandidateSkill {
  candidate_id: string
  skill_id: string
  confidence: number
}

export interface SearchResult {
  candidate_id: string
  candidate_name: string
  email?: string
  phone?: string
  summary?: string
  similarity_score: number
  skills: string[]
  relevant_text?: string
  gcs_url?: string
}

export interface SearchResponse {
  query: string
  results: SearchResult[]
  total_results: number
  rag_answer?: string
  search_method?: string
}

export interface UploadResponse {
  message: string
  candidate_id: string
  filename: string
  skills_extracted: string[]
}

export interface SearchQuery {
  query: string
  top_k?: number
  similarity_threshold?: number
}

export interface HealthCheck {
  status: string
  timestamp: string
  services: {
    database: boolean
    vertex_ai: boolean
    cloud_storage: boolean
    matching_engine: boolean
  }
} 