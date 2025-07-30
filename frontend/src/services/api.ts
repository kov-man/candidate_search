import axios from 'axios'
import type { 
  SearchQuery, 
  SearchResponse, 
  UploadResponse, 
  Candidate, 
  HealthCheck 
} from '../types'

const api = axios.create({
  baseURL: 'http://localhost:8000',
  headers: {
    'Content-Type': 'application/json',
  },
})

export const apiService = {
  // health check
  async healthCheck(): Promise<HealthCheck> {
    const response = await api.get<HealthCheck>('/health')
    return response.data
  },

  // upload CV file
  async uploadCV(file: File): Promise<UploadResponse> {
    const formData = new FormData()
    formData.append('file', file)
    
    const response = await api.post<UploadResponse>('/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    return response.data
  },

  // search for candidates
  async searchCandidates(query: SearchQuery): Promise<SearchResponse> {
    const response = await api.post<SearchResponse>('/search', query)
    return response.data
  },

  // get single candidate
  async getCandidate(candidateId: string): Promise<Candidate> {
    const response = await api.get<Candidate>(`/candidates/${candidateId}`)
    return response.data
  },

  // delete a candidate
  async deleteCandidate(candidateId: string): Promise<{ success: boolean; message: string; candidate_id: string; file_deleted: boolean }> {
    const response = await api.delete(`/candidates/${candidateId}`)
    return response.data
  },

  // get all candidates  
  async listCandidates(skip: number = 0, limit: number = 20): Promise<{
    candidates: Candidate[]
    skip: number
    limit: number
    total: number
  }> {
    const response = await api.get('/candidates', {
      params: { skip, limit }
    })
    return response.data
  },

  // get skills list
  async listSkills(): Promise<{ skills: { name: string; candidate_count: number }[] }> {
    const response = await api.get('/skills')
    return response.data
  },

  // download CV file
  async downloadCV(candidateId: string): Promise<Blob> {
    const response = await api.get(`/download/${candidateId}`, {
      responseType: 'blob'
    })
    return response.data
  },
} 