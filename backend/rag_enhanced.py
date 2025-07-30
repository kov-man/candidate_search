#!/usr/bin/env python3
"""
RAG System for Candidate Search
Vector search and LLM integration for semantic candidate matching.
"""

import os
import logging
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import uuid
import re
import difflib

import google.generativeai as genai
from google.cloud import spanner
from google.cloud import storage
import spacy
from sentence_transformers import SentenceTransformer
import faiss

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GCP project settings
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

# LLM configuration
USE_LLM = os.getenv('USE_LLM', 'true').lower() == 'true'

# Setup Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logger.warning("GOOGLE_API_KEY not found. LLM features will be disabled.")
    USE_LLM = False

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    logger.info("Gemini API configured successfully")

# Load NLP models
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("spaCy model loaded successfully")
except OSError:
    logger.warning("spaCy model not found. Installing...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")
except ImportError:
    logger.warning("spaCy not available. Continuing without NLP features.")
    nlp = None

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
logger.info("Sentence transformer model loaded")

# Setup FAISS index for vector search
VECTOR_DIMENSION = 384  # Dimension of all-MiniLM-L6-v2 embeddings
faiss_index = faiss.IndexFlatIP(VECTOR_DIMENSION)  # Inner product for cosine similarity
logger.info("FAISS index initialized")

class RAGEnhancer:
    """RAG system for candidate search with vector embeddings and LLM integration."""
    
    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        if use_llm:
            try:
                self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                self.gemini_model.timeout = 20
                logger.info("Gemini 1.5 Flash LLM initialized successfully")
            except Exception as e:
                logger.warning(f"Could not initialize Gemini model: {e}")
                self.gemini_model = None
        else:
            self.gemini_model = None
            logger.info("LLM disabled, using fallback responses only")
        
        self.candidate_embeddings = {}  # Store embeddings in memory
        self.candidate_vectors = []  # Store vectors for FAISS
        self.candidate_ids = []  # Store candidate IDs for mapping
        
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using sentence transformer."""
        try:
            embedding = embedding_model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return [0.0] * VECTOR_DIMENSION

    def generate_gemini_embedding(self, text: str) -> List[float]:
        """Generate embedding using Gemini API as fallback."""
        try:
            # For now, use sentence transformer as backup
            return self.generate_embedding(text)
        except Exception as e:
            logger.error(f"Error generating Gemini embedding: {e}")
            return self.generate_embedding(text)

    def extract_candidate_text_for_embedding(self, candidate: Dict[str, Any]) -> str:
        """Extract text from candidate data for embedding generation."""
        text_parts = []
        
        # Add name
        if candidate.get('name'):
            text_parts.append(f"Name: {candidate['name']}")
        
        # Add skills  
        if candidate.get('skills'):
            skills_text = " ".join(candidate['skills'])
            text_parts.append(f"Skills: {skills_text}")
        
        # Add summary
        if candidate.get('summary'):
            text_parts.append(f"Summary: {candidate['summary']}")
        
        # Add email domain if available
        if candidate.get('email'):
            email_domain = candidate['email'].split('@')[-1] if '@' in candidate['email'] else ''
            if email_domain:
                text_parts.append(f"Email domain: {email_domain}")
        
        return " | ".join(text_parts)
    
    def build_vector_index(self):
        """Build FAISS index from all candidates in database."""
        try:
            logger.info("Building vector index from database...")
            
            # Get all candidates
            query = """
            SELECT candidate_id, name, email, phone, summary, skills, gcs_url
            FROM candidates
            """
            
            with database.snapshot() as snapshot:
                results = snapshot.execute_sql(query)
                
                self.candidate_vectors = []
                self.candidate_ids = []
                
                for row in results:
                    candidate = {
                        'candidate_id': str(row[0]),
                        'name': str(row[1]),
                        'email': str(row[2]) if row[2] else None,
                        'phone': str(row[3]) if row[3] else None,
                        'summary': str(row[4]) if row[4] else "",
                        'skills': list(row[5]) if row[5] else [],
                        'gcs_url': str(row[6]) if row[6] else None
                    }
                    
                    # Generate text for embedding
                    candidate_text = self.extract_candidate_text_for_embedding(candidate)
                    
                    # Generate embedding
                    embedding = self.generate_embedding(candidate_text)
                    
                    # Store for FAISS
                    self.candidate_vectors.append(embedding)
                    self.candidate_ids.append(candidate['candidate_id'])
                    
                    # Store full candidate data
                    self.candidate_embeddings[candidate['candidate_id']] = candidate
                
                # Build FAISS index
                if self.candidate_vectors:
                    vectors_array = np.array(self.candidate_vectors, dtype=np.float32)
                    faiss_index.reset()
                    faiss_index.add(vectors_array)
                    logger.info(f"Built FAISS index with {len(self.candidate_vectors)} candidates")
                    logger.info(f"Stored {len(self.candidate_ids)} candidate IDs")
                    logger.info(f"Stored {len(self.candidate_embeddings)} candidate embeddings")
                else:
                    logger.warning("No candidates found for vector index")
                    
        except Exception as e:
            logger.error(f"Error building vector index: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def semantic_search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Perform semantic search using vector similarity."""
        try:
            # Check if we have candidates
            if not self.candidate_vectors or not self.candidate_ids:
                logger.warning("No candidate vectors available for semantic search")
                return []
            
            # Generate query embedding
            query_embedding = np.array([self.generate_embedding(query)], dtype=np.float32)
            
            # Check if FAISS index is empty
            if faiss_index.ntotal == 0:
                logger.warning("FAISS index is empty")
                return []
            
            # Perform search
            top_k = min(top_k, len(self.candidate_vectors))
            if top_k <= 0:
                logger.warning("No vectors to search")
                return []
            
            # Search in FAISS
            similarities, indices = faiss_index.search(query_embedding, top_k)
            
            # Format results
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx < len(self.candidate_ids):  # Valid index
                    candidate_id = self.candidate_ids[idx]
                    results.append((candidate_id, float(similarity)))
            
            logger.info(f"Semantic search found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []
    
    def generate_rag_response(self, query: str, candidates: List[Dict[str, Any]]) -> str:
        """Generate contextual response using Gemini LLM with fallback."""
        try:
            if not candidates:
                return f"No candidates found matching '{query}'."
            
            # Check if Gemini model is available
            if not self.gemini_model:
                return self._generate_fallback_response(query, candidates)
            
            # Prepare detailed context for LLM
            context_parts = []
            for i, candidate in enumerate(candidates[:5]):  # Limit to top 5 for context
                skills_str = ', '.join(candidate.get('skills', []))
                summary = candidate.get('summary', '')[:300]  # Get more summary text
                similarity = candidate.get('similarity_score', 0)
                match_percentage = int(similarity * 100)
                
                role_line = summary.split('|')[0] if '|' in summary else summary.split('\n')[0]
                candidate_info = f"""
Candidate {i+1}: {candidate['name']} (Match: {match_percentage}%)
Role: {role_line}
Skills: {skills_str}
Experience: {summary}
"""
                context_parts.append(candidate_info)
            
            context = "\n".join(context_parts)
            
            # Create enhanced prompt for Gemini
            prompt = f"""
You are an expert HR recruiter and talent acquisition specialist. Analyze the following candidates based on the user's search query and provide a detailed, insightful response.

User Query: "{query}"

Candidates Found ({len(candidates)} total):
{context}

Please provide a comprehensive analysis that includes:

1. Query Understanding: Briefly acknowledge what the user is looking for
2. Candidate Overview: Summarize the types of candidates found and their key strengths
3. Skill Analysis: Highlight the most relevant skills and experience areas
4. Top Matches: Mention 2-3 candidates who seem like the best fits and why
5. Recommendations: Suggest specific next steps for the user

Make the response:
- Professional yet conversational
- Specific to the query and candidates
- Actionable with clear next steps
- Insightful about the candidate pool quality

Focus on providing real value and insights, not just generic statements.
"""
            
            # Generate response using Gemini with timeout
            try:
                response = self.gemini_model.generate_content(prompt)
                
                if response and response.text:
                    return response.text.strip()
                else:
                    return self._generate_fallback_response(query, candidates)
                    
            except Exception as gemini_error:
                logger.warning(f"Gemini model error: {gemini_error}")
                return self._generate_fallback_response(query, candidates)
                
        except Exception as e:
            logger.error(f"Error generating RAG response: {e}")
            return self._generate_fallback_response(query, candidates)
    
    def _generate_fallback_response(self, query: str, candidates: List[Dict[str, Any]]) -> str:
        """Generate a fallback response without LLM."""
        try:
            if not candidates:
                return f"No candidates found matching '{query}'."
            
            # Find top candidates
            top_candidates = sorted(candidates, key=lambda x: x.get('similarity_score', 0), reverse=True)[:3]
            
            response_parts = [
                f"Found {len(candidates)} candidates matching '{query}'.",
                "",
                "Top matches:"
            ]
            
            for i, candidate in enumerate(top_candidates, 1):
                name = candidate.get('name', 'Unknown')
                skills = ', '.join(candidate.get('skills', [])[:5])  # Top 5 skills
                score = int(candidate.get('similarity_score', 0) * 100)
                
                response_parts.append(f"{i}. {name} ({score}% match)")
                response_parts.append(f"   Skills: {skills}")
                response_parts.append("")
            
            response_parts.append("Please review their full profiles for more details.")
            
            return "\n".join(response_parts)
            
        except Exception as e:
            logger.error(f"Error in fallback response: {e}")
            return f"Found {len(candidates)} candidates matching '{query}'. Please review their profiles for more details."
    
    def hybrid_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Perform hybrid search combining semantic and keyword search."""
        try:
            # Semantic search
            semantic_results = self.semantic_search(query, top_k)
            
            # Keyword search (existing SQL-based)
            keyword_results = self.keyword_search(query, top_k)
            
            # Combine and rank results
            combined_results = self.combine_search_results(semantic_results, keyword_results, top_k)
            
            return combined_results
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return self.keyword_search(query, top_k)  # Fallback to keyword search
    
    def keyword_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Perform traditional keyword search using SQL."""
        try:
            # Check if query contains multiple skills (common skill keywords)
            skill_keywords = ['python', 'java', 'javascript', 'react', 'node.js', 'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'git', 'github', 'machine learning', 'ai', 'data science', 'nlp', 'devops', 'ci/cd', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'kafka', 'jenkins', 'terraform', 'ansible', 'powershell', 'bash', 'linux', 'windows', 'vmware', 'hyper-v', 'spring', 'flask', 'django', 'angular', 'vue', 'swift', 'go', 'rust', 'c++', 'c#', 'ruby', 'php', 'scala', 'kotlin', 'typescript', 'html', 'css', 'sass', 'less', 'webpack', 'babel', 'jest', 'junit', 'pytest', 'selenium', 'cypress', 'jira', 'confluence', 'slack', 'teams', 'zoom', 'figma', 'sketch', 'adobe', 'photoshop', 'illustrator', 'blender', 'unity', 'unreal', 'maya', '3ds max', 'autocad', 'solidworks', 'matlab', 'sas', 'spss', 'tableau', 'powerbi', 'qlik', 'looker', 'snowflake', 'databricks', 'hadoop', 'spark', 'hive', 'pig', 'hbase', 'cassandra', 'neo4j', 'graphql', 'rest', 'soap', 'grpc', 'microservices', 'api', 'websocket', 'oauth', 'jwt', 'ldap', 'saml', 'oauth2', 'openid', 'ssl', 'tls', 'vpn', 'firewall', 'ids', 'ips', 'siem', 'soc', 'penetration testing', 'vulnerability assessment', 'risk assessment', 'compliance', 'gdpr', 'hipaa', 'sox', 'pci', 'iso', 'nist', 'cobit', 'itil', 'agile', 'scrum', 'kanban', 'lean', 'six sigma', 'waterfall', 'prince2', 'pmp', 'capm', 'pmi', 'csm', 'cspo', 'safe', 'less', 'nexus', 'dsdm', 'fdd', 'crystal', 'tdd', 'bdd', 'atdd', 'ddd', 'mvc', 'mvvm', 'mvp', 'clean architecture', 'solid', 'dry', 'kiss', 'yagni']
            
            # Extract potential skills from query with better word boundary detection
            query_lower = query.lower()
            found_skills = []
            
            for skill in skill_keywords:
                # Use word boundary detection to avoid partial matches
                import re
                pattern = r'\b' + re.escape(skill) + r'\b'
                if re.search(pattern, query_lower):
                    found_skills.append(skill)
            
            if len(found_skills) >= 2:
                # Multi-skill search with AND logic
                return self.multi_skill_search(found_skills, top_k)
            else:
                # Single skill or general search
                search_query = """
                SELECT candidate_id, name, email, phone, summary, gcs_url, created_at, skills
                FROM candidates
                WHERE LOWER(name) LIKE LOWER(@query) 
                   OR LOWER(summary) LIKE LOWER(@query)
                   OR EXISTS(SELECT 1 FROM UNNEST(skills) AS skill WHERE LOWER(skill) LIKE LOWER(@query))
                ORDER BY created_at DESC
                LIMIT @limit
                """
                
                with database.snapshot() as snapshot:
                    results = snapshot.execute_sql(
                        search_query,
                        params={'query': f'%{query}%', 'limit': top_k},
                        param_types={'query': spanner.param_types.STRING, 'limit': spanner.param_types.INT64}
                    )
                    
                    candidates = []
                    for row in results:
                        candidates.append({
                            'candidate_id': str(row[0]),
                            'name': str(row[1]),
                            'email': str(row[2]) if row[2] else None,
                            'phone': str(row[3]) if row[3] else None,
                            'summary': str(row[4]) if row[4] else "",
                            'gcs_url': str(row[5]) if row[5] else None,
                            'created_at': row[6],
                            'skills': list(row[7]) if row[7] else [],
                            'similarity_score': 0.8  # Default score for keyword search
                        })
                    
                    return candidates
                    
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return []
    
    def multi_skill_search(self, skills: List[str], top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for candidates who have ALL the specified skills (AND logic)."""
        try:
            # Build dynamic SQL query for multiple skills
            skill_conditions = []
            for i, skill in enumerate(skills):
                skill_conditions.append(f"EXISTS(SELECT 1 FROM UNNEST(skills) AS skill{i} WHERE LOWER(skill{i}) LIKE LOWER(@skill{i}))")
            
            search_query = f"""
            SELECT candidate_id, name, email, phone, summary, gcs_url, created_at, skills
            FROM candidates
            WHERE {' AND '.join(skill_conditions)}
            ORDER BY created_at DESC
            LIMIT @limit
            """
            
            # Build parameters
            params = {'limit': top_k}
            param_types = {'limit': spanner.param_types.INT64}
            
            for i, skill in enumerate(skills):
                params[f'skill{i}'] = f'%{skill}%'
                param_types[f'skill{i}'] = spanner.param_types.STRING
            
            with database.snapshot() as snapshot:
                results = snapshot.execute_sql(
                    search_query,
                    params=params,
                    param_types=param_types
                )
                
                candidates = []
                for row in results:
                    candidate_skills = list(row[7]) if row[7] else []
                    # Calculate score based on how many required skills they have
                    matching_skills = sum(1 for skill in skills if any(skill in s.lower() for s in candidate_skills))
                    score = 0.8 + (matching_skills / len(skills)) * 0.2  # Base 0.8 + bonus for matching skills
                    
                    candidates.append({
                        'candidate_id': str(row[0]),
                        'name': str(row[1]),
                        'email': str(row[2]) if row[2] else None,
                        'phone': str(row[3]) if row[3] else None,
                        'summary': str(row[4]) if row[4] else "",
                        'gcs_url': str(row[5]) if row[5] else None,
                        'created_at': row[6],
                        'skills': candidate_skills,
                        'similarity_score': score
                    })
                
                logger.info(f"Multi-skill search found {len(candidates)} candidates with skills: {skills}")
                return candidates
                
        except Exception as e:
            logger.error(f"Error in multi-skill search: {e}")
            return []
    
    def semantic_search_only(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Perform semantic search only (no LLM generation)."""
        try:
            # Perform semantic search
            semantic_results = self.semantic_search(query, top_k)
            
            if not semantic_results:
                logger.warning("No semantic results found, falling back to keyword search")
                return self.keyword_search(query, top_k)
            
            # Get full candidate data for semantic results
            candidates = []
            for candidate_id, similarity_score in semantic_results:
                if candidate_id in self.candidate_embeddings:
                    candidate = self.candidate_embeddings[candidate_id].copy()
                    candidate['similarity_score'] = similarity_score
                    candidates.append(candidate)
            
            # Sort by similarity score
            candidates.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
            
            logger.info(f"Semantic search found {len(candidates)} candidates")
            return candidates[:top_k]
            
        except Exception as e:
            logger.error(f"Error in semantic search only: {e}")
            # Fallback to keyword search
            return self.keyword_search(query, top_k)
    
    def combine_search_results(self, semantic_results: List[Tuple[str, float]], 
                             keyword_results: List[Dict[str, Any]], 
                             top_k: int) -> List[Dict[str, Any]]:
        """Combine and rank search results from both methods."""
        try:
            # Create a combined scoring system
            combined_candidates = {}
            
            # Add semantic search results
            for candidate_id, similarity in semantic_results:
                if candidate_id in self.candidate_embeddings:
                    candidate = self.candidate_embeddings[candidate_id].copy()
                    candidate['similarity_score'] = similarity
                    combined_candidates[candidate_id] = candidate
            
            # Add keyword search results with adjusted scoring
            for candidate in keyword_results:
                candidate_id = candidate['candidate_id']
                if candidate_id in combined_candidates:
                    # Boost score if found in both searches
                    combined_candidates[candidate_id]['similarity_score'] = max(
                        combined_candidates[candidate_id]['similarity_score'],
                        candidate['similarity_score'] + 0.1
                    )
                else:
                    combined_candidates[candidate_id] = candidate
            
            # If no semantic results, use keyword results as fallback
            if not semantic_results and keyword_results:
                logger.info("No semantic results found, using keyword search results")
                return keyword_results[:top_k]
            
            # Sort by similarity score and return top_k
            sorted_candidates = sorted(
                combined_candidates.values(),
                key=lambda x: x.get('similarity_score', 0),
                reverse=True
            )
            
            return sorted_candidates[:top_k]
            
        except Exception as e:
            logger.error(f"Error combining search results: {e}")
            return keyword_results[:top_k]  # Fallback
    
    def extract_relevant_text(self, text: str, query: str, max_length: int = 300) -> str:
        """Extract relevant text snippets using Gemini for better context understanding."""
        try:
            if not text or not query:
                return "No relevant text found."
            
            # Use Gemini to extract relevant context
            prompt = f"""
Extract the most relevant text snippet from the following candidate summary that relates to the search query.

Search Query: "{query}"

Candidate Summary:
{text}

Please extract 1-2 sentences that are most relevant to the search query. Keep it concise (max 300 characters).
If no relevant text is found, return "No relevant text found."
"""
            
            response = self.gemini_model.generate_content(prompt)
            
            if response and response.text:
                relevant_text = response.text.strip()
                if len(relevant_text) > max_length:
                    relevant_text = relevant_text[:max_length] + "..."
                return relevant_text
            else:
                # Fallback to original method
                return self._extract_relevant_text_fallback(text, query, max_length)
                
        except Exception as e:
            logger.error(f"Error extracting relevant text with Gemini: {e}")
            return self._extract_relevant_text_fallback(text, query, max_length)
    
    def _extract_relevant_text_fallback(self, text: str, query: str, max_length: int = 300) -> str:
        """Fallback method for extracting relevant text."""
        if not text or not query:
            return "No relevant text found."
        
        text_lower = text.lower()
        query_lower = query.lower()
        query_words = query_lower.split()
        
        sentences = []
        for sentence in text.split('.'):
            sentence = sentence.strip()
            if sentence:
                sentence_lower = sentence.lower()
                if any(word in sentence_lower for word in query_words):
                    sentences.append(sentence)
        
        if sentences:
            relevant_text = '. '.join(sentences[:3])
            if len(relevant_text) > max_length:
                relevant_text = relevant_text[:max_length] + "..."
            return relevant_text
        
        return "No relevant text found."
    
    def calculate_levenshtein_distance(self, str1: str, str2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        try:
            # Simple implementation of Levenshtein distance
            if len(str1) < len(str2):
                return self.calculate_levenshtein_distance(str2, str1)
            
            if len(str2) == 0:
                return len(str1)
            
            previous_row = list(range(len(str2) + 1))
            for i, c1 in enumerate(str1):
                current_row = [i + 1]
                for j, c2 in enumerate(str2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        except Exception as e:
            logger.error(f"Error calculating Levenshtein distance: {e}")
            return len(str1) + len(str2)  # Maximum distance as fallback
    
    def calculate_similarity_score(self, str1: str, str2: str) -> float:
        """Calculate similarity score between two strings using multiple methods."""
        try:
            if not str1 or not str2:
                return 0.0
            
            # Normalize strings
            str1_norm = str1.lower().strip()
            str2_norm = str2.lower().strip()
            
            # Method 1: Levenshtein distance
            levenshtein_dist = self.calculate_levenshtein_distance(str1_norm, str2_norm)
            max_len = max(len(str1_norm), len(str2_norm))
            levenshtein_similarity = 1.0 - (levenshtein_dist / max_len) if max_len > 0 else 0.0
            
            # Method 2: Sequence matcher (difflib)
            sequence_similarity = difflib.SequenceMatcher(None, str1_norm, str2_norm).ratio()
            
            # Method 3: Jaccard similarity for word sets
            words1 = set(str1_norm.split())
            words2 = set(str2_norm.split())
            if words1 or words2:
                intersection = len(words1.intersection(words2))
                union = len(words1.union(words2))
                jaccard_similarity = intersection / union if union > 0 else 0.0
            else:
                jaccard_similarity = 0.0
            
            # Combine scores (weighted average)
            combined_score = (levenshtein_similarity * 0.4 + 
                           sequence_similarity * 0.4 + 
                           jaccard_similarity * 0.2)
            
            return max(0.0, min(1.0, combined_score))
            
        except Exception as e:
            logger.error(f"Error calculating similarity score: {e}")
            return 0.0
    
    def semantic_search_with_similarity(self, query: str, top_k: int = 10, 
                                      similarity_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Enhanced semantic search with LLM-powered query understanding and matching."""
        try:
            # Use LLM to understand and enhance the query if available
            enhanced_query = query
            if self.gemini_model:
                try:
                    enhanced_query = self._enhance_query_with_llm(query)
                    logger.info(f"LLM enhanced query: '{query}' → '{enhanced_query}'")
                except Exception as e:
                    logger.warning(f"LLM query enhancement failed: {e}")

            # First, try keyword search to find exact skill matches
            keyword_results = self.keyword_search(enhanced_query, top_k * 2)

            # Then perform vector-based semantic search
            semantic_results = self.semantic_search(enhanced_query, top_k * 2)  # Get more results for filtering

            # Combine both approaches
            all_candidates = {}

            # Add keyword search results (these have exact skill matches)
            for candidate in keyword_results:
                candidate_id = candidate['candidate_id']
                all_candidates[candidate_id] = candidate
                # Boost score for keyword matches (exact skill matches)
                all_candidates[candidate_id]['similarity_score'] = max(
                    all_candidates[candidate_id]['similarity_score'],
                    0.8  # High score for exact matches
                )

            # Add semantic search results
            for candidate_id, vector_similarity in semantic_results:
                if candidate_id in self.candidate_embeddings:
                    candidate = self.candidate_embeddings[candidate_id].copy()

                    # Calculate text similarity for different fields
                    name_similarity = self.calculate_similarity_score(enhanced_query, candidate.get('name', ''))
                    summary_similarity = self.calculate_similarity_score(enhanced_query, candidate.get('summary', ''))
                    skills_similarity = self.calculate_similarity_score(enhanced_query, ' '.join(candidate.get('skills', [])))

                    # Combine similarities
                    text_similarity = max(name_similarity, summary_similarity, skills_similarity)

                    # Enhanced scoring: combine vector similarity with text similarity
                    enhanced_score = (vector_similarity * 0.7) + (text_similarity * 0.3)

                    if candidate_id in all_candidates:
                        # If already found by keyword search, boost the score
                        all_candidates[candidate_id]['similarity_score'] = max(
                            all_candidates[candidate_id]['similarity_score'],
                            enhanced_score + 0.1
                        )
                    else:
                        # Apply similarity threshold for semantic-only results
                        if enhanced_score >= similarity_threshold:
                            candidate['similarity_score'] = enhanced_score
                            candidate['vector_similarity'] = vector_similarity
                            candidate['text_similarity'] = text_similarity
                            all_candidates[candidate_id] = candidate

            # Use LLM to re-rank candidates if available
            if self.gemini_model and all_candidates:
                try:
                    all_candidates = self._rerank_candidates_with_llm(query, all_candidates)
                    logger.info("LLM re-ranking applied")
                except Exception as e:
                    logger.warning(f"LLM re-ranking failed: {e}")

            # Convert to list and sort by similarity score
            enhanced_candidates = list(all_candidates.values())
            enhanced_candidates.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)

            logger.info(f"Enhanced search found {len(enhanced_candidates)} candidates (keyword: {len(keyword_results)}, semantic: {len(semantic_results)})")
            return enhanced_candidates[:top_k]

        except Exception as e:
            logger.error(f"Error in enhanced semantic search: {e}")
            # Fallback to keyword search
            return self.keyword_search(query, top_k)

    def _enhance_query_with_llm(self, query: str) -> str:
        """Use LLM to enhance and expand the search query."""
        try:
            prompt = f"""
You are an expert HR recruiter. Enhance this search query to find better candidates by:
1. Adding related technical skills
2. Including relevant experience levels
3. Adding industry-specific terms
4. Expanding abbreviations

Original query: "{query}"

Return only the enhanced query, nothing else. Make it concise but comprehensive.
"""
            response = self.gemini_model.generate_content(prompt)
            if response and response.text:
                enhanced = response.text.strip()
                # Remove quotes if present
                enhanced = enhanced.strip('"').strip("'")
                return enhanced if enhanced else query
            return query
        except Exception as e:
            logger.warning(f"Query enhancement failed: {e}")
            return query

    def _rerank_candidates_with_llm(self, query: str, candidates: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Use LLM to re-rank candidates based on query relevance."""
        try:
            # Prepare candidate summaries for LLM
            candidate_summaries = []
            for candidate_id, candidate in candidates.items():
                skills_str = ', '.join(candidate.get('skills', [])[:10])  # Top 10 skills
                summary = candidate.get('summary', '')[:200]  # First 200 chars
                candidate_summaries.append(f"ID: {candidate_id} | {candidate['name']} | Skills: {skills_str} | Summary: {summary}")

            prompt = f"""
You are an expert HR recruiter. Given this search query: "{query}"

Rank these candidates by relevance (1=most relevant, 5=least relevant). Consider:
- Skill match
- Experience level
- Role alignment
- Industry fit

Candidates:
{chr(10).join(candidate_summaries[:10])}  # Limit to top 10 for LLM

Return only a JSON array of candidate IDs in order of relevance (most relevant first).
Example: ["candidate-123", "candidate-456", ...]
"""
            response = self.gemini_model.generate_content(prompt)
            if response and response.text:
                try:
                    import json
                    ranked_ids = json.loads(response.text.strip())
                    # Re-rank candidates based on LLM ranking
                    reranked_candidates = {}
                    for i, candidate_id in enumerate(ranked_ids):
                        if candidate_id in candidates:
                            candidate = candidates[candidate_id].copy()
                            # Boost score based on LLM ranking
                            candidate['similarity_score'] = candidate.get('similarity_score', 0) + (0.1 * (10 - i))
                            reranked_candidates[candidate_id] = candidate
                    
                    # Add any remaining candidates
                    for candidate_id, candidate in candidates.items():
                        if candidate_id not in reranked_candidates:
                            reranked_candidates[candidate_id] = candidate
                    
                    return reranked_candidates
                except Exception as e:
                    logger.warning(f"Failed to parse LLM ranking: {e}")
                    return candidates
            return candidates
        except Exception as e:
            logger.warning(f"LLM re-ranking failed: {e}")
            return candidates
    
    def fuzzy_search(self, query: str, top_k: int = 10, 
                    fuzzy_threshold: float = 0.6) -> List[Dict[str, Any]]:
        """Fuzzy search using Levenshtein distance and text similarity."""
        try:
            # Get all candidates from database
            with database.snapshot() as snapshot:
                results = snapshot.execute_sql(
                    "SELECT candidate_id, name, email, phone, summary, skills, gcs_url FROM candidates",
                    param_types={}
                )
                
                fuzzy_candidates = []
                query_lower = query.lower()
                
                for row in results:
                    candidate = {
                        'candidate_id': str(row[0]),
                        'name': str(row[1]),
                        'email': str(row[2]) if row[2] else None,
                        'phone': str(row[3]) if row[3] else None,
                        'summary': str(row[4]) if row[4] else "",
                        'skills': list(row[5]) if row[5] else [],
                        'gcs_url': str(row[6]) if row[6] else None
                    }
                    
                    # Calculate similarity scores for different fields
                    name_similarity = self.calculate_similarity_score(query, candidate['name'])
                    summary_similarity = self.calculate_similarity_score(query, candidate['summary'])
                    skills_text = ' '.join(candidate['skills'])
                    skills_similarity = self.calculate_similarity_score(query, skills_text)
                    
                    # Get the best similarity score
                    best_similarity = max(name_similarity, summary_similarity, skills_similarity)
                    
                    if best_similarity >= fuzzy_threshold:
                        candidate['similarity_score'] = best_similarity
                        candidate['name_similarity'] = name_similarity
                        candidate['summary_similarity'] = summary_similarity
                        candidate['skills_similarity'] = skills_similarity
                        fuzzy_candidates.append(candidate)
                
                # Sort by similarity score
                fuzzy_candidates.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
                
                logger.info(f"Fuzzy search found {len(fuzzy_candidates)} candidates")
                return fuzzy_candidates[:top_k]
                
        except Exception as e:
            logger.error(f"Error in fuzzy search: {e}")
            return []

# Initialize RAG enhancer (can be overridden)
rag_enhancer = None

def initialize_rag_system(use_llm: bool = None):
    """Initialize the RAG system by building the vector index."""
    global rag_enhancer
    try:
        logger.info("Initializing RAG system...")
        # Use environment variable if not explicitly set
        if use_llm is None:
            use_llm = USE_LLM
        rag_enhancer = RAGEnhancer(use_llm=use_llm)
        rag_enhancer.build_vector_index()
        logger.info("RAG system initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing RAG system: {e}")

if __name__ == "__main__":
    initialize_rag_system() 