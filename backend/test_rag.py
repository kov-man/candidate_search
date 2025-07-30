#!/usr/bin/env python3
"""
Test suite for RAG-enhanced candidate search system.
Tests vector search, semantic matching, and LLM integration.
"""

import os
import sys
import logging
from pathlib import Path

# Add the backend directory to Python path for imports
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import RAG components
try:
    from rag_enhanced import RAGEnhancer, initialize_rag_system
    from gcp_main_rag import database
except ImportError as e:
    logger.error(f"Import error: {e}")
    print("Make sure all dependencies are installed and the backend is properly set up.")
    sys.exit(1)

def test_rag_initialization():
    """Test RAG system initialization."""
    print("Testing RAG System Initialization...")
    
    try:
        # Initialize RAG system
        initialize_rag_system()
        print("RAG system initialized successfully")
        return True
        
    except Exception as e:
        print(f"Error initializing RAG system: {e}")
        return False

def test_semantic_search():
    """Test semantic search functionality."""
    print("\nTesting Semantic Search...")
    
    try:
        from rag_enhanced import rag_enhancer
        
        if not rag_enhancer:
            print("RAG enhancer not initialized")
            return False
        
        # Test queries
        test_queries = [
            "Python developer",
            "Machine learning engineer", 
            "React frontend developer",
            "Data scientist with PhD",
            "Cloud architect with AWS"
        ]
        
        success_count = 0
        for query in test_queries:
            try:
                results = rag_enhancer.semantic_search_with_similarity(query, top_k=3)
                print(f"Query: '{query}' -> {len(results)} results")
                
                for i, result in enumerate(results[:2]):  # Show top 2
                    similarity = result.get('similarity_score', 0.0)
                    name = result.get('name', 'Unknown')
                    print(f"  {i+1}. {name} (similarity: {similarity:.3f})")
                
                success_count += 1
                
            except Exception as e:
                print(f"Error testing query '{query}': {e}")
        
        print(f"Semantic search test completed: {success_count}/{len(test_queries)} queries successful")
        return success_count == len(test_queries)
        
    except Exception as e:
        print(f"Error testing semantic search: {e}")
        return False

def test_database_connectivity():
    """Test database connectivity and candidate retrieval."""
    print("\nTesting Database Connectivity...")
    
    try:
        # Test basic database query
        query = "SELECT COUNT(*) as count FROM candidates"
        
        with database.snapshot() as snapshot:
            results = list(snapshot.execute_sql(query))
            if results:
                count = results[0][0]
                print(f"Database connected. Total candidates: {count}")
                
                if count == 0:
                    print("No candidates found for testing")
                    return False
                    
            else:
                print("No results from database query")
                return False
        
        return True
        
    except Exception as e:
        print(f"Database connectivity test failed: {e}")
        return False

def test_rag_response_generation():
    """Test RAG response generation with LLM."""
    print("\nTesting RAG Response Generation...")
    
    try:
        from rag_enhanced import rag_enhancer
        
        if not rag_enhancer:
            print("RAG enhancer not initialized")
            return False
        
        # Test with sample candidates
        test_query = "Find Python developers"
        candidates = rag_enhancer.semantic_search_with_similarity(test_query, top_k=2)
        
        if candidates:
            response = rag_enhancer.generate_rag_response(test_query, candidates)
            print(f"RAG response for '{test_query}':")
            print(f"Response: {response[:200]}...")
            return True
        else:
            print("No candidates available for RAG response testing")
            return False
            
    except Exception as e:
        print(f"Error testing RAG response generation: {e}")
        return False

def test_relevant_text_extraction():
    """Test relevant text extraction functionality."""
    print("\nTesting Relevant Text Extraction...")
    
    try:
        from rag_enhanced import rag_enhancer
        
        if not rag_enhancer:
            print("RAG enhancer not initialized")
            return False
        
        # Get candidates with summaries
        candidates = []
        for candidate_data in rag_enhancer.candidate_embeddings.values():
            if candidate_data.get('summary'):
                candidates.append(candidate_data)
                
        if not candidates:
            print("No candidates with summaries found for testing")
            return False
        
        # Test text extraction with different queries
        test_queries = ["Python", "machine learning", "web development"]
        
        for query in test_queries:
            for i, candidate in enumerate(candidates[:2]):  # Test first 2 candidates
                summary = candidate.get('summary', '')
                if summary:
                    relevant_text = rag_enhancer._extract_relevant_text_fallback(summary, query, 100)
                    print(f"Query: '{query}', Candidate: {candidate.get('name', 'Unknown')}")
                    print(f"Relevant text: {relevant_text}")
                    print()
        
        return True
        
    except Exception as e:
        print(f"Error testing relevant text extraction: {e}")
        return False

def test_embedding_generation():
    """Test embedding generation functionality."""
    print("\nTesting Embedding Generation...")
    
    try:
        from rag_enhanced import rag_enhancer
        
        if not rag_enhancer:
            print("RAG enhancer not initialized")
            return False
        
        # Test embedding generation
        test_texts = [
            "Python developer with machine learning experience",
            "Frontend developer with React and JavaScript skills",
            "Data scientist with PhD in computer science"
        ]
        
        for text in test_texts:
            embedding = rag_enhancer.generate_embedding(text)
            print(f"Text: '{text[:50]}...'")
            print(f"Embedding dimension: {len(embedding)}")
            print(f"First 5 values: {embedding[:5]}")
            print()
        
        return True
        
    except Exception as e:
        print(f"Error testing embedding generation: {e}")
        return False

def test_vector_index():
    """Test vector index functionality."""
    print("\nTesting Vector Index...")
    
    try:
        from rag_enhanced import rag_enhancer
        
        if not rag_enhancer:
            print("RAG enhancer not initialized")
            return False
        
        # Rebuild vector index
        rag_enhancer.build_vector_index()
        
        print(f"Vector index built with {len(rag_enhancer.candidate_vectors)} candidates")
        
        # Test search
        test_query = "software engineer"
        results = rag_enhancer.semantic_search(test_query, top_k=3)
        
        print(f"Search results for '{test_query}':")
        for i, (candidate_id, similarity) in enumerate(results):
            candidate = rag_enhancer.candidate_embeddings.get(candidate_id)
            name = candidate.get('name', 'Unknown') if candidate else 'Unknown'
            print(f"  {i+1}. {name} (ID: {candidate_id}, Similarity: {similarity:.3f})")
        
        return True
        
    except Exception as e:
        print(f"Error testing vector index: {e}")
        return False

def run_comprehensive_test():
    """Run all tests and provide summary."""
    print("Starting Comprehensive RAG System Test")
    print("=" * 50)
    
    tests = [
        ("RAG Initialization", test_rag_initialization),
        ("Database Connectivity", test_database_connectivity),
        ("Vector Index", test_vector_index),
        ("Semantic Search", test_semantic_search),
        ("Embedding Generation", test_embedding_generation),
        ("Relevant Text Extraction", test_relevant_text_extraction),
        ("RAG Response Generation", test_rag_response_generation),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*20} {test_name} {'='*20}")
            success = test_func()
            status = "PASSED" if success else "FAILED"
            print(f"{test_name}: {status}")
            if success:
                passed += 1
        except Exception as e:
            print(f"ERROR: {test_name} - {e}")
            print(f"{test_name}: FAILED")
    
    print(f"\n{'='*50}")
    print(f"Test Summary:")
    print(f"Passed: {passed}/{total}")
    print(f"Failed: {total-passed}/{total}")
    
    if passed == total:
        print("All tests passed!")
        return True
    else:
        print("Some tests failed. Check the logs above for details.")
        return False

if __name__ == "__main__":
    # Check environment
    if not os.getenv("GOOGLE_API_KEY"):
        print("Warning: GOOGLE_API_KEY not set. Some features may not work.")
    
    # Run tests
    success = run_comprehensive_test()
    
    if success:
        print("\nRAG system is working correctly!")
        sys.exit(0)
    else:
        print("\nRAG system has issues. Please check the configuration.")
        sys.exit(1) 