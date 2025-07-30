#!/usr/bin/env python3
"""Script to check the current state of the candidates table."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import db
from config import settings

def main():
    """Check the database state."""
    print("Checking candidates table...")
    
    try:
        # Check if we can connect to the database
        print(f"Connecting to Spanner instance: {settings.SPANNER_INSTANCE}")
        print(f"Database: {settings.SPANNER_DATABASE}")
        
        # Get candidate count
        count = db.get_candidates_count()
        print(f"Total candidates in database: {count}")
        
        if count > 0:
            # Get all candidates
            candidates = db.get_all_candidates()
            print(f"\nFound {len(candidates)} candidates:")
            for i, candidate in enumerate(candidates, 1):
                print(f"{i}. ID: {candidate.candidate_id}")
                print(f"   Name: {candidate.name}")
                print(f"   Summary: {candidate.summary[:100] if candidate.summary else 'No summary'}...")
                print(f"   Created: {candidate.created_at}")
                print()
        else:
            print("No candidates found in the database.")
            
        # Check skills table
        with db.database.snapshot() as snapshot:
            skills_result = snapshot.execute_sql("SELECT COUNT(*) FROM Skills")
            for row in skills_result:
                skills_count = row[0]
                print(f"Total skills in database: {skills_count}")
                
            candidate_skills_result = snapshot.execute_sql("SELECT COUNT(*) FROM CandidateSkills")
            for row in candidate_skills_result:
                candidate_skills_count = row[0]
                print(f"Total candidate-skill relationships: {candidate_skills_count}")
                
    except Exception as e:
        print(f"Error checking database: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 