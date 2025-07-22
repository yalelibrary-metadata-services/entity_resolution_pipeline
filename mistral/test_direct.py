#!/usr/bin/env python3
"""
Direct test of Mistral predictor without subprocess
"""

import os
import sys
import json

# Add parent directory to path to import the predictor
sys.path.insert(0, os.path.dirname(__file__))

from predict_mistral_classifier import MistralPredictor

def test_classification():
    """Test the classifier with sample texts."""
    
    # Check API key
    if not os.environ.get("MISTRAL_API_KEY"):
        print("ERROR: MISTRAL_API_KEY environment variable not set")
        return
    
    # Initialize predictor
    model_id = "ft:classifier:ministral-3b-latest:2bec22ef:20250702:a2707cf5"
    
    try:
        predictor = MistralPredictor(model_id)
    except Exception as e:
        print(f"Failed to initialize predictor: {e}")
        return
    
    # Test texts
    test_texts = [
        "Title: Quartette für zwei Violinen, Viola, Violoncell\\nSubjects: String quartets--Scores",
        "Title: Strategic management : concepts and cases\\nSubjects: Strategic planning; Management; Business planning",
        "Title: Organic chemistry : structure and function\\nSubjects: Chemistry, Organic; Organic compounds--Structure"
    ]
    
    print("Testing classification...\n")
    
    for i, text in enumerate(test_texts):
        print(f"Test {i+1}:")
        print(f"Text: {text[:80]}...")
        
        # Get classification
        domain_scores, parent_scores = predictor.classify_text(text)
        
        if domain_scores:
            # Get top domain
            top_domain = max(domain_scores, key=domain_scores.get)
            top_score = domain_scores[top_domain]
            
            print(f"Top Domain: {top_domain} (score: {top_score:.4f})")
            
            # Show top 3 domains
            sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            print("Top 3 domains:")
            for domain, score in sorted_domains:
                print(f"  - {domain}: {score:.4f}")
        else:
            print("No classification results")
        
        print("-" * 50)

if __name__ == "__main__":
    test_classification()