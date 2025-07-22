#!/usr/bin/env python3
"""
Mistral Entity Classification Prediction CLI

Make predictions using a fine-tuned Mistral hierarchical classifier.
This is a drop-in replacement for the SetFit classifier with the same CLI interface.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from mistralai import Mistral


class MistralPredictor:
    """
    Load and use a trained Mistral classifier for predictions.
    """
    
    def __init__(self, model_id: str, api_key: Optional[str] = None):
        self.model_id = model_id
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        
        if not self.api_key:
            raise ValueError("Mistral API key required. Set MISTRAL_API_KEY environment variable.")
        
        self.client = Mistral(api_key=self.api_key)
        print(f"Loaded Mistral model: {model_id}")
    
    def classify_text(self, text: str) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Classify a single text and return scores for all classes.
        
        Returns:
            Tuple of (domain_scores, parent_category_scores)
        """
        try:
            response = self.client.classifiers.classify(
                model=self.model_id,
                inputs=[text]
            )
            
            # Parse the response structure based on the example format
            response_dict = response.model_dump() if hasattr(response, 'model_dump') else response
            
            # Extract scores from the nested structure
            if 'results' in response_dict and len(response_dict['results']) > 0:
                result = response_dict['results'][0]
                domain_scores = result.get('domain', {}).get('scores', {})
                parent_scores = result.get('parent_category', {}).get('scores', {})
            else:
                domain_scores = {}
                parent_scores = {}
            
            return domain_scores, parent_scores
            
        except Exception as e:
            print(f"Error classifying text: {e}")
            return {}, {}
    
    def predict(self, texts: List[str], confidence_threshold: float = 0.0) -> List[str]:
        """
        Make predictions on texts, returning the highest scoring domain.
        
        Args:
            texts: List of text strings to classify
            confidence_threshold: Minimum confidence score (not used for compatibility)
        
        Returns:
            List of predicted domain labels
        """
        if isinstance(texts, str):
            texts = [texts]
        
        predictions = []
        
        for text in texts:
            domain_scores, _ = self.classify_text(text)
            
            if domain_scores:
                # Get highest scoring domain
                best_domain = max(domain_scores, key=domain_scores.get)
                predictions.append(best_domain)
            else:
                predictions.append("")
        
        return predictions
    
    def predict_with_metadata(self, texts: List[str], confidence_threshold: float = 0.5) -> List[Dict]:
        """
        Make predictions with additional metadata about the prediction process.
        
        Returns:
            List of dictionaries with prediction details
        """
        if isinstance(texts, str):
            texts = [texts]
        
        results = []
        
        for text in texts:
            domain_scores, parent_scores = self.classify_text(text)
            
            if domain_scores and parent_scores:
                # Get highest scoring domain and parent
                best_domain = max(domain_scores, key=domain_scores.get)
                best_domain_score = domain_scores[best_domain]
                
                best_parent = max(parent_scores, key=parent_scores.get)
                best_parent_score = parent_scores[best_parent]
                
                # Check if this is a parent category
                # In the taxonomy, parent categories are the high-level categories
                parent_categories = ["Arts, Culture, and Creative Expression", 
                                   "Sciences, Research, and Discovery",
                                   "Humanities, Thought, and Interpretation",
                                   "Society, Governance, and Public Life"]
                
                is_parent = best_domain in parent_categories
                
                result = {
                    'text': text[:100] + '...' if len(text) > 100 else text,
                    'prediction': best_domain,
                    'confidence': best_domain_score,
                    'parent_category': best_parent,
                    'parent_confidence': best_parent_score,
                    'is_parent_category': is_parent,
                    'all_domain_scores': domain_scores,
                    'all_parent_scores': parent_scores
                }
            else:
                result = {
                    'text': text[:100] + '...' if len(text) > 100 else text,
                    'prediction': '',
                    'confidence': 0.0,
                    'parent_category': '',
                    'parent_confidence': 0.0,
                    'is_parent_category': False,
                    'all_domain_scores': {},
                    'all_parent_scores': {}
                }
            
            results.append(result)
            
            # Rate limiting
            time.sleep(0.1)
        
        return results
    
    def batch_predict_csv(self, input_csv: str, text_column: str = 'composite', 
                         output_csv: Optional[str] = None, batch_size: int = 50) -> pd.DataFrame:
        """
        Make predictions on a CSV file.
        
        Args:
            input_csv: Path to input CSV file
            text_column: Name of column containing text to classify
            output_csv: Path to output CSV file (optional)
            batch_size: Number of texts to process at once
        
        Returns:
            DataFrame with predictions added
        """
        df = pd.read_csv(input_csv)
        
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in CSV. Available columns: {list(df.columns)}")
        
        print(f"Making predictions on {len(df)} rows...")
        
        # Initialize new columns
        df['mistral_prediction'] = ''
        df['is_parent_category'] = False
        df['mistral_confidence'] = 0.0
        df['mistral_parent_category'] = ''
        
        # Process in batches
        for i in range(0, len(df), batch_size):
            batch_end = min(i + batch_size, len(df))
            batch_texts = df[text_column].iloc[i:batch_end].tolist()
            
            print(f"Processing batch {i//batch_size + 1}/{(len(df)-1)//batch_size + 1}...")
            
            # Get predictions with metadata
            prediction_results = self.predict_with_metadata(batch_texts)
            
            # Update dataframe
            for j, result in enumerate(prediction_results):
                idx = i + j
                df.at[idx, 'mistral_prediction'] = result['prediction']
                df.at[idx, 'is_parent_category'] = result['is_parent_category']
                df.at[idx, 'mistral_confidence'] = result['confidence']
                df.at[idx, 'mistral_parent_category'] = result['parent_category']
        
        if output_csv:
            df.to_csv(output_csv, index=False)
            print(f"Results saved to {output_csv}")
        
        return df


def main():
    parser = argparse.ArgumentParser(description='Make predictions with Mistral classifier')
    
    # Model configuration
    parser.add_argument('--model_id', type=str, 
                        default='ft:classifier:ministral-3b-latest:2bec22ef:20250702:a2707cf5',
                        help='Mistral model ID')
    parser.add_argument('--api_key', type=str,
                        help='Mistral API key (or set MISTRAL_API_KEY env var)')
    
    # Input options
    parser.add_argument('--input', type=str,
                        help='Input text file or CSV file')
    parser.add_argument('--text', type=str,
                        help='Single text string to classify')
    parser.add_argument('--text_column', type=str, default='composite',
                        help='Column name for text in CSV input')
    
    # Output options
    parser.add_argument('--output', type=str,
                        help='Output CSV file (for CSV input)')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed prediction metadata')
    
    # Processing options
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size for API calls')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                        help='Minimum confidence threshold')
    
    args = parser.parse_args()
    
    # Initialize predictor
    try:
        print(f"Loading Mistral model {args.model_id}...")
        predictor = MistralPredictor(args.model_id, args.api_key)
    except Exception as e:
        print(f"Error initializing Mistral client: {e}")
        sys.exit(1)
    
    if args.text:
        # Single text prediction
        print(f"\nClassifying text: {args.text[:100]}...")
        
        if args.verbose:
            results = predictor.predict_with_metadata([args.text], args.confidence_threshold)
            result = results[0]
            
            print(f"\nPrediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Parent category: {result['parent_category']}")
            print(f"Is parent category: {result['is_parent_category']}")
            
            print("\nTop 5 domain scores:")
            sorted_domains = sorted(result['all_domain_scores'].items(), 
                                  key=lambda x: x[1], reverse=True)[:5]
            for domain, score in sorted_domains:
                print(f"  {domain}: {score:.4f}")
        else:
            prediction = predictor.predict([args.text])[0]
            print(f"Prediction: {prediction}")
    
    elif args.input:
        input_path = Path(args.input)
        
        if input_path.suffix.lower() == '.csv':
            # CSV file prediction
            print(f"\nProcessing CSV file: {args.input}")
            df = predictor.batch_predict_csv(args.input, args.text_column, 
                                           args.output, args.batch_size)
            
            print(f"\nPrediction summary:")
            print(df['mistral_prediction'].value_counts().head(10))
            
            if args.verbose:
                parent_count = df['is_parent_category'].sum()
                print(f"\nParent category predictions: {parent_count}/{len(df)} ({parent_count/len(df)*100:.1f}%)")
                print(f"Average confidence: {df['mistral_confidence'].mean():.3f}")
        
        elif input_path.suffix.lower() == '.txt':
            # Text file prediction
            with open(args.input, 'r') as f:
                texts = [line.strip() for line in f if line.strip()]
            
            print(f"\nProcessing {len(texts)} texts from file...")
            
            if args.verbose:
                results = predictor.predict_with_metadata(texts, args.confidence_threshold)
                for i, result in enumerate(results):
                    print(f"\nText {i+1}: {result['text']}")
                    print(f"Prediction: {result['prediction']} (confidence: {result['confidence']:.3f})")
                    if result['is_parent_category']:
                        print("  (Parent category)")
            else:
                predictions = predictor.predict(texts)
                for i, (text, pred) in enumerate(zip(texts, predictions)):
                    print(f"Text {i+1}: {text[:50]}... -> {pred}")
        
        else:
            print(f"Unsupported file format: {input_path.suffix}")
            sys.exit(1)
    
    else:
        print("Please provide either --text or --input")
        sys.exit(1)


if __name__ == "__main__":
    main()