#!/usr/bin/env python3
"""
SetFit Entity Classification Prediction CLI

Make predictions using a trained SetFit hierarchical classifier.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import pandas as pd
from setfit import SetFitModel


class HierarchicalPredictor:
    """
    Load and use a trained hierarchical classifier for predictions.
    """
    
    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self.model = None
        self.label_mapping = None
        self.parent_mapping = None
        self.rare_classes = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model and metadata."""
        # Load the SetFit model
        model_path = self.model_dir / "setfit_model"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        self.model = SetFitModel.from_pretrained(str(model_path))
        print(f"Loaded SetFit model from {model_path}")
        
        # Load metadata
        metadata_path = self.model_dir / "metadata.pkl"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")
        
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.label_mapping = metadata['label_mapping']
        self.parent_mapping = metadata['parent_mapping']
        self.rare_classes = set(metadata['rare_classes'])
        
        print(f"Loaded metadata: {len(self.label_mapping)} label mappings, {len(self.rare_classes)} rare classes")
    
    def predict(self, texts, return_confidence=False):
        """
        Make predictions on texts.
        
        Args:
            texts: List of text strings to classify
            return_confidence: Whether to return confidence scores (if available)
        
        Returns:
            List of predictions, optionally with confidence scores
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Get model predictions (hierarchical level)
        predictions = self.model.predict(texts)
        
        if return_confidence:
            # SetFit doesn't directly provide confidence scores
            # For production use, you might want to modify the model to return probabilities
            return [(pred, None) for pred in predictions]
        
        return predictions
    
    def predict_with_metadata(self, texts):
        """
        Make predictions with additional metadata about the prediction process.
        
        Returns:
            List of dictionaries with prediction details
        """
        if isinstance(texts, str):
            texts = [texts]
        
        predictions = self.predict(texts)
        results = []
        
        for text, pred in zip(texts, predictions):
            # Determine if this prediction used hierarchical mapping
            is_hierarchical = any(
                mapped_label == pred and original_label in self.rare_classes
                for original_label, mapped_label in self.label_mapping.items()
            )
            
            # Find original labels that map to this prediction
            original_labels = [
                orig for orig, mapped in self.label_mapping.items() 
                if mapped == pred
            ]
            
            result = {
                'text': text[:100] + '...' if len(text) > 100 else text,
                'prediction': pred,                
                'is_parent_category': pred in self.parent_mapping.values()
            }
            results.append(result)
        
        return results
    
    def batch_predict_csv(self, input_csv, text_column='composite', output_csv=None):
        """
        Make predictions on a CSV file.
        
        Args:
            input_csv: Path to input CSV file
            text_column: Name of column containing text to classify
            output_csv: Path to output CSV file (optional)
        
        Returns:
            DataFrame with predictions added
        """
        df = pd.read_csv(input_csv)
        
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in CSV. Available columns: {list(df.columns)}")
        
        print(f"Making predictions on {len(df)} rows...")
        
        # Get predictions with metadata
        texts = df[text_column].tolist()
        prediction_results = self.predict_with_metadata(texts)
        
        # Add predictions to dataframe
        df['setfit_prediction'] = [r['prediction'] for r in prediction_results]
        df['is_parent_category'] = [r['is_parent_category'] for r in prediction_results]
        
        if output_csv:
            df.to_csv(output_csv, index=False)
            print(f"Results saved to {output_csv}")
        
        return df


def main():
    parser = argparse.ArgumentParser(description='Make predictions with trained SetFit classifier')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing trained model')
    parser.add_argument('--input', type=str,
                        help='Input text file or CSV file')
    parser.add_argument('--text', type=str,
                        help='Single text string to classify')
    parser.add_argument('--text_column', type=str, default='composite',
                        help='Column name for text in CSV input')
    parser.add_argument('--output', type=str,
                        help='Output CSV file (for CSV input)')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed prediction metadata')
    
    args = parser.parse_args()
    
    # Load predictor
    print(f"Loading model from {args.model_dir}...")
    predictor = HierarchicalPredictor(args.model_dir)
    
    if args.text:
        # Single text prediction
        print(f"\nPredicting for text: {args.text[:100]}...")
        
        if args.verbose:
            results = predictor.predict_with_metadata([args.text])
            result = results[0]
            print(f"\nPrediction: {result['prediction']}")
            print(f"Is parent category: {result['is_parent_category']}")            
        else:
            prediction = predictor.predict([args.text])[0]
            print(f"Prediction: {prediction}")
    
    elif args.input:
        input_path = Path(args.input)
        
        if input_path.suffix.lower() == '.csv':
            # CSV file prediction
            print(f"\nProcessing CSV file: {args.input}")
            df = predictor.batch_predict_csv(args.input, args.text_column, args.output)
            
            print(f"\nPrediction summary:")
            print(df['setfit_prediction'].value_counts())
            
            # if args.verbose:
            #     hierarchical_count = df['is_hierarchical_prediction'].sum()
            #     print(f"\nHierarchical predictions: {hierarchical_count}/{len(df)} ({hierarchical_count/len(df)*100:.1f}%)")
        
        elif input_path.suffix.lower() == '.txt':
            # Text file prediction
            with open(args.input, 'r') as f:
                texts = [line.strip() for line in f if line.strip()]
            
            print(f"\nProcessing {len(texts)} texts from file...")
            
            if args.verbose:
                results = predictor.predict_with_metadata(texts)
                for i, result in enumerate(results):
                    print(f"\nText {i+1}: {result['text']}")
                    print(f"Prediction: {result['prediction']}")
                    # if result['is_hierarchical_prediction']:
                    #     print("  (Hierarchical prediction)")
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