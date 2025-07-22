#!/usr/bin/env python3
"""
Enhanced Reporting Module for Entity Resolution

This module provides comprehensive logging and reporting capabilities for the feature optimization
pipeline. It generates structured logs, detailed CSV reports, and interactive visualizations
to help analyze and interpret optimization results.

Key features:
1. Structured logging framework that captures configurations with their performance metrics
2. CSV reporting of all tested configurations with detailed metrics
3. Parameter correlation analysis to identify relationships between settings and performance
4. Interactive HTML dashboard for result exploration
5. Automatic visualization generation for result analysis
"""

import os
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import re
import shutil
from datetime import datetime
from pathlib import Path

# Add file locking for thread safety
try:
    from filelock import FileLock
except ImportError:
    # Simple fallback implementation if filelock is not available
    class FileLock:
        def __init__(self, lock_file):
            self.lock_file = lock_file
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

logger = logging.getLogger(__name__)


def generate_report(config, hash_lookup=None, string_dict=None, prefix='entity_resolution', template_path=None):
    """
    Generate comprehensive report from pipeline results.
    
    Args:
        config: Configuration dictionary
        hash_lookup: Hash lookup dictionary for entity resolution
        string_dict: String dictionary for entity resolution
        prefix: Prefix for output files
        template_path: Path to report template (optional)
        
    Returns:
        Dictionary with reporting metrics
    """
    # Initialize parameters
    output_dir = Path(config.get('output_dir', 'data/output'))
    reporter = FeatureOptimizationReporter(output_dir, config)
    
    # Generate the report
    report_path = reporter.generate_configuration_report(output_dir / f"{prefix}_report.csv")
    
    # Generate HTML report if template provided
    html_report_path = output_dir / f"{prefix}_report.html"
    
    # Generate dashboard visualization
    dashboard_path = reporter.generate_interactive_dashboard(output_dir / f"configuration_dashboard.html")
    
    # Return metrics dictionary
    return {
        'status': 'completed',
        'report_path': str(report_path),
        'dashboard_path': str(dashboard_path),
        'entities_processed': len(hash_lookup) if hash_lookup else 0,
        'unique_strings': len(string_dict) if string_dict else 0
    }

def generate_detailed_test_results(classifier, test_pairs, test_features, test_labels, normalized_features, feature_names, output_dir=None, config=None, feature_engineering=None, include_normalized=True):
    """
    Generate detailed test results for model evaluation.
    
    This function creates a detailed CSV report of model predictions for test pairs,
    including both raw and normalized feature values, confidence scores, and correctness of predictions.
    
    Args:
        classifier: The trained classifier model
        test_pairs: List of test pairs (left_id, right_id, match_label)
        test_features: Raw feature vectors used for testing
        test_labels: Ground truth labels
        normalized_features: Normalized feature vectors used for model predictions
        feature_names: Names of features
        output_dir: Directory to save results (default: None)
        config: Configuration dictionary
        feature_engineering: Optional FeatureEngineering instance for diagnostics
        
    Returns:
        Path to the generated detailed test results CSV file
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path
    from datetime import datetime
    from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc
    
    # Create output directory if provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        return None  # Can't generate report without output directory
    
    # Get predictions from the classifier (use appropriate features based on scaling_disabled flag)
    scaling_disabled = config.get('scaling_disabled', False) if config else False
    
    # If scaling is disabled, we should use raw features for prediction
    features_for_prediction = test_features if scaling_disabled else normalized_features
    
    y_proba = classifier.predict_proba(features_for_prediction)
    y_pred = classifier.predict(features_for_prediction)
    
    if scaling_disabled:
        logger.warning("*** SCALING IS DISABLED - Using raw feature values for prediction ***")
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, y_pred, average='binary', zero_division=0
    )
    
    # Calculate confusion matrix components
    true_positives = np.sum((y_pred == 1) & (test_labels == 1))
    false_positives = np.sum((y_pred == 1) & (test_labels == 0))
    true_negatives = np.sum((y_pred == 0) & (test_labels == 0))
    false_negatives = np.sum((y_pred == 0) & (test_labels == 1))
    
    # Calculate additional metrics
    specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
    accuracy = (true_positives + true_negatives) / len(test_labels) if len(test_labels) > 0 else 0
    
    # Create detailed results data
    results_data = []
    
    # Add diagnostic output for identical strings with incorrect indicator values
    problematic_pairs = []
    
    # Look for any binary indicator features that are actually enabled
    binary_indicator_features = [
        'person_low_cosine_indicator',
        'person_low_levenshtein_indicator', 
        'person_low_jaro_winkler_indicator'
    ]
    
    # Find the first enabled binary indicator feature for diagnostics
    indicator_feature_name = None
    indicator_idx = -1
    for feature_name in binary_indicator_features:
        if feature_name in feature_names:
            indicator_feature_name = feature_name
            indicator_idx = feature_names.index(feature_name)
            break
    
    # If feature_engineering was passed as a parameter, use it directly
    # Otherwise check config or globals
    if feature_engineering is None:
        # Try to access feature engineering instance for string lookup
        # First, check if feature_engineering_instance is directly in config
        if config and hasattr(config, 'get') and config.get('feature_engineering_instance'):
            feature_engineering = config.get('feature_engineering_instance')
        # Then try to access it via the global scope if available
        elif 'feature_engineering' in globals():
            feature_engineering = globals()['feature_engineering']
    
    # Only run diagnostics if we have an enabled indicator feature
    if indicator_feature_name:
        # Log diagnostic info only when relevant
        logger.info(f"Running diagnostics for indicator feature: {indicator_feature_name}")
        logger.debug(f"Feature engineering instance available: {feature_engineering is not None}")
        logger.debug(f"{indicator_feature_name} feature index: {indicator_idx}")
    else:
        logger.debug(f"No binary indicator features enabled - skipping indicator diagnostics")
    
    # Log feature names to help with debugging
    logger.info(f"Feature names: {feature_names}")
    
    for i, (left_id, right_id, match_label) in enumerate(test_pairs):
        # Convert match label to boolean
        is_match = match_label.lower() == 'true'
        
        # Get prediction details
        predicted_prob = y_proba[i]
        predicted_match = y_pred[i] == 1
        is_correct = (predicted_match == is_match)
        
        # Get raw feature values
        feature_values = {}
        for j, name in enumerate(feature_names):
            feature_values[f"raw_{name}"] = test_features[i, j]
            
        # Get normalized feature values
        normalized_feature_values = {}
        for j, name in enumerate(feature_names):
            normalized_feature_values[f"norm_{name}"] = normalized_features[i, j]
        
        # Add to results data
        result = {
            'left_id': left_id,
            'right_id': right_id,
            'true_match': is_match,
            'predicted_match': predicted_match,
            'confidence': predicted_prob,
            'is_correct': is_correct,
            'prediction_type': (
                'true_positive' if is_match and predicted_match else
                'false_positive' if not is_match and predicted_match else
                'true_negative' if not is_match and not predicted_match else
                'false_negative'
            )
        }
        
        # Add raw and normalized feature values
        result.update(feature_values)
        result.update(normalized_feature_values)
        
        # DIAGNOSTIC: Check for identical strings with incorrect indicator values
        # Only run this check if we have an enabled binary indicator feature
        if indicator_idx >= 0 and indicator_feature_name and feature_engineering is not None:
            try:
                # Get string values
                left_person = feature_engineering._get_string_value(left_id, 'person')
                right_person = feature_engineering._get_string_value(right_id, 'person')
                
                # Check if strings are identical but indicator value is wrong
                is_identical = (left_person == right_person and left_person != "")
                raw_indicator_value = test_features[i, indicator_idx]
                norm_indicator_value = normalized_features[i, indicator_idx]
                
                if is_identical and (raw_indicator_value != 0.0 or norm_indicator_value != 0.0):
                    # Log the anomaly
                    logger.error(f"CRITICAL: Identical strings with incorrect {indicator_feature_name} value:")
                    logger.error(f"  Pair: ({left_id}, {right_id})")
                    logger.error(f"  Strings: '{left_person}' vs '{right_person}'")
                    logger.error(f"  Raw indicator value: {raw_indicator_value}")
                    logger.error(f"  Normalized indicator value: {norm_indicator_value}")
                    logger.error(f"  Index in batch: {i}")
                    
                    # Add to problematic pairs list
                    problematic_pairs.append({
                        "left_id": left_id,
                        "right_id": right_id, 
                        "left_person": left_person,
                        "right_person": right_person,
                        "raw_indicator_value": float(raw_indicator_value),
                        "norm_indicator_value": float(norm_indicator_value),
                        "is_identical": is_identical,
                        "batch_index": i,
                        "prediction_type": result['prediction_type'],
                        "feature_name": indicator_feature_name
                    })
            except Exception as e:
                logger.warning(f"Error checking string values: {str(e)}")
        
        # Add to results list
        results_data.append(result)
    
    # Always save report, even if empty
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    problematic_path = output_dir / f"problematic_indicators_{timestamp}.json"
    
    try:
        # If we couldn't check strings due to missing feature_engineering or no indicator features, log that
        if indicator_idx < 0 or feature_engineering is None:
            diagnostic_info = {
                "timestamp": timestamp,
                "error": "Could not check for problematic indicator values",
                "reasons": {
                    "feature_engineering_available": feature_engineering is not None,
                    "indicator_feature_name": indicator_feature_name,
                    "indicator_index": indicator_idx,
                    "feature_names": feature_names,
                    "available_indicators": [f for f in binary_indicator_features if f in feature_names]
                }
            }
            # Only add diagnostic info if no actual problems were found
            if not problematic_pairs:
                problematic_pairs = [diagnostic_info]
        
        # Convert numpy values to Python types for JSON serialization
        for pair in problematic_pairs:
            for key, value in pair.items():
                if isinstance(value, np.ndarray):
                    pair[key] = value.tolist()
                elif isinstance(value, np.floating):
                    pair[key] = float(value)
                elif isinstance(value, np.integer):
                    pair[key] = int(value)
                elif hasattr(value, '__name__'):  # For functions and classes
                    pair[key] = value.__name__
        
        # Create diagnostic output with summary
        diagnostic_output = {
            "timestamp": timestamp,
            "problematic_pairs_count": len(problematic_pairs),
            "feature_engineering_available": feature_engineering is not None,
            "indicator_feature_name": indicator_feature_name,
            "indicator_index": indicator_idx,
            "pairs_checked": len(test_pairs),
            "pairs_with_issues": problematic_pairs
        }
        
        # Write to JSON file
        with open(problematic_path, 'w') as f:
            json.dump(diagnostic_output, f, indent=2)
            
        # Only report problems if we actually found indicator issues (not just missing features)
        actual_problems = [p for p in problematic_pairs if not p.get('error')]
        if actual_problems:
            logger.warning(f"Found {len(actual_problems)} pairs with identical strings but incorrect {indicator_feature_name} values")
        elif indicator_feature_name:
            logger.info(f"No problematic {indicator_feature_name} values found in {len(test_pairs)} test pairs")
        else:
            logger.debug(f"Skipped indicator diagnostics - no binary indicator features enabled")
            
        logger.info(f"Saved diagnostic report to {problematic_path}")
    except Exception as e:
        logger.error(f"Error saving diagnostic report: {str(e)}")
    
    # Create DataFrame from results data
    results_df = pd.DataFrame(results_data)
    
    # Create timestamped filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_filename = f"detailed_test_results_{timestamp}.csv"
    results_path = output_dir / results_filename
    
    # Save to CSV
    results_df.to_csv(results_path, index=False)
    
    # Also generate summary metrics
    summary = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'accuracy': accuracy,
        'true_positives': int(true_positives),
        'false_positives': int(false_positives),
        'true_negatives': int(true_negatives),
        'false_negatives': int(false_negatives),
        'test_count': len(test_pairs),
        'timestamp': timestamp,
        'includes_raw_and_normalized_features': True,
        'raw_feature_prefix': 'raw_',
        'normalized_feature_prefix': 'norm_'
    }
    
    # Save summary metrics
    summary_path = output_dir / f"test_summary_{timestamp}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Generate visualizations if output directory is provided
    # Create feature importance visualization
    if hasattr(classifier, 'weights') and classifier.weights is not None:
        # Use actual model weights for feature importance
        feature_weights = classifier.weights
        
        plt.figure(figsize=(10, 8))
        feature_indices = np.argsort(np.abs(feature_weights))[::-1]
        plt.barh(range(len(feature_indices)), np.abs(feature_weights)[feature_indices])
        plt.yticks(range(len(feature_indices)), [feature_names[i] for i in feature_indices])
        plt.xlabel('Feature Importance')
        plt.title('Feature Importance Analysis')
        
        viz_path = output_dir / f"feature_importance_{timestamp}.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return str(results_path)


class FeatureOptimizationReporter:
    """
    Comprehensive reporting system for feature optimization results.
    
    This class provides structured logging, detailed reports, and visualizations
    for analyzing feature optimization results. It helps identify the most effective
    feature configurations and understand parameter influence on performance.
    """
    
    def __init__(self, output_dir, base_config=None):
        """
        Initialize the reporter with output directory.
        
        Args:
            output_dir: Directory for saving reports and visualizations
            base_config: Optional base configuration for reference
        """
        self.output_dir = Path(output_dir)
        self.base_config = base_config
        self.results = []
        self.csv_initialized = False
        self.common_parameters = [
            'person_low_jaro_winkler_indicator_threshold',
            'person_low_levenshtein_indicator_threshold',
            'person_jaro_winkler_similarity_weight',
            'person_levenshtein_similarity_weight'
        ]
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create visualization directory
        self.viz_dir = self.output_dir / 'visualizations'
        self.viz_dir.mkdir(exist_ok=True)
        
        # Create logs directory within output directory
        self.logs_dir = self.output_dir / 'logs'
        self.logs_dir.mkdir(exist_ok=True)
        
        # CSV configuration
        self.csv_path = self.output_dir / 'configuration_results.csv'
        self.csv_columns = [
            "configuration_name", "precision", "recall", "f1", "specificity", 
            "accuracy", "npv", "balanced_score", "precision_weight", "decision_threshold", "perfect_precision", "perfect_recall", "execution_time",
            "similarity_approach", "enabled_feature_count",
            "person_low_jaro_winkler_indicator_threshold", "person_low_levenshtein_indicator_threshold",
            "person_jaro_winkler_similarity_weight", "person_levenshtein_similarity_weight",
            "person_low_jaro_winkler_indicator_enabled", "person_low_levenshtein_indicator_enabled",
            "person_jaro_winkler_similarity_enabled", "person_levenshtein_similarity_enabled",
            "source"
        ]
        self._initialize_csv()
        
        logger.info(f"Initialized FeatureOptimizationReporter with output dir: {output_dir}")
        
    def _initialize_csv(self):
        """Initialize CSV with headers if it doesn't exist"""
        try:
            # Use file locking for thread safety
            lock_file = self.csv_path.with_suffix(self.csv_path.suffix + ".lock")
            try:
                # Create lock file
                with open(lock_file, 'w') as f:
                    f.write(str(datetime.now()))
                    
                # Create the directory if it doesn't exist
                self.csv_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Initialize with headers if the file doesn't exist or is empty
                if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
                    with open(self.csv_path, 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=self.csv_columns)
                        writer.writeheader()
                        self.csv_initialized = True
                        logger.debug(f"Initialized CSV with headers at {self.csv_path}")
            finally:
                # Remove lock file
                if lock_file.exists():
                    lock_file.unlink()
        except Exception as e:
            logger.error(f"Error initializing CSV file: {str(e)}")
    
    def _log_to_jsonl(self, result):
        """Write configuration result to JSONL file"""
        log_file = self.logs_dir / 'configuration_results.jsonl'
        with open(log_file, 'a') as f:
            f.write(json.dumps(result) + '\n')
            
    def _log_to_csv(self, result):
        """Write single configuration to CSV file with file locking"""
        # Extract configuration details
        config = result.get("configuration", {})
        metrics = result.get("metrics", {})
        
        # Calculate additional metrics if necessary (like confusion matrix components)
        tp = metrics.get('true_positives', 0)
        fp = metrics.get('false_positives', 0)
        tn = metrics.get('true_negatives', 0)
        fn = metrics.get('false_negatives', 0)
        
        # Calculate specificity if not available
        specificity = metrics.get('specificity')
        if specificity is None and (tn + fp) > 0:
            specificity = tn / (tn + fp)
        
        # Calculate accuracy if not available
        accuracy = metrics.get('accuracy')
        if accuracy is None and (tp + tn + fp + fn) > 0:
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            
        # Calculate NPV (Negative Predictive Value) if not available
        npv = metrics.get('npv')
        if npv is None and (tn + fn) > 0:
            npv = tn / (tn + fn)
        
        # Prepare row for CSV - with explicit string conversions for every field
        row = {
            "configuration_name": str(result.get("configuration_name", "")),
            "precision": str(metrics.get("precision", "")),
            "recall": str(metrics.get("recall", "")),
            "f1": str(metrics.get("f1", metrics.get("f1_score", ""))),
            "specificity": str(specificity) if specificity is not None else "",
            "accuracy": str(accuracy) if accuracy is not None else "",
            "balanced_score": str(metrics.get("balanced_score", "")),
            "precision_weight": str(metrics.get("precision_weight", "0.5")),
            "decision_threshold": str(metrics.get("decision_threshold", "")),
            "perfect_precision": "1" if metrics.get("perfect_precision") else "0",
            "perfect_recall": "1" if metrics.get("perfect_recall") else "0",
            "execution_time": str(result.get("execution_time", "")),
            "source": "optimization_run",
            "npv": str(npv) if npv is not None else ""
        }
        
        # Determine similarity approach - using string values
        if "similarity_metrics" in config:
            similarity_metrics = config.get("similarity_metrics", {})
            if similarity_metrics.get("use_binary_indicators", False):
                if similarity_metrics.get("include_both_metrics", False):
                    row["similarity_approach"] = "hybrid"
                else:
                    row["similarity_approach"] = "binary"
            else:
                row["similarity_approach"] = "direct"
        else:
            row["similarity_approach"] = ""
        
        # Add enabled feature count - explicitly as string
        enabled_features = config.get("enabled", [])
        row["enabled_feature_count"] = str(len(enabled_features)) if enabled_features else ""
        
        # Add specific parameters from configuration with enhanced extraction - all as strings
        if "parameters" in config:
            params = config.get("parameters", {})
            
            # Extract thresholds and weights with complete error handling - explicitly as strings
            try:
                if "person_low_jaro_winkler_indicator" in params:
                    if isinstance(params["person_low_jaro_winkler_indicator"], dict):
                        threshold = params["person_low_jaro_winkler_indicator"].get("threshold")
                        row["person_low_jaro_winkler_indicator_threshold"] = str(threshold) if threshold is not None else ""
                    elif isinstance(params["person_low_jaro_winkler_indicator"], (int, float)):
                        row["person_low_jaro_winkler_indicator_threshold"] = str(params["person_low_jaro_winkler_indicator"])
                else:
                    row["person_low_jaro_winkler_indicator_threshold"] = ""
            except Exception as e:
                logger.debug(f"Error extracting jaro_winkler threshold: {str(e)}")
                row["person_low_jaro_winkler_indicator_threshold"] = ""
                
            try:
                if "person_low_levenshtein_indicator" in params:
                    if isinstance(params["person_low_levenshtein_indicator"], dict):
                        threshold = params["person_low_levenshtein_indicator"].get("threshold")
                        row["person_low_levenshtein_indicator_threshold"] = str(threshold) if threshold is not None else ""
                    elif isinstance(params["person_low_levenshtein_indicator"], (int, float)):
                        row["person_low_levenshtein_indicator_threshold"] = str(params["person_low_levenshtein_indicator"])
                else:
                    row["person_low_levenshtein_indicator_threshold"] = ""
            except Exception as e:
                logger.debug(f"Error extracting levenshtein threshold: {str(e)}")
                row["person_low_levenshtein_indicator_threshold"] = ""
                
            try:
                if "person_jaro_winkler_similarity" in params:
                    if isinstance(params["person_jaro_winkler_similarity"], dict):
                        weight = params["person_jaro_winkler_similarity"].get("weight")
                        row["person_jaro_winkler_similarity_weight"] = str(weight) if weight is not None else ""
                    elif isinstance(params["person_jaro_winkler_similarity"], (int, float)):
                        row["person_jaro_winkler_similarity_weight"] = str(params["person_jaro_winkler_similarity"])
                else:
                    row["person_jaro_winkler_similarity_weight"] = ""
            except Exception as e:
                logger.debug(f"Error extracting jaro_winkler weight: {str(e)}")
                row["person_jaro_winkler_similarity_weight"] = ""
                
            try:
                if "person_levenshtein_similarity" in params:
                    if isinstance(params["person_levenshtein_similarity"], dict):
                        weight = params["person_levenshtein_similarity"].get("weight")
                        row["person_levenshtein_similarity_weight"] = str(weight) if weight is not None else ""
                    elif isinstance(params["person_levenshtein_similarity"], (int, float)):
                        row["person_levenshtein_similarity_weight"] = str(params["person_levenshtein_similarity"])
                else:
                    row["person_levenshtein_similarity_weight"] = ""
            except Exception as e:
                logger.debug(f"Error extracting levenshtein weight: {str(e)}")
                row["person_levenshtein_similarity_weight"] = ""
        
        # Add feature enabled flags - explicitly as strings
        for feature_name in ["person_low_jaro_winkler_indicator", "person_low_levenshtein_indicator", 
                           "person_jaro_winkler_similarity", "person_levenshtein_similarity"]:
            row[f"{feature_name}_enabled"] = "1" if feature_name in enabled_features else "0"
                
        # Add balanced score calculation if not present but we have precision and recall
        if 'balanced_score' not in row or not row['balanced_score']:
            try:
                precision = float(row.get('precision', 0))
                recall = float(row.get('recall', 0))
                precision_weight = float(row.get('precision_weight', 0.5))
                
                # Calculate balanced score using weighted harmonic mean
                if precision > 0 and recall > 0 and 0 <= precision_weight <= 1:
                    weight_ratio = precision_weight / (1.0 - precision_weight) if precision_weight < 1.0 else 1.0
                    balanced_score = ((1 + weight_ratio) * precision * recall) / \
                                   (weight_ratio * precision + recall) if (weight_ratio * precision + recall) > 0 else 0.0
                    row['balanced_score'] = str(balanced_score)
            except Exception as e:
                logger.debug(f"Error calculating balanced score: {str(e)}")
                row['balanced_score'] = ""
            
        # Ensure all required columns are populated (fill with empty strings if missing)
        for col in self.csv_columns:
            if col not in row or row[col] is None:
                row[col] = ""
        
        # Use file locking to safely write to CSV
        lock_file = self.csv_path.with_suffix(self.csv_path.suffix + ".lock")
        try:
            # Create lock file
            with open(lock_file, 'w') as f:
                f.write(str(datetime.now()))
                
            # Append to CSV file
            file_exists = self.csv_path.exists() and self.csv_path.stat().st_size > 0
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.csv_columns)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
                
        finally:
            # Remove lock file
            if lock_file.exists():
                lock_file.unlink()
                
    def initialize_results_csv(self, file_path=None):
        """Initialize the CSV file for storing configuration results with appropriate headers."""
        if file_path is None:
            file_path = self.csv_path
        else:
            file_path = Path(file_path)
        
        # Create the directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create the CSV file with headers
        try:
            with FileLock(str(file_path) + ".lock"):
                # Check if file exists and verify header alignment
                needs_new_file = False
                if file_path.exists() and file_path.stat().st_size > 0:
                    try:
                        with open(file_path, 'r', newline='') as f:
                            reader = csv.reader(f)
                            existing_headers = next(reader, [])
                            
                            # If headers don't match our columns, create a corrected file
                            if existing_headers != self.csv_columns:
                                logger.warning(f"CSV column misalignment detected - creating corrected file")
                                
                                # Skip backup creation - just fix the file directly
                                
                                # Re-create file with correct headers
                                needs_new_file = True
                                
                                # Try to recover valid rows directly from the file
                                try:
                                    # Re-read all rows from current file
                                    recovered_rows = []
                                    with open(file_path, 'r', newline='') as f:
                                        reader = csv.reader(f)
                                        _ = next(reader)  # Skip header row
                                        
                                        # Process each row, skipping header repetitions
                                        for row_data in reader:
                                            # Skip rows that look like headers
                                            if row_data and row_data[0] != 'configuration_name':
                                                recovered_rows.append(row_data)
                                    
                                    if recovered_rows:
                                        logger.info(f"Recovered {len(recovered_rows)} valid rows from CSV")
                                        
                                        # Create new file with correct header and recovered rows
                                        with open(file_path, 'w', newline='') as f:
                                            writer = csv.writer(f)
                                            writer.writerow(self.csv_columns)  # Write correct header
                                            writer.writerows(recovered_rows)    # Write recovered rows
                                        
                                        needs_new_file = False  # File already created with recovered rows
                                        logger.info(f"Created corrected CSV file")
                                except Exception as recover_error:
                                    logger.warning(f"Could not recover rows from CSV: {recover_error}")
                    except Exception as e:
                        logger.warning(f"Error checking CSV headers: {e}")
                        needs_new_file = True
                else:
                    # File doesn't exist or is empty
                    needs_new_file = True
                
                # Create a new file if needed
                if needs_new_file:
                    with open(file_path, 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=self.csv_columns)
                        writer.writeheader()
                    logger.info(f"Initialized configuration results CSV at {file_path}")
                else:
                    logger.info(f"Configuration results CSV already exists at {file_path}")
                
            self.csv_initialized = True
            return file_path
        except Exception as e:
            logger.error(f"Error initializing CSV file at {file_path}: {str(e)}")
            return None
    
    def _determine_similarity_approach(self, config):
        """
        Determine the similarity approach used in the configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            String describing the similarity approach
        """
        if config.get('similarity_metrics', {}).get('use_binary_indicators', False):
            if config.get('similarity_metrics', {}).get('include_both_metrics', False):
                return 'hybrid'
            else:
                return 'binary'
        else:
            return 'direct'
    
    def _extract_parameter_value(self, config, param_name):
        """
        Extract parameter value from configuration.
        
        Args:
            config: Configuration dictionary
            param_name: Name of the parameter to extract
            
        Returns:
            Parameter value or None if not found
        """
        # Handle composite parameter names (e.g., person_low_jaro_winkler_indicator_threshold)
        parts = param_name.split('_')
        
        # Find the base feature name and property
        if len(parts) >= 2 and parts[-1] in ['threshold', 'weight']:
            feature_name = '_'.join(parts[:-1])
            property_name = parts[-1]
            
            # Check if the parameter exists in the configuration
            if 'parameters' in config and feature_name in config['parameters']:
                param_value = config['parameters'][feature_name].get(property_name)
                return param_value
        
        # Direct parameter lookup
        return config.get('parameters', {}).get(param_name)
    
    def _extract_result_data(self, result):
        """
        Extract data from a result dictionary for CSV row.
        
        Args:
            result: Result dictionary from configuration evaluation
            
        Returns:
            List of values for CSV row
        """
        config = result.get('configuration', {})
        metrics = result.get('metrics', {})
        
        # Calculate metrics that might be missing
        specificity = metrics.get('specificity', 0)
        accuracy = metrics.get('accuracy', 0)
        
        # Determine similarity approach
        similarity_approach = self._determine_similarity_approach(config)
        
        # Extract enabled features
        enabled_features = config.get('enabled', [])
        enabled_feature_count = len(enabled_features) if enabled_features else 0
        
        # Build the basic row data
        row_data = [
            result.get('configuration_name', 'unknown'),
            metrics.get('precision', 0),
            metrics.get('recall', 0),
            metrics.get('f1', 0),
            specificity,
            accuracy,
            metrics.get('decision_threshold', 0),
            1 if metrics.get('perfect_precision', False) else 0,
            result.get('execution_time', 0),
            similarity_approach,
            enabled_feature_count
        ]
        
        # Add parameter values
        for param in self.common_parameters:
            param_value = self._extract_parameter_value(config, param)
            row_data.append(param_value if param_value is not None else '')
        
        # Add feature toggle flags
        feature_names = [
            'person_low_jaro_winkler_indicator',
            'person_low_levenshtein_indicator',
            'person_jaro_winkler_similarity',
            'person_levenshtein_similarity'
        ]
        for feature in feature_names:
            row_data.append(1 if feature in enabled_features else 0)
        
        return row_data
    
    def append_result_to_csv(self, result, file_path=None):
        """
        Append a single configuration result to the CSV file.
        
        Args:
            result: Result dictionary from configuration evaluation
            file_path: Path to the CSV file (if None, uses default path)
            
        Returns:
            Boolean indicating success
        """
        if file_path is None:
            file_path = self.csv_path
        else:
            file_path = Path(file_path)
        
        try:
            # Use the new _log_to_csv method to write result to CSV
            self._log_to_csv(result)
            logger.debug(f"Appended result for {result.get('configuration_name')} to CSV")
            return True
        except Exception as e:
            logger.error(f"Error appending result to CSV: {str(e)}")
            return False
    
    def log_configuration_result(self, result):
        """
        Log structured information about a configuration and its results.
        
        Args:
            result: Dictionary containing configuration and metrics data
        """
        self.results.append(result)
        
        # Extract key data for logging
        config_name = result.get('configuration_name', 'unknown')
        metrics = result.get('metrics', {})
        parameters = result.get('configuration', {}).get('parameters', {})
        
        # Process confusion matrix metrics if available
        tp = metrics.get('true_positives', 0)
        fp = metrics.get('false_positives', 0)
        tn = metrics.get('true_negatives', 0)
        fn = metrics.get('false_negatives', 0)
        
        # Calculate additional metrics if necessary
        if not metrics.get('specificity') and tn + fp > 0:
            metrics['specificity'] = tn / (tn + fp)
            
        if not metrics.get('accuracy') and (tp + tn + fp + fn) > 0:
            metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
            
        if not metrics.get('npv') and (tn + fn) > 0:
            metrics['npv'] = tn / (tn + fn)  # Negative Predictive Value
        
        # Create structured log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "configuration_id": config_name,
            "parameters": parameters,
            "metrics": metrics,
            "execution_time": result.get('execution_time', 0)
        }
        
        # Save to structured log file
        log_file = self.logs_dir / 'configuration_results.jsonl'
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        # Log summary to application log
        logger.info(f"Configuration {config_name} - precision={metrics.get('precision', 0):.4f}, "
                   f"recall={metrics.get('recall', 0):.4f}, f1={metrics.get('f1', 0):.4f}, "
                   f"specificity={metrics.get('specificity', 0):.4f}, accuracy={metrics.get('accuracy', 0):.4f}")
        
        # Write to JSONL and CSV simultaneously
        self._log_to_jsonl(result)
        self._log_to_csv(result)
    
    def _extract_missing_results_from_logs(self):
        """
        Extract configuration results from log files that might not be in our dataset.
        This helps capture results that weren't properly logged through the normal flow.
        
        Returns:
            List of additional configuration results extracted from logs
        """
        # CRITICAL FIX: Disable log extraction completely as it's causing data contamination
        # This prevents invalid configuration names from polluting the results
        logger.info("Log extraction disabled to prevent result contamination")
        return []
        
        # Previous implementation removed to stop invalid data from being included
    
    # Define canonical column ordering - MUST match order in _persist_configuration_state
    # This ensures correct column alignment between modules
    canonical_columns = [
        "configuration_name", "execution_stage", "timestamp", "status", 
        "fold_idx", "error_state", "error", "precision", "recall", "f1", 
        "specificity", "accuracy", "npv", "balanced_score", "precision_weight", "decision_threshold", 
        "perfect_precision", "perfect_recall", "execution_time", "worker_id",
        "similarity_approach", "enabled_feature_count",
        "person_low_jaro_winkler_indicator_threshold",
        "person_low_levenshtein_indicator_threshold",
        "person_jaro_winkler_similarity_weight",
        "person_levenshtein_similarity_weight",
        "person_low_jaro_winkler_indicator_enabled",
        "person_low_levenshtein_indicator_enabled",
        "person_jaro_winkler_similarity_enabled",
        "person_levenshtein_similarity_enabled",
        "source", "true_positives", "false_positives", 
        "true_negatives", "false_negatives"
    ]
    
    def generate_configuration_report(self, output_path=None):
        """
        Generate a comprehensive CSV report with all configurations and their metrics.
        If the file already exists, it ensures all results are properly included.
        
        Args:
            output_path: Path where the CSV report will be saved. If None, uses default path.
            
        Returns:
            Path to the generated CSV report
        """
        if output_path is None:
            output_path = self.csv_path
        
        # Extract configuration details and metrics
        report_data = []
        for result in self.results:
            # Skip configurations with errors
            if 'error' in result:
                continue
                
            # Extract feature parameters
            config = result.get('configuration', {})
            metrics = result.get('metrics', {})
            
            # Calculate additional metrics if they are not already present
            if ('true_positives' in metrics and 'false_positives' in metrics 
                and 'true_negatives' in metrics and 'false_negatives' in metrics):
                # Calculate specificity (true negative rate)
                tp = metrics.get('true_positives', 0)
                fp = metrics.get('false_positives', 0)
                tn = metrics.get('true_negatives', 0)
                fn = metrics.get('false_negatives', 0)
                
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
                npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative Predictive Value
                
                # Calculate balanced score if not provided and we have precision and recall
                if 'balanced_score' not in metrics and 'precision' in metrics and 'recall' in metrics:
                    precision = metrics.get('precision', 0.0)
                    recall = metrics.get('recall', 0.0)
                    precision_weight = metrics.get('precision_weight', 0.5)
                    # Weighted harmonic mean of precision and recall
                    if precision > 0 and recall > 0 and 0 <= precision_weight <= 1:
                        weight_ratio = precision_weight / (1.0 - precision_weight) if precision_weight < 1.0 else 1.0
                        balanced_score = ((1 + weight_ratio) * precision * recall) / \
                                         (weight_ratio * precision + recall) if (weight_ratio * precision + recall) > 0 else 0.0
                        metrics['balanced_score'] = balanced_score
            else:
                specificity = metrics.get('specificity', 0.0)
                accuracy = metrics.get('accuracy', 0.0)
                npv = metrics.get('npv', 0.0)
                
                # Calculate balanced score with default precision_weight if not provided
                if 'balanced_score' not in metrics and 'precision' in metrics and 'recall' in metrics:
                    precision = metrics.get('precision', 0.0)
                    recall = metrics.get('recall', 0.0)
                    precision_weight = metrics.get('precision_weight', 0.5)
                    # Weighted harmonic mean of precision and recall
                    if precision > 0 and recall > 0 and 0 <= precision_weight <= 1:
                        weight_ratio = precision_weight / (1.0 - precision_weight) if precision_weight < 1.0 else 1.0
                        balanced_score = ((1 + weight_ratio) * precision * recall) / \
                                         (weight_ratio * precision + recall) if (weight_ratio * precision + recall) > 0 else 0.0
                        metrics['balanced_score'] = balanced_score
            
            # Create report entry with configuration details
            entry = {
                'configuration_name': result.get('configuration_name', 'unknown'),
                'execution_time': result.get('execution_time', 0),
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'f1': metrics.get('f1', 0),
                'specificity': specificity,
                'accuracy': accuracy,
                'npv': npv,
                'balanced_score': metrics.get('balanced_score', 0),
                'precision_weight': metrics.get('precision_weight', 0.5),
                'decision_threshold': metrics.get('decision_threshold', 0),
                'perfect_precision': metrics.get('perfect_precision', False)
            }
            
            # Add feature-specific parameters
            for param_name, param_value in config.get('parameters', {}).items():
                if isinstance(param_value, dict):
                    for sub_param, sub_value in param_value.items():
                        entry[f"{param_name}_{sub_param}"] = sub_value
                else:
                    entry[param_name] = param_value
            
            # Add configuration type
            if config.get('similarity_metrics', {}).get('use_binary_indicators', False):
                if config.get('similarity_metrics', {}).get('include_both_metrics', False):
                    entry['similarity_approach'] = 'hybrid'
                else:
                    entry['similarity_approach'] = 'binary'
            else:
                entry['similarity_approach'] = 'direct'
            
            # Add enabled features
            enabled_features = config.get('enabled', [])
            entry['enabled_feature_count'] = len(enabled_features)
            for feature in ['person_low_jaro_winkler_indicator', 
                           'person_low_levenshtein_indicator',
                           'person_jaro_winkler_similarity',
                           'person_levenshtein_similarity']:
                entry[f'{feature}_enabled'] = 1 if feature in enabled_features else 0
                
            report_data.append(entry)
        
        # Create a new DataFrame from current results
        new_df = pd.DataFrame(report_data)

        # Check if file exists and already has data - with enhanced error recovery
        existing_df = None
        original_csv_path = None
        try:
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                with FileLock(str(output_path) + ".lock"):
                    try:
                        # CRITICAL: First check CSV header alignment
                        with open(output_path, 'r', newline='') as f:
                            reader = csv.reader(f)
                            existing_headers = next(reader, [])
                            
                        # Check if headers match the canonical ordering
                        if existing_headers and existing_headers != self.canonical_columns:
                            logger.warning(f"CSV column misalignment detected - headers do not match canonical ordering")
                            
                            # Create a backup of the problematic file
                            backup_path = f"{output_path}.misaligned.{int(time.time())}"
                            shutil.copy2(output_path, backup_path)
                            logger.info(f"Backed up misaligned CSV to {backup_path}")
                            
                            # Create a corrected file with proper column alignment
                            corrected_path = f"{output_path}.corrected.csv"
                            with open(output_path, 'r', newline='') as fin, open(corrected_path, 'w', newline='') as fout:
                                reader = csv.reader(fin)
                                writer = csv.writer(fout)
                                
                                # Write corrected header
                                writer.writerow(self.canonical_columns)
                                
                                # Map old columns to their indices
                                header_map = {h: i for i, h in enumerate(existing_headers) if h in self.canonical_columns}
                                
                                # Create a mapping from old positions to new positions
                                col_map = {}
                                for i, col in enumerate(self.canonical_columns):
                                    if col in header_map:
                                        col_map[header_map[col]] = i
                                
                                # Process data rows with proper column alignment
                                for i, row in enumerate(reader):
                                    if i == 0:  # Skip header row which we've already processed
                                        continue
                                        
                                    # Create a new row with the correct column ordering
                                    new_row = [None] * len(self.canonical_columns)
                                    
                                    # Map values from old columns to new columns
                                    for old_idx, new_idx in col_map.items():
                                        if old_idx < len(row):
                                            new_row[new_idx] = row[old_idx]
                                    
                                    writer.writerow(new_row)
                            
                            # Use the corrected file instead
                            logger.info(f"Using corrected CSV file: {corrected_path}")
                            existing_df = pd.read_csv(corrected_path)
                        else:
                            # Headers match canonical ordering, read normally
                            existing_df = pd.read_csv(output_path)
                            
                        logger.info(f"Loaded existing configuration report with {len(existing_df)} entries")
                        
                    except Exception as csv_error:
                        # Error recovery: Create a backup of problematic file
                        backup_path = f"{output_path}.backup.{datetime.now().strftime('%Y%m%d%H%M%S')}"
                        shutil.copy2(output_path, backup_path)
                        logger.warning(f"Created backup of problematic CSV at {backup_path}")
                        
                        # Try with more tolerant CSV reading options
                        try:
                            existing_df = pd.read_csv(
                                output_path,
                                error_bad_lines=False,  # Skip lines with incorrect field counts
                                warn_bad_lines=True,    # Show warnings
                                on_bad_lines='skip',    # Skip unparseable lines
                                engine='python',        # More flexible parsing engine
                                encoding='utf-8'        # Explicitly specify encoding
                            )
                            logger.info(f"Recovered {len(existing_df)} entries with tolerant parsing")
                        except Exception as e:
                            logger.warning(f"Could not parse CSV even with tolerant options: {str(e)}")
                            
                            # As last resort, try manual CSV parsing with exact column count matching
                            try:
                                # Read raw lines from file
                                with open(output_path, 'r', encoding='utf-8') as f:
                                    csv_lines = f.readlines()
                                    
                                if csv_lines:
                                    # Extract headers
                                    import csv
                                    reader = csv.reader([csv_lines[0]])
                                    headers = next(reader)
                                    
                                    # Create a cleaned CSV file with consistent column counts
                                    cleaned_path = f"{output_path}.cleaned.csv"
                                    with open(cleaned_path, 'w', encoding='utf-8') as f:
                                        # Write header row
                                        f.write(csv_lines[0])
                                        
                                        # Process and validate each data row
                                        for i, line in enumerate(csv_lines[1:], 1):
                                            # Basic validation - count fields by commas
                                            # This is a simplified approach just to get key data
                                            fields = line.count(',')
                                            expected_fields = len(headers) - 1
                                            
                                            if fields == expected_fields:
                                                f.write(line)
                                            else:
                                                logger.debug(f"Skipping line {i+1} with {fields+1} fields")
                                    
                                    # Try to read the cleaned file
                                    existing_df = pd.read_csv(cleaned_path)
                                    logger.info(f"Recovered {len(existing_df)} entries with manual CSV cleaning")
                                    original_csv_path = cleaned_path
                            except Exception as manual_error:
                                logger.warning(f"Manual CSV recovery failed: {str(manual_error)}")
        except Exception as e:
            logger.warning(f"Could not read existing configuration report: {str(e)}")
        
        # Attempt to recover any intermediate states from the raw incremental CSV
        try:
            # Read raw incremental states if different from the output path
            incremental_csv_path = self.output_dir / 'configuration_results.csv'
            if incremental_csv_path != Path(output_path) and \
               os.path.exists(incremental_csv_path) and os.path.getsize(incremental_csv_path) > 0:
                try:
                    # Try to read with flexible options
                    incremental_df = pd.read_csv(
                        incremental_csv_path,
                        error_bad_lines=False,
                        warn_bad_lines=True,
                        on_bad_lines='skip',
                        engine='python'
                    )
                    
                    # Filter for COMPLETION states only to avoid duplicates
                    if 'execution_stage' in incremental_df.columns:
                        completion_states = incremental_df[incremental_df['execution_stage'] == 'COMPLETION']
                        if not completion_states.empty:
                            logger.info(f"Recovered {len(completion_states)} completion states from incremental CSV")
                            
                            # If we have existing data, combine it
                            if existing_df is not None and not existing_df.empty:
                                # Ensure column compatibility
                                common_cols = list(set(existing_df.columns) & set(completion_states.columns))
                                if common_cols and 'configuration_name' in common_cols:
                                    # Use only common columns from both DataFrames
                                    existing_df_common = existing_df[common_cols]
                                    completion_states_common = completion_states[common_cols]
                                    
                                    # Combine with existing data
                                    existing_df = pd.concat([existing_df_common, completion_states_common])
                                    logger.info(f"Combined existing data with recovered completion states: {len(existing_df)} entries")
                            else:
                                # Just use the completion states as our existing data
                                existing_df = completion_states
                                logger.info(f"Using recovered completion states as existing data: {len(existing_df)} entries")
                except Exception as inc_error:
                    logger.warning(f"Could not recover states from incremental CSV: {str(inc_error)}")
        except Exception as recovery_error:
            logger.warning(f"Error during state recovery process: {str(recovery_error)}")
        
        # Initialize final DataFrame
        if existing_df is not None and not existing_df.empty:
            try:
                # Merge existing and new data, removing duplicates based on configuration_name
                # First ensure we have all needed columns in both DataFrames
                if len(new_df) > 0:
                    # Get the union of all columns
                    all_columns = list(set(existing_df.columns) | set(new_df.columns))
                    
                    # Add missing columns to each DataFrame with NaN values
                    for col in all_columns:
                        if col not in existing_df.columns:
                            existing_df[col] = None
                        if col not in new_df.columns:
                            new_df[col] = None
                    
                    # Now concat and deduplicate
                    combined_df = pd.concat([existing_df, new_df])
                    final_df = combined_df.drop_duplicates(subset=['configuration_name'], keep='last')
                    logger.info(f"Combined {len(existing_df)} existing and {len(new_df)} new entries, "
                               f"resulting in {len(final_df)} total configurations")
                else:
                    # If no new results, just use existing data
                    final_df = existing_df
                    logger.info(f"Using existing data with {len(final_df)} configurations")
            except Exception as merge_error:
                logger.warning(f"Error merging existing and new data: {str(merge_error)}")
                # Fallback to just using new data
                final_df = new_df
                logger.info(f"Fallback to new data only: {len(final_df)} entries")
        else:
            final_df = new_df
            logger.info(f"Created new configuration report with {len(final_df)} entries")
        
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Additional step: Scan log files for training metrics that aren't in our dataset
        additional_results = self._extract_missing_results_from_logs()
        if additional_results:
            logger.info(f"Extracted {len(additional_results)} additional configurations from log files")
            # Add to new_df
            additional_df = pd.DataFrame(additional_results)
            if not final_df.empty:
                final_df = pd.concat([final_df, additional_df])
            else:
                final_df = additional_df
            
            # Remove duplicates again after adding additional results
            final_df = final_df.drop_duplicates(subset=['configuration_name'], keep='last')
            logger.info(f"Final configuration count after adding log-extracted data: {len(final_df)}")
        
        # Save to CSV with file locking for thread safety
        with FileLock(str(output_path) + ".lock"):
            final_df.to_csv(output_path, index=False)
            logger.info(f"Updated configuration report saved to {output_path} with {len(final_df)} entries")
        
        return output_path
    
    def analyze_parameter_correlations(self, output_path=None):
        """
        Analyze correlations between parameter settings and performance metrics.
        
        Args:
            output_path: Path to save the correlation matrix visualization
            
        Returns:
            Pandas DataFrame with correlation analysis
        """
        if output_path is None:
            output_path = self.viz_dir / 'parameter_correlations.png'
            
        correlation_data = []
        
        # Extract parameter values and corresponding metrics
        for result in self.results:
            # CRITICAL FIX: Comprehensive validation to ensure only clean, valid data
            # Skip any problematic configurations
            if ('error' in result or 
                result.get('metrics', {}).get('error_state', False) or 
                result.get('metrics', {}).get('decision_threshold', 0) >= 0.99 or
                result.get('configuration_name', '').startswith('Validation_Result_') or
                not result.get('configuration', {}).get('parameters', {})):
                continue
                
            config = result.get('configuration', {})
            metrics = result.get('metrics', {})
            
            # Skip if no metrics available or if metrics have error indicators
            if not metrics or metrics.get('precision', 0) <= 0 or metrics.get('recall', 0) <= 0:
                continue
                
            # Extract key parameters
            entry = {}
            
            # Add metrics - handle None values explicitly
            entry['precision'] = float(metrics.get('precision', 0) or 0)
            entry['recall'] = float(metrics.get('recall', 0) or 0)
            entry['f1'] = float(metrics.get('f1', 0) or 0)
            entry['specificity'] = float(metrics.get('specificity', 0) or 0)
            entry['accuracy'] = float(metrics.get('accuracy', 0) or 0)
            entry['decision_threshold'] = float(metrics.get('decision_threshold', 0) or 0)
            
            # Add configuration parameters - with explicit numeric validation
            for param_name, param_value in config.get('parameters', {}).items():
                if isinstance(param_value, dict):
                    for sub_param, sub_value in param_value.items():
                        # Ensure numeric values only
                        if isinstance(sub_value, (int, float)):
                            entry[f"{param_name}_{sub_param}"] = float(sub_value)
                elif isinstance(param_value, (int, float)):
                    entry[param_name] = float(param_value)
            
            # Add binary flags for enabled features
            enabled_features = config.get('enabled', [])
            for feature in ['person_low_jaro_winkler_indicator', 
                           'person_low_levenshtein_indicator',
                           'person_jaro_winkler_similarity',
                           'person_levenshtein_similarity']:
                entry[f'{feature}_enabled'] = 1 if feature in enabled_features else 0
                
            # Add flags for approach
            entry['use_binary_indicators'] = 1 if config.get('similarity_metrics', {}).get('use_binary_indicators', False) else 0
            entry['include_both_metrics'] = 1 if config.get('similarity_metrics', {}).get('include_both_metrics', False) else 0
            
            correlation_data.append(entry)
        
        # CRITICAL FIX: Early return with appropriate logging if no valid data
        if not correlation_data:
            logger.warning("No valid configuration data available for correlation analysis")
            # Create empty correlation matrix to avoid crashes
            empty_df = pd.DataFrame()
            return empty_df
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(correlation_data)
        
        # Check if we have numeric data to compute correlations
        if df.empty or df.select_dtypes(include=[np.number]).empty:
            logger.warning("No numeric data available for correlation analysis")
            return df
        
        # Compute correlation matrix only for numeric columns with explicit error handling
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            correlation_matrix = numeric_df.corr()
            
            # Additional validation - if any columns have all zeros, drop them
            zero_columns = numeric_df.columns[numeric_df.sum() == 0]
            if not zero_columns.empty:
                logger.warning(f"Dropping {len(zero_columns)} columns with all zeros from correlation analysis")
                numeric_df = numeric_df.drop(columns=zero_columns)
                correlation_matrix = numeric_df.corr() if not numeric_df.empty else pd.DataFrame()
                
            logger.info(f"Successfully computed correlation matrix with {len(correlation_matrix.columns)} parameters")
        except Exception as e:
            logger.error(f"Error computing correlation matrix: {str(e)}")
            # Return empty DataFrame to avoid crashes
            correlation_matrix = pd.DataFrame()
        
        # Visualize correlations
        plt.figure(figsize=(14, 12))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        heatmap = sns.heatmap(correlation_matrix, mask=mask, annot=True, 
                             cmap='coolwarm', fmt=".2f", linewidths=0.5,
                             vmin=-1, vmax=1)
        plt.title('Parameter-Metric Correlations', fontsize=16)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save correlation matrix to CSV for further analysis
        csv_path = self.output_dir / 'correlation_matrix.csv'
        correlation_matrix.to_csv(csv_path)
        
        logger.info(f"Parameter correlation analysis saved to {output_path}")
        logger.info(f"Correlation matrix data saved to {csv_path}")
        
        return correlation_matrix
    
    def generate_parameter_influence_chart(self, param_name, metric='recall', output_path=None):
        """
        Generate chart showing the influence of a specific parameter on performance metrics.
        
        Args:
            param_name: Name of the parameter to analyze (e.g., 'person_low_jaro_winkler_indicator_threshold')
            metric: Metric to evaluate (default: 'recall')
            output_path: Path to save the chart
            
        Returns:
            Path to the saved chart
        """
        if output_path is None:
            output_path = self.viz_dir / f'influence_{param_name}_{metric}.png'
            
        # Extract parameter values and metrics
        param_data = []
        
        for result in self.results:
            # CRITICAL FIX: Apply comprehensive filtering criteria for clean data
            # Skip any problematic configurations
            if ('error' in result or 
                result.get('metrics', {}).get('error_state', False) or 
                result.get('metrics', {}).get('decision_threshold', 0) >= 0.99 or
                result.get('configuration_name', '').startswith('Validation_Result_') or
                not result.get('configuration', {}).get('parameters', {})):
                continue
                
            config = result.get('configuration', {})
            metrics = result.get('metrics', {})
            
            # Skip if no metrics available or if metric values are suspicious
            if not metrics or metrics.get(metric, 0) <= 0:
                continue
            
            # Get parameter value - handle both direct and nested parameters
            param_value = None
            
            # Check if it's a composite parameter name with parts
            if '_' in param_name:
                parts = param_name.split('_')
                if len(parts) >= 3:  # Like 'person_low_jaro_winkler_indicator_threshold'
                    feature_name = '_'.join(parts[:-1])
                    property_name = parts[-1]
                    
                    # Enhanced parameter extraction with type checking
                    if feature_name in config.get('parameters', {}):
                        if isinstance(config['parameters'][feature_name], dict):
                            param_value = config['parameters'][feature_name].get(property_name)
                        elif property_name == 'threshold' or property_name == 'weight':
                            # Handle case where parameter might be a direct value
                            param_value = config['parameters'].get(feature_name)
            
            # If not found as composite, try direct lookup
            if param_value is None and param_name in config.get('parameters', {}):
                param_value = config['parameters'][param_name]
                
            # Skip if parameter not found or not numeric
            if param_value is None or not isinstance(param_value, (int, float)):
                continue
                
            # Create entry with parameter value and metric - with explicit numeric conversion
            try:
                entry = {
                    'parameter_value': float(param_value),
                    'metric_value': float(metrics.get(metric, 0)),
                    'configuration': result.get('configuration_name', 'unknown'),
                    'perfect_precision': bool(metrics.get('perfect_precision', False)),
                    'specificity': float(metrics.get('specificity', 0)),
                    'accuracy': float(metrics.get('accuracy', 0))
                }
                param_data.append(entry)
            except (ValueError, TypeError) as e:
                # Skip entries that can't be properly converted
                logger.debug(f"Skipping entry due to conversion error: {e}")
                continue
        
        # CRITICAL FIX: Ensure we have enough data points for meaningful analysis
        if not param_data or len(param_data) < 3:
            logger.warning(f"Insufficient data available for parameter influence chart: {param_name} (found {len(param_data)} points)")
            return None
            
        df = pd.DataFrame(param_data)
        
        # Verify again that we have numeric values
        try:
            if not pd.api.types.is_numeric_dtype(df['parameter_value']) or not pd.api.types.is_numeric_dtype(df['metric_value']):
                logger.warning(f"Parameter {param_name} or metric {metric} is not numeric, cannot generate influence chart")
                return None
                
            # Check for all identical values
            if df['parameter_value'].nunique() <= 1:
                logger.warning(f"Parameter {param_name} has only one unique value, cannot generate meaningful influence chart")
                return None
        except Exception as e:
            logger.error(f"Error validating data for parameter influence chart: {str(e)}")
            return None
        
        # Create chart
        plt.figure(figsize=(12, 8))
        
        # Plot all configurations
        ax = plt.scatter(df['parameter_value'], df['metric_value'], alpha=0.7, label='All configurations')
        
        # Highlight perfect precision configurations
        perfect_mask = df['perfect_precision']
        if perfect_mask.any():
            plt.scatter(df[perfect_mask]['parameter_value'], df[perfect_mask]['metric_value'], 
                       edgecolor='green', s=100, facecolors='none', linewidth=2, 
                       label='Perfect Precision')
        
        # Add trend line
        try:
            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                df['parameter_value'], df['metric_value'])
            x_line = np.linspace(df['parameter_value'].min(), df['parameter_value'].max(), 100)
            y_line = slope * x_line + intercept
            plt.plot(x_line, y_line, 'r--', 
                    label=f'Trend (r²={r_value**2:.2f}, p={p_value:.3f})')
        except Exception as e:
            logger.warning(f"Could not generate trend line: {str(e)}")
        
        # Label chart
        plt.title(f'Influence of {param_name} on {metric.capitalize()}')
        plt.xlabel(f'{param_name}')
        plt.ylabel(f'{metric.capitalize()}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add text annotations for top configurations
        if len(df) > 0:
            # Sort by metric value (descending) and highlight top 3
            top_configs = df.sort_values('metric_value', ascending=False).head(3)
            for _, row in top_configs.iterrows():
                plt.annotate(row['configuration'], 
                            (row['parameter_value'], row['metric_value']),
                            xytext=(5, 5), textcoords='offset points')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        logger.info(f"Parameter influence chart saved to {output_path}")
        
        return output_path
    
    def generate_interactive_dashboard(self, output_path=None):
        """
        Generate an interactive HTML dashboard for exploring configuration results.
        This version embeds the data directly to avoid cross-origin restrictions.
        
        Args:
            output_path: Path to save the HTML dashboard
            
        Returns:
            Path to the generated dashboard
        """
        if output_path is None:
            output_path = self.output_dir / 'configuration_dashboard.html'
            
        # First, ensure we have the CSV report to reference
        csv_report_path = self.output_dir / 'configuration_results.csv'
        if not csv_report_path.exists():
            self.generate_configuration_report(csv_report_path)
        
        # Read the CSV data into a DataFrame
        try:
            df = pd.read_csv(csv_report_path)
            # Convert DataFrame to JSON string
            json_data = df.to_json(orient='records')
            logger.info(f"Embedded {len(df)} configuration records into dashboard")
        except Exception as e:
            logger.error(f"Error reading CSV data for dashboard: {str(e)}")
            json_data = "[]"
            
        # Keep the relative path for fallback mode
        csv_rel_path = os.path.relpath(csv_report_path, output_path.parent)
        
        # CRITICAL FIX: Create a filtered set of legitimate configurations only
        # This prevents invalid entries from corrupting the dashboard
        filtered_df = df.copy()
        
        # Apply strict filtering to remove error cases and invalid configurations
        filtered_df = filtered_df[
            ~filtered_df['configuration_name'].str.startswith('Validation_Result_') & 
            (filtered_df['decision_threshold'] < 0.99) & 
            (filtered_df['precision'] > 0) & 
            (filtered_df['recall'] > 0)
        ]
        
        # Handle the DataFrame is empty after filtering
        if len(filtered_df) == 0:
            logger.warning("No valid configurations found after filtering!")
            if len(df) > 0:
                # Use at most 10 valid configurations
                filtered_df = df.nlargest(10, 'recall')
        
        # Convert filtered DataFrame to JSON string
        json_data = filtered_df.to_json(orient='records')
        logger.info(f"Embedded {len(filtered_df)} configuration records into dashboard (filtered from {len(df)})")
            
        # Create the HTML dashboard
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Feature Configuration Analysis Dashboard</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/papaparse@5.3.0/papaparse.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/jquery@3.6.0/dist/jquery.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/datatables@1.10.18/media/js/jquery.dataTables.min.js"></script>
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css">
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/datatables@1.10.18/media/css/jquery.dataTables.min.css">
            <style>
                .chart-container {{
                    width: 100%;
                    height: 400px;
                    margin-bottom: 30px;
                }}
                .card {{
                    margin-bottom: 20px;
                }}
                .table-container {{
                    margin-top: 30px;
                    margin-bottom: 50px;
                }}
                .filters {{
                    margin-bottom: 20px;
                    padding: 15px;
                    background-color: #f8f9fa;
                    border-radius: 5px;
                }}
                .header {{
                    background-color: #343a40;
                    color: white;
                    padding: 20px 0;
                    margin-bottom: 30px;
                }}
                .metric-card {{
                    text-align: center;
                    padding: 15px;
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                }}
                .metric-label {{
                    font-size: 14px;
                    color: #6c757d;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <div class="container">
                    <h1>Feature Configuration Analysis Dashboard</h1>
                    <p>Analysis of feature configurations for entity resolution</p>
                </div>
            </div>
            
            <div class="container">
                <!-- Summary Metrics -->
                <div class="row mb-4" id="summaryMetrics">
                    <!-- Will be populated by JavaScript -->
                </div>
                
                <!-- Filters -->
                <div class="row filters">
                    <div class="col-md-2">
                        <label for="perfectPrecisionFilter">Precision Requirements:</label>
                        <select id="perfectPrecisionFilter" class="form-select">
                            <option value="all">All Configurations</option>
                            <option value="perfect" selected>Only Perfect Precision</option>
                            <option value="imperfect">Imperfect Precision</option>
                        </select>
                    </div>
                    <div class="col-md-2">
                        <label for="precisionWeightFilter">Precision Weight:</label>
                        <select id="precisionWeightFilter" class="form-select">
                            <option value="all" selected>All Weights</option>
                            <option value="0.3">Low (0.3)</option>
                            <option value="0.5">Balanced (0.5)</option>
                            <option value="0.7">High (0.7)</option>
                            <option value="0.9">Very High (0.9)</option>
                        </select>
                    </div>
                    <div class="col-md-3">
                        <label for="approachFilter">Similarity Approach:</label>
                        <select id="approachFilter" class="form-select">
                            <option value="all" selected>All Approaches</option>
                            <option value="binary">Binary Indicators</option>
                            <option value="direct">Direct Similarity</option>
                            <option value="hybrid">Hybrid Approach</option>
                        </select>
                    </div>
                    <div class="col-md-2">
                        <label for="metricSort">Sort By:</label>
                        <select id="metricSort" class="form-select">
                            <option value="recall" selected>Recall</option>
                            <option value="precision">Precision</option>
                            <option value="f1">F1 Score</option>
                            <option value="balanced_score">Balanced Score</option>
                            <option value="specificity">Specificity</option>
                            <option value="accuracy">Accuracy</option>
                        </select>
                    </div>
                    <div class="col-md-3">
                        <label for="configCount">Top Configurations:</label>
                        <select id="configCount" class="form-select">
                            <option value="10" selected>Top 10</option>
                            <option value="20">Top 20</option>
                            <option value="50">Top 50</option>
                            <option value="all">All</option>
                        </select>
                    </div>
                </div>
                
                <!-- Charts Section -->
                <div class="row">
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header">
                                <h5>Performance by Configuration</h5>
                            </div>
                            <div class="card-body">
                                <div class="chart-container">
                                    <canvas id="performanceChart"></canvas>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header">
                                <h5>Precision vs Recall</h5>
                            </div>
                            <div class="card-body">
                                <div class="chart-container">
                                    <canvas id="scatterChart"></canvas>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="row">
                    <div class="col-md-12">
                        <div class="card">
                            <div class="card-header">
                                <h5>Approach Comparison</h5>
                            </div>
                            <div class="card-body">
                                <div class="chart-container">
                                    <canvas id="approachChart"></canvas>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Data Table Section -->
                <div class="table-container">
                    <h3>Configuration Details</h3>
                    <table id="configurationsTable" class="table table-striped table-bordered" width="100%">
                        <!-- Table will be populated by JavaScript -->
                    </table>
                </div>
            </div>
            
            <script>
                // Dashboard initialization
                document.addEventListener('DOMContentLoaded', function() {{
                    try {{
                        // Check if data is loaded from embedded source or needs to be loaded from CSV
                        loadConfigurationData();
                    }} catch (e) {{
                        console.error('Dashboard initialization error:', e);
                        showError(`Dashboard initialization error: ${{e.message}}`);
                    }}
                }});

                // Function to load data from CSV
                function loadConfigurationData() {{
                    const csvPath = 'configuration_results.csv';
                    console.log('Attempting to load from CSV file:', csvPath);
                    
                    Papa.parse(csvPath, {{
                        download: true,
                        header: true,
                        dynamicTyping: true,
                        complete: function(results) {{
                            const data = results.data;
                            
                            // Filter out rows with null configuration_name (indicates errors)
                            const validData = data.filter(row => row.configuration_name);
                            
                            if (validData && validData.length > 0) {{
                                console.log(`Loaded ${{validData.length}} configuration records from CSV`);
                                
                                // Ensure balanced_score is calculated for all configurations
                                validData.forEach(row => {{
                                    if (!row.balanced_score && row.precision > 0 && row.recall > 0) {{
                                        const precision = parseFloat(row.precision);
                                        const recall = parseFloat(row.recall);
                                        const precisionWeight = parseFloat(row.precision_weight || 0.5);
                                        
                                        // Calculate balanced score
                                        if (precision > 0 && recall > 0 && 0 <= precisionWeight && precisionWeight <= 1) {{
                                            const weightRatio = precisionWeight < 1.0 ? 
                                                precisionWeight / (1.0 - precisionWeight) : 1.0;
                                            
                                            const denominator = weightRatio * precision + recall;
                                            if (denominator > 0) {{
                                                row.balanced_score = ((1 + weightRatio) * precision * recall) / denominator;
                                            }} else {{
                                                row.balanced_score = 0.0;
                                            }}
                                        }}
                                    }}
                                    
                                    // Ensure precision_weight has a value
                                    if (!row.precision_weight) {{
                                        row.precision_weight = 0.5;
                                    }}
                                }});
                                
                                initializeDashboard(validData);
                            }} else {{
                                showError('No valid configuration data found in CSV file.');
                            }}
                        }},
                        error: function(error) {{
                            console.error('Error loading CSV:', error);
                            showError(`Error loading CSV: ${{error.message}}. Try serving this file from an HTTP server.`);
                        }}
                    }});
                }}
                
                // Function to display error messages
                function showError(message) {{
                    const errorDiv = document.createElement('div');
                    errorDiv.className = 'alert alert-danger';
                    errorDiv.innerHTML = `
                        <h4>Error loading configuration data</h4>
                        <p>${{message}}</p>
                        <hr>
                        <h5>Troubleshooting:</h5>
                        <ol>
                            <li>Try opening this dashboard using a local web server:</li>
                            <pre>python -m http.server 8000</pre>
                            <li>Then navigate to: <code>http://localhost:8000/configuration_dashboard.html</code></li>
                            <li>Check that the configuration_results.csv file exists in the expected location</li>
                        </ol>
                    `;
                    document.body.prepend(errorDiv);
                }}
                
                function initializeDashboard(data) {{
                    // Update summary metrics
                    updateSummaryMetrics(data);
                    
                    // Initialize charts
                    createPerformanceChart(data);
                    createScatterChart(data);
                    createApproachComparisonChart(data);
                    
                    // Initialize data table
                    initializeDataTable(data);
                    
                    // Set up event listeners for filters
                    setupFilterListeners(data);
                }}
                
                function updateSummaryMetrics(data) {{
                    // Calculate summary metrics
                    const perfectPrecisionCount = data.filter(d => d.perfect_precision).length;
                    const highRecallCount = data.filter(d => d.recall >= 0.9).length;
                    const averagePrecision = data.reduce((sum, d) => sum + d.precision, 0) / data.length;
                    const averageRecall = data.reduce((sum, d) => sum + d.recall, 0) / data.length;
                    const averageBalancedScore = data.reduce((sum, d) => sum + (d.balanced_score || 0), 0) / data.length;
                    const averageSpecificity = data.reduce((sum, d) => sum + (d.specificity || 0), 0) / data.length;
                    const averageAccuracy = data.reduce((sum, d) => sum + (d.accuracy || 0), 0) / data.length;
                    
                    // Find best configuration (highest recall with perfect precision)
                    let bestConfig = null;
                    let bestRecall = 0;
                    
                    data.filter(d => d.perfect_precision).forEach(d => {{
                        if (d.recall > bestRecall) {{
                            bestRecall = d.recall;
                            bestConfig = d;
                        }}
                    }});
                    
                    // Create HTML for summary metrics
                    const summaryHTML = `
                        <div class="col-md-2">
                            <div class="card metric-card">
                                <div class="metric-value">${{data.length}}</div>
                                <div class="metric-label">Total Configurations</div>
                            </div>
                        </div>
                        <div class="col-md-2">
                            <div class="card metric-card">
                                <div class="metric-value">${{perfectPrecisionCount}}</div>
                                <div class="metric-label">Perfect Precision</div>
                            </div>
                        </div>
                        <div class="col-md-2">
                            <div class="card metric-card">
                                <div class="metric-value">${{highRecallCount}}</div>
                                <div class="metric-label">High Recall (≥0.9)</div>
                            </div>
                        </div>
                        <div class="col-md-2">
                            <div class="card metric-card">
                                <div class="metric-value">${{averagePrecision.toFixed(3)}}</div>
                                <div class="metric-label">Avg Precision</div>
                            </div>
                        </div>
                        <div class="col-md-2">
                            <div class="card metric-card">
                                <div class="metric-value">${{averageRecall.toFixed(3)}}</div>
                                <div class="metric-label">Avg Recall</div>
                            </div>
                        </div>
                        <div class="col-md-2">
                            <div class="card metric-card" style="background-color: #e8f5e9;">
                                <div class="metric-value">${{bestConfig ? bestRecall.toFixed(3) : 'N/A'}}</div>
                                <div class="metric-label">Best Recall</div>
                            </div>
                        </div>
                    `;
                    
                    $('#summaryMetrics').html(summaryHTML);
                }}
                
                function createPerformanceChart(data) {{
                    // Apply filters
                    const perfectOnly = $('#perfectPrecisionFilter').val() === 'perfect';
                    const approachFilter = $('#approachFilter').val();
                    const precisionWeightFilter = $('#precisionWeightFilter').val();
                    const sortMetric = $('#metricSort').val();
                    const configCount = $('#configCount').val();
                    
                    // Filter data
                    let filteredData = data;
                    if (perfectOnly) {{
                        filteredData = filteredData.filter(d => d.perfect_precision);
                    }} else if ($('#perfectPrecisionFilter').val() === 'imperfect') {{
                        filteredData = filteredData.filter(d => !d.perfect_precision);
                    }}
                    
                    if (approachFilter !== 'all') {{
                        filteredData = filteredData.filter(d => d.similarity_approach === approachFilter);
                    }}
                    
                    if (precisionWeightFilter !== 'all') {{
                        const targetWeight = parseFloat(precisionWeightFilter);
                        filteredData = filteredData.filter(d => {{
                            // Allow small tolerance for floating point comparison
                            const weight = parseFloat(d.precision_weight || 0.5);
                            return Math.abs(weight - targetWeight) < 0.01;
                        }});
                    }}
                    
                    // Sort by selected metric
                    filteredData.sort((a, b) => b[sortMetric] - a[sortMetric]);
                    
                    // Limit to selected count
                    if (configCount !== 'all') {{
                        filteredData = filteredData.slice(0, parseInt(configCount));
                    }}
                    
                    // Prepare chart data
                    const labels = filteredData.map(d => d.configuration_name);
                    const precisionData = filteredData.map(d => d.precision);
                    const recallData = filteredData.map(d => d.recall);
                    const f1Data = filteredData.map(d => d.f1);
                    const balancedScoreData = filteredData.map(d => d.balanced_score || 0);
                    const specificityData = filteredData.map(d => d.specificity || 0);
                    const accuracyData = filteredData.map(d => d.accuracy || 0);
                    
                    // Create chart
                    const ctx = document.getElementById('performanceChart').getContext('2d');
                    
                    // Destroy existing chart if it exists
                    if (window.performanceChart) {{
                        window.performanceChart.destroy();
                    }}
                    
                    window.performanceChart = new Chart(ctx, {{
                        type: 'bar',
                        data: {{
                            labels: labels,
                            datasets: [
                                {{
                                    label: 'Precision',
                                    data: precisionData,
                                    backgroundColor: 'rgba(54, 162, 235, 0.5)',
                                    borderColor: 'rgba(54, 162, 235, 1)',
                                    borderWidth: 1
                                }},
                                {{
                                    label: 'Recall',
                                    data: recallData,
                                    backgroundColor: 'rgba(75, 192, 192, 0.5)',
                                    borderColor: 'rgba(75, 192, 192, 1)',
                                    borderWidth: 1
                                }},
                                {{
                                    label: 'F1 Score',
                                    data: f1Data,
                                    backgroundColor: 'rgba(255, 159, 64, 0.5)',
                                    borderColor: 'rgba(255, 159, 64, 1)',
                                    borderWidth: 1
                                }},
                                {{
                                    label: 'Balanced Score',
                                    data: balancedScoreData,
                                    backgroundColor: 'rgba(255, 99, 132, 0.5)',
                                    borderColor: 'rgba(255, 99, 132, 1)',
                                    borderWidth: 1
                                }},
                                {{
                                    label: 'Specificity',
                                    data: specificityData,
                                    backgroundColor: 'rgba(153, 102, 255, 0.5)',
                                    borderColor: 'rgba(153, 102, 255, 1)',
                                    borderWidth: 1
                                }},
                                {{
                                    label: 'Accuracy',
                                    data: accuracyData,
                                    backgroundColor: 'rgba(201, 203, 207, 0.5)',
                                    borderColor: 'rgba(201, 203, 207, 1)',
                                    borderWidth: 1
                                }}
                            ]
                        }},
                        options: {{
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {{
                                x: {{
                                    ticks: {{
                                        maxRotation: 45,
                                        minRotation: 45
                                    }}
                                }},
                                y: {{
                                    beginAtZero: true,
                                    max: 1.0
                                }}
                            }},
                            plugins: {{
                                title: {{
                                    display: true,
                                    text: `Performance Metrics by Configuration (Top ${{configCount !== 'all' ? configCount : labels.length}})`
                                }}
                            }}
                        }}
                    }});
                }}
                
                function createScatterChart(data) {{
                    // Apply filters
                    const perfectOnly = $('#perfectPrecisionFilter').val() === 'perfect';
                    const approachFilter = $('#approachFilter').val();
                    const precisionWeightFilter = $('#precisionWeightFilter').val();
                    
                    // Filter data
                    let filteredData = data;
                    if (perfectOnly) {{
                        filteredData = filteredData.filter(d => d.perfect_precision);
                    }} else if ($('#perfectPrecisionFilter').val() === 'imperfect') {{
                        filteredData = filteredData.filter(d => !d.perfect_precision);
                    }}
                    
                    if (approachFilter !== 'all') {{
                        filteredData = filteredData.filter(d => d.similarity_approach === approachFilter);
                    }}
                    
                    if (precisionWeightFilter !== 'all') {{
                        const targetWeight = parseFloat(precisionWeightFilter);
                        filteredData = filteredData.filter(d => {{
                            // Allow small tolerance for floating point comparison
                            const weight = parseFloat(d.precision_weight || 0.5);
                            return Math.abs(weight - targetWeight) < 0.01;
                        }});
                    }}
                    
                    // Group by approach
                    const binaryData = filteredData.filter(d => d.similarity_approach === 'binary');
                    const directData = filteredData.filter(d => d.similarity_approach === 'direct');
                    const hybridData = filteredData.filter(d => d.similarity_approach === 'hybrid');
                    
                    // Create chart
                    const ctx = document.getElementById('scatterChart').getContext('2d');
                    
                    // Destroy existing chart if it exists
                    if (window.scatterChart) {{
                        window.scatterChart.destroy();
                    }}
                    
                    window.scatterChart = new Chart(ctx, {{
                        type: 'scatter',
                        data: {{
                            datasets: [
                                {{
                                    label: 'Binary Indicators',
                                    data: binaryData.map(d => ({{ x: d.recall, y: d.precision }})),
                                    backgroundColor: 'rgba(255, 99, 132, 0.7)',
                                    pointRadius: 5,
                                    pointHoverRadius: 7
                                }},
                                {{
                                    label: 'Direct Similarity',
                                    data: directData.map(d => ({{ x: d.recall, y: d.precision }})),
                                    backgroundColor: 'rgba(54, 162, 235, 0.7)',
                                    pointRadius: 5,
                                    pointHoverRadius: 7
                                }},
                                {{
                                    label: 'Hybrid Approach',
                                    data: hybridData.map(d => ({{ x: d.recall, y: d.precision }})),
                                    backgroundColor: 'rgba(75, 192, 192, 0.7)',
                                    pointRadius: 5,
                                    pointHoverRadius: 7
                                }}
                            ]
                        }},
                        options: {{
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {{
                                x: {{
                                    title: {{
                                        display: true,
                                        text: 'Recall'
                                    }},
                                    min: 0,
                                    max: 1.0
                                }},
                                y: {{
                                    title: {{
                                        display: true,
                                        text: 'Precision'
                                    }},
                                    min: 0,
                                    max: 1.0
                                }}
                            }},
                            plugins: {{
                                title: {{
                                    display: true,
                                    text: 'Precision vs Recall by Approach'
                                }},
                                tooltip: {{
                                    callbacks: {{
                                        label: function(context) {{
                                            const index = context.dataIndex;
                                            const dataset = context.dataset;
                                            const datasetIndex = context.datasetIndex;
                                            
                                            let originalData;
                                            if (datasetIndex === 0) originalData = binaryData[index];
                                            else if (datasetIndex === 1) originalData = directData[index];
                                            else originalData = hybridData[index];
                                            
                                            if (!originalData) return '';
                                            
                                            return [
                                                `Config: ${{originalData.configuration_name}}`,
                                                `Precision: ${{originalData.precision.toFixed(4)}}`,
                                                `Recall: ${{originalData.recall.toFixed(4)}}`,
                                                `F1: ${{originalData.f1.toFixed(4)}}`,
                                                `Balanced Score: ${{(originalData.balanced_score || 0).toFixed(4)}}`,
                                                `Precision Weight: ${{(originalData.precision_weight || 0.5).toFixed(2)}}`,
                                                `Specificity: ${{(originalData.specificity || 0).toFixed(4)}}`,
                                                `Accuracy: ${{(originalData.accuracy || 0).toFixed(4)}}`,
                                                `Perfect Precision: ${{originalData.perfect_precision ? 'Yes' : 'No'}}`
                                            ];
                                        }}
                                    }}
                                }}
                            }}
                        }}
                    }});
                }}
                
                function createApproachComparisonChart(data) {{
                    // Apply the precision weight filter if specified
                    const precisionWeightFilter = $('#precisionWeightFilter').val();
                    let filteredData = data;
                    
                    if (precisionWeightFilter !== 'all') {{
                        const targetWeight = parseFloat(precisionWeightFilter);
                        filteredData = filteredData.filter(d => {{
                            // Allow small tolerance for floating point comparison
                            const weight = parseFloat(d.precision_weight || 0.5);
                            return Math.abs(weight - targetWeight) < 0.01;
                        }});
                    }}
                    
                    // Group by approach
                    const approaches = {{'binary': [], 'direct': [], 'hybrid': []}};
                    
                    filteredData.forEach(d => {{
                        if (d.similarity_approach && approaches.hasOwnProperty(d.similarity_approach)) {{
                            approaches[d.similarity_approach].push(d);
                        }}
                    }});
                    
                    // Calculate average metrics for each approach
                    const avgMetrics = {{}};
                    Object.keys(approaches).forEach(approach => {{
                        const configs = approaches[approach];
                        if (configs.length > 0) {{
                            avgMetrics[approach] = {{
                                precision: configs.reduce((sum, d) => sum + d.precision, 0) / configs.length,
                                recall: configs.reduce((sum, d) => sum + d.recall, 0) / configs.length,
                                f1: configs.reduce((sum, d) => sum + d.f1, 0) / configs.length,
                                balanced_score: configs.reduce((sum, d) => sum + (d.balanced_score || 0), 0) / configs.length,
                                specificity: configs.reduce((sum, d) => sum + (d.specificity || 0), 0) / configs.length,
                                accuracy: configs.reduce((sum, d) => sum + (d.accuracy || 0), 0) / configs.length,
                                perfectPrecisionPct: configs.filter(d => d.perfect_precision).length / configs.length * 100
                            }};
                        }}
                    }});
                    
                    // Create chart
                    const ctx = document.getElementById('approachChart').getContext('2d');
                    
                    // Destroy existing chart if it exists
                    if (window.approachChart) {{
                        window.approachChart.destroy();
                    }}
                    
                    // Prepare data
                    const labels = Object.keys(avgMetrics).map(a => a.charAt(0).toUpperCase() + a.slice(1));
                    const precisionData = Object.values(avgMetrics).map(m => m.precision);
                    const recallData = Object.values(avgMetrics).map(m => m.recall);
                    const f1Data = Object.values(avgMetrics).map(m => m.f1);
                    const balancedScoreData = Object.values(avgMetrics).map(m => m.balanced_score);
                    const specificityData = Object.values(avgMetrics).map(m => m.specificity);
                    const accuracyData = Object.values(avgMetrics).map(m => m.accuracy);
                    const perfectPrecisionData = Object.values(avgMetrics).map(m => m.perfectPrecisionPct);
                    
                    window.approachChart = new Chart(ctx, {{
                        type: 'bar',
                        data: {{
                            labels: labels,
                            datasets: [
                                {{
                                    label: 'Avg Precision',
                                    data: precisionData,
                                    backgroundColor: 'rgba(54, 162, 235, 0.5)',
                                    borderColor: 'rgba(54, 162, 235, 1)',
                                    borderWidth: 1,
                                    yAxisID: 'y'
                                }},
                                {{
                                    label: 'Avg Recall',
                                    data: recallData,
                                    backgroundColor: 'rgba(75, 192, 192, 0.5)',
                                    borderColor: 'rgba(75, 192, 192, 1)',
                                    borderWidth: 1,
                                    yAxisID: 'y'
                                }},
                                {{
                                    label: 'Avg F1 Score',
                                    data: f1Data,
                                    backgroundColor: 'rgba(255, 159, 64, 0.5)',
                                    borderColor: 'rgba(255, 159, 64, 1)',
                                    borderWidth: 1,
                                    yAxisID: 'y'
                                }},
                                {{
                                    label: 'Avg Balanced Score',
                                    data: balancedScoreData,
                                    backgroundColor: 'rgba(255, 99, 132, 0.5)',
                                    borderColor: 'rgba(255, 99, 132, 1)',
                                    borderWidth: 1,
                                    yAxisID: 'y'
                                }},
                                {{
                                    label: 'Avg Specificity',
                                    data: specificityData,
                                    backgroundColor: 'rgba(153, 102, 255, 0.5)',
                                    borderColor: 'rgba(153, 102, 255, 1)',
                                    borderWidth: 1,
                                    yAxisID: 'y'
                                }},
                                {{
                                    label: 'Avg Accuracy',
                                    data: accuracyData,
                                    backgroundColor: 'rgba(201, 203, 207, 0.5)',
                                    borderColor: 'rgba(201, 203, 207, 1)',
                                    borderWidth: 1,
                                    yAxisID: 'y'
                                }},
                                {{
                                    label: '% Perfect Precision',
                                    data: perfectPrecisionData,
                                    backgroundColor: 'rgba(255, 205, 86, 0.5)',
                                    borderColor: 'rgba(255, 205, 86, 1)',
                                    borderWidth: 1,
                                    yAxisID: 'y1'
                                }}
                            ]
                        }},
                        options: {{
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {{
                                y: {{
                                    type: 'linear',
                                    position: 'left',
                                    beginAtZero: true,
                                    max: 1.0,
                                    title: {{
                                        display: true,
                                        text: 'Average Score'
                                    }}
                                }},
                                y1: {{
                                    type: 'linear',
                                    position: 'right',
                                    beginAtZero: true,
                                    max: 100,
                                    title: {{
                                        display: true,
                                        text: 'Percentage'
                                    }},
                                    grid: {{
                                        drawOnChartArea: false
                                    }}
                                }}
                            }},
                            plugins: {{
                                title: {{
                                    display: true,
                                    text: 'Performance by Similarity Approach'
                                }}
                            }}
                        }}
                    }});
                }}
                
                function initializeDataTable(data) {{
                    // Apply the initial filters
                    const perfectOnly = $('#perfectPrecisionFilter').val() === 'perfect';
                    const approachFilter = $('#approachFilter').val();
                    const precisionWeightFilter = $('#precisionWeightFilter').val();
                    
                    // Filter data
                    let filteredData = data;
                    if (perfectOnly) {{
                        filteredData = filteredData.filter(d => d.perfect_precision);
                    }} else if ($('#perfectPrecisionFilter').val() === 'imperfect') {{
                        filteredData = filteredData.filter(d => !d.perfect_precision);
                    }}
                    
                    if (approachFilter !== 'all') {{
                        filteredData = filteredData.filter(d => d.similarity_approach === approachFilter);
                    }}
                    
                    if (precisionWeightFilter !== 'all') {{
                        const targetWeight = parseFloat(precisionWeightFilter);
                        filteredData = filteredData.filter(d => {{
                            // Allow small tolerance for floating point comparison
                            const weight = parseFloat(d.precision_weight || 0.5);
                            return Math.abs(weight - targetWeight) < 0.01;
                        }});
                    }}
                    
                    // Format data for DataTable
                    const tableData = filteredData.map(row => {{
                        return {{
                            configuration_name: row.configuration_name,
                            precision: row.precision ? row.precision.toFixed(4) : 'N/A',
                            recall: row.recall ? row.recall.toFixed(4) : 'N/A',
                            f1: row.f1 ? row.f1.toFixed(4) : 'N/A',
                            balanced_score: row.balanced_score ? row.balanced_score.toFixed(4) : 'N/A',
                            precision_weight: row.precision_weight ? row.precision_weight.toFixed(2) : '0.50',
                            specificity: row.specificity ? row.specificity.toFixed(4) : 'N/A',
                            accuracy: row.accuracy ? row.accuracy.toFixed(4) : 'N/A',
                            similarity_approach: row.similarity_approach || 'unknown',
                            perfect_precision: row.perfect_precision ? 'Yes' : 'No',
                            decision_threshold: row.decision_threshold ? row.decision_threshold.toFixed(4) : 'N/A',
                            execution_time: row.execution_time || 'N/A'
                        }};
                    }});
                    
                    // Destroy existing DataTable if it exists
                    if ($.fn.DataTable.isDataTable('#configurationsTable')) {{
                        $('#configurationsTable').DataTable().destroy();
                    }}
                    
                    // Initialize DataTable
                    $('#configurationsTable').DataTable({{
                        data: tableData,
                        columns: [
                            {{ title: "Configuration", data: "configuration_name" }},
                            {{ title: "Precision", data: "precision" }},
                            {{ title: "Recall", data: "recall" }},
                            {{ title: "F1 Score", data: "f1" }},
                            {{ title: "Balanced Score", data: "balanced_score" }},
                            {{ title: "Precision Weight", data: "precision_weight" }},
                            {{ title: "Perfect Precision", data: "perfect_precision" }},
                            {{ title: "Approach", data: "similarity_approach" }},
                            {{ title: "Threshold", data: "decision_threshold" }},
                            {{ title: "Exec Time (s)", data: "execution_time" }}
                        ],
                        pageLength: 10,
                        lengthMenu: [10, 25, 50, 100],
                        order: [[3, 'desc']], // Default sort by F1 score
                        responsive: true,
                        dom: 'Bfrtip',
                        language: {{
                            search: "Filter Results:"
                        }}
                    }});
                }}
                
                function setupFilterListeners(data) {{
                    // Set up event listeners for filters
                    $('#perfectPrecisionFilter, #approachFilter, #precisionWeightFilter, #metricSort, #configCount').on('change', function() {{
                        // Update charts with filtered data
                        createPerformanceChart(data);
                        createScatterChart(data);
                        createApproachComparisonChart(data);
                        initializeDataTable(data);
                    }});
                }}
            </script>
        </body>
        </html>
        """
        
        # Save HTML dashboard
        with open(output_path, 'w') as f:
            f.write(html_content)
            
        logger.info(f"Interactive dashboard generated at {output_path}")
        
        return output_path
    
    def generate_parameter_importance_visualization(self, output_path=None):
        """
        Generate feature importance visualization based on correlation with performance metrics.
        
        Args:
            output_path: Path to save the visualization
            
        Returns:
            Path to the saved visualization
        """
        if output_path is None:
            output_path = self.viz_dir / 'parameter_importance.png'
            
        # First analyze parameter correlations to get correlation matrix
        correlation_matrix = self.analyze_parameter_correlations()
        
        if correlation_matrix is None:
            logger.warning("Could not generate parameter importance visualization - correlation analysis failed")
            return None
            
        # Extract correlations with performance metrics
        metrics = ['precision', 'recall', 'f1', 'specificity', 'accuracy']
        
        importance_data = []
        
        for metric in metrics:
            if metric in correlation_matrix.index:
                # Extract correlations with this metric
                correlations = correlation_matrix[metric].drop(metrics)
                
                # Filter to parameters only (skip other metrics)
                param_correlations = correlations[correlations.index.map(lambda x: not x in metrics)]
                
                # Take absolute values for importance
                param_importance = param_correlations.abs().sort_values(ascending=False)
                
                # Add to data
                for param, importance in param_importance.items():
                    importance_data.append({
                        'parameter': param,
                        'metric': metric,
                        'importance': importance,
                        'correlation': correlations[param]
                    })
        
        # Check if we have data to plot
        if not importance_data:
            logger.warning("No parameter importance data available for visualization")
            return None
            
        df = pd.DataFrame(importance_data)
        
        # Only keep top 15 parameters by average importance across metrics
        param_avg_importance = df.groupby('parameter')['importance'].mean().sort_values(ascending=False)
        top_params = param_avg_importance.head(15).index.tolist()
        
        df_filtered = df[df['parameter'].isin(top_params)]
        
        # Plot parameter importance
        plt.figure(figsize=(14, 10))
        
        # Create grouped bar chart
        sns.barplot(x='importance', y='parameter', hue='metric', data=df_filtered)
        
        plt.title('Parameter Importance for Performance Metrics', fontsize=16)
        plt.xlabel('Absolute Correlation (Importance)', fontsize=14)
        plt.ylabel('Parameter', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(title='Metric', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Parameter importance visualization saved to {output_path}")
        
        return output_path
    
    def generate_comprehensive_report(self):
        """
        Generate all reports and visualizations in a single comprehensive report.
        
        Returns:
            Dictionary with paths to all generated reports
        """
        logger.info("Generating comprehensive report...")
        
        report_paths = {}
        
        # Generate CSV report
        report_paths['csv_report'] = self.generate_configuration_report(self.csv_path)
        
        # Generate correlation analysis
        correlation_path = self.viz_dir / 'parameter_correlations.png'
        report_paths['correlation_analysis'] = self.analyze_parameter_correlations(correlation_path)
        
        # Generate parameter influence charts for key parameters
        influence_charts = {}
        
        # Identify common parameters used across configurations
        parameter_keys = set()
        
        for result in self.results:
            config = result.get('configuration', {})
            for param_name, param_value in config.get('parameters', {}).items():
                if isinstance(param_value, dict):
                    for sub_param in param_value:
                        parameter_keys.add(f"{param_name}_{sub_param}")
                else:
                    parameter_keys.add(param_name)
        
        # Generate charts for common parameters
        for param in parameter_keys:
            chart_path = self.generate_parameter_influence_chart(param, 'recall')
            if chart_path:
                influence_charts[param] = chart_path
        
        report_paths['influence_charts'] = influence_charts
        
        # Generate parameter importance visualization
        importance_path = self.viz_dir / 'parameter_importance.png'
        report_paths['parameter_importance'] = self.generate_parameter_importance_visualization(importance_path)
        
        # Generate interactive dashboard
        dashboard_path = self.output_dir / 'configuration_dashboard.html'
        report_paths['dashboard'] = self.generate_interactive_dashboard(dashboard_path)
        
        logger.info(f"Comprehensive report generated with {len(report_paths)} components")
        
        return report_paths


def generate_subject_enhancement_report(config: dict, hash_lookup: dict = None, string_dict: dict = None) -> dict:
    """
    Generate comprehensive JSON report for subject quality audit and imputation results.
    
    This function creates a detailed report that resolves opaque hashes to human-readable strings,
    providing insights into the subject enhancement process including quality improvements and
    missing subject imputations.
    
    Args:
        config: Configuration dictionary
        hash_lookup: Hash lookup dictionary mapping record IDs to field hashes
        string_dict: String dictionary mapping hashes to original string values
        
    Returns:
        Dictionary with report generation metrics and file paths
    """
    import time
    import glob
    from pathlib import Path
    
    logger.info("Generating subject enhancement report")
    start_time = time.time()
    
    # Set up directories
    output_dir = Path(config.get('output_dir', 'data/output'))
    reports_dir = output_dir / 'reports'
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Load required data if not provided
    if hash_lookup is None or string_dict is None:
        from src.preprocessing import load_hash_lookup, load_string_dict
        checkpoint_dir = config.get('checkpoint_dir', 'data/checkpoints')
        
        if hash_lookup is None:
            hash_lookup_path = Path(checkpoint_dir) / 'hash_lookup.pkl'
            if hash_lookup_path.exists():
                hash_lookup = load_hash_lookup(str(hash_lookup_path))
            else:
                logger.warning("Hash lookup not found, report will be limited")
                hash_lookup = {}
        
        if string_dict is None:
            string_dict_path = Path(checkpoint_dir) / 'string_dict.pkl'
            if string_dict_path.exists():
                string_dict = load_string_dict(str(string_dict_path))
            else:
                logger.warning("String dictionary not found, hashes will not be resolved")
                string_dict = {}
    
    def resolve_hash(hash_value: str) -> str:
        """Resolve hash to human-readable string with fallback."""
        if not hash_value or hash_value == "NULL":
            return None
        return string_dict.get(hash_value, f"[Hash: {hash_value}]")
    
    # Initialize report structure
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    report = {
        'metadata': {
            'generated_at': timestamp,
            'generated_by': 'Subject Enhancement Reporting Module',
            'pipeline_config': {
                'quality_audit_enabled': config.get('subject_quality_audit', {}).get('enabled', True),
                'imputation_enabled': config.get('subject_imputation', {}).get('enabled', True),
                'quality_auto_remediate': config.get('subject_quality_audit', {}).get('auto_remediate', True),
                'quality_confidence_threshold': config.get('subject_quality_audit', {}).get('confidence_threshold', 0.80),
                'imputation_confidence_threshold': config.get('subject_imputation', {}).get('confidence_threshold', 0.70)
            }
        },
        'summary': {
            'total_records': len(hash_lookup),
            'records_with_subjects': 0,
            'records_missing_subjects': 0,
            'quality_evaluated_records': 0,
            'quality_remediated_records': 0,
            'imputed_records': 0,
            'enhancement_coverage': 0.0
        },
        'quality_audit': {
            'enabled': config.get('subject_quality_audit', {}).get('enabled', True),
            'evaluated_records': [],
            'remediated_records': [],
            'statistics': {}
        },
        'subject_imputation': {
            'enabled': config.get('subject_imputation', {}).get('enabled', True),
            'imputed_records': [],
            'statistics': {}
        },
        'detailed_results': []
    }
    
    # Process hash_lookup to gather enhancement results
    quality_evaluated = 0
    quality_remediated = 0
    imputed_subjects = 0
    records_with_subjects = 0
    records_missing_subjects = 0
    
    for record_id, record_data in hash_lookup.items():
        subject_hash = record_data.get('subjects')
        has_subject = subject_hash and subject_hash != "NULL"
        
        if has_subject:
            records_with_subjects += 1
        else:
            records_missing_subjects += 1
        
        # Create detailed record entry
        record_detail = {
            'record_id': record_id,
            'composite_field': resolve_hash(record_data.get('composite')),
            'person_name': resolve_hash(record_data.get('person')),
            'title': resolve_hash(record_data.get('title')),
            'current_subject': resolve_hash(subject_hash),
            'enhancement_history': []
        }
        
        # Check for quality audit results
        if record_data.get('subject_evaluated', False):
            quality_evaluated += 1
            
            enhancement_entry = {
                'type': 'quality_audit',
                'timestamp': timestamp,
                'remediation_required': record_data.get('subject_remediation_required', False),
                'confidence': record_data.get('imputation_confidence', 0.0)
            }
            
            if record_data.get('subject_remediated', False):
                quality_remediated += 1
                original_subject = resolve_hash(record_data.get('original_subject_hash'))
                enhancement_entry.update({
                    'action_taken': 'subject_improved',
                    'original_subject': original_subject,
                    'improved_subject': resolve_hash(subject_hash),
                    'improvement_reason': 'semantic_similarity_analysis'
                })
                
                report['quality_audit']['remediated_records'].append({
                    'record_id': record_id,
                    'person_name': record_detail['person_name'],
                    'original_subject': original_subject,
                    'improved_subject': record_detail['current_subject'],
                    'confidence': enhancement_entry['confidence']
                })
            else:
                enhancement_entry['action_taken'] = 'no_improvement_needed'
            
            record_detail['enhancement_history'].append(enhancement_entry)
            
            if record_data.get('subject_remediation_required', False):
                report['quality_audit']['evaluated_records'].append({
                    'record_id': record_id,
                    'person_name': record_detail['person_name'],
                    'current_subject': record_detail['current_subject'],
                    'remediation_required': True,
                    'alternative_available': bool(record_data.get('alternative_subject_hash'))
                })
        
        # Check for imputation results
        if record_data.get('imputed_subject', False):
            imputed_subjects += 1
            
            enhancement_entry = {
                'type': 'subject_imputation',
                'timestamp': timestamp,
                'action_taken': 'subject_imputed',
                'imputed_subject': resolve_hash(subject_hash),
                'confidence': record_data.get('imputation_confidence', 0.0),
                'alternative_count': record_data.get('imputation_alternatives', 0),
                'method': 'composite_field_vector_similarity'
            }
            
            record_detail['enhancement_history'].append(enhancement_entry)
            
            report['subject_imputation']['imputed_records'].append({
                'record_id': record_id,
                'person_name': record_detail['person_name'],
                'composite_field': record_detail['composite_field'],
                'imputed_subject': record_detail['current_subject'],
                'confidence': enhancement_entry['confidence'],
                'alternative_count': enhancement_entry['alternative_count']
            })
        
        # Add record to detailed results if it has enhancement history
        if record_detail['enhancement_history']:
            report['detailed_results'].append(record_detail)
    
    # Update summary statistics
    report['summary'].update({
        'records_with_subjects': records_with_subjects,
        'records_missing_subjects': records_missing_subjects,
        'quality_evaluated_records': quality_evaluated,
        'quality_remediated_records': quality_remediated,
        'imputed_records': imputed_subjects,
        'enhancement_coverage': (quality_evaluated + imputed_subjects) / max(len(hash_lookup), 1)
    })
    
    # Add quality audit statistics
    if quality_evaluated > 0:
        report['quality_audit']['statistics'] = {
            'total_evaluated': quality_evaluated,
            'remediation_rate': quality_remediated / quality_evaluated,
            'records_needing_remediation': len(report['quality_audit']['evaluated_records']),
            'successful_remediations': quality_remediated
        }
    
    # Add imputation statistics
    if imputed_subjects > 0:
        # Calculate average confidence for imputed subjects
        confidence_scores = [r['confidence'] for r in report['subject_imputation']['imputed_records']]
        report['subject_imputation']['statistics'] = {
            'total_imputed': imputed_subjects,
            'imputation_rate': imputed_subjects / max(records_missing_subjects, 1),
            'average_confidence': sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0,
            'high_confidence_count': sum(1 for c in confidence_scores if c >= 0.80)
        }
    
    # Load external audit reports if available
    quality_reports = glob.glob(str(reports_dir / 'subject_quality_audit_*.json'))
    imputation_reports = glob.glob(str(output_dir / 'subject_imputation_results_*.json'))
    
    if quality_reports:
        # Load the most recent quality audit report
        latest_quality_report = max(quality_reports, key=os.path.getctime)
        try:
            with open(latest_quality_report, 'r') as f:
                external_quality_data = json.load(f)
            report['quality_audit']['external_report'] = {
                'source_file': os.path.basename(latest_quality_report),
                'statistics': external_quality_data.get('statistics', {}),
                'configuration': external_quality_data.get('configuration', {})
            }
        except Exception as e:
            logger.warning(f"Could not load external quality report: {e}")
    
    if imputation_reports:
        # Load the most recent imputation report
        latest_imputation_report = max(imputation_reports, key=os.path.getctime)
        try:
            with open(latest_imputation_report, 'r') as f:
                external_imputation_data = json.load(f)
            report['subject_imputation']['external_report'] = {
                'source_file': os.path.basename(latest_imputation_report),
                'statistics': external_imputation_data.get('statistics', {}),
                'configuration': external_imputation_data.get('configuration', {})
            }
        except Exception as e:
            logger.warning(f"Could not load external imputation report: {e}")
    
    # Save the comprehensive report
    report_filename = f'subject_enhancement_comprehensive_report_{timestamp}.json'
    report_path = reports_dir / report_filename
    
    try:
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        elapsed_time = time.time() - start_time
        
        logger.info(f"Subject enhancement report generated in {elapsed_time:.2f} seconds")
        logger.info(f"Report saved to: {report_path}")
        logger.info(f"Enhanced {quality_remediated + imputed_subjects} records total")
        
        return {
            'status': 'completed',
            'report_path': str(report_path),
            'elapsed_time': elapsed_time,
            'total_records': len(hash_lookup),
            'enhanced_records': quality_remediated + imputed_subjects,
            'quality_evaluated': quality_evaluated,
            'quality_remediated': quality_remediated,
            'subjects_imputed': imputed_subjects,
            'enhancement_coverage': report['summary']['enhancement_coverage']
        }
        
    except Exception as e:
        logger.error(f"Error saving subject enhancement report: {e}")
        return {
            'status': 'failed',
            'error': str(e),
            'elapsed_time': time.time() - start_time
        }
