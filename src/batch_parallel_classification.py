"""
Classification module for entity resolution pipeline.

This module trains and evaluates the entity resolution classifier
using gradient descent for logistic regression.
"""

import os
import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, roc_auc_score, confusion_matrix
)
import networkx as nx
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.utils import (
    Timer, get_memory_usage, save_checkpoint, load_checkpoint
)

logger = logging.getLogger(__name__)

class Classifier:
    """
    Handles classification for entity resolution.
    
    Features:
    - Logistic regression with gradient descent
    - Batch processing of training data
    - Parallel processing of predictions
    - Model persistence and loading
    - Evaluation metrics
    - Clustering of matching records
    - Detailed test results export
    """
    
    def __init__(self, config):
        """
        Initialize the classifier with configuration parameters.
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = config
        
        # Classification parameters
        self.algorithm = config['classification']['algorithm']
        self.regularization = config['classification']['regularization']
        self.regularization_strength = config['classification']['regularization_strength']
        self.learning_rate = config['classification']['learning_rate']
        self.max_iterations = config['classification']['max_iterations']
        self.convergence_tolerance = config['classification']['convergence_tolerance']
        self.batch_size = config['classification']['batch_size']
        self.class_weight = config['classification']['class_weight']
        self.decision_threshold = config['classification']['decision_threshold']
        self.precision_recall_tradeoff = config['classification']['precision_recall_tradeoff']
        
        # Clustering parameters
        self.clustering_algorithm = config['clustering']['algorithm']
        self.min_edge_weight = config['clustering']['min_edge_weight']
        self.transitivity_enabled = config['clustering']['transitivity_enabled']
        self.resolve_conflicts = config['clustering']['resolve_conflicts']
        self.min_cluster_size = config['clustering']['min_cluster_size']
        
        # Batch processing parameters
        self.system_batch_size = config['system']['batch_size']
        self.max_workers = config['system']['max_workers']
        
        # Data paths
        self.output_dir = Path(config['system']['output_dir'])
        self.temp_dir = Path(config['system']['temp_dir'])
        self.checkpoint_dir = Path(config['system']['checkpoint_dir'])
        self.train_test_split = config['data']['train_test_split']
        
        # Create directories if they don't exist
        for dir_path in [self.output_dir, self.temp_dir, self.checkpoint_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize model parameters
        self.weights = None
        self.bias = None
        self.feature_names = None
        
        logger.info("Classifier initialized")
    
    def execute(self, checkpoint=None):
        """
        Execute classification (train and evaluate).
        
        Args:
            checkpoint (str, optional): Path to checkpoint file. Defaults to None.
            
        Returns:
            dict: Classification results
        """
        # Load checkpoint if provided
        if checkpoint and os.path.exists(checkpoint):
            state = load_checkpoint(checkpoint)
            self.weights = state.get('weights')
            self.bias = state.get('bias')
            self.feature_names = state.get('feature_names')
            metrics = state.get('metrics', {})
            logger.info(f"Resumed classification from checkpoint: {checkpoint}")
            logger.info(f"Loaded weights with shape {self.weights.shape if self.weights is not None else None}")
            logger.info(f"Loaded {len(self.feature_names)} feature names")
            
            # If weights and metrics are available, return results
            if self.weights is not None and metrics:
                logger.info("Classification already completed")
                return metrics
        
        # Load ground truth features
        features, labels, pair_ids = self._load_ground_truth_features()
        logger.info(f"Loaded {len(features)} feature vectors")
        
        # Split into train and test sets
        train_features, train_labels, train_pair_ids, test_features, test_labels, test_pair_ids = self._train_test_split(
            features, labels, pair_ids
        )
        logger.info(f"Split data into {len(train_features)} train and {len(test_features)} test samples")
        
        # Train model
        with Timer() as timer:
            self._train(train_features, train_labels)
            
            # Evaluate model
            metrics = self._evaluate(test_features, test_labels, test_pair_ids)
            metrics['training_duration'] = timer.duration
        
        # Save results
        self._save_results(metrics)
        
        logger.info(
            f"Classification completed: accuracy={metrics['accuracy']:.4f}, "
            f"precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, "
            f"f1={metrics['f1']:.4f}, "
            f"{timer.duration:.2f} seconds"
        )
        
        return metrics
    
    def execute_prediction(self, checkpoint=None):
        """
        Execute prediction on candidate pairs.
        
        Args:
            checkpoint (str, optional): Path to checkpoint file. Defaults to None.
            
        Returns:
            dict: Prediction results
        """
        # Load checkpoint if provided
        if checkpoint and os.path.exists(checkpoint):
            state = load_checkpoint(checkpoint)
            self.weights = state.get('weights')
            self.bias = state.get('bias')
            self.feature_names = state.get('feature_names')
            processed_pairs = set(state.get('processed_pairs', []))
            predictions = state.get('predictions', {})
            logger.info(f"Resumed prediction from checkpoint: {checkpoint}")
            logger.info(f"Loaded weights with shape {self.weights.shape if self.weights is not None else None}")
            logger.info(f"Loaded {len(processed_pairs)} processed pairs")
            logger.info(f"Loaded {len(predictions)} predictions")
        else:
            processed_pairs = set()
            predictions = {}
        
        # Load model if not loaded from checkpoint
        if self.weights is None:
            self._load_model()
        
        # Load candidate features
        feature_set = self._load_candidate_features()
        logger.info(f"Loaded {len(feature_set)} candidate feature vectors")
        
        # Filter out already processed pairs
        pairs_to_process = {
            pair_id: data
            for pair_id, data in feature_set.items()
            if pair_id not in processed_pairs
        }
        
        if not pairs_to_process:
            logger.info("No new pairs to process")
            return {
                'pairs_processed': len(processed_pairs),
                'predictions': len(predictions),
                'duration': 0.0
            }
        
        logger.info(f"Processing {len(pairs_to_process)} candidate pairs")
        
        if self.config['system']['mode'] == 'dev':
            # In dev mode, limit the number of pairs to process
            dev_sample_size = min(
                self.config['system']['dev_sample_size'],
                len(pairs_to_process)
            )
            pair_ids = list(pairs_to_process.keys())[:dev_sample_size]
            pairs_to_process = {
                pair_id: pairs_to_process[pair_id]
                for pair_id in pair_ids
            }
            logger.info(f"Dev mode: limited to {len(pairs_to_process)} pairs")
        
        # Create batches of pairs for parallel processing
        pair_batches = self._create_pair_batches(pairs_to_process)
        logger.info(f"Created {len(pair_batches)} pair batches")
        
        # Process pair batches
        with Timer() as timer:
            for batch_idx, batch in enumerate(tqdm(pair_batches, desc="Processing prediction batches")):
                try:
                    # Get predictions for batch
                    batch_predictions = self._predict_batch(batch)
                    
                    # Update predictions
                    predictions.update(batch_predictions)
                    
                    # Update processed pairs
                    processed_pairs.update(batch.keys())
                    
                    # Save checkpoint periodically
                    if (batch_idx + 1) % 10 == 0:
                        logger.info(
                            f"Processed {batch_idx + 1}/{len(pair_batches)} batches, "
                            f"{len(processed_pairs)}/{len(feature_set)} pairs"
                        )
                        
                        checkpoint_path = self.checkpoint_dir / f"prediction_{batch_idx + 1}.ckpt"
                        save_checkpoint({
                            'weights': self.weights,
                            'bias': self.bias,
                            'feature_names': self.feature_names,
                            'processed_pairs': list(processed_pairs),
                            'predictions': predictions
                        }, checkpoint_path)
                
                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx}: {e}")
                    
                    # Save checkpoint on error
                    error_checkpoint = self.checkpoint_dir / f"prediction_error_{batch_idx}.ckpt"
                    save_checkpoint({
                        'weights': self.weights,
                        'bias': self.bias,
                        'feature_names': self.feature_names,
                        'processed_pairs': list(processed_pairs),
                        'predictions': predictions
                    }, error_checkpoint)
                    
                    # Continue with next batch
                    continue
            
            # Save all predictions to CSV for analysis
            self._save_predictions_to_csv(predictions, feature_set)
            
            # Perform clustering
            clusters = self._cluster_predictions(predictions)
        
        # Save results
        self._save_prediction_results(predictions, clusters, processed_pairs)
        
        results = {
            'pairs_processed': len(processed_pairs),
            'predictions': len(predictions),
            'match_predictions': sum(1 for p in predictions.values() if p['match']),
            'clusters': len(clusters),
            'duration': timer.duration
        }
        
        logger.info(
            f"Prediction completed: {results['pairs_processed']} pairs, "
            f"{results['match_predictions']} matches, "
            f"{results['clusters']} clusters, "
            f"{timer.duration:.2f} seconds"
        )
        
        return results
    
    def _load_ground_truth_features(self):
        """
        Load ground truth features for classification.
        
        Returns:
            tuple: (features, labels, pair_ids)
        """
        # Load metadata
        metadata_path = self.output_dir / "ground_truth_features_metadata.json"
        if not metadata_path.exists():
            logger.error("Ground truth features metadata not found")
            return [], [], []
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load feature names
        self.feature_names = metadata.get('feature_names', [])
        
        # Load feature set
        feature_set_path = metadata.get('feature_set_path')
        if not feature_set_path:
            logger.error("Feature set path not found in metadata")
            return [], [], []
        
        with open(feature_set_path, 'rb') as f:
            import pickle
            feature_set = pickle.load(f)
        
        # Extract features and labels
        features = []
        labels = []
        pair_ids = []
        
        for pair_id, data in feature_set.items():
            feature_dict = data.get('features', {})
            match = data.get('labels', {}).get('match')
            
            # Skip pairs with empty features or no match label
            if not feature_dict or match is None:
                continue
            
            # Convert feature dict to vector in consistent order
            feature_vector = [feature_dict.get(name, 0.0) for name in self.feature_names]
            
            features.append(feature_vector)
            labels.append(1 if match else 0)
            pair_ids.append(pair_id)
        
        return np.array(features), np.array(labels), pair_ids
    
    def _load_candidate_features(self):
        """
        Load candidate features for prediction.
        
        Returns:
            dict: Feature set
        """
        # Load metadata
        metadata_path = self.output_dir / "candidate_features_metadata.json"
        if not metadata_path.exists():
            logger.error("Candidate features metadata not found")
            return {}
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load feature set
        feature_set_path = metadata.get('feature_set_path')
        if not feature_set_path:
            logger.error("Feature set path not found in metadata")
            return {}
        
        with open(feature_set_path, 'rb') as f:
            import pickle
            feature_set = pickle.load(f)
        
        return feature_set
    
    def _load_model(self):
        """
        Load trained model from file.
        """
        # Load metadata
        metadata_path = self.output_dir / "classification_metadata.json"
        if not metadata_path.exists():
            logger.error("Classification metadata not found")
            return
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load model
        model_path = metadata.get('model_path')
        if not model_path:
            logger.error("Model path not found in metadata")
            return
        
        with open(model_path, 'rb') as f:
            import pickle
            model = pickle.load(f)
        
        # Extract weights and bias
        self.weights = model.get('weights')
        self.bias = model.get('bias')
        self.feature_names = model.get('feature_names')
        
        logger.info(f"Loaded model with {len(self.feature_names)} features")
    
    def _train_test_split(self, features, labels, pair_ids=None):
        """
        Split data into train and test sets.
        
        Args:
            features (ndarray): Feature vectors
            labels (ndarray): Labels
            pair_ids (list, optional): Pair IDs for tracking. Defaults to None.
            
        Returns:
            tuple: Split datasets
        """
        # Set random seed for reproducibility
        np.random.seed(self.config['system']['random_seed'])
        
        # Shuffle data with a fixed permutation
        indices = np.random.permutation(len(features))
        features = features[indices]
        labels = labels[indices]
        
        # Split data
        split_idx = int(len(features) * self.train_test_split)
        train_features = features[:split_idx]
        train_labels = labels[:split_idx]
        test_features = features[split_idx:]
        test_labels = labels[split_idx:]
        
        # Split pair IDs if provided
        if pair_ids:
            pair_ids = np.array(pair_ids)[indices]
            train_pair_ids = pair_ids[:split_idx].tolist()
            test_pair_ids = pair_ids[split_idx:].tolist()
            return train_features, train_labels, train_pair_ids, test_features, test_labels, test_pair_ids
        
        return train_features, train_labels, None, test_features, test_labels, None
    
    def _create_pair_batches(self, pairs):
        """
        Create batches of pairs for parallel processing.
        
        Args:
            pairs (dict): Dictionary of pair ID -> data
            
        Returns:
            list: List of batch dictionaries
        """
        # Create batches
        batches = []
        pair_ids = list(pairs.keys())
        
        for i in range(0, len(pair_ids), self.system_batch_size):
            batch_ids = pair_ids[i:i + self.system_batch_size]
            batch = {
                pair_id: pairs[pair_id]
                for pair_id in batch_ids
            }
            batches.append(batch)
        
        return batches
    
    def _train(self, features, labels):
        """
        Train the classifier using gradient descent.
        
        Args:
            features (ndarray): Feature vectors
            labels (ndarray): Labels
        """
        # Initialize weights and bias
        n_features = features.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        # Initialize class weights
        if self.class_weight == "balanced":
            # Compute class weights inversely proportional to class frequencies
            class_counts = np.bincount(labels)
            total_samples = len(labels)
            
            # Avoid division by zero
            class_counts = np.maximum(class_counts, 1)
            
            class_weights = {
                0: total_samples / (2 * class_counts[0]),
                1: total_samples / (2 * class_counts[1])
            }
            
            logger.info(f"Using balanced class weights: {class_weights}")
        else:
            class_weights = {0: 1.0, 1: 1.0}
        
        # Initialize training variables
        prev_loss = float('inf')
        
        # Gradient descent
        for iteration in range(self.max_iterations):
            # Process data in batches
            batch_losses = []
            
            for i in range(0, len(features), self.batch_size):
                batch_features = features[i:i + self.batch_size]
                batch_labels = labels[i:i + self.batch_size]
                
                # Forward pass
                z = np.dot(batch_features, self.weights) + self.bias
                predictions = self._sigmoid(z)
                
                # Compute loss
                sample_weights = np.array([class_weights[label] for label in batch_labels])
                loss = self._binary_cross_entropy(batch_labels, predictions, sample_weights)
                batch_losses.append(loss)
                
                # Backward pass
                d_predictions = (predictions - batch_labels) * sample_weights
                d_weights = np.dot(batch_features.T, d_predictions) / len(batch_labels)
                d_bias = np.mean(d_predictions)
                
                # Update weights with regularization
                if self.regularization == "l2":
                    d_weights += (self.regularization_strength * self.weights) / len(batch_labels)
                elif self.regularization == "l1":
                    d_weights += (self.regularization_strength * np.sign(self.weights)) / len(batch_labels)
                
                # Update parameters
                self.weights -= self.learning_rate * d_weights
                self.bias -= self.learning_rate * d_bias
            
            # Compute average loss across batches
            avg_loss = np.mean(batch_losses)
            
            # Check for convergence
            if iteration % 10 == 0:
                logger.info(f"Iteration {iteration}, Loss: {avg_loss:.6f}")
            
            if abs(prev_loss - avg_loss) < self.convergence_tolerance:
                logger.info(f"Converged at iteration {iteration}, Loss: {avg_loss:.6f}")
                break
            
            prev_loss = avg_loss
        
        logger.info(f"Training completed after {iteration + 1} iterations, Final Loss: {avg_loss:.6f}")
    
    def _evaluate(self, features, labels, pair_ids=None):
        """
        Evaluate the classifier on test data.
        
        Args:
            features (ndarray): Feature vectors
            labels (ndarray): Labels
            pair_ids (list, optional): Pair IDs for tracking. Defaults to None.
            
        Returns:
            dict: Evaluation metrics
        """
        # Make predictions
        probabilities = self._predict(features)
        
        # Convert probabilities to binary predictions based on threshold
        predictions = (probabilities >= self.decision_threshold).astype(int)
        
        # Compute metrics
        metrics = {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions),
            'recall': recall_score(labels, predictions),
            'f1': f1_score(labels, predictions),
            'roc_auc': roc_auc_score(labels, probabilities)
        }
        
        # Compute confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        # Convert confusion matrix to dictionary
        metrics['confusion_matrix'] = {
            'true_negatives': int(cm[0, 0]),
            'false_positives': int(cm[0, 1]),
            'false_negatives': int(cm[1, 0]),
            'true_positives': int(cm[1, 1])
        }
        
        # Adjust threshold based on precision-recall tradeoff
        if self.precision_recall_tradeoff != "balanced":
            # Find optimal threshold based on preference
            thresholds = np.linspace(0.1, 0.9, 9)
            best_threshold = self.decision_threshold
            best_score = 0.0
            
            for threshold in thresholds:
                predictions = (probabilities >= threshold).astype(int)
                precision = precision_score(labels, predictions)
                recall = recall_score(labels, predictions)
                
                if self.precision_recall_tradeoff == "precision":
                    score = precision
                elif self.precision_recall_tradeoff == "recall":
                    score = recall
                else:  # f1
                    score = f1_score(labels, predictions)
                
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
            
            # Update threshold
            self.decision_threshold = best_threshold
            metrics['decision_threshold'] = self.decision_threshold
            
            # Update predictions and metrics with new threshold
            predictions = (probabilities >= self.decision_threshold).astype(int)
            metrics['accuracy'] = accuracy_score(labels, predictions)
            metrics['precision'] = precision_score(labels, predictions)
            metrics['recall'] = recall_score(labels, predictions)
            metrics['f1'] = f1_score(labels, predictions)
            
            # Update confusion matrix
            cm = confusion_matrix(labels, predictions)
            metrics['confusion_matrix'] = {
                'true_negatives': int(cm[0, 0]),
                'false_positives': int(cm[0, 1]),
                'false_negatives': int(cm[1, 0]),
                'true_positives': int(cm[1, 1])
            }
        
        # Compute feature importance
        feature_importance = self._compute_feature_importance()
        metrics['feature_importance'] = feature_importance
        
        # Save detailed test results to CSV
        self._save_test_results_to_csv(features, labels, predictions, probabilities, pair_ids)
        
        return metrics
    
    def _save_test_results_to_csv(self, features, labels, predictions, probabilities, pair_ids=None):
        """
        Save detailed test results to CSV file.
        
        Args:
            features (ndarray): Feature vectors
            labels (ndarray): True labels
            predictions (ndarray): Predicted labels
            probabilities (ndarray): Prediction probabilities
            pair_ids (list, optional): Pair IDs for tracking. Defaults to None.
        """
        print("\n=== DEBUG: SAVING TEST RESULTS ===")
        print(f"Feature array shape: {features.shape}")
        
        # Sample a few feature vectors
        for i in range(min(3, features.shape[0])):
            # Print cosine features
            cosine_indices = [j for j, name in enumerate(self.feature_names) if 'cosine' in name]
            cosine_values = {self.feature_names[j]: features[i, j] for j in cosine_indices}
            print(f"Sample {i} cosine features: {cosine_values}")

        # Create test results dataframe
        results_dict = {
            'true_label': labels,
            'predicted_label': predictions,
            'confidence': probabilities,
            'correct': labels == predictions
        }

        for i, feature_name in enumerate(self.feature_names):
            feature_values = features[:, i]
            results_dict[feature_name] = feature_values
            
            # Check if cosine feature to debug
            if 'cosine' in feature_name:
                non_zero = np.count_nonzero(feature_values)
                print(f"Feature {feature_name}: non-zero values={non_zero}/{len(feature_values)}")
                print(f"First 5 values: {feature_values[:5]}")
        
        # Add pair IDs if available
        if pair_ids and len(pair_ids) == len(labels):
            results_dict['pair_id'] = pair_ids
            # Split into left and right IDs
            left_ids = []
            right_ids = []
            for pair_id in pair_ids:
                try:
                    left_id, right_id = pair_id.split('_', 1)
                    left_ids.append(left_id)
                    right_ids.append(right_id)
                except:
                    left_ids.append('')
                    right_ids.append('')
            
            results_dict['left_id'] = left_ids
            results_dict['right_id'] = right_ids
        
        # Add feature values
        for i, feature_name in enumerate(self.feature_names):
            results_dict[feature_name] = features[:, i]
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results_dict)
        
        # Save to CSV
        csv_path = self.output_dir / "test_results_detailed.csv"
        results_df.to_csv(csv_path, index=False)
        logger.info(f"Saved detailed test results to {csv_path}")
        
        # Create a simpler version with just essential columns for easy review
        essential_cols = ['pair_id', 'left_id', 'right_id', 'true_label', 'predicted_label', 
                          'confidence', 'correct']
        
        # Add top 5 most important features
        if len(self.feature_names) > 0:
            feature_importance = self._compute_feature_importance()
            top_features = list(feature_importance.keys())[:5]
            essential_cols.extend(top_features)
        
        # Filter columns that exist in the dataframe
        essential_cols = [col for col in essential_cols if col in results_df.columns]
        
        # Save simplified version
        simple_csv_path = self.output_dir / "test_results_summary.csv"
        results_df[essential_cols].to_csv(simple_csv_path, index=False)
        logger.info(f"Saved simplified test results to {simple_csv_path}")
    
    def _save_predictions_to_csv(self, predictions, feature_set):
        """
        Save full prediction results to CSV for analysis.
        
        Args:
            predictions (dict): Dictionary of predictions
            feature_set (dict): Feature set
        """
        # Create predictions dataframe
        rows = []
        
        for pair_id, prediction in predictions.items():
            # Get features
            features = {}
            if pair_id in feature_set:
                features = feature_set[pair_id].get('features', {})
            
            # Create row
            row = {
                'pair_id': pair_id,
                'left_id': prediction.get('left_id', ''),
                'right_id': prediction.get('right_id', ''),
                'predicted_match': prediction.get('match', False),
                'confidence': prediction.get('probability', 0.0)
            }
            
            # Add feature values
            for feature_name in self.feature_names:
                if feature_name in features:
                    row[feature_name] = features[feature_name]
            
            rows.append(row)
        
        # Convert to DataFrame
        predictions_df = pd.DataFrame(rows)
        
        # Save to CSV
        csv_path = self.output_dir / "prediction_results.csv"
        predictions_df.to_csv(csv_path, index=False)
        logger.info(f"Saved prediction results to {csv_path}")
    
    def _predict(self, features):
        """
        Make predictions for feature vectors.
        
        Args:
            features (ndarray): Feature vectors
            
        Returns:
            ndarray: Predicted probabilities
        """
        z = np.dot(features, self.weights) + self.bias
        return self._sigmoid(z)
    
    def _predict_batch(self, batch):
        """
        Make predictions for a batch of pairs.
        
        Args:
            batch (dict): Dictionary of pair ID -> data
            
        Returns:
            dict: Dictionary of pair ID -> prediction
        """
        # Create a process pool for parallel processing
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit jobs
            future_to_pair = {
                executor.submit(
                    self._predict_pair,
                    pair_id,
                    data
                ): pair_id
                for pair_id, data in batch.items()
            }
            
            # Process results as they complete
            batch_predictions = {}
            for future in as_completed(future_to_pair):
                pair_id = future_to_pair[future]
                
                try:
                    prediction = future.result()
                    if prediction:
                        batch_predictions[pair_id] = prediction
                
                except Exception as e:
                    logger.error(f"Error predicting pair {pair_id}: {e}")
        
        return batch_predictions
    
    def _predict_pair(self, pair_id, data):
        """
        Make prediction for a single pair.
        
        Args:
            pair_id (str): Pair ID
            data (dict): Pair data
            
        Returns:
            dict: Prediction result
        """
        features = data.get('features', {})
        
        # Skip pairs with empty features
        if not features:
            return None
        
        # Convert feature dict to vector in consistent order
        feature_vector = np.array([features.get(name, 0.0) for name in self.feature_names])
        
        # Make prediction
        probability = self._sigmoid(np.dot(feature_vector, self.weights) + self.bias)
        match = probability >= self.decision_threshold
        
        # Parse pair ID
        left_id, right_id = pair_id.split('_', 1)
        
        return {
            'left_id': left_id,
            'right_id': right_id,
            'probability': float(probability),
            'match': bool(match)
        }
    
    def _cluster_predictions(self, predictions):
        """
        Cluster matching records based on predictions.
        
        Args:
            predictions (dict): Dictionary of pair ID -> prediction
            
        Returns:
            list: List of clusters
        """
        # Create graph
        G = nx.Graph()
        
        # Add edges for matching pairs
        for pair_id, prediction in predictions.items():
            if prediction['match']:
                left_id = prediction['left_id']
                right_id = prediction['right_id']
                probability = prediction['probability']
                
                # Add nodes if they don't exist
                if left_id not in G:
                    G.add_node(left_id)
                
                if right_id not in G:
                    G.add_node(right_id)
                
                # Add edge with probability as weight
                G.add_edge(left_id, right_id, weight=probability)
        
        # Apply clustering algorithm
        if self.clustering_algorithm == "connected_components":
            # Remove edges with weight below threshold
            for u, v, w in list(G.edges(data='weight')):
                if w < self.min_edge_weight:
                    G.remove_edge(u, v)
            
            # Find connected components
            clusters = list(nx.connected_components(G))
        
        elif self.clustering_algorithm == "louvain":
            try:
                from community import best_partition
                
                # Apply Louvain algorithm
                partition = best_partition(G)
                
                # Group nodes by community
                communities = {}
                for node, community_id in partition.items():
                    if community_id not in communities:
                        communities[community_id] = set()
                    
                    communities[community_id].add(node)
                
                clusters = list(communities.values())
            
            except ImportError:
                logger.warning("community package not found, falling back to connected components")
                
                # Remove edges with weight below threshold
                for u, v, w in list(G.edges(data='weight')):
                    if w < self.min_edge_weight:
                        G.remove_edge(u, v)
                
                # Find connected components
                clusters = list(nx.connected_components(G))
        
        elif self.clustering_algorithm == "label_propagation":
            try:
                from networkx.algorithms.community import label_propagation_communities
                
                # Apply label propagation algorithm
                clusters = list(label_propagation_communities(G))
            
            except ImportError:
                logger.warning("label_propagation_communities not available, falling back to connected components")
                
                # Remove edges with weight below threshold
                for u, v, w in list(G.edges(data='weight')):
                    if w < self.min_edge_weight:
                        G.remove_edge(u, v)
                
                # Find connected components
                clusters = list(nx.connected_components(G))
        
        else:
            logger.warning(f"Unknown clustering algorithm: {self.clustering_algorithm}, falling back to connected components")
            
            # Remove edges with weight below threshold
            for u, v, w in list(G.edges(data='weight')):
                if w < self.min_edge_weight:
                    G.remove_edge(u, v)
            
            # Find connected components
            clusters = list(nx.connected_components(G))
        
        # Filter clusters by size
        clusters = [cluster for cluster in clusters if len(cluster) >= self.min_cluster_size]
        
        # Save clusters to CSV
        self._save_clusters_to_csv(clusters)
        
        return clusters
    
    def _save_clusters_to_csv(self, clusters):
        """
        Save clusters to CSV file.
        
        Args:
            clusters (list): List of clusters
        """
        # Create clusters dataframe
        rows = []
        
        for cluster_id, cluster in enumerate(clusters):
            for entity_id in cluster:
                rows.append({
                    'cluster_id': cluster_id,
                    'entity_id': entity_id,
                    'cluster_size': len(cluster)
                })
        
        # Convert to DataFrame
        clusters_df = pd.DataFrame(rows)
        
        # Save to CSV
        csv_path = self.output_dir / "clusters.csv"
        clusters_df.to_csv(csv_path, index=False)
        logger.info(f"Saved {len(clusters)} clusters to {csv_path}")
    
    def _sigmoid(self, z):
        """
        Sigmoid activation function.
        
        Args:
            z (ndarray): Input values
            
        Returns:
            ndarray: Sigmoid of input values
        """
        return 1.0 / (1.0 + np.exp(-np.clip(z, -100, 100)))
    
    def _binary_cross_entropy(self, y_true, y_pred, sample_weights=None):
        """
        Binary cross-entropy loss function.
        
        Args:
            y_true (ndarray): True labels
            y_pred (ndarray): Predicted probabilities
            sample_weights (ndarray, optional): Sample weights. Defaults to None.
            
        Returns:
            float: Loss value
        """
        # Avoid numerical issues
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        
        if sample_weights is None:
            sample_weights = np.ones_like(y_true)
        
        loss = -np.mean(
            sample_weights * (
                y_true * np.log(y_pred) +
                (1 - y_true) * np.log(1 - y_pred)
            )
        )
        
        return loss
    
    def _compute_feature_importance(self):
        """
        Compute feature importance based on model weights.
        
        Returns:
            dict: Feature importance
        """
        # Compute absolute weights
        abs_weights = np.abs(self.weights)
        
        # Normalize weights
        norm_weights = abs_weights / np.sum(abs_weights)
        
        # Create feature importance dictionary
        feature_importance = {}
        for i, name in enumerate(self.feature_names):
            feature_importance[name] = {
                'weight': float(self.weights[i]),
                'abs_weight': float(abs_weights[i]),
                'importance': float(norm_weights[i])
            }
        
        # Sort by importance
        feature_importance = {
            k: v for k, v in sorted(
                feature_importance.items(),
                key=lambda item: item[1]['importance'],
                reverse=True
            )
        }
        
        return feature_importance
    
    def _save_results(self, metrics):
        """
        Save classification results.
        
        Args:
            metrics (dict): Evaluation metrics
        """
        # Save model
        model = {
            'weights': self.weights,
            'bias': self.bias,
            'feature_names': self.feature_names,
            'decision_threshold': self.decision_threshold
        }
        
        model_path = self.temp_dir / "classification_model.pkl"
        with open(model_path, 'wb') as f:
            import pickle
            pickle.dump(model, f)
        
        # Save metrics
        with open(self.output_dir / "classification_metrics.json", 'w') as f:
            # Convert numpy values to native Python types
            def convert_numpy(obj):
                if isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(v) for v in obj]
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.number):
                    return obj.item()
                else:
                    return obj
            
            json.dump(convert_numpy(metrics), f, indent=2)
        
        # Save metadata
        with open(self.output_dir / "classification_metadata.json", 'w') as f:
            json.dump({
                'model_path': str(model_path),
                'feature_count': len(self.feature_names),
                'decision_threshold': float(self.decision_threshold),
                'feature_names': self.feature_names,
                'test_results_path': str(self.output_dir / "test_results_detailed.csv")
            }, f, indent=2)
        
        # Save final checkpoint
        checkpoint_path = self.checkpoint_dir / "classification_final.ckpt"
        save_checkpoint({
            'weights': self.weights,
            'bias': self.bias,
            'feature_names': self.feature_names,
            'metrics': metrics
        }, checkpoint_path)
        
        logger.info(f"Classification results saved to {self.output_dir}")
    
    def _save_prediction_results(self, predictions, clusters, processed_pairs):
        """
        Save prediction results.
        
        Args:
            predictions (dict): Dictionary of pair ID -> prediction
            clusters (list): List of clusters
            processed_pairs (set): Set of processed pair IDs
        """
        # Save predictions
        predictions_path = self.output_dir / "predictions.json"
        with open(predictions_path, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        # Save clusters
        clusters_path = self.output_dir / "clusters.json"
        with open(clusters_path, 'w') as f:
            # Convert sets to lists for JSON serialization
            clusters_list = [list(cluster) for cluster in clusters]
            json.dump(clusters_list, f, indent=2)
        
        # Save prediction statistics
        prediction_stats = {
            'total_pairs': len(predictions),
            'matches': sum(1 for p in predictions.values() if p['match']),
            'non_matches': sum(1 for p in predictions.values() if not p['match']),
            'match_percentage': sum(1 for p in predictions.values() if p['match']) / len(predictions) * 100 if predictions else 0,
            'cluster_count': len(clusters),
            'average_cluster_size': sum(len(cluster) for cluster in clusters) / len(clusters) if clusters else 0,
            'max_cluster_size': max(len(cluster) for cluster in clusters) if clusters else 0,
            'min_cluster_size': min(len(cluster) for cluster in clusters) if clusters else 0
        }
        
        with open(self.output_dir / "prediction_statistics.json", 'w') as f:
            json.dump(prediction_stats, f, indent=2)
        
        # Save metadata
        with open(self.output_dir / "prediction_metadata.json", 'w') as f:
            json.dump({
                'predictions_path': str(predictions_path),
                'clusters_path': str(clusters_path),
                'prediction_results_path': str(self.output_dir / "prediction_results.csv"),
                'clusters_csv_path': str(self.output_dir / "clusters.csv"),
                'total_pairs': len(predictions),
                'processed_pairs': len(processed_pairs),
                'cluster_count': len(clusters)
            }, f, indent=2)
        
        # Save final checkpoint
        checkpoint_path = self.checkpoint_dir / "prediction_final.ckpt"
        save_checkpoint({
            'weights': self.weights,
            'bias': self.bias,
            'feature_names': self.feature_names,
            'processed_pairs': list(processed_pairs),
            'predictions': predictions
        }, checkpoint_path)
        
        logger.info(f"Prediction results saved to {self.output_dir}")
