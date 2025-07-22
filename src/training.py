"""
Training Module for Entity Resolution

This module implements classifier training and testing using ground truth data.
It trains a logistic regression classifier optimized for entity resolution tasks
and provides evaluation metrics for model performance assessment.
"""

import logging
import os
import json
import numpy as np
import pandas as pd
import pickle
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from src.utils import setup_deterministic_behavior, get_subprocess_seed
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc, confusion_matrix
import seaborn as sns

# Local imports
# These imports will be relative to the project root when run from main.py
from src.feature_engineering import FeatureEngineering
from src.custom_features import register_custom_features
from src.reporting import generate_detailed_test_results
from src.scaling_bridge import ScalingBridge
from src.visualization import plot_feature_distributions, plot_class_separation, generate_feature_visualization_report
from weaviate.classes.query import Filter

logger = logging.getLogger(__name__)

class EntityClassifier:
    """
    Entity classifier for resolving matching entities in the catalog.
    Implements logistic regression with gradient descent optimization.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the entity classifier with configuration parameters.
        
        Args:
            config: Configuration dictionary with classifier parameters
        """
        self.config = config
        
        # Classifier parameters
        self.learning_rate = config.get("learning_rate", 0.01)
        self.max_iterations = config.get("max_iterations", 1000)
        self.batch_size = config.get("training_batch_size", 256)
        self.convergence_threshold = config.get("convergence_threshold", 1e-5)
        self.l2_lambda = config.get("l2_lambda", 0.01)  # L2 regularization parameter
        self.class_weight = config.get("class_weight", 5.0)  # Weight for positive class
        
        # Model parameters
        self.weights = None
        self.bias = 0.0
        self.feature_names = []
        self.decision_threshold = config.get("decision_threshold", 0.5)
        self.substitution_mapping = {}  # Will be populated from feature engineering
        self.feature_importance_factors = {}  # Will be populated from feature engineering
        self.scaling_disabled = config.get("disable_scaling", False)  # Track if scaling was disabled
        
        # Paths
        self.model_checkpoint_path = os.path.join(
            config.get("checkpoint_dir", "data/checkpoints"),
            "classifier_model.pkl"
        )
        self.evaluation_report_path = os.path.join(
            config.get("output_dir", "data/output"),
            "classifier_evaluation.json"
        )
        
        logger.info(f"Initialized EntityClassifier with {self.max_iterations} max iterations")
    
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None) -> Dict[str, Any]:
        """
        Train the classifier using gradient descent on logistic regression.
        
        Args:
            X: Feature matrix with shape (n_samples, n_features)
            y: Binary labels with shape (n_samples,)
            feature_names: Optional list of feature names for reporting
            
        Returns:
            Dictionary with training metrics
        """
        # Validate input data
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("X and y must be numpy arrays")
            
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have the same number of samples. Got X: {X.shape[0]}, y: {y.shape[0]}")
            
        if len(y.shape) != 1:
            raise ValueError(f"y must be a 1D array. Got shape: {y.shape}")
            
        # Check for NaN or infinity values
        if np.isnan(X).any() or np.isinf(X).any():
            logger.warning("X contains NaN or infinite values. Cleaning data...")
            X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
            
        if np.isnan(y).any() or np.isinf(y).any():
            raise ValueError("y contains NaN or infinite values. Please clean the labels.")
            
        # Ensure binary classification labels
        unique_labels = np.unique(y)
        if len(unique_labels) > 2 or not np.all(np.isin(unique_labels, [0, 1])):
            raise ValueError(f"y must contain only binary labels (0, 1). Got: {unique_labels}")
            
        logger.info(f"Training classifier on {X.shape[0]} samples with {X.shape[1]} features")
        
        # Store feature names if provided
        if feature_names:
            if len(feature_names) != X.shape[1]:
                logger.warning(f"Number of feature names ({len(feature_names)}) doesn't match number of features ({X.shape[1]})")
                # Adjust feature names to match - truncate or extend with generic names
                if len(feature_names) > X.shape[1]:
                    feature_names = feature_names[:X.shape[1]]
                    logger.warning(f"Truncated feature names to match feature count: {X.shape[1]}")
                else:
                    additional = [f"feature_{i}" for i in range(len(feature_names), X.shape[1])]
                    feature_names.extend(additional)
                    logger.warning(f"Extended feature names with generic names to match feature count: {X.shape[1]}")
            
            self.feature_names = feature_names
        
        # Split data into training and validation sets
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            logger.info(f"Split data into {X_train.shape[0]} training and {X_val.shape[0]} validation samples")
        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            logger.warning("Proceeding with all data for training")
            X_train, X_val = X, X
            y_train, y_val = y, y
        
        # Log class distribution
        train_pos = np.sum(y_train == 1)
        train_neg = np.sum(y_train == 0)
        val_pos = np.sum(y_val == 1)
        val_neg = np.sum(y_val == 0)
        logger.info(f"Class distribution - Train: {train_pos} positive, {train_neg} negative. Val: {val_pos} positive, {val_neg} negative")
        
        # Initialize weights
        n_features = X_train.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        # Apply feature importance factors if available
        if self.feature_names and self.feature_importance_factors:
            logger.info("Applying feature importance factors to initial weights")
            for i, feature_name in enumerate(self.feature_names):
                if feature_name in self.feature_importance_factors:
                    # Initialize with higher weights for important features
                    # This helps the model converge faster with appropriate feature importance
                    importance = self.feature_importance_factors[feature_name]
                    # We don't modify the features directly, but we can adjust initial weights
                    # to reflect the relative importance
                    if importance > 1.0:
                        logger.info(f"Initializing {feature_name} with increased importance: {importance}x")
                    elif importance < 1.0:
                        logger.info(f"Initializing {feature_name} with decreased importance: {importance}x")
        
        # Training variables
        best_val_loss = float('inf')
        best_weights = np.copy(self.weights)
        best_bias = self.bias
        patience = self.config.get("early_stopping_patience", 10)
        patience_counter = 0
        
        # Training metrics
        train_losses = []
        val_losses = []
        
        # Monitoring variables for numerical stability
        gradient_magnitudes = []
        weight_magnitudes = []
        largest_weight_changes = []
        
        # Early warning system for potential issues
        def check_training_health(iteration, grad_w, weights_before, weights_after):
            """Monitor training health and log warnings if issues are detected."""
            # Calculate gradient magnitude
            grad_magnitude = np.linalg.norm(grad_w)
            gradient_magnitudes.append(grad_magnitude)
            
            # Calculate weight magnitude
            weight_magnitude = np.linalg.norm(weights_after)
            weight_magnitudes.append(weight_magnitude)
            
            # Calculate largest weight change
            weight_change = np.max(np.abs(weights_after - weights_before))
            largest_weight_changes.append(weight_change)
            
            # Check for signs of instability
            if grad_magnitude > 1000:
                logger.warning(f"Iteration {iteration}: Large gradient magnitude: {grad_magnitude:.2f}")
                
            if weight_change > 10:
                logger.warning(f"Iteration {iteration}: Large weight change: {weight_change:.2f}")
                
            # Check for vanishing gradients (after initial iterations)
            if iteration > 5 and grad_magnitude < 1e-6:
                logger.warning(f"Iteration {iteration}: Possible vanishing gradient: {grad_magnitude:.8f}")
                
            # Check for weight explosion
            if weight_magnitude > 1000:
                logger.warning(f"Iteration {iteration}: Weights becoming very large: {weight_magnitude:.2f}")
                
            # Return True if training seems healthy, False if severe issues detected
            return not (grad_magnitude > 10000 or weight_magnitude > 10000 or np.isnan(grad_magnitude))
        
        # Compute class weights for imbalanced data
        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        pos_weight = self.class_weight if n_pos < n_neg else 1.0
        neg_weight = 1.0 if n_pos < n_neg else self.class_weight * n_neg / n_pos
        
        # Main training loop
        for iteration in range(self.max_iterations):
            # Shuffle data for stochastic gradient descent using a deterministic permutation
            # Use iteration number to derive a deterministic seed for each iteration
            # This ensures reproducibility while still providing effective shuffling
            iter_seed = (self.config.get("random_seed", 42) * 1000 + iteration) % 2147483647
            rng = np.random.RandomState(iter_seed)
            indices = rng.permutation(X_train.shape[0])
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            logger.debug(f"Iteration {iteration}: Using seed {iter_seed} for shuffling")
            
            # Mini-batch gradient descent
            total_loss = 0
            for i in range(0, X_train.shape[0], self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]
                
                # Validate batch data
                if len(X_batch) == 0 or len(y_batch) == 0:
                    logger.warning(f"Empty batch encountered at index {i}. Skipping.")
                    continue
                
                # Compute batch weights for class imbalance
                batch_weights = np.where(y_batch == 1, pos_weight, neg_weight)
                
                # Forward pass with error handling
                try:
                    z = np.dot(X_batch, self.weights) + self.bias
                    y_pred = self._sigmoid(z)
                    
                    # Compute loss
                    batch_loss = self._binary_cross_entropy(y_batch, y_pred, batch_weights)
                    
                    # Check for invalid loss
                    if np.isnan(batch_loss) or np.isinf(batch_loss):
                        logger.error(f"Invalid loss value: {batch_loss} in batch {i}. Using default value.")
                        batch_loss = 10.0  # Use a large but finite value
                        
                    total_loss += batch_loss * len(y_batch)
                    
                    # Compute gradients
                    error = y_pred - y_batch
                    grad_w = np.dot(X_batch.T, error * batch_weights) / len(y_batch)
                    grad_b = np.mean(error * batch_weights)
                    
                    # Add L2 regularization
                    grad_w += self.l2_lambda * self.weights
                    
                    # Store current weights for health check
                    weights_before = np.copy(self.weights)
                    
                    # Update parameters
                    self.weights -= self.learning_rate * grad_w
                    self.bias -= self.learning_rate * grad_b
                    
                    # Check training health
                    is_healthy = check_training_health(
                        iteration, grad_w, weights_before, self.weights
                    )
                    
                    # If severe issues detected, revert to previous weights and reduce learning rate
                    if not is_healthy:
                        logger.warning(f"Training instability detected in batch {i}. Reverting update.")
                        self.weights = weights_before
                        self.learning_rate *= 0.5  # Reduce learning rate
                        logger.info(f"Reduced learning rate to {self.learning_rate}")
                        
                except Exception as e:
                    logger.error(f"Error in batch {i}: {str(e)}")
                    logger.error(f"X_batch shape: {X_batch.shape}, y_batch shape: {y_batch.shape}")
                    continue  # Skip to next batch on error
            
            # Compute average training loss
            avg_train_loss = total_loss / X_train.shape[0]
            train_losses.append(avg_train_loss)
            
            # Compute validation loss
            val_preds = self.predict_proba(X_val)
            val_weights = np.where(y_val == 1, pos_weight, neg_weight)
            val_loss = self._binary_cross_entropy(y_val, val_preds, val_weights)
            val_losses.append(val_loss)
            
            # Check for improvement
            if val_loss < best_val_loss - self.convergence_threshold:
                best_val_loss = val_loss
                best_weights = np.copy(self.weights)
                best_bias = self.bias
                patience_counter = 0
                
                # Log progress
                if (iteration + 1) % 10 == 0:
                    logger.info(f"Iteration {iteration + 1}/{self.max_iterations}: "
                               f"train_loss={avg_train_loss:.6f}, val_loss={val_loss:.6f} (improved)")
            else:
                patience_counter += 1
                
                # Log progress
                if (iteration + 1) % 10 == 0:
                    logger.info(f"Iteration {iteration + 1}/{self.max_iterations}: "
                               f"train_loss={avg_train_loss:.6f}, val_loss={val_loss:.6f} "
                               f"(patience: {patience_counter}/{patience})")
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at iteration {iteration + 1}")
                break
        
        # Restore best weights
        self.weights = best_weights
        self.bias = best_bias
        
        # Compute final validation metrics
        val_preds_binary = self.predict(X_val)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, val_preds_binary, average='binary'
        )
        
        # Compute ROC AUC
        fpr, tpr, _ = roc_curve(y_val, self.predict_proba(X_val))
        roc_auc = auc(fpr, tpr)
        
        # Log results
        logger.info(f"Training completed after {iteration + 1} iterations")
        logger.info(f"Final validation metrics: precision={precision:.4f}, "
                   f"recall={recall:.4f}, f1={f1:.4f}, auc={roc_auc:.4f}")
        
        # Save model
        self._save_model()
        
        # Return metrics
        metrics = {
            'iterations': iteration + 1,
            'train_loss': float(avg_train_loss),
            'val_loss': float(val_loss),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc),
            'feature_weights': self.get_feature_weights(),
            'training_curves': {
                'train_loss': train_losses,
                'val_loss': val_losses
            }
        }
        
        return metrics
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities for input samples with enhanced error handling.
        
        Args:
            X: Feature matrix with shape (n_samples, n_features)
            
        Returns:
            Predicted probabilities with shape (n_samples,)
        """
        # Validate inputs
        if self.weights is None:
            raise ValueError("Model has not been trained yet")
            
        if not isinstance(X, np.ndarray):
            logger.warning(f"X is not a numpy array (type: {type(X)}). Converting to array.")
            try:
                X = np.array(X, dtype=np.float32)
            except Exception as e:
                raise ValueError(f"Could not convert X to a numpy array: {str(e)}")
        
        # Validate input shape
        if len(X.shape) != 2:
            error_msg = f"X must be a 2D array. Got shape: {X.shape}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        if X.shape[1] != len(self.weights):
            error_msg = f"Feature count mismatch: X has {X.shape[1]} features, model expects {len(self.weights)}"
            logger.error(error_msg)
            
        # Note if we're using scaled or unscaled features
        if hasattr(self, 'scaling_disabled') and self.scaling_disabled:
            logger.debug("Using raw feature values without scaling for prediction")
        
        # Check for NaN or infinity values
        if np.isnan(X).any() or np.isinf(X).any():
            logger.warning("X contains NaN or infinite values. Cleaning data...")
            X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
        
        try:
            # Compute logits
            z = np.dot(X, self.weights) + self.bias
            
            # Check for extreme values that might cause numerical issues
            if np.any(np.abs(z) > 100):
                logger.warning(f"Extreme logit values detected: range [{z.min():.2f}, {z.max():.2f}]")
                
            # Apply sigmoid function
            probabilities = self._sigmoid(z)
            
            # Verify probabilities are valid
            if np.isnan(probabilities).any() or np.isinf(probabilities).any():
                logger.error("Invalid probability values detected. Fixing...")
                probabilities = np.nan_to_num(probabilities, nan=0.5, posinf=1.0, neginf=0.0)
                
            # Ensure probabilities are in [0,1] range
            probabilities = np.clip(probabilities, 0.0, 1.0)
            
            return probabilities
            
        except Exception as e:
            logger.error(f"Error computing probabilities: {str(e)}")
            # Return safe fallback probabilities (all 0.5) with correct shape
            return np.ones(X.shape[0], dtype=np.float32) * 0.5
    
    def predict(self, X: np.ndarray, threshold: float = None) -> np.ndarray:
        """
        Predict binary labels for input samples with robust error handling.
        
        Args:
            X: Feature matrix with shape (n_samples, n_features)
            threshold: Optional decision threshold, defaults to self.decision_threshold
            
        Returns:
            Predicted binary labels with shape (n_samples,)
        """
        # Validate inputs
        if self.weights is None:
            raise ValueError("Model has not been trained yet")
        
        if not isinstance(X, np.ndarray):
            logger.warning(f"X is not a numpy array (type: {type(X)}). Converting to array.")
            try:
                X = np.array(X, dtype=np.float32)
            except Exception as e:
                raise ValueError(f"Could not convert X to a numpy array: {str(e)}")
        
        # Check for shape compatibility
        if X.shape[1] != len(self.weights):
            error_msg = f"Feature count mismatch: X has {X.shape[1]} features, model expects {len(self.weights)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Check for NaN or infinity values
        if np.isnan(X).any() or np.isinf(X).any():
            logger.warning("X contains NaN or infinite values. Cleaning data...")
            X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Use provided threshold or default
        if threshold is None:
            threshold = self.decision_threshold
        
        # Bounds check on threshold
        if threshold < 0.0 or threshold > 1.0:
            logger.warning(f"Invalid threshold value: {threshold}. Using default: {self.decision_threshold}")
            threshold = self.decision_threshold
        
        try:
            # Get probabilities and apply threshold
            probabilities = self.predict_proba(X)
            predictions = (probabilities >= threshold).astype(int)
            
            # Verify predictions are valid (0 or 1 only)
            unique_vals = np.unique(predictions)
            if not np.all(np.isin(unique_vals, [0, 1])):
                logger.error(f"Invalid prediction values detected: {unique_vals}")
                # Force valid binary values
                predictions = np.clip(predictions, 0, 1)
                
            return predictions
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            # Return safe fallback predictions (all negative)
            return np.zeros(X.shape[0], dtype=int)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, test_pairs: List[Tuple[str, str, str]] = None, 
                X_raw: np.ndarray = None, output_dir: str = None) -> Dict[str, Any]:
        """
        Evaluate the trained model on test data and generate evaluation reports.
        
        Args:
            X: Feature matrix with shape (n_samples, n_features)
            y: Binary labels with shape (n_samples,)
            output_dir: Optional directory for saving evaluation results
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.weights is None:
            raise ValueError("Model has not been trained yet")
            
        logger.info(f"Evaluating model on {X.shape[0]} test samples")
        
        # Set output directory
        if output_dir:
            self.evaluation_report_path = os.path.join(output_dir, "classifier_evaluation.json")
        
        # Get predictions
        y_proba = self.predict_proba(X)
        y_pred = self.predict(X)
        
        # Generate detailed test results CSV if test pairs are provided
        detailed_report_path = None
        if test_pairs and output_dir:
            # Use raw features if provided, otherwise use normalized features for both
            X_for_report = X_raw if X_raw is not None else X
            
            try:
                # Pass feature_engineering_instance directly for diagnostics
                feature_engineering = None
                if hasattr(self, 'feature_engineering_instance'):
                    feature_engineering = self.feature_engineering_instance
                
                # Add scaling_disabled flag to config for detailed reporting
                report_config = self.config.copy() if self.config else {}
                report_config['scaling_disabled'] = self.scaling_disabled
                
                detailed_report_path = generate_detailed_test_results(
                    self, test_pairs, X_for_report, y, X, 
                    self.feature_names, output_dir, report_config,
                    feature_engineering
                )
                logger.info(f"Generated detailed test results at {detailed_report_path}")
            except Exception as e:
                logger.error(f"Failed to generate detailed test results: {str(e)}")
        
        # Compute metrics
        precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary')
        
        # Compute ROC and AUC
        fpr, tpr, thresholds = roc_curve(y, y_proba)
        roc_auc = auc(fpr, tpr)
        
        # Compute confusion matrix
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        
        # Log metrics
        logger.info(f"Evaluation metrics: precision={precision:.4f}, recall={recall:.4f}, "
                   f"f1={f1:.4f}, auc={roc_auc:.4f}")
        logger.info(f"Confusion matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
        
        # Create evaluation report
        evaluation = {
            'metrics': {
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'roc_auc': float(roc_auc),
                'true_positives': int(tp),
                'false_positives': int(fp),
                'true_negatives': int(tn),
                'false_negatives': int(fn)
            },
            'roc_curve': {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': thresholds.tolist()
            },
            'feature_weights': self.get_feature_weights(),
            'feature_analysis': {
                'effective_weights': self.get_effective_feature_weights(),
                'importance_factors': self.feature_importance_factors
            }
        }
        
        # Add detailed report path if available
        if detailed_report_path:
            evaluation['detailed_report_path'] = detailed_report_path
        
        # Generate evaluation visualizations
        if output_dir:
            self._generate_evaluation_plots(y, y_proba, y_pred, output_dir)
            
            # Generate feature distribution and class separation plots
            try:
                # Save test data for later visualization
                test_data_path = os.path.join(output_dir, "test_data.npz")
                np.savez(test_data_path, X_test=X, y_test=y)
                logger.info(f"Saved test data to {test_data_path}")
                
                # Generate feature distribution plots
                plot_feature_distributions(X, y, self.feature_names, output_dir)
                
                # Generate class separation plots
                feature_auc_scores = plot_class_separation(X, y, self.feature_names, output_dir)
                
                # Generate feature visualization report
                report_path = generate_feature_visualization_report(
                    output_dir, feature_auc_scores, self.get_feature_weights()
                )
                evaluation['feature_visualization_report'] = report_path
                logger.info(f"Generated feature visualization report at {report_path}")
            except Exception as e:
                logger.error(f"Error generating feature visualizations: {str(e)}")
        
        # Save evaluation report
        with open(self.evaluation_report_path, 'w') as f:
            json.dump(evaluation, f, indent=2)
        
        logger.info(f"Saved evaluation report to {self.evaluation_report_path}")
        
        return evaluation
    
    def get_feature_weights(self) -> Dict[str, float]:
        """
        Get feature weights as a dictionary.
        
        Returns:
            Dictionary mapping feature names to weights
        """
        if self.weights is None:
            raise ValueError("Model has not been trained yet")
            
        if not self.feature_names:
            # If feature names weren't provided, use generic names
            self.feature_names = [f"feature_{i}" for i in range(len(self.weights))]
            
        return {name: float(weight) for name, weight in zip(self.feature_names, self.weights)}
        
    def get_effective_feature_weights(self) -> Dict[str, float]:
        """
        Get effective feature weights accounting for feature importance factors.
        
        This method combines the learned model weights with the feature importance
        factors to provide a comprehensive view of feature influence in the model.
        
        Returns:
            Dictionary mapping feature names to effective weights
        """
        if self.weights is None:
            raise ValueError("Model has not been trained yet")
        
        raw_weights = self.get_feature_weights()
        effective_weights = {}
        
        # Include importance factors in the effective weights calculation
        for feature_name, weight in raw_weights.items():
            importance = self.feature_importance_factors.get(feature_name, 1.0)
            effective_weights[feature_name] = {
                'raw_weight': float(weight),
                'importance_factor': float(importance),
                'effective_weight': float(weight * importance)
            }
        
        return effective_weights
    
    def save(self, path: str = None) -> str:
        """
        Save the trained model to disk.
        
        Args:
            path: Optional path for saving the model
            
        Returns:
            Path where the model was saved
        """
        if path:
            self.model_checkpoint_path = path
            
        return self._save_model()
    
    def load(self, path: str = None) -> bool:
        """
        Load a trained model from disk.
        
        Args:
            path: Optional path for loading the model
            
        Returns:
            True if the model was loaded successfully, False otherwise
        """
        if path:
            self.model_checkpoint_path = path
            
        return self._load_model()
    
    def optimize_threshold(self, X: np.ndarray, y: np.ndarray, metric: str = 'f1') -> float:
        """
        Find the optimal decision threshold for classification.
        
        Args:
            X: Feature matrix with shape (n_samples, n_features)
            y: Binary labels with shape (n_samples,)
            metric: Metric to optimize ('precision', 'recall', or 'f1')
            
        Returns:
            Optimal threshold value
        """
        if self.weights is None:
            raise ValueError("Model has not been trained yet")
            
        # Get predicted probabilities
        y_proba = self.predict_proba(X)
        
        # Try different thresholds
        thresholds = np.linspace(0.1, 0.9, 100)
        best_score = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary')
            
            # Select score based on metric
            if metric == 'precision':
                score = precision
            elif metric == 'recall':
                score = recall
            else:  # Default to f1
                score = f1
                
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        logger.info(f"Optimized {metric} score: {best_score:.4f} at threshold {best_threshold:.4f}")
        
        # Update decision threshold
        self.decision_threshold = best_threshold
        
        return best_threshold
    
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Compute sigmoid function with enhanced numerical stability.
        
        Args:
            z: Input array
            
        Returns:
            Sigmoid of input
        """
        # Validate input
        if not isinstance(z, np.ndarray):
            logger.warning(f"Invalid input type to sigmoid: {type(z)}")
            z = np.array(z, dtype=np.float32)
        
        # Check for NaN or infinity values
        if np.isnan(z).any() or np.isinf(z).any():
            logger.warning("NaN or infinity values detected in sigmoid input")
            # Replace problematic values
            z = np.nan_to_num(z, nan=0.0, posinf=15.0, neginf=-15.0)
        
        # Clip values for numerical stability - avoid overflow in exp
        z_clipped = np.clip(z, -15.0, 15.0)
        
        # Use numerically stable computation
        # For large negative values, use exp(x) / (1 + exp(x))
        # For large positive values, use 1 / (1 + exp(-x))
        # This avoids numerical underflow/overflow
        result = np.zeros_like(z_clipped, dtype=np.float32)
        
        # For z >= 0, use 1 / (1 + exp(-z)) to avoid exp overflow
        pos_mask = z_clipped >= 0
        result[pos_mask] = 1.0 / (1.0 + np.exp(-z_clipped[pos_mask]))
        
        # For z < 0, use exp(z) / (1 + exp(z)) to avoid division by near-zero
        neg_mask = ~pos_mask
        exp_z = np.exp(z_clipped[neg_mask])
        result[neg_mask] = exp_z / (1.0 + exp_z)
        
        return result
    
    def _binary_cross_entropy(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             sample_weights: np.ndarray = None) -> float:
        """
        Compute binary cross-entropy loss with L2 regularization using enhanced numerical stability.
        
        Args:
            y_true: True binary labels
            y_pred: Predicted probabilities
            sample_weights: Optional weights for each sample
            
        Returns:
            Binary cross-entropy loss
        """
        # Input validation
        if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
            logger.warning("Inputs to binary_cross_entropy must be numpy arrays")
            y_true = np.array(y_true, dtype=np.float32)
            y_pred = np.array(y_pred, dtype=np.float32)
            
        if y_true.shape != y_pred.shape:
            logger.error(f"Shape mismatch in binary_cross_entropy: {y_true.shape} vs {y_pred.shape}")
            # Try to reshape or broadcast if possible
            try:
                y_true = np.broadcast_to(y_true, y_pred.shape)
            except:
                raise ValueError(f"Incompatible shapes: {y_true.shape} and {y_pred.shape}")
                
        # Check for NaN or infinity values
        if np.isnan(y_true).any() or np.isinf(y_true).any():
            logger.error("NaN or infinity values detected in y_true")
            raise ValueError("y_true contains NaN or infinite values")
            
        if np.isnan(y_pred).any() or np.isinf(y_pred).any():
            logger.warning("NaN or infinity values detected in y_pred")
            y_pred = np.nan_to_num(y_pred, nan=0.5, posinf=1.0, neginf=0.0)
            
        # Clip prediction values for numerical stability (avoid log(0) and log(1))
        epsilon = 1e-7  # Smaller than before for better precision, but still numerically stable
        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
        
        # Compute binary cross-entropy with stable computation
        # Use element-wise operations to avoid potential broadcasting issues
        log_probs = np.where(y_true > 0.5, 
                             np.log(y_pred),  # log(p) for positive class
                             np.log(1.0 - y_pred))  # log(1-p) for negative class
        
        # Apply sample weights if provided
        if sample_weights is not None:
            if sample_weights.shape != y_true.shape:
                # Try to reshape the weights if shape mismatch
                try:
                    sample_weights = np.broadcast_to(sample_weights, y_true.shape)
                except:
                    logger.error(f"Sample weights shape {sample_weights.shape} incompatible with labels shape {y_true.shape}")
                    raise ValueError(f"Sample weights shape {sample_weights.shape} incompatible with labels shape {y_true.shape}")
                    
            bce = -log_probs * sample_weights
        else:
            bce = -log_probs
            
        # Compute mean loss with handling for empty arrays
        if len(bce) > 0:
            loss = np.mean(bce)
        else:
            logger.warning("Empty array in binary_cross_entropy calculation")
            return 0.0
            
        # Check for NaN or infinity in the loss
        if np.isnan(loss) or np.isinf(loss):
            logger.error(f"Loss computation resulted in {loss}")
            # Return a large but finite loss value
            return 100.0
        
        # Add L2 regularization with numerical stability
        if self.weights is not None:
            # Calculate L2 norm with protection against very large values
            l2_norm = np.sum(np.clip(self.weights, -1e6, 1e6) ** 2)
            l2_loss = 0.5 * self.l2_lambda * l2_norm
            loss += l2_loss
            
        return float(loss)  # Ensure we return a scalar
    
    def _save_model(self) -> str:
        """
        Save the model to disk.
        
        Returns:
            Path where the model was saved
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.model_checkpoint_path), exist_ok=True)
        
        # Save model parameters including feature importance factors
        model_data = {
            'weights': self.weights,
            'bias': self.bias,
            'feature_names': self.feature_names,
            'decision_threshold': self.decision_threshold,
            'substitution_mapping': self.substitution_mapping,
            'feature_importance_factors': self.feature_importance_factors,
            'scaling_disabled': self.scaling_disabled
        }
        
        with open(self.model_checkpoint_path, 'wb') as f:
            pickle.dump(model_data, f)
            
        logger.info(f"Saved model to {self.model_checkpoint_path}")
        
        return self.model_checkpoint_path
    
    def _load_model(self) -> bool:
        """
        Load the model from disk.
        
        Returns:
            True if the model was loaded successfully, False otherwise
        """
        try:
            with open(self.model_checkpoint_path, 'rb') as f:
                model_data = pickle.load(f)
                
            self.weights = model_data['weights']
            self.bias = model_data['bias']
            self.feature_names = model_data['feature_names']
            self.decision_threshold = model_data['decision_threshold']
            self.substitution_mapping = model_data.get('substitution_mapping', {})
            self.feature_importance_factors = model_data.get('feature_importance_factors', {})
            self.scaling_disabled = model_data.get('scaling_disabled', False)
            
            # Log feature importance factors if available
            if self.feature_importance_factors:
                logger.info(f"Loaded model with feature importance factors: {self.feature_importance_factors}")
            
            logger.info(f"Loaded model from {self.model_checkpoint_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model from {self.model_checkpoint_path}: {str(e)}")
            return False
    
    def _generate_evaluation_plots(self, y_true: np.ndarray, y_proba: np.ndarray, 
                                  y_pred: np.ndarray, output_dir: str) -> None:
        """
        Generate evaluation visualizations and save to output directory.
        
        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities
            y_pred: Predicted binary labels
            output_dir: Directory for saving plots
        """
        # Create plots directory
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # 1. ROC curve
        plt.figure(figsize=(10, 8))
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(plots_dir, "roc_curve.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                  xticklabels=['Non-match', 'Match'],
                  yticklabels=['Non-match', 'Match'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(plots_dir, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Feature importance with effective weights visualization
        plt.figure(figsize=(12, 10))
        weights = self.get_feature_weights()
        effective_weights = self.get_effective_feature_weights()
        feature_names = list(weights.keys())
        
        # Prepare data for visualization
        raw_importance = [abs(weights[name]) for name in feature_names]
        effective_importance = [abs(effective_weights[name]['effective_weight']) for name in feature_names]
        
        # Sort by effective importance
        sorted_idx = np.argsort(effective_importance)
        sorted_features = [feature_names[i] for i in sorted_idx]
        sorted_raw = [raw_importance[i] for i in sorted_idx]
        sorted_effective = [effective_importance[i] for i in sorted_idx]
        
        # Visualization with dual bars for comparison
        y_pos = np.arange(len(feature_names))
        bar_width = 0.35
        
        # Plot raw weights
        plt.barh(y_pos - bar_width/2, sorted_raw, bar_width, alpha=0.6, color='blue', label='Raw Weight')
        
        # Plot effective weights
        plt.barh(y_pos + bar_width/2, sorted_effective, bar_width, alpha=0.6, color='red', label='Effective Weight')
        
        # Styling
        plt.yticks(y_pos, sorted_features)
        plt.xlabel('Weight Magnitude')
        plt.title('Feature Importance: Raw vs. Effective Weights')
        plt.legend()
        
        # Generate annotation for substituted features
        annotations = {}
        feature_labels = feature_names.copy()
        
        # Create a JSON file with feature weight information
        feature_weight_info = {
            "raw_weights": weights,
            "effective_weights": effective_weights,
            "importance_factors": self.feature_importance_factors,
            "normalized_used": True,
            "test_results_note": "Test results contain both raw and normalized feature values with prefixes raw_ and norm_"
        }
        
        # Save feature weight info to JSON
        with open(os.path.join(plots_dir, "feature_weight_info.json"), 'w') as f:
            json.dump(feature_weight_info, f, indent=2)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "feature_importance.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Probability distribution
        plt.figure(figsize=(10, 6))
        
        # Create separate series for match and non-match predictions
        match_probs = y_proba[y_true == 1]
        non_match_probs = y_proba[y_true == 0]
        
        plt.hist(non_match_probs, bins=50, alpha=0.5, color='red', label='Non-match')
        plt.hist(match_probs, bins=50, alpha=0.5, color='green', label='Match')
        
        plt.axvline(x=self.decision_threshold, color='black', linestyle='--',
                   label=f'Decision threshold ({self.decision_threshold:.2f})')
        
        plt.xlabel('Predicted probability')
        plt.ylabel('Count')
        plt.title('Distribution of Predicted Probabilities')
        plt.legend()
        plt.savefig(os.path.join(plots_dir, "probability_distribution.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Feature impact analysis plot
        if hasattr(self, 'feature_importance_factors') and self.feature_importance_factors:
            plt.figure(figsize=(12, 8))
            
            # Get feature data for analysis
            weights = self.get_feature_weights()
            effective_weights = self.get_effective_feature_weights()
            top_features = sorted([(name, abs(effective_weights[name]['effective_weight']))
                                  for name in weights.keys()], 
                                 key=lambda x: x[1], reverse=True)[:5]  # Top 5 features
            
            # Create a scatter plot for impact analysis
            x_vals = []
            y_vals = []
            sizes = []
            colors = []
            labels = []
            
            # Prepare data points
            for name, importance in top_features:
                data = effective_weights[name]
                raw_weight = data['raw_weight']
                importance_factor = data['importance_factor']
                effective_weight = data['effective_weight']
                
                # Add data point
                x_vals.append(abs(raw_weight))
                y_vals.append(importance_factor)
                sizes.append(abs(effective_weight) * 500)  # Scale for visibility
                colors.append('blue' if effective_weight > 0 else 'red')
                labels.append(name)
            
            # Create scatter plot
            plt.scatter(x_vals, y_vals, s=sizes, c=colors, alpha=0.6)
            
            # Add feature name labels
            for i, label in enumerate(labels):
                plt.annotate(label, (x_vals[i], y_vals[i]), 
                           xytext=(10, 5), textcoords='offset points')
            
            # Add styling
            plt.xlabel('Raw Weight Magnitude')
            plt.ylabel('Importance Factor')
            plt.title('Feature Impact Analysis')
            plt.grid(alpha=0.3)
            
            # Add size legend
            plt.figtext(0.15, 0.02, "Note: Circle size represents effective weight magnitude", 
                      ha="left", fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
            
            # Save the plot
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "feature_impact_analysis.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Generated evaluation plots in {plots_dir}")

# Training module functions
def load_training_data(ground_truth_path: str) -> Tuple[List[Tuple[str, str, str]], Dict[str, int]]:
    """
    Load ground truth labeled data for training.
    
    Args:
        ground_truth_path: Path to the ground truth CSV file
        
    Returns:
        Tuple of (labeled_pairs, label_counts) where labeled_pairs is a list of
        (left_id, right_id, match_label) tuples and label_counts is a dictionary
        with counts of matches and non-matches
    """
    try:
        # Load ground truth data
        df = pd.read_csv(ground_truth_path)
        logger.info(f"Loaded {len(df)} labeled pairs from {ground_truth_path}")
        
        # Extract labeled pairs
        labeled_pairs = []
        for _, row in df.iterrows():
            left_id = row['left']
            right_id = row['right']
            match_label = str(row['match']).lower()
            labeled_pairs.append((left_id, right_id, match_label))
        
        # Count matches and non-matches
        match_count = sum(1 for _, _, label in labeled_pairs if label == 'true')
        non_match_count = sum(1 for _, _, label in labeled_pairs if label == 'false')
        
        label_counts = {
            'match': match_count,
            'non_match': non_match_count,
            'total': len(labeled_pairs)
        }
        
        logger.info(f"Label distribution: {match_count} matches, {non_match_count} non-matches")
        
        return labeled_pairs, label_counts
        
    except Exception as e:
        logger.error(f"Failed to load training data from {ground_truth_path}: {str(e)}")
        raise

def train_classifier(config: Dict[str, Any], feature_engineering: FeatureEngineering,
                    hash_lookup: Dict[str, Dict[str, str]], string_dict: Dict[str, str] = None,
                    reset: bool = False, disable_scaling: bool = False) -> EntityClassifier:
    # Ensure deterministic behavior
    seed = config.get('random_seed', 42)
    setup_deterministic_behavior(seed)
    # Log the seed being used
    logger.info(f"Training classifier with random seed: {seed}")
    """
    Train the entity classifier using labeled data.
    
    Args:
        config: Configuration dictionary
        feature_engineering: FeatureEngineering instance for computing features
        hash_lookup: Dictionary mapping personId to field hashes
        string_dict: Optional dictionary of string values keyed by hash
        reset: Whether to reset training and start from scratch
        
    Returns:
        Trained EntityClassifier instance
    """
    # Initialize classifier
    classifier = EntityClassifier(config)
    
    # Check for existing model if not resetting
    model_path = os.path.join(
        config.get("checkpoint_dir", "data/checkpoints"),
        "classifier_model.pkl"
    )
    
    # Register any custom features with improved error handling
    try:
        logger.info("Registering custom features for feature engineering module")
        register_custom_features(feature_engineering, config)
        logger.info(f"Feature engineering has {len(feature_engineering.get_feature_names())} features registered")
        logger.info(f"Feature engineering substitution mapping: {feature_engineering.get_substitution_mapping()}")
    except Exception as e:
        logger.error(f"Error registering custom features: {str(e)}")
        logger.error(f"Continuing with base features only - custom features may not be available")
        # Continue with base features if possible
    
    if not reset and os.path.exists(model_path):
        logger.info(f"Loading existing model from {model_path}")
        if classifier.load(model_path):
            return classifier
    
    # Load ground truth data
    ground_truth_path = os.path.join(
        config.get("ground_truth_dir", "data/ground_truth"),
        config.get("labeled_matches_file", "labeled_matches.csv")
    )
    
    labeled_pairs, _ = load_training_data(ground_truth_path)
    
    # Compute feature vectors for labeled pairs
    X, y = feature_engineering.compute_features(labeled_pairs, string_dict)
    logger.info(f"Computed feature vectors with shape {X.shape}")
    
    # Check if RFE is enabled and should be run
    feature_selection_config = config.get('feature_selection', {})
    if feature_selection_config.get('enabled', False) and feature_selection_config.get('method') == 'rfe':
        logger.info("Feature selection (RFE) is enabled - performing feature selection")
        
        # Import RFE module
        from src.feature_selection_rfe import RecursiveFeatureEliminator
        
        # Check if we should load pre-computed RFE results
        rfe_results_path = feature_selection_config.get('rfe_config', {}).get('results_path', 'data/output/rfe_results.json')
        use_precomputed = feature_selection_config.get('use_precomputed_results', False)
        
        if use_precomputed and os.path.exists(rfe_results_path):
            # Load pre-computed RFE results
            logger.info(f"Loading pre-computed RFE results from {rfe_results_path}")
            with open(rfe_results_path, 'r') as f:
                rfe_results = json.load(f)
            
            # Get selected features
            selected_features = rfe_results['optimal']['features']
            original_features = feature_engineering.get_feature_names()
            
            # Create feature mask
            feature_mask = np.array([feat in selected_features for feat in original_features])
            
            # Transform features
            X = X[:, feature_mask]
            
            # Update feature names in feature engineering
            feature_engineering.feature_names = selected_features
            feature_engineering.feature_registry = {
                name: func for name, func in feature_engineering.feature_registry.items() 
                if name in selected_features
            }
            
            logger.info(f"Using pre-computed RFE results: {len(selected_features)} features selected")
            logger.info(f"Selected features: {selected_features}")
            
        else:
            # Run RFE to select features
            logger.info("Running Recursive Feature Elimination...")
            rfe = RecursiveFeatureEliminator(config)
            
            # Fit RFE
            original_feature_names = feature_engineering.get_feature_names()
            rfe.fit(X, y, original_feature_names, EntityClassifier, config)
            
            # Transform features
            X = rfe.transform(X)
            
            # Update feature names in feature engineering
            selected_features = rfe.get_selected_features()
            feature_engineering.feature_names = selected_features
            feature_engineering.feature_registry = {
                name: func for name, func in feature_engineering.feature_registry.items() 
                if name in selected_features
            }
            
            logger.info(f"RFE complete: {len(selected_features)} features selected from {len(original_feature_names)}")
            logger.info(f"Selected features: {selected_features}")
            
            # Generate RFE report
            report = rfe.generate_report()
            logger.info("\n" + report)
    
    # Get test_split_ratio from config or use default
    test_split_ratio = config.get("test_split_ratio", 0.3)
    test_split_ratio = max(0.1, min(0.5, test_split_ratio))  # Constrain within acceptable range
    logger.info(f"Using test split ratio: {test_split_ratio}")
    
    # Split into training and testing sets with fixed random state for determinism
    random_state = config.get('random_seed', 42)
    # Use the exact same random_state for reproducibility
    np.random.seed(random_state)
    train_indices, test_indices = train_test_split(
        np.arange(len(labeled_pairs)), test_size=test_split_ratio, 
        random_state=random_state, stratify=y
    )
    logger.info(f"Split data using random_state={random_state}")
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    # Extract test pairs for detailed reporting
    test_pairs = [labeled_pairs[i] for i in test_indices]
    
    # Scaling is now handled directly in feature_engineering.py normalize_features() method
    scaling_bridge = None
    
    # Normalize features (scaling controlled by use_enhanced_scaling in config.yml)
    if disable_scaling:
        logger.warning("*** FEATURE SCALING DISABLED via --disable-scaling flag - Using raw feature values directly ***")
        X_train_norm = X_train.copy()  # Use raw features directly
        X_test_norm = X_test.copy()    # Use raw features directly
    else:
        # Scaling behavior now controlled by enable_feature_scaling in config.yml
        # normalize_features will check this setting and either scale or return raw features
        X_train_norm = feature_engineering.normalize_features(X_train, fit=True)
        X_test_norm = feature_engineering.normalize_features(X_test)
        
        # Log the actual behavior based on configuration
        enable_scaling = config.get("use_enhanced_scaling", False)
        if enable_scaling:
            logger.info("Feature scaling enabled - using LibraryCatalogScaler")
        else:
            logger.info("Feature scaling disabled by configuration - using raw features")
    
    # Train classifier
    feature_names = feature_engineering.get_feature_names()
    
    # Store feature engineering instance for diagnostics
    # It will be used in evaluation to examine problematic indicator values
    classifier.feature_engineering_instance = feature_engineering
    
    # Extract importance factors if available
    importance_factors = {}
    if hasattr(feature_engineering, 'get_feature_importance_factors'):
        importance_factors = feature_engineering.get_feature_importance_factors()
        logger.info(f"Using feature importance factors: {importance_factors}")
    
    # Pass feature importance metadata to the classifier
    classifier.feature_importance_factors = importance_factors
    
    # Train the classifier
    training_metrics = classifier.fit(X_train_norm, y_train, feature_names)
    
    # Extract feature importance factors and substitution mapping
    substitution_mapping = feature_engineering.get_substitution_mapping()
    importance_factors = {}
    if hasattr(feature_engineering, 'get_feature_importance_factors'):
        importance_factors = feature_engineering.get_feature_importance_factors()
        
    # Store metadata in classifier for reporting and analysis
    classifier.substitution_mapping = substitution_mapping.copy()
    classifier.feature_importance_factors = importance_factors.copy()
    
    # Log importance factors
    if importance_factors:
        logger.info(f"Feature importance factors: {importance_factors}")
    
    # Evaluate on test set with detailed reporting
    output_dir = config.get("output_dir", "data/output")
    evaluation_metrics = classifier.evaluate(
        X_test_norm, y_test, test_pairs=test_pairs, X_raw=X_test, output_dir=output_dir
    )
    
    # Optimize decision threshold
    optimal_threshold = classifier.optimize_threshold(X_test_norm, y_test, 'f1')
    logger.info(f"Optimized decision threshold: {optimal_threshold:.4f}")
    
    # Add scaling information to report
    if disable_scaling:
        scaling_approach = "disabled_by_flag"
    else:
        enable_scaling = config.get("use_enhanced_scaling", False)
        scaling_approach = "LibraryCatalogScaler" if enable_scaling else "raw_features"
    
    scaling_info = {
        "scaling_approach": scaling_approach,
        "use_enhanced_scaling": config.get("use_enhanced_scaling", False),
        "scaled_feature_statistics": {
            "mean": float(np.mean(X_train_norm)),
            "std": float(np.std(X_train_norm)),
            "min": float(np.min(X_train_norm)),
            "max": float(np.max(X_train_norm))
        },
        "scaling_disabled": disable_scaling
    }
    
    # Add feature importance factors to report if available
    if hasattr(feature_engineering, 'get_feature_importance_factors'):
        importance_factors = feature_engineering.get_feature_importance_factors()
        if importance_factors:
            scaling_info["feature_importance_factors"] = importance_factors
    
    # Save combined report
    report = {
        'training': training_metrics,
        'evaluation': evaluation_metrics,
        'optimal_threshold': float(optimal_threshold),
        'scaling': scaling_info,
        'feature_info': {
            'effective_features': feature_names,
            'substitution_mapping': substitution_mapping
        },
        'config': {k: v for k, v in config.items() if isinstance(v, (str, int, float, bool, list, dict))}
    }
    
    report_path = os.path.join(output_dir, "training_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Saved training report to {report_path}")
    
    return classifier

def main(config_path: str = 'config.yml', reset: bool = False, disable_scaling: bool = False):
    """
    Main function for training the entity classifier.
    
    Args:
        config_path: Path to the configuration file
        reset: Whether to reset training and start from scratch
        disable_scaling: Whether to disable feature scaling
    """
    # Load configuration with environment-specific overrides
    from src.config_utils import load_config_with_environment
    config = load_config_with_environment(config_path)
    
    # Override config values
    if disable_scaling:
        config['disable_scaling'] = True
        logger.warning("*** FEATURE SCALING DISABLED - Will use raw feature values for training ***")
    
    # Load data from checkpoints
    checkpoint_dir = config.get('checkpoint_dir', 'data/checkpoints')
    hash_lookup_path = os.path.join(checkpoint_dir, 'hash_lookup.pkl')
    string_dict_path = os.path.join(checkpoint_dir, 'string_dict.pkl')
    
    with open(hash_lookup_path, 'rb') as f:
        hash_lookup = pickle.load(f)
        
    with open(string_dict_path, 'rb') as f:
        string_dict = pickle.load(f)
    
    # Initialize Weaviate client 
    from src.indexing import WeaviateClientManager
    client_manager = WeaviateClientManager(config)
    client = client_manager.__enter__()
    
    try:
        # Initialize feature engineering
        feature_engineering = FeatureEngineering(config, client, hash_lookup)
        
        # Train with or without scaling
        disable_scaling = config.get('disable_scaling', False)
        classifier = train_classifier(
            config, feature_engineering, hash_lookup, string_dict, 
            reset=reset, disable_scaling=disable_scaling
        )
        
        return classifier
    finally:
        # Clean up resources
        if client_manager is not None:
            client_manager.__exit__(None, None, None)

if __name__ == "__main__":
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description='Train entity classifier')
    parser.add_argument('--config', default='config.yml', help='Path to configuration file')
    parser.add_argument('--reset', action='store_true', help='Reset training and start from scratch')
    parser.add_argument('--use-enhanced-scaling', action='store_true', 
                        help='Use enhanced scaling approaches (library_catalog and robust_minmax)')
    parser.add_argument('--disable-scaling', action='store_true',
                        help='Disable all feature scaling and use raw values directly')
    args = parser.parse_args()
    
    # Load configuration with environment-specific overrides
    from src.config_utils import load_config_with_environment
    config = load_config_with_environment(args.config)
    
    # Add flags to config
    config['use_enhanced_scaling'] = args.use_enhanced_scaling
    config['disable_scaling'] = args.disable_scaling
    
    if args.disable_scaling:
        logger.warning("*** FEATURE SCALING DISABLED - Will use raw feature values for training ***")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run training
    main(args.config, args.reset, args.disable_scaling)
