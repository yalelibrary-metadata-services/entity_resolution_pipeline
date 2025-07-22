"""
Validation Metrics Collector Module

This module provides comprehensive metrics collection and analysis capabilities
for feature validation, including advanced statistical measures, learning curves,
feature stability analysis, and confidence interval estimation.

Classes:
    ValidationMetricsCollector: Main class for collecting validation metrics
    LearningCurveAnalysis: Learning curve analysis and convergence diagnostics
    FeatureStabilityAnalysis: Feature stability across cross-validation folds
    ConfidenceIntervalEstimator: Bootstrap and parametric CI estimation
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import json
from datetime import datetime

# Scientific computing and statistics
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.metrics import (
    precision_recall_fscore_support, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    brier_score_loss, log_loss
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import learning_curve, validation_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Local imports
from src.training import EntityClassifier

logger = logging.getLogger(__name__)

@dataclass
class AdvancedMetrics:
    """Advanced performance metrics beyond basic precision/recall/F1."""
    # Basic metrics
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    
    # Additional classification metrics
    average_precision: float  # Area under PR curve
    balanced_accuracy: float
    matthews_correlation: float
    specificity: float
    negative_predictive_value: float
    false_positive_rate: float
    false_negative_rate: float
    
    # Probabilistic metrics
    brier_score: float
    log_loss: float
    calibration_error: float
    
    # Threshold analysis
    optimal_threshold: float
    threshold_metrics: Dict[str, List[float]]  # Metrics at different thresholds
    
    # Distribution metrics
    prediction_entropy: float
    confidence_distribution: Dict[str, float]

@dataclass
class LearningCurveAnalysis:
    """Results from learning curve analysis."""
    train_sizes: np.ndarray
    train_scores_mean: np.ndarray
    train_scores_std: np.ndarray
    validation_scores_mean: np.ndarray
    validation_scores_std: np.ndarray
    convergence_score: float
    convergence_iteration: int
    overfitting_score: float
    learning_rate_estimate: float

@dataclass
class FeatureStabilityAnalysis:
    """Feature stability analysis across cross-validation folds."""
    feature_importance_by_fold: Dict[str, List[float]]
    stability_scores: Dict[str, float]  # Per-feature stability
    rank_consistency: Dict[str, float]  # Ranking consistency across folds
    variance_ratio: Dict[str, float]    # Variance to mean ratio
    overall_stability: float

@dataclass
class ValidationResults:
    """Comprehensive validation results."""
    feature_set: List[str]
    cv_folds: int
    advanced_metrics: AdvancedMetrics
    learning_curve_analysis: LearningCurveAnalysis
    feature_stability: FeatureStabilityAnalysis
    confidence_intervals: Dict[str, Tuple[float, float]]
    cross_validation_details: Dict[str, Any]
    computational_metrics: Dict[str, float]
    timestamp: datetime

class ValidationMetricsCollector:
    """
    Comprehensive metrics collection and analysis for feature validation.
    
    This class provides advanced metrics beyond basic precision/recall/F1,
    including learning curve analysis, feature stability assessment,
    confidence interval estimation, and calibration analysis.
    """
    
    def __init__(self, cv_folds: int = 5, random_seed: int = 42,
                 bootstrap_samples: int = 1000):
        """
        Initialize the validation metrics collector.
        
        Args:
            cv_folds: Number of cross-validation folds
            random_seed: Random seed for reproducibility
            bootstrap_samples: Number of bootstrap samples for CI estimation
        """
        self.cv_folds = cv_folds
        self.random_seed = random_seed
        self.bootstrap_samples = bootstrap_samples
        
        # Set random seed
        np.random.seed(random_seed)
        
        logger.info(f"Initialized ValidationMetricsCollector with {cv_folds} CV folds")
    
    def collect_comprehensive_metrics(self, X: np.ndarray, y: np.ndarray, 
                                    feature_names: List[str], 
                                    config: Dict[str, Any]) -> ValidationResults:
        """
        Collect comprehensive validation metrics for a feature set.
        
        Args:
            X: Feature matrix
            y: Labels
            feature_names: Feature names
            config: Configuration dictionary
            
        Returns:
            Comprehensive validation results
        """
        start_time = datetime.now()
        
        logger.info(f"Collecting comprehensive metrics for {len(feature_names)} features")
        
        # Basic cross-validation metrics
        cv_details = self._perform_detailed_cv(X, y, feature_names, config)
        
        # Advanced metrics calculation
        advanced_metrics = self._calculate_advanced_metrics(
            cv_details['y_true_all'], 
            cv_details['y_pred_all'], 
            cv_details['y_pred_proba_all']
        )
        
        # Learning curve analysis
        learning_analysis = self._analyze_learning_curves(X, y, feature_names, config)
        
        # Feature stability analysis
        stability_analysis = self._analyze_feature_stability(
            cv_details['feature_importance_by_fold'], feature_names
        )
        
        # Confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(cv_details)
        
        # Computational metrics
        end_time = datetime.now()
        computational_metrics = {
            'total_time_seconds': (end_time - start_time).total_seconds(),
            'time_per_fold': (end_time - start_time).total_seconds() / self.cv_folds,
            'memory_usage_mb': self._estimate_memory_usage(X),
            'feature_computation_time': cv_details.get('feature_computation_time', 0.0)
        }
        
        return ValidationResults(
            feature_set=feature_names.copy(),
            cv_folds=self.cv_folds,
            advanced_metrics=advanced_metrics,
            learning_curve_analysis=learning_analysis,
            feature_stability=stability_analysis,
            confidence_intervals=confidence_intervals,
            cross_validation_details=cv_details,
            computational_metrics=computational_metrics,
            timestamp=end_time
        )
    
    def _perform_detailed_cv(self, X: np.ndarray, y: np.ndarray, 
                           feature_names: List[str], 
                           config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform detailed cross-validation with comprehensive data collection."""
        from sklearn.model_selection import StratifiedKFold
        
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_seed)
        
        # Storage for detailed results
        fold_results = []
        y_true_all = []
        y_pred_all = []
        y_pred_proba_all = []
        feature_importance_by_fold = {feature: [] for feature in feature_names}
        training_times = []
        prediction_times = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            fold_start = datetime.now()
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train classifier
            classifier = EntityClassifier(config)
            train_start = datetime.now()
            classifier.fit(X_train, y_train, feature_names)
            train_time = (datetime.now() - train_start).total_seconds()
            
            # Predict on validation set
            pred_start = datetime.now()
            y_pred_proba = classifier.predict_proba(X_val)
            y_pred = classifier.predict(X_val)
            pred_time = (datetime.now() - pred_start).total_seconds()
            
            # Store results
            y_true_all.extend(y_val)
            y_pred_all.extend(y_pred)
            y_pred_proba_all.extend(y_pred_proba)
            
            # Feature importance
            if hasattr(classifier, 'weights') and classifier.weights is not None:
                for i, feature in enumerate(feature_names):
                    feature_importance_by_fold[feature].append(classifier.weights[i])
            
            # Fold-specific metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_val, y_pred, average='binary', zero_division=0
            )
            
            try:
                auc_score = roc_auc_score(y_val, y_pred_proba)
            except ValueError:
                auc_score = 0.5
            
            fold_result = {
                'fold': fold,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc_score,
                'train_size': len(X_train),
                'val_size': len(X_val),
                'train_time': train_time,
                'pred_time': pred_time,
                'fold_time': (datetime.now() - fold_start).total_seconds()
            }
            fold_results.append(fold_result)
            training_times.append(train_time)
            prediction_times.append(pred_time)
            
            logger.debug(f"Fold {fold}: F1={f1:.4f}, AUC={auc_score:.4f}")
        
        return {
            'fold_results': fold_results,
            'y_true_all': np.array(y_true_all),
            'y_pred_all': np.array(y_pred_all),
            'y_pred_proba_all': np.array(y_pred_proba_all),
            'feature_importance_by_fold': feature_importance_by_fold,
            'training_times': training_times,
            'prediction_times': prediction_times,
            'avg_train_time': np.mean(training_times),
            'avg_pred_time': np.mean(prediction_times)
        }
    
    def _calculate_advanced_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  y_pred_proba: np.ndarray) -> AdvancedMetrics:
        """Calculate advanced performance metrics."""
        # Basic metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        
        try:
            roc_auc = roc_auc_score(y_true, y_pred_proba)
            avg_precision = average_precision_score(y_true, y_pred_proba)
        except ValueError:
            roc_auc = 0.5
            avg_precision = 0.5
        
        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Additional classification metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        balanced_accuracy = (recall + specificity) / 2
        
        # Matthews Correlation Coefficient
        mcc_num = (tp * tn) - (fp * fn)
        mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        matthews_corr = mcc_num / mcc_den if mcc_den > 0 else 0.0
        
        # Probabilistic metrics
        try:
            brier = brier_score_loss(y_true, y_pred_proba)
            logloss = log_loss(y_true, y_pred_proba)
        except ValueError:
            brier = np.nan
            logloss = np.nan
        
        # Calibration error
        calibration_error = self._calculate_calibration_error(y_true, y_pred_proba)
        
        # Optimal threshold analysis
        optimal_threshold, threshold_metrics = self._analyze_thresholds(y_true, y_pred_proba)
        
        # Prediction confidence analysis
        prediction_entropy = -np.mean(
            y_pred_proba * np.log(y_pred_proba + 1e-15) + 
            (1 - y_pred_proba) * np.log(1 - y_pred_proba + 1e-15)
        )
        
        confidence_dist = {
            'mean_confidence': np.mean(np.maximum(y_pred_proba, 1 - y_pred_proba)),
            'median_confidence': np.median(np.maximum(y_pred_proba, 1 - y_pred_proba)),
            'low_confidence_ratio': np.mean(np.maximum(y_pred_proba, 1 - y_pred_proba) < 0.6),
            'high_confidence_ratio': np.mean(np.maximum(y_pred_proba, 1 - y_pred_proba) > 0.9)
        }
        
        return AdvancedMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1,
            roc_auc=roc_auc,
            average_precision=avg_precision,
            balanced_accuracy=balanced_accuracy,
            matthews_correlation=matthews_corr,
            specificity=specificity,
            negative_predictive_value=npv,
            false_positive_rate=fpr,
            false_negative_rate=fnr,
            brier_score=brier,
            log_loss=logloss,
            calibration_error=calibration_error,
            optimal_threshold=optimal_threshold,
            threshold_metrics=threshold_metrics,
            prediction_entropy=prediction_entropy,
            confidence_distribution=confidence_dist
        )
    
    def _calculate_calibration_error(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                   n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error (ECE)."""
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_pred_proba, n_bins=n_bins
            )
            
            # Calculate bin weights
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0
            for bin_lower, bin_upper, frac_pos, mean_pred in zip(
                bin_lowers, bin_uppers, fraction_of_positives, mean_predicted_value
            ):
                # Calculate bin weight (fraction of samples in bin)
                in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    ece += np.abs(frac_pos - mean_pred) * prop_in_bin
            
            return ece
        except (ValueError, IndexError):
            return np.nan
    
    def _analyze_thresholds(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                          n_thresholds: int = 100) -> Tuple[float, Dict[str, List[float]]]:
        """Analyze performance across different classification thresholds."""
        thresholds = np.linspace(0.01, 0.99, n_thresholds)
        
        precisions = []
        recalls = []
        f1_scores = []
        specificities = []
        
        best_f1 = 0
        optimal_threshold = 0.5
        
        for threshold in thresholds:
            y_pred_thresh = (y_pred_proba >= threshold).astype(int)
            
            # Skip if all predictions are the same class
            if len(np.unique(y_pred_thresh)) == 1:
                precisions.append(np.nan)
                recalls.append(np.nan)
                f1_scores.append(np.nan)
                specificities.append(np.nan)
                continue
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred_thresh, average='binary', zero_division=0
            )
            
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_thresh).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
            specificities.append(specificity)
            
            # Track best F1 threshold
            if f1 > best_f1:
                best_f1 = f1
                optimal_threshold = threshold
        
        return optimal_threshold, {
            'thresholds': thresholds.tolist(),
            'precisions': precisions,
            'recalls': recalls,
            'f1_scores': f1_scores,
            'specificities': specificities
        }
    
    def _analyze_learning_curves(self, X: np.ndarray, y: np.ndarray, 
                               feature_names: List[str], 
                               config: Dict[str, Any]) -> LearningCurveAnalysis:
        """Analyze learning curves to assess training convergence and overfitting."""
        from sklearn.base import BaseEstimator, ClassifierMixin
        
        # Create a sklearn-compatible wrapper for EntityClassifier
        class EntityClassifierWrapper(BaseEstimator, ClassifierMixin):
            def __init__(self, config, feature_names):
                self.config = config
                self.feature_names = feature_names
                self.classifier = None
            
            def fit(self, X, y):
                self.classifier = EntityClassifier(self.config)
                self.classifier.fit(X, y, self.feature_names)
                return self
            
            def predict(self, X):
                return self.classifier.predict(X)
            
            def predict_proba(self, X):
                proba = self.classifier.predict_proba(X)
                # Return probabilities for both classes
                return np.column_stack([1 - proba, proba])
            
            def score(self, X, y):
                y_pred = self.predict(X)
                return np.mean(y == y_pred)
        
        wrapper = EntityClassifierWrapper(config, feature_names)
        
        # Generate learning curves
        train_sizes = np.linspace(0.1, 1.0, 10)
        
        try:
            train_sizes_abs, train_scores, validation_scores = learning_curve(
                wrapper, X, y, 
                train_sizes=train_sizes,
                cv=min(3, self.cv_folds),  # Use fewer folds for efficiency
                scoring='f1',
                random_state=self.random_seed,
                n_jobs=1  # Avoid multiprocessing issues
            )
            
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            validation_scores_mean = np.mean(validation_scores, axis=1)
            validation_scores_std = np.std(validation_scores, axis=1)
            
            # Analyze convergence
            convergence_score, convergence_iter = self._analyze_convergence(validation_scores_mean)
            
            # Analyze overfitting
            overfitting_score = self._calculate_overfitting_score(
                train_scores_mean, validation_scores_mean
            )
            
            # Estimate learning rate
            learning_rate_est = self._estimate_learning_rate(train_sizes_abs, validation_scores_mean)
            
        except Exception as e:
            logger.warning(f"Learning curve analysis failed: {e}")
            # Return default values
            train_sizes_abs = np.array([])
            train_scores_mean = np.array([])
            train_scores_std = np.array([])
            validation_scores_mean = np.array([])
            validation_scores_std = np.array([])
            convergence_score = 0.0
            convergence_iter = 0
            overfitting_score = 0.0
            learning_rate_est = 0.0
        
        return LearningCurveAnalysis(
            train_sizes=train_sizes_abs,
            train_scores_mean=train_scores_mean,
            train_scores_std=train_scores_std,
            validation_scores_mean=validation_scores_mean,
            validation_scores_std=validation_scores_std,
            convergence_score=convergence_score,
            convergence_iteration=convergence_iter,
            overfitting_score=overfitting_score,
            learning_rate_estimate=learning_rate_est
        )
    
    def _analyze_convergence(self, validation_scores: np.ndarray) -> Tuple[float, int]:
        """Analyze convergence characteristics of the learning curve."""
        if len(validation_scores) < 3:
            return 0.0, 0
        
        # Calculate rate of improvement
        improvements = np.diff(validation_scores)
        
        # Find where improvements become minimal
        convergence_threshold = 0.001
        convergence_iter = 0
        
        for i, improvement in enumerate(improvements):
            if improvement < convergence_threshold:
                convergence_iter = i + 1
                break
        
        # Convergence score based on stability of final scores
        if len(validation_scores) >= 3:
            final_scores = validation_scores[-3:]
            convergence_score = 1.0 - np.std(final_scores)
        else:
            convergence_score = 0.0
        
        return max(0.0, min(1.0, convergence_score)), convergence_iter
    
    def _calculate_overfitting_score(self, train_scores: np.ndarray, 
                                   val_scores: np.ndarray) -> float:
        """Calculate overfitting score (gap between training and validation)."""
        if len(train_scores) == 0 or len(val_scores) == 0:
            return 0.0
        
        # Use final scores for overfitting assessment
        final_train = train_scores[-1] if len(train_scores) > 0 else 0.0
        final_val = val_scores[-1] if len(val_scores) > 0 else 0.0
        
        overfitting_gap = final_train - final_val
        return max(0.0, overfitting_gap)
    
    def _estimate_learning_rate(self, train_sizes: np.ndarray, 
                              validation_scores: np.ndarray) -> float:
        """Estimate learning rate from validation score progression."""
        if len(train_sizes) < 3 or len(validation_scores) < 3:
            return 0.0
        
        try:
            # Fit exponential model: score = a * (1 - exp(-b * size))
            def exponential_model(x, a, b):
                return a * (1 - np.exp(-b * x))
            
            # Normalize train sizes
            x_norm = train_sizes / train_sizes[-1]
            
            popt, _ = curve_fit(
                exponential_model, x_norm, validation_scores,
                p0=[1.0, 1.0], maxfev=1000
            )
            
            return popt[1]  # Learning rate parameter
        except:
            return 0.0
    
    def _analyze_feature_stability(self, feature_importance_by_fold: Dict[str, List[float]], 
                                 feature_names: List[str]) -> FeatureStabilityAnalysis:
        """Analyze feature importance stability across CV folds."""
        stability_scores = {}
        rank_consistency = {}
        variance_ratio = {}
        
        # Calculate stability for each feature
        for feature in feature_names:
            if feature in feature_importance_by_fold and feature_importance_by_fold[feature]:
                importances = np.array(feature_importance_by_fold[feature])
                
                # Stability as 1 - coefficient of variation
                if np.mean(importances) != 0:
                    cv_coeff = np.std(importances) / abs(np.mean(importances))
                    stability_scores[feature] = max(0.0, 1.0 - cv_coeff)
                else:
                    stability_scores[feature] = 1.0 if np.std(importances) == 0 else 0.0
                
                # Variance to mean ratio
                if abs(np.mean(importances)) > 1e-8:
                    variance_ratio[feature] = np.var(importances) / abs(np.mean(importances))
                else:
                    variance_ratio[feature] = 0.0
            else:
                stability_scores[feature] = 0.0
                variance_ratio[feature] = 0.0
        
        # Calculate rank consistency
        importance_matrix = []
        for fold in range(len(next(iter(feature_importance_by_fold.values())))):
            fold_importances = []
            for feature in feature_names:
                if (feature in feature_importance_by_fold and 
                    len(feature_importance_by_fold[feature]) > fold):
                    fold_importances.append(feature_importance_by_fold[feature][fold])
                else:
                    fold_importances.append(0.0)
            importance_matrix.append(fold_importances)
        
        importance_matrix = np.array(importance_matrix)
        
        # Calculate rank consistency using Spearman correlation
        for i, feature in enumerate(feature_names):
            ranks = []
            for fold in range(importance_matrix.shape[0]):
                fold_ranks = stats.rankdata(-importance_matrix[fold])  # Negative for descending
                ranks.append(fold_ranks[i])
            
            if len(set(ranks)) > 1:
                # Rank consistency as 1 - (std of ranks / max possible std)
                max_std = np.sqrt((len(ranks) - 1) * len(feature_names)**2 / 12)
                rank_consistency[feature] = max(0.0, 1.0 - (np.std(ranks) / max_std))
            else:
                rank_consistency[feature] = 1.0
        
        # Overall stability
        overall_stability = np.mean(list(stability_scores.values()))
        
        return FeatureStabilityAnalysis(
            feature_importance_by_fold=feature_importance_by_fold,
            stability_scores=stability_scores,
            rank_consistency=rank_consistency,
            variance_ratio=variance_ratio,
            overall_stability=overall_stability
        )
    
    def _calculate_confidence_intervals(self, cv_details: Dict[str, Any], 
                                      confidence: float = 0.95) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for key metrics."""
        confidence_intervals = {}
        
        # Extract metrics from fold results
        metrics = ['precision', 'recall', 'f1', 'auc']
        
        for metric in metrics:
            values = [fold[metric] for fold in cv_details['fold_results']]
            
            if len(values) > 1:
                # Bootstrap confidence interval
                ci = self._bootstrap_confidence_interval(values, confidence)
                confidence_intervals[metric] = ci
            else:
                confidence_intervals[metric] = (values[0], values[0]) if values else (0.0, 0.0)
        
        return confidence_intervals
    
    def _bootstrap_confidence_interval(self, data: List[float], 
                                     confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval."""
        data = np.array(data)
        
        # Bootstrap resampling
        bootstrap_samples = []
        for _ in range(self.bootstrap_samples):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_samples.append(np.mean(sample))
        
        # Calculate confidence interval
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_samples, (alpha/2) * 100)
        upper = np.percentile(bootstrap_samples, (1 - alpha/2) * 100)
        
        return (lower, upper)
    
    def _estimate_memory_usage(self, X: np.ndarray) -> float:
        """Estimate memory usage in MB."""
        # Basic estimation based on array size
        array_size_bytes = X.nbytes
        
        # Add overhead estimates
        overhead_factor = 3.0  # Conservative estimate for processing overhead
        total_bytes = array_size_bytes * overhead_factor
        
        return total_bytes / (1024 * 1024)  # Convert to MB
    
    def save_validation_results(self, results: ValidationResults, 
                              output_dir: Path) -> str:
        """
        Save validation results to file.
        
        Args:
            results: Validation results to save
            output_dir: Directory to save the results
            
        Returns:
            Path to saved results file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        timestamp = results.timestamp.strftime('%Y%m%d_%H%M%S')
        results_path = output_dir / f"validation_results_{timestamp}.json"
        
        # Convert to serializable format
        results_dict = asdict(results)
        
        # Handle numpy arrays and other non-serializable objects
        def convert_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.float64):
                return float(obj)
            elif isinstance(obj, np.int64):
                return int(obj)
            elif hasattr(obj, 'isoformat'):
                return obj.isoformat()
            elif np.isnan(obj) if isinstance(obj, (int, float)) else False:
                return None
            return obj
        
        def recursive_convert(d):
            if isinstance(d, dict):
                return {k: recursive_convert(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [recursive_convert(item) for item in d]
            else:
                return convert_types(d)
        
        results_dict = recursive_convert(results_dict)
        
        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        logger.info(f"Validation results saved to {results_path}")
        return str(results_path)