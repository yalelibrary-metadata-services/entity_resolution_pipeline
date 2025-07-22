"""
Feature Optimization Validation Module

This module provides comprehensive feature optimization and validation capabilities
for the entity resolution pipeline, including systematic testing of feature
combinations, cross-validation, statistical significance testing, and automated
configuration optimization.

Classes:
    FeatureOptimizationValidator: Main coordinator for feature optimization
    SearchStrategy: Enumeration of available search strategies
    ValidationResults: Container for validation results with statistical analysis
"""

import logging
import os
import json
import copy
import itertools
import time
from datetime import datetime
from enum import Enum
from typing import Dict, List, Tuple, Any, Optional, Set, Union
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import yaml

# Scientific computing and statistics
from scipy import stats
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix

# Local imports
from src.training import EntityClassifier
from src.feature_engineering import FeatureEngineering
from src.custom_features import register_custom_features
from src.scaling_bridge import ScalingBridge
from src.utils import setup_deterministic_behavior

logger = logging.getLogger(__name__)

class SearchStrategy(Enum):
    """Available search strategies for feature optimization."""
    EXHAUSTIVE = "exhaustive"
    GREEDY_FORWARD = "greedy_forward"
    GREEDY_BACKWARD = "greedy_backward"
    RANDOM_SEARCH = "random_search"
    STEPWISE = "stepwise"
    RFE_GUIDED = "rfe_guided"

@dataclass
class FeatureTestResult:
    """Results from testing a specific feature combination."""
    feature_set: List[str]
    cv_scores: Dict[str, np.ndarray]  # precision, recall, f1, auc
    mean_scores: Dict[str, float]
    std_scores: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    training_time: float
    feature_importance: Dict[str, float]
    config_hash: str
    timestamp: datetime

@dataclass
class ComparisonResult:
    """Results from comparing two feature configurations."""
    baseline_result: FeatureTestResult
    test_result: FeatureTestResult
    statistical_tests: Dict[str, Dict[str, float]]  # metric -> {p_value, effect_size, test_statistic}
    significant_improvements: Dict[str, bool]
    recommendation: str

class FeatureOptimizationValidator:
    """
    Comprehensive feature optimization and validation system.
    
    This class provides systematic testing of feature combinations using
    various search strategies, cross-validation for robust evaluation,
    statistical significance testing, and automated report generation.
    """
    
    def __init__(self, config_path: str, search_strategy: str = "greedy_forward",
                 cv_folds: int = 5, random_seed: int = 42):
        """
        Initialize the feature optimization validator.
        
        Args:
            config_path: Path to the configuration file
            search_strategy: Strategy for searching feature combinations
            cv_folds: Number of cross-validation folds
            random_seed: Random seed for reproducibility
        """
        self.config_path = Path(config_path)
        self.search_strategy = SearchStrategy(search_strategy)
        self.cv_folds = cv_folds
        self.random_seed = random_seed
        
        # Load configuration
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup deterministic behavior
        setup_deterministic_behavior(random_seed)
        
        # Initialize paths
        self.output_dir = Path(self.config.get("output_dir", "data/output"))
        self.validation_dir = self.output_dir / "validation"
        self.validation_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.feature_engineering = None
        self.scaling_bridge = None
        self.current_experiment_id = None
        
        # Results storage
        self.experiment_results = {}
        self.baseline_result = None
        
        # Initialize dependencies as None
        self._hash_lookup = None
        self._weaviate_client = None
        
        # Feature universe - all available features
        self.available_features = self._discover_available_features()
        
        # Optimization parameters
        self.max_features = self.config.get("validation", {}).get("max_features", 10)
        self.max_combinations = self.config.get("validation", {}).get("max_combinations", 1000)
        self.significance_level = self.config.get("validation", {}).get("significance_level", 0.05)
        self.min_improvement_threshold = self.config.get("validation", {}).get("min_improvement_threshold", 0.01)
        
        logger.info(f"Initialized FeatureOptimizationValidator with {search_strategy} strategy")
        logger.info(f"Available features: {len(self.available_features)}")
        logger.info(f"Max features per combination: {self.max_features}")
    
    def _discover_available_features(self) -> List[str]:
        """
        Discover all available features from the configuration.
        
        Returns:
            List of available feature names
        """
        # Get features from config
        enabled_features = self.config.get("features", {}).get("enabled", [])
        
        # Add commented out features that could be enabled
        all_possible_features = [
            "person_cosine",
            "person_title_squared", 
            "composite_cosine",
            "composite_cosine_squared",
            "taxonomy_dissimilarity",
            "roles_cosine",
            "roles_cosine_weighted",
            "marcKey_cosine",
            "birth_death_match",
            "title_cosine_squared",
            "title_role_adjusted",
            "person_role_squared",
            "person_title_adjusted_squared",
            "marcKey_title_squared",
            # Binary features
            "has_related_work",
            # Non-linear features
            "confidence_amplifier_exponential",
            "confidence_amplifier_polynomial",
            "person_title_interaction_cubic",
            "marc_roles_polynomial_interaction",
            "confidence_cascade_indicator",
            "evidence_strength_weighted",
            "harmonic_mean_primary_features",
            "geometric_confidence_scaling",
            # Composite features
            "combined_person_title_role_adjusted",
            "combined_title_person",
            # Binary indicators
            "person_low_levenshtein_indicator",
            "person_low_jaro_winkler_indicator",
            "person_low_cosine_indicator",
        ]
        
        # Filter to features that have parameters defined or are currently enabled
        feature_params = self.config.get("features", {}).get("parameters", {})
        available = []
        
        for feature in all_possible_features:
            if feature in enabled_features or feature in feature_params:
                available.append(feature)
        
        # Also add any custom features
        custom_features = self.config.get("custom_features", {})
        for feature_name, feature_config in custom_features.items():
            if feature_config.get("enabled", False):
                available.append(feature_name)
        
        return sorted(list(set(available)))
    
    def set_baseline(self, feature_set: Optional[List[str]] = None) -> FeatureTestResult:
        """
        Set baseline performance using current or specified feature configuration.
        
        Args:
            feature_set: Optional specific feature set, uses current config if None
            
        Returns:
            Baseline test result
        """
        if feature_set is None:
            feature_set = self.config.get("features", {}).get("enabled", [])
        
        logger.info(f"Setting baseline with {len(feature_set)} features: {feature_set}")
        
        # Test the baseline configuration
        self.baseline_result = self._test_feature_combination(feature_set, is_baseline=True)
        
        logger.info(f"Baseline F1-score: {self.baseline_result.mean_scores['f1']:.4f} "
                   f"(±{self.baseline_result.std_scores['f1']:.4f})")
        
        return self.baseline_result
    
    def run_optimization(self, target_metric: str = "f1") -> Dict[str, Any]:
        """
        Run feature optimization using the configured search strategy.
        
        Args:
            target_metric: Metric to optimize (precision, recall, f1, auc)
            
        Returns:
            Dictionary with optimization results
        """
        self.current_experiment_id = f"optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting feature optimization with {self.search_strategy.value} strategy")
        logger.info(f"Target metric: {target_metric}")
        
        # Ensure baseline is set
        if self.baseline_result is None:
            self.set_baseline()
        
        # Run optimization based on strategy
        if self.search_strategy == SearchStrategy.EXHAUSTIVE:
            results = self._exhaustive_search(target_metric)
        elif self.search_strategy == SearchStrategy.GREEDY_FORWARD:
            results = self._greedy_forward_search(target_metric)
        elif self.search_strategy == SearchStrategy.GREEDY_BACKWARD:
            results = self._greedy_backward_search(target_metric)
        elif self.search_strategy == SearchStrategy.RANDOM_SEARCH:
            results = self._random_search(target_metric)
        elif self.search_strategy == SearchStrategy.STEPWISE:
            results = self._stepwise_search(target_metric)
        elif self.search_strategy == SearchStrategy.RFE_GUIDED:
            results = self._rfe_guided_search(target_metric)
        else:
            raise ValueError(f"Unsupported search strategy: {self.search_strategy}")
        
        # Find best configuration
        best_config = self._find_best_configuration(results, target_metric)
        
        # Generate comprehensive report
        report = self._generate_optimization_report(results, best_config, target_metric)
        
        return {
            "best_configuration": best_config,
            "optimization_results": results,
            "report": report,
            "experiment_id": self.current_experiment_id
        }
    
    def compare_configurations(self, config_a: List[str], config_b: List[str]) -> ComparisonResult:
        """
        Compare two feature configurations with statistical significance testing.
        
        Args:
            config_a: First feature configuration
            config_b: Second feature configuration
            
        Returns:
            Detailed comparison results
        """
        logger.info(f"Comparing configurations:")
        logger.info(f"  Config A ({len(config_a)}): {config_a}")
        logger.info(f"  Config B ({len(config_b)}): {config_b}")
        
        # Test both configurations
        result_a = self._test_feature_combination(config_a)
        result_b = self._test_feature_combination(config_b)
        
        # Perform statistical tests
        statistical_tests = {}
        significant_improvements = {}
        
        for metric in ['precision', 'recall', 'f1', 'auc']:
            scores_a = result_a.cv_scores[metric]
            scores_b = result_b.cv_scores[metric]
            
            # Paired t-test
            t_stat, p_value = stats.ttest_rel(scores_b, scores_a)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt((np.var(scores_a) + np.var(scores_b)) / 2)
            effect_size = (np.mean(scores_b) - np.mean(scores_a)) / pooled_std
            
            # Wilcoxon signed-rank test (non-parametric)
            wilcoxon_stat, wilcoxon_p = stats.wilcoxon(scores_b, scores_a)
            
            statistical_tests[metric] = {
                't_statistic': t_stat,
                't_test_p_value': p_value,
                'effect_size': effect_size,
                'wilcoxon_statistic': wilcoxon_stat,
                'wilcoxon_p_value': wilcoxon_p
            }
            
            # Check for significant improvement
            is_significant = p_value < self.significance_level
            is_meaningful = abs(effect_size) > self.min_improvement_threshold
            significant_improvements[metric] = is_significant and is_meaningful and (np.mean(scores_b) > np.mean(scores_a))
        
        # Generate recommendation
        recommendation = self._generate_comparison_recommendation(result_a, result_b, significant_improvements)
        
        return ComparisonResult(
            baseline_result=result_a,
            test_result=result_b,
            statistical_tests=statistical_tests,
            significant_improvements=significant_improvements,
            recommendation=recommendation
        )
    
    def _test_feature_combination(self, feature_set: List[str], is_baseline: bool = False) -> FeatureTestResult:
        """
        Test a specific feature combination using cross-validation.
        
        Args:
            feature_set: List of features to test
            is_baseline: Whether this is a baseline test
            
        Returns:
            Detailed test results
        """
        start_time = time.time()
        
        # Create temporary config with specified features
        temp_config = copy.deepcopy(self.config)
        temp_config["features"]["enabled"] = feature_set
        
        # Initialize required dependencies
        from src.preprocessing import load_hash_lookup
        from src.feature_engineering import FeatureEngineering
        
        # Load hash lookup if not already loaded
        if not hasattr(self, '_hash_lookup') or self._hash_lookup is None:
            hash_lookup_path = os.path.join(
                temp_config.get("checkpoint_dir", "data/checkpoints"),
                "hash_lookup.pkl"
            )
            if os.path.exists(hash_lookup_path):
                self._hash_lookup = load_hash_lookup(hash_lookup_path)
            else:
                logger.warning(f"Hash lookup not found at {hash_lookup_path}, using empty lookup")
                self._hash_lookup = {}
        
        # Initialize Weaviate client if not already initialized
        if not hasattr(self, '_weaviate_client') or self._weaviate_client is None:
            try:
                # Use the embedding and indexing pipeline to get a Weaviate client
                from src.embedding_and_indexing import EmbeddingAndIndexingPipeline
                temp_pipeline = EmbeddingAndIndexingPipeline(temp_config)
                self._weaviate_client = temp_pipeline.weaviate_client
                logger.info("Initialized Weaviate client for feature validation")
            except Exception as e:
                logger.warning(f"Could not initialize Weaviate client: {e}")
                logger.warning("Feature validation will run without vector-based features")
                self._weaviate_client = None
        
        # Initialize components with temp config
        feature_eng = FeatureEngineering(temp_config, self._weaviate_client, self._hash_lookup)
        
        # Register custom features
        register_custom_features(feature_eng, temp_config)
        
        # Load data using the same approach as training.py
        labeled_matches_path = os.path.join(
            temp_config.get("ground_truth_dir", "data/ground_truth"),
            temp_config.get("labeled_matches_file", "labeled_matches.csv")
        )
        
        if not os.path.exists(labeled_matches_path):
            raise FileNotFoundError(f"Labeled matches file not found: {labeled_matches_path}")
        
        # Load ground truth data
        from src.training import load_training_data
        labeled_pairs, _ = load_training_data(labeled_matches_path)
        
        # Load string dict if needed
        string_dict = None
        string_dict_path = os.path.join(
            temp_config.get("checkpoint_dir", "data/checkpoints"),
            "string_dict.pkl"
        )
        if os.path.exists(string_dict_path):
            import pickle
            with open(string_dict_path, 'rb') as f:
                string_dict = pickle.load(f)
        
        # Compute features using the same method as training
        X, y = feature_eng.compute_features(labeled_pairs, string_dict)
        
        if X.shape[0] == 0:
            raise ValueError(f"No training data generated for features: {feature_set}")
        
        # Handle scaling if enabled (using same approach as training.py)
        if temp_config.get("use_enhanced_scaling", False):
            X = feature_eng.normalize_features(X, fit=True)
        else:
            logger.debug("Enhanced scaling disabled, using raw features")
        
        # Perform cross-validation
        cv_results = self._cross_validate(X, y, feature_set, temp_config)
        
        # Calculate feature importance (using the full dataset)
        classifier = EntityClassifier(temp_config)
        classifier.fit(X, y, feature_set)
        feature_importance = dict(zip(feature_set, classifier.weights.tolist()))
        
        training_time = time.time() - start_time
        
        # Create config hash for tracking
        config_hash = hash(str(sorted(feature_set)))
        
        result = FeatureTestResult(
            feature_set=feature_set.copy(),
            cv_scores=cv_results["cv_scores"],
            mean_scores=cv_results["mean_scores"],
            std_scores=cv_results["std_scores"],
            confidence_intervals=cv_results["confidence_intervals"],
            training_time=training_time,
            feature_importance=feature_importance,
            config_hash=str(config_hash),
            timestamp=datetime.now()
        )
        
        logger.info(f"Tested {len(feature_set)} features in {training_time:.2f}s: "
                   f"F1={result.mean_scores['f1']:.4f}±{result.std_scores['f1']:.4f}")
        
        return result
    
    def _cross_validate(self, X: np.ndarray, y: np.ndarray, feature_names: List[str], 
                       config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform cross-validation with comprehensive metrics collection.
        
        Args:
            X: Feature matrix
            y: Labels
            feature_names: Feature names
            config: Configuration dictionary
            
        Returns:
            Cross-validation results with statistics
        """
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_seed)
        
        metrics = {
            'precision': [],
            'recall': [],
            'f1': [],
            'auc': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train classifier
            classifier = EntityClassifier(config)
            classifier.fit(X_train, y_train, feature_names)
            
            # Predict on validation set
            y_pred_proba = classifier.predict_proba(X_val)
            y_pred = classifier.predict(X_val)
            
            # Calculate metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_val, y_pred, average='binary', zero_division=0
            )
            
            try:
                auc_score = roc_auc_score(y_val, y_pred_proba)
            except ValueError:
                # Handle case where only one class is present
                auc_score = 0.5
            
            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
            metrics['f1'].append(f1)
            metrics['auc'].append(auc_score)
        
        # Convert to numpy arrays
        cv_scores = {metric: np.array(scores) for metric, scores in metrics.items()}
        
        # Calculate statistics
        mean_scores = {metric: np.mean(scores) for metric, scores in cv_scores.items()}
        std_scores = {metric: np.std(scores) for metric, scores in cv_scores.items()}
        
        # Calculate 95% confidence intervals
        confidence_intervals = {}
        for metric, scores in cv_scores.items():
            ci_lower = np.percentile(scores, 2.5)
            ci_upper = np.percentile(scores, 97.5)
            confidence_intervals[metric] = (ci_lower, ci_upper)
        
        return {
            "cv_scores": cv_scores,
            "mean_scores": mean_scores,
            "std_scores": std_scores,
            "confidence_intervals": confidence_intervals
        }
    
    def _greedy_forward_search(self, target_metric: str) -> List[FeatureTestResult]:
        """
        Perform greedy forward search for optimal feature combination.
        
        Args:
            target_metric: Metric to optimize
            
        Returns:
            List of test results from the search
        """
        logger.info("Starting greedy forward search")
        
        current_features = []
        remaining_features = self.available_features.copy()
        results = []
        
        # Start with empty set
        best_score = 0.0
        
        while remaining_features and len(current_features) < self.max_features:
            best_candidate = None
            best_candidate_score = best_score
            
            # Test adding each remaining feature
            for feature in remaining_features:
                test_features = current_features + [feature]
                
                try:
                    result = self._test_feature_combination(test_features)
                    score = result.mean_scores[target_metric]
                    
                    if score > best_candidate_score:
                        best_candidate = feature
                        best_candidate_score = score
                        best_result = result
                        
                except Exception as e:
                    logger.warning(f"Failed to test feature combination {test_features}: {e}")
                    continue
            
            # Add best candidate if it improves performance
            if best_candidate and best_candidate_score > best_score + self.min_improvement_threshold:
                current_features.append(best_candidate)
                remaining_features.remove(best_candidate)
                best_score = best_candidate_score
                results.append(best_result)
                
                logger.info(f"Added feature '{best_candidate}': {target_metric}={best_score:.4f}")
            else:
                logger.info("No beneficial feature found, stopping forward search")
                break
        
        return results
    
    def _greedy_backward_search(self, target_metric: str) -> List[FeatureTestResult]:
        """
        Perform greedy backward elimination for optimal feature combination.
        
        Args:
            target_metric: Metric to optimize
            
        Returns:
            List of test results from the search
        """
        logger.info("Starting greedy backward search")
        
        current_features = self.available_features.copy()
        results = []
        
        # Start with all features
        if len(current_features) > self.max_features:
            current_features = current_features[:self.max_features]
        
        current_result = self._test_feature_combination(current_features)
        best_score = current_result.mean_scores[target_metric]
        results.append(current_result)
        
        while len(current_features) > 1:
            worst_feature = None
            best_removal_score = best_score
            
            # Test removing each feature
            for feature in current_features:
                test_features = [f for f in current_features if f != feature]
                
                try:
                    result = self._test_feature_combination(test_features)
                    score = result.mean_scores[target_metric]
                    
                    # If removing this feature doesn't hurt (or helps), consider it
                    if score >= best_removal_score - self.min_improvement_threshold:
                        worst_feature = feature
                        best_removal_score = score
                        best_result = result
                        
                except Exception as e:
                    logger.warning(f"Failed to test feature combination {test_features}: {e}")
                    continue
            
            # Remove the worst feature if beneficial
            if worst_feature and best_removal_score >= best_score - self.min_improvement_threshold:
                current_features.remove(worst_feature)
                best_score = best_removal_score
                results.append(best_result)
                
                logger.info(f"Removed feature '{worst_feature}': {target_metric}={best_score:.4f}")
            else:
                logger.info("No beneficial feature removal found, stopping backward search")
                break
        
        return results
    
    def _random_search(self, target_metric: str, n_iterations: int = 100) -> List[FeatureTestResult]:
        """
        Perform random search over feature combinations.
        
        Args:
            target_metric: Metric to optimize
            n_iterations: Number of random combinations to test
            
        Returns:
            List of test results from the search
        """
        logger.info(f"Starting random search with {n_iterations} iterations")
        
        results = []
        np.random.seed(self.random_seed)
        
        for i in range(min(n_iterations, self.max_combinations)):
            # Random feature subset size (1 to max_features)
            n_features = np.random.randint(1, min(len(self.available_features), self.max_features) + 1)
            
            # Random feature selection
            feature_set = list(np.random.choice(
                self.available_features, 
                size=n_features, 
                replace=False
            ))
            
            try:
                result = self._test_feature_combination(feature_set)
                results.append(result)
                
                logger.info(f"Random iteration {i+1}/{n_iterations}: "
                           f"{target_metric}={result.mean_scores[target_metric]:.4f}")
                
            except Exception as e:
                logger.warning(f"Failed to test random combination {feature_set}: {e}")
                continue
        
        return results
    
    def _exhaustive_search(self, target_metric: str) -> List[FeatureTestResult]:
        """
        Perform exhaustive search over all feature combinations.
        
        Args:
            target_metric: Metric to optimize
            
        Returns:
            List of test results from the search
        """
        logger.warning("Exhaustive search can be computationally expensive!")
        
        results = []
        total_combinations = 0
        
        # Calculate total combinations for progress tracking
        for r in range(1, min(len(self.available_features), self.max_features) + 1):
            total_combinations += len(list(itertools.combinations(self.available_features, r)))
        
        if total_combinations > self.max_combinations:
            logger.warning(f"Too many combinations ({total_combinations}), limiting to {self.max_combinations}")
        
        combination_count = 0
        
        for r in range(1, min(len(self.available_features), self.max_features) + 1):
            for feature_combination in itertools.combinations(self.available_features, r):
                if combination_count >= self.max_combinations:
                    break
                
                feature_set = list(feature_combination)
                
                try:
                    result = self._test_feature_combination(feature_set)
                    results.append(result)
                    combination_count += 1
                    
                    if combination_count % 10 == 0:
                        logger.info(f"Tested {combination_count}/{min(total_combinations, self.max_combinations)} combinations")
                        
                except Exception as e:
                    logger.warning(f"Failed to test combination {feature_set}: {e}")
                    continue
        
        return results
    
    def _stepwise_search(self, target_metric: str) -> List[FeatureTestResult]:
        """
        Perform stepwise feature selection (combination of forward and backward).
        
        Args:
            target_metric: Metric to optimize
            
        Returns:
            List of test results from the search
        """
        logger.info("Starting stepwise search")
        
        # Start with forward selection
        forward_results = self._greedy_forward_search(target_metric)
        
        if not forward_results:
            return []
        
        # Get best feature set from forward selection
        best_forward = max(forward_results, key=lambda r: r.mean_scores[target_metric])
        
        # Now try backward elimination from this set
        temp_available = self.available_features.copy()
        self.available_features = best_forward.feature_set.copy()
        
        backward_results = self._greedy_backward_search(target_metric)
        
        # Restore original available features
        self.available_features = temp_available
        
        return forward_results + backward_results
    
    def _rfe_guided_search(self, target_metric: str) -> List[FeatureTestResult]:
        """
        Use RFE (Recursive Feature Elimination) to guide the search.
        
        Args:
            target_metric: Metric to optimize
            
        Returns:
            List of test results from the search
        """
        logger.info("Starting RFE-guided search")
        
        # This would integrate with the existing RFE module
        # For now, we'll implement a simplified version
        
        from sklearn.feature_selection import RFE
        from sklearn.linear_model import LogisticRegression
        
        # Build full dataset
        temp_config = copy.deepcopy(self.config)
        temp_config["features"]["enabled"] = self.available_features
        
        feature_eng = FeatureEngineering(temp_config)
        register_custom_features(feature_eng, temp_config)
        
        labeled_matches_path = os.path.join(
            temp_config.get("ground_truth_dir", "data/ground_truth"),
            temp_config.get("labeled_matches_file", "labeled_matches.csv")
        )
        
        pairs_data = feature_eng.build_training_dataset(labeled_matches_path)
        X = np.array([pair["features"] for pair in pairs_data])
        y = np.array([pair["label"] for pair in pairs_data])
        
        # Use RFE to rank features
        estimator = LogisticRegression(random_state=self.random_seed)
        rfe = RFE(estimator, n_features_to_select=1, step=1)
        rfe.fit(X, y)
        
        # Get feature ranking
        feature_ranking = list(zip(self.available_features, rfe.ranking_))
        feature_ranking.sort(key=lambda x: x[1])  # Sort by rank (1 is best)
        
        # Test progressively larger feature sets based on RFE ranking
        results = []
        current_features = []
        
        for feature, rank in feature_ranking:
            current_features.append(feature)
            
            if len(current_features) > self.max_features:
                break
            
            try:
                result = self._test_feature_combination(current_features)
                results.append(result)
                
                logger.info(f"RFE step {len(current_features)}: "
                           f"{target_metric}={result.mean_scores[target_metric]:.4f}")
                
            except Exception as e:
                logger.warning(f"Failed to test RFE combination {current_features}: {e}")
                current_features.pop()  # Remove the problematic feature
                continue
        
        return results
    
    def _find_best_configuration(self, results: List[FeatureTestResult], target_metric: str) -> FeatureTestResult:
        """
        Find the best configuration from search results.
        
        Args:
            results: List of test results
            target_metric: Metric to optimize
            
        Returns:
            Best configuration result
        """
        if not results:
            raise ValueError("No results to analyze")
        
        # Sort by target metric
        best_result = max(results, key=lambda r: r.mean_scores[target_metric])
        
        logger.info(f"Best configuration found: {len(best_result.feature_set)} features")
        logger.info(f"Features: {best_result.feature_set}")
        logger.info(f"Performance: {target_metric}={best_result.mean_scores[target_metric]:.4f}±{best_result.std_scores[target_metric]:.4f}")
        
        return best_result
    
    def _generate_comparison_recommendation(self, result_a: FeatureTestResult, 
                                         result_b: FeatureTestResult,
                                         significant_improvements: Dict[str, bool]) -> str:
        """
        Generate recommendation based on comparison results.
        
        Args:
            result_a: First result
            result_b: Second result
            significant_improvements: Dictionary of significant improvements
            
        Returns:
            Recommendation string
        """
        improvements = sum(significant_improvements.values())
        
        if improvements >= 3:
            return "STRONG RECOMMENDATION: Configuration B shows significant improvements in multiple metrics"
        elif improvements >= 2:
            return "MODERATE RECOMMENDATION: Configuration B shows significant improvements in key metrics"
        elif improvements >= 1:
            return "WEAK RECOMMENDATION: Configuration B shows limited significant improvements"
        else:
            return "NO RECOMMENDATION: No significant improvements detected"
    
    def _generate_optimization_report(self, results: List[FeatureTestResult], 
                                    best_config: FeatureTestResult, 
                                    target_metric: str) -> Dict[str, Any]:
        """
        Generate comprehensive optimization report.
        
        Args:
            results: All optimization results
            best_config: Best configuration found
            target_metric: Target metric optimized
            
        Returns:
            Comprehensive report dictionary
        """
        report = {
            "experiment_id": self.current_experiment_id,
            "timestamp": datetime.now().isoformat(),
            "search_strategy": self.search_strategy.value,
            "target_metric": target_metric,
            "total_combinations_tested": len(results),
            "best_configuration": {
                "features": best_config.feature_set,
                "feature_count": len(best_config.feature_set),
                "performance": best_config.mean_scores,
                "std_dev": best_config.std_scores,
                "confidence_intervals": best_config.confidence_intervals,
                "training_time": best_config.training_time
            },
            "baseline_comparison": None,
            "performance_distribution": self._analyze_performance_distribution(results, target_metric),
            "feature_importance_analysis": self._analyze_feature_importance(results),
            "recommendations": self._generate_actionable_recommendations(results, best_config, target_metric)
        }
        
        # Add baseline comparison if available
        if self.baseline_result:
            comparison = self.compare_configurations(
                self.baseline_result.feature_set,
                best_config.feature_set
            )
            report["baseline_comparison"] = asdict(comparison)
        
        # Save report
        report_path = self.validation_dir / f"optimization_report_{self.current_experiment_id}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Optimization report saved to {report_path}")
        
        return report
    
    def _analyze_performance_distribution(self, results: List[FeatureTestResult], 
                                        target_metric: str) -> Dict[str, Any]:
        """Analyze the distribution of performance across all tested configurations."""
        scores = [r.mean_scores[target_metric] for r in results]
        
        return {
            "mean": np.mean(scores),
            "median": np.median(scores),
            "std": np.std(scores),
            "min": np.min(scores),
            "max": np.max(scores),
            "percentiles": {
                "25th": np.percentile(scores, 25),
                "75th": np.percentile(scores, 75),
                "90th": np.percentile(scores, 90),
                "95th": np.percentile(scores, 95)
            }
        }
    
    def _analyze_feature_importance(self, results: List[FeatureTestResult]) -> Dict[str, Any]:
        """Analyze feature importance across all tested configurations."""
        feature_scores = {}
        feature_counts = {}
        
        for result in results:
            for feature in result.feature_set:
                if feature not in feature_scores:
                    feature_scores[feature] = []
                    feature_counts[feature] = 0
                
                feature_scores[feature].append(result.mean_scores['f1'])
                feature_counts[feature] += 1
        
        # Calculate average performance when each feature is included
        feature_avg_performance = {}
        for feature, scores in feature_scores.items():
            feature_avg_performance[feature] = np.mean(scores)
        
        # Sort by average performance
        sorted_features = sorted(feature_avg_performance.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "feature_frequency": feature_counts,
            "feature_avg_performance": feature_avg_performance,
            "top_features": sorted_features[:10],
            "bottom_features": sorted_features[-10:]
        }
    
    def _generate_actionable_recommendations(self, results: List[FeatureTestResult],
                                           best_config: FeatureTestResult,
                                           target_metric: str) -> List[str]:
        """Generate actionable recommendations based on optimization results."""
        recommendations = []
        
        # Feature count recommendation
        if len(best_config.feature_set) <= 5:
            recommendations.append(f"Optimal feature count appears to be {len(best_config.feature_set)}. Consider this for production deployment.")
        else:
            recommendations.append(f"Best configuration uses {len(best_config.feature_set)} features. Consider feature reduction for simplicity.")
        
        # Performance improvement recommendation
        if self.baseline_result:
            improvement = best_config.mean_scores[target_metric] - self.baseline_result.mean_scores[target_metric]
            if improvement > 0.01:
                recommendations.append(f"Significant {target_metric} improvement of {improvement:.4f} achieved over baseline.")
            else:
                recommendations.append("Limited improvement over baseline. Consider alternative approaches.")
        
        # Feature stability recommendation
        feature_importance = self._analyze_feature_importance(results)
        top_features = [f[0] for f in feature_importance["top_features"][:5]]
        recommendations.append(f"Most consistently beneficial features: {', '.join(top_features)}")
        
        # Computational efficiency recommendation
        avg_training_time = np.mean([r.training_time for r in results])
        if avg_training_time > 300:  # 5 minutes
            recommendations.append("Consider feature reduction to improve training efficiency.")
        
        return recommendations
    
    def save_optimal_configuration(self, best_config: FeatureTestResult, 
                                 backup_current: bool = True) -> str:
        """
        Save the optimal configuration to the config file.
        
        Args:
            best_config: Best configuration to save
            backup_current: Whether to backup current config
            
        Returns:
            Path to backup file if created
        """
        backup_path = None
        
        if backup_current:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = str(self.config_path.with_suffix(f'.backup_{timestamp}.yml'))
            
            # Copy current config to backup
            import shutil
            shutil.copy2(self.config_path, backup_path)
            logger.info(f"Current configuration backed up to {backup_path}")
        
        # Update configuration
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        config["features"]["enabled"] = best_config.feature_set
        
        # Add optimization metadata
        config["optimization_metadata"] = {
            "experiment_id": self.current_experiment_id,
            "optimization_date": datetime.now().isoformat(),
            "search_strategy": self.search_strategy.value,
            "performance": best_config.mean_scores,
            "feature_count": len(best_config.feature_set)
        }
        
        # Save updated configuration
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f, indent=2)
        
        logger.info(f"Optimal configuration saved to {self.config_path}")
        
        return backup_path
    
    def cleanup(self):
        """
        Clean up resources, particularly the Weaviate client.
        """
        if hasattr(self, '_weaviate_client') and self._weaviate_client is not None:
            try:
                self._weaviate_client.close()
                logger.info("Closed Weaviate client connection")
            except Exception as e:
                logger.warning(f"Error closing Weaviate client: {e}")
            finally:
                self._weaviate_client = None