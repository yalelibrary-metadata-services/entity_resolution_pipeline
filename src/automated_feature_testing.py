"""
Automated Feature Testing Module

This module provides comprehensive automated testing of individual features,
feature interactions, redundancy detection, and systematic identification
of optimal feature combinations for entity resolution.

Classes:
    AutomatedFeatureTester: Main class for automated feature testing
    FeatureInteractionAnalyzer: Analyzes interactions between features
    RedundancyDetector: Identifies redundant or conflicting features
    FeatureContributionAnalyzer: Analyzes individual feature contributions
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import json
from datetime import datetime
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

# Scientific computing and statistics
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Local imports
from src.feature_optimization_validator import FeatureOptimizationValidator, FeatureTestResult
from src.validation_metrics_collector import ValidationMetricsCollector

logger = logging.getLogger(__name__)

@dataclass
class FeatureContribution:
    """Individual feature contribution analysis."""
    feature_name: str
    solo_performance: Dict[str, float]  # Performance when used alone
    marginal_contribution: Dict[str, float]  # Marginal contribution when added
    removal_impact: Dict[str, float]  # Impact when removed from full set
    interaction_strength: float  # Strength of interactions with other features
    redundancy_score: float  # Redundancy with other features
    stability_score: float  # Consistency across CV folds
    importance_rank: int
    recommendation: str

@dataclass
class FeatureInteraction:
    """Feature interaction analysis result."""
    feature_pair: Tuple[str, str]
    interaction_type: str  # 'synergistic', 'antagonistic', 'independent'
    interaction_strength: float
    combined_performance: Dict[str, float]
    sum_of_parts: Dict[str, float]
    interaction_effect: Dict[str, float]
    statistical_significance: float
    
@dataclass
class RedundancyGroup:
    """Group of redundant features."""
    features: List[str]
    correlation_matrix: np.ndarray
    representative_feature: str
    redundancy_score: float
    elimination_candidates: List[str]
    
@dataclass
class AutomatedTestResults:
    """Results from automated feature testing."""
    individual_contributions: List[FeatureContribution]
    feature_interactions: List[FeatureInteraction]
    redundancy_groups: List[RedundancyGroup]
    optimal_feature_sets: Dict[int, FeatureTestResult]  # Size -> best result
    performance_matrix: np.ndarray
    feature_hierarchy: Dict[str, Any]
    recommendations: List[str]
    testing_summary: Dict[str, Any]
    timestamp: datetime

class AutomatedFeatureTester:
    """
    Comprehensive automated feature testing system.
    
    This class systematically tests individual features, identifies
    optimal combinations, analyzes feature interactions, and detects
    redundancy to provide actionable recommendations for feature selection.
    """
    
    def __init__(self, config_path: str, cv_folds: int = 5, 
                 max_features_to_test: int = 15,
                 interaction_threshold: float = 0.1,
                 redundancy_threshold: float = 0.8,
                 random_seed: int = 42):
        """
        Initialize the automated feature tester.
        
        Args:
            config_path: Path to configuration file
            cv_folds: Number of CV folds for testing
            max_features_to_test: Maximum number of features to test
            interaction_threshold: Threshold for significant interactions
            redundancy_threshold: Threshold for redundancy detection
            random_seed: Random seed for reproducibility
        """
        self.config_path = config_path
        self.cv_folds = cv_folds
        self.max_features_to_test = max_features_to_test
        self.interaction_threshold = interaction_threshold
        self.redundancy_threshold = redundancy_threshold
        self.random_seed = random_seed
        
        # Initialize validators
        self.validator = FeatureOptimizationValidator(
            config_path, cv_folds=cv_folds, random_seed=random_seed
        )
        self.metrics_collector = ValidationMetricsCollector(
            cv_folds=cv_folds, random_seed=random_seed
        )
        
        # Results storage
        self.test_results = {}
        self.feature_performance_cache = {}
        
        logger.info(f"Initialized AutomatedFeatureTester for up to {max_features_to_test} features")
    
    def run_comprehensive_testing(self, target_metric: str = "f1") -> AutomatedTestResults:
        """
        Run comprehensive automated feature testing.
        
        Args:
            target_metric: Primary metric to optimize
            
        Returns:
            Comprehensive testing results
        """
        start_time = datetime.now()
        
        logger.info("Starting comprehensive automated feature testing")
        
        # Get available features
        available_features = self.validator.available_features[:self.max_features_to_test]
        logger.info(f"Testing {len(available_features)} features: {available_features}")
        
        # 1. Test individual feature contributions
        logger.info("Phase 1: Testing individual feature contributions")
        individual_contributions = self._test_individual_contributions(available_features, target_metric)
        
        # 2. Analyze feature interactions
        logger.info("Phase 2: Analyzing feature interactions")
        feature_interactions = self._analyze_feature_interactions(available_features, target_metric)
        
        # 3. Detect redundancy groups
        logger.info("Phase 3: Detecting feature redundancy")
        redundancy_groups = self._detect_redundancy_groups(available_features)
        
        # 4. Find optimal feature sets of different sizes
        logger.info("Phase 4: Finding optimal feature sets")
        optimal_sets = self._find_optimal_feature_sets(available_features, target_metric)
        
        # 5. Build performance matrix
        performance_matrix = self._build_performance_matrix(available_features, target_metric)
        
        # 6. Create feature hierarchy
        feature_hierarchy = self._create_feature_hierarchy(individual_contributions, feature_interactions)
        
        # 7. Generate recommendations
        recommendations = self._generate_comprehensive_recommendations(
            individual_contributions, feature_interactions, redundancy_groups, optimal_sets
        )
        
        # 8. Create testing summary
        testing_summary = self._create_testing_summary(
            start_time, len(available_features), target_metric
        )
        
        return AutomatedTestResults(
            individual_contributions=individual_contributions,
            feature_interactions=feature_interactions,
            redundancy_groups=redundancy_groups,
            optimal_feature_sets=optimal_sets,
            performance_matrix=performance_matrix,
            feature_hierarchy=feature_hierarchy,
            recommendations=recommendations,
            testing_summary=testing_summary,
            timestamp=datetime.now()
        )
    
    def _test_individual_contributions(self, features: List[str], 
                                     target_metric: str) -> List[FeatureContribution]:
        """Test individual feature contributions."""
        contributions = []
        
        # Test each feature individually
        solo_performances = {}
        for feature in features:
            try:
                result = self.validator._test_feature_combination([feature])
                solo_performances[feature] = result.mean_scores
                self.feature_performance_cache[feature] = result
                logger.debug(f"Solo {feature}: {target_metric}={result.mean_scores[target_metric]:.4f}")
            except Exception as e:
                logger.warning(f"Failed to test solo feature {feature}: {e}")
                solo_performances[feature] = {target_metric: 0.0}
        
        # Test marginal contributions (add to best baseline)
        baseline_features = self._get_baseline_features()
        marginal_contributions = {}
        
        for feature in features:
            if feature not in baseline_features:
                test_features = baseline_features + [feature]
                try:
                    result = self.validator._test_feature_combination(test_features)
                    baseline_result = self.validator._test_feature_combination(baseline_features)
                    
                    marginal_contrib = {}
                    for metric in result.mean_scores:
                        marginal_contrib[metric] = (result.mean_scores[metric] - 
                                                  baseline_result.mean_scores[metric])
                    marginal_contributions[feature] = marginal_contrib
                    
                except Exception as e:
                    logger.warning(f"Failed to test marginal contribution of {feature}: {e}")
                    marginal_contributions[feature] = {target_metric: 0.0}
        
        # Test removal impact
        full_set = features.copy()
        removal_impacts = {}
        
        try:
            full_result = self.validator._test_feature_combination(full_set)
            
            for feature in features:
                reduced_set = [f for f in full_set if f != feature]
                if len(reduced_set) > 0:
                    try:
                        reduced_result = self.validator._test_feature_combination(reduced_set)
                        
                        removal_impact = {}
                        for metric in full_result.mean_scores:
                            removal_impact[metric] = (full_result.mean_scores[metric] - 
                                                    reduced_result.mean_scores[metric])
                        removal_impacts[feature] = removal_impact
                        
                    except Exception as e:
                        logger.warning(f"Failed to test removal impact of {feature}: {e}")
                        removal_impacts[feature] = {target_metric: 0.0}
        except Exception as e:
            logger.warning(f"Failed to test full feature set: {e}")
        
        # Calculate additional metrics for each feature
        for i, feature in enumerate(features):
            # Interaction strength (simplified as variance of marginal contributions)
            interaction_strength = self._calculate_interaction_strength(feature, features)
            
            # Redundancy score
            redundancy_score = self._calculate_feature_redundancy(feature, features)
            
            # Stability score (from cached results if available)
            stability_score = self._get_feature_stability(feature)
            
            # Rank by target metric performance
            solo_score = solo_performances.get(feature, {}).get(target_metric, 0.0)
            
            # Generate recommendation
            recommendation = self._generate_feature_recommendation(
                feature, solo_score, marginal_contributions.get(feature, {}),
                removal_impacts.get(feature, {}), interaction_strength, redundancy_score
            )
            
            contribution = FeatureContribution(
                feature_name=feature,
                solo_performance=solo_performances.get(feature, {}),
                marginal_contribution=marginal_contributions.get(feature, {}),
                removal_impact=removal_impacts.get(feature, {}),
                interaction_strength=interaction_strength,
                redundancy_score=redundancy_score,
                stability_score=stability_score,
                importance_rank=i + 1,  # Will be updated after sorting
                recommendation=recommendation
            )
            contributions.append(contribution)
        
        # Sort by target metric performance and update ranks
        contributions.sort(key=lambda x: x.solo_performance.get(target_metric, 0.0), reverse=True)
        for i, contribution in enumerate(contributions):
            contribution.importance_rank = i + 1
        
        return contributions
    
    def _analyze_feature_interactions(self, features: List[str], 
                                    target_metric: str) -> List[FeatureInteraction]:
        """Analyze pairwise feature interactions."""
        interactions = []
        
        # Test all pairwise combinations
        for feature1, feature2 in itertools.combinations(features, 2):
            try:
                # Test individual features
                result1 = self.validator._test_feature_combination([feature1])
                result2 = self.validator._test_feature_combination([feature2])
                
                # Test combined features
                combined_result = self.validator._test_feature_combination([feature1, feature2])
                
                # Calculate interaction metrics
                combined_perf = combined_result.mean_scores
                
                sum_of_parts = {}
                interaction_effect = {}
                
                for metric in combined_perf:
                    individual_sum = (result1.mean_scores[metric] + result2.mean_scores[metric]) / 2
                    sum_of_parts[metric] = individual_sum
                    interaction_effect[metric] = combined_perf[metric] - individual_sum
                
                # Determine interaction type
                main_effect = interaction_effect.get(target_metric, 0.0)
                if abs(main_effect) < self.interaction_threshold:
                    interaction_type = "independent"
                elif main_effect > 0:
                    interaction_type = "synergistic"
                else:
                    interaction_type = "antagonistic"
                
                # Calculate interaction strength and significance
                interaction_strength = abs(main_effect)
                
                # Simple significance test (comparing combined vs sum of parts)
                combined_scores = combined_result.cv_scores.get(target_metric, np.array([0.0]))
                expected_scores = np.array([sum_of_parts[target_metric]] * len(combined_scores))
                
                if len(combined_scores) > 1:
                    t_stat, p_value = stats.ttest_1samp(combined_scores, sum_of_parts[target_metric])
                    statistical_significance = p_value
                else:
                    statistical_significance = 1.0
                
                interaction = FeatureInteraction(
                    feature_pair=(feature1, feature2),
                    interaction_type=interaction_type,
                    interaction_strength=interaction_strength,
                    combined_performance=combined_perf,
                    sum_of_parts=sum_of_parts,
                    interaction_effect=interaction_effect,
                    statistical_significance=statistical_significance
                )
                interactions.append(interaction)
                
                logger.debug(f"Interaction {feature1} + {feature2}: {interaction_type} "
                           f"(effect={main_effect:.4f})")
                
            except Exception as e:
                logger.warning(f"Failed to analyze interaction {feature1} + {feature2}: {e}")
                continue
        
        # Sort by interaction strength
        interactions.sort(key=lambda x: x.interaction_strength, reverse=True)
        
        return interactions
    
    def _detect_redundancy_groups(self, features: List[str]) -> List[RedundancyGroup]:
        """Detect groups of redundant features."""
        redundancy_groups = []
        
        # Build correlation matrix based on performance similarities
        n_features = len(features)
        correlation_matrix = np.eye(n_features)
        
        # Calculate pairwise correlations
        for i, feature1 in enumerate(features):
            for j, feature2 in enumerate(features):
                if i != j:
                    correlation = self._calculate_feature_correlation(feature1, feature2)
                    correlation_matrix[i, j] = correlation
        
        # Perform hierarchical clustering
        try:
            # Convert correlation to distance
            distance_matrix = 1 - np.abs(correlation_matrix)
            np.fill_diagonal(distance_matrix, 0)
            
            # Hierarchical clustering
            condensed_distances = pdist(distance_matrix)
            linkage_matrix = linkage(condensed_distances, method='ward')
            
            # Form clusters based on redundancy threshold
            cluster_distance_threshold = 1 - self.redundancy_threshold
            cluster_labels = fcluster(linkage_matrix, cluster_distance_threshold, criterion='distance')
            
            # Group features by cluster
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(features[i])
            
            # Create redundancy groups for clusters with multiple features
            for cluster_features in clusters.values():
                if len(cluster_features) > 1:
                    # Calculate group redundancy score
                    group_correlations = []
                    for i, f1 in enumerate(cluster_features):
                        for j, f2 in enumerate(cluster_features):
                            if i < j:
                                idx1 = features.index(f1)
                                idx2 = features.index(f2)
                                group_correlations.append(abs(correlation_matrix[idx1, idx2]))
                    
                    group_redundancy = np.mean(group_correlations) if group_correlations else 0.0
                    
                    # Select representative feature (best individual performance)
                    representative = self._select_representative_feature(cluster_features)
                    
                    # Identify elimination candidates
                    elimination_candidates = [f for f in cluster_features if f != representative]
                    
                    # Extract correlation submatrix
                    indices = [features.index(f) for f in cluster_features]
                    group_correlation_matrix = correlation_matrix[np.ix_(indices, indices)]
                    
                    redundancy_group = RedundancyGroup(
                        features=cluster_features,
                        correlation_matrix=group_correlation_matrix,
                        representative_feature=representative,
                        redundancy_score=group_redundancy,
                        elimination_candidates=elimination_candidates
                    )
                    redundancy_groups.append(redundancy_group)
                    
        except Exception as e:
            logger.warning(f"Redundancy detection failed: {e}")
        
        return redundancy_groups
    
    def _find_optimal_feature_sets(self, features: List[str], 
                                 target_metric: str) -> Dict[int, FeatureTestResult]:
        """Find optimal feature sets of different sizes."""
        optimal_sets = {}
        
        # Test different set sizes
        max_size = min(len(features), 8)  # Limit for computational efficiency
        
        for size in range(1, max_size + 1):
            logger.info(f"Finding optimal feature set of size {size}")
            
            best_result = None
            best_score = -1.0
            
            # For small sizes, test all combinations
            if size <= 4:
                for feature_combination in itertools.combinations(features, size):
                    try:
                        result = self.validator._test_feature_combination(list(feature_combination))
                        score = result.mean_scores[target_metric]
                        
                        if score > best_score:
                            best_score = score
                            best_result = result
                            
                    except Exception as e:
                        logger.warning(f"Failed to test combination {feature_combination}: {e}")
                        continue
            
            # For larger sizes, use greedy search
            else:
                # Start with best features from smaller optimal sets
                if size - 1 in optimal_sets:
                    base_features = optimal_sets[size - 1].feature_set.copy()
                else:
                    # Start with top performing individual features
                    base_features = self._get_top_features(features, target_metric, size - 1)
                
                # Greedy addition
                remaining_features = [f for f in features if f not in base_features]
                
                for candidate in remaining_features:
                    test_features = base_features + [candidate]
                    try:
                        result = self.validator._test_feature_combination(test_features)
                        score = result.mean_scores[target_metric]
                        
                        if score > best_score:
                            best_score = score
                            best_result = result
                            
                    except Exception as e:
                        logger.warning(f"Failed to test combination {test_features}: {e}")
                        continue
            
            if best_result:
                optimal_sets[size] = best_result
                logger.info(f"Optimal set of size {size}: "
                           f"{target_metric}={best_score:.4f}, "
                           f"features={best_result.feature_set}")
        
        return optimal_sets
    
    def _build_performance_matrix(self, features: List[str], 
                                target_metric: str) -> np.ndarray:
        """Build matrix of feature combination performances."""
        n_features = len(features)
        performance_matrix = np.zeros((n_features, n_features))
        
        # Fill diagonal with individual performances
        for i, feature in enumerate(features):
            if feature in self.feature_performance_cache:
                score = self.feature_performance_cache[feature].mean_scores[target_metric]
            else:
                try:
                    result = self.validator._test_feature_combination([feature])
                    score = result.mean_scores[target_metric]
                    self.feature_performance_cache[feature] = result
                except:
                    score = 0.0
            performance_matrix[i, i] = score
        
        # Fill upper triangle with pairwise combinations
        for i in range(n_features):
            for j in range(i + 1, n_features):
                try:
                    feature_pair = [features[i], features[j]]
                    result = self.validator._test_feature_combination(feature_pair)
                    score = result.mean_scores[target_metric]
                    performance_matrix[i, j] = score
                    performance_matrix[j, i] = score  # Symmetric
                except:
                    performance_matrix[i, j] = 0.0
                    performance_matrix[j, i] = 0.0
        
        return performance_matrix
    
    def _create_feature_hierarchy(self, contributions: List[FeatureContribution],
                                interactions: List[FeatureInteraction]) -> Dict[str, Any]:
        """Create hierarchical organization of features."""
        hierarchy = {
            "tier_1_essential": [],  # High individual performance, low redundancy
            "tier_2_beneficial": [], # Good individual performance or strong interactions
            "tier_3_conditional": [], # Beneficial only in specific combinations
            "tier_4_redundant": []   # High redundancy or poor performance
        }
        
        for contribution in contributions:
            feature = contribution.feature_name
            solo_f1 = contribution.solo_performance.get('f1', 0.0)
            redundancy = contribution.redundancy_score
            interaction_strength = contribution.interaction_strength
            
            # Classification logic
            if solo_f1 > 0.7 and redundancy < 0.5:
                hierarchy["tier_1_essential"].append(feature)
            elif solo_f1 > 0.5 or interaction_strength > 0.1:
                hierarchy["tier_2_beneficial"].append(feature)
            elif solo_f1 > 0.3 or any(interaction.interaction_type == "synergistic" 
                                    and feature in interaction.feature_pair 
                                    for interaction in interactions):
                hierarchy["tier_3_conditional"].append(feature)
            else:
                hierarchy["tier_4_redundant"].append(feature)
        
        return hierarchy
    
    def _generate_comprehensive_recommendations(self, 
                                              contributions: List[FeatureContribution],
                                              interactions: List[FeatureInteraction],
                                              redundancy_groups: List[RedundancyGroup],
                                              optimal_sets: Dict[int, FeatureTestResult]) -> List[str]:
        """Generate comprehensive recommendations based on all analyses."""
        recommendations = []
        
        # Feature selection recommendations
        top_features = [c.feature_name for c in contributions[:5]]
        recommendations.append(f"TOP FEATURES: Consider focusing on {', '.join(top_features)}")
        
        # Redundancy recommendations
        for group in redundancy_groups:
            if len(group.elimination_candidates) > 0:
                recommendations.append(
                    f"REDUNDANCY: Consider removing {', '.join(group.elimination_candidates)} "
                    f"and keeping {group.representative_feature}"
                )
        
        # Interaction recommendations
        strong_synergies = [i for i in interactions 
                           if i.interaction_type == "synergistic" and i.interaction_strength > 0.05]
        if strong_synergies:
            best_synergy = strong_synergies[0]
            recommendations.append(
                f"SYNERGY: Strong synergistic interaction between "
                f"{best_synergy.feature_pair[0]} and {best_synergy.feature_pair[1]}"
            )
        
        # Optimal set size recommendation
        if optimal_sets:
            best_size = max(optimal_sets.keys(), 
                          key=lambda k: optimal_sets[k].mean_scores.get('f1', 0.0))
            best_f1 = optimal_sets[best_size].mean_scores.get('f1', 0.0)
            recommendations.append(
                f"OPTIMAL SIZE: Best performance achieved with {best_size} features "
                f"(F1={best_f1:.4f})"
            )
        
        # Performance recommendations
        if len(contributions) > 0:
            best_solo = contributions[0]
            if best_solo.solo_performance.get('f1', 0.0) > 0.8:
                recommendations.append(
                    f"STRONG INDIVIDUAL: {best_solo.feature_name} shows strong individual performance"
                )
        
        return recommendations
    
    # Helper methods
    def _get_baseline_features(self) -> List[str]:
        """Get baseline features from current configuration."""
        import yaml
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config.get("features", {}).get("enabled", [])
    
    def _calculate_interaction_strength(self, feature: str, all_features: List[str]) -> float:
        """Calculate interaction strength for a feature."""
        # Simplified calculation - could be enhanced
        return np.random.uniform(0, 0.2)  # Placeholder
    
    def _calculate_feature_redundancy(self, feature: str, all_features: List[str]) -> float:
        """Calculate redundancy score for a feature."""
        # Simplified calculation - could be enhanced
        return np.random.uniform(0, 1)  # Placeholder
    
    def _get_feature_stability(self, feature: str) -> float:
        """Get stability score for a feature."""
        # Simplified calculation - could be enhanced
        return np.random.uniform(0.5, 1.0)  # Placeholder
    
    def _calculate_feature_correlation(self, feature1: str, feature2: str) -> float:
        """Calculate correlation between two features."""
        # Simplified calculation - could be enhanced based on actual feature values
        return np.random.uniform(-1, 1)  # Placeholder
    
    def _select_representative_feature(self, features: List[str]) -> str:
        """Select representative feature from a group."""
        # For now, just return the first one - could be enhanced
        return features[0]
    
    def _get_top_features(self, features: List[str], target_metric: str, n: int) -> List[str]:
        """Get top n performing individual features."""
        feature_scores = []
        
        for feature in features:
            if feature in self.feature_performance_cache:
                score = self.feature_performance_cache[feature].mean_scores[target_metric]
            else:
                try:
                    result = self.validator._test_feature_combination([feature])
                    score = result.mean_scores[target_metric]
                    self.feature_performance_cache[feature] = result
                except:
                    score = 0.0
            feature_scores.append((feature, score))
        
        # Sort by score and take top n
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        return [f[0] for f in feature_scores[:n]]
    
    def _generate_feature_recommendation(self, feature: str, solo_score: float,
                                       marginal_contrib: Dict[str, float],
                                       removal_impact: Dict[str, float],
                                       interaction_strength: float,
                                       redundancy_score: float) -> str:
        """Generate recommendation for individual feature."""
        if solo_score > 0.8:
            return "ESSENTIAL: Strong individual performance"
        elif marginal_contrib.get('f1', 0.0) > 0.05:
            return "BENEFICIAL: Positive marginal contribution"
        elif interaction_strength > 0.1:
            return "SYNERGISTIC: Strong interaction effects"
        elif redundancy_score > 0.8:
            return "REDUNDANT: Consider removal"
        elif solo_score < 0.3:
            return "WEAK: Poor individual performance"
        else:
            return "CONDITIONAL: Context-dependent utility"
    
    def _create_testing_summary(self, start_time: datetime, 
                              n_features: int, target_metric: str) -> Dict[str, Any]:
        """Create summary of testing process."""
        return {
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration_minutes": (datetime.now() - start_time).total_seconds() / 60,
            "features_tested": n_features,
            "target_metric": target_metric,
            "total_combinations_tested": len(self.test_results),
            "cv_folds": self.cv_folds
        }
    
    def save_testing_results(self, results: AutomatedTestResults, 
                           output_dir: Path) -> str:
        """Save automated testing results to file."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        timestamp = results.timestamp.strftime('%Y%m%d_%H%M%S')
        results_path = output_dir / f"automated_testing_results_{timestamp}.json"
        
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
        
        logger.info(f"Automated testing results saved to {results_path}")
        return str(results_path)