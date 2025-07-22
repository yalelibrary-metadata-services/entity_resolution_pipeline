"""
Feature Set Comparator Module

This module provides comprehensive A/B testing capabilities for comparing
different feature configurations with statistical rigor, including
significance testing, effect size analysis, and detailed performance
comparisons.

Classes:
    FeatureSetComparator: Main class for comparing feature configurations
    StatisticalTest: Enumeration of available statistical tests
    ComparisonReport: Detailed comparison results with statistical analysis
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import json
from pathlib import Path
from datetime import datetime

# Statistical analysis
from scipy import stats
from scipy.stats import bootstrap
import matplotlib.pyplot as plt
import seaborn as sns

# Local imports
from src.feature_optimization_validator import FeatureTestResult

logger = logging.getLogger(__name__)

class StatisticalTest(Enum):
    """Available statistical tests for comparing configurations."""
    PAIRED_T_TEST = "paired_t_test"
    WILCOXON_SIGNED_RANK = "wilcoxon_signed_rank"
    MANN_WHITNEY_U = "mann_whitney_u"
    BOOTSTRAP_DIFFERENCE = "bootstrap_difference"

@dataclass
class StatisticalTestResult:
    """Results from a statistical test."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    interpretation: str

@dataclass
class MetricComparison:
    """Comparison results for a specific metric."""
    metric_name: str
    baseline_mean: float
    baseline_std: float
    test_mean: float
    test_std: float
    difference: float
    percent_change: float
    statistical_tests: List[StatisticalTestResult]
    is_significant: bool
    is_meaningful: bool
    recommendation: str

@dataclass
class ComparisonReport:
    """Comprehensive comparison report between two feature configurations."""
    baseline_config: List[str]
    test_config: List[str]
    baseline_result: FeatureTestResult
    test_result: FeatureTestResult
    metric_comparisons: Dict[str, MetricComparison]
    overall_recommendation: str
    summary_statistics: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    timestamp: datetime

class FeatureSetComparator:
    """
    Comprehensive feature set comparison with statistical analysis.
    
    This class provides detailed A/B testing capabilities for comparing
    feature configurations, including multiple statistical tests,
    effect size analysis, and risk assessment.
    """
    
    def __init__(self, significance_level: float = 0.05, 
                 min_effect_size: float = 0.1,
                 bootstrap_samples: int = 10000,
                 random_seed: int = 42):
        """
        Initialize the feature set comparator.
        
        Args:
            significance_level: Alpha level for statistical significance
            min_effect_size: Minimum effect size to consider meaningful
            bootstrap_samples: Number of bootstrap samples for CI estimation
            random_seed: Random seed for reproducibility
        """
        self.significance_level = significance_level
        self.min_effect_size = min_effect_size
        self.bootstrap_samples = bootstrap_samples
        self.random_seed = random_seed
        
        # Set random seed
        np.random.seed(random_seed)
        
        logger.info(f"Initialized FeatureSetComparator with α={significance_level}, "
                   f"min_effect_size={min_effect_size}")
    
    def compare_configurations(self, baseline_result: FeatureTestResult,
                             test_result: FeatureTestResult,
                             metrics: List[str] = None) -> ComparisonReport:
        """
        Perform comprehensive comparison between two feature configurations.
        
        Args:
            baseline_result: Results from baseline configuration
            test_result: Results from test configuration
            metrics: List of metrics to compare (defaults to all available)
            
        Returns:
            Detailed comparison report
        """
        if metrics is None:
            metrics = ['precision', 'recall', 'f1', 'auc']
        
        logger.info(f"Comparing configurations:")
        logger.info(f"  Baseline ({len(baseline_result.feature_set)}): {baseline_result.feature_set}")
        logger.info(f"  Test ({len(test_result.feature_set)}): {test_result.feature_set}")
        
        # Compare each metric
        metric_comparisons = {}
        for metric in metrics:
            comparison = self._compare_metric(
                baseline_result.cv_scores[metric],
                test_result.cv_scores[metric],
                metric
            )
            metric_comparisons[metric] = comparison
        
        # Generate overall recommendation
        overall_recommendation = self._generate_overall_recommendation(metric_comparisons)
        
        # Calculate summary statistics
        summary_stats = self._calculate_summary_statistics(baseline_result, test_result, metric_comparisons)
        
        # Perform risk assessment
        risk_assessment = self._assess_risk(baseline_result, test_result, metric_comparisons)
        
        report = ComparisonReport(
            baseline_config=baseline_result.feature_set.copy(),
            test_config=test_result.feature_set.copy(),
            baseline_result=baseline_result,
            test_result=test_result,
            metric_comparisons=metric_comparisons,
            overall_recommendation=overall_recommendation,
            summary_statistics=summary_stats,
            risk_assessment=risk_assessment,
            timestamp=datetime.now()
        )
        
        return report
    
    def _compare_metric(self, baseline_scores: np.ndarray, 
                       test_scores: np.ndarray, 
                       metric_name: str) -> MetricComparison:
        """
        Compare a specific metric between baseline and test configurations.
        
        Args:
            baseline_scores: CV scores from baseline configuration
            test_scores: CV scores from test configuration
            metric_name: Name of the metric being compared
            
        Returns:
            Detailed metric comparison
        """
        # Basic statistics
        baseline_mean = np.mean(baseline_scores)
        baseline_std = np.std(baseline_scores)
        test_mean = np.mean(test_scores)
        test_std = np.std(test_scores)
        
        difference = test_mean - baseline_mean
        percent_change = (difference / baseline_mean) * 100 if baseline_mean != 0 else 0
        
        # Perform statistical tests
        statistical_tests = []
        
        # 1. Paired t-test
        if len(baseline_scores) == len(test_scores):
            t_stat, p_value = stats.ttest_rel(test_scores, baseline_scores)
            effect_size = self._calculate_cohens_d(baseline_scores, test_scores, paired=True)
            ci = self._calculate_paired_difference_ci(baseline_scores, test_scores)
            interpretation = self._interpret_t_test(t_stat, p_value, effect_size)
            
            statistical_tests.append(StatisticalTestResult(
                test_name="Paired t-test",
                statistic=t_stat,
                p_value=p_value,
                effect_size=effect_size,
                confidence_interval=ci,
                interpretation=interpretation
            ))
        
        # 2. Wilcoxon signed-rank test (non-parametric paired)
        if len(baseline_scores) == len(test_scores) and len(baseline_scores) > 1:
            try:
                w_stat, w_p_value = stats.wilcoxon(test_scores, baseline_scores)
                # Effect size for Wilcoxon (r = Z / sqrt(N))
                z_score = stats.norm.ppf(w_p_value/2)
                w_effect_size = abs(z_score) / np.sqrt(len(baseline_scores))
                
                statistical_tests.append(StatisticalTestResult(
                    test_name="Wilcoxon signed-rank",
                    statistic=w_stat,
                    p_value=w_p_value,
                    effect_size=w_effect_size,
                    confidence_interval=(np.nan, np.nan),  # Not easily calculated for Wilcoxon
                    interpretation=self._interpret_wilcoxon(w_stat, w_p_value, w_effect_size)
                ))
            except ValueError as e:
                logger.warning(f"Wilcoxon test failed for {metric_name}: {e}")
        
        # 3. Mann-Whitney U test (independent samples)
        u_stat, u_p_value = stats.mannwhitneyu(test_scores, baseline_scores, alternative='two-sided')
        u_effect_size = self._calculate_rank_biserial_correlation(baseline_scores, test_scores)
        
        statistical_tests.append(StatisticalTestResult(
            test_name="Mann-Whitney U",
            statistic=u_stat,
            p_value=u_p_value,
            effect_size=u_effect_size,
            confidence_interval=(np.nan, np.nan),  # Complex to calculate
            interpretation=self._interpret_mann_whitney(u_stat, u_p_value, u_effect_size)
        ))
        
        # 4. Bootstrap confidence interval for difference
        bootstrap_ci = self._bootstrap_difference_ci(baseline_scores, test_scores)
        
        statistical_tests.append(StatisticalTestResult(
            test_name="Bootstrap difference",
            statistic=difference,
            p_value=np.nan,  # Not applicable for bootstrap CI
            effect_size=effect_size if 'effect_size' in locals() else np.nan,
            confidence_interval=bootstrap_ci,
            interpretation=self._interpret_bootstrap_ci(bootstrap_ci, difference)
        ))
        
        # Determine significance and meaningfulness
        # Use the most conservative test (paired t-test if available, otherwise Mann-Whitney)
        primary_test = statistical_tests[0] if len(statistical_tests) > 0 else None
        is_significant = primary_test.p_value < self.significance_level if primary_test else False
        is_meaningful = abs(primary_test.effect_size) >= self.min_effect_size if primary_test else False
        
        # Generate recommendation
        recommendation = self._generate_metric_recommendation(
            metric_name, difference, percent_change, is_significant, is_meaningful
        )
        
        return MetricComparison(
            metric_name=metric_name,
            baseline_mean=baseline_mean,
            baseline_std=baseline_std,
            test_mean=test_mean,
            test_std=test_std,
            difference=difference,
            percent_change=percent_change,
            statistical_tests=statistical_tests,
            is_significant=is_significant,
            is_meaningful=is_meaningful,
            recommendation=recommendation
        )
    
    def _calculate_cohens_d(self, group1: np.ndarray, group2: np.ndarray, paired: bool = False) -> float:
        """Calculate Cohen's d effect size."""
        if paired:
            differences = group2 - group1
            return np.mean(differences) / np.std(differences, ddof=1)
        else:
            pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + 
                                (len(group2) - 1) * np.var(group2, ddof=1)) / 
                               (len(group1) + len(group2) - 2))
            return (np.mean(group2) - np.mean(group1)) / pooled_std
    
    def _calculate_rank_biserial_correlation(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate rank-biserial correlation (effect size for Mann-Whitney U)."""
        n1, n2 = len(group1), len(group2)
        u_stat, _ = stats.mannwhitneyu(group2, group1, alternative='two-sided')
        return (2 * u_stat) / (n1 * n2) - 1
    
    def _calculate_paired_difference_ci(self, group1: np.ndarray, group2: np.ndarray, 
                                      confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for paired differences."""
        differences = group2 - group1
        mean_diff = np.mean(differences)
        se_diff = stats.sem(differences)
        
        alpha = 1 - confidence
        df = len(differences) - 1
        t_critical = stats.t.ppf(1 - alpha/2, df)
        
        margin_error = t_critical * se_diff
        return (mean_diff - margin_error, mean_diff + margin_error)
    
    def _bootstrap_difference_ci(self, group1: np.ndarray, group2: np.ndarray,
                               confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for difference in means."""
        def difference_statistic(x, y):
            return np.mean(y) - np.mean(x)
        
        # Bootstrap resampling
        differences = []
        for _ in range(self.bootstrap_samples):
            # Resample with replacement
            idx1 = np.random.choice(len(group1), len(group1), replace=True)
            idx2 = np.random.choice(len(group2), len(group2), replace=True)
            
            boot_group1 = group1[idx1]
            boot_group2 = group2[idx2]
            
            diff = difference_statistic(boot_group1, boot_group2)
            differences.append(diff)
        
        # Calculate confidence interval
        alpha = 1 - confidence
        lower = np.percentile(differences, (alpha/2) * 100)
        upper = np.percentile(differences, (1 - alpha/2) * 100)
        
        return (lower, upper)
    
    def _interpret_t_test(self, t_stat: float, p_value: float, effect_size: float) -> str:
        """Interpret t-test results."""
        significance = "significant" if p_value < self.significance_level else "not significant"
        
        if abs(effect_size) < 0.2:
            effect_desc = "negligible"
        elif abs(effect_size) < 0.5:
            effect_desc = "small"
        elif abs(effect_size) < 0.8:
            effect_desc = "medium"
        else:
            effect_desc = "large"
        
        direction = "positive" if t_stat > 0 else "negative"
        
        return f"Result is {significance} (p={p_value:.4f}) with {effect_desc} {direction} effect (d={effect_size:.3f})"
    
    def _interpret_wilcoxon(self, w_stat: float, p_value: float, effect_size: float) -> str:
        """Interpret Wilcoxon signed-rank test results."""
        significance = "significant" if p_value < self.significance_level else "not significant"
        
        if abs(effect_size) < 0.1:
            effect_desc = "negligible"
        elif abs(effect_size) < 0.3:
            effect_desc = "small"
        elif abs(effect_size) < 0.5:
            effect_desc = "medium"
        else:
            effect_desc = "large"
        
        return f"Non-parametric test is {significance} (p={p_value:.4f}) with {effect_desc} effect (r={effect_size:.3f})"
    
    def _interpret_mann_whitney(self, u_stat: float, p_value: float, effect_size: float) -> str:
        """Interpret Mann-Whitney U test results."""
        significance = "significant" if p_value < self.significance_level else "not significant"
        
        if abs(effect_size) < 0.1:
            effect_desc = "negligible"
        elif abs(effect_size) < 0.3:
            effect_desc = "small"
        elif abs(effect_size) < 0.5:
            effect_desc = "medium"
        else:
            effect_desc = "large"
        
        return f"Independent samples test is {significance} (p={p_value:.4f}) with {effect_desc} effect (r={effect_size:.3f})"
    
    def _interpret_bootstrap_ci(self, ci: Tuple[float, float], observed_diff: float) -> str:
        """Interpret bootstrap confidence interval."""
        lower, upper = ci
        
        if lower > 0:
            return f"Bootstrap CI [{lower:.4f}, {upper:.4f}] suggests positive effect (CI excludes zero)"
        elif upper < 0:
            return f"Bootstrap CI [{lower:.4f}, {upper:.4f}] suggests negative effect (CI excludes zero)"
        else:
            return f"Bootstrap CI [{lower:.4f}, {upper:.4f}] includes zero (effect uncertain)"
    
    def _generate_metric_recommendation(self, metric_name: str, difference: float, 
                                      percent_change: float, is_significant: bool, 
                                      is_meaningful: bool) -> str:
        """Generate recommendation for a specific metric."""
        if is_significant and is_meaningful:
            direction = "improvement" if difference > 0 else "degradation"
            return f"STRONG {direction.upper()}: {percent_change:+.2f}% change is both statistically significant and practically meaningful"
        elif is_significant:
            direction = "improvement" if difference > 0 else "degradation"
            return f"WEAK {direction.upper()}: {percent_change:+.2f}% change is statistically significant but may not be practically meaningful"
        elif is_meaningful:
            direction = "improvement" if difference > 0 else "degradation"
            return f"UNCERTAIN {direction.upper()}: {percent_change:+.2f}% change is practically meaningful but not statistically significant"
        else:
            return f"NO CHANGE: {percent_change:+.2f}% change is neither statistically significant nor practically meaningful"
    
    def _generate_overall_recommendation(self, metric_comparisons: Dict[str, MetricComparison]) -> str:
        """Generate overall recommendation based on all metric comparisons."""
        strong_improvements = sum(1 for comp in metric_comparisons.values() 
                                if comp.is_significant and comp.is_meaningful and comp.difference > 0)
        strong_degradations = sum(1 for comp in metric_comparisons.values() 
                                if comp.is_significant and comp.is_meaningful and comp.difference < 0)
        weak_improvements = sum(1 for comp in metric_comparisons.values() 
                              if comp.is_significant and comp.difference > 0 and not comp.is_meaningful)
        
        total_metrics = len(metric_comparisons)
        
        if strong_improvements >= 3:
            return "STRONG RECOMMENDATION: Test configuration shows significant improvements across multiple key metrics"
        elif strong_improvements >= 2 and strong_degradations == 0:
            return "MODERATE RECOMMENDATION: Test configuration shows improvements in key metrics with no significant degradations"
        elif strong_improvements >= 1 and strong_degradations == 0:
            return "WEAK RECOMMENDATION: Test configuration shows some improvement with no significant degradations"
        elif strong_degradations >= 2:
            return "STRONG REJECTION: Test configuration shows significant degradations in multiple metrics"
        elif strong_degradations >= 1:
            return "MODERATE REJECTION: Test configuration shows significant degradation in at least one key metric"
        elif weak_improvements >= 2:
            return "CAUTIOUS CONSIDERATION: Test configuration shows statistical improvements but questionable practical significance"
        else:
            return "NO CLEAR RECOMMENDATION: Test configuration shows no consistent improvements or degradations"
    
    def _calculate_summary_statistics(self, baseline_result: FeatureTestResult,
                                    test_result: FeatureTestResult,
                                    metric_comparisons: Dict[str, MetricComparison]) -> Dict[str, Any]:
        """Calculate summary statistics for the comparison."""
        return {
            "feature_count_change": len(test_result.feature_set) - len(baseline_result.feature_set),
            "training_time_change": test_result.training_time - baseline_result.training_time,
            "training_time_percent_change": ((test_result.training_time - baseline_result.training_time) / 
                                           baseline_result.training_time) * 100,
            "significant_metrics": sum(1 for comp in metric_comparisons.values() if comp.is_significant),
            "meaningful_metrics": sum(1 for comp in metric_comparisons.values() if comp.is_meaningful),
            "improved_metrics": sum(1 for comp in metric_comparisons.values() if comp.difference > 0),
            "degraded_metrics": sum(1 for comp in metric_comparisons.values() if comp.difference < 0),
            "features_added": list(set(test_result.feature_set) - set(baseline_result.feature_set)),
            "features_removed": list(set(baseline_result.feature_set) - set(test_result.feature_set)),
            "features_common": list(set(baseline_result.feature_set) & set(test_result.feature_set))
        }
    
    def _assess_risk(self, baseline_result: FeatureTestResult,
                    test_result: FeatureTestResult,
                    metric_comparisons: Dict[str, MetricComparison]) -> Dict[str, Any]:
        """Assess risks associated with adopting the test configuration."""
        risks = []
        risk_level = "LOW"
        
        # Performance risk
        precision_comp = metric_comparisons.get('precision')
        if precision_comp and precision_comp.difference < -0.01:
            risks.append("Precision degradation risk")
            risk_level = "MEDIUM"
        
        recall_comp = metric_comparisons.get('recall')
        if recall_comp and recall_comp.difference < -0.02:
            risks.append("Significant recall degradation risk")
            risk_level = "HIGH"
        
        # Complexity risk
        feature_count_increase = len(test_result.feature_set) - len(baseline_result.feature_set)
        if feature_count_increase > 3:
            risks.append("Increased model complexity risk")
            if risk_level == "LOW":
                risk_level = "MEDIUM"
        
        # Training time risk
        time_increase_percent = ((test_result.training_time - baseline_result.training_time) / 
                               baseline_result.training_time) * 100
        if time_increase_percent > 50:
            risks.append("Significant training time increase risk")
            if risk_level == "LOW":
                risk_level = "MEDIUM"
        
        # Variance risk (stability)
        f1_variance_increase = (test_result.std_scores['f1'] - baseline_result.std_scores['f1']) / baseline_result.std_scores['f1']
        if f1_variance_increase > 0.5:
            risks.append("Increased performance variance risk (less stable)")
            if risk_level == "LOW":
                risk_level = "MEDIUM"
        
        return {
            "risk_level": risk_level,
            "identified_risks": risks,
            "risk_mitigation_suggestions": self._suggest_risk_mitigations(risks),
            "confidence_assessment": self._assess_confidence(metric_comparisons)
        }
    
    def _suggest_risk_mitigations(self, risks: List[str]) -> List[str]:
        """Suggest mitigations for identified risks."""
        mitigations = []
        
        if any("Precision degradation" in risk for risk in risks):
            mitigations.append("Consider threshold tuning to recover precision")
            mitigations.append("Validate precision performance on additional test sets")
        
        if any("recall degradation" in risk for risk in risks):
            mitigations.append("Investigate feature interactions that may be reducing recall")
            mitigations.append("Consider ensemble approaches or threshold adjustment")
        
        if any("complexity" in risk for risk in risks):
            mitigations.append("Perform feature importance analysis to identify redundant features")
            mitigations.append("Consider regularization to manage complexity")
        
        if any("training time" in risk for risk in risks):
            mitigations.append("Optimize feature computation and caching")
            mitigations.append("Consider feature selection to reduce computational burden")
        
        if any("variance" in risk for risk in risks):
            mitigations.append("Increase cross-validation folds for more stable estimates")
            mitigations.append("Investigate data-dependent feature instability")
        
        return mitigations
    
    def _assess_confidence(self, metric_comparisons: Dict[str, MetricComparison]) -> str:
        """Assess confidence in the comparison results."""
        significant_count = sum(1 for comp in metric_comparisons.values() if comp.is_significant)
        meaningful_count = sum(1 for comp in metric_comparisons.values() if comp.is_meaningful)
        total_metrics = len(metric_comparisons)
        
        # Check p-value consistency
        p_values = [comp.statistical_tests[0].p_value for comp in metric_comparisons.values() 
                   if comp.statistical_tests]
        min_p_value = min(p_values) if p_values else 1.0
        
        if significant_count >= 3 and meaningful_count >= 2 and min_p_value < 0.01:
            return "HIGH: Multiple metrics show strong statistical significance"
        elif significant_count >= 2 and meaningful_count >= 1:
            return "MEDIUM: Some metrics show statistical significance with practical relevance"
        elif significant_count >= 1:
            return "LOW: Limited statistical evidence for meaningful differences"
        else:
            return "VERY LOW: No statistically significant differences detected"
    
    def save_comparison_report(self, report: ComparisonReport, output_dir: Path) -> str:
        """
        Save comparison report to file.
        
        Args:
            report: Comparison report to save
            output_dir: Directory to save the report
            
        Returns:
            Path to saved report file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        timestamp = report.timestamp.strftime('%Y%m%d_%H%M%S')
        report_path = output_dir / f"feature_comparison_{timestamp}.json"
        
        # Convert to serializable format
        report_dict = asdict(report)
        
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
        
        report_dict = recursive_convert(report_dict)
        
        with open(report_path, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        logger.info(f"Comparison report saved to {report_path}")
        return str(report_path)
    
    def visualize_comparison(self, report: ComparisonReport, output_dir: Path) -> List[str]:
        """
        Create visualizations for the comparison report.
        
        Args:
            report: Comparison report to visualize
            output_dir: Directory to save visualizations
            
        Returns:
            List of paths to generated visualization files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        timestamp = report.timestamp.strftime('%Y%m%d_%H%M%S')
        plot_paths = []
        
        # 1. Metric comparison bar plot
        plt.figure(figsize=(12, 8))
        metrics = list(report.metric_comparisons.keys())
        baseline_means = [report.metric_comparisons[m].baseline_mean for m in metrics]
        test_means = [report.metric_comparisons[m].test_mean for m in metrics]
        baseline_stds = [report.metric_comparisons[m].baseline_std for m in metrics]
        test_stds = [report.metric_comparisons[m].test_std for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.bar(x - width/2, baseline_means, width, yerr=baseline_stds, 
                label='Baseline', alpha=0.8, capsize=5)
        plt.bar(x + width/2, test_means, width, yerr=test_stds, 
                label='Test', alpha=0.8, capsize=5)
        
        plt.xlabel('Metrics')
        plt.ylabel('Performance')
        plt.title('Feature Configuration Comparison')
        plt.xticks(x, metrics)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        bar_plot_path = output_dir / f"metric_comparison_bar_{timestamp}.png"
        plt.savefig(bar_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(str(bar_plot_path))
        
        # 2. Effect size plot
        plt.figure(figsize=(10, 6))
        effect_sizes = []
        metric_names = []
        colors = []
        
        for metric, comparison in report.metric_comparisons.items():
            if comparison.statistical_tests:
                effect_size = comparison.statistical_tests[0].effect_size
                if not np.isnan(effect_size):
                    effect_sizes.append(effect_size)
                    metric_names.append(metric)
                    colors.append('green' if effect_size > 0 else 'red')
        
        if effect_sizes:
            plt.barh(metric_names, effect_sizes, color=colors, alpha=0.7)
            plt.xlabel('Effect Size (Cohen\'s d)')
            plt.title('Effect Sizes by Metric')
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
            plt.axvline(x=self.min_effect_size, color='blue', linestyle='--', alpha=0.5, 
                       label=f'Min meaningful effect ({self.min_effect_size})')
            plt.axvline(x=-self.min_effect_size, color='blue', linestyle='--', alpha=0.5)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            effect_plot_path = output_dir / f"effect_sizes_{timestamp}.png"
            plt.savefig(effect_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths.append(str(effect_plot_path))
        
        return plot_paths