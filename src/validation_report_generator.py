"""
Validation Report Generator Module

This module provides comprehensive report generation capabilities for
feature validation results, creating actionable insights for decision-makers
through detailed analysis, visualizations, and executive summaries.

Classes:
    ValidationReportGenerator: Main class for generating validation reports
    ExecutiveSummary: Executive summary data structure
    TechnicalAnalysis: Technical analysis data structure
    RecommendationEngine: Automated recommendation generation
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template

# Local imports
from src.feature_optimization_validator import FeatureTestResult
from src.feature_set_comparator import ComparisonReport
from src.validation_metrics_collector import ValidationResults
from src.automated_feature_testing import AutomatedTestResults

logger = logging.getLogger(__name__)

@dataclass
class ExecutiveSummary:
    """Executive summary of validation results."""
    key_findings: List[str]
    performance_highlights: Dict[str, float]
    recommended_action: str
    risk_assessment: str
    business_impact: str
    implementation_effort: str
    timeline_estimate: str
    roi_projection: str

@dataclass
class TechnicalAnalysis:
    """Technical analysis of validation results."""
    optimal_configuration: List[str]
    performance_comparison: Dict[str, Dict[str, float]]
    feature_importance_ranking: List[Tuple[str, float]]
    interaction_effects: List[Dict[str, Any]]
    redundancy_analysis: Dict[str, List[str]]
    stability_assessment: Dict[str, float]
    computational_impact: Dict[str, float]

@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    report_id: str
    generation_timestamp: datetime
    executive_summary: ExecutiveSummary
    technical_analysis: TechnicalAnalysis
    detailed_results: Dict[str, Any]
    visualizations: List[str]
    recommendations: List[str]
    appendices: Dict[str, Any]

class ValidationReportGenerator:
    """
    Comprehensive validation report generator.
    
    This class creates detailed, actionable reports from validation results,
    including executive summaries, technical analysis, visualizations,
    and specific recommendations for implementation.
    """
    
    def __init__(self, output_dir: str = "data/output/validation"):
        """
        Initialize the validation report generator.
        
        Args:
            output_dir: Directory for saving reports and visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "reports").mkdir(exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)
        
        logger.info(f"Initialized ValidationReportGenerator with output dir: {self.output_dir}")
    
    def generate_comprehensive_report(self, 
                                    optimization_results: Dict[str, Any],
                                    comparison_results: Optional[List[ComparisonReport]] = None,
                                    automated_test_results: Optional[AutomatedTestResults] = None,
                                    baseline_results: Optional[FeatureTestResult] = None) -> ValidationReport:
        """
        Generate comprehensive validation report.
        
        Args:
            optimization_results: Results from feature optimization
            comparison_results: Results from feature comparisons
            automated_test_results: Results from automated testing
            baseline_results: Baseline performance results
            
        Returns:
            Comprehensive validation report
        """
        report_id = f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Generating comprehensive validation report: {report_id}")
        
        # Generate executive summary
        executive_summary = self._generate_executive_summary(
            optimization_results, comparison_results, automated_test_results, baseline_results
        )
        
        # Generate technical analysis
        technical_analysis = self._generate_technical_analysis(
            optimization_results, comparison_results, automated_test_results
        )
        
        # Create visualizations
        visualizations = self._create_visualizations(
            optimization_results, comparison_results, automated_test_results, report_id
        )
        
        # Generate recommendations
        recommendations = self._generate_detailed_recommendations(
            optimization_results, comparison_results, automated_test_results
        )
        
        # Compile detailed results
        detailed_results = self._compile_detailed_results(
            optimization_results, comparison_results, automated_test_results
        )
        
        # Create appendices
        appendices = self._create_appendices(
            optimization_results, comparison_results, automated_test_results
        )
        
        report = ValidationReport(
            report_id=report_id,
            generation_timestamp=datetime.now(),
            executive_summary=executive_summary,
            technical_analysis=technical_analysis,
            detailed_results=detailed_results,
            visualizations=visualizations,
            recommendations=recommendations,
            appendices=appendices
        )
        
        # Save report
        self._save_report(report)
        
        # Generate HTML report
        self._generate_html_report(report)
        
        return report
    
    def _generate_executive_summary(self, optimization_results: Dict[str, Any],
                                  comparison_results: Optional[List[ComparisonReport]],
                                  automated_test_results: Optional[AutomatedTestResults],
                                  baseline_results: Optional[FeatureTestResult]) -> ExecutiveSummary:
        """Generate executive summary."""
        key_findings = []
        performance_highlights = {}
        
        # Extract key findings from optimization results
        if optimization_results:
            best_config = optimization_results.get("best_configuration")
            if best_config:
                best_f1 = best_config.mean_scores.get("f1", 0.0)
                feature_count = len(best_config.feature_set)
                
                key_findings.append(f"Optimal configuration identified with {feature_count} features achieving {best_f1:.1%} F1-score")
                performance_highlights["best_f1_score"] = best_f1
                performance_highlights["optimal_feature_count"] = feature_count
                
                if baseline_results:
                    baseline_f1 = baseline_results.mean_scores.get("f1", 0.0)
                    improvement = best_f1 - baseline_f1
                    performance_highlights["f1_improvement"] = improvement
                    
                    if improvement > 0.01:
                        key_findings.append(f"Significant improvement of {improvement:.1%} F1-score over baseline")
                    else:
                        key_findings.append("Limited improvement over baseline configuration")
        
        # Extract findings from automated testing
        if automated_test_results:
            if automated_test_results.redundancy_groups:
                total_redundant = sum(len(group.elimination_candidates) 
                                    for group in automated_test_results.redundancy_groups)
                key_findings.append(f"Identified {total_redundant} potentially redundant features for elimination")
            
            strong_interactions = [i for i in automated_test_results.feature_interactions 
                                 if i.interaction_type == "synergistic" and i.interaction_strength > 0.05]
            if strong_interactions:
                key_findings.append(f"Discovered {len(strong_interactions)} strong synergistic feature interactions")
        
        # Generate recommendation
        if performance_highlights.get("f1_improvement", 0) > 0.02:
            recommended_action = "IMPLEMENT: Strong evidence for adopting optimized configuration"
            risk_assessment = "LOW: Consistent improvements with minimal risk"
        elif performance_highlights.get("f1_improvement", 0) > 0.005:
            recommended_action = "PILOT: Test optimized configuration in production with monitoring"
            risk_assessment = "MEDIUM: Modest improvements, recommend careful monitoring"
        else:
            recommended_action = "INVESTIGATE: No clear improvement, consider alternative approaches"
            risk_assessment = "LOW: No evidence of degradation, but limited benefit"
        
        # Business impact assessment
        f1_improvement = performance_highlights.get("f1_improvement", 0)
        if f1_improvement > 0.02:
            business_impact = f"Significant operational improvement: ~{f1_improvement*100:.1f}% better match detection"
        elif f1_improvement > 0.005:
            business_impact = f"Moderate operational improvement: ~{f1_improvement*100:.1f}% better match detection"
        else:
            business_impact = "Minimal operational impact expected"
        
        # Implementation effort
        feature_count = performance_highlights.get("optimal_feature_count", 0)
        if feature_count <= 5:
            implementation_effort = "LOW: Simple configuration change"
            timeline_estimate = "1-2 weeks including testing"
        elif feature_count <= 8:
            implementation_effort = "MEDIUM: Configuration update with validation"
            timeline_estimate = "2-4 weeks including validation"
        else:
            implementation_effort = "HIGH: Complex configuration requiring careful testing"
            timeline_estimate = "4-8 weeks including extensive validation"
        
        # ROI projection
        if f1_improvement > 0.02:
            roi_projection = "HIGH: Significant productivity gains expected"
        elif f1_improvement > 0.005:
            roi_projection = "MEDIUM: Moderate productivity gains expected"
        else:
            roi_projection = "LOW: Limited productivity impact"
        
        return ExecutiveSummary(
            key_findings=key_findings,
            performance_highlights=performance_highlights,
            recommended_action=recommended_action,
            risk_assessment=risk_assessment,
            business_impact=business_impact,
            implementation_effort=implementation_effort,
            timeline_estimate=timeline_estimate,
            roi_projection=roi_projection
        )
    
    def _generate_technical_analysis(self, optimization_results: Dict[str, Any],
                                   comparison_results: Optional[List[ComparisonReport]],
                                   automated_test_results: Optional[AutomatedTestResults]) -> TechnicalAnalysis:
        """Generate technical analysis section."""
        # Extract optimal configuration
        optimal_config = []
        if optimization_results and optimization_results.get("best_configuration"):
            optimal_config = optimization_results["best_configuration"].feature_set
        
        # Performance comparison
        performance_comparison = {}
        if optimization_results:
            best_result = optimization_results.get("best_configuration")
            if best_result:
                performance_comparison["optimized"] = best_result.mean_scores
        
        if comparison_results:
            for comparison in comparison_results:
                performance_comparison["baseline"] = comparison.baseline_result.mean_scores
                performance_comparison["test"] = comparison.test_result.mean_scores
                break  # Use first comparison for now
        
        # Feature importance ranking
        feature_importance = []
        if automated_test_results:
            for contribution in automated_test_results.individual_contributions:
                solo_f1 = contribution.solo_performance.get("f1", 0.0)
                feature_importance.append((contribution.feature_name, solo_f1))
        
        # Interaction effects
        interaction_effects = []
        if automated_test_results:
            for interaction in automated_test_results.feature_interactions[:5]:  # Top 5
                effect_dict = {
                    "feature_pair": interaction.feature_pair,
                    "type": interaction.interaction_type,
                    "strength": interaction.interaction_strength,
                    "effect": interaction.interaction_effect
                }
                interaction_effects.append(effect_dict)
        
        # Redundancy analysis
        redundancy_analysis = {}
        if automated_test_results:
            for group in automated_test_results.redundancy_groups:
                redundancy_analysis[group.representative_feature] = group.elimination_candidates
        
        # Stability assessment
        stability_assessment = {}
        if automated_test_results:
            for contribution in automated_test_results.individual_contributions:
                stability_assessment[contribution.feature_name] = contribution.stability_score
        
        # Computational impact
        computational_impact = {}
        if optimization_results:
            best_result = optimization_results.get("best_configuration")
            if best_result:
                computational_impact["training_time"] = best_result.training_time
                computational_impact["feature_count"] = len(best_result.feature_set)
                computational_impact["complexity_score"] = len(best_result.feature_set) * best_result.training_time
        
        return TechnicalAnalysis(
            optimal_configuration=optimal_config,
            performance_comparison=performance_comparison,
            feature_importance_ranking=feature_importance,
            interaction_effects=interaction_effects,
            redundancy_analysis=redundancy_analysis,
            stability_assessment=stability_assessment,
            computational_impact=computational_impact
        )
    
    def _create_visualizations(self, optimization_results: Dict[str, Any],
                             comparison_results: Optional[List[ComparisonReport]],
                             automated_test_results: Optional[AutomatedTestResults],
                             report_id: str) -> List[str]:
        """Create visualizations for the report."""
        viz_paths = []
        viz_dir = self.output_dir / "visualizations"
        
        # 1. Feature Performance Comparison
        if automated_test_results:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            features = [c.feature_name for c in automated_test_results.individual_contributions[:10]]
            f1_scores = [c.solo_performance.get("f1", 0.0) for c in automated_test_results.individual_contributions[:10]]
            
            bars = ax.bar(range(len(features)), f1_scores, alpha=0.7)
            ax.set_xlabel("Features")
            ax.set_ylabel("F1 Score")
            ax.set_title("Individual Feature Performance")
            ax.set_xticks(range(len(features)))
            ax.set_xticklabels(features, rotation=45, ha='right')
            
            # Color bars based on performance
            for i, bar in enumerate(bars):
                if f1_scores[i] > 0.7:
                    bar.set_color('green')
                elif f1_scores[i] > 0.5:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')
            
            plt.tight_layout()
            viz_path = viz_dir / f"{report_id}_feature_performance.png"
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            viz_paths.append(str(viz_path))
        
        # 2. Performance Metrics Comparison
        if comparison_results:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            comparison = comparison_results[0]  # Use first comparison
            metrics = ['precision', 'recall', 'f1', 'auc']
            
            for i, metric in enumerate(metrics):
                ax = axes[i // 2, i % 2]
                
                baseline_val = comparison.baseline_result.mean_scores[metric]
                test_val = comparison.test_result.mean_scores[metric]
                baseline_std = comparison.baseline_result.std_scores[metric]
                test_std = comparison.test_result.std_scores[metric]
                
                x = ['Baseline', 'Optimized']
                y = [baseline_val, test_val]
                yerr = [baseline_std, test_std]
                
                bars = ax.bar(x, y, yerr=yerr, capsize=5, alpha=0.7)
                ax.set_ylabel(metric.title())
                ax.set_title(f"{metric.title()} Comparison")
                
                # Color based on improvement
                if test_val > baseline_val:
                    bars[1].set_color('green')
                else:
                    bars[1].set_color('red')
            
            plt.tight_layout()
            viz_path = viz_dir / f"{report_id}_metrics_comparison.png"
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            viz_paths.append(str(viz_path))
        
        # 3. Feature Interaction Network
        if automated_test_results and automated_test_results.feature_interactions:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create interaction strength heatmap
            interactions = automated_test_results.feature_interactions
            features = list(set([f for interaction in interactions for f in interaction.feature_pair]))
            
            if len(features) > 1:
                n_features = len(features)
                interaction_matrix = np.zeros((n_features, n_features))
                
                for interaction in interactions:
                    f1, f2 = interaction.feature_pair
                    try:
                        i1, i2 = features.index(f1), features.index(f2)
                        interaction_matrix[i1, i2] = interaction.interaction_strength
                        interaction_matrix[i2, i1] = interaction.interaction_strength
                    except ValueError:
                        continue
                
                sns.heatmap(interaction_matrix, 
                           xticklabels=features, 
                           yticklabels=features,
                           annot=True, 
                           fmt='.3f',
                           cmap='RdYlBu_r',
                           ax=ax)
                ax.set_title("Feature Interaction Strength Matrix")
                
                plt.tight_layout()
                viz_path = viz_dir / f"{report_id}_interaction_matrix.png"
                plt.savefig(viz_path, dpi=300, bbox_inches='tight')
                plt.close()
                viz_paths.append(str(viz_path))
        
        # 4. Optimization Progress
        if optimization_results and optimization_results.get("optimization_results"):
            fig, ax = plt.subplots(figsize=(10, 6))
            
            results = optimization_results["optimization_results"]
            if isinstance(results, list) and len(results) > 1:
                iterations = range(1, len(results) + 1)
                f1_scores = [result.mean_scores.get("f1", 0.0) for result in results]
                
                ax.plot(iterations, f1_scores, marker='o', linewidth=2, markersize=6)
                ax.set_xlabel("Optimization Iteration")
                ax.set_ylabel("F1 Score")
                ax.set_title("Optimization Progress")
                ax.grid(True, alpha=0.3)
                
                # Highlight best result
                best_idx = np.argmax(f1_scores)
                ax.scatter(iterations[best_idx], f1_scores[best_idx], 
                          color='red', s=100, zorder=5, label=f'Best: {f1_scores[best_idx]:.4f}')
                ax.legend()
                
                plt.tight_layout()
                viz_path = viz_dir / f"{report_id}_optimization_progress.png"
                plt.savefig(viz_path, dpi=300, bbox_inches='tight')
                plt.close()
                viz_paths.append(str(viz_path))
        
        return viz_paths
    
    def _generate_detailed_recommendations(self, optimization_results: Dict[str, Any],
                                         comparison_results: Optional[List[ComparisonReport]],
                                         automated_test_results: Optional[AutomatedTestResults]) -> List[str]:
        """Generate detailed recommendations."""
        recommendations = []
        
        # Configuration recommendations
        if optimization_results:
            best_config = optimization_results.get("best_configuration")
            if best_config:
                recommendations.append(
                    f"IMPLEMENTATION: Deploy configuration with {len(best_config.feature_set)} features: "
                    f"{', '.join(best_config.feature_set)}"
                )
        
        # Feature elimination recommendations
        if automated_test_results and automated_test_results.redundancy_groups:
            for group in automated_test_results.redundancy_groups:
                if group.elimination_candidates:
                    recommendations.append(
                        f"REDUNDANCY: Consider eliminating {', '.join(group.elimination_candidates)} "
                        f"while keeping {group.representative_feature}"
                    )
        
        # Interaction recommendations
        if automated_test_results:
            strong_synergies = [i for i in automated_test_results.feature_interactions 
                               if i.interaction_type == "synergistic" and i.interaction_strength > 0.1]
            for synergy in strong_synergies[:3]:  # Top 3
                recommendations.append(
                    f"SYNERGY: Leverage strong interaction between "
                    f"{synergy.feature_pair[0]} and {synergy.feature_pair[1]}"
                )
        
        # Performance monitoring recommendations
        if comparison_results:
            for comparison in comparison_results:
                if comparison.risk_assessment["risk_level"] != "LOW":
                    recommendations.append(
                        f"MONITORING: {comparison.risk_assessment['risk_level']} risk detected - "
                        f"implement careful monitoring during deployment"
                    )
        
        # Testing recommendations
        recommendations.append(
            "VALIDATION: Conduct A/B testing in production environment before full deployment"
        )
        
        recommendations.append(
            "MONITORING: Establish performance monitoring to track real-world effectiveness"
        )
        
        return recommendations
    
    def _compile_detailed_results(self, optimization_results: Dict[str, Any],
                                comparison_results: Optional[List[ComparisonReport]],
                                automated_test_results: Optional[AutomatedTestResults]) -> Dict[str, Any]:
        """Compile detailed results section."""
        return {
            "optimization_results": optimization_results,
            "comparison_results": [asdict(comp) for comp in comparison_results] if comparison_results else [],
            "automated_test_results": asdict(automated_test_results) if automated_test_results else {},
            "metadata": {
                "generation_time": datetime.now().isoformat(),
                "validation_framework_version": "1.0.0"
            }
        }
    
    def _create_appendices(self, optimization_results: Dict[str, Any],
                         comparison_results: Optional[List[ComparisonReport]],
                         automated_test_results: Optional[AutomatedTestResults]) -> Dict[str, Any]:
        """Create appendices with supporting information."""
        appendices = {}
        
        # Statistical analysis appendix
        if comparison_results:
            statistical_summary = {}
            for comparison in comparison_results:
                for metric, metric_comparison in comparison.metric_comparisons.items():
                    statistical_summary[metric] = {
                        "p_values": [test.p_value for test in metric_comparison.statistical_tests],
                        "effect_sizes": [test.effect_size for test in metric_comparison.statistical_tests],
                        "significance": metric_comparison.is_significant
                    }
            appendices["statistical_analysis"] = statistical_summary
        
        # Feature analysis appendix
        if automated_test_results:
            feature_analysis = {
                "total_features_tested": len(automated_test_results.individual_contributions),
                "redundancy_groups_found": len(automated_test_results.redundancy_groups),
                "significant_interactions": len([i for i in automated_test_results.feature_interactions 
                                               if i.statistical_significance < 0.05]),
                "performance_distribution": {
                    "min_f1": min(c.solo_performance.get("f1", 0.0) 
                                 for c in automated_test_results.individual_contributions),
                    "max_f1": max(c.solo_performance.get("f1", 0.0) 
                                 for c in automated_test_results.individual_contributions),
                    "mean_f1": np.mean([c.solo_performance.get("f1", 0.0) 
                                       for c in automated_test_results.individual_contributions])
                }
            }
            appendices["feature_analysis"] = feature_analysis
        
        # Configuration history appendix
        if optimization_results:
            config_history = {
                "search_strategy": optimization_results.get("report", {}).get("search_strategy", "unknown"),
                "total_combinations_tested": optimization_results.get("report", {}).get("total_combinations_tested", 0),
                "optimization_duration": "estimated_from_results"
            }
            appendices["configuration_history"] = config_history
        
        return appendices
    
    def _save_report(self, report: ValidationReport) -> None:
        """Save report to JSON file."""
        report_path = self.output_dir / "reports" / f"{report.report_id}.json"
        
        # Convert to serializable format
        report_dict = asdict(report)
        
        # Handle datetime and other non-serializable objects
        def convert_types(obj):
            if hasattr(obj, 'isoformat'):
                return obj.isoformat()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.float64):
                return float(obj)
            elif isinstance(obj, np.int64):
                return int(obj)
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
        
        logger.info(f"Validation report saved to {report_path}")
    
    def _generate_html_report(self, report: ValidationReport) -> None:
        """Generate HTML report."""
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Feature Validation Report - {{ report.report_id }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
        .header { background-color: #f4f4f4; padding: 20px; border-radius: 5px; }
        .section { margin: 30px 0; }
        .highlight { background-color: #e8f4f8; padding: 15px; border-radius: 5px; }
        .recommendation { background-color: #f0f8f0; padding: 10px; margin: 10px 0; border-left: 4px solid #4CAF50; }
        .warning { background-color: #fff8e1; padding: 10px; margin: 10px 0; border-left: 4px solid #FF9800; }
        .metric { display: inline-block; margin: 10px; padding: 10px; background-color: #f9f9f9; border-radius: 3px; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .visualization { text-align: center; margin: 20px 0; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Feature Validation Report</h1>
        <p><strong>Report ID:</strong> {{ report.report_id }}</p>
        <p><strong>Generated:</strong> {{ report.generation_timestamp.strftime('%Y-%m-%d %H:%M:%S') }}</p>
    </div>

    <div class="section">
        <h2>Executive Summary</h2>
        <div class="highlight">
            <h3>Recommended Action</h3>
            <p><strong>{{ report.executive_summary.recommended_action }}</strong></p>
            
            <h3>Key Findings</h3>
            <ul>
            {% for finding in report.executive_summary.key_findings %}
                <li>{{ finding }}</li>
            {% endfor %}
            </ul>
            
            <h3>Business Impact</h3>
            <p>{{ report.executive_summary.business_impact }}</p>
            
            <h3>Implementation</h3>
            <p><strong>Effort:</strong> {{ report.executive_summary.implementation_effort }}</p>
            <p><strong>Timeline:</strong> {{ report.executive_summary.timeline_estimate }}</p>
            <p><strong>ROI:</strong> {{ report.executive_summary.roi_projection }}</p>
        </div>
    </div>

    <div class="section">
        <h2>Performance Metrics</h2>
        {% for metric, value in report.executive_summary.performance_highlights.items() %}
        <div class="metric">
            <strong>{{ metric.replace('_', ' ').title() }}:</strong> 
            {% if value < 1 %}{{ "%.2f%%" | format(value * 100) }}{% else %}{{ value }}{% endif %}
        </div>
        {% endfor %}
    </div>

    <div class="section">
        <h2>Technical Analysis</h2>
        <h3>Optimal Configuration</h3>
        <p><strong>Recommended Features:</strong></p>
        <ul>
        {% for feature in report.technical_analysis.optimal_configuration %}
            <li>{{ feature }}</li>
        {% endfor %}
        </ul>
        
        {% if report.technical_analysis.redundancy_analysis %}
        <h3>Redundancy Analysis</h3>
        {% for representative, candidates in report.technical_analysis.redundancy_analysis.items() %}
        <div class="warning">
            <strong>Keep:</strong> {{ representative }}<br>
            <strong>Consider removing:</strong> {{ candidates | join(', ') }}
        </div>
        {% endfor %}
        {% endif %}
    </div>

    <div class="section">
        <h2>Recommendations</h2>
        {% for recommendation in report.recommendations %}
        <div class="recommendation">{{ recommendation }}</div>
        {% endfor %}
    </div>

    {% if report.visualizations %}
    <div class="section">
        <h2>Visualizations</h2>
        {% for viz_path in report.visualizations %}
        <div class="visualization">
            <img src="{{ viz_path | basename }}" alt="Visualization" style="max-width: 100%; height: auto;">
        </div>
        {% endfor %}
    </div>
    {% endif %}

    <div class="section">
        <h2>Risk Assessment</h2>
        <p>{{ report.executive_summary.risk_assessment }}</p>
    </div>
</body>
</html>
        """
        
        template = Template(html_template)
        html_content = template.render(report=report)
        
        html_path = self.output_dir / "reports" / f"{report.report_id}.html"
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML report generated: {html_path}")

    def generate_quick_summary(self, results: Dict[str, Any]) -> str:
        """Generate quick text summary for immediate feedback."""
        if not results:
            return "No results to summarize."
        
        best_config = results.get("best_configuration")
        if not best_config:
            return "No optimal configuration found."
        
        f1_score = best_config.mean_scores.get("f1", 0.0)
        feature_count = len(best_config.feature_set)
        
        summary = f"""
FEATURE OPTIMIZATION SUMMARY
===========================
Best Configuration: {feature_count} features
F1-Score: {f1_score:.1%}
Features: {', '.join(best_config.feature_set)}

Performance:
- Precision: {best_config.mean_scores.get('precision', 0.0):.1%}
- Recall: {best_config.mean_scores.get('recall', 0.0):.1%}
- F1-Score: {f1_score:.1%}
- AUC: {best_config.mean_scores.get('auc', 0.0):.1%}

Recommendation: {'IMPLEMENT' if f1_score > 0.9 else 'INVESTIGATE' if f1_score > 0.8 else 'RECONSIDER'}
        """
        
        return summary.strip()