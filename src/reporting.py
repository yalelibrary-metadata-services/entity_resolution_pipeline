"""
Reporting module for entity resolution pipeline.

This module generates reports and visualizations for entity resolution results.
"""

import os
import logging
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc, confusion_matrix
)

logger = logging.getLogger(__name__)

class ReportGenerator:
    """
    Generates reports and visualizations for entity resolution results.
    
    Features:
    - Classification metrics reporting
    - Feature importance visualization
    - Test results analysis and visualization
    - Cluster statistics reporting
    - Confusion matrix visualization
    - Precision-recall curve visualization
    """
    
    def __init__(self, config):
        """
        Initialize the report generator with configuration parameters.
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = config
        
        # Reporting configuration
        self.generate_metrics_report = config['reporting']['generate_metrics_report']
        self.generate_feature_importance = config['reporting']['generate_feature_importance']
        self.generate_error_analysis = config['reporting']['generate_error_analysis']
        self.generate_cluster_statistics = config['reporting']['generate_cluster_statistics']
        self.visualization_enabled = config['reporting']['visualization_enabled']
        self.metrics_to_report = config['reporting']['metrics_to_report']
        
        # Data paths
        self.output_dir = Path(config['system']['output_dir'])
        self.reports_dir = self.output_dir / "reports"
        self.figures_dir = self.output_dir / "figures"
        
        # Create directories if they don't exist
        for dir_path in [self.reports_dir, self.figures_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("ReportGenerator initialized")
    
    def execute(self):
        """
        Execute report generation.
        
        Returns:
            dict: Report generation results
        """
        # Track generated reports and figures
        generated_reports = []
        generated_figures = []
        
        # Generate classification metrics report
        if self.generate_metrics_report:
            metrics_report = self._generate_metrics_report()
            if metrics_report:
                generated_reports.append(metrics_report)
        
        # Generate feature importance visualization
        if self.generate_feature_importance:
            feature_importance_report = self._generate_feature_importance()
            if feature_importance_report:
                generated_reports.append(feature_importance_report)
        
        # Generate test results analysis
        test_results_report = self._analyze_test_results()
        if test_results_report:
            generated_reports.append(test_results_report)
        
        # Generate error analysis
        if self.generate_error_analysis:
            error_analysis_report = self._generate_error_analysis()
            if error_analysis_report:
                generated_reports.append(error_analysis_report)
        
        # Generate cluster statistics
        if self.generate_cluster_statistics:
            cluster_stats_report = self._generate_cluster_statistics()
            if cluster_stats_report:
                generated_reports.append(cluster_stats_report)
        
        # Generate visualizations
        if self.visualization_enabled:
            figures = self._generate_visualizations()
            generated_figures.extend(figures)
        
        results = {
            'reports': generated_reports,
            'figures': generated_figures
        }
        
        logger.info(
            f"Report generation completed: {len(generated_reports)} reports, "
            f"{len(generated_figures)} figures"
        )
        
        return results
    
    def _generate_metrics_report(self):
        """
        Generate classification metrics report.
        
        Returns:
            str: Path to report file
        """
        logger.info("Generating classification metrics report")
        
        # Load classification metrics
        metrics_path = self.output_dir / "classification_metrics.json"
        if not metrics_path.exists():
            logger.warning("Classification metrics not found")
            return None
        
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # Create metrics report
        report = {
            'title': 'Entity Resolution Classification Metrics',
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': {}
        }
        
        # Add requested metrics
        for metric_name in self.metrics_to_report:
            if metric_name in metrics:
                report['metrics'][metric_name] = metrics[metric_name]
        
        # Save report
        report_path = self.reports_dir / "classification_metrics_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Create readable summary in Markdown
        summary_path = self.reports_dir / "classification_metrics_summary.md"
        with open(summary_path, 'w') as f:
            f.write(f"# Entity Resolution Classification Metrics\n\n")
            f.write(f"Generated: {report['timestamp']}\n\n")
            
            f.write("## Performance Metrics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            
            for metric_name, value in report['metrics'].items():
                if metric_name != 'confusion_matrix' and metric_name != 'feature_importance':
                    if isinstance(value, (int, float)):
                        f.write(f"| {metric_name.capitalize()} | {value:.4f} |\n")
                    else:
                        f.write(f"| {metric_name.capitalize()} | {value} |\n")
            
            if 'confusion_matrix' in report['metrics']:
                cm = report['metrics']['confusion_matrix']
                f.write("\n## Confusion Matrix\n\n")
                f.write("| | Predicted Negative | Predicted Positive |\n")
                f.write("|---------------------|--------------------|\n")
                f.write(f"| **Actual Negative** | {cm['true_negatives']} | {cm['false_positives']} |\n")
                f.write(f"| **Actual Positive** | {cm['false_negatives']} | {cm['true_positives']} |\n")
                
                # Calculate derived metrics
                total = cm['true_negatives'] + cm['false_positives'] + cm['false_negatives'] + cm['true_positives']
                accuracy = (cm['true_negatives'] + cm['true_positives']) / total if total > 0 else 0
                precision = cm['true_positives'] / (cm['true_positives'] + cm['false_positives']) if (cm['true_positives'] + cm['false_positives']) > 0 else 0
                recall = cm['true_positives'] / (cm['true_positives'] + cm['false_negatives']) if (cm['true_positives'] + cm['false_negatives']) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                f.write("\n## Derived Metrics\n\n")
                f.write(f"- **Accuracy**: {accuracy:.4f}\n")
                f.write(f"- **Precision**: {precision:.4f}\n")
                f.write(f"- **Recall**: {recall:.4f}\n")
                f.write(f"- **F1 Score**: {f1:.4f}\n")
        
        logger.info(f"Classification metrics report saved to {report_path}")
        logger.info(f"Classification metrics summary saved to {summary_path}")
        
        return str(report_path)
    
    def _generate_feature_importance(self):
        """
        Generate feature importance report.
        
        Returns:
            str: Path to report file
        """
        logger.info("Generating feature importance report")
        
        # Load classification metrics
        metrics_path = self.output_dir / "classification_metrics.json"
        if not metrics_path.exists():
            logger.warning("Classification metrics not found")
            return None
        
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # Get feature importance
        if 'feature_importance' not in metrics:
            logger.warning("Feature importance not found in metrics")
            return None
        
        feature_importance = metrics['feature_importance']
        
        # Create feature importance report
        report = {
            'title': 'Entity Resolution Feature Importance',
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'feature_importance': feature_importance
        }
        
        # Save report
        report_path = self.reports_dir / "feature_importance_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Create readable summary in Markdown
        summary_path = self.reports_dir / "feature_importance_summary.md"
        with open(summary_path, 'w') as f:
            f.write(f"# Entity Resolution Feature Importance\n\n")
            f.write(f"Generated: {report['timestamp']}\n\n")
            
            f.write("## Feature Importance Ranking\n\n")
            f.write("| Feature | Weight | Absolute Weight | Importance |\n")
            f.write("|---------|--------|----------------|------------|\n")
            
            for feature_name, details in feature_importance.items():
                weight = details['weight']
                abs_weight = details['abs_weight']
                importance = details['importance']
                
                f.write(f"| {feature_name} | {weight:.4f} | {abs_weight:.4f} | {importance:.4f} |\n")
            
            # Group features by type
            feature_types = {}
            for feature_name, details in feature_importance.items():
                # Extract feature type from name (e.g., person_cosine -> cosine)
                feature_type = feature_name.split('_')[-1] if '_' in feature_name else 'other'
                
                if feature_type not in feature_types:
                    feature_types[feature_type] = 0
                
                feature_types[feature_type] += details['importance']
            
            # Sort by importance
            feature_types = {k: v for k, v in sorted(feature_types.items(), key=lambda item: item[1], reverse=True)}
            
            f.write("\n## Feature Type Importance\n\n")
            f.write("| Feature Type | Importance |\n")
            f.write("|--------------|------------|\n")
            
            for feature_type, importance in feature_types.items():
                f.write(f"| {feature_type} | {importance:.4f} |\n")
            
            # Generate insights on feature importance
            f.write("\n## Insights\n\n")
            f.write("### Top Features\n")
            top_features = list(feature_importance.keys())[:5]
            for i, feature in enumerate(top_features, 1):
                details = feature_importance[feature]
                f.write(f"{i}. **{feature}** (Importance: {details['importance']:.4f})\n")
            
            f.write("\n### Top Feature Types\n")
            top_types = list(feature_types.keys())[:3]
            for i, feature_type in enumerate(top_types, 1):
                f.write(f"{i}. **{feature_type}** (Importance: {feature_types[feature_type]:.4f})\n")
        
        # Generate CSV export of feature importance
        feature_importance_df = pd.DataFrame([
            {
                'feature': feature_name,
                'weight': details['weight'],
                'abs_weight': details['abs_weight'],
                'importance': details['importance'],
                'feature_type': feature_name.split('_')[-1] if '_' in feature_name else 'other'
            }
            for feature_name, details in feature_importance.items()
        ])
        
        # Save to CSV
        csv_path = self.output_dir / "feature_importance.csv"
        feature_importance_df.to_csv(csv_path, index=False)
        logger.info(f"Feature importance CSV saved to {csv_path}")
        
        # Visualize feature importance
        if self.visualization_enabled:
            try:
                # Create DataFrame from feature importance
                df = feature_importance_df.sort_values('importance', ascending=False)
                
                # Plot top features
                plt.figure(figsize=(12, 8))
                
                # Bar plot of feature importance
                plt.subplot(2, 1, 1)
                sns.barplot(x='importance', y='feature', data=df.head(15), palette='viridis')
                plt.title('Top 15 Features by Importance')
                plt.xlabel('Importance')
                plt.ylabel('Feature')
                plt.tight_layout()
                
                # Bar plot of feature weights
                plt.subplot(2, 1, 2)
                sns.barplot(x='weight', y='feature', data=df.head(15), palette='viridis')
                plt.title('Top 15 Features by Weight')
                plt.xlabel('Weight')
                plt.ylabel('Feature')
                plt.tight_layout()
                
                # Save figure
                fig_path = self.figures_dir / "feature_importance.png"
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                # Grouped by feature type
                plt.figure(figsize=(10, 6))
                feature_type_df = df.groupby('feature_type')['importance'].sum().reset_index().sort_values('importance', ascending=False)
                sns.barplot(x='importance', y='feature_type', data=feature_type_df, palette='viridis')
                plt.title('Feature Importance by Type')
                plt.xlabel('Total Importance')
                plt.ylabel('Feature Type')
                plt.tight_layout()
                
                # Save figure
                fig_path = self.figures_dir / "feature_importance_by_type.png"
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Feature importance visualizations saved to {self.figures_dir}")
            
            except Exception as e:
                logger.error(f"Error generating feature importance visualization: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        logger.info(f"Feature importance report saved to {report_path}")
        logger.info(f"Feature importance summary saved to {summary_path}")
        
        return str(report_path)
    
    def _analyze_test_results(self):
        """
        Analyze detailed test results.
        
        Returns:
            str: Path to report file
        """
        logger.info("Analyzing detailed test results")
        
        # Load test results CSV
        test_results_path = self.output_dir / "test_results_detailed.csv"
        if not test_results_path.exists():
            logger.warning("Detailed test results CSV not found")
            return None
        
        try:
            # Load test results
            test_df = pd.read_csv(test_results_path)
            logger.info(f"Loaded {len(test_df)} test results from {test_results_path}")
            
            # Create error analysis dataframe
            error_df = test_df[test_df['true_label'] != test_df['predicted_label']]
            
            # Split errors into false positives and false negatives
            false_positives = error_df[error_df['predicted_label'] == 1]
            false_negatives = error_df[error_df['predicted_label'] == 0]
            
            # Get high confidence errors
            high_conf_threshold = 0.8
            high_conf_errors = error_df[error_df['confidence'] > high_conf_threshold]
            high_conf_fps = false_positives[false_positives['confidence'] > high_conf_threshold]
            high_conf_fns = false_negatives[false_negatives['confidence'] > high_conf_threshold]
            
            # Calculate average confidence by true/predicted label
            confidence_stats = test_df.groupby(['true_label', 'predicted_label'])['confidence'].agg(['mean', 'min', 'max', 'count']).reset_index()
            
            # Generate analysis report
            report = {
                'title': 'Entity Resolution Test Results Analysis',
                'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_test_instances': len(test_df),
                'correct_count': len(test_df[test_df['correct']]),
                'error_count': len(error_df),
                'false_positive_count': len(false_positives),
                'false_negative_count': len(false_negatives),
                'high_confidence_error_count': len(high_conf_errors),
                'high_confidence_fp_count': len(high_conf_fps),
                'high_confidence_fn_count': len(high_conf_fns),
                'confidence_stats': confidence_stats.to_dict(orient='records')
            }
            
            # Save report
            report_path = self.reports_dir / "test_results_analysis_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            # Create readable summary in Markdown
            summary_path = self.reports_dir / "test_results_analysis_summary.md"
            with open(summary_path, 'w') as f:
                f.write(f"# Entity Resolution Test Results Analysis\n\n")
                f.write(f"Generated: {report['timestamp']}\n\n")
                
                f.write("## Overview\n\n")
                
                # Calculate accuracy
                accuracy = report['correct_count'] / report['total_test_instances'] if report['total_test_instances'] > 0 else 0
                error_rate = report['error_count'] / report['total_test_instances'] if report['total_test_instances'] > 0 else 0
                
                f.write(f"- **Total Test Instances**: {report['total_test_instances']}\n")
                f.write(f"- **Correct Predictions**: {report['correct_count']} ({accuracy:.2%})\n")
                f.write(f"- **Errors**: {report['error_count']} ({error_rate:.2%})\n\n")
                
                f.write("## Error Analysis\n\n")
                f.write(f"- **False Positives**: {report['false_positive_count']} ({report['false_positive_count']/report['error_count']:.2%} of errors)\n")
                f.write(f"- **False Negatives**: {report['false_negative_count']} ({report['false_negative_count']/report['error_count']:.2%} of errors)\n")
                f.write(f"- **High Confidence Errors** (> {high_conf_threshold}): {report['high_confidence_error_count']} ({report['high_confidence_error_count']/report['error_count']:.2%} of errors)\n")
                f.write(f"  - High Confidence False Positives: {report['high_confidence_fp_count']}\n")
                f.write(f"  - High Confidence False Negatives: {report['high_confidence_fn_count']}\n\n")
                
                f.write("## Confidence Statistics\n\n")
                f.write("| True Label | Predicted Label | Count | Mean Confidence | Min Confidence | Max Confidence |\n")
                f.write("|------------|-----------------|-------|-----------------|---------------|---------------|\n")
                
                for stat in report['confidence_stats']:
                    f.write(f"| {int(stat['true_label'])} | {int(stat['predicted_label'])} | {int(stat['count'])} | {stat['mean']:.4f} | {stat['min']:.4f} | {stat['max']:.4f} |\n")
                
                # Feature analysis for errors
                if len(error_df) > 0:
                    # Get feature columns (excluding metadata columns)
                    feature_cols = [col for col in error_df.columns if col not in ['pair_id', 'left_id', 'right_id', 'true_label', 'predicted_label', 'confidence', 'correct']]
                    
                    if len(feature_cols) > 0:
                        # Calculate average feature values for correct and incorrect predictions
                        f.write("\n## Average Feature Values in Errors vs Correct Predictions\n\n")
                        f.write("| Feature | Correct Predictions | False Positives | False Negatives |\n")
                        f.write("|---------|---------------------|----------------|----------------|\n")
                        
                        correct_df = test_df[test_df['correct']]
                        
                        for feature in feature_cols[:10]:  # Limit to top 10 features
                            correct_avg = correct_df[feature].mean()
                            fp_avg = false_positives[feature].mean() if len(false_positives) > 0 else float('nan')
                            fn_avg = false_negatives[feature].mean() if len(false_negatives) > 0 else float('nan')
                            
                            f.write(f"| {feature} | {correct_avg:.4f} | {fp_avg:.4f} | {fn_avg:.4f} |\n")
                
                # Sample errors for inspection
                if len(error_df) > 0:
                    f.write("\n## Sample Error Cases\n\n")
                    
                    # Check if we have entity IDs
                    has_ids = 'left_id' in error_df.columns and 'right_id' in error_df.columns
                    
                    if has_ids:
                        f.write("### False Positives (Predicted Match, Actually Different)\n\n")
                        for i, row in false_positives.head(5).iterrows():
                            f.write(f"- **Pair**: {row['left_id']} - {row['right_id']}\n")
                            f.write(f"  - Confidence: {row['confidence']:.4f}\n")
                            f.write("  - Top Features:\n")
                            
                            # Show top 3 features (if available)
                            top_features = feature_cols[:3] if len(feature_cols) >= 3 else feature_cols
                            for feature in top_features:
                                f.write(f"    - {feature}: {row[feature]:.4f}\n")
                            f.write("\n")
                        
                        f.write("### False Negatives (Predicted Different, Actually Match)\n\n")
                        for i, row in false_negatives.head(5).iterrows():
                            f.write(f"- **Pair**: {row['left_id']} - {row['right_id']}\n")
                            f.write(f"  - Confidence: {row['confidence']:.4f}\n")
                            f.write("  - Top Features:\n")
                            
                            # Show top 3 features (if available)
                            for feature in top_features:
                                f.write(f"    - {feature}: {row[feature]:.4f}\n")
                            f.write("\n")
            
            # Generate visualizations
            if self.visualization_enabled:
                try:
                    # Confusion matrix visualization
                    plt.figure(figsize=(8, 6))
                    cm = confusion_matrix(test_df['true_label'], test_df['predicted_label'])
                    sns.heatmap(
                        cm,
                        annot=True,
                        fmt='d',
                        cmap='Blues',
                        xticklabels=['Non-Match', 'Match'],
                        yticklabels=['Non-Match', 'Match']
                    )
                    plt.title('Confusion Matrix')
                    plt.xlabel('Predicted Label')
                    plt.ylabel('True Label')
                    
                    # Save figure
                    fig_path = self.figures_dir / "test_confusion_matrix.png"
                    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    # Confidence distribution by prediction correctness
                    plt.figure(figsize=(10, 6))
                    sns.histplot(
                        data=test_df,
                        x='confidence',
                        hue='correct',
                        multiple='stack',
                        bins=20,
                        palette=['red', 'green']
                    )
                    plt.title('Confidence Distribution by Prediction Correctness')
                    plt.xlabel('Confidence')
                    plt.ylabel('Count')
                    plt.legend(['Incorrect', 'Correct'])
                    
                    # Save figure
                    fig_path = self.figures_dir / "confidence_by_correctness.png"
                    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    # Confidence distribution by true label
                    plt.figure(figsize=(10, 6))
                    sns.histplot(
                        data=test_df,
                        x='confidence',
                        hue='true_label',
                        multiple='layer',
                        bins=20,
                        palette=['blue', 'orange']
                    )
                    plt.title('Confidence Distribution by True Label')
                    plt.xlabel('Confidence')
                    plt.ylabel('Count')
                    plt.legend(['Non-Match', 'Match'])
                    
                    # Save figure
                    fig_path = self.figures_dir / "confidence_by_true_label.png"
                    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    # Feature boxplots for errors
                    if len(feature_cols) > 0:
                        # Create a label for group analysis
                        test_df['prediction_group'] = 'Correct'
                        test_df.loc[(test_df['true_label'] == 0) & (test_df['predicted_label'] == 1), 'prediction_group'] = 'False Positive'
                        test_df.loc[(test_df['true_label'] == 1) & (test_df['predicted_label'] == 0), 'prediction_group'] = 'False Negative'
                        
                        # Plot top 5 features
                        for feature in feature_cols[:5]:
                            plt.figure(figsize=(10, 6))
                            sns.boxplot(
                                data=test_df,
                                x='prediction_group',
                                y=feature,
                                order=['Correct', 'False Positive', 'False Negative']
                            )
                            plt.title(f'Distribution of {feature} by Prediction Result')
                            plt.xlabel('Prediction Result')
                            plt.ylabel(feature)
                            
                            # Save figure
                            fig_path = self.figures_dir / f"feature_{feature}_by_result.png"
                            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                            plt.close()
                    
                    logger.info(f"Test result visualizations saved to {self.figures_dir}")
                
                except Exception as e:
                    logger.error(f"Error generating test result visualizations: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
            
            # Also save error cases to CSV for further analysis
            error_csv_path = self.output_dir / "error_analysis.csv"
            error_df.to_csv(error_csv_path, index=False)
            logger.info(f"Error analysis CSV saved to {error_csv_path}")
            
            logger.info(f"Test results analysis report saved to {report_path}")
            logger.info(f"Test results analysis summary saved to {summary_path}")
            
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Error analyzing test results: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _generate_error_analysis(self):
        """
        Generate error analysis report based on test results.
        
        Returns:
            str: Path to report file
        """
        logger.info("Generating error analysis report")
        
        # If we already analyzed test results, we don't need a separate error analysis
        # The test result analysis already covers error analysis
        return None
    
    def _generate_cluster_statistics(self):
        """
        Generate cluster statistics report.
        
        Returns:
            str: Path to report file
        """
        logger.info("Generating cluster statistics report")
        
        # Load clusters from CSV (more detailed than the JSON file)
        clusters_csv_path = self.output_dir / "clusters.csv"
        if not clusters_csv_path.exists():
            # Fall back to JSON file
            clusters_path = self.output_dir / "clusters.json"
            if not clusters_path.exists():
                logger.warning("Clusters not found")
                return None
            
            with open(clusters_path, 'r') as f:
                clusters = json.load(f)
            
            # Convert to DataFrame for analysis
            rows = []
            for cluster_id, cluster in enumerate(clusters):
                for entity_id in cluster:
                    rows.append({
                        'cluster_id': cluster_id,
                        'entity_id': entity_id,
                        'cluster_size': len(cluster)
                    })
            
            clusters_df = pd.DataFrame(rows)
        else:
            # Load directly from CSV
            clusters_df = pd.read_csv(clusters_csv_path)
        
        # Compute cluster statistics
        cluster_sizes = clusters_df['cluster_size'].unique()
        cluster_counts = clusters_df.groupby('cluster_id').size().reset_index(name='count')
        
        stats = {
            'title': 'Entity Resolution Cluster Statistics',
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_clusters': clusters_df['cluster_id'].nunique(),
            'total_records': len(clusters_df),
            'average_cluster_size': clusters_df['cluster_size'].mean(),
            'median_cluster_size': clusters_df['cluster_size'].median(),
            'min_cluster_size': clusters_df['cluster_size'].min(),
            'max_cluster_size': clusters_df['cluster_size'].max(),
            'std_cluster_size': clusters_df['cluster_size'].std(),
            'cluster_size_distribution': {
                '1': sum(1 for size in cluster_sizes if size == 1),
                '2': sum(1 for size in cluster_sizes if size == 2),
                '3-5': sum(1 for size in cluster_sizes if 3 <= size <= 5),
                '6-10': sum(1 for size in cluster_sizes if 6 <= size <= 10),
                '11-20': sum(1 for size in cluster_sizes if 11 <= size <= 20),
                '21+': sum(1 for size in cluster_sizes if size > 20)
            }
        }
        
        # Save report
        report_path = self.reports_dir / "cluster_statistics_report.json"
        with open(report_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Create readable summary in Markdown
        summary_path = self.reports_dir / "cluster_statistics_summary.md"
        with open(summary_path, 'w') as f:
            f.write(f"# Entity Resolution Cluster Statistics\n\n")
            f.write(f"Generated: {stats['timestamp']}\n\n")
            
            f.write("## Cluster Overview\n\n")
            f.write(f"- Total Clusters: {stats['total_clusters']}\n")
            f.write(f"- Total Records in Clusters: {stats['total_records']}\n\n")
            
            f.write("## Cluster Size Statistics\n\n")
            f.write(f"- Average Cluster Size: {stats['average_cluster_size']:.2f}\n")
            f.write(f"- Median Cluster Size: {stats['median_cluster_size']:.2f}\n")
            f.write(f"- Minimum Cluster Size: {stats['min_cluster_size']}\n")
            f.write(f"- Maximum Cluster Size: {stats['max_cluster_size']}\n")
            f.write(f"- Standard Deviation: {stats['std_cluster_size']:.2f}\n\n")
            
            f.write("## Cluster Size Distribution\n\n")
            f.write("| Cluster Size | Count | Percentage |\n")
            f.write("|--------------|-------|------------|\n")
            
            for size_range, count in stats['cluster_size_distribution'].items():
                percentage = count / stats['total_clusters'] * 100 if stats['total_clusters'] > 0 else 0
                f.write(f"| {size_range} | {count} | {percentage:.2f}% |\n")
            
            # Top largest clusters
            f.write("\n## Largest Clusters\n\n")
            
            # Group by cluster_id and count entities
            top_clusters = clusters_df.groupby('cluster_id').size().reset_index(name='entity_count')
            top_clusters = top_clusters.sort_values('entity_count', ascending=False).head(10)
            
            f.write("| Cluster ID | Entity Count |\n")
            f.write("|------------|-------------|\n")
            
            for _, row in top_clusters.iterrows():
                f.write(f"| {row['cluster_id']} | {row['entity_count']} |\n")
        
        # Visualize cluster statistics
        if self.visualization_enabled:
            try:
                # Cluster size distribution
                plt.figure(figsize=(10, 6))
                
                # Prepare data for visualization
                size_ranges = list(stats['cluster_size_distribution'].keys())
                counts = list(stats['cluster_size_distribution'].values())
                
                # Bar plot
                plt.bar(size_ranges, counts, color='skyblue')
                plt.title('Cluster Size Distribution')
                plt.xlabel('Cluster Size')
                plt.ylabel('Count')
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                
                # Save figure
                fig_path = self.figures_dir / "cluster_size_distribution.png"
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                # Histogram of cluster sizes
                plt.figure(figsize=(10, 6))
                plt.hist(clusters_df['cluster_size'], bins=20, color='skyblue', edgecolor='black')
                plt.title('Histogram of Cluster Sizes')
                plt.xlabel('Cluster Size')
                plt.ylabel('Frequency')
                plt.grid(linestyle='--', alpha=0.7)
                
                # Save figure
                fig_path = self.figures_dir / "cluster_size_histogram.png"
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                # Top 10 largest clusters
                plt.figure(figsize=(12, 6))
                top_clusters = clusters_df.groupby('cluster_id').size().reset_index(name='entity_count')
                top_clusters = top_clusters.sort_values('entity_count', ascending=False).head(10)
                
                plt.bar(top_clusters['cluster_id'].astype(str), top_clusters['entity_count'], color='skyblue')
                plt.title('Top 10 Largest Clusters')
                plt.xlabel('Cluster ID')
                plt.ylabel('Entity Count')
                plt.xticks(rotation=45)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                
                # Save figure
                fig_path = self.figures_dir / "top_10_largest_clusters.png"
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Cluster statistics visualizations saved to {self.figures_dir}")
            
            except Exception as e:
                logger.error(f"Error generating cluster statistics visualization: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        logger.info(f"Cluster statistics report saved to {report_path}")
        logger.info(f"Cluster statistics summary saved to {summary_path}")
        
        return str(report_path)
    
    def _generate_visualizations(self):
        """
        Generate visualizations for entity resolution results.
        
        Returns:
            list: Paths to figure files
        """
        logger.info("Generating overall visualizations")
        
        figure_paths = []
        
        # Generate ROC curve if possible
        roc_curve_path = self._visualize_roc_curve()
        if roc_curve_path:
            figure_paths.append(roc_curve_path)
        
        # Generate precision-recall curve if possible
        pr_curve_path = self._visualize_precision_recall_curve()
        if pr_curve_path:
            figure_paths.append(pr_curve_path)
        
        # Generate pipeline summary visualization if possible
        pipeline_summary_path = self._visualize_pipeline_summary()
        if pipeline_summary_path:
            figure_paths.append(pipeline_summary_path)
        
        return figure_paths
    
    def _visualize_roc_curve(self):
        """
        Visualize ROC curve if test results are available.
        
        Returns:
            str: Path to figure file
        """
        test_results_path = self.output_dir / "test_results_detailed.csv"
        if not test_results_path.exists():
            return None
        
        try:
            # Load test results
            test_df = pd.read_csv(test_results_path)
            
            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(test_df['true_label'], test_df['confidence'])
            roc_auc = auc(fpr, tpr)
            
            # Plot ROC curve
            plt.figure(figsize=(10, 8))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.grid(linestyle='--', alpha=0.7)
            
            # Add decision threshold point
            decision_idx = np.argmin(np.abs(thresholds - self.config['classification']['decision_threshold']))
            plt.scatter(fpr[decision_idx], tpr[decision_idx], color='red', s=100, label=f'Decision Threshold ({self.config["classification"]["decision_threshold"]:.2f})')
            plt.legend(loc="lower right")
            
            # Save figure
            fig_path = self.figures_dir / "roc_curve.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"ROC curve visualization saved to {fig_path}")
            
            return str(fig_path)
        
        except Exception as e:
            logger.error(f"Error generating ROC curve: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _visualize_precision_recall_curve(self):
        """
        Visualize precision-recall curve if test results are available.
        
        Returns:
            str: Path to figure file
        """
        test_results_path = self.output_dir / "test_results_detailed.csv"
        if not test_results_path.exists():
            return None
        
        try:
            # Load test results
            test_df = pd.read_csv(test_results_path)
            
            # Calculate precision-recall curve
            precision, recall, thresholds = precision_recall_curve(test_df['true_label'], test_df['confidence'])
            
            # Calculate F1 scores for each threshold
            f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
            best_threshold_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_threshold_idx]
            best_f1 = f1_scores[best_threshold_idx]
            
            # Plot precision-recall curve
            plt.figure(figsize=(10, 8))
            plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
            plt.axhline(y=0.5, color='navy', linestyle='--', label='Baseline (50% precision)')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc="lower left")
            plt.grid(linestyle='--', alpha=0.7)
            
            # Add decision threshold point
            decision_threshold = self.config['classification']['decision_threshold']
            # Find closest threshold
            if len(thresholds) > 0:
                decision_idx = min(range(len(thresholds)), key=lambda i: abs(thresholds[i] - decision_threshold))
                plt.scatter(recall[decision_idx], precision[decision_idx], color='red', s=100, 
                            label=f'Decision Threshold ({decision_threshold:.2f})')
            
            # Add best F1 threshold point
            plt.scatter(recall[best_threshold_idx], precision[best_threshold_idx], color='green', s=100, 
                        label=f'Best F1 Threshold ({best_threshold:.2f}, F1={best_f1:.2f})')
            
            plt.legend(loc="lower left")
            
            # Save figure
            fig_path = self.figures_dir / "precision_recall_curve.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Precision-recall curve visualization saved to {fig_path}")
            
            # Also create a threshold analysis plot
            plt.figure(figsize=(12, 8))
            
            # Only include valid indices where precision and recall are defined
            valid_indices = ~np.isnan(f1_scores)
            valid_thresholds = thresholds[valid_indices]
            valid_precision = precision[:-1][valid_indices]
            valid_recall = recall[:-1][valid_indices]
            valid_f1_scores = f1_scores[valid_indices]
            
            if len(valid_thresholds) > 0:
                # Plot precision, recall, and F1 score vs threshold
                plt.plot(valid_thresholds, valid_precision, 'b-', label='Precision')
                plt.plot(valid_thresholds, valid_recall, 'g-', label='Recall')
                plt.plot(valid_thresholds, valid_f1_scores, 'r-', label='F1 Score')
                
                # Add decision threshold line
                plt.axvline(x=decision_threshold, color='black', linestyle='--', 
                            label=f'Decision Threshold ({decision_threshold:.2f})')
                
                # Add best F1 threshold line
                plt.axvline(x=best_threshold, color='purple', linestyle='--', 
                            label=f'Best F1 Threshold ({best_threshold:.2f})')
                
                plt.xlabel('Threshold')
                plt.ylabel('Score')
                plt.title('Precision, Recall, and F1 Score vs Threshold')
                plt.legend(loc="best")
                plt.grid(linestyle='--', alpha=0.7)
                
                # Save figure
                threshold_fig_path = self.figures_dir / "threshold_analysis.png"
                plt.savefig(threshold_fig_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Threshold analysis visualization saved to {threshold_fig_path}")
            
            return str(fig_path)
        
        except Exception as e:
            logger.error(f"Error generating precision-recall curve: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _visualize_pipeline_summary(self):
        """
        Generate a visual summary of the entity resolution pipeline results.
        
        Returns:
            str: Path to figure file
        """
        try:
            # Load metrics from various pipeline stages
            metrics_path = self.output_dir / "classification_metrics.json"
            if not metrics_path.exists():
                return None
            
            with open(metrics_path, 'r') as f:
                classification_metrics = json.load(f)
            
            # Load pipeline results if available
            pipeline_results_path = self.output_dir / "pipeline_results.json"
            if pipeline_results_path.exists():
                with open(pipeline_results_path, 'r') as f:
                    pipeline_results = json.load(f)
            else:
                pipeline_results = {}
            
            # Create summary visualization
            plt.figure(figsize=(15, 10))
            
            # Create grid for subplots
            gs = plt.GridSpec(2, 2)
            
            # Performance metrics
            ax1 = plt.subplot(gs[0, 0])
            metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            values = [classification_metrics.get(m, 0) for m in metrics]
            
            ax1.bar(metrics, values, color='skyblue')
            ax1.set_ylim(0, 1.1)
            ax1.set_title('Classification Performance')
            ax1.set_ylabel('Score')
            ax1.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Confusion matrix
            ax2 = plt.subplot(gs[0, 1])
            cm = classification_metrics.get('confusion_matrix', {})
            if cm:
                cm_array = np.array([
                    [cm.get('true_negatives', 0), cm.get('false_positives', 0)],
                    [cm.get('false_negatives', 0), cm.get('true_positives', 0)]
                ])
                
                sns.heatmap(
                    cm_array,
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    xticklabels=['Non-Match', 'Match'],
                    yticklabels=['Non-Match', 'Match'],
                    ax=ax2
                )
                ax2.set_title('Confusion Matrix')
                ax2.set_xlabel('Predicted')
                ax2.set_ylabel('Actual')
            
            # Pipeline statistics
            ax3 = plt.subplot(gs[1, 0])
            stages = []
            durations = []
            
            if pipeline_results:
                for stage, results in pipeline_results.items():
                    if isinstance(results, dict) and 'duration' in results:
                        stages.append(stage)
                        durations.append(results['duration'])
            
            if stages:
                ax3.barh(stages, durations, color='lightgreen')
                ax3.set_title('Processing Time by Stage')
                ax3.set_xlabel('Duration (seconds)')
                ax3.grid(axis='x', linestyle='--', alpha=0.7)
            
            # Add feature importance
            ax4 = plt.subplot(gs[1, 1])
            if 'feature_importance' in classification_metrics:
                feature_importance = classification_metrics['feature_importance']
                top_features = list(feature_importance.keys())[:10]
                importance_values = [feature_importance[f]['importance'] for f in top_features]
                
                y_pos = np.arange(len(top_features))
                ax4.barh(y_pos, importance_values, color='salmon')
                ax4.set_yticks(y_pos)
                ax4.set_yticklabels(top_features)
                ax4.invert_yaxis()  # Labels read top-to-bottom
                ax4.set_title('Top 10 Feature Importance')
                ax4.set_xlabel('Importance')
            
            plt.tight_layout()
            
            # Add title
            plt.suptitle('Entity Resolution Pipeline Summary', fontsize=16, y=1.02)
            
            # Save figure
            fig_path = self.figures_dir / "pipeline_summary.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Pipeline summary visualization saved to {fig_path}")
            
            return str(fig_path)
        
        except Exception as e:
            logger.error(f"Error generating pipeline summary visualization: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
