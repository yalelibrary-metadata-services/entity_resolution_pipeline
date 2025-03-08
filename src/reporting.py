"""
Reporting module for entity resolution pipeline.

This module generates reports and visualizations for entity resolution results
with enhanced feature distribution visualizations and proper normalization.
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
from matplotlib.colors import LinearSegmentedColormap

logger = logging.getLogger(__name__)

class ReportGenerator:
    """
    Generates reports and visualizations for entity resolution results.
    
    Features:
    - Classification metrics reporting
    - Feature importance visualization
    - Feature distribution visualization
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
        
        # Create output directories if they don't exist
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Reports will be saved to {self.reports_dir}")
        logger.info(f"Figures will be saved to {self.figures_dir}")
        
        # Normalize features in test results
        test_results_path = self.output_dir / "test_results_detailed.csv"
        if test_results_path.exists():
            normalized_path, normalized_df = self._normalize_features_for_reporting(test_results_path)
            logger.info(f"Normalized test results saved to {normalized_path}")
            
            # Create enhanced feature distribution plots
            logger.info("Generating enhanced feature distribution visualizations...")
            top_features = self._create_feature_distribution_plots(normalized_df, top_n=15)
            logger.info(f"Generated distribution plots for top {len(top_features)} features")
            
            # Track generated figures
            for feature in top_features:
                generated_figures.append(str(self.figures_dir / f'feature_dist_{feature}.png'))
            generated_figures.append(str(self.figures_dir / 'feature_separation_power.png'))
        
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
        
        # Check that reports were actually created
        report_files = list(self.reports_dir.glob("*.json")) + list(self.reports_dir.glob("*.md"))
        figure_files = list(self.figures_dir.glob("*.png"))
        
        logger.info(f"Generated {len(report_files)} report files and {len(figure_files)} figures")
        for file in report_files[:5]:  # Log the first few files
            logger.info(f"  - {file.name} ({file.stat().st_size} bytes)")
        
        results = {
            'reports': generated_reports,
            'figures': generated_figures
        }
        
        logger.info(
            f"Report generation completed: {len(generated_reports)} reports, "
            f"{len(generated_figures)} figures"
        )
        
        return results
    
    def _normalize_features_for_reporting(self, csv_path):
        """
        Normalize all features in test results to 0-1 similarity range.
        
        Args:
            csv_path (Path): Path to test results CSV
            
        Returns:
            tuple: (normalized_path, normalized_df)
        """
        # Load the test results CSV
        df = pd.read_csv(csv_path)
        
        # Identify feature columns (exclude metadata columns)
        metadata_cols = ['pair_id', 'left_id', 'right_id', 'true_label', 
                         'predicted_label', 'confidence', 'correct']
        feature_cols = [col for col in df.columns if col not in metadata_cols]
        
        # Process each feature column
        for col in feature_cols:
            # Check if column contains negative values
            if df[col].min() < 0:
                if df[col].min() >= -1 and df[col].max() <= 1:
                    # If values are in [-1, 1] range (like correlations), rescale to [0, 1]
                    logger.info(f"Normalizing {col} from [-1, 1] to [0, 1] range")
                    df[col] = (df[col] + 1) / 2
                else:
                    # For other ranges, use min-max scaling
                    logger.info(f"Applying min-max scaling to {col}")
                    min_val = df[col].min()
                    max_val = df[col].max()
                    df[col] = (df[col] - min_val) / (max_val - min_val)
        
        # Save normalized version
        normalized_path = str(csv_path).replace('.csv', '_normalized.csv')
        df.to_csv(normalized_path, index=False)
        
        return normalized_path, df
    
    def _create_feature_distribution_plots(self, df, top_n=10):
        """
        Create feature distribution plots showing separation between classes.
        
        Args:
            df (DataFrame): Normalized test results dataframe
            top_n (int): Number of top features to visualize
            
        Returns:
            list: Top features
        """
        # Get feature columns
        metadata_cols = ['pair_id', 'left_id', 'right_id', 'true_label', 
                         'predicted_label', 'confidence', 'correct']
        feature_cols = [col for col in df.columns if col not in metadata_cols]
        
        # Calculate feature importance based on class separation
        feature_separation = {}
        for col in feature_cols:
            pos_mean = df[df['true_label'] == 1][col].mean()
            neg_mean = df[df['true_label'] == 0][col].mean()
            separation = abs(pos_mean - neg_mean)
            feature_separation[col] = {
                'separation': separation,
                'pos_mean': pos_mean,
                'neg_mean': neg_mean
            }
        
        # Sort features by separation
        sorted_features = sorted(feature_separation.items(), 
                                key=lambda x: x[1]['separation'], 
                                reverse=True)
        
        # Select top N features
        top_features = [f[0] for f in sorted_features[:top_n]]
        
        # Create custom colormaps
        pos_cmap = LinearSegmentedColormap.from_list('positive', ['#c8e6c9', '#2e7d32'])
        neg_cmap = LinearSegmentedColormap.from_list('negative', ['#ffcdd2', '#c62828'])
        
        # Create plots for each feature
        for feature in top_features:
            plt.figure(figsize=(12, 8))
            
            # Get data and statistics
            pos_data = df[df['true_label'] == 1][feature]
            neg_data = df[df['true_label'] == 0][feature]
            pos_mean = feature_separation[feature]['pos_mean']
            neg_mean = feature_separation[feature]['neg_mean']
            separation = feature_separation[feature]['separation']
            
            # Plot histograms with KDE
            sns.histplot(pos_data, kde=True, stat='density', alpha=0.6, 
                        color='#4caf50', label='Matching Entities', 
                        edgecolor='white', linewidth=0.5)
            sns.histplot(neg_data, kde=True, stat='density', alpha=0.6, 
                        color='#f44336', label='Non-matching Entities', 
                        edgecolor='white', linewidth=0.5)
            
            # Add mean lines
            plt.axvline(pos_mean, color='#2e7d32', linestyle='--', 
                       label=f'Match Mean: {pos_mean:.4f}')
            plt.axvline(neg_mean, color='#c62828', linestyle='--', 
                       label=f'Non-match Mean: {neg_mean:.4f}')
            
            # Styling
            plt.title(f'Distribution of {feature} by Class (Separation: {separation:.4f})', 
                     fontsize=16)
            plt.xlabel('Feature Value', fontsize=14)
            plt.ylabel('Density', fontsize=14)
            plt.grid(alpha=0.3)
            plt.legend(fontsize=12)
            plt.tight_layout()
            
            # Save figure
            plt.savefig(self.figures_dir / f'feature_dist_{feature}.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Create a summary plot of feature separation
        plt.figure(figsize=(14, 8))
        separations = [feature_separation[f]['separation'] for f in top_features]
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(top_features)))
        
        # Create barplot
        bars = plt.barh(top_features, separations, color=colors)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.4f}', ha='left', va='center', fontsize=10)
        
        plt.title('Feature Separation Power (Higher = Better Class Separation)', fontsize=16)
        plt.xlabel('Mean Absolute Difference Between Classes', fontsize=14)
        plt.ylabel('Feature', fontsize=14)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(self.figures_dir / 'feature_separation_power.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return top_features
    
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
                sns.barplot(x='importance', y='feature', data=df.head(15), hue='feature', legend=False, palette='viridis')
                plt.title('Top 15 Features by Importance')
                plt.xlabel('Importance')
                plt.ylabel('Feature')
                plt.tight_layout()
                
                # Bar plot of feature weights
                plt.subplot(2, 1, 2)
                sns.barplot(x='weight', y='feature', data=df.head(15), hue='feature', legend=False, palette='viridis')
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
                sns.barplot(x='importance', y='feature_type', data=feature_type_df, hue='feature_type', legend=False, palette='viridis')
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
                    # Normalize features before visualizing
                    normalized_path, normalized_df = self._normalize_features_for_reporting(test_results_path)
                    test_df = normalized_df  # Use normalized data for visualizations
                    
                    # Confusion matrix visualization
                    plt.figure(figsize=(10, 8))
                    cm = confusion_matrix(test_df['true_label'], test_df['predicted_label'])
                    sns.heatmap(
                        cm,
                        annot=True,
                        fmt='d',
                        cmap='Blues',
                        xticklabels=['Non-Match', 'Match'],
                        yticklabels=['Non-Match', 'Match'],
                        annot_kws={'size': 14},
                        cbar_kws={'label': 'Count'}
                    )
                    plt.title('Confusion Matrix', fontsize=16)
                    plt.xlabel('Predicted Label', fontsize=14)
                    plt.ylabel('True Label', fontsize=14)
                    
                    # Save figure
                    fig_path = self.figures_dir / "test_confusion_matrix.png"
                    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    # Confidence distribution by prediction correctness
                    plt.figure(figsize=(12, 8))
                    sns.histplot(
                        data=test_df,
                        x='confidence',
                        hue='correct',
                        multiple='stack',
                        bins=20,
                        palette=['#EF5350', '#66BB6A']
                    )
                    plt.title('Confidence Distribution by Prediction Correctness', fontsize=16)
                    plt.xlabel('Confidence', fontsize=14)
                    plt.ylabel('Count', fontsize=14)
                    plt.legend(['Incorrect', 'Correct'], fontsize=12)
                    plt.grid(alpha=0.3)
                    
                    # Save figure
                    fig_path = self.figures_dir / "confidence_by_correctness.png"
                    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    # Confidence distribution by true label
                    plt.figure(figsize=(12, 8))
                    sns.histplot(
                        data=test_df,
                        x='confidence',
                        hue='true_label',
                        multiple='layer',
                        bins=20,
                        palette=['#42A5F5', '#FFA726'],
                        alpha=0.7
                    )
                    plt.title('Confidence Distribution by True Label', fontsize=16)
                    plt.xlabel('Confidence', fontsize=14)
                    plt.ylabel('Count', fontsize=14)
                    plt.legend(['Non-Match', 'Match'], fontsize=12)
                    plt.grid(alpha=0.3)
                    
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
                            plt.figure(figsize=(12, 8))
                            sns.boxplot(
                                data=test_df,
                                x='prediction_group',
                                y=feature,
                                order=['Correct', 'False Positive', 'False Negative'],
                                palette={'Correct': '#66BB6A', 'False Positive': '#FF7043', 'False Negative': '#42A5F5'},
                                orientation='vertical'
                            )
                            plt.title(f'Distribution of {feature} by Prediction Result', fontsize=16)
                            plt.xlabel('Prediction Result', fontsize=14)
                            plt.ylabel(feature, fontsize=14)
                            plt.grid(alpha=0.3)
                            
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
                plt.figure(figsize=(12, 8))
                
                # Prepare data for visualization
                size_ranges = list(stats['cluster_size_distribution'].keys())
                counts = list(stats['cluster_size_distribution'].values())
                
                # Bar plot with modern colors
                plt.bar(size_ranges, counts, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(size_ranges))))
                plt.title('Cluster Size Distribution', fontsize=16)
                plt.xlabel('Cluster Size', fontsize=14)
                plt.ylabel('Count', fontsize=14)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                
                # Add value labels
                for i, count in enumerate(counts):
                    plt.text(i, count + 0.5, str(count), ha='center', fontsize=12)
                
                # Save figure
                fig_path = self.figures_dir / "cluster_size_distribution.png"
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                # Histogram of cluster sizes (more detailed view)
                plt.figure(figsize=(12, 8))
                actual_sizes = clusters_df['cluster_size'].value_counts().reset_index()
                actual_sizes.columns = ['size', 'count']
                actual_sizes = actual_sizes.sort_values('size')
                
                # Use log scale if there's high variance in cluster sizes
                if stats['max_cluster_size'] / stats['min_cluster_size'] > 20:
                    plt.bar(actual_sizes['size'], actual_sizes['count'], 
                           color=plt.cm.viridis(np.linspace(0.2, 0.8, len(actual_sizes))))
                    plt.xscale('log')
                    plt.title('Histogram of Cluster Sizes (Log Scale)', fontsize=16)
                else:
                    plt.bar(actual_sizes['size'], actual_sizes['count'],
                           color=plt.cm.viridis(np.linspace(0.2, 0.8, len(actual_sizes))))
                    plt.title('Histogram of Cluster Sizes', fontsize=16)
                
                plt.xlabel('Cluster Size', fontsize=14)
                plt.ylabel('Frequency', fontsize=14)
                plt.grid(linestyle='--', alpha=0.7)
                
                # Save figure
                fig_path = self.figures_dir / "cluster_size_histogram.png"
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                # Top 10 largest clusters visualization
                plt.figure(figsize=(14, 8))
                top_clusters = clusters_df.groupby('cluster_id').size().reset_index(name='entity_count')
                top_clusters = top_clusters.sort_values('entity_count', ascending=False).head(10)
                
                plt.bar(top_clusters['cluster_id'].astype(str), top_clusters['entity_count'], 
                       color=plt.cm.plasma(np.linspace(0.2, 0.8, len(top_clusters))))
                plt.title('Top 10 Largest Clusters', fontsize=16)
                plt.xlabel('Cluster ID', fontsize=14)
                plt.ylabel('Entity Count', fontsize=14)
                plt.xticks(rotation=45)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                
                # Add value labels
                for i, count in enumerate(top_clusters['entity_count']):
                    plt.text(i, count + 0.5, str(count), ha='center', fontsize=12)
                
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
            
            # Plot ROC curve with enhanced styling
            plt.figure(figsize=(12, 10))
            plt.plot(fpr, tpr, color='#1976D2', lw=3, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='#455A64', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=14)
            plt.ylabel('True Positive Rate', fontsize=14)
            plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
            plt.legend(loc="lower right", fontsize=12)
            plt.grid(linestyle='--', alpha=0.7)
            
            # Add decision threshold point
            decision_idx = np.argmin(np.abs(thresholds - self.config['classification']['decision_threshold']))
            plt.scatter(fpr[decision_idx], tpr[decision_idx], color='red', s=100, 
                       label=f'Decision Threshold ({self.config["classification"]["decision_threshold"]:.2f})')
            plt.legend(loc="lower right", fontsize=12)
            
            # Add labels at key points
            plt.annotate('Perfect Classification', xy=(0, 1), xytext=(0.2, 0.9),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))
            
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
            
            # Calculate baseline performance
            baseline = test_df['true_label'].mean()
            
            # Plot precision-recall curve with enhanced styling
            plt.figure(figsize=(12, 10))
            plt.plot(recall, precision, color='#7B1FA2', lw=3, label='Precision-Recall curve')
            plt.axhline(y=baseline, color='#FF5722', linestyle='--', 
                       label=f'Baseline precision: {baseline:.2f}')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall', fontsize=14)
            plt.ylabel('Precision', fontsize=14)
            plt.title('Precision-Recall Curve', fontsize=16)
            plt.legend(loc="lower left", fontsize=12)
            plt.grid(linestyle='--', alpha=0.7)
            
            # Add decision threshold point
            decision_threshold = self.config['classification']['decision_threshold']
            # Find closest threshold
            if len(thresholds) > 0:
                decision_idx = min(range(len(thresholds)), key=lambda i: abs(thresholds[i] - decision_threshold))
                plt.scatter(recall[decision_idx], precision[decision_idx], color='#E91E63', s=100, 
                           label=f'Decision Threshold ({decision_threshold:.2f})')
            
            # Add best F1 threshold point
            plt.scatter(recall[best_threshold_idx], precision[best_threshold_idx], color='#4CAF50', s=100, 
                       label=f'Best F1 Threshold ({best_threshold:.2f}, F1={best_f1:.2f})')
            
            plt.legend(loc="lower left", fontsize=12)
            
            # Save figure
            fig_path = self.figures_dir / "precision_recall_curve.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Precision-recall curve visualization saved to {fig_path}")
            
            # Also create a threshold analysis plot
            plt.figure(figsize=(14, 10))
            
            # Only include valid indices where precision and recall are defined
            valid_indices = ~np.isnan(f1_scores)
            valid_thresholds = thresholds[valid_indices]
            valid_precision = precision[:-1][valid_indices]
            valid_recall = recall[:-1][valid_indices]
            valid_f1_scores = f1_scores[valid_indices]
            
            if len(valid_thresholds) > 0:
                # Plot precision, recall, and F1 score vs threshold
                plt.plot(valid_thresholds, valid_precision, 'b-', lw=3, label='Precision')
                plt.plot(valid_thresholds, valid_recall, 'g-', lw=3, label='Recall')
                plt.plot(valid_thresholds, valid_f1_scores, 'r-', lw=3, label='F1 Score')
                
                # Add decision threshold line
                plt.axvline(x=decision_threshold, color='#E91E63', linestyle='--', lw=2, 
                           label=f'Decision Threshold ({decision_threshold:.2f})')
                
                # Add best F1 threshold line
                plt.axvline(x=best_threshold, color='#4CAF50', linestyle='--', lw=2, 
                           label=f'Best F1 Threshold ({best_threshold:.2f})')
                
                plt.xlabel('Threshold', fontsize=14)
                plt.ylabel('Score', fontsize=14)
                plt.title('Precision, Recall, and F1 Score vs Threshold', fontsize=16)
                plt.legend(loc="best", fontsize=12)
                plt.grid(linestyle='--', alpha=0.7)
                
                # Save figure
                threshold_fig_path = self.figures_dir / "threshold_analysis.png"
                plt.savefig(threshold_fig_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Threshold analysis visualization saved to {threshold_fig_path}")
                figure_paths = [str(fig_path), str(threshold_fig_path)]
                return figure_paths
            
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
            
            # Create summary visualization with enhanced styling
            plt.figure(figsize=(16, 12))
            
            # Create grid for subplots with more spacing
            from matplotlib.gridspec import GridSpec
            gs = GridSpec(2, 2, wspace=0.3, hspace=0.4)
            
            # Performance metrics
            ax1 = plt.subplot(gs[0, 0])
            metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            values = [classification_metrics.get(m, 0) for m in metrics]
            
            bars = ax1.bar(metrics, values, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(metrics))))
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=12)
            
            ax1.set_ylim(0, 1.1)
            ax1.set_title('Classification Performance', fontsize=16)
            ax1.set_ylabel('Score', fontsize=14)
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
                    annot_kws={'size': 14},
                    cbar_kws={'label': 'Count'},
                    ax=ax2
                )
                ax2.set_title('Confusion Matrix', fontsize=16)
                ax2.set_xlabel('Predicted', fontsize=14)
                ax2.set_ylabel('Actual', fontsize=14)
            
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
                bars = ax3.barh(stages, durations, color=plt.cm.plasma(np.linspace(0.2, 0.8, len(stages))))
                
                # Add value labels
                for bar in bars:
                    width = bar.get_width()
                    ax3.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                            f'{width:.1f}s', ha='left', va='center', fontsize=12)
                
                ax3.set_title('Processing Time by Stage', fontsize=16)
                ax3.set_xlabel('Duration (seconds)', fontsize=14)
                ax3.grid(axis='x', linestyle='--', alpha=0.7)
            
            # Add feature importance
            ax4 = plt.subplot(gs[1, 1])
            if 'feature_importance' in classification_metrics:
                feature_importance = classification_metrics['feature_importance']
                top_features = list(feature_importance.keys())[:10]
                importance_values = [feature_importance[f]['importance'] for f in top_features]
                
                y_pos = np.arange(len(top_features))
                bars = ax4.barh(y_pos, importance_values, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(top_features))))
                
                # Add value labels
                for bar in bars:
                    width = bar.get_width()
                    ax4.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                            f'{width:.3f}', ha='left', va='center', fontsize=12)
                
                ax4.set_yticks(y_pos)
                ax4.set_yticklabels(top_features)
                ax4.invert_yaxis()  # Labels read top-to-bottom
                ax4.set_title('Top 10 Feature Importance', fontsize=16)
                ax4.set_xlabel('Importance', fontsize=14)
                ax4.grid(axis='x', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            
            # Add title
            plt.suptitle('Entity Resolution Pipeline Summary', fontsize=20, y=1.02)
            
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