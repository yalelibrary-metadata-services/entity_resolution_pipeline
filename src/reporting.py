"""
Reporting module for entity resolution pipeline.

This module generates reports and visualizations for entity resolution results
with enhanced feature representation and visualization.
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
    - Feature distribution visualization with enhanced color schemes
    - Multiple feature representations (raw, normalized, standardized)
    - Test results analysis and visualization
    - Cluster statistics reporting
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
        
        # Color scheme for visualizations
        self.colors = {
            'match': '#a6cee3',  # Light blue for matching entities
            'non_match': '#fdb462',  # Light orange for non-matching entities
            'match_line': '#1f78b4',  # Darker blue for match mean line
            'non_match_line': '#e08214',  # Darker orange for non-match mean line
        }
        
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
        
        # Process test results with multiple feature representations
        test_results_path = self.output_dir / "test_results_detailed.csv"
        if test_results_path.exists():
            enhanced_path, test_df = self._process_test_results(test_results_path)
            logger.info(f"Enhanced test results saved to {enhanced_path}")
            
            # Create feature distribution plots
            top_features = self._create_feature_distribution_plots(test_df, top_n=15)
            logger.info(f"Generated distribution plots for top {len(top_features)} features")
            
            # Track generated figures
            for feature in top_features:
                generated_figures.append(str(self.figures_dir / f'feature_dist_{feature}.png'))
                # Check if comparison plot exists
                comparison_path = self.figures_dir / f'feature_comparison_{feature}.png'
                if comparison_path.exists():
                    generated_figures.append(str(comparison_path))
                    
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
        
        rfe_report = self._generate_rfe_report()
        if rfe_report:
            generated_reports.append(rfe_report)

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
    
    def _process_test_results(self, csv_path):
        """
        Process test results and add derived feature representations if needed.
        
        Args:
            csv_path (Path): Path to test results CSV
            
        Returns:
            tuple: (enhanced_path, enhanced_df)
        """
        try:
            # Load the test results CSV
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
            
            # Identify feature columns (exclude metadata columns)
            metadata_cols = ['pair_id', 'left_id', 'right_id', 'true_label', 
                            'predicted_label', 'confidence', 'correct']
            feature_cols = [col for col in df.columns if col not in metadata_cols]
            
            # Log feature statistics without modifying them
            logger.info("Feature statistics (preserving original values):")
            cosine_cols = [col for col in feature_cols if 'cosine' in col and 
                          not col.endswith(('_raw', '_norm', '_std'))]
            for col in cosine_cols:
                min_val = df[col].min()
                max_val = df[col].max()
                mean_val = df[col].mean()
                std_val = df[col].std()
                logger.info(f"  {col}: range=[{min_val:.4f}, {max_val:.4f}], mean={mean_val:.4f}, std={std_val:.4f}")
            
            # Check if raw/norm/std columns already exist
            raw_cols = [col for col in feature_cols if col.endswith('_raw')]
            norm_cols = [col for col in feature_cols if col.endswith('_norm')]
            std_cols = [col for col in feature_cols if col.endswith('_std')]
            
            if not raw_cols and not norm_cols:
                logger.info("Adding derived feature representations for enhanced analysis")
                
                # For each cosine feature, add raw and normalized versions if not present
                for col in cosine_cols:
                    # Check if we need to derive representations
                    if f"{col}_raw" not in df.columns and f"{col}_norm" not in df.columns:
                        # Get current values (likely StandardScaler values)
                        std_values = df[col].values
                        
                        # Create raw and normalized representations
                        if df[col].min() < 0 or df[col].max() > 1:
                            logger.info(f"  Creating representations for {col} (appears to be StandardScaler values)")
                            
                            # Derive raw values (approximately back to [-1, 1] range)
                            # Note: This is a rough approximation without the original scaler parameters
                            df[f"{col}_raw"] = np.clip(std_values, -3, 3)  # Limit extreme values
                            
                            # Create normalized values (domain normalization)
                            df[f"{col}_norm"] = (df[f"{col}_raw"] + 1) / 2
                            df[f"{col}_norm"] = df[f"{col}_norm"].clip(0, 1)  # Ensure [0,1] range
                            
                            # Store StandardScaler values explicitly
                            df[f"{col}_std"] = std_values
                        else:
                            # Values already in [0,1] range - might be already normalized
                            logger.info(f"  {col} appears to be in [0,1] range - might be domain-normalized already")
                            df[f"{col}_norm"] = df[col].clip(0, 1)  # Ensure proper range
                            df[f"{col}_raw"] = df[col] * 2 - 1  # Approximate raw values
                            df[f"{col}_std"] = df[col]  # Copy for consistency
            else:
                logger.info("Multiple feature representations already present in data")
            
            # Save enhanced version
            enhanced_path = str(csv_path).replace('.csv', '_enhanced.csv')
            df.to_csv(enhanced_path, index=False)
            logger.info(f"Enhanced test results saved to {enhanced_path}")
            
            return enhanced_path, df
        
        except Exception as e:
            logger.error(f"Error processing test results: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Return original path and dataframe as fallback
            return csv_path, pd.read_csv(csv_path)
    
    def _create_feature_distribution_plots(self, df, top_n=10):
        """
        Create feature distribution plots showing separation between classes.
        
        Args:
            df (DataFrame): Test results dataframe with enhanced features
            top_n (int): Number of top features to visualize
            
        Returns:
            list: Top features
        """
        # Get feature columns (exclude metadata and derived columns)
        metadata_cols = ['pair_id', 'left_id', 'right_id', 'true_label', 
                        'predicted_label', 'confidence', 'correct']
        derived_suffixes = ['_raw', '_norm', '_std']
        
        # Get base feature columns (excluding derived versions)
        feature_cols = [col for col in df.columns 
                        if col not in metadata_cols and 
                        not any(col.endswith(suffix) for suffix in derived_suffixes)]
        
        logger.info(f"Creating distribution plots for {len(feature_cols)} feature columns")
        
        # Calculate feature importance based on class separation
        feature_separation = {}
        for col in feature_cols:
            pos_data = df[df['true_label'] == 1][col]
            neg_data = df[df['true_label'] == 0][col]
            
            # Skip features with no variance
            if pos_data.nunique() <= 1 and neg_data.nunique() <= 1:
                logger.warning(f"Skipping feature {col} - no variance in values")
                continue
                
            pos_mean = pos_data.mean()
            neg_mean = neg_data.mean()
            separation = abs(pos_mean - neg_mean)
            
            feature_separation[col] = {
                'separation': separation,
                'pos_mean': pos_mean,
                'neg_mean': neg_mean
            }
        
        # If no features have separation value, return empty list
        if not feature_separation:
            logger.warning("No features with sufficient class separation found")
            return []
        
        # Sort features by separation
        sorted_features = sorted(feature_separation.items(), 
                                key=lambda x: x[1]['separation'], 
                                reverse=True)
        
        # Select top N features
        top_features = [f[0] for f in sorted_features[:top_n]]
        logger.info(f"Top {len(top_features)} features by separation: {top_features}")
        
        # Create plots for each feature
        # For each feature in top_features
        for feature in top_features:
            try:
                plt.figure(figsize=(12, 8))
                
                # Get data and statistics
                pos_data = df[df['true_label'] == 1][feature]
                neg_data = df[df['true_label'] == 0][feature]
                pos_mean = feature_separation[feature]['pos_mean']
                neg_mean = feature_separation[feature]['neg_mean']
                separation = feature_separation[feature]['separation']
                
                # Check if feature has sufficient variance for histogram
                if pos_data.nunique() <= 1 or neg_data.nunique() <= 1:
                    # Use a simple bar chart for features with limited variance
                    plt.bar(['Non-Match', 'Match'], [neg_mean, pos_mean], 
                        color=[self.colors['non_match'], self.colors['match']])
                    
                    plt.title(f'Values of {feature} by Class (Limited Distribution)', 
                            fontsize=16)
                else:
                    # Data validation - filter out NaN, inf values and check for extreme outliers
                    pos_data = pos_data.replace([np.inf, -np.inf], np.nan).dropna()
                    neg_data = neg_data.replace([np.inf, -np.inf], np.nan).dropna()
                    
                    # Calculate reasonable limits to exclude extreme outliers
                    # Using 5 standard deviations as a threshold
                    pos_std = pos_data.std()
                    neg_std = neg_data.std()
                    pos_limit = pos_mean + 5 * pos_std if not np.isnan(pos_std) else pos_data.max()
                    neg_limit = neg_mean + 5 * neg_std if not np.isnan(neg_std) else neg_data.max()
                    
                    # Filter outliers
                    pos_data = pos_data[pos_data.between(pos_mean - 5 * pos_std, pos_mean + 5 * pos_std)] if not np.isnan(pos_std) else pos_data
                    neg_data = neg_data[neg_data.between(neg_mean - 5 * neg_std, neg_mean + 5 * neg_std)] if not np.isnan(neg_std) else neg_data
                    
                    # Explicitly set a reasonable number of bins
                    bins = min(30, max(10, int(np.sqrt(len(pos_data) + len(neg_data)))))
                    
                    # Plot histograms with KDE using the updated color scheme and explicit bins
                    sns.histplot(pos_data, kde=True, stat='density', alpha=0.6, 
                                color=self.colors['match'], label='Matching Entities', 
                                edgecolor='white', linewidth=0.5, bins=bins)
                    sns.histplot(neg_data, kde=True, stat='density', alpha=0.6, 
                                color=self.colors['non_match'], label='Non-matching Entities', 
                                edgecolor='white', linewidth=0.5, bins=bins)
                    
                    # Add mean lines with consistent colors
                    plt.axvline(pos_mean, color=self.colors['match_line'], linestyle='--', 
                            label=f'Match Mean: {pos_mean:.4f}')
                    plt.axvline(neg_mean, color=self.colors['non_match_line'], linestyle='--', 
                            label=f'Non-match Mean: {neg_mean:.4f}')
                    
                    plt.title(f'Distribution of {feature} by Class (Separation: {separation:.4f})', 
                            fontsize=16)
                
                # Common styling
                plt.xlabel('Feature Value', fontsize=14)
                plt.ylabel('Density', fontsize=14)
                plt.grid(alpha=0.3)
                plt.legend(fontsize=12)
                plt.tight_layout()
                
                # Save figure
                plt.savefig(self.figures_dir / f'feature_dist_{feature}.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                # If we have multiple representations, create a comparison plot
                if (f"{feature}_raw" in df.columns and 
                    f"{feature}_norm" in df.columns and 
                    f"{feature}_std" in df.columns):
                    self._create_feature_comparison_plot(df, feature)

            except Exception as e:
                logger.error(f"Error creating plot for feature {feature}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                plt.close()  # Make sure to close any open figures
                continue                
        
        # Create a summary plot of feature separation power
        self._create_feature_separation_plot(top_features, feature_separation)
        
        return top_features
    
    def _create_feature_separation_plot(self, top_features, feature_separation):
        """
        Create a summary plot showing feature separation power.
        
        Args:
            top_features (list): List of top features
            feature_separation (dict): Dictionary of feature separation metrics
        """
        try:
            plt.figure(figsize=(14, 8))
            
            # Extract separation values for the top features
            features = top_features
            separations = [feature_separation[f]['separation'] for f in features]
            
            # Create a DataFrame for easier plotting
            data = pd.DataFrame({
                'Feature': features,
                'Separation': separations
            }).sort_values('Separation', ascending=False)
            
            # Use the blue color palette for consistency
            colors = plt.cm.Blues(np.linspace(0.6, 0.9, len(features)))
            
            # Plot horizontal bars
            bars = plt.barh(data['Feature'], data['Separation'], color=colors)
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                plt.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{width:.4f}', ha='left', va='center', fontsize=10)
            
            plt.title('Feature Separation Power (Higher = Better Class Separation)', fontsize=16)
            plt.xlabel('Mean Absolute Difference Between Classes', fontsize=14)
            plt.ylabel('Feature', fontsize=14)
            plt.grid(axis='x', linestyle='--', alpha=0.3)
            plt.tight_layout()
            
            # Save figure
            plt.savefig(self.figures_dir / 'feature_separation_power.png', dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Created feature separation power plot")
            
        except Exception as e:
            logger.error(f"Error creating feature separation plot: {e}")
            import traceback
            logger.error(traceback.format_exc())
            plt.close()  # Make sure to close any open figures

    def _create_feature_comparison_plot(self, df, feature):
        """
        Create a comparison plot showing raw, normalized, and standardized representations.
        
        Args:
            df (DataFrame): Test results dataframe
            feature (str): Base feature name to plot
        """
        try:
            # Create a multi-panel plot showing all three representations
            plt.figure(figsize=(18, 6))
            
            # Plot 1: Raw values
            plt.subplot(1, 3, 1)
            raw_pos = df[df['true_label'] == 1][f"{feature}_raw"]
            raw_neg = df[df['true_label'] == 0][f"{feature}_raw"]
            
            # Data validation and outlier filtering
            raw_pos = raw_pos.replace([np.inf, -np.inf], np.nan).dropna()
            raw_neg = raw_neg.replace([np.inf, -np.inf], np.nan).dropna()
            
            # Calculate statistics for outlier detection
            raw_pos_mean = raw_pos.mean()
            raw_neg_mean = raw_neg.mean()
            raw_pos_std = raw_pos.std()
            raw_neg_std = raw_neg.std()
            
            # Filter outliers if we have valid statistics
            if not np.isnan(raw_pos_std) and raw_pos_std > 0:
                raw_pos = raw_pos[raw_pos.between(raw_pos_mean - 5 * raw_pos_std, raw_pos_mean + 5 * raw_pos_std)]
            if not np.isnan(raw_neg_std) and raw_neg_std > 0:
                raw_neg = raw_neg[raw_neg.between(raw_neg_mean - 5 * raw_neg_std, raw_neg_mean + 5 * raw_neg_std)]
            
            # Explicitly set a reasonable number of bins
            bins = min(30, max(10, int(np.sqrt(len(raw_pos) + len(raw_neg)))))
            
            if raw_pos.nunique() > 1 and raw_neg.nunique() > 1:
                sns.histplot(raw_pos, kde=True, stat='density', alpha=0.6, 
                            color=self.colors['match'], label='Match', edgecolor='white', bins=bins)
                sns.histplot(raw_neg, kde=True, stat='density', alpha=0.6, 
                            color=self.colors['non_match'], label='Non-match', edgecolor='white', bins=bins)
            else:
                plt.bar(['Non-Match', 'Match'], [raw_neg.mean(), raw_pos.mean()], 
                    color=[self.colors['non_match'], self.colors['match']])
            
            plt.title(f'Raw Values (-1 to 1)', fontsize=14)
            plt.xlabel('Raw Value', fontsize=12)
            plt.ylabel('Density', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(alpha=0.3)
            
            # Plot 2: Normalized values
            plt.subplot(1, 3, 2)
            norm_pos = df[df['true_label'] == 1][f"{feature}_norm"]
            norm_neg = df[df['true_label'] == 0][f"{feature}_norm"]
            
            # Data validation and outlier filtering
            norm_pos = norm_pos.replace([np.inf, -np.inf], np.nan).dropna()
            norm_neg = norm_neg.replace([np.inf, -np.inf], np.nan).dropna()
            
            # Calculate statistics for outlier detection
            norm_pos_mean = norm_pos.mean()
            norm_neg_mean = norm_neg.mean()
            norm_pos_std = norm_pos.std()
            norm_neg_std = norm_neg.std()
            
            # Filter outliers if we have valid statistics
            if not np.isnan(norm_pos_std) and norm_pos_std > 0:
                norm_pos = norm_pos[norm_pos.between(norm_pos_mean - 5 * norm_pos_std, norm_pos_mean + 5 * norm_pos_std)]
            if not np.isnan(norm_neg_std) and norm_neg_std > 0:
                norm_neg = norm_neg[norm_neg.between(norm_neg_mean - 5 * norm_neg_std, norm_neg_mean + 5 * norm_neg_std)]
            
            if norm_pos.nunique() > 1 and norm_neg.nunique() > 1:
                sns.histplot(norm_pos, kde=True, stat='density', alpha=0.6, 
                            color=self.colors['match'], label='Match', edgecolor='white', bins=bins)
                sns.histplot(norm_neg, kde=True, stat='density', alpha=0.6, 
                            color=self.colors['non_match'], label='Non-match', edgecolor='white', bins=bins)
            else:
                plt.bar(['Non-Match', 'Match'], [norm_neg.mean(), norm_pos.mean()], 
                    color=[self.colors['non_match'], self.colors['match']])
            
            plt.title(f'Domain-Normalized (0 to 1)', fontsize=14)
            plt.xlabel('Normalized Value', fontsize=12)
            plt.grid(alpha=0.3)
            plt.legend(fontsize=10)
            
            # Plot 3: StandardScaler values
            plt.subplot(1, 3, 3)
            std_pos = df[df['true_label'] == 1][feature]  # Main column has standardized values
            std_neg = df[df['true_label'] == 0][feature]
            
            # Data validation and outlier filtering
            std_pos = std_pos.replace([np.inf, -np.inf], np.nan).dropna()
            std_neg = std_neg.replace([np.inf, -np.inf], np.nan).dropna()
            
            # Calculate statistics for outlier detection
            std_pos_mean = std_pos.mean()
            std_neg_mean = std_neg.mean()
            std_pos_std = std_pos.std()
            std_neg_std = std_neg.std()
            
            # Filter outliers if we have valid statistics
            if not np.isnan(std_pos_std) and std_pos_std > 0:
                std_pos = std_pos[std_pos.between(std_pos_mean - 5 * std_pos_std, std_pos_mean + 5 * std_pos_std)]
            if not np.isnan(std_neg_std) and std_neg_std > 0:
                std_neg = std_neg[std_neg.between(std_neg_mean - 5 * std_neg_std, std_neg_mean + 5 * std_neg_std)]
            
            if std_pos.nunique() > 1 and std_neg.nunique() > 1:
                sns.histplot(std_pos, kde=True, stat='density', alpha=0.6, 
                            color=self.colors['match'], label='Match', edgecolor='white', bins=bins)
                sns.histplot(std_neg, kde=True, stat='density', alpha=0.6, 
                            color=self.colors['non_match'], label='Non-match', edgecolor='white', bins=bins)
            else:
                plt.bar(['Non-Match', 'Match'], [std_neg.mean(), std_pos.mean()], 
                    color=[self.colors['non_match'], self.colors['match']])
            
            plt.title(f'StandardScaler Values', fontsize=14)
            plt.xlabel('Standardized Value', fontsize=12)
            plt.grid(alpha=0.3)
            plt.legend(fontsize=10)
            
            # Add overall title
            plt.suptitle(f'Comparison of {feature} Representations', fontsize=16, y=1.05)
            plt.tight_layout()
            
            # Save comparison figure
            plt.savefig(self.figures_dir / f'feature_comparison_{feature}.png', dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Created comparison plot for {feature}")
        
        except Exception as e:
            logger.error(f"Error creating comparison plot for feature {feature}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            plt.close()  # Make sure to close the figure even if there's an error
    
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
            'metrics': {},
            'feature_representations': {
                'standard': 'Main feature columns contain StandardScaler values used for training',
                'raw': 'Raw feature values in original range (e.g., [-1,1] for cosine similarity)',
                'normalized': 'Domain-normalized feature values in [0,1] range'
            }
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
            
            f.write("## Feature Representations\n\n")
            f.write("This system uses multiple feature representations for clarity:\n\n")
            f.write("- **StandardScaler Values**: Used for model training (mean=0, std=1)\n")
            f.write("- **Domain-Normalized Values**: Intuitive [0,1] range for interpretation\n")
            f.write("- **Raw Values**: Original values (e.g., [-1,1] for cosine similarity)\n\n")
            
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
            'feature_importance': feature_importance,
            'feature_representations': {
                'note': 'Feature importance is based on the StandardScaler values used for model training'
            }
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
            
            f.write("## Understanding Feature Values\n\n")
            f.write("Feature importance is calculated based on the StandardScaler values used for model training.\n")
            f.write("These features are normalized to have mean=0 and standard deviation=1, which helps the model\n")
            f.write("give appropriate weight to each feature regardless of its original scale.\n\n")
            
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
            self._visualize_feature_importance(feature_importance_df)
        
        logger.info(f"Feature importance report saved to {report_path}")
        logger.info(f"Feature importance summary saved to {summary_path}")
        
        return str(report_path)
    
    def _visualize_feature_importance(self, df):
        """
        Create visualizations of feature importance.
        
        Args:
            df (DataFrame): Feature importance dataframe
        """
        try:
            # Sort by importance
            df = df.sort_values('importance', ascending=False)
            
            # Plot top features
            plt.figure(figsize=(12, 8))
            
            # Bar plot of feature importance
            plt.subplot(2, 1, 1)
            sns.barplot(x='importance', y='feature', data=df.head(15), palette='Blues_r')
            plt.title('Top 15 Features by Importance', fontsize=16)
            plt.xlabel('Importance', fontsize=14)
            plt.ylabel('Feature', fontsize=14)
            plt.grid(axis='x', alpha=0.3, linestyle='--')
            
            # Bar plot of feature weights
            plt.subplot(2, 1, 2)
            # Use different colors for positive and negative weights
            colors = ['#d73027' if w < 0 else '#4575b4' for w in df.head(15)['weight']]
            sns.barplot(x='weight', y='feature', data=df.head(15), palette=colors)
            plt.title('Top 15 Features by Weight', fontsize=16)
            plt.xlabel('Weight', fontsize=14)
            plt.ylabel('Feature', fontsize=14)
            plt.grid(axis='x', alpha=0.3, linestyle='--')
            
            plt.tight_layout()
            
            # Save figure
            fig_path = self.figures_dir / "feature_importance.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Grouped by feature type
            plt.figure(figsize=(10, 6))
            feature_type_df = df.groupby('feature_type')['importance'].sum().reset_index().sort_values('importance', ascending=False)
            sns.barplot(x='importance', y='feature_type', data=feature_type_df, palette='Blues_r')
            plt.title('Feature Importance by Type', fontsize=16)
            plt.xlabel('Total Importance', fontsize=14)
            plt.ylabel('Feature Type', fontsize=14)
            plt.grid(axis='x', alpha=0.3, linestyle='--')
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
            # Process test results to ensure we have all representations
            enhanced_path, test_df = self._process_test_results(test_results_path)
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
                'confidence_stats': confidence_stats.to_dict(orient='records'),
                'feature_representations': {
                    'standard': 'Main feature columns contain StandardScaler values',
                    'raw': 'Columns with _raw suffix contain original [-1,1] values',
                    'normalized': 'Columns with _norm suffix contain domain-normalized [0,1] values' 
                }
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
                
                f.write("## Feature Representation Guide\n\n")
                f.write("This report includes multiple representations of feature values:\n\n")
                f.write("- **Standard Features** (e.g., `person_cosine`): StandardScaler values used for model training\n")
                f.write("- **Raw Features** (e.g., `person_cosine_raw`): Original values in their native range\n") 
                f.write("- **Normalized Features** (e.g., `person_cosine_norm`): Domain-normalized values in [0,1] range\n\n")
                
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
                    metadata_cols = ['pair_id', 'left_id', 'right_id', 'true_label', 
                                    'predicted_label', 'confidence', 'correct']
                    derived_suffixes = ['_raw', '_norm', '_std']
                    feature_cols = [col for col in test_df.columns 
                                    if col not in metadata_cols and 
                                    not any(col.endswith(suffix) for suffix in derived_suffixes)]
                    
                    if len(feature_cols) > 0:
                        # Calculate average standardized feature values for correct and incorrect predictions
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
                                raw_value = row.get(f"{feature}_raw", "N/A")
                                norm_value = row.get(f"{feature}_norm", "N/A")
                                std_value = row[feature]
                                
                                # Format the display based on what values we have
                                if isinstance(raw_value, (int, float)) and isinstance(norm_value, (int, float)):
                                    f.write(f"    - {feature}: {std_value:.4f} (raw: {raw_value:.4f}, norm: {norm_value:.4f})\n")
                                else:
                                    f.write(f"    - {feature}: {std_value:.4f}\n")
                            f.write("\n")
                        
                        f.write("### False Negatives (Predicted Different, Actually Match)\n\n")
                        for i, row in false_negatives.head(5).iterrows():
                            f.write(f"- **Pair**: {row['left_id']} - {row['right_id']}\n")
                            f.write(f"  - Confidence: {row['confidence']:.4f}\n")
                            f.write("  - Top Features:\n")
                            
                            top_features = feature_cols[:3] if len(feature_cols) >= 3 else feature_cols
                            # Show top 3 features (if available)
                            for feature in top_features:
                                raw_value = row.get(f"{feature}_raw", "N/A")
                                norm_value = row.get(f"{feature}_norm", "N/A")
                                std_value = row[feature]
                                
                                # Format the display based on what values we have
                                if isinstance(raw_value, (int, float)) and isinstance(norm_value, (int, float)):
                                    f.write(f"    - {feature}: {std_value:.4f} (raw: {raw_value:.4f}, norm: {norm_value:.4f})\n")
                                else:
                                    f.write(f"    - {feature}: {std_value:.4f}\n")
                            f.write("\n")
            
            # Generate error visualizations
            if self.visualization_enabled:
                self._create_error_visualizations(test_df)
            
            # Save error cases to CSV for further analysis
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
    
    def _create_error_visualizations(self, df):
        """
        Create visualizations for error analysis.
        
        Args:
            df (DataFrame): Test results dataframe
        """
        try:
            # Confusion matrix visualization
            plt.figure(figsize=(10, 8))
            cm = confusion_matrix(df['true_label'], df['predicted_label'])
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
                data=df,
                x='confidence',
                hue='correct',
                multiple='stack',
                bins=20,
                palette={True: self.colors['match'], False: self.colors['non_match']}
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
            
            logger.info(f"Error visualizations saved to {self.figures_dir}")
        
        except Exception as e:
            logger.error(f"Error creating error visualizations: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _generate_error_analysis(self):
        """
        Generate error analysis report.
        
        Returns:
            str: Path to report file
        """
        # This is already covered in _analyze_test_results
        return None
    
    def _generate_cluster_statistics(self):
        """
        Generate cluster statistics report.
        
        Returns:
            str: Path to report file
        """
        logger.info("Generating cluster statistics report")
        
        # Load clusters from CSV
        clusters_csv_path = self.output_dir / "clusters.csv"
        if not clusters_csv_path.exists():
            # Fall back to JSON file
            clusters_path = self.output_dir / "clusters.json"
            if not clusters_path.exists():
                logger.warning("Clusters not found")
                return None
            
            # Load from JSON
            with open(clusters_path, 'r') as f:
                clusters = json.load(f)
            
            # Convert to DataFrame
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
            # Load from CSV
            clusters_df = pd.read_csv(clusters_csv_path)
        
        # Compute cluster statistics
        cluster_sizes = clusters_df['cluster_size'].unique()
        cluster_counts = clusters_df.groupby('cluster_id').size().reset_index(name='count')
        
        stats = {
            'title': 'Entity Resolution Cluster Statistics',
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_clusters': clusters_df['cluster_id'].nunique(),
            'total_records': len(clusters_df),
            'average_cluster_size': float(clusters_df['cluster_size'].mean()),
            'median_cluster_size': float(clusters_df['cluster_size'].median()),
            'min_cluster_size': int(clusters_df['cluster_size'].min()),
            'max_cluster_size': int(clusters_df['cluster_size'].max()),
            'std_cluster_size': float(clusters_df['cluster_size'].std()),
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
            self._create_cluster_visualizations(clusters_df, stats)
        
        logger.info(f"Cluster statistics report saved to {report_path}")
        logger.info(f"Cluster statistics summary saved to {summary_path}")
        
        return str(report_path)
    
    def _create_cluster_visualizations(self, clusters_df, stats):
        """
        Create visualizations for cluster statistics.
        
        Args:
            clusters_df (DataFrame): Clusters dataframe
            stats (dict): Cluster statistics
        """
        try:
            # Cluster size distribution
            plt.figure(figsize=(12, 8))
            
            # Prepare data for visualization
            size_ranges = list(stats['cluster_size_distribution'].keys())
            counts = list(stats['cluster_size_distribution'].values())
            
            # Bar plot with modern colors
            plt.bar(size_ranges, counts, color=plt.cm.Blues(np.linspace(0.4, 0.8, len(size_ranges))))
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
            
            # Top 10 largest clusters visualization
            plt.figure(figsize=(14, 8))
            top_clusters = clusters_df.groupby('cluster_id').size().reset_index(name='entity_count')
            top_clusters = top_clusters.sort_values('entity_count', ascending=False).head(10)
            
            plt.bar(top_clusters['cluster_id'].astype(str), top_clusters['entity_count'], 
                   color=plt.cm.Blues(np.linspace(0.4, 0.8, len(top_clusters))))
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
            
            logger.info(f"Cluster visualizations saved to {self.figures_dir}")
        
        except Exception as e:
            logger.error(f"Error creating cluster visualizations: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
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
            
            # Plot ROC curve with updated color scheme
            plt.figure(figsize=(12, 10))
            plt.plot(fpr, tpr, color=self.colors['match_line'], lw=3, label=f'ROC curve (area = {roc_auc:.2f})')
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
            if 0 <= decision_idx < len(fpr):
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
            f1_scores = np.zeros_like(thresholds)
            for i in range(len(thresholds)):
                f1_scores[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i] + 1e-10)
            
            best_threshold_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_threshold_idx]
            best_f1 = f1_scores[best_threshold_idx]
            
            # Calculate baseline performance (random classifier)
            baseline = test_df['true_label'].mean()
            
            # Plot precision-recall curve with updated color scheme
            plt.figure(figsize=(12, 10))
            plt.plot(recall, precision, color=self.colors['match_line'], lw=3, label='Precision-Recall curve')
            plt.axhline(y=baseline, color=self.colors['non_match_line'], linestyle='--', 
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
                
                return [str(fig_path), str(threshold_fig_path)]
            
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
            
            bars = ax1.bar(metrics, values, color=plt.cm.Blues(np.linspace(0.4, 0.8, len(metrics))))
            
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
                bars = ax3.barh(stages, durations, color=plt.cm.Blues(np.linspace(0.4, 0.8, len(stages))))
                
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
                bars = ax4.barh(y_pos, importance_values, color=plt.cm.Blues(np.linspace(0.4, 0.8, len(top_features))))
                
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
    
    def _generate_rfe_report(self):
        """
        Generate a report on the recursive feature elimination results.
        
        Returns:
            str: Path to report file
        """
        logger.info("Generating recursive feature elimination report")
        
        # Check if RFE was enabled in configuration
        if not self.config['features']['rfe_enabled']:
            logger.info("RFE was not enabled in configuration, skipping report")
            return None
        
        # Load feature engineering metadata to find RFE results
        metadata_path = self.output_dir / "ground_truth_features_metadata.json"
        if not metadata_path.exists():
            logger.warning("Feature engineering metadata not found")
            return None
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load RFE model if available
        rfe_path = metadata.get('rfe_path')
        if not rfe_path or not os.path.exists(rfe_path):
            logger.warning("RFE model not found")
            return None
        
        with open(rfe_path, 'rb') as f:
            import pickle
            rfe_model = pickle.load(f)
        
        # Extract feature names
        feature_names = metadata.get('feature_names', [])
        if not feature_names:
            logger.warning("Feature names not found in metadata")
            return None
        
        # Create RFE report
        report = {
            'title': 'Recursive Feature Elimination Results',
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'config': {
                'rfe_step_size': self.config['features']['rfe_step_size'],
                'rfe_cv_folds': self.config['features']['rfe_cv_folds']
            },
            'selected_features': [],
            'eliminated_features': [],
            'cross_validation_scores': []
        }
        
        # Extract selected features
        if hasattr(rfe_model, 'support_'):
            report['selected_features'] = [
                feature_names[i] for i in range(len(feature_names)) 
                if i < len(rfe_model.support_) and rfe_model.support_[i]
            ]
            report['eliminated_features'] = [
                feature_names[i] for i in range(len(feature_names)) 
                if i < len(rfe_model.support_) and not rfe_model.support_[i]
            ]
        
        # Extract cross-validation scores if available
        if hasattr(rfe_model, 'grid_scores_'):
            report['cross_validation_scores'] = rfe_model.grid_scores_.tolist()
        elif hasattr(rfe_model, 'cv_results_'):
            report['cross_validation_scores'] = rfe_model.cv_results_.get('mean_test_score', []).tolist()
        
        # Extract ranking if available
        if hasattr(rfe_model, 'ranking_'):
            feature_ranking = []
            for i, rank in enumerate(rfe_model.ranking_):
                if i < len(feature_names):
                    feature_ranking.append({
                        'feature': feature_names[i],
                        'rank': int(rank)
                    })
            report['feature_ranking'] = sorted(feature_ranking, key=lambda x: x['rank'])
        
        # Save report
        report_path = self.reports_dir / "rfe_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Create readable summary in Markdown
        summary_path = self.reports_dir / "rfe_summary.md"
        with open(summary_path, 'w') as f:
            f.write(f"# Recursive Feature Elimination Results\n\n")
            f.write(f"Generated: {report['timestamp']}\n\n")
            
            f.write("## Configuration\n\n")
            f.write(f"- RFE Step Size: {report['config']['rfe_step_size']}\n")
            f.write(f"- Cross-Validation Folds: {report['config']['rfe_cv_folds']}\n\n")
            
            f.write("## Selected Features\n\n")
            for i, feature in enumerate(report['selected_features'], 1):
                f.write(f"{i}. {feature}\n")
            
            if 'feature_ranking' in report:
                f.write("\n## Feature Ranking\n\n")
                f.write("| Rank | Feature |\n")
                f.write("|------|--------|\n")
                
                for item in report['feature_ranking']:
                    f.write(f"| {item['rank']} | {item['feature']} |\n")
            
            if report['cross_validation_scores']:
                f.write("\n## Cross-Validation Performance\n\n")
                f.write("The following scores show model performance at each step of feature elimination:\n\n")
                for i, score in enumerate(report['cross_validation_scores']):
                    num_features = len(report['selected_features']) + len(report['eliminated_features']) - i * report['config']['rfe_step_size']
                    f.write(f"- {num_features} features: {score:.4f}\n")
                
            f.write("\n## Eliminated Features\n\n")
            for i, feature in enumerate(report['eliminated_features'], 1):
                f.write(f"{i}. {feature}\n")
        
        # Visualize RFE results
        if self.visualization_enabled:
            self._visualize_rfe_results(report)
        
        logger.info(f"RFE report saved to {report_path}")
        logger.info(f"RFE summary saved to {summary_path}")
        
        return str(report_path)

    def _visualize_rfe_results(self, rfe_report):
        """
        Create visualizations for recursive feature elimination results.
        
        Args:
            rfe_report (dict): RFE report data
        """
        try:
            # Check if we have cross-validation scores for performance plot
            if 'cross_validation_scores' in rfe_report and rfe_report['cross_validation_scores']:
                plt.figure(figsize=(12, 8))
                
                scores = rfe_report['cross_validation_scores']
                num_features_range = []
                
                # Calculate number of features at each step
                total_features = len(rfe_report['selected_features']) + len(rfe_report['eliminated_features'])
                step_size = rfe_report['config']['rfe_step_size']
                
                for i in range(len(scores)):
                    num_features = total_features - i * step_size
                    num_features_range.append(num_features)
                
                # Plot performance vs number of features
                plt.plot(num_features_range, scores, 'o-', color=self.colors['match_line'], 
                        linewidth=3, markersize=8)
                
                # Add labels for key points
                best_idx = np.argmax(scores)
                best_score = scores[best_idx]
                best_num_features = num_features_range[best_idx]
                
                plt.scatter([best_num_features], [best_score], s=200, c='red', zorder=3,
                        label=f'Best: {best_num_features} features, score: {best_score:.4f}')
                
                # Styling
                plt.xlabel('Number of Features', fontsize=14)
                plt.ylabel('Cross-Validation Score (F1)', fontsize=14)
                plt.title('Performance vs Number of Features in RFE', fontsize=16)
                plt.grid(linestyle='--', alpha=0.7)
                
                # Add vertical line for optimal feature count
                plt.axvline(x=best_num_features, color='red', linestyle='--', alpha=0.5)
                
                # If x-axis has enough points, use integer ticks
                if len(num_features_range) > 1:
                    plt.xticks(num_features_range)
                
                plt.legend(loc='best', fontsize=12)
                plt.tight_layout()
                
                # Save figure
                fig_path = self.figures_dir / "rfe_performance_curve.png"
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"RFE performance curve saved to {fig_path}")
            
            # Feature ranking visualization
            if 'feature_ranking' in rfe_report:
                # Get top 20 features by rank
                feature_ranking = rfe_report['feature_ranking']
                top_n = min(20, len(feature_ranking))
                top_features = feature_ranking[:top_n]
                
                plt.figure(figsize=(12, 10))
                
                # Invert the order so the highest ranked feature is at the top
                features = [item['feature'] for item in reversed(top_features)]
                ranks = [item['rank'] for item in reversed(top_features)]
                
                # Create a colormap from blue to light blue based on rank
                colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(features)))
                
                bars = plt.barh(features, [max(ranks) - r + 1 for r in ranks], color=colors)
                
                # Add value labels
                for bar in bars:
                    width = bar.get_width()
                    plt.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                            f'{width:.0f}', ha='left', va='center', fontsize=10)
                
                plt.xlabel('Relative Importance (inverse of rank)', fontsize=14)
                plt.ylabel('Feature', fontsize=14)
                plt.title('Top Features by RFE Ranking', fontsize=16)
                plt.grid(axis='x', linestyle='--', alpha=0.5)
                plt.tight_layout()
                
                # Save figure
                fig_path = self.figures_dir / "rfe_feature_ranking.png"
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"RFE feature ranking visualization saved to {fig_path}")
                
            # Selected vs Eliminated Features Comparison
            plt.figure(figsize=(10, 6))
            counts = [len(rfe_report['selected_features']), len(rfe_report['eliminated_features'])]
            labels = ['Selected', 'Eliminated']
            
            plt.bar(labels, counts, color=[self.colors['match'], self.colors['non_match']])
            
            # Add count labels
            for i, count in enumerate(counts):
                plt.text(i, count + 0.5, str(count), ha='center', va='bottom', fontsize=12)
            
            plt.ylabel('Number of Features', fontsize=14)
            plt.title('Feature Selection Results', fontsize=16)
            plt.grid(axis='y', linestyle='--', alpha=0.5)
            plt.tight_layout()
            
            # Save figure
            fig_path = self.figures_dir / "rfe_feature_counts.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"RFE feature counts visualization saved to {fig_path}")
            
        except Exception as e:
            logger.error(f"Error creating RFE visualizations: {e}")
            import traceback
            logger.error(traceback.format_exc())