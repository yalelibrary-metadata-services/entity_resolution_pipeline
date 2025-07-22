"""
Feature Scaling Visualization for Library Catalog Entity Resolution

This module provides visualization tools to assess and monitor feature distributions
before and after scaling, helping identify compression issues and evaluate the
effectiveness of different scaling approaches.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Any, Optional, Union

logger = logging.getLogger(__name__)

class ScalingVisualizer:
    """
    Visualization tools for analyzing feature scaling in entity resolution.
    
    Provides methods for visualizing feature distributions, compression effects,
    and separation power before and after scaling.
    """
    
    def __init__(self, output_dir="./scaling_viz"):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set default styling
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        
        logger.info(f"ScalingVisualizer initialized. Outputs will be saved to {self.output_dir}")
    
    def visualize_scaling_comparison(self, X_original, X_scaled, feature_names, 
                                    y_true=None, method_name="robust_scaling"):
        """
        Create comprehensive scaling comparison visualizations.
        
        Args:
            X_original: Original feature values
            X_scaled: Scaled feature values
            feature_names: Names of features
            y_true: True labels (optional)
            method_name: Name of scaling method for file naming
            
        Returns:
            Dictionary mapping visualization types to file paths
        """
        output_files = {}
        
        # Create distribution comparison visualization
        dist_file = self.visualize_distributions(X_original, X_scaled, feature_names, method_name)
        output_files["distributions"] = dist_file
        
        # Create compression visualization
        compression_file = self.visualize_compression(X_original, X_scaled, feature_names, method_name)
        output_files["compression"] = compression_file
        
        # Create separation visualization if labels provided
        if y_true is not None:
            separation_file = self.visualize_separation(X_original, X_scaled, feature_names, y_true, method_name)
            output_files["separation"] = separation_file
            
            # Create ROC curves
            roc_file = self.visualize_roc_curves(X_original, X_scaled, feature_names, y_true, method_name)
            output_files["roc_curves"] = roc_file
        
        # Create correlation matrix visualization
        corr_file = self.visualize_correlation(X_scaled, feature_names, method_name)
        output_files["correlation"] = corr_file
        
        # Create report with all visualizations
        report_file = self.create_scaling_report(output_files, method_name)
        output_files["report"] = report_file
        
        return output_files
    
    def visualize_distributions(self, X_original, X_scaled, feature_names, method_name="robust_scaling"):
        """
        Visualize feature distributions before and after scaling.
        
        Args:
            X_original: Original feature values
            X_scaled: Scaled feature values
            feature_names: Names of features
            method_name: Name of scaling method for file naming
            
        Returns:
            Path to saved visualization
        """
        # Determine number of features and create figure layout
        n_features = len(feature_names)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        # Create figure with subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Create distribution plots for each feature
        for i, (feature_idx, ax) in enumerate(zip(range(n_features), axes)):
            feature_name = feature_names[feature_idx]
            
            # Get original and scaled values
            orig_vals = X_original[:, feature_idx]
            scaled_vals = X_scaled[:, feature_idx]
            
            # Create KDE plots
            sns.kdeplot(orig_vals, ax=ax, color="blue", label="Original", fill=True, alpha=0.3)
            sns.kdeplot(scaled_vals, ax=ax, color="red", label="Scaled", fill=True, alpha=0.3)
            
            # Add distribution statistics
            orig_stats = f"Original: μ={np.mean(orig_vals):.3f}, σ={np.std(orig_vals):.3f}"
            scaled_stats = f"Scaled: μ={np.mean(scaled_vals):.3f}, σ={np.std(scaled_vals):.3f}"
            
            # Add plot elements
            ax.set_title(feature_name)
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
            ax.text(0.05, 0.95, orig_stats, transform=ax.transAxes, va="top", fontsize=10)
            ax.text(0.05, 0.90, scaled_stats, transform=ax.transAxes, va="top", fontsize=10)
            ax.legend()
            
            # Check for binary feature
            if set(np.unique(orig_vals)).issubset({0, 1}) and set(np.unique(scaled_vals)).issubset({0, 1}):
                ax.set_xticks([0, 1])
                ax.set_xticklabels(["0", "1"])
                ax.set_title(f"{feature_name} (Binary)")
        
        # Hide empty subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        # Add overall title
        plt.suptitle(f"Feature Distributions Before and After {method_name.replace('_', ' ').title()}", 
                    fontsize=16, y=1.02)
        plt.tight_layout()
        
        # Save figure
        output_file = self.output_dir / f"distributions_{method_name}.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Distribution visualization saved to {output_file}")
        return str(output_file)
    
    def visualize_compression(self, X_original, X_scaled, feature_names, method_name="robust_scaling"):
        """
        Visualize compression effect of scaling on feature ranges.
        
        Args:
            X_original: Original feature values
            X_scaled: Scaled feature values
            feature_names: Names of features
            method_name: Name of scaling method for file naming
            
        Returns:
            Path to saved visualization
        """
        # Calculate range metrics
        compression_data = []
        
        for i, name in enumerate(feature_names):
            orig_vals = X_original[:, i]
            scaled_vals = X_scaled[:, i]
            
            # Calculate original and scaled ranges
            orig_range = np.max(orig_vals) - np.min(orig_vals)
            scaled_range = np.max(scaled_vals) - np.min(scaled_vals)
            
            # Calculate distribution statistics
            orig_stats = {
                "min": np.min(orig_vals),
                "max": np.max(orig_vals),
                "mean": np.mean(orig_vals),
                "median": np.median(orig_vals),
                "q1": np.percentile(orig_vals, 25),
                "q3": np.percentile(orig_vals, 75),
                "p5": np.percentile(orig_vals, 5),
                "p95": np.percentile(orig_vals, 95)
            }
            
            scaled_stats = {
                "min": np.min(scaled_vals),
                "max": np.max(scaled_vals),
                "mean": np.mean(scaled_vals),
                "median": np.median(scaled_vals),
                "q1": np.percentile(scaled_vals, 25),
                "q3": np.percentile(scaled_vals, 75),
                "p5": np.percentile(scaled_vals, 5),
                "p95": np.percentile(scaled_vals, 95)
            }
            
            # Calculate compression metrics
            compression_factor = orig_range / scaled_range if scaled_range > 0 else float('inf')
            pct_range_used = (scaled_range / orig_range) * 100 if orig_range > 0 else 100.0
            
            # Create p5-p95 range for comparison (exclude outliers)
            orig_p5_p95_range = orig_stats["p95"] - orig_stats["p5"]
            scaled_p5_p95_range = scaled_stats["p95"] - scaled_stats["p5"]
            p5_p95_compression = orig_p5_p95_range / scaled_p5_p95_range if scaled_p5_p95_range > 0 else float('inf')
            
            # Save to data list
            compression_data.append({
                "feature": name,
                "orig_min": orig_stats["min"],
                "orig_max": orig_stats["max"],
                "orig_range": orig_range,
                "orig_p5": orig_stats["p5"],
                "orig_p95": orig_stats["p95"],
                "orig_p5_p95_range": orig_p5_p95_range,
                "scaled_min": scaled_stats["min"],
                "scaled_max": scaled_stats["max"],
                "scaled_range": scaled_range,
                "scaled_p5": scaled_stats["p5"],
                "scaled_p95": scaled_stats["p95"],
                "scaled_p5_p95_range": scaled_p5_p95_range,
                "compression_factor": compression_factor,
                "p5_p95_compression": p5_p95_compression,
                "pct_range_used": pct_range_used
            })
        
        # Create DataFrame for visualization
        df = pd.DataFrame(compression_data)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Plot 1: Range comparison
        range_df = pd.melt(df, id_vars=["feature"], 
                          value_vars=["orig_range", "orig_p5_p95_range", "scaled_range", "scaled_p5_p95_range"],
                          var_name="range_type", value_name="range")
        
        # Add dataset column for proper grouping
        range_df["dataset"] = range_df["range_type"].apply(
            lambda x: "Original" if x.startswith("orig") else "Scaled"
        )
        range_df["range_type"] = range_df["range_type"].apply(
            lambda x: "Full Range" if x.endswith("range") else "P5-P95 Range"
        )
        
        # Create grouped bar chart
        sns.barplot(x="feature", y="range", hue="dataset", data=range_df, ax=ax1, 
                   palette={"Original": "skyblue", "Scaled": "coral"})
        
        # Add styling
        ax1.set_title("Feature Ranges Before and After Scaling", fontsize=14)
        ax1.set_xlabel("Feature", fontsize=12)
        ax1.set_ylabel("Range", fontsize=12)
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right")
        ax1.legend(title="Dataset")
        
        # Plot 2: Compression factors
        compression_df = pd.melt(df, id_vars=["feature"], 
                                value_vars=["compression_factor", "p5_p95_compression", "pct_range_used"],
                                var_name="metric", value_name="value")
        
        # Map metric names to display names
        metric_mapping = {
            "compression_factor": "Full Range Compression",
            "p5_p95_compression": "P5-P95 Compression",
            "pct_range_used": "% Range Used"
        }
        compression_df["metric"] = compression_df["metric"].map(metric_mapping)
        
        # Use separate axes for compression and percentage
        ax2a = ax2
        ax2b = ax2.twinx()
        
        # Plot compression factors on primary y-axis
        compression_factors = compression_df[compression_df["metric"].isin(
            ["Full Range Compression", "P5-P95 Compression"]
        )]
        sns.barplot(x="feature", y="value", hue="metric", data=compression_factors, ax=ax2a,
                   palette={"Full Range Compression": "darkblue", "P5-P95 Compression": "royalblue"})
        
        # Plot percentage on secondary y-axis
        pct_range = compression_df[compression_df["metric"] == "% Range Used"]
        sns.lineplot(x="feature", y="value", data=pct_range, ax=ax2b, 
                   marker="o", color="red", label="% Range Used")
        
        # Add styling
        ax2a.set_title("Feature Compression Metrics", fontsize=14)
        ax2a.set_xlabel("Feature", fontsize=12)
        ax2a.set_ylabel("Compression Factor", fontsize=12)
        ax2b.set_ylabel("Percentage", fontsize=12)
        ax2a.set_xticklabels(ax2a.get_xticklabels(), rotation=45, ha="right")
        
        # Combine legends
        h1, l1 = ax2a.get_legend_handles_labels()
        h2, l2 = ax2b.get_legend_handles_labels()
        ax2a.legend(h1 + h2, l1 + l2, loc="upper right")
        
        # Remove redundant legend
        ax2b.get_legend().remove()
        
        # Add overall title
        plt.suptitle(f"Feature Compression Analysis for {method_name.replace('_', ' ').title()}", 
                    fontsize=16, y=1.02)
        plt.tight_layout()
        
        # Save figure
        output_file = self.output_dir / f"compression_{method_name}.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        
        # Save metrics to CSV for further analysis
        metrics_file = self.output_dir / f"compression_metrics_{method_name}.csv"
        df.to_csv(metrics_file, index=False)
        
        logger.info(f"Compression visualization saved to {output_file}")
        logger.info(f"Compression metrics saved to {metrics_file}")
        
        return str(output_file)
    
    def visualize_separation(self, X_original, X_scaled, feature_names, y_true, method_name="robust_scaling"):
        """
        Visualize separation power of features before and after scaling.
        
        Args:
            X_original: Original feature values
            X_scaled: Scaled feature values
            feature_names: Names of features
            y_true: True labels
            method_name: Name of scaling method for file naming
            
        Returns:
            Path to saved visualization
        """
        # Determine number of features and create figure layout
        n_features = len(feature_names)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        # Create figure with subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Calculate separation metrics
        separation_data = []
        
        # Create separation plots for each feature
        for i, (feature_idx, ax) in enumerate(zip(range(n_features), axes)):
            feature_name = feature_names[feature_idx]
            
            # Get original and scaled values
            orig_vals = X_original[:, feature_idx]
            scaled_vals = X_scaled[:, feature_idx]
            
            # Split by match status
            match_mask = y_true == 1
            non_match_mask = y_true == 0
            
            orig_match = orig_vals[match_mask]
            orig_non_match = orig_vals[non_match_mask]
            scaled_match = scaled_vals[match_mask]
            scaled_non_match = scaled_vals[non_match_mask]
            
            # Create plot for original values (top subplot)
            sns.kdeplot(orig_match, ax=ax, color="green", label="Match", fill=True, alpha=0.3)
            sns.kdeplot(orig_non_match, ax=ax, color="red", label="Non-match", fill=True, alpha=0.3)
            
            # Create second plot for scaled values (overlay with dashed lines)
            sns.kdeplot(scaled_match, ax=ax, color="green", label="Match (scaled)", linestyle="--")
            sns.kdeplot(scaled_non_match, ax=ax, color="red", label="Non-match (scaled)", linestyle="--")
            
            # Calculate separation metrics
            try:
                from sklearn.metrics import roc_auc_score
                orig_auc = roc_auc_score(y_true, orig_vals)
                scaled_auc = roc_auc_score(y_true, scaled_vals)
            except:
                orig_auc = 0.5
                scaled_auc = 0.5
            
            # Calculate effect sizes
            if len(orig_match) > 0 and len(orig_non_match) > 0:
                # Cohen's d for original
                pooled_std_orig = np.sqrt((np.var(orig_match) + np.var(orig_non_match)) / 2)
                orig_effect_size = (np.mean(orig_match) - np.mean(orig_non_match)) / pooled_std_orig if pooled_std_orig > 0 else 0
                
                # Cohen's d for scaled
                pooled_std_scaled = np.sqrt((np.var(scaled_match) + np.var(scaled_non_match)) / 2)
                scaled_effect_size = (np.mean(scaled_match) - np.mean(scaled_non_match)) / pooled_std_scaled if pooled_std_scaled > 0 else 0
            else:
                orig_effect_size = 0
                scaled_effect_size = 0
            
            # Add separation metrics to plot
            orig_stats = f"Original: AUC={orig_auc:.3f}, Effect Size={orig_effect_size:.3f}"
            scaled_stats = f"Scaled: AUC={scaled_auc:.3f}, Effect Size={scaled_effect_size:.3f}"
            
            # Add styling
            ax.set_title(feature_name)
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
            ax.text(0.05, 0.95, orig_stats, transform=ax.transAxes, va="top", fontsize=10)
            ax.text(0.05, 0.90, scaled_stats, transform=ax.transAxes, va="top", fontsize=10)
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())
            
            # Save separation metrics
            separation_data.append({
                "feature": feature_name,
                "orig_match_mean": np.mean(orig_match) if len(orig_match) > 0 else np.nan,
                "orig_non_match_mean": np.mean(orig_non_match) if len(orig_non_match) > 0 else np.nan,
                "orig_effect_size": orig_effect_size,
                "orig_auc": orig_auc,
                "scaled_match_mean": np.mean(scaled_match) if len(scaled_match) > 0 else np.nan,
                "scaled_non_match_mean": np.mean(scaled_non_match) if len(scaled_non_match) > 0 else np.nan,
                "scaled_effect_size": scaled_effect_size,
                "scaled_auc": scaled_auc,
                "effect_size_change": scaled_effect_size - orig_effect_size,
                "auc_change": scaled_auc - orig_auc
            })
        
        # Hide empty subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        # Add overall title
        plt.suptitle(f"Feature Separation Power Before and After {method_name.replace('_', ' ').title()}", 
                    fontsize=16, y=1.02)
        plt.tight_layout()
        
        # Save figure
        output_file = self.output_dir / f"separation_{method_name}.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        
        # Save metrics to CSV for further analysis
        separation_df = pd.DataFrame(separation_data)
        metrics_file = self.output_dir / f"separation_metrics_{method_name}.csv"
        separation_df.to_csv(metrics_file, index=False)
        
        logger.info(f"Separation visualization saved to {output_file}")
        logger.info(f"Separation metrics saved to {metrics_file}")
        
        return str(output_file)
    
    def visualize_roc_curves(self, X_original, X_scaled, feature_names, y_true, method_name="robust_scaling"):
        """
        Visualize ROC curves for key features before and after scaling.
        
        Args:
            X_original: Original feature values
            X_scaled: Scaled feature values
            feature_names: Names of features
            y_true: True labels
            method_name: Name of scaling method for file naming
            
        Returns:
            Path to saved visualization
        """
        from sklearn.metrics import roc_curve, auc
        
        # Select top features (to avoid overcrowding)
        max_features = 5
        
        # Calculate AUC for each feature
        feature_aucs = []
        for i, name in enumerate(feature_names):
            try:
                auc_val = auc(y_true, X_scaled[:, i])
                feature_aucs.append((name, i, auc_val))
            except:
                feature_aucs.append((name, i, 0.0))
        
        # Sort by AUC and select top features
        feature_aucs.sort(key=lambda x: x[2], reverse=True)
        top_features = feature_aucs[:max_features]
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot ROC curves for top features
        for name, idx, _ in top_features:
            # Original values
            fpr, tpr, _ = roc_curve(y_true, X_original[:, idx])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, linestyle=":", lw=2, 
                    label=f"{name} Original (AUC = {roc_auc:.3f})")
            
            # Scaled values
            fpr, tpr, _ = roc_curve(y_true, X_scaled[:, idx])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, 
                    label=f"{name} Scaled (AUC = {roc_auc:.3f})")
        
        # Add diagonal reference line
        plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
        
        # Add styling
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curves Before and After {method_name.replace('_', ' ').title()}")
        plt.legend(loc="lower right")
        
        # Save figure
        output_file = self.output_dir / f"roc_curves_{method_name}.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"ROC curves visualization saved to {output_file}")
        return str(output_file)
    
    def visualize_correlation(self, X_scaled, feature_names, method_name="robust_scaling"):
        """
        Visualize correlation matrix of scaled features.
        
        Args:
            X_scaled: Scaled feature values
            feature_names: Names of features
            method_name: Name of scaling method for file naming
            
        Returns:
            Path to saved visualization
        """
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(X_scaled, rowvar=False)
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
                   xticklabels=feature_names, yticklabels=feature_names)
        
        # Add styling
        plt.title(f"Feature Correlation Matrix After {method_name.replace('_', ' ').title()}")
        plt.tight_layout()
        
        # Save figure
        output_file = self.output_dir / f"correlation_{method_name}.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Correlation matrix visualization saved to {output_file}")
        return str(output_file)
    
    def create_scaling_report(self, visualization_files, method_name="robust_scaling"):
        """
        Create HTML report with all visualizations.
        
        Args:
            visualization_files: Dictionary mapping visualization types to file paths
            method_name: Name of scaling method for file naming
            
        Returns:
            Path to saved report
        """
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Feature Scaling Analysis Report - {method_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                .section {{ margin-bottom: 30px; }}
                .image-container {{ margin: 20px 0; }}
                img {{ max-width: 100%; border: 1px solid #ddd; }}
                .summary {{ background-color: #f8f9fa; padding: 15px; border-left: 4px solid #4e73df; }}
            </style>
        </head>
        <body>
            <h1>Feature Scaling Analysis Report</h1>
            <div class="summary">
                <h3>Scaling Method: {method_name.replace('_', ' ').title()}</h3>
                <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Feature Distributions</h2>
                <p>This visualization shows the distribution of feature values before and after scaling.</p>
                <div class="image-container">
                    <img src="{visualization_files.get('distributions', '')}" alt="Feature Distributions">
                </div>
            </div>
            
            <div class="section">
                <h2>Feature Compression Analysis</h2>
                <p>This visualization shows how scaling affects the range of feature values and quantifies compression.</p>
                <div class="image-container">
                    <img src="{visualization_files.get('compression', '')}" alt="Feature Compression">
                </div>
            </div>
        """
        
        # Add separation visualization if available
        if 'separation' in visualization_files:
            html_content += f"""
            <div class="section">
                <h2>Feature Separation Power</h2>
                <p>This visualization shows how well features separate matches from non-matches before and after scaling.</p>
                <div class="image-container">
                    <img src="{visualization_files.get('separation', '')}" alt="Feature Separation">
                </div>
            </div>
            """
        
        # Add ROC curves if available
        if 'roc_curves' in visualization_files:
            html_content += f"""
            <div class="section">
                <h2>ROC Curves</h2>
                <p>This visualization shows ROC curves for top features before and after scaling.</p>
                <div class="image-container">
                    <img src="{visualization_files.get('roc_curves', '')}" alt="ROC Curves">
                </div>
            </div>
            """
        
        # Add correlation matrix
        html_content += f"""
            <div class="section">
                <h2>Feature Correlation Matrix</h2>
                <p>This visualization shows the correlation between scaled features.</p>
                <div class="image-container">
                    <img src="{visualization_files.get('correlation', '')}" alt="Feature Correlation">
                </div>
            </div>
            
            <footer>
                <p>Entity Resolution Pipeline - Library Catalog Scaling Analysis</p>
            </footer>
        </body>
        </html>
        """
        
        # Save HTML report
        output_file = self.output_dir / f"scaling_report_{method_name}.html"
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Scaling report saved to {output_file}")
        return str(output_file)
    
    def compare_scaling_methods(self, X_original, methods_results, feature_names, y_true=None):
        """
        Compare multiple scaling methods visually.
        
        Args:
            X_original: Original feature values
            methods_results: Dictionary mapping method names to scaled features
            feature_names: Names of features
            y_true: True labels (optional)
            
        Returns:
            Path to saved comparison report
        """
        # Generate visualizations for each method
        method_viz_files = {}
        
        for method_name, X_scaled in methods_results.items():
            viz_files = self.visualize_scaling_comparison(
                X_original, X_scaled, feature_names, y_true, method_name
            )
            method_viz_files[method_name] = viz_files
        
        # Create comparison metrics
        comparison_metrics = self._calculate_comparison_metrics(
            X_original, methods_results, feature_names, y_true
        )
        
        # Create comparison visualizations
        comparison_files = self._create_method_comparison_visualizations(
            X_original, methods_results, feature_names, y_true
        )
        
        # Save comparison metrics
        metrics_file = self.output_dir / "scaling_methods_comparison.json"
        with open(metrics_file, 'w') as f:
            json.dump(comparison_metrics, f, indent=2)
        
        # Create comparison report
        report_file = self._create_comparison_report(method_viz_files, comparison_files)
        
        logger.info(f"Scaling methods comparison saved to {report_file}")
        return str(report_file)
    
    def _calculate_comparison_metrics(self, X_original, methods_results, feature_names, y_true=None):
        """
        Calculate comparative metrics for different scaling methods.
        
        Args:
            X_original: Original feature values
            methods_results: Dictionary mapping method names to scaled features
            feature_names: Names of features
            y_true: True labels (optional)
            
        Returns:
            Dictionary of comparison metrics
        """
        comparison_metrics = {}
        
        # Calculate information retention for each method
        for method_name, X_scaled in methods_results.items():
            method_metrics = {}
            
            # Calculate range metrics
            range_metrics = []
            for i, name in enumerate(feature_names):
                orig_vals = X_original[:, i]
                scaled_vals = X_scaled[:, i]
                
                # Calculate original and scaled ranges
                orig_range = np.max(orig_vals) - np.min(orig_vals)
                scaled_range = np.max(scaled_vals) - np.min(scaled_vals)
                
                # Calculate compression factor
                compression_factor = orig_range / scaled_range if scaled_range > 0 else float('inf')
                
                range_metrics.append({
                    "feature": name,
                    "orig_range": float(orig_range),
                    "scaled_range": float(scaled_range),
                    "compression_factor": float(compression_factor)
                })
            
            method_metrics["range"] = range_metrics
            
            # Calculate separation metrics if labels provided
            if y_true is not None:
                separation_metrics = []
                for i, name in enumerate(feature_names):
                    try:
                        from sklearn.metrics import roc_auc_score
                        auc = float(roc_auc_score(y_true, X_scaled[:, i]))
                    except:
                        auc = 0.5
                    
                    # Calculate means for matches and non-matches
                    match_mean = float(np.mean(X_scaled[y_true == 1, i])) if np.any(y_true == 1) else 0.0
                    non_match_mean = float(np.mean(X_scaled[y_true == 0, i])) if np.any(y_true == 0) else 0.0
                    
                    separation_metrics.append({
                        "feature": name,
                        "auc": auc,
                        "match_mean": match_mean,
                        "non_match_mean": non_match_mean,
                        "mean_difference": match_mean - non_match_mean
                    })
                
                method_metrics["separation"] = separation_metrics
            
            comparison_metrics[method_name] = method_metrics
        
        return comparison_metrics
    
    def _create_method_comparison_visualizations(self, X_original, methods_results, feature_names, y_true=None):
        """
        Create visualizations comparing different scaling methods.
        
        Args:
            X_original: Original feature values
            methods_results: Dictionary mapping method names to scaled features
            feature_names: Names of features
            y_true: True labels (optional)
            
        Returns:
            Dictionary mapping visualization types to file paths
        """
        comparison_files = {}
        
        # Create AUC comparison visualization if labels provided
        if y_true is not None:
            comparison_files["auc_comparison"] = self._visualize_auc_comparison(
                methods_results, feature_names, y_true
            )
        
        # Create range utilization comparison
        comparison_files["range_comparison"] = self._visualize_range_comparison(
            X_original, methods_results, feature_names
        )
        
        return comparison_files
    
    def _visualize_auc_comparison(self, methods_results, feature_names, y_true):
        """
        Visualize AUC comparison across scaling methods.
        
        Args:
            methods_results: Dictionary mapping method names to scaled features
            feature_names: Names of features
            y_true: True labels
            
        Returns:
            Path to saved visualization
        """
        from sklearn.metrics import roc_auc_score
        
        # Calculate AUC for each feature and method
        auc_data = []
        
        for method_name, X_scaled in methods_results.items():
            for i, name in enumerate(feature_names):
                try:
                    auc = float(roc_auc_score(y_true, X_scaled[:, i]))
                except:
                    auc = 0.5
                
                auc_data.append({
                    "method": method_name,
                    "feature": name,
                    "auc": auc
                })
        
        # Create DataFrame
        auc_df = pd.DataFrame(auc_data)
        
        # Create comparison visualization
        plt.figure(figsize=(12, 8))
        
        # Create grouped bar chart
        ax = sns.barplot(x="feature", y="auc", hue="method", data=auc_df)
        
        # Add styling
        plt.title("AUC Comparison Across Scaling Methods")
        plt.xlabel("Feature")
        plt.ylabel("AUC")
        plt.xticks(rotation=45, ha="right")
        plt.legend(title="Method")
        plt.tight_layout()
        
        # Save figure
        output_file = self.output_dir / "auc_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"AUC comparison visualization saved to {output_file}")
        return str(output_file)
    
    def _visualize_range_comparison(self, X_original, methods_results, feature_names):
        """
        Visualize range utilization comparison across scaling methods.
        
        Args:
            X_original: Original feature values
            methods_results: Dictionary mapping method names to scaled features
            feature_names: Names of features
            
        Returns:
            Path to saved visualization
        """
        # Calculate range metrics
        range_data = []
        
        for method_name, X_scaled in methods_results.items():
            for i, name in enumerate(feature_names):
                orig_vals = X_original[:, i]
                scaled_vals = X_scaled[:, i]
                
                # Calculate original and scaled ranges
                orig_range = np.max(orig_vals) - np.min(orig_vals)
                scaled_range = np.max(scaled_vals) - np.min(scaled_vals)
                
                # Calculate percentage of range utilized
                pct_range = (scaled_range / 1.0) * 100  # Assuming scaled to [0,1]
                
                # Calculate compression factor
                compression_factor = orig_range / scaled_range if scaled_range > 0 else float('inf')
                
                range_data.append({
                    "method": method_name,
                    "feature": name,
                    "orig_range": orig_range,
                    "scaled_range": scaled_range,
                    "pct_range": pct_range,
                    "compression_factor": compression_factor
                })
        
        # Create DataFrame
        range_df = pd.DataFrame(range_data)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Range utilization
        sns.barplot(x="feature", y="pct_range", hue="method", data=range_df, ax=ax1)
        
        # Add styling
        ax1.set_title("Percentage of Potential Range Utilized")
        ax1.set_xlabel("Feature")
        ax1.set_ylabel("% of [0,1] Range Used")
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right")
        ax1.legend(title="Method")
        
        # Plot 2: Compression factor
        # Cap compression factor for visualization
        range_df["compression_factor_capped"] = range_df["compression_factor"].apply(
            lambda x: min(x, 10.0)  # Cap at 10 for better visualization
        )
        
        sns.barplot(x="feature", y="compression_factor_capped", hue="method", data=range_df, ax=ax2)
        
        # Add styling
        ax2.set_title("Compression Factor Comparison (capped at 10)")
        ax2.set_xlabel("Feature")
        ax2.set_ylabel("Compression Factor")
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")
        ax2.legend(title="Method")
        
        # Add overall title
        plt.suptitle("Feature Range Comparison Across Scaling Methods", fontsize=16, y=1.02)
        plt.tight_layout()
        
        # Save figure
        output_file = self.output_dir / "range_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Range comparison visualization saved to {output_file}")
        return str(output_file)
    
    def _create_comparison_report(self, method_viz_files, comparison_files):
        """
        Create HTML report comparing scaling methods.
        
        Args:
            method_viz_files: Dictionary mapping method names to visualization files
            comparison_files: Dictionary of comparison visualization files
            
        Returns:
            Path to saved report
        """
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Scaling Methods Comparison Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                .section {{ margin-bottom: 30px; }}
                .image-container {{ margin: 20px 0; }}
                img {{ max-width: 100%; border: 1px solid #ddd; }}
                .summary {{ background-color: #f8f9fa; padding: 15px; border-left: 4px solid #4e73df; }}
                .method-links {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 20px 0; }}
                .method-links a {{ 
                    display: inline-block; 
                    padding: 8px 15px; 
                    background-color: #4e73df; 
                    color: white; 
                    text-decoration: none; 
                    border-radius: 4px; 
                }}
                .method-links a:hover {{ background-color: #2e59d9; }}
            </style>
        </head>
        <body>
            <h1>Scaling Methods Comparison Report</h1>
            <div class="summary">
                <h3>Methods Compared: {", ".join(method_viz_files.keys())}</h3>
                <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Method-Specific Reports</h2>
                <p>Click on a method to view its detailed report:</p>
                <div class="method-links">
        """
        
        # Add links to method-specific reports
        for method_name, viz_files in method_viz_files.items():
            if 'report' in viz_files:
                html_content += f"""
                    <a href="{viz_files['report']}" target="_blank">{method_name.replace('_', ' ').title()}</a>
                """
        
        html_content += """
                </div>
            </div>
            
            <div class="section">
                <h2>Direct Method Comparison</h2>
        """
        
        # Add comparison visualizations
        for viz_type, file_path in comparison_files.items():
            viz_title = viz_type.replace('_', ' ').title()
            html_content += f"""
                <h3>{viz_title}</h3>
                <div class="image-container">
                    <img src="{file_path}" alt="{viz_title}">
                </div>
            """
        
        html_content += """
            </div>
            
            <footer>
                <p>Entity Resolution Pipeline - Scaling Methods Comparison</p>
            </footer>
        </body>
        </html>
        """
        
        # Save HTML report
        output_file = self.output_dir / "scaling_methods_comparison.html"
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Scaling methods comparison report saved to {output_file}")
        return str(output_file)