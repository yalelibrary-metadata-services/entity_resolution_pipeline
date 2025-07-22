"""
Fixed Feature Visualization Module for Entity Resolution

This module provides visualization functions for feature distributions and class separation
to help analyze and understand the feature engineering process with correct color mapping.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

logger = logging.getLogger(__name__)

def plot_feature_distributions(X: np.ndarray, y: np.ndarray, feature_names: List[str], 
                              output_dir: str, n_features_per_row: int = 2) -> None:
    """
    Plot the distribution of each feature split by class (match/non-match).
    Colors are intelligently assigned based on feature behavior:
    - Higher values (matches) = GREEN
    - Lower values (non-matches) = RED
    
    Args:
        X: Feature matrix
        y: Binary labels
        feature_names: Names of features
        output_dir: Directory to save plots
        n_features_per_row: Number of features to plot per row
    """
    n_features = X.shape[1]
    if n_features == 0 or len(feature_names) == 0:
        logger.warning("No features to plot distributions for")
        return
        
    # Create plots directory
    plots_dir = os.path.join(output_dir, "plots", "feature_distributions")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Calculate number of rows and columns for subplots
    n_rows = (n_features + n_features_per_row - 1) // n_features_per_row
    
    # Create a combined plot for all features
    plt.figure(figsize=(n_features_per_row * 6, n_rows * 5))
    
    for i, feature_name in enumerate(feature_names):
        if i >= n_features:
            break
            
        # Create subplot
        plt.subplot(n_rows, n_features_per_row, i + 1)
        
        # Get feature values
        feature_values = X[:, i]
        
        # Convert to DataFrame for easier plotting with seaborn
        df = pd.DataFrame({
            'Feature Value': feature_values,
            'Class': y
        })
        
        # Plot distributions with correct color mapping
        # y=1 indicates a match (green), y=0 indicates a non-match (red)
        unique_classes = sorted(df['Class'].unique())
        if len(unique_classes) == 2:
            # Create color map based on actual class labels
            # Class 1 = Match = Green
            # Class 0 = Non-match = Red
            color_map = {1: 'green', 0: 'red'}
            
            # Create a copy of the dataframe with proper labels
            df_labeled = df.copy()
            df_labeled['Class'] = df_labeled['Class'].map({0: 'Non-match', 1: 'Match'})
            
            sns.histplot(data=df_labeled, x='Feature Value', hue='Class', multiple='dodge', 
                       stat='density', common_norm=False, 
                       palette={'Match': 'green', 'Non-match': 'red'})
        else:
            # Fallback for single class
            if unique_classes[0] == 1:
                sns.histplot(data=df, x='Feature Value', hue='Class', element='step', 
                           stat='density', common_norm=False, palette=['green'])
                legend_labels = ['Match']
            else:
                sns.histplot(data=df, x='Feature Value', hue='Class', element='step', 
                           stat='density', common_norm=False, palette=['red'])
                legend_labels = ['Non-match']
        
        # Add vertical line at decision threshold if available (use median for demonstration)
        if len(feature_values) > 0:
            plt.axvline(x=np.median(feature_values), color='k', linestyle='--', alpha=0.5)
        
        plt.title(f'Distribution of {feature_name}')
        plt.xlabel('Feature Value')
        plt.ylabel('Density')
        
        # Adjust layout
        plt.tight_layout()
    
    # Save combined plot
    plt.savefig(os.path.join(plots_dir, "all_feature_distributions.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also create individual plots for each feature
    for i, feature_name in enumerate(feature_names):
        if i >= n_features:
            break
            
        plt.figure(figsize=(10, 6))
        
        # Get feature values
        feature_values = X[:, i]
        
        # Convert to DataFrame for easier plotting with seaborn
        df = pd.DataFrame({
            'Feature Value': feature_values,
            'Class': y
        })
        
        # Plot distributions with correct color mapping
        # y=1 indicates a match (green), y=0 indicates a non-match (red)
        unique_classes = sorted(df['Class'].unique())
        if len(unique_classes) == 2:
            # Create a copy of the dataframe with proper labels
            df_labeled = df.copy()
            df_labeled['Class'] = df_labeled['Class'].map({0: 'Non-match', 1: 'Match'})
            
            sns.histplot(data=df_labeled, x='Feature Value', hue='Class', element='step', 
                       stat='density', common_norm=False, 
                       palette={'Match': 'green', 'Non-match': 'red'})
        else:
            # Fallback for single class
            df_labeled = df.copy()
            if unique_classes[0] == 1:
                df_labeled['Class'] = 'Match'
                sns.histplot(data=df_labeled, x='Feature Value', hue='Class', element='step', 
                           stat='density', common_norm=False, palette={'Match': 'green'})
            else:
                df_labeled['Class'] = 'Non-match'
                sns.histplot(data=df_labeled, x='Feature Value', hue='Class', element='step', 
                           stat='density', common_norm=False, palette={'Non-match': 'red'})
        
        # Add vertical line at decision threshold if available (use median for demonstration)
        if len(feature_values) > 0:
            plt.axvline(x=np.median(feature_values), color='k', linestyle='--', alpha=0.5)
        
        plt.title(f'Distribution of {feature_name}')
        plt.xlabel('Feature Value')
        plt.ylabel('Density')
        
        # Save individual plot
        plt.savefig(os.path.join(plots_dir, f"distribution_{feature_name}.png"), 
                  dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info(f"Generated feature distribution plots in {plots_dir}")

# Rest of the functions remain the same...
def plot_class_separation(X: np.ndarray, y: np.ndarray, feature_names: List[str], 
                         output_dir: str, n_features_per_row: int = 2) -> Dict[str, float]:
    """
    Plot class separation metrics (ROC, PR curves) for each feature.
    
    Args:
        X: Feature matrix
        y: Binary labels
        feature_names: Names of features
        output_dir: Directory to save plots
        n_features_per_row: Number of features to plot per row
        
    Returns:
        Dictionary mapping feature names to AUC scores
    """
    n_features = X.shape[1]
    if n_features == 0 or len(feature_names) == 0:
        logger.warning("No features to plot class separation for")
        return {}
        
    # Create plots directory
    plots_dir = os.path.join(output_dir, "plots", "class_separation")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Calculate number of rows for subplots
    n_rows = (n_features + n_features_per_row - 1) // n_features_per_row
    
    # Store AUC scores for each feature
    feature_auc_scores = {}
    
    # Create combined ROC plot
    plt.figure(figsize=(n_features_per_row * 6, n_rows * 5))
    
    for i, feature_name in enumerate(feature_names):
        if i >= n_features:
            break
            
        # Create subplot
        plt.subplot(n_rows, n_features_per_row, i + 1)
        
        # Get feature values
        feature_values = X[:, i]
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y, feature_values)
        roc_auc = auc(fpr, tpr)
        feature_auc_scores[feature_name] = roc_auc
        
        # Plot ROC curve
        plt.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC for {feature_name}')
        plt.legend(loc="lower right")
        
    # Adjust layout
    plt.tight_layout()
    
    # Save combined ROC plot
    plt.savefig(os.path.join(plots_dir, "all_feature_roc_curves.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create combined precision-recall plot
    plt.figure(figsize=(n_features_per_row * 6, n_rows * 5))
    
    for i, feature_name in enumerate(feature_names):
        if i >= n_features:
            break
            
        # Create subplot
        plt.subplot(n_rows, n_features_per_row, i + 1)
        
        # Get feature values
        feature_values = X[:, i]
        
        # Calculate precision-recall curve
        precision, recall, _ = precision_recall_curve(y, feature_values)
        avg_precision = average_precision_score(y, feature_values)
        
        # Plot precision-recall curve
        plt.plot(recall, precision, lw=2, label=f'AP = {avg_precision:.2f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall for {feature_name}')
        plt.legend(loc="lower left")
        
    # Adjust layout
    plt.tight_layout()
    
    # Save combined precision-recall plot
    plt.savefig(os.path.join(plots_dir, "all_feature_pr_curves.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also create individual plots for each feature
    for i, feature_name in enumerate(feature_names):
        if i >= n_features:
            break
            
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        
        # Get feature values
        feature_values = X[:, i]
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y, feature_values)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {feature_name}')
        plt.legend(loc="lower right")
        
        # Plot precision-recall curve
        plt.subplot(1, 2, 2)
        precision, recall, _ = precision_recall_curve(y, feature_values)
        avg_precision = average_precision_score(y, feature_values)
        
        plt.plot(recall, precision, lw=2, label=f'AP = {avg_precision:.2f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {feature_name}')
        plt.legend(loc="lower left")
        
        plt.tight_layout()
        
        # Save individual plot
        plt.savefig(os.path.join(plots_dir, f"separation_{feature_name}.png"), 
                  dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create a feature comparison bar chart
    plt.figure(figsize=(12, 6))
    features = list(feature_auc_scores.keys())
    auc_scores = list(feature_auc_scores.values())
    
    # Sort by AUC score
    sorted_indices = np.argsort(auc_scores)
    sorted_features = [features[i] for i in sorted_indices]
    sorted_scores = [auc_scores[i] for i in sorted_indices]
    
    plt.barh(sorted_features, sorted_scores, color='skyblue')
    plt.xlabel('AUC Score')
    plt.title('Feature Comparison by AUC Score')
    plt.xlim([0, 1])
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    # Save feature comparison
    plt.savefig(os.path.join(plots_dir, "feature_auc_comparison.png"), 
              dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Generated class separation plots in {plots_dir}")
    return feature_auc_scores

def generate_feature_visualization_report(output_dir: str, feature_auc_scores: Dict[str, float],
                                         feature_weights: Dict[str, float]) -> str:
    """
    Generate a feature visualization summary report.
    
    Args:
        output_dir: Directory to save the report
        feature_auc_scores: Dictionary mapping feature names to AUC scores
        feature_weights: Dictionary mapping feature names to feature weights
        
    Returns:
        Path to the generated report
    """
    # Create reports directory
    reports_dir = os.path.join(output_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    # Create DataFrame with feature metrics
    data = []
    for feature_name in feature_weights.keys():
        auc_score = feature_auc_scores.get(feature_name, 0.0)
        weight = feature_weights.get(feature_name, 0.0)
        
        data.append({
            'Feature': feature_name,
            'AUC Score': auc_score,
            'Weight': weight,
            'Abs Weight': abs(weight)
        })
    
    df = pd.DataFrame(data)
    
    # Sort by absolute weight
    df = df.sort_values('Abs Weight', ascending=False)
    
    # Generate report
    report_path = os.path.join(reports_dir, "feature_visualization_report.html")
    
    # Create HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Feature Visualization Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; }}
            h1, h2, h3 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .visualization {{ margin: 30px 0; }}
            .visualization img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
        </style>
    </head>
    <body>
        <h1>Feature Visualization Report</h1>
        
        <h2>Feature Metrics</h2>
        <table>
            <tr>
                <th>Feature</th>
                <th>Model Weight</th>
                <th>AUC Score</th>
                <th>Absolute Weight</th>
            </tr>
    """
    
    # Add rows for each feature
    for _, row in df.iterrows():
        html_content += f"""
            <tr>
                <td>{row['Feature']}</td>
                <td>{row['Weight']:.4f}</td>
                <td>{row['AUC Score']:.4f}</td>
                <td>{row['Abs Weight']:.4f}</td>
            </tr>
        """
    
    html_content += """
        </table>
        
        <h2>Feature Distributions</h2>
        <p>These plots show how each feature is distributed across the match and non-match classes.</p>
        <div class="visualization">
            <img src="../plots/feature_distributions/all_feature_distributions.png" alt="Feature Distributions">
        </div>
        
        <h2>Class Separation</h2>
        <p>These plots show how well each feature separates the match and non-match classes.</p>
        <div class="visualization">
            <img src="../plots/class_separation/all_feature_roc_curves.png" alt="ROC Curves">
        </div>
        <div class="visualization">
            <img src="../plots/class_separation/all_feature_pr_curves.png" alt="Precision-Recall Curves">
        </div>
        
        <h2>Feature Importance Comparison</h2>
        <div class="visualization">
            <img src="../plots/class_separation/feature_auc_comparison.png" alt="Feature AUC Comparison">
        </div>
    </body>
    </html>
    """
    
    # Write to file
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Generated feature visualization report at {report_path}")
    return report_path