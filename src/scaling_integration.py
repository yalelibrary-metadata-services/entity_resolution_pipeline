"""
Integration module for enhanced feature scaling in entity resolution pipeline.

This module integrates the robust scaling approaches with the existing 
feature engineering and classification components of the entity resolution pipeline.
"""

import os
import yaml
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import matplotlib.pyplot as plt
import warnings

# Import custom components
from src.robust_scaler import RobustMinMaxScaler, LibraryCatalogScaler
from src.scaling_visualizer import ScalingVisualizer

logger = logging.getLogger(__name__)

class ScalingIntegration:
    """
    Integrates enhanced scaling with the entity resolution pipeline.
    
    Provides methods for comparing scaling approaches, evaluating their
    effectiveness, and integrating the selected approach into the pipeline.
    """
    
    def __init__(self, config_path="scaling_config.yml"):
        """
        Initialize the scaling integration module.
        
        Args:
            config_path: Path to scaling configuration file
        """
        # Load scaling configuration
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize scaling components
        self.scalers = self._initialize_scalers()
        self.evaluator = ScalingEvaluator(self.config)
        
        # Initialize visualizer if enabled
        if self.config.get('visualization', {}).get('enabled', True):
            output_dir = self.config.get('visualization', {}).get('output_dir', 'scaling_viz')
            self.visualizer = ScalingVisualizer(output_dir)
        else:
            self.visualizer = None
        
        # Track selected scaler
        self.selected_scaler = None
        self.scaling_results = {}
        
        logger.info(f"ScalingIntegration initialized with config from {config_path}")
    
    def _load_config(self):
        """
        Load scaling configuration from file.
        
        Returns:
            Configuration dictionary
        """
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.warning(f"Error loading scaling config from {self.config_path}: {e}")
            logger.warning("Using default scaling configuration")
            return {
                'scaling_approaches': {
                    'robust_minmax': True
                },
                'scaling_default': {
                    'feature_range': [0, 1],
                    'clip_percentile': 95
                }
            }
    
    def _initialize_scalers(self):
        """
        Initialize scaling approaches based on configuration.
        
        Returns:
            Dictionary of initialized scalers
        """
        from sklearn.preprocessing import MinMaxScaler, StandardScaler
        
        # Get enabled approaches
        approaches = self.config.get('scaling_approaches', {})
        default_settings = self.config.get('scaling_default', {})
        
        # Initialize enabled scalers
        scalers = {}
        
        # Always include LibraryCatalogScaler as the primary scaling approach
        scalers['library_catalog'] = LibraryCatalogScaler(self.config)
        
        # Include other scalers for backward compatibility or comparative analysis
        if approaches.get('robust_minmax', False):
            # Robust min-max with outlier handling
            logger.info("Initializing RobustMinMaxScaler (legacy support)")
            scalers['robust_minmax'] = RobustMinMaxScaler(
                feature_range=default_settings.get('feature_range', (0, 1)),
                clip_percentile=default_settings.get('clip_percentile', 95),
                clip_lower=default_settings.get('clip_lower', False)
            )
        
        # These scaling methods are deprecated but included for comparative analysis
        if approaches.get('standard_minmax', False):
            # Standard min-max scaling 
            logger.info("Initializing standard MinMaxScaler (for comparison only)")
            scalers['standard_minmax'] = MinMaxScaler(
                feature_range=default_settings.get('feature_range', (0, 1))
            )
        
        if approaches.get('standard_scaler', False):
            # StandardScaler for comparison
            logger.info("Initializing StandardScaler (for comparison only)")
            scalers['standard_scaler'] = StandardScaler()
        
        logger.info(f"Initialized {len(scalers)} scaling approaches: {', '.join(scalers.keys())}")
        return scalers
    
    def compare_scaling_approaches(self, X, feature_names, y=None):
        """
        Compare different scaling approaches on training data.
        
        Args:
            X: Feature matrix
            feature_names: Names of features
            y: Labels (optional)
            
        Returns:
            Dictionary of evaluation results
        """
        logger.info(f"Comparing {len(self.scalers)} scaling approaches on {X.shape[0]} samples")
        
        # Verify input data is suitable for scaling
        if not isinstance(X, np.ndarray):
            logger.warning(f"Input data is not a NumPy array. Converting to array.")
            X = np.asarray(X)
        
        # Check data types
        if not np.issubdtype(X.dtype, np.number):
            raise TypeError("Input data must contain only numeric values for scaling")
            
        # Check for and handle NaN or infinity values
        if np.isnan(X).any() or np.isinf(X).any():
            logger.warning("Input data contains NaN or infinite values. Cleaning data...")
            # Make a copy to avoid modifying the original
            X_clean = np.copy(X)
            
            # Replace NaN values with column means
            col_means = np.nanmean(X_clean, axis=0)
            for i in range(X_clean.shape[1]):
                mask = np.isnan(X_clean[:, i])
                X_clean[mask, i] = col_means[i]
            
            # Replace infinity values with large but finite values
            X_clean[np.isinf(X_clean) & (X_clean > 0)] = np.finfo(np.float64).max / 10
            X_clean[np.isinf(X_clean) & (X_clean < 0)] = np.finfo(np.float64).min / 10
            
            # Use cleaned data
            X = X_clean
            logger.info("Data cleaned: NaN and infinity values replaced")
        
        # Scale data with each approach
        scaled_data = {}
        for name, scaler in self.scalers.items():
            try:
                # Handle different scaler interfaces
                if name == 'library_catalog':
                    X_scaled = scaler.fit_transform(X, feature_names)
                else:
                    X_scaled = scaler.fit_transform(X)
                
                scaled_data[name] = X_scaled
                logger.info(f"Successfully scaled data with {name}")
            except Exception as e:
                logger.error(f"Error scaling data with {name}: {e}")
                import traceback
                logger.debug(f"Traceback: {traceback.format_exc()}")
                
                # Try with a more defensive approach if the original failed
                try:
                    # Skip problematic features and use sklearn's standard scaler as fallback
                    from sklearn.preprocessing import MinMaxScaler
                    logger.warning(f"Attempting fallback scaling for {name}")
                    
                    # Use standard scaler with fixed feature range
                    fallback_scaler = MinMaxScaler()
                    X_scaled = fallback_scaler.fit_transform(X)
                    
                    scaled_data[name] = X_scaled
                    logger.info(f"Successfully scaled data with fallback scaler for {name}")
                    
                except Exception as e2:
                    logger.error(f"Fallback scaling also failed for {name}: {e2}")
                    # Skip this scaler completely
        
        # Store scaled data
        self.scaling_results = scaled_data
        
        # Evaluate scaling approaches
        evaluation_results = {}
        for name, X_scaled in scaled_data.items():
            try:
                # Run evaluation
                results = self.evaluator.evaluate_scaling(X, X_scaled, feature_names, y)
                evaluation_results[name] = results
                logger.info(f"Successfully evaluated {name} scaling approach")
            except Exception as e:
                logger.error(f"Error evaluating {name} scaling approach: {e}")
        
        # Visualize comparison if enabled
        if self.visualizer:
            try:
                comparison_report = self.visualizer.compare_scaling_methods(
                    X, scaled_data, feature_names, y
                )
                logger.info(f"Scaling comparison report saved to {comparison_report}")
            except Exception as e:
                logger.error(f"Error creating comparison report: {e}")
        
        return evaluation_results
    
    def select_best_approach(self, evaluation_results=None):
        """
        Select the best scaling approach based on evaluation.
        
        Args:
            evaluation_results: Evaluation results from compare_scaling_approaches()
            
        Returns:
            Name of selected approach
        """
        # Use evaluation results if provided, otherwise use library_catalog as default
        if evaluation_results is None:
            # Always default to library_catalog when no evaluation results available
            if 'library_catalog' in self.scalers:
                self.selected_scaler = 'library_catalog'
            # Fallback to other options only if library_catalog is not available
            elif 'robust_minmax' in self.scalers:
                self.selected_scaler = 'robust_minmax'
                logger.warning("Falling back to robust_minmax because library_catalog not available")
            elif 'standard_minmax' in self.scalers:
                self.selected_scaler = 'standard_minmax'
                logger.warning("Falling back to standard_minmax because preferred scalers not available")
            else:
                # Use the first available scaler as last resort
                if self.scalers:
                    self.selected_scaler = next(iter(self.scalers.keys()))
                else:
                    logger.error("No scalers available for selection")
                    return None
                    
            logger.info(f"Selected default scaling approach: {self.selected_scaler}")
            return self.selected_scaler
        
        # Calculate overall score for each approach
        approach_scores = {}
        
        for name, results in evaluation_results.items():
            score = 0.0
            
            # Score based on separation (if available)
            if 'separation' in results:
                separation = results['separation']
                
                # Check if we have the necessary metrics before using them
                if separation and isinstance(separation, dict):
                    # Calculate average AUC and effect size if available
                    auc_values = [feat.get('auc', 0) for feat_name, feat in separation.items() 
                                  if isinstance(feat, dict) and 'auc' in feat]
                    
                    effect_values = [abs(feat.get('effect_size', 0)) for feat_name, feat in separation.items() 
                                     if isinstance(feat, dict) and 'effect_size' in feat]
                    
                    if auc_values:
                        avg_auc = np.mean(auc_values)
                        # Add to score (higher is better)
                        score += avg_auc * 2.0  # Weight AUC more heavily
                    
                    if effect_values:
                        avg_effect = np.mean(effect_values)
                        # Add to score
                        score += avg_effect * 1.0
            
            # Score based on information retention
            if 'info_retention' in results:
                retention = results['info_retention']
                
                # Check if we have the necessary metrics
                if retention and isinstance(retention, dict):
                    # Calculate average rank correlation if available
                    corr_values = [feat.get('rank_correlation', 0) for feat_name, feat in retention.items() 
                                  if isinstance(feat, dict) and 'rank_correlation' in feat]
                    
                    if corr_values:
                        avg_corr = np.mean(corr_values)
                        # Add to score (higher is better)
                        score += avg_corr * 1.5
            
            # Score based on distribution
            if 'distribution' in results:
                distributions = results['distribution']
                
                # Check if we have the necessary metrics
                if distributions and isinstance(distributions, dict):
                    # Collect skew values where available
                    skew_values = []
                    for feat_name, feat in distributions.items():
                        if isinstance(feat, dict) and 'scaled' in feat and isinstance(feat['scaled'], dict) and 'skew' in feat['scaled']:
                            skew_values.append(abs(feat['scaled']['skew']))
                    
                    if skew_values:
                        # Penalize extreme skew in scaled values
                        avg_skew = np.mean(skew_values)
                        # Subtract from score (lower skew is better)
                        score -= avg_skew * 0.5
            
            approach_scores[name] = score
        
        # Select approach with highest score
        if approach_scores:
            self.selected_scaler = max(approach_scores.items(), key=lambda x: x[1])[0]
            logger.info(f"Selected best scaling approach: {self.selected_scaler} (score: {approach_scores[self.selected_scaler]:.4f})")
        else:
            # Fall back to library_catalog if no scores
            if 'library_catalog' in self.scalers:
                self.selected_scaler = 'library_catalog'
            # Fallback to other options only if library_catalog is not available
            elif 'robust_minmax' in self.scalers:
                self.selected_scaler = 'robust_minmax'
                logger.warning("Falling back to robust_minmax because library_catalog not available")
            elif 'standard_minmax' in self.scalers:
                self.selected_scaler = 'standard_minmax'
                logger.warning("Falling back to standard_minmax because preferred scalers not available")
            # Use the first available scaler as last resort if none of the above are available
            elif self.scalers:
                self.selected_scaler = next(iter(self.scalers.keys()))
            else:
                logger.error("No scalers available for selection")
                return None
                
            logger.info(f"No evaluation results available, selected default: {self.selected_scaler}")
        
        return self.selected_scaler
    
    def integrate_with_feature_engineering(self, feature_engineering):
        """
        Integrate selected scaling approach with feature engineering.
        
        Args:
            feature_engineering: FeatureEngineering instance to integrate with
            
        Returns:
            Updated FeatureEngineering instance
        """
        # Select best approach if not already selected
        if self.selected_scaler is None:
            self.select_best_approach()
        
        # Get selected scaler
        scaler = self.scalers.get(self.selected_scaler)
        if scaler is None:
            logger.error(f"Selected scaler {self.selected_scaler} not initialized")
            return feature_engineering
        
        # Replace feature engineering's scaler with selected scaler
        try:
            # Store original scaler for reference
            original_scaler = feature_engineering.scaler
            
            # Replace with selected scaler
            feature_engineering.scaler = scaler
            feature_engineering.is_fitted = False  # Force re-fit on next use
            
            logger.info(f"Replaced feature engineering scaler with {self.selected_scaler}")
            
            # Override normalize_features method if using library catalog scaler
            if self.selected_scaler == 'library_catalog':
                # Save original method
                original_normalize = feature_engineering.normalize_features
                
                # Create new method with library catalog functionality
                def new_normalize_features(X, fit=False):
                    # Get feature names
                    feature_names = feature_engineering.get_feature_names()
                    
                    # Use library catalog scaler
                    if fit or not feature_engineering.is_fitted:
                        X_scaled = feature_engineering.scaler.fit_transform(X, feature_names)
                        feature_engineering.is_fitted = True
                    else:
                        X_scaled = feature_engineering.scaler.transform(X)
                    
                    return X_scaled
                
                # Replace method
                feature_engineering.normalize_features = new_normalize_features.__get__(
                    feature_engineering, feature_engineering.__class__
                )
                
                logger.info("Enhanced normalize_features method for library catalog scaling")
        
        except Exception as e:
            logger.error(f"Error integrating with feature engineering: {e}")
        
        return feature_engineering
    
    def get_scaler(self, name=None):
        """
        Get a specific scaler instance.
        
        Args:
            name: Name of scaler to get (uses selected scaler if None)
            
        Returns:
            Scaler instance
        """
        if name is None:
            name = self.selected_scaler or self.select_best_approach()
        
        return self.scalers.get(name)
    
    def save_scaling_report(self, output_path=None):
        """
        Save scaling integration report.
        
        Args:
            output_path: Path to save report (uses default if None)
            
        Returns:
            Path to saved report
        """
        if output_path is None:
            output_dir = self.config.get('visualization', {}).get('output_dir', 'scaling_viz')
            output_path = os.path.join(output_dir, "scaling_integration_report.html")
        
        # Create report directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Feature Scaling Integration Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                .section {{ margin-bottom: 30px; }}
                .code {{ 
                    font-family: monospace; 
                    background-color: #f8f9fa; 
                    padding: 10px; 
                    border-radius: 4px;
                    white-space: pre-wrap;
                }}
                .summary {{ background-color: #f8f9fa; padding: 15px; border-left: 4px solid #4e73df; }}
                .selected {{ background-color: #d4edda; border-left: 4px solid #28a745; padding: 15px; }}
            </style>
        </head>
        <body>
            <h1>Feature Scaling Integration Report</h1>
            <div class="summary">
                <h3>Entity Resolution Pipeline - Library Catalog</h3>
                <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Scaling Approaches Compared</h2>
                <p>The following scaling approaches were evaluated:</p>
                <ul>
        """
        
        # Add scaling approaches
        for name in self.scalers:
            html_content += f"<li>{name.replace('_', ' ').title()}</li>"
        
        html_content += """
                </ul>
            </div>
        """
        
        # Add selected approach
        if self.selected_scaler:
            html_content += f"""
            <div class="section">
                <h2>Selected Scaling Approach</h2>
                <div class="selected">
                    <h3>{self.selected_scaler.replace('_', ' ').title()}</h3>
                    <p>This approach was selected based on evaluation metrics.</p>
                </div>
            </div>
            """
        
        # Add integration code example
        html_content += """
            <div class="section">
                <h2>Integration with Entity Resolution Pipeline</h2>
                <p>Use the following code to integrate the selected scaling approach:</p>
                <div class="code">
from src.scaling_integration import ScalingIntegration

# Initialize scaling integration
scaling = ScalingIntegration('scaling_config.yml')

# Compare scaling approaches (optional)
evaluation_results = scaling.compare_scaling_approaches(X_train, feature_names, y_train)

# Select best approach
best_approach = scaling.select_best_approach(evaluation_results)

# Integrate with feature engineering
enhanced_feature_engineering = scaling.integrate_with_feature_engineering(feature_engineering)
                </div>
            </div>
            
            <div class="section">
                <h2>Configuration</h2>
                <p>The scaling configuration used:</p>
                <div class="code">
        """
        
        # Add scaling configuration
        import yaml
        html_content += yaml.dump(self.config, default_flow_style=False)
        
        html_content += """
                </div>
            </div>
            
            <footer>
                <p>Entity Resolution Pipeline - Feature Scaling Integration</p>
            </footer>
        </body>
        </html>
        """
        
        # Save HTML report
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Scaling integration report saved to {output_path}")
        return output_path
    
    def visualize_threshold_impact(self, X, y, feature_names, base_threshold=0.6, thresholds=None):
        """
        Visualize impact of different decision thresholds with various scaling approaches.
        
        Args:
            X: Feature matrix
            y: True labels
            feature_names: Names of features
            base_threshold: Base decision threshold
            thresholds: List of thresholds to evaluate (default: [0.5, 0.6, 0.7, 0.8, 0.9])
            
        Returns:
            Path to saved visualization
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
        
        # Use default thresholds if not provided
        if thresholds is None:
            thresholds = self.config.get('evaluation', {}).get(
                'compare_thresholds', [0.5, 0.6, 0.7, 0.8, 0.9]
            )
        
        # Check if we have enough positive samples to calculate precision
        unique_classes, class_counts = np.unique(y, return_counts=True)
        class_balance = {cls: count for cls, count in zip(unique_classes, class_counts)}
        
        if 1 not in class_balance or class_balance[1] < 5:
            logger.warning(f"Insufficient positive samples ({class_balance.get(1, 0)}) for reliable metrics. "
                          f"Adding synthetic positive samples for visualization.")
            
            # Add synthetic positive samples for visualization
            positive_count = 5 if 1 not in class_balance else class_balance[1]
            synth_count = max(10 - positive_count, 0)
            
            if synth_count > 0:
                # Create synthetic samples with high feature values (positive examples)
                X_synth = np.ones((synth_count, X.shape[1])) * 0.8
                y_synth = np.ones(synth_count)
                
                # Append to original data
                X = np.vstack([X, X_synth])
                y = np.append(y, y_synth)
                logger.info(f"Added {synth_count} synthetic positive samples for visualization")
        
        # Suppress specific warning types
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message="Precision is ill-defined")
        
            # Scale data with each approach
            if not self.scaling_results:
                # Scale data with each approach
                scaled_data = {}
                for name, scaler in self.scalers.items():
                    try:
                        # Handle different scaler interfaces
                        if name == 'library_catalog':
                            X_scaled = scaler.fit_transform(X, feature_names)
                        else:
                            X_scaled = scaler.fit_transform(X)
                        
                        scaled_data[name] = X_scaled
                    except Exception as e:
                        logger.error(f"Error scaling data with {name}: {e}")
                
                self.scaling_results = scaled_data
            else:
                scaled_data = self.scaling_results
            
            # Train a simple model for each scaling approach
            models = {}
            for name, X_scaled in scaled_data.items():
                try:
                    # Use class_weight='balanced' to handle imbalanced classes
                    model = LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear')
                    model.fit(X_scaled, y)
                    models[name] = model
                except Exception as e:
                    logger.error(f"Error training model for {name}: {e}")
            
            # Evaluate with different thresholds
            threshold_results = []
            
            for name, model in models.items():
                X_scaled = scaled_data[name]
                
                # Get predicted probabilities
                try:
                    y_proba = model.predict_proba(X_scaled)[:, 1]
                    
                    # Evaluate at each threshold
                    for threshold in thresholds:
                        y_pred = (y_proba >= threshold).astype(int)
                        
                        # Calculate metrics with zero_division=0 to handle undefined precision
                        precision = precision_score(y, y_pred, zero_division=0)
                        recall = recall_score(y, y_pred, zero_division=0)
                        f1 = f1_score(y, y_pred, zero_division=0)
                        accuracy = accuracy_score(y, y_pred)
                        
                        # Save results
                        threshold_results.append({
                            'scaling_method': name,
                            'threshold': threshold,
                            'precision': precision,
                            'recall': recall,
                            'f1': f1,
                            'accuracy': accuracy
                        })
                except Exception as e:
                    logger.error(f"Error evaluating thresholds for {name}: {e}")
        
        # Create DataFrame
        if threshold_results:
            threshold_df = pd.DataFrame(threshold_results)
            
            # Create visualization
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            metrics = ['precision', 'recall', 'f1', 'accuracy']
            
            for i, (metric, ax) in enumerate(zip(metrics, axes.flatten())):
                # Create line plot
                for name in scaled_data.keys():
                    if name in [result['scaling_method'] for result in threshold_results]:
                        method_data = threshold_df[threshold_df['scaling_method'] == name]
                        ax.plot(method_data['threshold'], method_data[metric], marker='o', label=name)
                
                # Add styling
                ax.set_title(f"{metric.title()} vs. Threshold")
                ax.set_xlabel("Decision Threshold")
                ax.set_ylabel(metric.title())
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.legend()
            
            # Add overall title
            plt.suptitle("Impact of Decision Threshold Across Scaling Methods", fontsize=16, y=1.02)
            plt.tight_layout()
            
            # Save figure
            output_dir = self.config.get('visualization', {}).get('output_dir', 'scaling_viz')
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, "threshold_impact.png")
            
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            plt.close()
            
            logger.info(f"Threshold impact visualization saved to {output_file}")
            return output_file
        else:
            logger.error("Unable to visualize threshold impact due to errors or lack of data")
            return None
    
    def integrate_auto(self, feature_engineering, X=None, feature_names=None, y=None):
        """
        Automatically integrate best scaling approach with minimal setup.
        
        Args:
            feature_engineering: FeatureEngineering instance to integrate with
            X: Feature matrix for evaluation (optional)
            feature_names: Feature names for evaluation (optional)
            y: Labels for evaluation (optional)
            
        Returns:
            Updated FeatureEngineering instance
        """
        # Compare scaling approaches if training data provided
        if X is not None and feature_names is not None:
            self.compare_scaling_approaches(X, feature_names, y)
            self.select_best_approach()
        else:
            # Select default based on configuration
            self.select_best_approach()
        
        # Integrate with feature engineering
        enhanced_fe = self.integrate_with_feature_engineering(feature_engineering)
        
        # Save integration report
        self.save_scaling_report()
        
        return enhanced_fe
    
    def validate_with_cross_validation(self, X, y, feature_names, cv=5):
        """
        Validate scaling approaches with cross-validation.
        
        Args:
            X: Feature matrix
            y: True labels
            feature_names: Names of features
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary of cross-validation results
        """
        from sklearn.model_selection import StratifiedKFold
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
        
        logger.info(f"Validating scaling approaches with {cv}-fold cross-validation")
        
        # Check if we have enough positive samples to calculate precision
        unique_classes, class_counts = np.unique(y, return_counts=True)
        class_balance = {cls: count for cls, count in zip(unique_classes, class_counts)}
        
        if 1 not in class_balance or class_balance[1] < cv:
            logger.warning(f"Insufficient positive samples ({class_balance.get(1, 0)}) for reliable metrics. "
                          f"Cross-validation may not produce meaningful results.")
        
        # Initialize cross-validation splitter
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        # Store results for each approach
        cv_results = {}
        
        # Evaluate each scaling approach
        for name, scaler in self.scalers.items():
            # Initialize metrics storage
            fold_metrics = []
            
            # Suppress specific warning types
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, message="Precision is ill-defined")
            
                # Perform cross-validation
                for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                    # Split data
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    
                    try:
                        # Scale features
                        if name == 'library_catalog':
                            X_train_scaled = scaler.fit_transform(X_train, feature_names)
                            X_test_scaled = scaler.transform(X_test)
                        else:
                            X_train_scaled = scaler.fit_transform(X_train)
                            X_test_scaled = scaler.transform(X_test)
                        
                        # Train classifier with class weights to handle imbalance
                        clf = LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear')
                        clf.fit(X_train_scaled, y_train)
                        
                        # Make predictions
                        y_pred = clf.predict(X_test_scaled)
                        y_proba = clf.predict_proba(X_test_scaled)[:, 1]
                        
                        # Calculate metrics with zero_division=0 to handle undefined precision
                        metrics = {
                            'fold': fold,
                            'accuracy': accuracy_score(y_test, y_pred),
                            'precision': precision_score(y_test, y_pred, zero_division=0),
                            'recall': recall_score(y_test, y_pred, zero_division=0),
                            'f1': f1_score(y_test, y_pred, zero_division=0),
                        }
                        
                        # Calculate metrics at different thresholds
                        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
                        for threshold in thresholds:
                            y_pred_threshold = (y_proba >= threshold).astype(int)
                            metrics[f'precision_{threshold}'] = precision_score(y_test, y_pred_threshold, zero_division=0)
                            metrics[f'recall_{threshold}'] = recall_score(y_test, y_pred_threshold, zero_division=0)
                            metrics[f'f1_{threshold}'] = f1_score(y_test, y_pred_threshold, zero_division=0)
                        
                        fold_metrics.append(metrics)
                        
                    except Exception as e:
                        logger.error(f"Error in fold {fold} for {name}: {e}")
            
            # Calculate average metrics across folds
            avg_metrics = {}
            
            if fold_metrics:
                # Calculate averages for each metric
                for metric in fold_metrics[0].keys():
                    if metric != 'fold':  # Skip fold number
                        avg_metrics[metric] = np.mean([fold[metric] for fold in fold_metrics])
                
                # Store standard deviations for key metrics
                for metric in ['accuracy', 'precision', 'recall', 'f1']:
                    avg_metrics[f'{metric}_std'] = np.std([fold[metric] for fold in fold_metrics])
                
                # Store fold details
                avg_metrics['fold_details'] = fold_metrics
            
            # Store results for this approach
            cv_results[name] = avg_metrics
            
            logger.info(f"Cross-validation results for {name}: "
                       f"Accuracy: {avg_metrics.get('accuracy', 0):.4f}, "
                       f"F1: {avg_metrics.get('f1', 0):.4f}")
        
        # Create visualization if visualizer available
        if self.visualizer:
            try:
                self._visualize_cv_results(cv_results)
            except Exception as e:
                logger.error(f"Error visualizing cross-validation results: {e}")
        
        return cv_results
    
    def _visualize_cv_results(self, cv_results):
        """
        Visualize cross-validation results.
        
        Args:
            cv_results: Dictionary of cross-validation results
            
        Returns:
            Path to saved visualization
        """
        # Prepare data for visualization
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        methods = list(cv_results.keys())
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Extract values and errors
            values = [cv_results[method].get(metric, 0) for method in methods]
            errors = [cv_results[method].get(f'{metric}_std', 0) for method in methods]
            
            # Create bar chart with error bars
            bars = ax.bar(methods, values, yerr=errors, capsize=10, 
                    color='skyblue', edgecolor='navy', alpha=0.7)
            
            # Add values on top of bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=9)
            
            # Add styling
            ax.set_title(f'{metric.title()} by Scaling Method')
            ax.set_ylim(0, 1.1)  # Metrics are in range [0, 1]
            ax.set_ylabel(metric.title())
            ax.set_xticklabels([m.replace('_', ' ').title() for m in methods], rotation=30, ha='right')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add overall title
        plt.suptitle('Cross-Validation Results by Scaling Method', fontsize=16, y=1.02)
        plt.tight_layout()
        
        # Save visualization
        output_dir = self.config.get('visualization', {}).get('output_dir', 'scaling_viz')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "cross_validation_results.png")
        
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Cross-validation results visualization saved to {output_file}")
        return output_file
    
    def visualize_decision_boundaries(self, X, y, feature_names, feature_pair=(0, 1)):
        """
        Visualize decision boundaries before and after scaling.
        
        Args:
            X: Feature matrix
            y: True labels
            feature_names: Names of features
            feature_pair: Tuple of feature indices to visualize (default: first two features)
            
        Returns:
            Path to saved visualization
        """
        from sklearn.linear_model import LogisticRegression
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Check if we have enough positive samples to calculate precision
        unique_classes, class_counts = np.unique(y, return_counts=True)
        class_balance = {cls: count for cls, count in zip(unique_classes, class_counts)}
        
        if 1 not in class_balance or class_balance[1] < 3:
            logger.warning(f"Insufficient positive samples ({class_balance.get(1, 0)}) for reliable decision boundaries. "
                          f"Adding synthetic positive samples for visualization.")
            
            # Add synthetic positive samples for visualization
            positive_count = 3 if 1 not in class_balance else class_balance[1]
            synth_count = max(5 - positive_count, 0)
            
            if synth_count > 0:
                # Create synthetic samples with high feature values (positive examples)
                X_synth = np.ones((synth_count, X.shape[1])) * 0.8
                y_synth = np.ones(synth_count)
                
                # Append to original data
                X = np.vstack([X, X_synth])
                y = np.append(y, y_synth)
                logger.info(f"Added {synth_count} synthetic positive samples for visualization")
        
        # Select features to plot
        f1_idx, f2_idx = feature_pair
        f1_name = feature_names[f1_idx] if f1_idx < len(feature_names) else f"Feature {f1_idx}"
        f2_name = feature_names[f2_idx] if f2_idx < len(feature_names) else f"Feature {f2_idx}"
        
        # Create feature subset for visualization
        X_subset = X[:, [f1_idx, f2_idx]]
        
        # Create figure grid based on number of scaling methods
        n_methods = len(self.scalers) + 1  # +1 for original data
        n_cols = min(3, n_methods)
        n_rows = (n_methods + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Plot original data with decision boundary - suppress precision warning
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message="Precision is ill-defined")
            self._plot_decision_boundary(X_subset, y, axes[0], f1_name, f2_name, "Original Data")
        
        # Plot scaled data for each method
        for i, (name, scaler) in enumerate(self.scalers.items(), 1):
            try:
                # Scale data
                if name == 'library_catalog':
                    X_scaled = scaler.fit_transform(X, feature_names)
                else:
                    X_scaled = scaler.fit_transform(X)
                
                # Extract selected features
                X_scaled_subset = X_scaled[:, [f1_idx, f2_idx]]
                
                # Plot decision boundary - suppress precision warning
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning, message="Precision is ill-defined")
                    self._plot_decision_boundary(X_scaled_subset, y, axes[i], 
                                              f1_name, f2_name, 
                                              f"{name.replace('_', ' ').title()}")
            except Exception as e:
                logger.error(f"Error plotting decision boundary for {name}: {e}")
                # Add error message to plot
                axes[i].text(0.5, 0.5, f"Error: {str(e)}", 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f"{name.replace('_', ' ').title()}")
        
        # Hide empty subplots
        for i in range(n_methods, len(axes)):
            axes[i].set_visible(False)
        
        # Add overall title
        plt.suptitle(f'Decision Boundaries: {f1_name} vs {f2_name}', fontsize=16, y=1.02)
        plt.tight_layout()
        
        # Save visualization
        output_dir = self.config.get('visualization', {}).get('output_dir', 'scaling_viz')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"decision_boundaries_{f1_idx}_{f2_idx}.png")
        
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Decision boundaries visualization saved to {output_file}")
        return output_file
    
    def _plot_decision_boundary(self, X, y, ax, feature1_name, feature2_name, title):
        """
        Plot decision boundary for a feature pair.
        
        Args:
            X: Feature matrix (2 features only)
            y: True labels
            ax: Matplotlib axis to plot on
            feature1_name: Name of first feature
            feature2_name: Name of second feature
            title: Plot title
        """
        from sklearn.linear_model import LogisticRegression
        
        # Train classifier - handle class imbalance
        clf = LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear')
        clf.fit(X, y)
        
        # Create mesh grid for decision boundary
        x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
        y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        
        # Make predictions on mesh grid
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary and contours
        contour = ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
        ax.contour(xx, yy, Z, [0.5], linewidths=1, colors='k')
        
        # Plot training points
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k', alpha=0.7)
        
        # Add labels and title
        ax.set_xlabel(feature1_name)
        ax.set_ylabel(feature2_name)
        ax.set_title(title)
        
        # Set axis limits
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        
        return contour, scatter
    
    def visualize_feature_transformations(self, X, feature_names):
        """
        Visualize how each feature is transformed by different scaling methods.
        
        Args:
            X: Feature matrix
            feature_names: Names of features
            
        Returns:
            Path to saved visualization
        """
        # Scale data with each approach
        scaled_data = {}
        for name, scaler in self.scalers.items():
            try:
                # Handle different scaler interfaces
                if name == 'library_catalog':
                    X_scaled = scaler.fit_transform(X, feature_names)
                else:
                    X_scaled = scaler.fit_transform(X)
                
                scaled_data[name] = X_scaled
            except Exception as e:
                logger.error(f"Error scaling data with {name}: {e}")
        
        # Determine number of features and create figure layout
        n_features = len(feature_names)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        # Create figure with subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Plot histogram for each feature
        for i, (feature_idx, ax) in enumerate(zip(range(n_features), axes)):
            feature_name = feature_names[feature_idx] if feature_idx < len(feature_names) else f"Feature {feature_idx}"
            
            # Plot original distribution
            ax.hist(X[:, feature_idx], bins=30, alpha=0.5, label="Original", color="gray")
            
            # Plot scaled distributions
            colors = ['blue', 'green', 'red', 'purple', 'orange']
            for j, (name, X_scaled) in enumerate(scaled_data.items()):
                color = colors[j % len(colors)]
                ax.hist(X_scaled[:, feature_idx], bins=30, alpha=0.3, label=name, color=color)
            
            # Add plot styling
            ax.set_title(feature_name)
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
            
            # Add feature statistics
            orig_mean = np.mean(X[:, feature_idx])
            orig_std = np.std(X[:, feature_idx])
            stats_text = f"Original: μ={orig_mean:.3f}, σ={orig_std:.3f}\n"
            
            # Add statistics for each scaled version
            for name, X_scaled in scaled_data.items():
                scaled_mean = np.mean(X_scaled[:, feature_idx])
                scaled_std = np.std(X_scaled[:, feature_idx])
                stats_text += f"{name}: μ={scaled_mean:.3f}, σ={scaled_std:.3f}\n"
            
            # Add text box with statistics
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, va="top", ha="right",
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
                   fontsize=8)
        
        # Hide empty subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        # Add legend to figure
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.02),
                  fancybox=True, shadow=True, ncol=len(scaled_data)+1)
        
        # Add overall title
        plt.suptitle("Feature Transformations Across Scaling Methods", fontsize=16, y=1.02)
        plt.tight_layout(rect=[0, 0.05, 1, 0.98])
        
        # Save figure
        output_dir = self.config.get('visualization', {}).get('output_dir', 'scaling_viz')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "feature_transformations.png")
        
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Feature transformations visualization saved to {output_file}")
        return output_file
    
    def integrate_with_classification_pipeline(self, pipeline):
        """
        Integrate selected scaling approach with the full entity resolution pipeline.
        
        Args:
            pipeline: The main entity resolution pipeline
            
        Returns:
            Updated pipeline
        """
        # Select best approach if not already selected
        if self.selected_scaler is None:
            self.select_best_approach()
        
        # Get selected scaler
        scaler = self.scalers.get(self.selected_scaler)
        if scaler is None:
            logger.error(f"Selected scaler {self.selected_scaler} not initialized")
            return pipeline
        
        # Integration points will vary based on pipeline implementation
        # This is a generic implementation that assumes:
        # 1. The pipeline has a 'feature_engineering' component
        # 2. The pipeline has a 'classifier' component that accepts a 'scaler' parameter
        
        try:
            # Update feature engineering component
            if hasattr(pipeline, 'feature_engineering'):
                pipeline.feature_engineering = self.integrate_with_feature_engineering(
                    pipeline.feature_engineering
                )
                logger.info("Updated pipeline feature engineering component")
            
            # Update classifier component if it exists and accepts a scaler
            if hasattr(pipeline, 'classifier') and hasattr(pipeline.classifier, 'set_scaler'):
                pipeline.classifier.set_scaler(scaler)
                logger.info("Updated pipeline classifier with new scaler")
            
            # Update configuration if it exists
            if hasattr(pipeline, 'config'):
                pipeline.config['scaling'] = {
                    'method': self.selected_scaler,
                    'parameters': self.config.get('scaling_default', {})
                }
                logger.info("Updated pipeline configuration with scaling settings")
        
        except Exception as e:
            logger.error(f"Error integrating with classification pipeline: {e}")
        
        return pipeline


class ScalingEvaluator:
    """
    Evaluates the effectiveness of scaling strategies for entity resolution.
    
    Provides metrics and visualizations to analyze how scaling affects
    feature distributions and classification performance.
    """
    
    def __init__(self, config):
        """
        Initialize with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
    
    def evaluate_scaling(self, X_original, X_scaled, feature_names, y_true=None):
        """
        Evaluate effectiveness of scaling strategy.
        
        Args:
            X_original: Original feature values
            X_scaled: Scaled feature values
            feature_names: Names of features
            y_true: True match labels if available
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Calculate basic distribution statistics
        metrics["distribution"] = self._analyze_distributions(X_original, X_scaled, feature_names)
        
        # Calculate feature separation power (if labels available)
        if y_true is not None:
            metrics["separation"] = self._analyze_feature_separation(X_scaled, feature_names, y_true)
            
        # Analyze correlation between features
        metrics["correlation"] = self._analyze_feature_correlation(X_scaled, feature_names)
        
        # Calculate information retention
        metrics["info_retention"] = self._analyze_information_retention(X_original, X_scaled, feature_names)
        
        return metrics
    
    def _analyze_distributions(self, X_original, X_scaled, feature_names):
        """
        Analyze distribution changes from scaling.
        
        Args:
            X_original: Original feature values
            X_scaled: Scaled feature values
            feature_names: Names of features
            
        Returns:
            Dictionary of distribution metrics
        """
        from scipy.stats import skew, kurtosis
        
        distribution_metrics = {}
        
        for i, name in enumerate(feature_names):
            orig_vals = X_original[:, i]
            scaled_vals = X_scaled[:, i]
            
            # Calculate distribution statistics
            distribution_metrics[name] = {
                "original": {
                    "min": float(np.min(orig_vals)),
                    "max": float(np.max(orig_vals)),
                    "mean": float(np.mean(orig_vals)),
                    "median": float(np.median(orig_vals)),
                    "std": float(np.std(orig_vals)),
                    "skew": float(skew(orig_vals)),
                    "kurtosis": float(kurtosis(orig_vals))
                },
                "scaled": {
                    "min": float(np.min(scaled_vals)),
                    "max": float(np.max(scaled_vals)),
                    "mean": float(np.mean(scaled_vals)),
                    "median": float(np.median(scaled_vals)),
                    "std": float(np.std(scaled_vals)),
                    "skew": float(skew(scaled_vals)),
                    "kurtosis": float(kurtosis(scaled_vals))
                }
            }
            
        return distribution_metrics
    
    def _analyze_feature_separation(self, X_scaled, feature_names, y_true):
        """
        Analyze how well scaled features separate matches from non-matches.
        
        Args:
            X_scaled: Scaled feature values
            feature_names: Names of features
            y_true: True match labels
            
        Returns:
            Dictionary of separation metrics
        """
        from sklearn.metrics import roc_auc_score
        
        separation_metrics = {}
        
        # Split data by match status
        X_match = X_scaled[y_true == 1]
        X_non_match = X_scaled[y_true == 0]
        
        for i, name in enumerate(feature_names):
            match_vals = X_match[:, i] if X_match.size > 0 else np.array([])
            non_match_vals = X_non_match[:, i] if X_non_match.size > 0 else np.array([])
            
            # Skip if either class has no samples
            if len(match_vals) == 0 or len(non_match_vals) == 0:
                separation_metrics[name] = {
                    "error": "Insufficient samples for separation analysis"
                }
                continue
            
            # Calculate separation metrics
            try:
                # Calculate AUC for this feature
                auc = float(roc_auc_score(y_true, X_scaled[:, i]))
            except:
                auc = 0.5  # Default if calculation fails
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt((np.var(match_vals) + np.var(non_match_vals)) / 2)
            effect_size = float((np.mean(match_vals) - np.mean(non_match_vals)) / pooled_std) if pooled_std > 0 else 0.0
            
            separation_metrics[name] = {
                "match_mean": float(np.mean(match_vals)),
                "non_match_mean": float(np.mean(non_match_vals)),
                "mean_difference": float(np.mean(match_vals) - np.mean(non_match_vals)),
                "effect_size": effect_size,
                "auc": auc
            }
            
        return separation_metrics
    
    def _analyze_feature_correlation(self, X_scaled, feature_names):
        """
        Analyze correlation between scaled features.
        
        Args:
            X_scaled: Scaled feature values
            feature_names: Names of features
            
        Returns:
            Dictionary with correlation matrix
        """
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(X_scaled, rowvar=False)
        
        # Convert to dictionary format
        correlation = {
            "matrix": corr_matrix.tolist(),
            "feature_names": feature_names
        }
        
        return correlation
    
    def _analyze_information_retention(self, X_original, X_scaled, feature_names):
        """
        Analyze information retention after scaling.
        
        Args:
            X_original: Original feature values
            X_scaled: Scaled feature values
            feature_names: Names of features
            
        Returns:
            Dictionary of information retention metrics
        """
        retention_metrics = {}
        
        for i, name in enumerate(feature_names):
            orig_vals = X_original[:, i]
            scaled_vals = X_scaled[:, i]
            
            # Calculate rank correlation (Spearman's rho)
            # This measures how well the ordering of values is preserved
            from scipy.stats import spearmanr
            try:
                rho, p_value = spearmanr(orig_vals, scaled_vals)
            except:
                rho, p_value = 0.0, 1.0
            
            # Calculate mutual information
            # This measures how much information is shared between variables
            from sklearn.feature_selection import mutual_info_regression
            try:
                mi = mutual_info_regression(orig_vals.reshape(-1, 1), scaled_vals)[0]
            except:
                mi = 0.0
            
            retention_metrics[name] = {
                "rank_correlation": float(rho),
                "rank_correlation_p_value": float(p_value),
                "mutual_information": float(mi)
            }
            
        return retention_metrics