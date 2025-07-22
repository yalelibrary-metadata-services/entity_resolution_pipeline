"""
Recursive Feature Elimination (RFE) for Entity Resolution Pipeline

This module implements Recursive Feature Elimination to identify the optimal
subset of features for entity matching. It works with the existing logistic
regression classifier and supports configurable parameters.
"""

import json
import logging
import os
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

logger = logging.getLogger(__name__)


class RecursiveFeatureEliminator:
    """
    Implements Recursive Feature Elimination for feature selection in entity resolution.
    
    This class works with the existing EntityClassifier to systematically remove
    features and evaluate model performance, identifying the optimal feature subset.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Recursive Feature Eliminator.
        
        Args:
            config: Configuration dictionary containing RFE parameters
        """
        self.config = config
        rfe_config = config.get('feature_selection', {}).get('rfe_config', {})
        
        # RFE parameters
        self.min_features_to_select = rfe_config.get('min_features_to_select', 1)
        self.step = rfe_config.get('step', 1)
        self.scoring_metric = rfe_config.get('scoring_metric', 'precision')
        self.cv_folds = rfe_config.get('cv_folds', 5)
        self.verbose = rfe_config.get('verbose', True)
        self.save_results = rfe_config.get('save_results', True)
        self.results_path = rfe_config.get('results_path', 'data/output/rfe_results.json')
        
        # Decision threshold from main config
        self.decision_threshold = config.get('decision_threshold', 0.65)
        
        # Results storage
        self.ranking_ = None
        self.support_ = None
        self.n_features_ = None
        self.scores_ = {}
        self.feature_names_ = None
        self.elimination_history_ = []
        
        logger.info(f"Initialized RFE with min_features={self.min_features_to_select}, "
                   f"step={self.step}, metric={self.scoring_metric}")
    
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str], 
            classifier_class: Any, classifier_config: Dict[str, Any]) -> 'RecursiveFeatureEliminator':
        """
        Perform Recursive Feature Elimination.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Binary labels (n_samples,)
            feature_names: List of feature names
            classifier_class: EntityClassifier class
            classifier_config: Configuration for classifier
            
        Returns:
            Self for method chaining
        """
        n_samples, n_features = X.shape
        self.feature_names_ = feature_names.copy()
        
        if self.verbose:
            logger.info(f"Starting RFE with {n_features} features and {n_samples} samples")
            logger.info(f"Class distribution: {np.sum(y==1)} matches, {np.sum(y==0)} non-matches")
        
        # Initialize feature support and ranking
        support = np.ones(n_features, dtype=bool)
        ranking = np.ones(n_features, dtype=int)
        
        # Initialize scores tracking
        self.scores_ = {
            'n_features': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'accuracy': [],
            'feature_sets': []
        }
        
        # Stratified K-Fold for cross-validation
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        # RFE main loop
        n_features_remaining = n_features
        elimination_round = 0
        
        with tqdm(total=n_features - self.min_features_to_select, 
                 desc="RFE Progress", disable=not self.verbose) as pbar:
            
            while n_features_remaining > self.min_features_to_select:
                # Get current feature subset
                features_mask = support
                X_subset = X[:, features_mask]
                current_feature_names = [name for name, mask in zip(feature_names, features_mask) if mask]
                
                if self.verbose and elimination_round % 5 == 0:
                    logger.info(f"Round {elimination_round}: Evaluating {n_features_remaining} features")
                
                # Cross-validation scores
                cv_scores = {
                    'precision': [],
                    'recall': [],
                    'f1': [],
                    'accuracy': []
                }
                
                # Feature importances from each fold
                fold_importances = []
                
                for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_subset, y)):
                    X_train, X_val = X_subset[train_idx], X_subset[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    # Train classifier on this fold
                    classifier = classifier_class(classifier_config)
                    classifier.fit(X_train, y_train, current_feature_names)
                    
                    # Get predictions with decision threshold
                    y_pred_proba = classifier.predict_proba(X_val)
                    y_pred = (y_pred_proba >= self.decision_threshold).astype(int)
                    
                    # Calculate metrics
                    cv_scores['precision'].append(precision_score(y_val, y_pred, zero_division=0))
                    cv_scores['recall'].append(recall_score(y_val, y_pred, zero_division=0))
                    cv_scores['f1'].append(f1_score(y_val, y_pred, zero_division=0))
                    cv_scores['accuracy'].append(accuracy_score(y_val, y_pred))
                    
                    # Store feature importances (absolute weights)
                    fold_importances.append(np.abs(classifier.weights))
                
                # Average importances across folds
                avg_importances = np.mean(fold_importances, axis=0)
                
                # Store scores for this feature set
                self.scores_['n_features'].append(n_features_remaining)
                self.scores_['precision'].append(np.mean(cv_scores['precision']))
                self.scores_['recall'].append(np.mean(cv_scores['recall']))
                self.scores_['f1'].append(np.mean(cv_scores['f1']))
                self.scores_['accuracy'].append(np.mean(cv_scores['accuracy']))
                self.scores_['feature_sets'].append(current_feature_names.copy())
                
                # Record elimination history
                self.elimination_history_.append({
                    'round': elimination_round,
                    'n_features': n_features_remaining,
                    'features': current_feature_names.copy(),
                    'importances': dict(zip(current_feature_names, avg_importances)),
                    'cv_scores': {k: np.mean(v) for k, v in cv_scores.items()},
                    'cv_std': {k: np.std(v) for k, v in cv_scores.items()}
                })
                
                # If we've evaluated with all features, move to elimination
                if n_features_remaining <= self.min_features_to_select:
                    break
                
                # Find least important features
                importance_threshold = np.sort(avg_importances)[min(self.step, len(avg_importances)-1)-1]
                features_to_eliminate = avg_importances <= importance_threshold
                
                # Update support and ranking
                support_indices = np.where(support)[0]
                eliminated_indices = support_indices[features_to_eliminate]
                
                for idx in eliminated_indices:
                    support[idx] = False
                    ranking[idx] = n_features - n_features_remaining + 1
                
                n_features_remaining = np.sum(support)
                elimination_round += 1
                pbar.update(1)
        
        # Set final ranking for remaining features
        ranking[support] = 1
        
        # Find optimal number of features based on scoring metric
        metric_scores = self.scores_[self.scoring_metric]
        optimal_idx = np.argmax(metric_scores)
        optimal_n_features = self.scores_['n_features'][optimal_idx]
        
        # Set support for optimal feature set
        self.support_ = np.zeros(n_features, dtype=bool)
        optimal_features = self.scores_['feature_sets'][optimal_idx]
        for i, name in enumerate(feature_names):
            if name in optimal_features:
                self.support_[i] = True
        
        self.ranking_ = ranking
        self.n_features_ = optimal_n_features
        
        if self.verbose:
            logger.info(f"RFE complete. Optimal features: {optimal_n_features}")
            logger.info(f"Best {self.scoring_metric}: {metric_scores[optimal_idx]:.4f}")
            logger.info(f"Selected features: {optimal_features}")
        
        # Save results if requested
        if self.save_results:
            self._save_results()
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform feature matrix to selected features only.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Transformed feature matrix with selected features
        """
        if self.support_ is None:
            raise ValueError("RFE has not been fitted yet. Call fit() first.")
        
        return X[:, self.support_]
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray, feature_names: List[str],
                     classifier_class: Any, classifier_config: Dict[str, Any]) -> np.ndarray:
        """
        Fit RFE and transform features in one step.
        
        Args:
            X: Feature matrix
            y: Labels
            feature_names: Feature names
            classifier_class: EntityClassifier class
            classifier_config: Classifier configuration
            
        Returns:
            Transformed feature matrix
        """
        self.fit(X, y, feature_names, classifier_class, classifier_config)
        return self.transform(X)
    
    def get_support(self, indices: bool = False) -> Union[np.ndarray, List[int]]:
        """
        Get the boolean mask or indices of selected features.
        
        Args:
            indices: If True, return indices instead of boolean mask
            
        Returns:
            Boolean mask or indices of selected features
        """
        if self.support_ is None:
            raise ValueError("RFE has not been fitted yet. Call fit() first.")
        
        if indices:
            return np.where(self.support_)[0].tolist()
        return self.support_
    
    def get_ranking(self) -> Dict[str, int]:
        """
        Get feature ranking (1 is best).
        
        Returns:
            Dictionary mapping feature names to rankings
        """
        if self.ranking_ is None or self.feature_names_ is None:
            raise ValueError("RFE has not been fitted yet. Call fit() first.")
        
        return dict(zip(self.feature_names_, self.ranking_))
    
    def get_selected_features(self) -> List[str]:
        """
        Get names of selected features.
        
        Returns:
            List of selected feature names
        """
        if self.support_ is None or self.feature_names_ is None:
            raise ValueError("RFE has not been fitted yet. Call fit() first.")
        
        return [name for name, selected in zip(self.feature_names_, self.support_) if selected]
    
    def plot_scores(self, save_path: Optional[str] = None) -> None:
        """
        Plot performance metrics vs number of features.
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.scores_:
            raise ValueError("No scores available. Run fit() first.")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot all metrics
        for metric in ['precision', 'recall', 'f1', 'accuracy']:
            ax.plot(self.scores_['n_features'], self.scores_[metric], 
                   marker='o', label=metric.capitalize())
        
        # Highlight optimal point
        optimal_idx = np.argmax(self.scores_[self.scoring_metric])
        optimal_n = self.scores_['n_features'][optimal_idx]
        optimal_score = self.scores_[self.scoring_metric][optimal_idx]
        
        ax.axvline(x=optimal_n, color='red', linestyle='--', alpha=0.7, 
                  label=f'Optimal ({optimal_n} features)')
        ax.scatter([optimal_n], [optimal_score], color='red', s=100, zorder=5)
        
        ax.set_xlabel('Number of Features')
        ax.set_ylabel('Score')
        ax.set_title('RFE: Model Performance vs Number of Features')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Reverse x-axis to show elimination progression
        ax.invert_xaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved RFE scores plot to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, save_path: Optional[str] = None) -> None:
        """
        Plot feature importance/ranking.
        
        Args:
            save_path: Optional path to save the plot
        """
        if self.ranking_ is None or self.feature_names_ is None:
            raise ValueError("RFE has not been fitted yet. Call fit() first.")
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'Feature': self.feature_names_,
            'Ranking': self.ranking_,
            'Selected': self.support_
        })
        df = df.sort_values('Ranking')
        
        # Create color map
        colors = ['green' if selected else 'red' for selected in df['Selected']]
        
        fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.3)))
        
        y_pos = np.arange(len(df))
        ax.barh(y_pos, df['Ranking'], color=colors, alpha=0.7)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df['Feature'])
        ax.set_xlabel('Ranking (1 = Most Important)')
        ax.set_title('Feature Rankings from RFE')
        ax.invert_xaxis()  # Lower ranking = better
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, label='Selected'),
            Patch(facecolor='red', alpha=0.7, label='Eliminated')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved feature importance plot to {save_path}")
        
        plt.show()
    
    def _save_results(self) -> None:
        """Save RFE results to JSON file."""
        # Convert rankings to native Python types
        rankings = self.get_ranking()
        rankings_converted = {k: int(v) for k, v in rankings.items()}
        
        results = {
            'config': {
                'min_features_to_select': int(self.min_features_to_select),
                'step': int(self.step),
                'scoring_metric': self.scoring_metric,
                'cv_folds': int(self.cv_folds),
                'decision_threshold': float(self.decision_threshold)
            },
            'optimal': {
                'n_features': int(self.n_features_),
                'features': self.get_selected_features(),
                'scores': {
                    metric: float(self.scores_[metric][np.argmax(self.scores_[self.scoring_metric])])
                    for metric in ['precision', 'recall', 'f1', 'accuracy']
                }
            },
            'rankings': rankings_converted,
            'elimination_history': self._convert_history_types(self.elimination_history_),
            'all_scores': {
                'n_features': [int(n) for n in self.scores_['n_features']],
                'precision': [float(s) for s in self.scores_['precision']],
                'recall': [float(s) for s in self.scores_['recall']],
                'f1': [float(s) for s in self.scores_['f1']],
                'accuracy': [float(s) for s in self.scores_['accuracy']],
                'feature_sets': self.scores_['feature_sets']
            }
        }
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.results_path), exist_ok=True)
        
        with open(self.results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved RFE results to {self.results_path}")
        
        # Also save as CSV for easy analysis
        csv_path = self.results_path.replace('.json', '.csv')
        scores_df = pd.DataFrame({
            'n_features': self.scores_['n_features'],
            'precision': self.scores_['precision'],
            'recall': self.scores_['recall'],
            'f1': self.scores_['f1'],
            'accuracy': self.scores_['accuracy'],
            'features': [','.join(features) for features in self.scores_['feature_sets']]
        })
        scores_df.to_csv(csv_path, index=False)
        logger.info(f"Saved RFE scores to {csv_path}")
    
    def _convert_history_types(self, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert NumPy types in elimination history to native Python types."""
        converted_history = []
        for entry in history:
            converted_entry = {
                'round': int(entry['round']),
                'n_features': int(entry['n_features']),
                'features': entry['features'],
                'importances': {k: float(v) for k, v in entry['importances'].items()},
                'cv_scores': {k: float(v) for k, v in entry['cv_scores'].items()},
                'cv_std': {k: float(v) for k, v in entry['cv_std'].items()}
            }
            converted_history.append(converted_entry)
        return converted_history
    
    def generate_report(self) -> str:
        """
        Generate a text report of RFE results.
        
        Returns:
            Formatted report string
        """
        if self.ranking_ is None:
            return "RFE has not been fitted yet. Call fit() first."
        
        report = []
        report.append("=" * 80)
        report.append("RECURSIVE FEATURE ELIMINATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Configuration
        report.append("Configuration:")
        report.append(f"  Minimum features: {self.min_features_to_select}")
        report.append(f"  Step size: {self.step}")
        report.append(f"  Scoring metric: {self.scoring_metric}")
        report.append(f"  CV folds: {self.cv_folds}")
        report.append(f"  Decision threshold: {self.decision_threshold}")
        report.append("")
        
        # Optimal results
        report.append("Optimal Feature Set:")
        report.append(f"  Number of features: {self.n_features_}")
        report.append(f"  Selected features: {', '.join(self.get_selected_features())}")
        report.append("")
        
        optimal_idx = np.argmax(self.scores_[self.scoring_metric])
        report.append("  Performance metrics:")
        report.append(f"    Precision: {self.scores_['precision'][optimal_idx]:.4f}")
        report.append(f"    Recall: {self.scores_['recall'][optimal_idx]:.4f}")
        report.append(f"    F1-score: {self.scores_['f1'][optimal_idx]:.4f}")
        report.append(f"    Accuracy: {self.scores_['accuracy'][optimal_idx]:.4f}")
        report.append("")
        
        # Feature rankings
        report.append("Feature Rankings (1 = most important):")
        rankings = self.get_ranking()
        for feature, rank in sorted(rankings.items(), key=lambda x: x[1]):
            status = "SELECTED" if rank == 1 else f"Eliminated in round {rank-1}"
            report.append(f"  {rank}. {feature} - {status}")
        report.append("")
        
        # Performance progression
        report.append("Performance Progression:")
        report.append(f"  {'N Features':<12} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Accuracy':<10}")
        report.append("  " + "-" * 52)
        
        for i in range(len(self.scores_['n_features'])):
            n_feat = self.scores_['n_features'][i]
            prec = self.scores_['precision'][i]
            rec = self.scores_['recall'][i]
            f1 = self.scores_['f1'][i]
            acc = self.scores_['accuracy'][i]
            
            marker = " *" if i == optimal_idx else ""
            report.append(f"  {n_feat:<12} {prec:<10.4f} {rec:<10.4f} {f1:<10.4f} {acc:<10.4f}{marker}")
        
        report.append("")
        report.append("* = Optimal based on " + self.scoring_metric)
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)