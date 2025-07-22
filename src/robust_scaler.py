"""
Robust Feature Scaling for Library Catalog Entity Resolution

This module implements specialized scaling approaches for similarity features
in library catalog entity resolution, with outlier handling and feature-specific
scaling strategies.
"""

import numpy as np
import logging
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union

logger = logging.getLogger(__name__)

class RobustMinMaxScaler:
    """
    Enhanced MinMaxScaler with outlier handling for similarity features.
    
    Implements percentile-based clipping to prevent extreme values from
    compressing the similarity scale.
    """
    
    def __init__(self, feature_range=(0, 1), clip_percentile=95, clip_lower=False):
        """
        Initialize the robust scaler.
        
        Args:
            feature_range: Target range for scaling (default: [0, 1])
            clip_percentile: Percentile threshold for clipping extreme values (default: 95)
            clip_lower: Whether to also clip lower values (default: False)
        """
        self.feature_range = feature_range
        self.clip_percentile = clip_percentile
        self.clip_lower = clip_lower
        self.min_ = None
        self.max_ = None
        self.data_min_ = None  # Original min before clipping
        self.data_max_ = None  # Original max before clipping
        self.feature_names = None
        self.n_features = None
    
    def fit(self, X, feature_names=None):
        """
        Fit scaler to training data with outlier handling.
        
        Args:
            X: Training data (2D array-like)
            feature_names: Names of features in X (optional)
            
        Returns:
            self
        """
        # Convert to numpy array
        X_array = np.asarray(X)
        
        # Store feature names and dimensions
        self.n_features = X_array.shape[1]
        self.feature_names = feature_names if feature_names else [f"feature_{i}" for i in range(self.n_features)]
        
        # Store original min/max
        self.data_min_ = np.min(X_array, axis=0)
        self.data_max_ = np.max(X_array, axis=0)
        
        # Ensure data is numeric
        if not np.issubdtype(X_array.dtype, np.number):
            raise TypeError("Input array must contain only numeric values")
        
        # Check for NaN or infinity values
        if np.isnan(X_array).any() or np.isinf(X_array).any():
            logger.warning("Input array contains NaN or infinite values. These will be treated as missing.")
            # Replace NaN and inf with reasonable values for percentile calculation
            X_array_clean = np.copy(X_array)
            mask_nan_inf = np.logical_or(np.isnan(X_array_clean), np.isinf(X_array_clean))
            X_array_clean[mask_nan_inf] = np.nanmean(X_array_clean, axis=0, keepdims=True)
            
            # Calculate clipping thresholds on cleaned data
            upper_thresholds = np.percentile(X_array_clean, self.clip_percentile, axis=0)
            
            if self.clip_lower:
                lower_percentile = 100 - self.clip_percentile
                lower_thresholds = np.percentile(X_array_clean, lower_percentile, axis=0)
            else:
                lower_thresholds = self.data_min_
        else:
            # Calculate clipping thresholds on clean data
            upper_thresholds = np.percentile(X_array, self.clip_percentile, axis=0)
            
            if self.clip_lower:
                lower_percentile = 100 - self.clip_percentile
                lower_thresholds = np.percentile(X_array, lower_percentile, axis=0)
            else:
                lower_thresholds = self.data_min_
        
        # Store clipped min/max for scaling
        self.min_ = lower_thresholds
        self.max_ = upper_thresholds
        
        # Log scaling parameters
        for i, feat_name in enumerate(self.feature_names):
            # Avoid division by zero when min and max are the same
            denom = (self.max_[i] - self.min_[i])
            if denom == 0:
                compression_factor = 1.0  # No compression if min=max (all values are identical)
                logger.warning(f"Feature '{feat_name}' has identical min and max values: {self.min_[i]}")
            else:
                compression_factor = (self.data_max_[i] - self.data_min_[i]) / denom
            
            logger.debug(f"Feature '{feat_name}':")
            logger.debug(f"  Original range: [{self.data_min_[i]:.4f}, {self.data_max_[i]:.4f}]")
            logger.debug(f"  Clipped range: [{self.min_[i]:.4f}, {self.max_[i]:.4f}]")
            logger.debug(f"  Compression factor: {compression_factor:.4f}")
            
            # Warn about significant compression
            if compression_factor > 2.0:
                logger.warning(f"Feature '{feat_name}' has high compression factor ({compression_factor:.2f}). "
                              f"Consider adjusting clip_percentile or using feature-specific scaling.")
        
        return self
    
    def transform(self, X):
        """
        Transform features using robust scaling.
        
        Args:
            X: Features to transform (2D array-like)
            
        Returns:
            Scaled features
        """
        # Check if fitted
        if self.min_ is None or self.max_ is None:
            raise ValueError("RobustMinMaxScaler has not been fitted. Call fit() before transform().")
        
        # Convert to numpy array
        X_array = np.asarray(X)
        
        # Clip values to the fitted ranges
        X_clipped = np.clip(X_array, self.min_[np.newaxis, :], self.max_[np.newaxis, :])
        
        # Apply min-max scaling
        range_min, range_max = self.feature_range
        denominator = self.max_ - self.min_
        
        # Handle zero-width features
        denominator[denominator == 0.0] = 1.0
        
        # Scale using the clipped range
        X_scaled = range_min + (X_clipped - self.min_) * (range_max - range_min) / denominator
        
        return X_scaled
    
    def fit_transform(self, X, feature_names=None):
        """
        Fit to data, then transform it.
        
        Args:
            X: Features to fit and transform (2D array-like)
            feature_names: Names of features in X (optional)
            
        Returns:
            Scaled features
        """
        return self.fit(X, feature_names).transform(X)
    
    def inverse_transform(self, X_scaled):
        """
        Inverse transform scaled data back to original scale.
        
        Args:
            X_scaled: Scaled features (2D array-like)
            
        Returns:
            Features on original scale
        """
        # Check if fitted
        if self.min_ is None or self.max_ is None:
            raise ValueError("RobustMinMaxScaler has not been fitted. Call fit() before inverse_transform().")
        
        # Convert to numpy array
        X_array = np.asarray(X_scaled)
        
        # Get scaling parameters
        range_min, range_max = self.feature_range
        
        # Inverse transform
        X_original = self.min_ + (X_array - range_min) * (self.max_ - self.min_) / (range_max - range_min)
        
        return X_original
    
    def get_compression_metrics(self):
        """
        Get metrics describing how much features were compressed by outlier handling.
        
        Returns:
            Dictionary of compression metrics
        """
        # Check if fitted
        if self.min_ is None or self.max_ is None:
            raise ValueError("RobustMinMaxScaler has not been fitted. Call get_compression_metrics() after fit().")
        
        metrics = {}
        
        for i, feat_name in enumerate(self.feature_names):
            original_range = self.data_max_[i] - self.data_min_[i]
            clipped_range = self.max_[i] - self.min_[i]
            
            # Calculate metrics safely (handling division by zero)
            metrics[feat_name] = {
                'original_min': float(self.data_min_[i]),
                'original_max': float(self.data_max_[i]),
                'clipped_min': float(self.min_[i]),
                'clipped_max': float(self.max_[i]),
                'original_range': float(original_range),
                'clipped_range': float(clipped_range),
                'compression_factor': float(original_range / clipped_range) if clipped_range > 0 else 1.0,
                'percent_range_used': float((clipped_range / original_range) * 100) if original_range > 0 else 100.0
            }
        
        return metrics


class LibraryCatalogScaler:
    """
    Domain-specific scaler for library catalog entity resolution with feature groups.
    
    Applies different scaling strategies to different types of features based on
    their expected behavior in library catalog data.
    """
    
    def __init__(self, config):
        """
        Initialize with configuration parameters.
        
        Args:
            config: Configuration dictionary with feature parameters
        """
        self.config = config
        self.scaling_config = config.get('feature_scaling', {})
        self.feature_groups = self._define_feature_groups()
        self.scalers = {}
        self.feature_indices = {}
        self.feature_names = None
        self.binary_features = []
    
    def _define_feature_groups(self):
        """
        Define feature groups with specific library catalog scaling characteristics.
        
        Returns:
            Dictionary defining feature groups and their scaling properties
        """
        # Load custom feature groups if specified
        custom_groups = self.scaling_config.get('feature_groups', {})
        if custom_groups:
            return custom_groups
        
        # Default feature groups with domain-specific settings
        return {
            # Person name features - critical for entity disambiguation
            "person_features": {
                "patterns": ["person_cosine"],
                "range": self.scaling_config.get('person_range', (0, 1)),
                "percentile": self.scaling_config.get('person_percentile', 98),
            },
            
            # Title features - important but can be noisy
            "title_features": {
                "patterns": ["title_cosine", "title_cosine_squared", "person_title_squared"],
                "range": self.scaling_config.get('title_range', (0, 1)),
                "percentile": self.scaling_config.get('title_percentile', 95),
            },
            
            # Context features - composite signals
            "context_features": {
                "patterns": ["composite_cosine"],
                "range": self.scaling_config.get('context_range', (0, 1)),
                "percentile": self.scaling_config.get('context_percentile', 90),
            },
            
            # Binary indicators - preserve exact values
            "binary_features": {
                "patterns": [
                    "birth_death_match", 
                    "person_low_levenshtein_indicator",
                    "person_low_jaro_winkler_indicator"
                ],
                "range": (0, 1),
                "percentile": 100,  # No clipping
            },
            
            # Categorical features - preserve exact discrete values
            "categorical_features": {
                "patterns": ["taxonomy_dissimilarity"],
                "range": (0, 1),
                "percentile": 100,  # No clipping - preserve categorical values
            },
            
            # Role-related features
            "role_features": {
                "patterns": ["title_role_adjusted"],
                "range": self.scaling_config.get('role_range', (0, 1)),
                "percentile": self.scaling_config.get('role_percentile', 95),
            },
            
            # Default for other features
            "default": {
                "range": (0, 1),
                "percentile": 95
            }
        }
    
    def fit(self, X, feature_names):
        """
        Fit multiple scalers based on feature groups.
        
        Args:
            X: Feature matrix to fit
            feature_names: Names of features in X
            
        Returns:
            self
        """
        # Ensure feature_names is a list of strings
        if not isinstance(feature_names, (list, tuple)):
            raise TypeError("feature_names must be a list or tuple")
        if not all(isinstance(name, str) for name in feature_names):
            logger.warning("Non-string feature names detected - converting to strings")
            feature_names = [str(name) for name in feature_names]
        
        # Store feature names
        self.feature_names = list(feature_names)
        
        # Convert to numpy array and ensure numeric
        X_array = np.asarray(X)
        
        # Verify data is numeric
        if not np.issubdtype(X_array.dtype, np.number):
            raise TypeError(f"Input data must be numeric, got {X_array.dtype}")
        
        # Check dimensions match
        if X_array.shape[1] != len(self.feature_names):
            raise ValueError(f"Number of features ({X_array.shape[1]}) doesn't match number of feature names ({len(self.feature_names)})")
            
        # Check for and handle NaN/inf values
        if np.isnan(X_array).any() or np.isinf(X_array).any():
            logger.warning("Input data contains NaN or infinite values. Cleaning data...")
            X_array = np.copy(X_array)
            
            # Replace NaN with column means
            col_means = np.nanmean(X_array, axis=0)
            for i in range(X_array.shape[1]):
                mask = np.isnan(X_array[:, i])
                X_array[mask, i] = col_means[i]
            
            # Replace inf with large finite values
            inf_mask = np.isinf(X_array)
            if inf_mask.any():
                X_array[inf_mask & (X_array > 0)] = np.finfo(np.float64).max / 10
                X_array[inf_mask & (X_array < 0)] = np.finfo(np.float64).min / 10
        
        # Identify binary features
        self.binary_features = []
        for group_name, group_info in self.feature_groups.items():
            if group_name == "binary_features":
                for pattern in group_info.get("patterns", []):
                    matching_features = [feat for feat in feature_names if pattern in feat]
                    self.binary_features.extend(matching_features)
        
        # Initialize feature indices for each group
        self.feature_indices = {}
        
        # Assign features to groups based on patterns
        assigned_features = set()
        
        for group_name, group_info in self.feature_groups.items():
            if group_name == "default":
                continue
            
            # Find matching features for this group
            group_features = []
            
            for pattern in group_info.get("patterns", []):
                matching_features = [i for i, feat in enumerate(feature_names) if pattern in feat]
                group_features.extend(matching_features)
            
            if group_features:
                # Remove duplicates and sort
                group_features = sorted(set(group_features))
                
                # Store indices for this group
                self.feature_indices[group_name] = group_features
                
                # Track assigned features
                assigned_features.update(group_features)
                
                # Create and fit scaler for this group
                group_feature_names = [feature_names[i] for i in group_features]
                group_data = X_array[:, group_features]
                
                # Special handling for binary and categorical features (no scaling needed)
                if group_name in ["binary_features", "categorical_features"]:
                    if group_name == "categorical_features":
                        # Skip scaling for categorical features, preserve exact values
                        logger.info(f"Categorical features will not be scaled: {group_feature_names}")
                        
                        # Validate categorical feature values (especially taxonomy_dissimilarity)
                        for i, name in enumerate(group_feature_names):
                            col_data = group_data[:, i]
                            unique_values = np.unique(col_data)
                            
                            if "taxonomy" in name.lower():
                                expected_values = {0.0, 0.15, 0.4}
                                if not set(unique_values).issubset(expected_values):
                                    logger.warning(f"Categorical feature '{name}' contains unexpected values: {unique_values}")
                                    logger.warning(f"Expected values: {expected_values}")
                                else:
                                    logger.info(f"Categorical feature '{name}' validated with values: {sorted(unique_values)}")
                    
                    else:  # binary_features
                        # Skip scaling for binary features, but ensure correct values
                        logger.info(f"Binary features will not be scaled: {group_feature_names}")
                        
                        # Binary indicators should be already correct values
                        # Audit the feature values to ensure they are valid (0.0 or 1.0)
                        for i, name in enumerate(group_feature_names):
                            col_data = group_data[:, i]
                            unique_values = np.unique(col_data)
                            
                            # Check for invalid values
                            is_binary = np.all(np.isin(unique_values, [0.0, 1.0]))
                            if not is_binary:
                                logger.warning(f"Binary feature '{name}' contains non-binary values: {unique_values}")
                                logger.warning(f"This may indicate a calculation issue in the feature engineering pipeline")
                                
                                # CRITICAL FIX: For binary indicator features, ensure values are exactly 0.0 or 1.0
                                # This is crucial for consistent classification behavior
                                if "indicator" in name.lower() or "match" in name.lower():
                                    logger.info(f"CRITICAL FIX: Converting binary feature '{name}' to exact binary values")
                                    corrected_values = np.zeros_like(col_data)
                                    corrected_values[col_data >= 0.5] = 1.0
                                    group_data[:, i] = corrected_values
                                    
                                    # Verify fix worked
                                    fixed_values = np.unique(group_data[:, i])
                                    is_binary_now = np.all(np.isin(fixed_values, [0.0, 1.0]))
                                    if is_binary_now:
                                        logger.info(f"Successfully fixed binary values for {name}")
                                    else:
                                        logger.error(f"Failed to fix binary values for {name}: {fixed_values}")
                        
                        # Store the corrected binary features data in the feature engineering cache for reference
                        self._binary_features_data = group_data.copy()
                    
                    # Store None as the scaler to indicate no scaling
                else:
                    # Create robust scaler with group-specific settings
                    self.scalers[group_name] = RobustMinMaxScaler(
                        feature_range=group_info.get("range", (0, 1)),
                        clip_percentile=group_info.get("percentile", 95)
                    )
                    
                    # Fit scaler
                    self.scalers[group_name].fit(group_data, group_feature_names)
                    logger.info(f"Fitted {group_name} scaler for {len(group_features)} features: {group_feature_names}")
        
        # Handle remaining features with default scaler
        unassigned_features = list(set(range(len(feature_names))) - assigned_features)
        
        if unassigned_features:
            self.feature_indices["default"] = unassigned_features
            default_feature_names = [feature_names[i] for i in unassigned_features]
            default_data = X_array[:, unassigned_features]
            
            # Create and fit default scaler
            self.scalers["default"] = RobustMinMaxScaler(
                feature_range=self.feature_groups["default"].get("range", (0, 1)),
                clip_percentile=self.feature_groups["default"].get("percentile", 95)
            )
            
            self.scalers["default"].fit(default_data, default_feature_names)
            logger.info(f"Fitted default scaler for {len(unassigned_features)} features: {default_feature_names}")
        
        return self
    
    def transform(self, X):
        """
        Transform features using the appropriate scaler for each feature group.
        
        Args:
            X: Feature matrix to transform
            
        Returns:
            Scaled features
        """
        # Check if fitted
        if not self.scalers:
            raise ValueError("LibraryCatalogScaler has not been fitted. Call fit() before transform().")
        
        # Convert to numpy array
        X_array = np.asarray(X)
        X_scaled = X_array.copy()
        
        # Apply each scaler to its feature group
        for group_name, indices in self.feature_indices.items():
            if not indices:
                continue
            
            # Skip binary and categorical features (no scaling, preserve exact values)
            if group_name in ["binary_features", "categorical_features"] or self.scalers[group_name] is None:
                # Handle binary and categorical features appropriately
                for i, idx in enumerate(indices):
                    feature_name = self.feature_names[idx] if idx < len(self.feature_names) else f"feature_{idx}"
                    col_data = X_array[:, idx]
                    
                    if group_name == "categorical_features":
                        # For categorical features like taxonomy_dissimilarity, preserve exact values
                        # Expected values: {0.0, 0.15, 0.4} for taxonomy dissimilarity
                        X_scaled[:, idx] = col_data.copy()
                        
                        # Validate categorical values (optional validation)
                        unique_values = np.unique(col_data)
                        if "taxonomy" in feature_name.lower():
                            expected_values = {0.0, 0.15, 0.4}
                            unexpected = set(unique_values) - expected_values
                            if unexpected:
                                logger.warning(f"Categorical feature '{feature_name}' contains unexpected values: {unexpected}")
                        
                        logger.debug(f"Preserved categorical feature '{feature_name}' with values: {unique_values}")
                        
                    else:  # binary_features
                        # Always handle binary features consistently
                        # The lack of consistent handling may be causing our critical entity matching problem
                        if "indicator" in feature_name.lower() or "match" in feature_name.lower():
                            # For indicator and match features, we need exact binary values
                            logger.debug(f"Processing binary feature '{feature_name}' with threshold 0.5")
                            
                            # Apply strict binary conversion (exactly 0.0 or 1.0)
                            binary_values = np.zeros_like(col_data)
                            binary_values[col_data >= 0.5] = 1.0
                            X_scaled[:, idx] = binary_values
                            
                            # Check for any values that were changed significantly
                            significant_changes = np.abs(binary_values - col_data) > 0.1
                            if np.any(significant_changes):
                                changed_count = np.sum(significant_changes)
                                original_values = col_data[significant_changes]
                                logger.info(f"BINARY FIX: Corrected {changed_count} values in '{feature_name}' from non-binary to binary")
                                logger.debug(f"  Original values: {original_values[:5]} (showing up to 5)")
                        else:
                            # For other binary features, still ensure correct values
                            unique_values = np.unique(col_data)
                            if not np.all(np.isin(unique_values, [0.0, 1.0])):
                                logger.warning(f"Fixing non-binary values in '{feature_name}' during transform: {unique_values}")
                                
                                # Fix non-binary values
                                fixed_values = np.zeros_like(col_data)
                                fixed_values[col_data >= 0.5] = 1.0
                                X_scaled[:, idx] = fixed_values
                continue
            
            # Extract group data
            group_data = X_array[:, indices]
            
            # Transform with the appropriate scaler
            scaled_group_data = self.scalers[group_name].transform(group_data)
            
            # Update scaled array
            for i, idx in enumerate(indices):
                X_scaled[:, idx] = scaled_group_data[:, i]
        
        return X_scaled
    
    def fit_transform(self, X, feature_names):
        """
        Fit to data, then transform it.
        
        Args:
            X: Feature matrix to fit and transform
            feature_names: Names of features in X
            
        Returns:
            Scaled features
        """
        return self.fit(X, feature_names).transform(X)
    
    def get_compression_metrics(self):
        """
        Get metrics describing how features were compressed by each group's scaler.
        
        Returns:
            Dictionary of compression metrics by feature group
        """
        metrics = {}
        
        for group_name, scaler in self.scalers.items():
            if scaler is not None:  # Skip binary features and other non-scaled groups
                group_metrics = scaler.get_compression_metrics()
                metrics[group_name] = group_metrics
        
        return metrics


# Serialization functions for fitted scalers
def serialize_scaler(scaler, output_path):
    """
    Serialize fitted scaler to disk.
    
    Args:
        scaler: Fitted LibraryCatalogScaler instance
        output_path: Path to save serialized scaler
        
    Returns:
        Output path where scaler was saved
    """
    import os
    import json
    
    # Extract all critical scaling parameters
    serialized_data = {
        "feature_groups": scaler.feature_groups,
        "feature_indices": {k: list(v) for k, v in scaler.feature_indices.items() if v},
        "feature_names": scaler.feature_names,
        "binary_features": scaler.binary_features,
        "version": "1.0",
        "scalers": {}
    }
  
    # Serialize each individual scaler for feature groups
    for group_name, group_scaler in scaler.scalers.items():
        if group_scaler is not None:
            serialized_data["scalers"][group_name] = {
                "min_": group_scaler.min_.tolist() if hasattr(group_scaler, "min_") else None,
                "max_": group_scaler.max_.tolist() if hasattr(group_scaler, "max_") else None,
                "data_min_": group_scaler.data_min_.tolist() if hasattr(group_scaler, "data_min_") else None,
                "data_max_": group_scaler.data_max_.tolist() if hasattr(group_scaler, "data_max_") else None,
                "feature_range": group_scaler.feature_range,
                "clip_percentile": group_scaler.clip_percentile if hasattr(group_scaler, "clip_percentile") else None
            }
  
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
  
    # Save as JSON
    with open(output_path, 'w') as f:
        json.dump(serialized_data, f, indent=2)
  
    logger.info(f"Serialized scaler parameters saved to {output_path}")
    return output_path

def deserialize_scaler(input_path, config):
    """
    Deserialize scaler from disk for production use.
    
    Args:
        input_path: Path to serialized scaler
        config: Configuration dictionary
        
    Returns:
        Reconstructed fitted LibraryCatalogScaler
    """
    import json
    import numpy as np
    
    logger.info(f"Loading serialized scaler from {input_path}")
  
    with open(input_path, 'r') as f:
        serialized_data = json.load(f)
  
    # Create new scaler instance
    scaler = LibraryCatalogScaler(config)
  
    # Restore feature groups and other metadata
    scaler.feature_groups = serialized_data["feature_groups"]
    scaler.feature_indices = {k: np.array(v, dtype=int) for k, v in serialized_data["feature_indices"].items()}
    scaler.feature_names = serialized_data["feature_names"]
    scaler.binary_features = serialized_data["binary_features"]
  
    # Initialize scalers dictionary
    scaler.scalers = {}
  
    # Reconstruct each group scaler
    for group_name, scaler_data in serialized_data["scalers"].items():
        if scaler_data:
            # Initialize appropriate scaler
            group_scaler = RobustMinMaxScaler(
                feature_range=tuple(scaler_data["feature_range"]),
                clip_percentile=scaler_data["clip_percentile"]
            )
          
            # Set fitted parameters
            if scaler_data["min_"] is not None:
                group_scaler.min_ = np.array(scaler_data["min_"])
            if scaler_data["max_"] is not None:
                group_scaler.max_ = np.array(scaler_data["max_"])
            if scaler_data["data_min_"] is not None:
                group_scaler.data_min_ = np.array(scaler_data["data_min_"])
            if scaler_data["data_max_"] is not None:
                group_scaler.data_max_ = np.array(scaler_data["data_max_"])
              
            # Add reconstructed scaler
            scaler.scalers[group_name] = group_scaler
  
    # Mark as fitted
    scaler.is_fitted = True
    logger.info(f"Successfully loaded serialized scaler with {len(scaler.feature_names)} features and {len(scaler.scalers)} feature groups")
  
    return scaler


# ScalingEvaluator has been moved to scaling_integration.py for better separation of concerns