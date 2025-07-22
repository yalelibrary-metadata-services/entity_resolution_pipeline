"""
Scaling Bridge Module

A streamlined interface between feature engineering and advanced scaling strategies.
Provides a coherent API for scaling selection, evaluation, and integration with 
minimal coupling and clear dependency boundaries.

Key Components:
1. Scaling strategy selection and evaluation
2. Feature engineering integration
3. Configuration management
4. Error recovery and fallback mechanisms

Architectural Design Principles:
- Single responsibility: Each method handles one specific scaling task
- Clear interface: Well-defined entry points for external components
- Robust error handling: Graceful degradation with appropriate fallbacks
- Performance optimization: Efficient vector processing with minimal overhead
"""

import logging
import numpy as np
import os
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum

from src.scaling_integration import ScalingIntegration
from src.feature_engineering import FeatureEngineering
from src.robust_scaler import LibraryCatalogScaler

logger = logging.getLogger(__name__)

class ScalingStrategy(Enum):
    """Enumeration of supported scaling strategies for clear type safety."""
    LIBRARY_CATALOG = "library_catalog"
    ROBUST_MINMAX = "robust_minmax"  # Keeping for backward compatibility
    STANDARD_MINMAX = "standard_minmax"  # Deprecated
    STANDARD_SCALER = "standard_scaler"  # Deprecated


class ScalingBridge:
    """
    Bridge class providing a simplified interface between scaling functionality 
    and feature engineering pipeline.
    
    Implements a facade pattern to hide scaling complexity while providing a 
    consistent interface with robust error handling and performance optimization.
    """
    
    def __init__(self, config_path: str = "scaling_config.yml"):
        """
        Initialize the scaling bridge with configuration.
        
        Args:
            config_path: Path to scaling configuration file
        """
        self.config_path = config_path
        self.scaling = ScalingIntegration(config_path)
        self.feature_engineering = None
        self.feature_names = []
        self.scaler_state = {
            "is_active": False,
            "strategy": None,
            "feature_count": 0
        }
        logger.info(f"Initialized ScalingBridge with config from {config_path}")
    
    def connect(self, feature_engineering: FeatureEngineering) -> 'ScalingBridge':
        """
        Connect to feature engineering module with streamlined interface.
        
        Args:
            feature_engineering: Feature engineering instance to integrate with
            
        Returns:
            Self for method chaining
        """
        if feature_engineering is None:
            raise ValueError("Feature engineering instance cannot be None")
            
        self.feature_engineering = feature_engineering
        
        # Extract feature information
        if hasattr(feature_engineering, "feature_registry") and feature_engineering.feature_registry:
            self.feature_names = list(feature_engineering.feature_registry.keys())
        else:
            logger.warning("Feature registry not found or empty, using empty feature list")
            self.feature_names = []
        
        logger.info(f"Connected to feature engineering with {len(self.feature_names)} features")
        return self
    
    def evaluate(self, 
                feature_vectors: np.ndarray, 
                labels: Optional[np.ndarray] = None,
                auto_select: bool = True) -> Dict[str, Any]:
        """
        Evaluate scaling approaches and optionally select the best one.
        
        Args:
            feature_vectors: Feature vectors to evaluate scaling on
            labels: Optional ground truth labels for supervised evaluation
            auto_select: Whether to automatically select best approach
            
        Returns:
            Evaluation results dictionary with detailed metrics
        """
        # Validate inputs
        self._validate_feature_matrix(feature_vectors)
        
        # Ensure feature names match dimensions
        self._synchronize_feature_names(feature_vectors.shape[1])
        
        # Compare scaling approaches
        results = self.scaling.compare_scaling_approaches(feature_vectors, self.feature_names, labels)
        
        # Select best approach if requested
        if auto_select and results:
            strategy = self.scaling.select_best_approach(results)
            self.scaler_state["strategy"] = strategy
            logger.info(f"Selected optimal scaling strategy: {strategy}")
        
        # Generate visualization report in a standardized location
        self._generate_scaling_report()
        
        return results
    
    def apply(self, strategy: Optional[Union[str, ScalingStrategy]] = None) -> FeatureEngineering:
        """
        Apply selected scaling strategy to feature engineering with clean delegation.
        
        Args:
            strategy: Optional explicit strategy to use (overrides evaluated best strategy)
            
        Returns:
            Updated feature engineering instance
        """
        # Validate state
        if self.feature_engineering is None:
            raise ValueError("Feature engineering not connected. Call connect() first.")
        
        # Determine which strategy to use (explicit parameter > evaluated > default)
        if strategy is not None:
            # Handle enum values
            if isinstance(strategy, ScalingStrategy):
                strategy = strategy.value
            
            # Set and log the choice
            self.scaler_state["strategy"] = strategy
            logger.info(f"Using explicitly specified scaling strategy: {strategy}")
        elif not self.scaler_state.get("strategy"):
            # Default to library_catalog if no strategy set (standardized approach)
            self.scaler_state["strategy"] = "library_catalog"
            logger.info(f"Using default scaling strategy: library_catalog")
        
        # Get and validate scaler
        scaler = self._get_validated_scaler()
        if not scaler:
            logger.error(f"Could not obtain valid scaler for strategy: {self.scaler_state['strategy']}")
            return self.feature_engineering
            
        # Store scaler for delegation
        self.scaler = scaler
        
        # Update state
        self.scaler_state["is_active"] = True
        self.scaler_state["feature_count"] = len(self.feature_names)
        
        logger.info(f"Applied {self.scaler_state['strategy']} scaling strategy with clean delegation")
        return self.feature_engineering
        
    def apply_with_fitted_scaler(self, strategy: Optional[Union[str, ScalingStrategy]] = None) -> FeatureEngineering:
        """
        Apply selected scaling strategy to feature engineering using pre-fitted scaler.
        Unlike apply(), this method doesn't re-fit the scaler, ensuring consistent scaling with training.
        
        Args:
            strategy: Optional explicit strategy to use (overrides evaluated best strategy)
            
        Returns:
            Updated feature engineering instance
        """
        # Validate state
        if self.feature_engineering is None:
            raise ValueError("Feature engineering not connected. Call connect() first.")
        
        # Set strategy if provided
        if strategy is not None:
            if isinstance(strategy, ScalingStrategy):
                strategy = strategy.value
            self.scaler_state["strategy"] = strategy
            logger.info(f"Using explicitly specified scaling strategy: {strategy}")
        elif not self.scaler_state.get("strategy"):
            self.scaler_state["strategy"] = "library_catalog"
            logger.info(f"Using default scaling strategy: library_catalog")
        
        # Get the fitted scaler
        scaler = self._get_validated_scaler()
        if not scaler:
            logger.error(f"Could not obtain valid scaler for strategy: {self.scaler_state['strategy']}")
            return self.feature_engineering
            
        # Validate scaler is already fitted
        if not hasattr(scaler, 'scalers') or not scaler.scalers:
            logger.warning("Scaler is not fitted; consistency with training cannot be guaranteed")
            
        # Store scaler for delegation (production mode)
        self.scaler = scaler
        
        # Update state
        self.scaler_state["is_active"] = True
        self.scaler_state["feature_count"] = len(self.feature_names)
        
        logger.info(f"Applied pre-fitted {self.scaler_state['strategy']} scaler with clean delegation")
        return self.feature_engineering
    
    def reset(self) -> FeatureEngineering:
        """
        Reset feature engineering to use built-in scaling.
        
        Returns:
            Feature engineering instance
        """
        if self.feature_engineering is None:
            logger.warning("No feature engineering instance to reset")
            return None
            
        if not self.scaler_state.get("is_active", False):
            logger.info("No active scaling to reset")
            return self.feature_engineering
            
        # Clear scaling bridge scaler reference
        if hasattr(self, 'scaler'):
            delattr(self, 'scaler')
        
        # Update state
        self.scaler_state["is_active"] = False
        
        logger.info("Successfully reset scaling configuration")
        return self.feature_engineering
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get scaling configuration metadata.
        
        Returns:
            Dictionary with detailed scaling metadata
        """
        # Basic metadata
        metadata = {
            'strategy': self.scaler_state.get("strategy"),
            'is_active': self.scaler_state.get("is_active", False),
            'feature_count': len(self.feature_names),
            'config_path': self.config_path
        }
        
        # Add scaler-specific information if active
        if self.scaler_state.get("is_active") and self.feature_engineering and hasattr(self.feature_engineering, "scaler"):
            scaler = self.feature_engineering.scaler
            
            # Get scaler parameters based on type
            if hasattr(scaler, "data_min_") and hasattr(scaler, "data_max_"):
                metadata["scaling_range"] = {
                    "min": scaler.data_min_.tolist() if hasattr(scaler.data_min_, "tolist") else scaler.data_min_,
                    "max": scaler.data_max_.tolist() if hasattr(scaler.data_max_, "tolist") else scaler.data_max_
                }
            
            # Add configuration parameters
            if hasattr(scaler, "clip_percentile"):
                metadata["clip_percentile"] = scaler.clip_percentile
                
            if hasattr(scaler, "feature_groups"):
                metadata["has_feature_groups"] = True
                
        return metadata
    
    def _validate_feature_matrix(self, feature_vectors: np.ndarray) -> None:
        """
        Validate feature matrix for scaling operations.
        
        Args:
            feature_vectors: Feature vectors to validate
            
        Raises:
            ValueError: If feature matrix is invalid
        """
        if not isinstance(feature_vectors, np.ndarray):
            raise TypeError(f"Feature vectors must be numpy array, got {type(feature_vectors)}")
            
        if feature_vectors.ndim != 2:
            raise ValueError(f"Feature vectors must be 2D, got {feature_vectors.ndim}D")
            
        if feature_vectors.shape[0] == 0:
            raise ValueError("Feature vector array is empty (no samples)")
            
        if feature_vectors.shape[1] == 0:
            raise ValueError("Feature vector array has no features (zero width)")
    
    def _synchronize_feature_names(self, feature_count: int) -> None:
        """
        Ensure feature names match the expected feature count.
        
        Args:
            feature_count: Expected number of features
        """
        if len(self.feature_names) != feature_count:
            logger.warning(f"Feature names count ({len(self.feature_names)}) doesn't match "
                          f"expected feature count ({feature_count})")
            
            # Adjust feature names list length
            if len(self.feature_names) > feature_count:
                # Truncate if too many names
                self.feature_names = self.feature_names[:feature_count]
                logger.info(f"Truncated feature names list to {feature_count} entries")
            else:
                # Extend with generic names if too few
                additional = [f"feature_{i}" for i in range(len(self.feature_names), feature_count)]
                self.feature_names.extend(additional)
                logger.info(f"Extended feature names with {len(additional)} generic entries")
    
    def _get_validated_scaler(self) -> Any:
        """
        Get and validate scaler for selected strategy.
        
        Returns:
            Valid scaler instance or None if validation fails
        """
        strategy = self.scaler_state.get("strategy")
        if not strategy:
            logger.error("No scaling strategy selected")
            return None
            
        # Get scaler from integration module
        scaler = self.scaling.get_scaler(strategy)
        if scaler is None:
            logger.error(f"Failed to obtain scaler for strategy: {strategy}")
            return None
            
        # Basic validation
        if not hasattr(scaler, "fit") or not hasattr(scaler, "transform"):
            logger.error(f"Invalid scaler: missing required fit/transform methods")
            return None
            
        return scaler
    
    def normalize_features(self, feature_vectors: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Clean delegation-based normalization using LibraryCatalogScaler.
        
        Args:
            feature_vectors: Feature vectors to normalize
            fit: Whether to fit the scaler on this data
            
        Returns:
            Normalized feature vectors
        """
        # Ensure we have a valid scaler
        if not hasattr(self, 'scaler') or self.scaler is None:
            scaler = self._get_validated_scaler()
            if not scaler:
                logger.error("Could not obtain valid scaler")
                return feature_vectors.copy()
            self.scaler = scaler
        
        # Early exit for empty input
        if feature_vectors.size == 0:
            return feature_vectors.copy()
            
        try:
            # Ensure we have feature names
            if not self.feature_names and hasattr(self.feature_engineering, 'feature_registry'):
                self.feature_names = list(self.feature_engineering.feature_registry.keys())
            
            # Apply scaling using LibraryCatalogScaler
            if fit:
                scaled_features = self.scaler.fit_transform(feature_vectors, self.feature_names)
                logger.debug(f"Fitted and transformed features using {self.scaler_state.get('strategy', 'library_catalog')}")
            else:
                # Check if scaler is fitted
                if hasattr(self.scaler, 'scalers') and self.scaler.scalers:
                    scaled_features = self.scaler.transform(feature_vectors)
                else:
                    # Scaler not fitted, perform fit_transform
                    scaled_features = self.scaler.fit_transform(feature_vectors, self.feature_names)
                    logger.debug("Scaler not fitted, performing fit_transform")
            
            return scaled_features
            
        except Exception as e:
            logger.error(f"Error in ScalingBridge normalization: {str(e)}")
            return feature_vectors.copy()
    
    def normalize_features_production(self, feature_vectors: np.ndarray) -> np.ndarray:
        """
        Production normalization that never refits the scaler.
        Always uses the pre-fitted scaler from training.
        
        Args:
            feature_vectors: Feature vectors to normalize
            
        Returns:
            Normalized feature vectors using pre-fitted scaler
        """
        # Ensure we have a valid fitted scaler
        if not hasattr(self, 'scaler') or self.scaler is None:
            logger.error("No fitted scaler available for production normalization")
            return feature_vectors.copy()
            
        # Check if scaler is fitted
        if not hasattr(self.scaler, 'scalers') or not self.scaler.scalers:
            logger.error("Scaler is not fitted for production use")
            return feature_vectors.copy()
        
        # Early exit for empty input
        if feature_vectors.size == 0:
            return feature_vectors.copy()
            
        try:
            # Always use transform, never fit in production
            scaled_features = self.scaler.transform(feature_vectors)
            logger.debug(f"Applied pre-fitted scaler to feature vectors (transform only)")
            return scaled_features
            
        except Exception as e:
            logger.error(f"Error in production scaling: {str(e)}")
            return feature_vectors.copy()
    
    def _validate_binary_features(self, X_scaled, X_original, feature_names=None):
        """
        Enhanced validation of binary features with production monitoring and alerting.
        Ensures binary indicator features are preserved during scaling and provides
        comprehensive monitoring for production environments.
        
        Args:
            X_scaled: Scaled feature vectors
            X_original: Original feature vectors
            feature_names: Optional feature names for detailed logging
            
        Returns:
            Dictionary with validation results and monitoring metrics
        """
        validation_results = {
            'status': 'healthy',
            'warnings': [],
            'errors': [],
            'corrections': [],
            'binary_features_found': 0,
            'corrupted_features': 0,
            'timestamp': time.time()
        }
        
        if X_scaled.shape != X_original.shape:
            error_msg = f"Cannot validate binary features: shape mismatch {X_scaled.shape} vs {X_original.shape}"
            logger.error(error_msg)
            validation_results['status'] = 'error'
            validation_results['errors'].append(error_msg)
            return validation_results
        
        # Default feature names if not provided
        if feature_names is None:
            if hasattr(self.feature_engineering, "feature_registry") and self.feature_engineering.feature_registry:
                feature_names = list(self.feature_engineering.feature_registry.keys())
            else:
                feature_names = [f"feature_{i}" for i in range(X_scaled.shape[1])]
                
        # Find binary indicator features
        binary_features = []
        binary_indices = []
        
        for i, name in enumerate(feature_names):
            if i < X_scaled.shape[1] and ("indicator" in name.lower() or "match" in name.lower()):
                binary_features.append(name)
                binary_indices.append(i)
                
        validation_results['binary_features_found'] = len(binary_features)
        
        if not binary_indices:
            logger.debug("No binary features found for validation")
            return validation_results
            
        # Enhanced validation with comprehensive monitoring
        total_fixed = 0
        critical_issues = 0
        
        for idx, name in zip(binary_indices, binary_features):
            orig_col = X_original[:, idx]
            scaled_col = X_scaled[:, idx]
            
            # Analyze feature distribution for monitoring
            unique_scaled = np.unique(scaled_col)
            unique_orig = np.unique(orig_col)
            
            # Check if original values are properly binary
            orig_is_binary = np.all(np.isin(unique_orig, [0.0, 1.0]))
            scaled_is_binary = np.all(np.isin(unique_scaled, [0.0, 1.0]))
            
            if not orig_is_binary:
                warning_msg = f"Original feature '{name}' contains non-binary values: {unique_orig}"
                logger.warning(warning_msg)
                validation_results['warnings'].append(warning_msg)
            
            # Find values that should be binary but aren't exactly 0.0 or 1.0
            should_be_zero = (orig_col < 0.5) & ((scaled_col != 0.0) & (scaled_col > 0.001))
            should_be_one = (orig_col >= 0.5) & ((scaled_col != 1.0) & (scaled_col < 0.999))
            
            problem_count = np.sum(should_be_zero) + np.sum(should_be_one)
            
            if problem_count > 0:
                validation_results['corrupted_features'] += 1
                
                # Force binary values (correction)
                X_scaled[should_be_zero, idx] = 0.0
                X_scaled[should_be_one, idx] = 1.0
                total_fixed += problem_count
                
                correction_msg = f"Corrected {problem_count} non-binary values in '{name}'"
                validation_results['corrections'].append(correction_msg)
                logger.warning(f"BINARY VALIDATION: {correction_msg}")
                
                # Check for critical entity pairs (from specification)
                if "indicator" in name.lower() and problem_count > 0:
                    critical_issues += 1
                    
            # Final validation check
            final_scaled = X_scaled[:, idx]
            final_unique = np.unique(final_scaled)
            if not np.all(np.isin(final_unique, [0.0, 1.0])):
                error_msg = f"CRITICAL: Failed to fix binary feature '{name}' - still contains: {final_unique}"
                logger.error(error_msg)
                validation_results['errors'].append(error_msg)
                validation_results['status'] = 'error'
                
        # Summary logging and monitoring alerts
        if total_fixed > 0:
            summary_msg = f"Fixed {total_fixed} binary values across {len(binary_features)} features"
            logger.info(f"BINARY VALIDATION SUMMARY: {summary_msg}")
            validation_results['corrections'].append(summary_msg)
            
        if critical_issues > 0:
            critical_msg = f"CRITICAL: {critical_issues} indicator features had binary corruption"
            logger.error(critical_msg)
            validation_results['errors'].append(critical_msg)
            validation_results['status'] = 'critical' if validation_results['status'] != 'error' else 'error'
            
        # Production monitoring thresholds
        corruption_rate = validation_results['corrupted_features'] / max(1, validation_results['binary_features_found'])
        if corruption_rate > 0.1:  # More than 10% corruption
            warning_msg = f"High binary feature corruption rate: {corruption_rate:.2%}"
            logger.warning(warning_msg)
            validation_results['warnings'].append(warning_msg)
            if validation_results['status'] == 'healthy':
                validation_results['status'] = 'warning'
        
        # Log audit trail for production monitoring
        if validation_results['corrections'] or validation_results['errors']:
            audit_entry = {
                'timestamp': validation_results['timestamp'],
                'binary_features_checked': validation_results['binary_features_found'],
                'corrupted_features': validation_results['corrupted_features'],
                'corrections_made': len(validation_results['corrections']),
                'status': validation_results['status']
            }
            logger.info(f"BINARY VALIDATION AUDIT: {audit_entry}")
        
        return validation_results
    
    def validate_scaling_health(self, X_current: np.ndarray, feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Production health check for scaling configuration and data integrity.
        Implements monitoring thresholds from SCALING_DATA_VALIDATION.md.
        
        Args:
            X_current: Current feature data to validate
            feature_names: Names of features (optional)
            
        Returns:
            Dictionary with health check results and monitoring metrics
        """
        health_report = {
            'status': 'healthy',
            'warnings': [],
            'errors': [],
            'metrics': {},
            'timestamp': time.time()
        }
        
        try:
            # Ensure we have feature names
            if feature_names is None:
                if hasattr(self.feature_engineering, 'feature_registry'):
                    feature_names = list(self.feature_engineering.feature_registry.keys())
                else:
                    feature_names = [f"feature_{i}" for i in range(X_current.shape[1])]
            
            # Check if we have a scaler for comparison
            if not hasattr(self, 'scaler') or self.scaler is None:
                warning_msg = "No scaler available for health check"
                health_report['warnings'].append(warning_msg)
                logger.warning(warning_msg)
                health_report['status'] = 'warning'
                return health_report
            
            # 1. Compression Factor Analysis
            compression_metrics = {}
            if hasattr(self.scaler, 'get_compression_metrics'):
                try:
                    scaler_metrics = self.scaler.get_compression_metrics()
                    for group_name, group_metrics in scaler_metrics.items():
                        for feature_name, metrics in group_metrics.items():
                            compression_factor = metrics.get('compression_factor', 1.0)
                            compression_metrics[feature_name] = compression_factor
                            
                            # Apply thresholds from SCALING_DATA_VALIDATION.md
                            if compression_factor > 2.0:
                                error_msg = f"Feature '{feature_name}': compression factor {compression_factor:.2f} > 2.0 (critical threshold)"
                                health_report['errors'].append(error_msg)
                                logger.error(error_msg)
                            elif compression_factor > 1.5:
                                warning_msg = f"Feature '{feature_name}': compression factor {compression_factor:.2f} > 1.5 (warning threshold)"
                                health_report['warnings'].append(warning_msg)
                                logger.warning(warning_msg)
                                
                except Exception as e:
                    logger.warning(f"Could not get compression metrics: {e}")
            
            health_report['metrics']['compression_factors'] = compression_metrics
            
            # 2. Feature Range Drift Detection
            range_drift_metrics = {}
            for i, name in enumerate(feature_names):
                if i < X_current.shape[1]:
                    current_min = float(np.min(X_current[:, i]))
                    current_max = float(np.max(X_current[:, i]))
                    current_range = current_max - current_min
                    
                    range_drift_metrics[name] = {
                        'min': current_min,
                        'max': current_max,
                        'range': current_range
                    }
                    
                    # Check for extreme values (basic drift detection)
                    if current_min < -0.1 or current_max > 1.1:
                        warning_msg = f"Feature '{name}' outside expected range [0,1]: [{current_min:.3f}, {current_max:.3f}]"
                        health_report['warnings'].append(warning_msg)
                        logger.warning(warning_msg)
            
            health_report['metrics']['feature_ranges'] = range_drift_metrics
            
            # 3. Binary Feature Integrity Check
            binary_validation = self._validate_binary_features(X_current, X_current, feature_names)
            health_report['metrics']['binary_validation'] = binary_validation
            
            if binary_validation['status'] == 'error':
                health_report['status'] = 'error'
                health_report['errors'].extend(binary_validation['errors'])
            elif binary_validation['status'] in ['critical', 'warning']:
                if health_report['status'] == 'healthy':
                    health_report['status'] = binary_validation['status']
                health_report['warnings'].extend(binary_validation['warnings'])
            
            # 4. Overall Status Assessment
            if health_report['errors']:
                health_report['status'] = 'error'
            elif health_report['warnings']:
                if health_report['status'] == 'healthy':
                    health_report['status'] = 'warning'
            
            # 5. Summary Metrics
            health_report['metrics']['summary'] = {
                'total_features': len(feature_names),
                'binary_features_checked': binary_validation.get('binary_features_found', 0),
                'compression_issues': sum(1 for cf in compression_metrics.values() if cf > 1.5),
                'range_drift_issues': len([w for w in health_report['warnings'] if 'outside expected range' in w])
            }
            
            logger.info(f"Scaling health check completed: {health_report['status']} - "
                       f"{len(health_report['errors'])} errors, {len(health_report['warnings'])} warnings")
            
        except Exception as e:
            error_msg = f"Health check failed: {str(e)}"
            health_report['status'] = 'error'
            health_report['errors'].append(error_msg)
            logger.error(error_msg)
        
        return health_report
    
    def _generate_scaling_report(self) -> str:
        """
        Generate scaling visualization report in standardized location.
        
        Returns:
            Path to generated report or None if generation failed
        """
        try:
            # Create output directory
            output_dir = os.path.join('data', 'output', 'scaling_viz')
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate report
            report_path = self.scaling.save_scaling_report()
            
            if report_path and os.path.exists(report_path):
                logger.info(f"Generated scaling report at {report_path}")
                return report_path
            else:
                logger.warning("Failed to generate scaling report")
                return None
                
        except Exception as e:
            logger.error(f"Error generating scaling report: {e}")
            return None
