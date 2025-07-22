"""
Custom Features Module for Entity Resolution

This module provides functionality for registering and configuring custom features
to extend the feature engineering capabilities of the entity resolution pipeline.
"""

import logging
import numpy as np
import threading
from typing import Dict, Any, List, Set

logger = logging.getLogger(__name__)

def register_custom_features(feature_engineering, config: Dict[str, Any]) -> None:
    """
    Register custom features based on configuration.
    
    Args:
        feature_engineering: FeatureEngineering instance
        config: Configuration dictionary with custom feature definitions
    """
    # Thread safety with minimized locking - single lock acquisition
    registry_lock = getattr(feature_engineering, 'registry_lock', threading.RLock())
    
    # Extract custom features configuration
    custom_features = config.get("custom_features", {})
    
    if not custom_features:
        logger.info("No custom features configured")
        return
        
    # INITIALIZATION: Single critical section for initial state setup
    with registry_lock:
        # Initialize component registry if needed
        if not hasattr(feature_engineering, 'component_registry'):
            feature_engineering.component_registry = {}
        
        # Pre-populate component registry with all standard features in one operation
        # This ensures all base features are available as components
        for feature_name, feature_func in feature_engineering.feature_registry.items():
            feature_engineering.component_registry[feature_name] = feature_func
            
        # Get current enabled features for reference - use reference (not copy)
        initial_enabled_features = feature_engineering.enabled_features
    
    logger.info(f"Setting up {len(custom_features)} custom features")
    
    # Track successfully registered features
    enabled_count = 0
    enabled_feature_names = []
    
    # Register each custom feature - avoid unnecessary copies
    for feature_name, feature_config in custom_features.items():
        # Skip if disabled in config
        if not feature_config.get("enabled", True):
            continue
            
        # Check if already registered - single lock acquisition
        with registry_lock:
            if feature_name in feature_engineering.feature_registry:
                logger.debug(f"Feature '{feature_name}' already registered, skipping")
                continue
        
        # Skip if no type specified
        feature_type = feature_config.get("type")
        if not feature_type:
            logger.warning(f"Custom feature '{feature_name}' has no type, skipping")
            continue
        
        # Register feature based on type with streamlined error handling
        try:
            # Use specific registration functions optimized for each type
            if feature_type == "weighted_field_similarity":
                _register_weighted_field_similarity(feature_engineering, feature_name, feature_config)
                enabled_count += 1
                enabled_feature_names.append(feature_name)
                
            elif feature_type == "field_match":
                _register_field_match(feature_engineering, feature_name, feature_config)
                enabled_count += 1
                enabled_feature_names.append(feature_name)
                
            elif feature_type == "field_levenshtein":
                _register_field_levenshtein(feature_engineering, feature_name, feature_config)
                enabled_count += 1
                enabled_feature_names.append(feature_name)
                
            elif feature_type == "composite_feature":
                _register_composite_feature(feature_engineering, feature_name, feature_config)
                enabled_count += 1
                enabled_feature_names.append(feature_name)
                
            else:
                logger.warning(f"Unknown feature type '{feature_type}' for '{feature_name}'")
                continue
                
            # Also register in component registry - single update operation
            with registry_lock:
                if feature_name in feature_engineering.feature_registry and feature_name not in feature_engineering.component_registry:
                    feature_engineering.component_registry[feature_name] = feature_engineering.feature_registry[feature_name]
                    
        except Exception as e:
            logger.error(f"Error registering feature '{feature_name}': {str(e)}")
            continue
    
    if enabled_count > 0:
        logger.info(f"Successfully registered {enabled_count} custom features: {', '.join(enabled_feature_names[:5])}{' and more' if len(enabled_feature_names) > 5 else ''}")
    
    # Process and apply substitutions with minimal overhead
    try:
        substitutions = _process_feature_substitutions(config, custom_features)
        if substitutions:
            logger.info(f"Applying {len(substitutions)} feature substitutions")
            _apply_feature_substitutions(feature_engineering, substitutions)
    except Exception as e:
        logger.error(f"Error processing feature substitutions: {str(e)}")
    
    # Minimal verification - only check critical components if in debug mode
    if config.get("debug_custom_features", False):
        try:
            # Only validate component availability for composite features
            debug_components = set()
            missing_components = set()
            
            with registry_lock:
                component_registry_keys = set(feature_engineering.component_registry.keys())
                enabled_features = feature_engineering.enabled_features
                substitution_mapping = getattr(feature_engineering, 'substitution_mapping', {})
            
            # Only check composite features for missing components
            for feature_name, feature_config in custom_features.items():
                if feature_config.get("enabled", True) and feature_config.get("type") == "composite_feature":
                    components = feature_config.get("components", [])
                    debug_components.update(components)
                    
                    # Check for critical missing components
                    for component in components:
                        if component not in component_registry_keys:
                            missing_components.add(component)
            
            if missing_components:
                logger.error(f"Missing components for composite features: {missing_components}")
        except Exception as e:
            logger.warning(f"Error during validation: {str(e)}")
    
    # Final summary log
    logger.info(f"Custom feature setup complete ({enabled_count} features)")

def _process_feature_substitutions(config: Dict[str, Any], 
                                  custom_features: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Process feature substitution configuration.
    
    Args:
        config: Configuration dictionary
        custom_features: Custom features configuration dictionary
        
    Returns:
        Dictionary mapping custom feature names to lists of features they replace
    """
    # Get substitution configuration directly - avoid copying unless necessary
    substitutions_config = config.get("feature_substitutions", {})
    
    if not substitutions_config:
        return {}
    
    # Build substitution map directly - avoid intermediate copies
    substitutions = {}
    for feature_name, subst_config in substitutions_config.items():
        # Only process if the feature exists and is enabled
        if feature_name in custom_features and custom_features[feature_name].get("enabled", True):
            # Get features to replace
            replaces = subst_config.get("replaces", [])
            if replaces:
                substitutions[feature_name] = list(replaces)  # Make a copy only here at point of assignment
                logger.debug(f"Feature '{feature_name}' will substitute: {', '.join(replaces)}")
    
    if substitutions:
        logger.info(f"Processed {len(substitutions)} feature substitutions")
    
    return substitutions

def _apply_feature_substitutions(feature_engineering, substitutions: Dict[str, List[str]]) -> None:
    """
    Apply feature substitutions to enabled features in feature engineering.
    
    Args:
        feature_engineering: FeatureEngineering instance
        substitutions: Dictionary mapping custom feature names to lists of features they replace
    """
    if not substitutions:
        return
    
    # Thread safety with minimized locking
    registry_lock = getattr(feature_engineering, 'registry_lock', threading.RLock())
    
    # Single lock acquisition to get current state
    with registry_lock:
        # Initialize component registry if needed
        if not hasattr(feature_engineering, 'component_registry'):
            feature_engineering.component_registry = {}
        
        # Get current features and registry references
        current_features = set(feature_engineering.enabled_features)
        feature_registry = feature_engineering.feature_registry
    
    logger.info(f"Applying substitutions for features: {', '.join(substitutions.keys())}")
    
    # Prepare substitution data structures without modifying originals
    features_to_remove = set()
    active_substitutions = {}
    components_to_backup = set()
    
    # First pass: identify which features will be substituted and need backup
    for custom_feature, replaces in substitutions.items():
        if custom_feature in current_features:
            replaced_in_this_feature = []
            
            # Check which features will be replaced
            for replaced_feature in replaces:
                if replaced_feature in current_features:
                    features_to_remove.add(replaced_feature)
                    replaced_in_this_feature.append(replaced_feature)
                    # Track features that need backup in component registry
                    components_to_backup.add(replaced_feature)
            
            # Only add to active substitutions if actually replacing something
            if replaced_in_this_feature:
                active_substitutions[custom_feature] = replaced_in_this_feature
                
                # Also add components of composite features to backup list
                if feature_engineering.config.get("custom_features", {}).get(custom_feature, {}).get("type") == "composite_feature":
                    component_list = feature_engineering.config.get("custom_features", {}).get(custom_feature, {}).get("components", [])
                    components_to_backup.update(component_list)
    
    # Backup all required components in a single lock operation
    with registry_lock:
        # Backup all component functions that will be needed
        for component in components_to_backup:
            if component in feature_registry and component not in feature_engineering.component_registry:
                try:
                    feature_engineering.component_registry[component] = feature_registry[component]
                except Exception as e:
                    logger.error(f"Failed to back up {component}: {str(e)}")
        
        # Update substitution mapping in one atomic operation
        feature_engineering.substitution_mapping = active_substitutions.copy() if active_substitutions else {}
    
    # Only update feature registry if we have substitutions to apply
    if features_to_remove:
        # Prepare updated structures without modifying original
        updated_features = [f for f in current_features if f not in features_to_remove]
        
        # Prepare an updated feature registry without copying the whole structure
        updated_feature_registry = {}
        with registry_lock:
            for name, func in feature_registry.items():
                if name not in features_to_remove:
                    updated_feature_registry[name] = func
        
        # Update structures in one atomic operation
        with registry_lock:
            feature_engineering.enabled_features = updated_features
            feature_engineering.feature_registry = updated_feature_registry
            
        logger.info(f"Substituted features: {', '.join(str(x) for x in features_to_remove)}")
    
    # Quick verification - error messages only for actual missing components
    with registry_lock:
        component_registry_keys = set(feature_engineering.component_registry.keys())
        
    missing = components_to_backup - component_registry_keys
    if missing:
        logger.error(f"Missing components after substitution: {missing}")
        logger.error("These missing components may cause zero values - attempting emergency repair")
        
        # Last attempt to recover missing components
        with registry_lock:
            for missing_component in missing:
                if missing_component in feature_registry:
                    feature_engineering.component_registry[missing_component] = feature_registry[missing_component]
                    logger.info(f"Emergency recovery of component: {missing_component}")

def _register_weighted_field_similarity(feature_engineering, feature_name: str, 
                                       feature_config: Dict[str, Any]) -> None:
    """
    Register a weighted field similarity feature.
    
    Args:
        feature_engineering: FeatureEngineering instance
        feature_name: Name of the feature
        feature_config: Feature configuration dictionary
    """
    # Get feature parameters
    field = feature_config.get("field", "title")
    weight = feature_config.get("weight", 1.0)
    power = feature_config.get("power", 1.0)
    
    # Define feature function
    def weighted_field_similarity(left_id: str, right_id: str, 
                                field=field, weight=weight, power=power, **kwargs) -> float:
        # Get vectors
        left_vec = feature_engineering._get_vector(left_id, field)
        right_vec = feature_engineering._get_vector(right_id, field)
        
        # Calculate similarity
        similarity = feature_engineering._cosine_similarity(left_vec, right_vec)
        
        # Apply power and weight
        return (similarity ** power) * weight
        
    # Register feature
    feature_engineering.register_feature(feature_name, weighted_field_similarity)
    logger.info(f"Registered custom weighted_field_similarity feature '{feature_name}'")

def _register_field_match(feature_engineering, feature_name: str, 
                         feature_config: Dict[str, Any]) -> None:
    """
    Register a field exact match feature.
    
    Args:
        feature_engineering: FeatureEngineering instance
        feature_name: Name of the feature
        feature_config: Feature configuration dictionary
    """
    # Get feature parameters
    field = feature_config.get("field", "person")
    weight = feature_config.get("weight", 1.0)
    
    # Define feature function
    def field_match(left_id: str, right_id: str, field=field, weight=weight, **kwargs) -> float:
        # Get string values
        left_str = feature_engineering._get_string_value(left_id, field)
        right_str = feature_engineering._get_string_value(right_id, field)
        
        # Check for exact match
        if left_str == right_str and left_str.strip():
            return 1.0 * weight
        return 0.0
        
    # Register feature
    feature_engineering.register_feature(feature_name, field_match)
    logger.info(f"Registered custom field_match feature '{feature_name}'")

def _register_field_levenshtein(feature_engineering, feature_name: str, 
                               feature_config: Dict[str, Any]) -> None:
    """
    Register a field Levenshtein similarity feature.
    
    Args:
        feature_engineering: FeatureEngineering instance
        feature_name: Name of the feature
        feature_config: Feature configuration dictionary
    """
    # Get feature parameters
    field = feature_config.get("field", "person")
    weight = feature_config.get("weight", 1.0)
    threshold = feature_config.get("threshold", 0.0)
    normalize_names = feature_config.get("normalize_names", False)
    
    # Define feature function
    def field_levenshtein(left_id: str, right_id: str, 
                         field=field, weight=weight, threshold=threshold, 
                         normalize_names=normalize_names, **kwargs) -> float:
        # Get string values
        left_str = feature_engineering._get_string_value(left_id, field)
        right_str = feature_engineering._get_string_value(right_id, field)
        
        # Normalize names if requested
        if normalize_names and field == 'person':
            left_str = feature_engineering.birth_death_extractor.normalize_name(left_str)
            right_str = feature_engineering.birth_death_extractor.normalize_name(right_str)
        
        # Calculate Levenshtein similarity
        similarity = feature_engineering._levenshtein_similarity(left_str, right_str)
        
        # Apply threshold and weight
        if similarity > threshold:
            return similarity * weight
        return 0.0
        
    # Register feature
    feature_engineering.register_feature(feature_name, field_levenshtein)
    logger.info(f"Registered custom field_levenshtein feature '{feature_name}'")

def _register_composite_feature(feature_engineering, feature_name: str, 
                               feature_config: Dict[str, Any]) -> None:
    """
    Register a composite feature that combines other features.
    
    Args:
        feature_engineering: FeatureEngineering instance
        feature_name: Name of the feature
        feature_config: Feature configuration dictionary
    """
    # Get feature parameters - avoid unnecessary copying
    components = feature_config.get("components", [])
    operation = feature_config.get("operation", "multiply")
    weight = feature_config.get("weight", 1.0)
    
    if not components:
        logger.warning(f"Composite feature '{feature_name}' has no components, skipping")
        return
    
    # Check if all component features are available
    available_features = feature_engineering.get_feature_names()
    missing_components = []
    for component in components:
        if component not in available_features:
            missing_components.append(component)
    
    if missing_components:
        logger.warning(f"Component features not available for '{feature_name}': {missing_components}")
        logger.warning(f"Feature '{feature_name}' will be registered but might not work correctly")
    
    # Store a reference to the components list to avoid unnecessary copying
    # Only create a copy if absolutely necessary for safety
    component_list = list(components)
    
    # Define feature function with performance optimizations
    def composite_feature(left_id: str, right_id: str, 
                         operation=operation, weight=weight, **kwargs) -> float:
        # Pre-allocate space for component values - most features have 2-3 components
        component_values = []
        
        # Debug info is only needed if this is the first calculation for this feature/entity pair
        # Avoid overhead of debug tracking unless actually in debug mode
        debug_mode = feature_engineering.config.get("debug_custom_features", False)
        is_first_calc = False
        feature_count = 0

        # Only do expensive debug tracking if in debug mode
        if debug_mode:
            debug_key = f"{feature_name}:{left_id}:{right_id}"
            registry_lock = getattr(feature_engineering, 'registry_lock', threading.RLock())
            
            # Initialize debug tracking only if needed
            with registry_lock:
                if not hasattr(feature_engineering, '_debug_composite'):
                    feature_engineering._debug_composite = {}
                
                # Check if this is the first calculation
                if debug_key not in feature_engineering._debug_composite:
                    is_first_calc = True
                    feature_engineering._debug_composite[debug_key] = True
                    
                    # Count feature calculations only for first few instances
                    debug_keys = list(feature_engineering._debug_composite.keys())
                    for k in debug_keys:
                        if k.startswith(feature_name + ":"):
                            feature_count += 1
            
            # Detailed logging for first few calculations
            if is_first_calc and feature_count <= 5:
                logger.info(f"First calculation of {feature_name} for {left_id} - {right_id}")
                logger.info(f"Components to calculate: {component_list}")
        
        # Pre-fetch component functions for better performance
        component_funcs = {}
        component_sources = {}
        
        # Single lock acquisition to get all component functions at once
        registry_lock = getattr(feature_engineering, 'registry_lock', threading.RLock())
        with registry_lock:
            # Get references to registries without copying - performance optimization
            component_registry = getattr(feature_engineering, 'component_registry', {})
            feature_registry = getattr(feature_engineering, 'feature_registry', {})
            
            # Pre-fetch all component functions in a single lock operation
            for component in component_list:
                # Try component registry first (preferred lookup path)
                if component in component_registry:
                    component_funcs[component] = component_registry[component]
                    component_sources[component] = "component_registry"
                # Fall back to feature registry if needed
                elif component in feature_registry:
                    component_funcs[component] = feature_registry[component]
                    component_sources[component] = "feature_registry"
        
        # Calculate all component values - no locks needed here
        missing_count = 0
        error_count = 0
        
        # Calculate component values using pre-fetched functions
        for component in component_list:
            component_func = component_funcs.get(component)
            
            if component_func is not None:
                try:
                    # Calculate component value with pre-fetched function
                    component_value = component_func(left_id, right_id)
                    component_values.append(component_value)
                    
                    # Only log on first calculation in debug mode
                    if debug_mode and is_first_calc and feature_count <= 5:
                        logger.info(f"  {component} = {component_value} (from {component_sources.get(component)})")
                        
                except Exception as e:
                    if debug_mode:
                        logger.warning(f"Error calculating component value for {component}: {e}")
                    # Use fallback value for error cases
                    component_values.append(0.0)
                    error_count += 1
            else:
                if debug_mode:
                    logger.error(f"Component function for {component} not found in any registry")
                # Use fallback value for missing components
                component_values.append(0.0)
                missing_count += 1
        
        # Performance optimization: early return if all components missing or errors
        if missing_count + error_count >= len(component_list):
            if debug_mode:
                logger.error(f"All {len(component_list)} components for {feature_name} missing or errored")
            return 0.0 * weight
        
        # Calculate result based on operation - optimized for common operations
        result = 0.0
        try:
            if operation == "multiply":
                result = 1.0
                for value in component_values:
                    try:
                        # Fast path for common numeric types
                        if isinstance(value, float):
                            result *= value
                        elif isinstance(value, int):
                            result *= float(value)
                        # Handle NumPy types
                        elif isinstance(value, np.number):
                            result *= float(value.item()) 
                        # Fallback for other types
                        else:
                            result *= float(value)
                    except (TypeError, ValueError):
                        if debug_mode:
                            logger.warning(f"Error in multiply operation - non-numeric value: {value} ({type(value)})")
            
            elif operation == "add":
                for value in component_values:
                    try:
                        if isinstance(value, (int, float, np.number)):
                            result += float(value.item()) if isinstance(value, np.number) else float(value)
                    except (TypeError, ValueError):
                        continue
            
            elif operation == "average":
                # Streamlined average calculation
                sum_values = 0.0
                count = 0
                for value in component_values:
                    try:
                        if isinstance(value, (int, float, np.number)):
                            sum_values += float(value.item()) if isinstance(value, np.number) else float(value)
                            count += 1
                    except (TypeError, ValueError):
                        continue
                result = sum_values / count if count > 0 else 0.0
            
            elif operation == "max" or operation == "min":
                # Collect valid numeric values
                numeric_values = []
                for value in component_values:
                    try:
                        if isinstance(value, (int, float)):
                            numeric_values.append(float(value))
                        elif isinstance(value, np.number):
                            numeric_values.append(float(value.item()))
                    except (TypeError, ValueError):
                        continue
                
                # Apply max or min operation if we have values
                if numeric_values:
                    result = max(numeric_values) if operation == "max" else min(numeric_values)
            
            else:
                if debug_mode:
                    logger.warning(f"Unknown operation '{operation}' for feature '{feature_name}'")
                    
        except Exception as e:
            if debug_mode:
                logger.error(f"Error computing {operation} operation for {feature_name}: {str(e)}")
            # Use safe fallback
            result = 0.0
        
        # Debug diagnostics for zero results only if in debug mode
        if debug_mode and result == 0.0 and any(isinstance(v, (int, float, np.number)) and v > 0 
                                             for v in component_values):
            logger.error(f"Zero result detected with non-zero components for {feature_name}")
            logger.error(f"Operation: {operation}, Values: {component_values}")
        
        # Detailed logging for first few calculations in debug mode
        if debug_mode and is_first_calc and feature_count <= 5:
            logger.info(f"  Operation: {operation}, Result: {result}, Weight: {weight}")
            if result == 0.0:
                logger.warning(f"ZERO RESULT for {feature_name} with values: {component_values}")
        
        # Apply weight and ensure value is in range [0,1]
        final_result = result * weight
        
        # Fast range validation and clamping
        if not isinstance(final_result, (int, float, np.number)) or np.isnan(final_result) or np.isinf(final_result):
            return 0.0  # Default for invalid results
        elif final_result < 0.0:
            return 0.0  # Clamp to minimum
        elif final_result > 1.0:
            return 1.0  # Clamp to maximum
        else:
            return final_result
        
    # Register feature
    feature_engineering.register_feature(feature_name, composite_feature)
    logger.info(f"Registered custom composite feature '{feature_name}' with {len(components)} components")
