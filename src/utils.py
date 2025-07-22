"""
Utility functions for the Entity Resolution pipeline.

This module provides a collection of utility functions used throughout the pipeline,
including configuration management, logging setup, and performance monitoring.
"""

import os
import sys
import logging
import time
import yaml
import psutil
import random
import json
import numpy as np
import threading
import contextlib
from typing import Dict, List, Any, Tuple, Optional, Union, Callable, Iterator

# Global seeds lock for thread safety
_GLOBAL_SEED_LOCK = threading.RLock()
# Tracking for seeds to ensure determinism
_seed_initialized = False
# Global registry to track seed usage
_SEED_REGISTRY = {}
# Registry lock
_REGISTRY_LOCK = threading.RLock()


def get_seed_registry():
    """
    Get the current seed registry for debugging and analysis.
    
    Returns:
        Dictionary with seed registry information
    """
    with _REGISTRY_LOCK:
        return dict(_SEED_REGISTRY)

def reset_random_state(seed=None, context="manual_reset"):
    """
    Reset random state to a specific seed.
    Useful for ensuring deterministic behavior at specific points in code.
    
    Args:
        seed: Seed to use (if None, uses default 42)
        context: Context identifier for seed registry tracking
    """
    if seed is None:
        seed = 42
        
    # Reset all random states
    random.seed(seed)
    np.random.seed(seed)
    
    # Register the reset
    with _REGISTRY_LOCK:
        _SEED_REGISTRY[context] = {
            "seed": seed,
            "timestamp": time.time(),
            "thread_id": threading.current_thread().ident,
            "type": "manual_reset"
        }
        
    logging.info(f"Manually reset random state with seed {seed} in context '{context}'")
    
def setup_deterministic_behavior(seed=42, force=False, context="global"):
    """
    Set up deterministic behavior for random operations across the pipeline.
    Enhanced with seed registry and optional force reset to ensure reproducibility.
    
    Args:
        seed: Random seed to use
        force: Force re-initialization even if already initialized
        context: Context identifier for seed registry tracking
    """
    global _seed_initialized, _SEED_REGISTRY
    
    with _GLOBAL_SEED_LOCK:
        # Check if initialization needed or forced
        if not _seed_initialized or force:
            # Set random seeds
            random.seed(seed)
            np.random.seed(seed)
            
            # Set environment variables for other libraries
            os.environ['PYTHONHASHSEED'] = str(seed)
            
            # Set flag to avoid re-initialization
            _seed_initialized = True
            
            # Register the seed usage
            with _REGISTRY_LOCK:
                _SEED_REGISTRY[context] = {
                    "seed": seed,
                    "timestamp": time.time(),
                    "thread_id": threading.current_thread().ident
                }
            
            logging.info(f"Initialized deterministic behavior with seed {seed} in context '{context}'")
            
            # Return true to indicate initialization was performed
            return True
        else:
            # Register seed access without re-initialization
            with _REGISTRY_LOCK:
                if context not in _SEED_REGISTRY:
                    _SEED_REGISTRY[context] = {
                        "seed": seed,
                        "timestamp": time.time(),
                        "thread_id": threading.current_thread().ident,
                        "status": "reused"
                    }
            
            return False

def get_subprocess_seed(base_seed, modifier, context="subprocess"):
    """
    Get a derived seed for subprocesses to ensure deterministic behavior.
    Enhanced with registry tracking for better debugging and reproducibility.
    
    Args:
        base_seed: Base random seed
        modifier: Value to modify the seed (e.g., process ID or task ID)
        context: Context identifier for seed registry tracking
        
    Returns:
        Derived seed value
    """
    # Calculate deterministic derived seed
    derived_seed = (base_seed * 1000 + modifier) % 2147483647  # Max 32-bit integer
    
    # Register the derived seed
    context_key = f"{context}_{modifier}"
    with _REGISTRY_LOCK:
        _SEED_REGISTRY[context_key] = {
            "parent_seed": base_seed,
            "modifier": modifier,
            "derived_seed": derived_seed,
            "timestamp": time.time(),
            "thread_id": threading.current_thread().ident
        }
    
    logging.debug(f"Generated subprocess seed {derived_seed} from base {base_seed} with modifier {modifier}")
    return derived_seed

def setup_logging(config):
    """
    Setup logging configuration.
    
    Args:
        config: Configuration dictionary
    """
    log_level = config.get('log_level', 'INFO')
    log_dir = config.get('log_dir', 'logs')
    
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'pipeline.log')),
            logging.StreamHandler()
        ]
    )
    
    # Set more restrictive log level for noisy libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('weaviate').setLevel(logging.WARNING)

def create_directory_structure(config):
    """
    Create required directory structure.
    
    Args:
        config: Configuration dictionary
    """
    # Create standard directories
    dirs = [
        config.get('input_dir', 'data/input'),
        config.get('output_dir', 'data/output'),
        config.get('checkpoint_dir', 'data/checkpoints'),
        config.get('ground_truth_dir', 'data/ground_truth'),
        config.get('log_dir', 'logs')
    ]
    
    # Create each directory
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
    
    # Create additional subdirectories if specified
    output_dir = config.get('output_dir', 'data/output')
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'reports'), exist_ok=True)

def get_memory_usage():
    """
    Get current memory usage.
    
    Returns:
        Memory usage string
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return f"{memory_info.rss / (1024 * 1024):.2f} MB"

def analyze_feature_configuration(config):
    """
    Analyze feature configuration to determine effective feature set.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with feature analysis results
    """
    # Extract base configuration
    feature_config = config.get("features", {})
    enabled_features = feature_config.get("enabled", [])
    
    # Extract custom feature configuration
    custom_features_config = config.get("custom_features", {})
    enabled_custom_features = []
    
    for name, feature_config in custom_features_config.items():
        if feature_config.get("enabled", True):
            enabled_custom_features.append(name)
    
    # Extract substitution configuration
    substitutions_config = config.get("feature_substitutions", {})
    active_substitutions = {}
    
    for feature_name, subst_config in substitutions_config.items():
        if feature_name in enabled_custom_features:
            replaces = subst_config.get("replaces", [])
            if replaces:
                active_substitutions[feature_name] = replaces
    
    # Determine effective feature set
    substituted_features = []
    for replaces in active_substitutions.values():
        substituted_features.extend(replaces)
    
    effective_features = [f for f in enabled_features if f not in substituted_features]
    effective_features.extend(enabled_custom_features)
    
    return {
        "enabled_features": enabled_features,
        "custom_features": enabled_custom_features,
        "substitutions": active_substitutions,
        "substituted_features": substituted_features,
        "effective_features": effective_features
    }

class NumpyJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles NumPy data types by converting them
    to their Python standard library equivalents.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (set, frozenset)):
            return list(obj)
        
        # Handle NaN values
        try:
            if np.isnan(obj):
                return None
        except (TypeError, ValueError):
            pass
        
        # Handle other NumPy scalar types with .item() method
        try:
            if hasattr(obj, 'item') and callable(obj.item):
                return obj.item()
        except (ValueError, TypeError):
            pass
            
        return super(NumpyJSONEncoder, self).default(obj)


def safe_json_serialize(data: Any) -> Any:
    """
    Recursively convert a dictionary containing NumPy data types to standard Python types.
    
    Args:
        data: The data structure to convert
        
    Returns:
        A data structure with all NumPy types converted to Python standard types
    """
    if isinstance(data, dict):
        return {k: safe_json_serialize(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [safe_json_serialize(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(safe_json_serialize(item) for item in data)
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.bool_):
        return bool(data)
    elif isinstance(data, (set, frozenset)):
        return list(data)
    elif hasattr(data, 'item') and callable(data.item):
        try:
            return data.item()
        except (ValueError, TypeError):
            # If .item() fails, fall back to string representation
            return str(data)
            
    # Handle NaN, Infinity, etc.
    try:
        if np.isnan(data):
            return None
        elif np.isinf(data):
            return float('inf') if data > 0 else float('-inf')
    except (TypeError, ValueError):
        pass
        
    return data


def format_numpy_to_python(array_like: Any) -> Union[List, Dict, int, float, bool, str]:
    """
    Convert NumPy arrays and scalar types to standard Python types.
    Handles multi-dimensional arrays and mixed data structures.
    
    Args:
        array_like: NumPy array, scalar, or mixed Python/NumPy data structure
        
    Returns:
        Equivalent structure with standard Python types
    """
    return safe_json_serialize(array_like)


@contextlib.contextmanager
def acquire_timeout(lock: threading.Lock, timeout: float) -> Iterator[bool]:
    """
    Context manager for acquiring a lock with timeout.
    
    Args:
        lock: Lock to acquire
        timeout: Maximum time to wait for lock acquisition
        
    Yields:
        Boolean indicating whether lock was acquired
    """
    result = lock.acquire(timeout=timeout)
    try:
        yield result
    finally:
        if result:
            lock.release()


@contextlib.contextmanager
def resource_cleanup(cleanup_fn: Callable, *args, **kwargs) -> Iterator[None]:
    """
    Context manager for ensuring resource cleanup.
    
    Args:
        cleanup_fn: Function to call during cleanup
        *args: Arguments to pass to cleanup function
        **kwargs: Keyword arguments to pass to cleanup function
        
    Yields:
        None
    """
    try:
        yield
    finally:
        try:
            cleanup_fn(*args, **kwargs)
        except Exception as e:
            logging.warning(f"Error during resource cleanup: {str(e)}")


def serialize_to_json(data: Any, file_path: str) -> None:
    """
    Serialize data to JSON file with NumPy type handling.
    Uses atomic write pattern for reliability.
    
    Args:
        data: Data to serialize
        file_path: Path to output file
    """
    # Create temp file path
    temp_path = f"{file_path}.tmp"
    
    # Convert NumPy types to standard Python types
    safe_data = safe_json_serialize(data)
    
    # Write to temp file first
    with open(temp_path, 'w') as f:
        json.dump(safe_data, f, indent=2, cls=NumpyJSONEncoder)
    
    # Atomically replace target file
    os.replace(temp_path, file_path)
    

def load_config(config_path):
    """
    Load configuration from YAML file.
    Enhanced with deterministic behavior setup on load.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Add seed configuration if not present
        if 'random_seed' not in config:
            config['random_seed'] = 42
        
        # Set up deterministic behavior immediately to ensure consistency
        # This helps avoid race conditions where code might run before explicit setup
        seed = config['random_seed']
        setup_deterministic_behavior(seed, context="config_load")
        
        return config
    
    except Exception as e:
        print(f"Error loading configuration: {str(e)}")
        sys.exit(1)
