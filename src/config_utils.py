"""
Configuration Utilities for Entity Resolution Pipeline

This module provides shared configuration loading functionality that ensures
environment-specific settings are applied consistently across all pipeline modules,
whether run through the orchestrator or standalone.
"""

import os
import yaml
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def load_config_with_environment(config_path: str = 'config.yml') -> Dict[str, Any]:
    """
    Load configuration from YAML file with environment-specific overrides.
    
    This function replicates the environment configuration logic from the orchestrator
    to ensure consistent behavior when modules are run standalone.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary with environment-specific overrides applied
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Set default values for required parameters
        config.setdefault('input_dir', 'data/input')
        config.setdefault('output_dir', 'data/output')
        config.setdefault('checkpoint_dir', 'data/checkpoints')
        config.setdefault('ground_truth_dir', 'data/ground_truth')
        config.setdefault('log_dir', 'logs')
        
        # Apply environment-specific configuration
        apply_environment_config(config)
        
        return config
        
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise

def apply_environment_config(config: Dict[str, Any]) -> None:
    """
    Apply environment-specific configuration overrides.
    
    Args:
        config: Configuration dictionary to modify in-place
    """
    # Get environment from environment variable or default to 'local'
    environment = os.environ.get('PIPELINE_ENV', 'local').lower()
    
    if environment not in ['local', 'prod']:
        logger.warning(f"Unknown environment '{environment}', defaulting to 'local'")
        environment = 'local'
    
    logger.info(f"Applying {environment} environment configuration")
    
    # Apply resource configuration
    resource_key = f"{environment}_resources"
    if resource_key in config:
        resource_config = config[resource_key]
        config.update({
            'preprocessing_workers': resource_config.get('preprocessing_workers'),
            'preprocessing_batch_size': resource_config.get('preprocessing_batch_size'),
            'preprocessing_use_optimized': resource_config.get('preprocessing_use_optimized'),
            'embedding_workers': resource_config.get('embedding_workers'),
            'embedding_batch_size': resource_config.get('embedding_batch_size'),
            'embedding_checkpoint_batch': resource_config.get('embedding_checkpoint_batch'),
            'feature_workers': resource_config.get('feature_workers'),
            'feature_batch_size': resource_config.get('feature_batch_size'),
            'classification_workers': resource_config.get('classification_workers'),
            'classification_batch_size': resource_config.get('classification_batch_size')
        })
        logger.info(f"Applied {environment} resource configuration")
    
    # Apply Weaviate configuration
    weaviate_key = f"{environment}_weaviate"
    if weaviate_key in config:
        weaviate_config = config[weaviate_key]
        config.update({
            'weaviate_url': weaviate_config.get('weaviate_url'),
            'weaviate_timeout': weaviate_config.get('weaviate_timeout'),
            'weaviate_batch_size': weaviate_config.get('weaviate_batch_size'),
            'weaviate_ef': weaviate_config.get('weaviate_ef'),
            'weaviate_max_connections': weaviate_config.get('weaviate_max_connections'),
            'weaviate_ef_construction': weaviate_config.get('weaviate_ef_construction'),
            'weaviate_grpc_max_receive_size': weaviate_config.get('weaviate_grpc_max_receive_size'),
            'weaviate_connection_pool_size': weaviate_config.get('weaviate_connection_pool_size'),
            'weaviate_query_concurrent_limit': weaviate_config.get('weaviate_query_concurrent_limit')
        })
        logger.info(f"Applied {environment} Weaviate configuration")
    
    # Apply cluster validation configuration
    cluster_key = f"{environment}_cluster_validation"
    if cluster_key in config and 'cluster_validation' in config:
        cluster_config = config[cluster_key]
        config['cluster_validation'].update(cluster_config)
        logger.info(f"Applied {environment} cluster validation configuration")
    
    # Apply cache configuration
    cache_key = f"{environment}_cache"
    if cache_key in config:
        cache_config = config[cache_key]
        config.update({
            'disable_feature_caching': cache_config.get('disable_feature_caching'),
            'string_cache_size': cache_config.get('string_cache_size'),
            'vector_cache_size': cache_config.get('vector_cache_size'),
            'similarity_cache_size': cache_config.get('similarity_cache_size')
        })
        logger.info(f"Applied {environment} cache configuration")
    
    # Log final resource allocation
    if environment == 'prod':
        logger.info(f"Production mode: Using {config.get('preprocessing_workers', 'N/A')} preprocessing workers, "
                   f"{config.get('feature_workers', 'N/A')} feature workers, "
                   f"{config.get('classification_workers', 'N/A')} classification workers")
    else:
        logger.info(f"Local mode: Using {config.get('preprocessing_workers', 'N/A')} preprocessing workers, "
                   f"{config.get('feature_workers', 'N/A')} feature workers, "
                   f"{config.get('classification_workers', 'N/A')} classification workers")

def get_environment() -> str:
    """
    Get the current pipeline environment.
    
    Returns:
        Environment string ('local' or 'prod')
    """
    return os.environ.get('PIPELINE_ENV', 'local').lower()

def is_production_environment() -> bool:
    """
    Check if running in production environment.
    
    Returns:
        True if running in production mode, False otherwise
    """
    return get_environment() == 'prod'