#!/usr/bin/env python3
"""
Create EntityString collection with proper MUVERA configuration using raw API

This module creates a Weaviate collection with Jina ColBERT multi-vector embeddings
compressed via MUVERA. It uses the raw Weaviate API to ensure proper configuration.
"""

import requests
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def create_colbert_collection_with_muvera(
    weaviate_url: str = "http://localhost:8080", 
    muvera_config: Optional[Dict[str, Any]] = None,
    collection_name: str = "EntityString",
    recreate: bool = True,
    hnsw_config: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Create a collection with ColBERT and MUVERA using raw API.
    
    Args:
        weaviate_url: Base URL for Weaviate
        muvera_config: MUVERA configuration dict with ksim, dprojections, repetitions
        collection_name: Name of the collection to create
        recreate: Whether to delete existing collection first
        hnsw_config: Optional HNSW index configuration
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Default MUVERA configuration
    if muvera_config is None:
        muvera_config = {
            'ksim': 8,  # 2^8 = 256 buckets
            'dprojections': 16,  # 16-dimensional sub-vectors
            'repetitions': 1  # Single iteration
        }
    
    # Default HNSW configuration
    if hnsw_config is None:
        hnsw_config = {
            'ef': 128,
            'efConstruction': 128,
            'maxConnections': 64
        }
    
    # Delete existing collection if requested
    if recreate:
        try:
            response = requests.delete(f"{weaviate_url}/v1/schema/{collection_name}")
            if response.status_code in [200, 404]:
                logger.info(f"Deleted existing {collection_name} collection" if response.status_code == 200 
                           else f"No existing {collection_name} collection to delete")
            else:
                logger.warning(f"Unexpected response when deleting collection: {response.status_code}")
        except Exception as e:
            logger.warning(f"Error deleting collection: {e}")
    
    # Validate MUVERA configuration
    if not all(key in muvera_config for key in ['ksim', 'dprojections', 'repetitions']):
        raise ValueError("MUVERA config must contain ksim, dprojections, and repetitions")
    
    # Calculate expected dimensions
    expected_dims = (2 ** muvera_config['ksim']) * muvera_config['dprojections'] * muvera_config['repetitions']
    logger.info(f"MUVERA will compress to {expected_dims} dimensions")
    
    # Create schema with proper MUVERA configuration
    schema = {
        "class": collection_name,
        "description": "Collection for entity string values with ColBERT multi-vector embeddings compressed via MUVERA",
        "vectorConfig": {
            "jina_colbert": {
                "vectorIndexType": "hnsw",
                "vectorIndexConfig": {
                    "distance": "cosine",
                    "ef": hnsw_config['ef'],
                    "efConstruction": hnsw_config['efConstruction'],
                    "maxConnections": hnsw_config['maxConnections'],
                    "multivector": {
                        "enabled": True,
                        "aggregation": "maxSim",
                        "muvera": {
                            "enabled": muvera_config.get('enabled', True),
                            "ksim": muvera_config['ksim'],
                            "dprojections": muvera_config['dprojections'],
                            "repetitions": muvera_config['repetitions']
                        }
                    }
                },
                "vectorizer": {
                    "text2colbert-jinaai": {
                        "model": "jina-colbert-v2",
                        "properties": ["original_string"],
                        "vectorizeClassName": True
                    }
                }
            }
        },
        "properties": [
            {
                "name": "original_string",
                "dataType": ["text"],
                "description": "The original string value being indexed"
            },
            {
                "name": "hash_value",
                "dataType": ["text"],
                "description": "Hash of the original string",
                "moduleConfig": {
                    "text2colbert-jinaai": {
                        "skip": True
                    }
                }
            },
            {
                "name": "field_type",
                "dataType": ["text"],
                "description": "Type of field (person, title, composite, etc.)",
                "moduleConfig": {
                    "text2colbert-jinaai": {
                        "skip": True
                    }
                }
            },
            {
                "name": "frequency",
                "dataType": ["int"],
                "description": "Frequency of this string in the dataset",
                "moduleConfig": {
                    "text2colbert-jinaai": {
                        "skip": True
                    }
                }
            }
        ]
    }
    
    # Create collection
    try:
        response = requests.post(
            f"{weaviate_url}/v1/schema",
            json=schema,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            logger.info(f"{collection_name} collection created successfully with MUVERA")
            
            # Verify MUVERA is enabled
            check_response = requests.get(f"{weaviate_url}/v1/schema/{collection_name}")
            if check_response.status_code == 200:
                created = check_response.json()
                vc = created.get('vectorConfig', {}).get('jina_colbert', {})
                mv = vc.get('vectorIndexConfig', {}).get('multivector', {})
                
                if mv.get('muvera', {}).get('enabled', False):
                    logger.info("✓ MUVERA is ENABLED")
                    logger.info(f"MUVERA config: ksim={mv['muvera']['ksim']}, "
                               f"dprojections={mv['muvera']['dprojections']}, "
                               f"repetitions={mv['muvera']['repetitions']}")
                    logger.info(f"Expected MUVERA dimensions: {expected_dims}")
                    return True
                else:
                    logger.error("✗ MUVERA is DISABLED - configuration may have failed")
                    return False
            else:
                logger.error(f"Failed to verify collection: {check_response.text}")
                return False
        else:
            logger.error(f"Failed to create collection: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"Error creating collection: {e}")
        return False


def validate_muvera_config(muvera_config: Dict[str, Any]) -> None:
    """
    Validate MUVERA configuration parameters.
    
    Args:
        muvera_config: MUVERA configuration to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Check required keys
    required_keys = ['ksim', 'dprojections', 'repetitions']
    for key in required_keys:
        if key not in muvera_config:
            raise ValueError(f"MUVERA config missing required key: {key}")
    
    # Validate ranges
    ksim = muvera_config['ksim']
    if not (1 <= ksim <= 16):
        raise ValueError(f"ksim must be between 1 and 16, got {ksim}")
    
    dprojections = muvera_config['dprojections']
    if not (1 <= dprojections <= 512):
        raise ValueError(f"dprojections must be between 1 and 512, got {dprojections}")
    
    repetitions = muvera_config['repetitions']
    if not (1 <= repetitions <= 100):
        raise ValueError(f"repetitions must be between 1 and 100, got {repetitions}")
    
    # Warn about memory usage
    dims = (2 ** ksim) * dprojections * repetitions
    if dims > 10000:
        logger.warning(f"Large MUVERA dimensions ({dims}) may use significant memory")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example with custom configuration
    custom_muvera = {
        'ksim': 8,  # 256 buckets
        'dprojections': 16,
        'repetitions': 1
    }
    
    success = create_colbert_collection_with_muvera(muvera_config=custom_muvera)
    if success:
        print("Collection created successfully!")
    else:
        print("Failed to create collection")