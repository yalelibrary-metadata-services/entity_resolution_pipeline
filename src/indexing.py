"""
Indexing Module for Entity Resolution

This module handles the indexing of entity data in Weaviate, enabling efficient
similarity search and candidate pair generation for entity resolution.
"""

import os
import logging
import pickle
import time
import json
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from tqdm import tqdm
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.config import VectorDistances
from weaviate.util import generate_uuid5

logger = logging.getLogger(__name__)

def get_weaviate_client(config: Dict[str, Any]) -> Any:
    """
    Initialize and return a Weaviate client based on configuration.
    
    Args:
        config: Configuration dictionary with Weaviate parameters
        
    Returns:
        Initialized Weaviate client
    """
    # Get Weaviate connection parameters
    weaviate_url = config.get("weaviate_url", "http://localhost:8080")
    
    # Extract host and port information
    import urllib.parse
    parsed_url = urllib.parse.urlparse(weaviate_url)
    
    # Extract host (without protocol)
    host = parsed_url.netloc
    if ':' in host:
        host, port_str = host.split(':', 1)
        port = int(port_str)
    else:
        port = 8080  # Default HTTP port
    
    # Determine if secure connection (HTTPS)
    secure = parsed_url.scheme == 'https'
    
    # Default gRPC port is typically 50051
    grpc_port = config.get("weaviate_grpc_port", 50051)
    
    # Create API key authentication if provided
    auth_client_secret = None
    api_key = config.get("weaviate_api_key")
    if api_key:
        from weaviate.auth import AuthApiKey
        auth_client_secret = AuthApiKey(api_key)
    
    try:
        # Create connection parameters with from_params method
        from weaviate.connect import ConnectionParams
        connection_params = ConnectionParams.from_params(
            http_host=host,
            http_port=port,
            http_secure=secure,
            grpc_host=host,  # Using same host for gRPC
            grpc_port=grpc_port,
            grpc_secure=secure  # Using same security setting for gRPC
        )
        
        # Initialize client
        client = weaviate.WeaviateClient(
            connection_params=connection_params,
            auth_client_secret=auth_client_secret
        )
        
        # Connect to Weaviate
        client.connect()
        
        logger.info(f"Connected to Weaviate at {weaviate_url} (gRPC port: {grpc_port})")
        return client
        
    except Exception as e:
        logger.error(f"Error connecting to Weaviate: {str(e)}")
        raise


def close_weaviate_client(client: Any) -> None:
    """
    Explicitly close Weaviate client connection and release resources.
    
    Args:
        client: Active Weaviate client instance
    """
    try:
        if client is not None:
            client.close()
            logger.info("Weaviate client connection closed successfully")
    except Exception as e:
        logger.error(f"Error closing Weaviate client connection: {str(e)}")


class WeaviateClientManager:
    """
    Context manager for Weaviate client lifecycle management.
    Ensures proper connection closure even during exceptions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with configuration dictionary.
        
        Args:
            config: Configuration dictionary with Weaviate parameters
        """
        self.config = config
        self.client = None
        
    def __enter__(self):
        """
        Establish Weaviate connection when entering context.
        
        Returns:
            Initialized Weaviate client
        """
        self.client = get_weaviate_client(self.config)
        return self.client
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Close Weaviate connection when exiting context.
        
        Args:
            exc_type: Exception type if raised
            exc_val: Exception value if raised
            exc_tb: Exception traceback if raised
        """
        if self.client is not None:
            close_weaviate_client(self.client)
            logger.info("Weaviate client connection closed via context manager")

def create_schema(client: Any, config: Dict[str, Any]) -> None:
    """
    Create Weaviate schema for entity resolution.
    
    Args:
        client: Initialized Weaviate client
        config: Configuration dictionary with schema parameters
    """
    logger.info("Creating Weaviate schema")
    
    # Get index configuration parameters
    ef = config.get("weaviate_ef", 128)
    max_connections = config.get("weaviate_max_connections", 64)
    ef_construction = config.get("weaviate_ef_construction", 128)
    vector_dimensions = config.get("embedding_dimensions", 1536)
    
    try:
        # Check if collection exists, delete if it does
        try:
            client.collections.delete("EntityString")
            logger.info("Deleted existing EntityString collection")
        except Exception:
            logger.info("No existing EntityString collection found")
        
        # Create collection with named vectors
        collection = client.collections.create(
            name="EntityString",
            description="Collection for entity string values with their embeddings",
            vectorizer_config=[
                Configure.NamedVectors.none(
                    name="custom_vector",
                    vector_index_config=Configure.VectorIndex.hnsw(
                        ef=ef,
                        max_connections=max_connections,
                        ef_construction=ef_construction,
                        distance_metric=VectorDistances.COSINE
                    )
                )
            ],
            properties=[
                Property(name="original_string", data_type=DataType.TEXT, skip_vectorization=True),
                Property(name="hash_value", data_type=DataType.TEXT, skip_vectorization=True),
                Property(name="field_type", data_type=DataType.TEXT, skip_vectorization=True),
                Property(name="frequency", data_type=DataType.INT, skip_vectorization=True)
            ]
        )
        
        logger.info(f"Created EntityString collection with named vectors")
        
    except Exception as e:
        logger.error(f"Error creating Weaviate schema: {str(e)}")
        raise

def index_data_in_weaviate(config: Dict[str, Any], client: Any, 
                          string_dict: Dict[str, str]) -> Dict[str, Any]:
    """
    Index entity data in Weaviate.
    
    Args:
        config: Configuration dictionary
        client: Initialized Weaviate client
        string_dict: Dictionary mapping hash to string value
        
    Returns:
        Dictionary with indexing metrics
    """
    logger.info("Starting data indexing in Weaviate")
    start_time = time.time()
    
    # Create schema if needed
    create_schema(client, config)
    
    # Load embeddings
    checkpoint_dir = config.get("checkpoint_dir", "data/checkpoints")
    embedding_cache_path = os.path.join(checkpoint_dir, "embedding_cache.pkl")
    
    if not os.path.exists(embedding_cache_path):
        logger.error(f"Embedding cache not found at {embedding_cache_path}")
        return {'status': 'error', 'message': 'Embedding cache not found'}
    
    try:
        with open(embedding_cache_path, 'rb') as f:
            embedding_cache = pickle.load(f)
            
        logger.info(f"Loaded embedding cache with {len(embedding_cache)} entries")
        
    except Exception as e:
        logger.error(f"Error loading embedding cache: {str(e)}")
        return {'status': 'error', 'message': f'Error loading embedding cache: {str(e)}'}
    
    # Load field hash mapping
    field_hash_mapping_path = os.path.join(checkpoint_dir, "field_hash_mapping.pkl")
    
    if os.path.exists(field_hash_mapping_path):
        try:
            with open(field_hash_mapping_path, 'rb') as f:
                field_hash_mapping = pickle.load(f)
                
            logger.info(f"Loaded field hash mapping with {len(field_hash_mapping)} entries")
            
        except Exception as e:
            logger.error(f"Error loading field hash mapping: {str(e)}")
            field_hash_mapping = {}
    else:
        logger.warning(f"Field hash mapping not found at {field_hash_mapping_path}")
        field_hash_mapping = {}
    
    # Load string counts
    string_counts_path = os.path.join(checkpoint_dir, "string_counts.pkl")
    
    if os.path.exists(string_counts_path):
        try:
            with open(string_counts_path, 'rb') as f:
                string_counts = pickle.load(f)
                
            logger.info(f"Loaded string counts with {len(string_counts)} entries")
            
        except Exception as e:
            logger.error(f"Error loading string counts: {str(e)}")
            string_counts = {}
    else:
        logger.warning(f"String counts not found at {string_counts_path}")
        string_counts = {}
    
    # Get collection
    collection = client.collections.get("EntityString")
    
    # Index data in batches
    batch_size = config.get("weaviate_batch_size", 100)
    total_indexed = 0
    
    # Get embed fields
    embed_fields = config.get("embed_fields", ["composite", "person", "title"])
    
    # Prepare items to index
    items_to_index = []
    for hash_val, string_val in string_dict.items():
        # Skip NULL values
        if hash_val == "NULL" or not string_val:
            continue
            
        # Check if this hash has an embedding
        if hash_val not in embedding_cache:
            continue
            
        # Determine field type(s)
        field_types = []
        if hash_val in field_hash_mapping:
            for field_type, count in field_hash_mapping[hash_val].items():
                if field_type in embed_fields:
                    field_types.append((field_type, count))
        
        # If no specific field type found, skip
        if not field_types:
            continue
            
        # Create separate index entry for each field type
        for field_type, count in field_types:
            items_to_index.append({
                'hash_value': hash_val,
                'original_string': string_val,
                'field_type': field_type,
                'frequency': string_counts.get(hash_val, 1),
                'vector': embedding_cache[hash_val]
            })
    
    logger.info(f"Prepared {len(items_to_index)} items to index")
    
    # Index in batches
    with tqdm(total=len(items_to_index), desc="Indexing items", unit="item", ncols=100,
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
        
        # Performance metrics tracking
        start_batch_time = time.time()
        indexed_count = 0
        error_count = 0
        
        for i in range(0, len(items_to_index), batch_size):
            batch = items_to_index[i:i+batch_size]
            batch_start_time = time.time()
            
            # Log batch start with clear visual separator
            logger.info(f"\n{'='*30} BATCH {i//batch_size + 1}/{(len(items_to_index)-1)//batch_size + 1} {'='*30}")
            logger.info(f"Processing {len(batch)} items ({i+1}-{min(i+len(batch), len(items_to_index))})")
            
            try:
                with collection.batch.dynamic() as batch_writer:
                    for item in batch:
                        # Generate UUID from hash value and field type for idempotency
                        uuid_input = f"{item['hash_value']}_{item['field_type']}"
                        uuid = generate_uuid5(uuid_input)
                        
                        # Remove vector from properties
                        properties = {k: v for k, v in item.items() if k != 'vector'}
                        
                        # Add object with named vector
                        batch_writer.add_object(
                            properties=properties,
                            uuid=uuid,
                            vector={'custom_vector': item['vector'].tolist()}
                        )
                
                # Performance metrics for this batch
                batch_time = time.time() - batch_start_time
                items_per_sec = len(batch) / batch_time
                indexed_count += len(batch)
                total_indexed += len(batch)
                
                # Update progress bar
                pbar.update(len(batch))
                
                # Calculate and display throughput metrics
                elapsed = time.time() - start_batch_time
                overall_rate = indexed_count / elapsed if elapsed > 0 else 0
                progress_pct = (indexed_count / len(items_to_index)) * 100
                
                # Embed metrics in progress log
                logger.info(f"Batch complete: {len(batch)} items in {batch_time:.2f}s ({items_per_sec:.2f} items/sec)")
                logger.info(f"Overall progress: {indexed_count}/{len(items_to_index)} items "
                           f"({progress_pct:.1f}%) at {overall_rate:.2f} items/sec")
                
                # Estimate remaining time
                remaining_items = len(items_to_index) - indexed_count
                est_time_remaining = remaining_items / overall_rate if overall_rate > 0 else 0
                logger.info(f"Estimated time remaining: {est_time_remaining:.1f} seconds")
                
            except Exception as e:
                error_count += 1
                logger.error(f"Error indexing batch: {str(e)}")
                
                # Update progress bar even if there's an error
                pbar.update(len(batch))
                
                # Show error count in progress display
                pbar.set_postfix({"errors": error_count})
                
        # Display final statistics after all batches
        if indexed_count > 0:
            total_time = time.time() - start_batch_time
            final_rate = indexed_count / total_time if total_time > 0 else 0
            logger.info(f"\n{'='*30} INDEXING COMPLETE {'='*30}")
            logger.info(f"Successfully indexed {indexed_count}/{len(items_to_index)} items in {total_time:.2f}s")
            logger.info(f"Average throughput: {final_rate:.2f} items/sec")
            logger.info(f"Error count: {error_count}")
    
    # Get final collection stats
    try:
        result = collection.aggregate.over_all(
            total_count=True
        )
        collection_count = result.total_count
        
    except Exception as e:
        logger.error(f"Error getting collection stats: {str(e)}")
        collection_count = total_indexed
    
    elapsed_time = time.time() - start_time
    logger.info(f"Indexing completed in {elapsed_time:.2f} seconds")
    logger.info(f"Indexed {total_indexed} items, collection has {collection_count} objects")
    
    # Return metrics
    metrics = {
        'status': 'completed',
        'elapsed_time': elapsed_time,
        'total_indexed': total_indexed,
        'collection_count': collection_count
    }
    
    return metrics

def query_weaviate(client: Any, collection_name: str, vector: np.ndarray, 
                 field_type: str = None, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Query Weaviate for similar vectors.
    
    Args:
        client: Initialized Weaviate client
        collection_name: Name of the collection to query
        vector: Query vector
        field_type: Optional field type to filter by
        limit: Maximum number of results to return
        
    Returns:
        List of similar objects
    """
    try:
        # Get collection
        collection = client.collections.get(collection_name)
        
        # Prepare query
        query = collection.query.near_vector(
            near_vector={'vector': vector.tolist()},
            limit=limit
        )
        
        # Add field type filter if provided
        if field_type:
            from weaviate.classes.query import Filter
            query = query.with_where(
                Filter.by_property("field_type").equal(field_type)
            )
        
        # Execute query
        results = query.objects
        
        return results
        
    except Exception as e:
        logger.error(f"Error querying Weaviate: {str(e)}")
        return []

def main(config_path: str = 'config.yml'):
    """
    Main function for indexing data in Weaviate.
    
    Args:
        config_path: Path to the configuration file
    """
    import yaml
    import pickle
    import os
    
    # Load configuration with environment-specific overrides
    from src.config_utils import load_config_with_environment
    config = load_config_with_environment(config_path)
    
    # Load string dictionary
    checkpoint_dir = config.get("checkpoint_dir", "data/checkpoints")
    unique_strings_path = os.path.join(checkpoint_dir, "unique_strings.pkl")
    
    if not os.path.exists(unique_strings_path):
        logger.error(f"Unique strings file not found at {unique_strings_path}")
        return {'status': 'error', 'message': 'Unique strings file not found'}
    
    try:
        with open(unique_strings_path, 'rb') as f:
            string_dict = pickle.load(f)
            
        logger.info(f"Loaded unique strings dictionary with {len(string_dict)} entries")
        
        # Use context manager for Weaviate client
        with WeaviateClientManager(config) as client:
            metrics = index_data_in_weaviate(config, client, string_dict)
            
        return metrics
        
    except Exception as e:
        logger.error(f"Error in indexing main function: {str(e)}")
        return {'status': 'error', 'message': str(e)}

if __name__ == "__main__":
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description='Index data in Weaviate for entity resolution')
    parser.add_argument('--config', default='config.yml', help='Path to configuration file')
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run indexing
    result = main(args.config)
    logger.info(f"Indexing completed with result: {result}")
