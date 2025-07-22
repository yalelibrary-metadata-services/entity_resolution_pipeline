"""
Querying Module for Entity Resolution

This module provides querying capabilities for retrieving entity data from Weaviate.
It handles similarity searches, candidate pair generation, and efficient vector retrieval.
"""

import logging
import time
import random
import os
from typing import Dict, List, Tuple, Any, Optional, Set
import numpy as np
from src.utils import resource_cleanup, safe_json_serialize

logger = logging.getLogger(__name__)

class WeaviateQuerying:
    """
    Class for querying entity data from Weaviate for entity resolution.
    """
    
    def __init__(self, config: Dict[str, Any], weaviate_client: Any, 
                hash_lookup: Dict[str, Dict[str, str]]):
        """
        Initialize the Weaviate querying module.
        
        Args:
            config: Configuration dictionary
            weaviate_client: Initialized Weaviate client
            hash_lookup: Dictionary mapping personId to field hashes
        """
        self.config = config
        self.client = weaviate_client
        self.hash_lookup = hash_lookup
        
        # Load collections
        self.entity_collection = self.client.collections.get("EntityString")
        
        # Configure query parameters
        self.limit = config.get("query_limit", 100)
        self.query_batch_size = config.get("query_batch_size", 20)
        
        # Configure larger cache sizes for better performance
        self.vector_cache_size = config.get("vector_cache_size", 50000)
        self.string_cache_size = config.get("string_cache_size", 100000)
        
        # Initialize basic caches for compatibility
        self.vector_cache = {}  # Cache for vector retrievals
        self.string_cache = {}  # Cache for string retrievals
        
        # Add cache for similar hash lookups for efficient repeated searches
        self.similar_hash_cache = {}
        self.cache_stats = {
            "vector_cache_hits": 0,
            "string_cache_hits": 0,
            "similar_hash_cache_hits": 0,
            "vector_queries": 0,
            "string_queries": 0,
            "similar_hash_queries": 0
        }
        
        # Set default similarity thresholds for different field types
        # FIXED: Use more inclusive threshold (0.3 vector distance = 0.7 similarity)
        # to match the approach in classifying.py and avoid dropping potential matches
        self.similarity_thresholds = {
            'person': config.get("person_similarity_threshold", 0.60),  # Lowered from 0.70
            'title': config.get("title_similarity_threshold", 0.50),    # Lowered from 0.60
            'composite': config.get("composite_similarity_threshold", 0.55)  # Lowered from 0.65
        }
        
        # Track hash to personId mapping for reverse lookups
        self.hash_to_person_id = self._build_hash_to_person_id_mapping()
        
        logger.info("Initialized WeaviateQuerying module")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Return statistics about cache usage."""
        return self.cache_stats.copy()

    def query_similar_entities(self, hash_value: str, field_type: str = 'person',
                              threshold: float = None, limit: int = None,
                              ensure_source_hash: bool = True) -> List[str]:
        """
        Query Weaviate for entities with similar vectors.
        
        Args:
            hash_value: Hash value of the string to query
            field_type: Field type ('person', 'title', or 'composite')
            threshold: Optional similarity threshold, defaults to field-specific threshold
            limit: Optional result limit, defaults to self.limit
            
        Returns:
            List of entity IDs with similar vectors
        """
        if threshold is None:
            threshold = self.similarity_thresholds.get(field_type, 0.70)
            
        if limit is None:
            limit = self.limit
        
        logger.debug(f"Querying similar entities for hash {hash_value} with threshold {threshold}")
        
        # Check cache for similar hash results first
        cache_key = f"{hash_value}_{field_type}_{threshold}_{limit}"
        if cache_key in self.similar_hash_cache:
            self.cache_stats["similar_hash_cache_hits"] += 1
            return self.similar_hash_cache[cache_key].copy()  # Return copy to prevent cache modification
            
        # Track query count
        self.cache_stats["similar_hash_queries"] += 1
        
        try:
            # Get vector for this hash if not in cache
            if hash_value not in self.vector_cache:
                vector = self._get_vector(hash_value)
                if vector is None:
                    logger.warning(f"Vector not found for hash {hash_value}")
                    return []
                self.vector_cache[hash_value] = vector
            else:
                vector = self.vector_cache[hash_value]
            
            # Query similar vectors in Weaviate
            from weaviate.classes.query import Filter, MetadataQuery
            
            # Use near_vector for pure vector similarity search
            result = self.entity_collection.query.near_vector(
                near_vector=vector.tolist(),
                filters=Filter.by_property("field_type").equal(field_type),
                limit=limit,
                return_properties=["hash_value", "original_string", "field_type"],
                return_metadata=MetadataQuery(distance=True),
                include_vector=True
            )
            
            # Process results and extract entity IDs
            similar_entities = []
            for obj in result.objects:
                # Skip if distance is too high (similarity too low)
                if hasattr(obj, 'metadata') and hasattr(obj.metadata, 'distance'):
                    similarity = 1.0 - obj.metadata.distance  # Convert distance to similarity
                    if similarity < threshold:
                        continue
                    
                # Get hash value from properties
                obj_hash = obj.properties.get('hash_value')
                if not obj_hash:
                    continue
                    
                # Convert hash to personId using our lookup
                entity_ids = self.hash_to_person_id.get(obj_hash, [])
                similar_entities.extend(entity_ids)
            
            # CRITICAL FIX: Always ensure the source hash entities are included
            if ensure_source_hash and hash_value:
                source_entities = self.hash_to_person_id.get(hash_value, [])
                if source_entities:
                    # Add source entities if not already included
                    new_additions = [eid for eid in source_entities if eid not in similar_entities]
                    if new_additions:
                        similar_entities.extend(new_additions)
                        logger.debug(f"Added {len(new_additions)} source hash entities to ensure completeness")
            
            # Cache the results
            self.similar_hash_cache[cache_key] = similar_entities.copy()
            
            logger.debug(f"Found {len(similar_entities)} similar entities")
            return similar_entities
            
        except Exception as e:
            logger.error(f"Error querying similar entities: {str(e)}")
            return []
    
    def get_candidate_pairs(self, entity_ids: List[str], 
                           threshold: float = None,
                           ensure_source_hash: bool = True) -> List[Tuple[str, str, None]]:
        """
        Generate candidate pairs for a list of entity IDs.
        
        Args:
            entity_ids: List of entity IDs to generate pairs for
            threshold: Optional similarity threshold, defaults to person similarity threshold
            
        Returns:
            List of (left_id, right_id, None) tuples representing candidate pairs
        """
        if threshold is None:
            threshold = self.similarity_thresholds.get('person', 0.70)
            
        logger.info(f"Generating candidate pairs for {len(entity_ids)} entities with threshold {threshold}")
        start_time = time.time()
        
        # Process in batches to avoid memory issues
        all_candidate_pairs = []
        
        for i in range(0, len(entity_ids), self.query_batch_size):
            batch_ids = entity_ids[i:i+self.query_batch_size]
            logger.debug(f"Processing batch {i//self.query_batch_size + 1}/{(len(entity_ids)-1)//self.query_batch_size + 1}")
            
            # Process each entity in the batch
            batch_pairs = []
            for entity_id in batch_ids:
                # Get the person hash for this entity
                if entity_id not in self.hash_lookup:
                    logger.warning(f"Entity ID {entity_id} not found in hash lookup")
                    continue
                    
                field_hashes = self.hash_lookup[entity_id]
                person_hash = field_hashes.get('person')
                
                if not person_hash:
                    logger.warning(f"No person hash for entity {entity_id}")
                    continue
                
                # Query similar entities - ensure source hash inclusion for completeness
                similar_entities = self.query_similar_entities(
                    person_hash, 'person', threshold, ensure_source_hash=ensure_source_hash
                )
                
                # Create candidate pairs, filtering out self-pairs
                entity_pairs = [
                    (entity_id, similar_id, None)
                    for similar_id in similar_entities
                    if similar_id != entity_id
                ]
                
                batch_pairs.extend(entity_pairs)
            
            all_candidate_pairs.extend(batch_pairs)
        
        # Remove duplicate pairs and ensure consistent ordering
        unique_pairs = set()
        filtered_pairs = []
        
        for left_id, right_id, _ in all_candidate_pairs:
            # Sort entity IDs to ensure consistent ordering
            sorted_ids = sorted([left_id, right_id])
            pair = tuple(sorted_ids)
            
            # CRITICAL FIX: use the sorted order for the filtered pairs too!
            # This ensures pairs are always in the same order (A,B), never (B,A)
            if pair not in unique_pairs:
                unique_pairs.add(pair)
                filtered_pairs.append((sorted_ids[0], sorted_ids[1], None))
        
        elapsed_time = time.time() - start_time
        logger.info(f"Generated {len(filtered_pairs)} unique candidate pairs in {elapsed_time:.2f} seconds")
        
        return filtered_pairs
    
    def get_vectors_batch(self, entity_ids: List[str], field_type: str) -> Dict[str, np.ndarray]:
        """
        Get vectors for multiple entities in a batch.
        
        Args:
            entity_ids: List of entity IDs
            field_type: Field type ('person', 'title', or 'composite')
            
        Returns:
            Dictionary mapping entity ID to vector
        """
        logger.debug(f"Getting {field_type} vectors for {len(entity_ids)} entities")
        
        # Prepare hash values to query
        hash_values = []
        entity_hash_map = {}
        
        for entity_id in entity_ids:
            if entity_id in self.hash_lookup:
                field_hashes = self.hash_lookup[entity_id]
                hash_val = field_hashes.get(field_type)
                
                if hash_val:
                    hash_values.append(hash_val)
                    if hash_val not in entity_hash_map:
                        entity_hash_map[hash_val] = []
                    entity_hash_map[hash_val].append(entity_id)
        
        # Remove hashes that are already cached
        hash_values = [hash_val for hash_val in hash_values if hash_val not in self.vector_cache]
        
        if not hash_values:
            return {entity_id: self.vector_cache.get(self.hash_lookup.get(entity_id, {}).get(field_type))
                   for entity_id in entity_ids if entity_id in self.hash_lookup}
        
        # Query vectors from Weaviate
        vectors = {}
        
        # Process in smaller batches to avoid overwhelming Weaviate
        batch_size = 50
        for i in range(0, len(hash_values), batch_size):
            batch_hashes = hash_values[i:i+batch_size]
            
            # Use dictionary comprehension to get vectors for each hash value
            batch_vectors = {
                hash_val: self._get_vector(hash_val) 
                for hash_val in batch_hashes
            }
            
            # Map vectors to entity IDs
            for hash_val, vector in batch_vectors.items():
                if vector is not None and hash_val in entity_hash_map:
                    for entity_id in entity_hash_map[hash_val]:
                        vectors[entity_id] = vector
        
        logger.debug(f"Retrieved vectors for {len(vectors)} entities")
        return vectors
    
    def get_strings_batch(self, entity_ids: List[str], field_type: str) -> Dict[str, str]:
        """
        Get string values for multiple entities in a batch.
        
        Args:
            entity_ids: List of entity IDs
            field_type: Field type ('person', 'title', etc.)
            
        Returns:
            Dictionary mapping entity ID to string value
        """
        logger.debug(f"Getting {field_type} strings for {len(entity_ids)} entities")
        
        # Prepare hash values to query
        hash_values = []
        entity_hash_map = {}
        
        for entity_id in entity_ids:
            if entity_id in self.hash_lookup:
                field_hashes = self.hash_lookup[entity_id]
                hash_val = field_hashes.get(field_type)
                
                if hash_val:
                    hash_values.append(hash_val)
                    if hash_val not in entity_hash_map:
                        entity_hash_map[hash_val] = []
                    entity_hash_map[hash_val].append(entity_id)
        
        # Remove hashes that are already cached
        hash_values = [hash_val for hash_val in hash_values if hash_val not in self.string_cache]
        
        if not hash_values:
            return {entity_id: self.string_cache.get(self.hash_lookup.get(entity_id, {}).get(field_type))
                   for entity_id in entity_ids if entity_id in self.hash_lookup}
        
        # Query strings from Weaviate
        strings = {}
        
        # Process in smaller batches to avoid overwhelming Weaviate
        batch_size = 50
        for i in range(0, len(hash_values), batch_size):
            batch_hashes = hash_values[i:i+batch_size]
            
            try:
                from weaviate.classes.query import Filter
                from weaviate.util import generate_uuid5
                
                # Process each hash value individually to get its string
                for hash_val in batch_hashes:
                    # Try to look up by field type first - prioritize 'person' field
                    priority_field_type = field_type  # Use the requested field type
                    uuid_input = f"{hash_val}_{priority_field_type}"
                    uuid = generate_uuid5(uuid_input)
                    
                    # Query for this specific hash
                    result = self.entity_collection.query.fetch_objects(
                        filters=Filter.by_id().equal(uuid),
                        include_vector=False
                    )
                    
                    # Process result
                    if result and result.objects and len(result.objects) > 0:
                        obj = result.objects[0]
                        if hasattr(obj, 'properties'):
                            string_val = obj.properties.get('original_string', '')
                            
                            # Cache string
                            self.string_cache[hash_val] = string_val
                            
                            # Map string to entities
                            for entity_id in entity_hash_map[hash_val]:
                                strings[entity_id] = string_val
                
            except Exception as e:
                logger.error(f"Error querying strings batch: {str(e)}")
        
        # Fill in strings from cache for any remaining entities
        for entity_id in entity_ids:
            if entity_id not in strings and entity_id in self.hash_lookup:
                hash_val = self.hash_lookup[entity_id].get(field_type)
                if hash_val in self.string_cache:
                    strings[entity_id] = self.string_cache[hash_val]
        
        logger.debug(f"Retrieved strings for {len(strings)} entities")
        return strings
    
    def clear_caches(self) -> Dict[str, int]:
        """Clear all caches and return previous stats."""
        previous_stats = self.get_cache_stats()
        
        self.vector_cache.clear()
        self.string_cache.clear()
        self.similar_hash_cache.clear()
        
        return previous_stats
        
    def _get_vector(self, hash_value: str) -> Optional[np.ndarray]:
        """
        Get vector for a single hash value.
        
        Args:
            hash_value: Hash value to get vector for
            
        Returns:
            Vector as numpy array or None if not found
        """
        # First check cache to avoid unnecessary database calls
        if hash_value in self.vector_cache:
            self.cache_stats["vector_cache_hits"] += 1
            return self.vector_cache[hash_value]
            
        # Track query count
        self.cache_stats["vector_queries"] += 1
            
        # Implement retry logic with exponential backoff
        max_retries = 5   # Increased from 3
        retry_delay = 2.0  # Increased from 1.0 seconds
        
        for attempt in range(max_retries + 1):
            try:
                # Import necessary utilities
                from weaviate.classes.query import Filter
                from weaviate.util import generate_uuid5
                
                # Try to look up by field type first - prioritize 'person' field
                # This avoids querying for all field types unnecessarily
                priority_field_type = 'person'
                uuid_input = f"{hash_value}_{priority_field_type}"
                uuid = generate_uuid5(uuid_input)
                
                # Query just for the person field type first
                result = self.entity_collection.query.fetch_objects(
                    filters=Filter.by_id().equal(uuid),
                    include_vector=True
                )
                
                # If we don't get a result, try the other field types
                if not (result and result.objects and len(result.objects) > 0):
                    # Try other field types one at a time rather than all at once
                    # This helps avoid overwhelming the server
                    for field_type in ['title', 'composite']:
                        uuid_input = f"{hash_value}_{field_type}"
                        uuid = generate_uuid5(uuid_input)
                        
                        result = self.entity_collection.query.fetch_objects(
                            filters=Filter.by_id().equal(uuid),
                            include_vector=True
                        )
                        
                        if result and result.objects and len(result.objects) > 0:
                            break
                
                # Check if we got any results
                if result and result.objects and len(result.objects) > 0:
                    for obj in result.objects:
                        if hasattr(obj, 'vector') and obj.vector:
                            # Extract vector - handle both dictionary and direct formats
                            if isinstance(obj.vector, dict):
                                # Check for jina_colbert first, then fall back to default
                                if 'jina_colbert' in obj.vector:
                                    vector_data = obj.vector['jina_colbert']
                                elif 'default' in obj.vector:
                                    vector_data = obj.vector['default']
                                else:
                                    # Try to get any available vector
                                    vector_data = next(iter(obj.vector.values())) if obj.vector else None
                                
                                if vector_data is not None:
                                    # Handle both single vectors and multi-vectors (ColBERT)
                                    if isinstance(vector_data, list) and len(vector_data) > 0:
                                        if isinstance(vector_data[0], list):
                                            # Multi-vector format (ColBERT): aggregate for similarity search
                                            multi_vectors = np.array(vector_data, dtype=np.float32)
                                            vector = np.mean(multi_vectors, axis=0)
                                            logger.debug(f"Aggregated {len(vector_data)} token vectors for similarity search")
                                        else:
                                            # Single vector format
                                            vector = np.array(vector_data, dtype=np.float32)
                                    else:
                                        # Direct array format
                                        vector = np.array(vector_data, dtype=np.float32)
                                else:
                                    logger.warning(f"No vector data found in vector dict: {list(obj.vector.keys())}")
                                    return None
                            else:
                                vector = np.array(obj.vector, dtype=np.float32)
                            
                            # Cache the vector
                            self.vector_cache[hash_value] = vector
                            return vector
                
                # If we get here, no results were found
                return None
                
            except Exception as e:
                # Check if this is a timeout error
                error_str = str(e).lower()
                is_timeout = 'deadline' in error_str or 'timeout' in error_str or 'unavailable' in error_str
                
                if attempt < max_retries:
                    # Exponential backoff with jitter
                    wait_time = retry_delay * (2 ** attempt) * (0.5 + 0.5 * random.random())
                    
                    if is_timeout:
                        logger.warning(f"Timeout getting vector for hash {hash_value}. Retrying in {wait_time:.2f}s (attempt {attempt+1}/{max_retries})")
                    else:
                        logger.warning(f"Error getting vector for hash {hash_value}: {str(e)}. Retrying in {wait_time:.2f}s (attempt {attempt+1}/{max_retries})")
                        
                    time.sleep(wait_time)
                else:
                    logger.error(f"Error getting vector for hash {hash_value} after {max_retries} attempts: {str(e)}")
                    return None
            
        return None
    
    def get_entity_info(self, entity_id: str) -> Dict[str, Any]:
        """
        Get complete information for an entity.
        
        Args:
            entity_id: Entity ID to get info for
            
        Returns:
            Dictionary with entity field values and vectors
        """
        if entity_id not in self.hash_lookup:
            logger.warning(f"Entity ID {entity_id} not found in hash lookup")
            return {}
            
        field_hashes = self.hash_lookup[entity_id]
        
        # Collect field values and vectors
        entity_info = {'personId': entity_id, 'field_hashes': field_hashes}
        
        # Get string values
        strings = {}
        for field, hash_val in field_hashes.items():
            if hash_val in self.string_cache:
                strings[field] = self.string_cache[hash_val]
        
        # Get any missing string values
        missing_fields = [field for field in field_hashes if field not in strings]
        if missing_fields:
            for field in missing_fields:
                hash_val = field_hashes[field]
                # Don't query for NULL values
                if hash_val == 'NULL':
                    strings[field] = None
                    continue
                
                # Get string from Weaviate
                try:
                    string_values = self.get_strings_batch([entity_id], field)
                    strings[field] = string_values.get(entity_id)
                except Exception as e:
                    logger.error(f"Error getting string for field {field}: {str(e)}")
                    strings[field] = None
        
        entity_info['strings'] = strings
        
        # Get vectors for embedded fields
        vectors = {}
        for field in ['person', 'title', 'composite']:
            if field in field_hashes:
                hash_val = field_hashes[field]
                
                # Get from cache if available
                if hash_val in self.vector_cache:
                    vectors[field] = self.vector_cache[hash_val]
                else:
                    # Query vector
                    vector = self._get_vector(hash_val)
                    if vector is not None:
                        vectors[field] = vector
                        self.vector_cache[hash_val] = vector
        
        entity_info['vectors'] = vectors
        
        return entity_info
    
    def _build_hash_to_person_id_mapping(self) -> Dict[str, List[str]]:
        """
        Build a mapping from hash values to person IDs for reverse lookups.
        
        Returns:
            Dictionary mapping hash value to list of person IDs
        """
        hash_to_person_id = {}
        
        for person_id, field_hashes in self.hash_lookup.items():
            for field, hash_val in field_hashes.items():
                if field == 'person':
                    if hash_val not in hash_to_person_id:
                        hash_to_person_id[hash_val] = []
                    hash_to_person_id[hash_val].append(person_id)
        
        logger.info(f"Built hash to person ID mapping with {len(hash_to_person_id)} entries")
        return hash_to_person_id

# Module functions
def create_weaviate_querying(config: Dict[str, Any], weaviate_client: Any,
                           hash_lookup: Dict[str, Dict[str, str]]) -> WeaviateQuerying:
    """
    Create and initialize a WeaviateQuerying instance.
    Enhanced with production-ready resource handling.
    
    Args:
        config: Configuration dictionary
        weaviate_client: Initialized Weaviate client
        hash_lookup: Dictionary mapping personId to field hashes
        
    Returns:
        Initialized WeaviateQuerying instance
    """
    # Create the querying instance with proper initialization
    querying = WeaviateQuerying(config, weaviate_client, hash_lookup)
    
    # Load cached vectors if available
    cache_dir = config.get("cache_dir", os.path.join(config.get("checkpoint_dir", "data/checkpoints"), "cache"))
    os.makedirs(cache_dir, exist_ok=True)
    
    vector_cache_path = os.path.join(cache_dir, "vector_cache.json")
    if os.path.exists(vector_cache_path):
        try:
            with open(vector_cache_path, 'r') as f:
                # Load vector cache - convert lists back to numpy arrays
                vector_data = json.load(f)
                querying.vector_cache = {k: np.array(v, dtype=np.float32) 
                                      for k, v in vector_data.items()}
            logger.info(f"Loaded vector cache with {len(querying.vector_cache)} entries")
        except Exception as e:
            logger.warning(f"Error loading vector cache: {str(e)}")
    
    return querying


def persist_querying_caches(querying: WeaviateQuerying, config: Dict[str, Any]) -> bool:
    """
    Persist query caches to disk for faster subsequent startup.
    Uses atomic write pattern for reliability.
    
    Args:
        querying: WeaviateQuerying instance
        config: Configuration dictionary
        
    Returns:
        True if successful, False otherwise
    """
    try:
        cache_dir = config.get("cache_dir", os.path.join(config.get("checkpoint_dir", "data/checkpoints"), "cache"))
        os.makedirs(cache_dir, exist_ok=True)
        
        # Persist vector cache
        vector_cache_path = os.path.join(cache_dir, "vector_cache.json")
        temp_vector_cache_path = f"{vector_cache_path}.tmp"
        
        # Convert numpy arrays to lists for serialization
        vector_data = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                     for k, v in querying.vector_cache.items()}
        
        # Safely serialize with NumPy handling
        with open(temp_vector_cache_path, 'w') as f:
            json.dump(safe_json_serialize(vector_data), f)
        
        # Atomically replace the file
        os.replace(temp_vector_cache_path, vector_cache_path)
        
        # Persist cache stats
        stats_path = os.path.join(cache_dir, "query_cache_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(querying.get_cache_stats(), f)
        
        logger.info(f"Persisted query caches with {len(querying.vector_cache)} vector entries")
        return True
        
    except Exception as e:
        logger.error(f"Error persisting query caches: {str(e)}")
        return False

def main(config_path: str = 'config.yml'):
    """
    Main function for testing the querying module.
    
    Args:
        config_path: Path to the configuration file
    """
    # This function would typically be called from the orchestrator
    # Implemented here for standalone module testing
    pass

if __name__ == "__main__":
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description='Test Weaviate querying')
    parser.add_argument('--config', default='config.yml', help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration with environment-specific overrides
    from src.config_utils import load_config_with_environment
    config = load_config_with_environment(args.config)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run testing
    main(args.config)