"""
Querying module for entity resolution pipeline.

This module handles querying Weaviate for match candidates and retrieving
vectors for feature engineering and classification.
"""

import os
import logging
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import weaviate
from weaviate.classes.query import Filter, MetadataQuery
from tenacity import retry, stop_after_attempt, wait_exponential
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.utils import (
    Timer, get_memory_usage, save_checkpoint, load_checkpoint
)
from src.mmap_dict import MMapDict

logger = logging.getLogger(__name__)

class QueryEngine:
    """
    Handles querying Weaviate for match candidates.
    
    Features:
    - Batch and parallel processing of queries
    - Efficient retrieval of vectors for multiple fields
    - Support for ground truth data for training
    - Generation of candidate pairs for classification
    """
    
    def __init__(self, config):
        """
        Initialize the query engine with configuration parameters.
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = config
        
        # Weaviate connection parameters
        self.weaviate_host = config['weaviate']['host']
        self.weaviate_port = config['weaviate']['port']
        self.collection_name = config['weaviate']['collection_name']
        
        # Batch processing parameters
        self.batch_size = config['system']['batch_size']
        self.max_workers = config['system']['max_workers']
        
        # Data paths
        self.output_dir = Path(config['system']['output_dir'])
        self.temp_dir = Path(config['system']['temp_dir'])
        self.checkpoint_dir = Path(config['system']['checkpoint_dir'])
        self.ground_truth_file = Path(config['data']['ground_truth_file'])
        
        # Create directories if they don't exist
        for dir_path in [self.output_dir, self.temp_dir, self.checkpoint_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Connect to Weaviate
        self.client = self._connect_to_weaviate()
        
        # Fields to use for querying
        self.fields_to_embed = config['embedding']['fields_to_embed']
        
        logger.info("QueryEngine initialized")
    
    def execute_ground_truth_queries(self, checkpoint=None):
        """
        Execute queries for ground truth data.
        
        Args:
            checkpoint (str, optional): Path to checkpoint file. Defaults to None.
            
        Returns:
            dict: Query results
        """
        # Load checkpoint if provided
        if checkpoint and os.path.exists(checkpoint):
            state = load_checkpoint(checkpoint)
            record_vectors = state.get('record_vectors', {})
            processed_pairs = set(state.get('processed_pairs', []))
            logger.info(f"Resumed ground truth queries from checkpoint: {checkpoint}")
            logger.info(f"Loaded vectors for {len(record_vectors)} records")
            logger.info(f"Loaded {len(processed_pairs)} processed pairs")
        else:
            record_vectors = {}
            processed_pairs = set()
        
        # Load ground truth data
        match_pairs = self._load_ground_truth()
        logger.info(f"Loaded {len(match_pairs)} ground truth pairs")
        
        # Get unique record IDs from match pairs
        record_ids = set()
        for left_id, right_id, _ in match_pairs:
            record_ids.add(left_id)
            record_ids.add(right_id)
        
        logger.info(f"Found {len(record_ids)} unique records in ground truth")
        
        # Filter out already processed pairs
        pairs_to_process = [
            (left_id, right_id, match)
            for left_id, right_id, match in match_pairs
            if f"{left_id}_{right_id}" not in processed_pairs
        ]
        
        if not pairs_to_process:
            logger.info("No new pairs to process")
            return {
                'records_queried': len(record_ids),
                'pairs_processed': len(processed_pairs),
                'duration': 0.0
            }
        
        logger.info(f"Processing {len(pairs_to_process)} ground truth pairs")
        
        # Load record field hashes
        record_field_hashes = self._load_record_field_hashes()
        logger.info(f"Loaded {len(record_field_hashes)} record field hashes")
        
        # Create batches of record IDs for parallel processing
        record_id_batches = self._create_record_id_batches(list(record_ids))
        logger.info(f"Created {len(record_id_batches)} record ID batches")
        
        # Query vectors for all records in parallel
        with Timer() as timer:
            for batch_idx, batch in enumerate(tqdm(record_id_batches, desc="Querying record batches")):
                try:
                    # Get vectors for records in batch
                    batch_vectors = self._get_vectors_for_records(batch, record_field_hashes)
                    
                    # Update record vectors
                    record_vectors.update(batch_vectors)
                    
                    # Save checkpoint periodically
                    if (batch_idx + 1) % 10 == 0:
                        logger.info(f"Processed {batch_idx + 1}/{len(record_id_batches)} batches, {len(record_vectors)}/{len(record_ids)} records")
                        
                        checkpoint_path = self.checkpoint_dir / f"ground_truth_queries_{batch_idx + 1}.ckpt"
                        save_checkpoint({
                            'record_vectors': record_vectors,
                            'processed_pairs': list(processed_pairs)
                        }, checkpoint_path)
                
                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    
                    # Save checkpoint on error
                    error_checkpoint = self.checkpoint_dir / f"ground_truth_queries_error_{batch_idx}.ckpt"
                    save_checkpoint({
                        'record_vectors': record_vectors,
                        'processed_pairs': list(processed_pairs)
                    }, error_checkpoint)
                    
                    # Continue with next batch
                    continue
            
            # Process all pairs now that we have the vectors
            logger.info(f"Processing {len(pairs_to_process)} pairs")
            
            # Mark all pairs as processed
            for left_id, right_id, _ in pairs_to_process:
                processed_pairs.add(f"{left_id}_{right_id}")
            
            # Create pair vectors
            pair_vectors = []
            
            for left_id, right_id, match in tqdm(match_pairs, desc="Creating pair vectors"):
                # Skip if either record is missing vectors
                if left_id not in record_vectors or right_id not in record_vectors:
                    logger.warning(f"Missing vectors for pair {left_id}_{right_id}")
                    continue
                
                # Get vectors for both records
                left_vectors = record_vectors[left_id]
                right_vectors = record_vectors[right_id]
                
                # Get field hashes
                left_hashes = {}
                right_hashes = {}
                
                if left_id in record_field_hashes:
                    left_hashes = record_field_hashes[left_id]
                
                if right_id in record_field_hashes:
                    right_hashes = record_field_hashes[right_id]
                
                # Create pair vectors
                pair_vector = {
                    'left_id': left_id,
                    'right_id': right_id,
                    'left_vectors': left_vectors,
                    'right_vectors': right_vectors,
                    'hashes': {
                        'left': left_hashes,
                        'right': right_hashes
                    },
                    'match': match
                }
                
                pair_vectors.append(pair_vector)
        
        # Save final results
        self._save_ground_truth_results(record_vectors, pair_vectors, processed_pairs)
        
        results = {
            'records_queried': len(record_vectors),
            'pairs_processed': len(processed_pairs),
            'pair_vectors_created': len(pair_vectors),
            'duration': timer.duration
        }
        
        logger.info(
            f"Ground truth queries completed: {results['records_queried']} records, "
            f"{results['pairs_processed']} pairs, "
            f"{timer.duration:.2f} seconds"
        )
        
        return results
    
    def _connect_to_weaviate(self):
        """
        Connect to Weaviate instance with retry logic.
        
        Returns:
            weaviate.Client: Weaviate client
        """
        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
        def connect_with_retry():
            try:
                # Connect to Weaviate
                client = weaviate.connect_to_local(
                    host=self.weaviate_host,
                    port=self.weaviate_port
                )
                
                # Test connection
                is_ready = client.is_ready()
                if is_ready:
                    logger.info("Connected to Weaviate at %s:%s", 
                              self.weaviate_host, self.weaviate_port)
                else:
                    logger.warning("Weaviate is not ready")
                
                return client
            
            except Exception as e:
                logger.error(f"Error connecting to Weaviate: {e}")
                raise
        
        try:
            return connect_with_retry()
        except Exception as e:
            logger.error(f"Could not connect to Weaviate after retries: {e}")
            logger.warning("Continuing with limited functionality. Vector operations will fail.")
            return None
    
    def _load_ground_truth(self):
        """
        Load ground truth data from file.
        
        Returns:
            list: List of (left_id, right_id, match) tuples
        """
        match_pairs = []
        
        try:
            with open(self.ground_truth_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 3:
                        left_id = parts[0]
                        right_id = parts[1]
                        match = parts[2].lower() == 'true'
                        match_pairs.append((left_id, right_id, match))
        except FileNotFoundError:
            logger.error(f"Ground truth file not found: {self.ground_truth_file}")
        except Exception as e:
            logger.error(f"Error loading ground truth data: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        return match_pairs
    
    def _load_record_field_hashes(self):
        """
        Load record field hashes.
        
        Returns:
            dict: Record field hashes
        """
        # Load record field hashes from index
        record_index_path = self.output_dir / "record_index.json"
        if record_index_path.exists():
            try:
                # Load from memory-mapped file
                with open(record_index_path, 'r') as f:
                    record_index = json.load(f)
                
                location = record_index.get('location')
                if location and os.path.exists(location):
                    record_field_hashes = MMapDict(location)
                    logger.info(f"Loaded record field hashes from memory-mapped file: {len(record_field_hashes)} records")
                    return record_field_hashes
            except Exception as e:
                logger.error(f"Error loading record field hashes from index: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Fall back to JSON file
        record_field_hashes_path = self.output_dir / "record_field_hashes.json"
        if record_field_hashes_path.exists():
            try:
                with open(record_field_hashes_path, 'r') as f:
                    record_field_hashes = json.load(f)
                logger.info(f"Loaded record field hashes from JSON: {len(record_field_hashes)} records")
                return record_field_hashes
            except Exception as e:
                logger.error(f"Error loading record field hashes from JSON: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Fall back to sample
        record_field_hashes_sample_path = self.output_dir / "record_field_hashes_sample.json"
        if record_field_hashes_sample_path.exists():
            try:
                with open(record_field_hashes_sample_path, 'r') as f:
                    record_field_hashes = json.load(f)
                logger.warning(f"Loaded SAMPLE record field hashes: {len(record_field_hashes)} records")
                return record_field_hashes
            except Exception as e:
                logger.error(f"Error loading record field hashes sample: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        logger.warning("Could not load record field hashes")
        return {}
    
    def _create_record_id_batches(self, record_ids):
        """
        Create batches of record IDs for parallel processing.
        
        Args:
            record_ids (list): List of record IDs
            
        Returns:
            list: Batches of record IDs
        """
        # Create batches
        batches = []
        for i in range(0, len(record_ids), self.batch_size):
            batch = record_ids[i:i + self.batch_size]
            batches.append(batch)
        
        return batches
    
    def _get_vectors_for_records(self, record_ids, record_field_hashes):
        """
        Get vectors for a batch of records.
        
        Args:
            record_ids (list): List of record IDs
            record_field_hashes (dict): Record field hashes
            
        Returns:
            dict: Dictionary of record ID -> field vectors
        """
        # Create a process pool for parallel processing
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit jobs
            future_to_record = {
                executor.submit(
                    self._get_vectors_for_record,
                    record_id,
                    record_field_hashes.get(record_id, {})
                ): record_id
                for record_id in record_ids
            }
            
            # Process results as they complete
            record_vectors = {}
            for future in as_completed(future_to_record):
                record_id = future_to_record[future]
                
                try:
                    field_vectors = future.result()
                    record_vectors[record_id] = field_vectors
                
                except Exception as e:
                    logger.error(f"Error getting vectors for record {record_id}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
        
        return record_vectors
    
    def _get_vectors_for_record(self, record_id, field_hashes):
        """
        Get vectors for a single record.
        
        Args:
            record_id (str): Record ID
            field_hashes (dict): Field hashes
            
        Returns:
            dict: Dictionary of field -> vector
        """
        if not self.client:
            logger.warning(f"Weaviate client not available, cannot get vectors for record {record_id}")
            return {}
            
        # Connect to Weaviate for this process
        client = None
        try:
            client = weaviate.connect_to_local(
                host=self.weaviate_host,
                port=self.weaviate_port
            )
        except Exception as e:
            logger.error(f"Error connecting to Weaviate in process: {e}")
            return {}
        
        if not client:
            logger.warning(f"Could not connect to Weaviate in process for record {record_id}")
            return {}
        
        # Get the collection
        collection = None
        try:
            collection = client.collections.get(self.collection_name)
        except Exception as e:
            logger.error(f"Error getting collection: {e}")
            client.close()
            return {}
        
        if not collection:
            logger.warning(f"Collection {self.collection_name} not found")
            client.close()
            return {}
        
        field_vectors = {}
        
        # Query vectors for each field
        for field in self.fields_to_embed:
            if field in field_hashes and field_hashes[field] != "NULL":
                hash_value = field_hashes[field]
                vector = self._get_vector_by_hash(collection, hash_value, field)
                
                if vector is not None:
                    field_vectors[field] = vector
        
        # Close Weaviate client
        client.close()
        
        return field_vectors
    
    def _get_vector_by_hash(self, collection, hash_value, field_type):
        """
        Get vector for a hash value and field type.
        
        Args:
            collection (weaviate.Collection): Weaviate collection
            hash_value (str): Hash value
            field_type (str): Field type
            
        Returns:
            list: Vector or None if not found
        """
        try:
            # Create filter for hash and field type
            hash_filter = Filter.by_property("hash").equal(hash_value)
            field_filter = Filter.by_property("field_type").equal(field_type)
            combined_filter = Filter.all_of([hash_filter, field_filter])
            
            # Execute search
            results = collection.query.fetch_objects(
                filters=combined_filter,
                limit=1,
                include_vector=True
            )
            
            if results.objects:
                # Extract vector
                return results.objects[0].vector.get(field_type)
            
            return None
        
        except Exception as e:
            logger.error(f"Error getting vector for hash {hash_value}, field {field_type}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _save_ground_truth_results(self, record_vectors, pair_vectors, processed_pairs):
        """
        Save ground truth query results.
        
        Args:
            record_vectors (dict): Record vectors
            pair_vectors (list): Pair vectors
            processed_pairs (set): Processed pairs
        """
        # Save record vectors
        record_vectors_path = self.temp_dir / "record_vectors.pkl"
        with open(record_vectors_path, 'wb') as f:
            import pickle
            pickle.dump(record_vectors, f)
        
        # Save pair vectors
        pair_vectors_path = self.temp_dir / "pair_vectors.pkl"
        with open(pair_vectors_path, 'wb') as f:
            import pickle
            pickle.dump(pair_vectors, f)
        
        # Save metadata
        with open(self.output_dir / "ground_truth_query_metadata.json", 'w') as f:
            json.dump({
                'record_vectors_path': str(record_vectors_path),
                'pair_vectors_path': str(pair_vectors_path),
                'record_count': len(record_vectors),
                'pair_count': len(pair_vectors),
                'processed_pairs': len(processed_pairs)
            }, f, indent=2)
        
        # Save final checkpoint
        checkpoint_path = self.checkpoint_dir / "ground_truth_queries_final.ckpt"
        save_checkpoint({
            'record_vectors': record_vectors,
            'processed_pairs': list(processed_pairs)
        }, checkpoint_path)
        
        logger.info(f"Ground truth query results saved to {self.output_dir}")
    
    def __del__(self):
        """
        Clean up resources when object is garbage collected.
        """
        try:
            # Close Weaviate client if it exists
            if hasattr(self, 'client') and self.client:
                logger.info("Closing Weaviate client connection")
                self.client.close()
                self.client = None
        except Exception as e:
            logger.error(f"Error closing Weaviate client: {e}")
