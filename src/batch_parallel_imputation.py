"""
Vector-based imputation module for entity resolution.

This module provides the Imputator class, which handles the imputation of
missing values using vector-based hot deck approach with Weaviate.
"""

import os
import logging
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import weaviate
from weaviate.classes.query import Filter, MetadataQuery
from concurrent.futures import ProcessPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils import (
    Timer, get_memory_usage, save_checkpoint, load_checkpoint
)

from src.mmap_dict import MMapDict

logger = logging.getLogger(__name__)

class Imputator:
    """
    Handles imputation of missing values using vector-based hot deck approach.
    
    Features:
    - Uses vectors to find similar records for imputation
    - Supports multiple imputation methods (average, weighted average, nearest)
    - Configurable similarity threshold and candidate count
    - Batch and parallel processing for efficiency
    - Checkpointing for resuming interrupted imputation
    """
    
    def __init__(self, config):
        """
        Initialize the imputator with configuration parameters.
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = config
        
        # Imputation parameters
        self.fields_to_impute = config['imputation']['fields_to_impute']
        self.similarity_threshold = config['imputation']['vector_similarity_threshold']
        self.max_candidates = config['imputation']['max_candidates']
        self.imputation_method = config['imputation']['imputation_method']
        
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
        
        # Create directories if they don't exist
        for dir_path in [self.output_dir, self.temp_dir, self.checkpoint_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Connect to Weaviate
        self.client = self._connect_to_weaviate()
        
        # Initialize imputed values dictionary
        self.imputed_values = {}
        
        logger.info("Imputator initialized with fields to impute: %s", self.fields_to_impute)
    
    def execute(self, checkpoint=None):
        """
        Execute imputation of missing values.
        
        Args:
            checkpoint (str, optional): Path to checkpoint file. Defaults to None.
            
        Returns:
            dict: Imputation results
        """
        # Load checkpoint if provided
        if checkpoint and os.path.exists(checkpoint):
            state = load_checkpoint(checkpoint)
            self.imputed_values = state.get('imputed_values', {})
            processed_records = set(state.get('processed_records', []))
            logger.info(f"Resumed imputation from checkpoint: {checkpoint}")
            logger.info(f"Loaded {len(self.imputed_values)} imputed values")
            logger.info(f"Loaded {len(processed_records)} processed records")
        else:
            processed_records = set()
        
        # Load record field hashes
        record_field_hashes = self._load_record_field_hashes()
        logger.info(f"Loaded {len(record_field_hashes)} records")
        
        # Identify records with missing values
        records_to_impute = self._identify_missing_values(record_field_hashes)
        logger.info(f"Found {len(records_to_impute)} records with missing values")
        
        # Filter out already processed records
        records_to_impute = {
            record_id: missing_fields
            for record_id, missing_fields in records_to_impute.items()
            if record_id not in processed_records
        }
        
        if not records_to_impute:
            logger.info("No new records to impute")
            return {
                'records_processed': len(processed_records),
                'records_imputed': len(self.imputed_values),
                'fields_imputed': sum(len(fields) for fields in self.imputed_values.values()),
                'duration': 0.0
            }
        
        logger.info(f"Imputing values for {len(records_to_impute)} records")
        
        if self.config['system']['mode'] == 'dev':
            # In dev mode, limit the number of records to impute
            dev_sample_size = min(
                self.config['system']['dev_sample_size'],
                len(records_to_impute)
            )
            record_ids = list(records_to_impute.keys())[:dev_sample_size]
            records_to_impute = {
                record_id: records_to_impute[record_id]
                for record_id in record_ids
            }
            logger.info(f"Dev mode: limited to {len(records_to_impute)} records")
        
        # Create record batches for parallel processing
        record_batches = self._create_record_batches(records_to_impute, record_field_hashes)
        logger.info(f"Created {len(record_batches)} record batches")
        
        # Process record batches
        total_imputed = 0
        total_fields_imputed = 0
        
        with Timer() as timer:
            # Process each batch one at a time instead of using ProcessPoolExecutor
            for batch_idx, batch in enumerate(tqdm(record_batches, desc="Processing batches")):
                try:
                    # Process batch
                    batch_results = self._process_record_batch(batch, record_field_hashes)
                    
                    # Update imputed values
                    for record_id, imputed_fields in batch_results.items():
                        if record_id not in self.imputed_values:
                            self.imputed_values[record_id] = {}
                        
                        self.imputed_values[record_id].update(imputed_fields)
                        
                        # Update processed records
                        processed_records.add(record_id)
                        
                        # Update counters
                        if imputed_fields:
                            total_imputed += 1
                            total_fields_imputed += len(imputed_fields)
                    
                    # Save checkpoint periodically
                    if batch_idx % 10 == 0:
                        checkpoint_path = self.checkpoint_dir / f"imputation_{batch_idx}.ckpt"
                        save_checkpoint({
                            'imputed_values': self.imputed_values,
                            'processed_records': list(processed_records)
                        }, checkpoint_path)
                        
                        # Log progress
                        logger.info(
                            f"Processed {batch_idx + 1}/{len(record_batches)} batches, "
                            f"{total_imputed} records imputed, "
                            f"{total_fields_imputed} fields imputed"
                        )
                
                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx}: {e}")
                    
                    # Save checkpoint on error
                    error_checkpoint = self.checkpoint_dir / f"imputation_error_{batch_idx}.ckpt"
                    save_checkpoint({
                        'imputed_values': self.imputed_values,
                        'processed_records': list(processed_records)
                    }, error_checkpoint)
                    
                    # Continue with next batch
                    continue
        
        # Save final results
        self._save_results(processed_records)
        
        results = {
            'records_processed': len(processed_records),
            'records_imputed': total_imputed,
            'fields_imputed': total_fields_imputed,
            'duration': timer.duration
        }
        
        logger.info(
            f"Imputation completed: {results['records_imputed']} records imputed, "
            f"{results['fields_imputed']} fields, "
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
                client.is_ready()
                logger.info("Connected to Weaviate at %s:%s", 
                           self.weaviate_host, self.weaviate_port)
                
                return client
            
            except Exception as e:
                logger.error(f"Error connecting to Weaviate: {e}")
                raise
        
        return connect_with_retry()
    
    def _load_record_field_hashes(self):
        """
        Load record field hashes.
        
        Returns:
            dict: Record field hashes
        """
        # Load record field hashes
        record_index_path = self.output_dir / "record_index.json"
        if record_index_path.exists():
            # Load from memory-mapped file
            with open(record_index_path, 'r') as f:
                record_index = json.load(f)
            
            mmap_path = record_index['location']
            record_field_hashes = MMapDict(mmap_path)
            logger.info(f"Loaded record field hashes from memory-mapped file: {len(record_field_hashes)} records")
        else:
            # Try loading from JSON file
            record_field_hashes_path = self.output_dir / "record_field_hashes.json"
            with open(record_field_hashes_path, 'r') as f:
                record_field_hashes = json.load(f)
            logger.info(f"Loaded record field hashes from JSON: {len(record_field_hashes)} records")
        
        return record_field_hashes
    
    def _identify_missing_values(self, record_field_hashes):
        """
        Identify records with missing values.
        
        Args:
            record_field_hashes (dict): Record field hashes
            
        Returns:
            dict: Dictionary of record ID -> list of missing fields
        """
        records_to_impute = {}
        null_values = ["NULL", None, "null", "None", "N/A", ""]
        
        # Check each record for missing values in fields to impute
        for record_id, fields in record_field_hashes.items():
            missing_fields = []
            
            for field in self.fields_to_impute:
                # Check if field is missing or has null value
                if field not in fields or fields[field] in null_values:
                    missing_fields.append(field)
            
            if missing_fields:
                records_to_impute[record_id] = missing_fields
        
        # Count how many records have missing values for each field
        field_missing_counts = {}
        for field in self.fields_to_impute:
            field_missing_counts[field] = sum(
                1 for missing_fields in records_to_impute.values()
                if field in missing_fields
            )
        
        logger.info(f"Missing value counts by field: {field_missing_counts}")
        
        return records_to_impute
    
    def _create_record_batches(self, records_to_impute, record_field_hashes):
        """
        Create batches of records for parallel processing.
        
        Args:
            records_to_impute (dict): Records to impute
            record_field_hashes (dict): Record field hashes
            
        Returns:
            list: Batches of record IDs
        """
        record_ids = list(records_to_impute.keys())
        
        # Create batches
        batches = []
        for i in range(0, len(record_ids), self.batch_size):
            batch = record_ids[i:i + self.batch_size]
            batches.append(batch)
        
        return batches
    
    def _process_record_batch(self, batch, record_field_hashes):
        """
        Process a batch of records for imputation.
        
        Args:
            batch (list): Batch of record IDs
            record_field_hashes (dict): Record field hashes
            
        Returns:
            dict: Dictionary of record ID -> imputed fields
        """
        # Don't use self.client here - create a new client in each worker process
        try:
            # Connect to Weaviate for this process
            client = weaviate.connect_to_local(
                host=self.weaviate_host,
                port=self.weaviate_port
            )
            
            # Get the collection
            collection = client.collections.get(self.collection_name)
            
            batch_results = {}
            
            for record_id in batch:
                try:
                    # Get record field hashes
                    field_hashes = record_field_hashes[record_id]
                    
                    # Check if composite field is available
                    if 'composite' not in field_hashes or field_hashes['composite'] == "NULL":
                        logger.warning(f"Record {record_id} missing composite field, skipping")
                        batch_results[record_id] = {}
                        continue
                    
                    # Get composite vector
                    composite_hash = field_hashes['composite']
                    composite_vector = self._get_vector_by_hash(
                        collection, composite_hash, 'composite'
                    )
                    
                    if not composite_vector:
                        logger.warning(f"Failed to get composite vector for record {record_id}")
                        batch_results[record_id] = {}
                        continue
                    
                    # Identify missing fields
                    missing_fields = []
                    for field in self.fields_to_impute:
                        if field not in field_hashes or field_hashes[field] == "NULL":
                            missing_fields.append(field)
                    
                    # Initialize imputed fields
                    imputed_fields = {}
                    
                    # Impute each missing field
                    for field in missing_fields:
                        # Impute value using vector similarity
                        imputed_data = self._impute_field(
                            collection, field, composite_vector
                        )
                        
                        if imputed_data:
                            imputed_fields[field] = imputed_data
                    
                    batch_results[record_id] = imputed_fields
                
                except Exception as e:
                    logger.error(f"Error imputing values for record {record_id}: {e}")
                    batch_results[record_id] = {}
            
            # Close Weaviate client when done
            client.close()
            
            return batch_results
        
        except Exception as e:
            logger.error(f"Error connecting to Weaviate in worker process: {e}")
            return {}
    
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
            return None
    
    def _impute_field(self, collection, field, query_vector):
        """
        Impute value for a field using vector similarity.
        
        Args:
            collection (weaviate.Collection): Weaviate collection
            field (str): Field to impute
            query_vector (list): Query vector (composite field)
            
        Returns:
            dict: Imputed value information or None if not found
        """
        try:
            # Create filter for field type
            field_filter = Filter.by_property("field_type").equal(field)
            
            # Execute search
            results = collection.query.near_vector(
                near_vector=query_vector,
                filters=field_filter,
                target_vector=field,
                limit=self.max_candidates,
                return_metadata=MetadataQuery(distance=True),
                include_vector=True
            )
            
            if not results.objects:
                logger.warning(f"No candidates found for field {field}")
                return None
            
            # Filter candidates by similarity threshold
            candidates = []
            
            for obj in results.objects:
                # Convert distance to similarity (1 - distance)
                similarity = 1.0 - obj.metadata.distance
                
                if similarity >= self.similarity_threshold:
                    candidates.append({
                        'hash': obj.properties['hash'],
                        'value': obj.properties['value'],
                        'vector': obj.vector.get(field),
                        'similarity': similarity
                    })
            
            if not candidates:
                logger.warning(f"No candidates above similarity threshold {self.similarity_threshold} for field {field}")
                return None
            
            # Apply imputation method
            if self.imputation_method == 'nearest':
                # Use nearest neighbor
                imputed = candidates[0]
                imputed_data = {
                    'hash': imputed['hash'],
                    'value': imputed['value'],
                    'similarity': imputed['similarity']
                }
            
            elif self.imputation_method == 'weighted_average':
                # Compute weighted average of vectors
                weights = np.array([c['similarity'] for c in candidates])
                weights = weights / np.sum(weights)  # Normalize
                
                vectors = np.array([c['vector'] for c in candidates])
                imputed_vector = np.average(vectors, axis=0, weights=weights)
                
                # Use value from highest weight candidate
                best_candidate = candidates[np.argmax(weights)]
                
                imputed_data = {
                    'hash': best_candidate['hash'],
                    'value': best_candidate['value'],
                    'similarity': best_candidate['similarity']
                }
            
            else:  # 'average' (default)
                # Compute average of vectors
                vectors = np.array([c['vector'] for c in candidates])
                imputed_vector = np.mean(vectors, axis=0)
                
                # Use most frequent value
                value_counts = {}
                for c in candidates:
                    value = c['value']
                    value_counts[value] = value_counts.get(value, 0) + 1
                
                most_frequent_value = max(value_counts.items(), key=lambda x: x[1])[0]
                most_frequent_hash = next(c['hash'] for c in candidates if c['value'] == most_frequent_value)
                most_frequent_similarity = next(c['similarity'] for c in candidates if c['value'] == most_frequent_value)
                
                imputed_data = {
                    'hash': most_frequent_hash,
                    'value': most_frequent_value,
                    'similarity': most_frequent_similarity
                }
            
            return imputed_data
        
        except Exception as e:
            logger.error(f"Error imputing field {field}: {e}")
            return None
    
    def _save_results(self, processed_records):
        """
        Save imputation results.
        
        Args:
            processed_records (list): List of processed record IDs
        """
        # Save imputed values
        with open(self.output_dir / "imputed_values.json", 'w') as f:
            json.dump(self.imputed_values, f, indent=2)
        
        # Save sample of imputed values
        sample_size = min(100, len(self.imputed_values))
        sample_record_ids = list(self.imputed_values.keys())[:sample_size]
        sample_values = {r: self.imputed_values[r] for r in sample_record_ids}
        
        with open(self.output_dir / "imputed_values_sample.json", 'w') as f:
            json.dump(sample_values, f, indent=2)
        
        # Save statistics
        field_stats = {}
        for record_id, fields in self.imputed_values.items():
            for field in fields:
                if field not in field_stats:
                    field_stats[field] = 0
                
                field_stats[field] += 1
        
        with open(self.output_dir / "imputation_statistics.json", 'w') as f:
            json.dump(field_stats, f, indent=2)
        
        # Save final checkpoint
        checkpoint_path = self.checkpoint_dir / "imputation_final.ckpt"
        save_checkpoint({
            'imputed_values': self.imputed_values,
            'processed_records': list(processed_records)
        }, checkpoint_path)
        
        logger.info(f"Imputation results saved to {self.output_dir}")
    
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
