"""
Weaviate indexing module for entity resolution pipeline.

This module handles indexing of vector embeddings in Weaviate with batch processing
for efficient handling of large datasets.
"""

import os
import logging
import json
import time
import uuid
from pathlib import Path
from tqdm import tqdm
import weaviate
from weaviate.classes.config import Configure, Property, DataType, VectorDistances
from weaviate.classes.query import MetadataQuery, Filter
from weaviate.util import generate_uuid5
from tenacity import retry, stop_after_attempt, wait_exponential
import numpy as np
import pickle

from src.utils import (
    Timer, get_memory_usage, save_checkpoint, load_checkpoint, MMapDict
)

logger = logging.getLogger(__name__)

class Indexer:
    """
    Handles indexing of vector embeddings in Weaviate for entity resolution.
    
    Features:
    - Efficient schema optimization for vector search
    - Batch indexing with automatic retries
    - Support for named vectors by field type
    - Idempotent operations for re-indexing
    """
    
    def __init__(self, config):
        """
        Initialize the indexer with configuration parameters.
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = config
        
        # Weaviate connection parameters
        self.weaviate_host = config['weaviate']['host']
        self.weaviate_port = config['weaviate']['port']
        self.collection_name = config['weaviate']['collection_name']
        
        # Connect to Weaviate
        self.client = self._connect_to_weaviate()
        
        # Batch size for indexing
        self.batch_size = config['weaviate']['batch_size']
        
        # Weaviate schema parameters
        self.ef = config['weaviate']['ef']
        self.max_connections = config['weaviate']['max_connections']
        self.ef_construction = config['weaviate']['ef_construction']
        self.distance_metric = config['weaviate']['distance_metric']
        
        # Data paths
        self.output_dir = Path(config['system']['output_dir'])
        self.temp_dir = Path(config['system']['temp_dir'])
        self.checkpoint_dir = Path(config['system']['checkpoint_dir'])
        
        # Fields to embed
        self.fields_to_embed = config['embedding']['fields_to_embed']
        
        logger.info("Indexer initialized with Weaviate at %s:%s", 
                   self.weaviate_host, self.weaviate_port)
    
    def execute(self, checkpoint=None):
        """
        Execute indexing of embeddings in Weaviate.
        
        Args:
            checkpoint (str, optional): Path to checkpoint file. Defaults to None.
            
        Returns:
            dict: Indexing results
        """
        # Load checkpoint if provided
        if checkpoint and os.path.exists(checkpoint):
            state = load_checkpoint(checkpoint)
            indexed_hashes = set(state.get('indexed_hashes', []))
            logger.info(f"Resumed indexing from checkpoint: {checkpoint}")
            logger.info(f"Loaded {len(indexed_hashes)} indexed hashes")
        else:
            indexed_hashes = set()
        
        # Check if collection exists and create if needed
        self._create_or_update_schema()
        
        # Load embedded hashes
        embedded_hashes, unique_strings, field_hash_mapping = self._load_data()
        
        logger.info(f"Loaded {len(embedded_hashes)} embedded hashes")
        
        # Filter hashes that haven't been indexed yet
        hashes_to_index = [h for h in embedded_hashes if h not in indexed_hashes]
        
        if not hashes_to_index:
            logger.info("No new hashes to index")
            return {
                'objects_indexed': 0,
                'total_embedded': len(embedded_hashes),
                'completion_percentage': 100.0,
                'duration': 0.0,
                'skipped': True
            }
        
        logger.info(f"Indexing {len(hashes_to_index)} embedded hashes")
        
        if self.config['system']['mode'] == 'dev':
            # In dev mode, limit the number of hashes to index
            dev_sample_size = min(
                self.config['system']['dev_sample_size'],
                len(hashes_to_index)
            )
            hashes_to_index = hashes_to_index[:dev_sample_size]
            logger.info(f"Dev mode: limited to {len(hashes_to_index)} hashes")
        
        # Load embeddings
        embeddings = self._load_embeddings()
        
        # Create batches of objects to index
        batches = self._create_batches(
            hashes_to_index, 
            embeddings, 
            unique_strings, 
            field_hash_mapping
        )
        
        logger.info(f"Created {len(batches)} batches")
        
        # Index objects in batches
        total_indexed = 0
        batch_durations = []
        
        with Timer() as timer:
            # Get the collection
            collection = self._execute_weaviate_operation(
                lambda: self.client.collections.get(self.collection_name)
            )
            
            for batch_idx, batch in enumerate(tqdm(batches, desc="Indexing batches")):
                batch_start = time.time()
                
                try:
                    # Index batch
                    self._index_batch(collection, batch)
                    
                    # Update indexed hashes
                    batch_hashes = [obj['hash'] for obj in batch]
                    indexed_hashes.update(batch_hashes)
                    total_indexed += len(batch)
                    
                    # Record batch duration
                    batch_duration = time.time() - batch_start
                    batch_durations.append(batch_duration)
                    
                    # Log progress
                    if (batch_idx + 1) % 10 == 0:
                        logger.info("Indexed %d/%d batches, %d objects, %.2f seconds/batch", 
                                   batch_idx + 1, len(batches), total_indexed, 
                                   sum(batch_durations[-10:]) / 10)
                        logger.info("Memory usage: %.2f GB", get_memory_usage())
                    
                    # Save checkpoint periodically
                    if self.config['data']['checkpoints_enabled'] and (batch_idx + 1) % 50 == 0:
                        checkpoint_path = self.checkpoint_dir / f"indexing_{batch_idx + 1}.ckpt"
                        save_checkpoint({
                            'indexed_hashes': list(indexed_hashes)
                        }, checkpoint_path)
                
                except Exception as e:
                    logger.error("Error indexing batch %d: %s", batch_idx, str(e))
                    
                    # Save checkpoint on error
                    error_checkpoint = self.checkpoint_dir / f"indexing_error_{batch_idx}.ckpt"
                    save_checkpoint({
                        'indexed_hashes': list(indexed_hashes)
                    }, error_checkpoint)
                    
                    # Continue with next batch after a short delay
                    time.sleep(5)
                    continue
        
        # Save final results
        self._save_results(list(indexed_hashes))
        
        # Get collection statistics
        collection_stats = self._get_collection_stats()
        
        # Calculate completion percentage
        completion_pct = (len(indexed_hashes) / len(embedded_hashes)) * 100 if embedded_hashes else 0
        
        results = {
            'objects_indexed': total_indexed,
            'total_embedded': len(embedded_hashes),
            'completion_percentage': completion_pct,
            'total_in_collection': collection_stats.get('object_count', 0),
            'duration': timer.duration,
            'batch_durations': batch_durations
        }
        
        logger.info("Indexing completed: %d objects indexed (%.2f%%), %.2f seconds",
                   total_indexed, completion_pct, timer.duration)
        
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
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _execute_weaviate_operation(self, operation_func):
        """
        Execute a Weaviate operation with retry logic.
        
        Args:
            operation_func (callable): Function to execute
            
        Returns:
            Any: Result of operation
        """
        try:
            return operation_func()
        except Exception as e:
            logger.error(f"Error executing Weaviate operation: {e}")
            raise
    
    def _create_or_update_schema(self):
        """
        Create or update Weaviate schema for entity resolution.
        """
        try:
            # Check if collection exists
            collections = self._execute_weaviate_operation(
                lambda: self.client.collections.list_all()
            )
            
            if self.collection_name in collections:
                logger.info(f"Collection {self.collection_name} already exists")
                return
            
            logger.info(f"Creating collection {self.collection_name}")
            
            # Create collection with named vectors
            vector_configs = []
            for field_type in self.fields_to_embed:
                vector_configs.append(
                    Configure.NamedVectors.none(
                        name=field_type,
                        vector_index_config=Configure.VectorIndex.hnsw(
                            ef=self.ef,
                            max_connections=self.max_connections,
                            ef_construction=self.ef_construction,
                            distance_metric=VectorDistances.COSINE,
                        )
                    )
                )
            
            # Define properties
            properties = [
                Property(name="hash", data_type=DataType.TEXT, index_filterable=True, 
                        description="Hash of the text value"),
                Property(name="value", data_type=DataType.TEXT, index_filterable=True, 
                        description="Original text value"),
                Property(name="field_type", data_type=DataType.TEXT, index_filterable=True, 
                        description="Type of field (composite, person, title, etc.)"),
                Property(name="frequency", data_type=DataType.NUMBER, index_filterable=True, 
                        description="Frequency of occurrence in the dataset")
            ]
            
            # Create collection
            self._execute_weaviate_operation(
                lambda: self.client.collections.create(
                    name=self.collection_name,
                    description="Entity resolution vectors collection",
                    vectorizer_config=vector_configs,
                    properties=properties
                )
            )
            
            logger.info(f"Created collection {self.collection_name}")
        
        except Exception as e:
            logger.error(f"Error creating Weaviate schema: {e}")
            raise
    
    def _load_data(self):
        """
        Load embedded hashes, unique strings, and field hash mapping.
        
        Returns:
            tuple: (embedded_hashes, unique_strings, field_hash_mapping)
        """
        # Load embedded hashes
        embedded_hashes_path = self.output_dir / "embedded_hashes.json"
        with open(embedded_hashes_path, 'r') as f:
            embedded_hashes = json.load(f)
        
        # Load unique strings
        unique_strings_index_path = self.output_dir / "unique_strings_index.json"
        if unique_strings_index_path.exists():
            # Load from memory-mapped file
            with open(unique_strings_index_path, 'r') as f:
                unique_strings_index = json.load(f)
            
            mmap_path = unique_strings_index['location']
            unique_strings = MMapDict(mmap_path)
            logger.info(f"Loaded unique strings from memory-mapped file: {len(unique_strings)} strings")
        else:
            # Try loading from JSON file
            unique_strings_path = self.output_dir / "unique_strings.json"
            with open(unique_strings_path, 'r') as f:
                unique_strings = json.load(f)
            logger.info(f"Loaded unique strings from JSON: {len(unique_strings)} strings")
        
        # Load field hash mapping
        field_hash_index_path = self.output_dir / "field_hash_index.json"
        if field_hash_index_path.exists():
            # Load from memory-mapped file
            with open(field_hash_index_path, 'r') as f:
                field_hash_index = json.load(f)
            
            mmap_path = field_hash_index['location']
            field_hash_mapping = MMapDict(mmap_path)
            logger.info(f"Loaded field hash mapping from memory-mapped file: {len(field_hash_mapping)} mappings")
        else:
            # Try loading from JSON file
            field_hash_path = self.output_dir / "field_hash_mapping.json"
            with open(field_hash_path, 'r') as f:
                field_hash_mapping = json.load(f)
            logger.info(f"Loaded field hash mapping from JSON: {len(field_hash_mapping)} mappings")
        
        return embedded_hashes, unique_strings, field_hash_mapping
    
    def _load_embeddings(self):
        """
        Load embeddings with support for memory-mapped storage.
        
        Returns:
            dict: Embeddings dictionary
        """
        # Load embedding index
        embedding_index_path = self.output_dir / "embedding_index.json"
        with open(embedding_index_path, 'r') as f:
            embedding_index = json.load(f)
        
        # Load embeddings
        location = embedding_index['location']
        format = embedding_index.get('format', 'mmap')
        
        if format == 'pickle':
            # Load from pickle file
            with open(location, 'rb') as f:
                embeddings = pickle.load(f)
            logger.info(f"Loaded embeddings from pickle: {len(embeddings)} embeddings")
        else:
            # Load from memory-mapped file
            embeddings = MMapDict(location)
            logger.info(f"Loaded embeddings from memory-mapped file: {len(embeddings)} embeddings")
        
        return embeddings
    
    def _create_batches(self, hashes, embeddings, unique_strings, field_hash_mapping):
        """
        Create batches of objects to index.
        
        Args:
            hashes (list): Hashes to index
            embeddings (dict): Embeddings dictionary
            unique_strings (dict): Unique strings dictionary
            field_hash_mapping (dict): Field hash mapping dictionary
            
        Returns:
            list: Batches of objects to index
        """
        batch_size = self.batch_size
        batches = []
        current_batch = []
        
        for hash_value in hashes:
            # Skip if embedding is missing
            if hash_value not in embeddings:
                logger.warning(f"Skipping hash {hash_value}: No embedding found")
                continue
            
            # Skip if not in unique strings
            if hash_value not in unique_strings:
                logger.warning(f"Skipping hash {hash_value}: No unique string found")
                continue
            
            # Skip if not in field hash mapping
            if hash_value not in field_hash_mapping:
                logger.warning(f"Skipping hash {hash_value}: No field hash mapping found")
                continue
            
            # Get string value and embedding
            string_value = unique_strings[hash_value]
            embedding_vector = embeddings[hash_value]
            
            # Get field types and counts
            field_types = field_hash_mapping[hash_value]
            
            # Create an object for each field type
            for field_type, count in field_types.items():
                # Skip if field type is not in fields to embed
                if field_type not in self.fields_to_embed:
                    continue
                
                obj = {
                    'hash': hash_value,
                    'value': string_value,
                    'field_type': field_type,
                    'frequency': count,
                    'vector': embedding_vector
                }
                
                current_batch.append(obj)
                
                # If batch is full, add to batches and start a new one
                if len(current_batch) >= batch_size:
                    batches.append(current_batch)
                    current_batch = []
        
        # Add last batch if not empty
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def _index_batch(self, collection, batch):
        """
        Index a batch of objects in Weaviate.
        
        Args:
            collection (weaviate.Collection): Weaviate collection
            batch (list): Batch of objects to index
        """
        try:
            # Use Weaviate's batch import
            with collection.batch.dynamic() as batch_executor:
                for obj in batch:
                    # Generate UUID from hash and field type for idempotency
                    obj_uuid = generate_uuid5(f"{obj['hash']}_{obj['field_type']}")
                    
                    # Prepare object properties
                    properties = {
                        'hash': obj['hash'],
                        'value': obj['value'],
                        'field_type': obj['field_type'],
                        'frequency': obj['frequency']
                    }
                    
                    # Prepare vector
                    vector = {
                        obj['field_type']: obj['vector']
                    }
                    
                    # Add object to batch
                    batch_executor.add_object(
                        properties=properties,
                        uuid=obj_uuid,
                        vector=vector
                    )
        
        except Exception as e:
            logger.error(f"Error indexing batch: {e}")
            raise
    
    def _get_collection_stats(self):
        """
        Get statistics for the Weaviate collection.
        
        Returns:
            dict: Collection statistics
        """
        try:
            collection = self._execute_weaviate_operation(
                lambda: self.client.collections.get(self.collection_name)
            )
            
            # Get object count
            count_result = self._execute_weaviate_operation(
                lambda: collection.aggregate.over_all(total_count=True)
            )
            
            # Get field type distribution
            from weaviate.classes.aggregate import GroupByAggregate
            
            field_type_result = self._execute_weaviate_operation(
                lambda: collection.aggregate.over_all(
                    group_by=GroupByAggregate(prop="field_type"),
                    total_count=True
                )
            )
            
            field_counts = {}
            for group in field_type_result.groups:
                field_counts[group.grouped_by.value] = group.total_count
            
            stats = {
                'object_count': count_result.total_count,
                'field_counts': field_counts
            }
            
            return stats
        
        except Exception as e:
            logger.error(f"Error getting collection statistics: {e}")
            return {'object_count': 0, 'field_counts': {}}
    
    def _save_results(self, indexed_hashes):
        """
        Save indexing results.
        
        Args:
            indexed_hashes (list): List of indexed hash values
        """
        # Save list of indexed hashes
        with open(self.output_dir / "indexed_hashes.json", 'w') as f:
            json.dump(indexed_hashes, f)
        
        # Save collection statistics
        stats = self._get_collection_stats()
        with open(self.output_dir / "collection_statistics.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Save final checkpoint
        checkpoint_path = self.checkpoint_dir / "indexing_final.ckpt"
        save_checkpoint({
            'indexed_hashes': indexed_hashes
        }, checkpoint_path)
        
        logger.info(f"Indexing results saved to {self.output_dir}")
    
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
