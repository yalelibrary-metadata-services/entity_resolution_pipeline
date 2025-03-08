"""
Embedding module for entity resolution pipeline.

This module generates embeddings for unique strings using OpenAI's API
with efficient batch processing and rate limiting.
"""

import os
import logging
import json
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm
import openai
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import (
    retry, stop_after_attempt, wait_exponential, retry_if_exception_type
)

from src.utils import (
    Timer, get_memory_usage, save_checkpoint, load_checkpoint, MMapDict
)

logger = logging.getLogger(__name__)

class Embedder:
    """
    Handles embedding generation for entity resolution.
    
    Features:
    - Batch processing of embeddings
    - Efficient rate limiting
    - Retry logic for API failures
    - Checkpointing for resumable processing
    """
    
    def __init__(self, config):
        """
        Initialize the embedder with configuration parameters.
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = config
        
        # Data paths
        self.output_dir = Path(config['system']['output_dir'])
        self.temp_dir = Path(config['system']['temp_dir'])
        self.checkpoint_dir = Path(config['system']['checkpoint_dir'])
        
        # Create directories if they don't exist
        for dir_path in [self.output_dir, self.temp_dir, self.checkpoint_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # OpenAI API parameters
        self.model = config['embedding']['model']
        self.dimensions = config['embedding']['dimensions']
        self.batch_size = config['embedding']['batch_size']
        self.request_timeout = config['embedding']['request_timeout']
        self.retry_attempts = config['embedding']['retry_attempts']
        self.retry_delay = config['embedding']['retry_delay']
        
        # Rate limiting parameters
        self.rpm_limit = config['embedding']['rpm_limit']
        self.tpm_limit = config['embedding']['tpm_limit']
        self.tokens_per_minute = 0
        self.requests_per_minute = 0
        self.minute_start = time.time()
        
        # Fields to embed
        self.fields_to_embed = config['embedding']['fields_to_embed']
        
        # Initialize OpenAI client
        self.client = OpenAI()
        
        # Initialize embeddings dictionary
        self.embeddings = {}
        
        logger.info("Embedder initialized with model: %s", self.model)
    
    def execute(self, checkpoint=None):
        """
        Execute embedding generation for unique strings.
        
        Args:
            checkpoint (str, optional): Path to checkpoint file. Defaults to None.
            
        Returns:
            dict: Embedding results
        """
        # Load checkpoint if provided
        if checkpoint and os.path.exists(checkpoint):
            state = load_checkpoint(checkpoint)
            self.embeddings = state.get('embeddings', {})
            processed_hashes = set(state.get('processed_hashes', []))
            logger.info(f"Resumed embedding from checkpoint: {checkpoint}")
            logger.info(f"Loaded {len(self.embeddings)} embeddings")
            logger.info(f"Loaded {len(processed_hashes)} processed hashes")
        else:
            processed_hashes = set()
        
        # Load unique strings
        unique_strings, field_hash_mapping = self._load_data()
        
        # Filter strings to embed (only those in fields_to_embed)
        hashes_to_embed = set()
        for hash_value, field_counts in field_hash_mapping.items():
            for field in field_counts:
                if field in self.fields_to_embed:
                    hashes_to_embed.add(hash_value)
                    break
        
        logger.info(f"Found {len(hashes_to_embed)} unique strings to embed")
        
        # Filter out already processed hashes
        hashes_to_embed = hashes_to_embed - processed_hashes
        
        if not hashes_to_embed:
            logger.info("No new strings to embed")
            # Save results for already processed strings
            self._save_results(list(processed_hashes))
            
            return {
                'total_embeddings': len(processed_hashes),
                'new_embeddings': 0,
                'batch_count': 0,
                'token_count': 0,
                'request_count': 0,
                'duration': 0.0
            }
        
        logger.info(f"Embedding {len(hashes_to_embed)} unique strings")
        
        if self.config['system']['mode'] == 'dev':
            # In dev mode, limit the number of strings to embed
            dev_sample_size = min(
                self.config['system']['dev_sample_size'],
                len(hashes_to_embed)
            )
            hashes_to_embed = set(list(hashes_to_embed)[:dev_sample_size])
            logger.info(f"Dev mode: limited to {len(hashes_to_embed)} strings")
        
        # Create batches of strings to embed
        hash_list = list(hashes_to_embed)
        batches = [
            hash_list[i:i + self.batch_size]
            for i in range(0, len(hash_list), self.batch_size)
        ]
        
        logger.info(f"Created {len(batches)} batches")
        
        # Initialize counters
        total_embedded = 0
        total_tokens = 0
        total_requests = 0
        batch_durations = []
        
        with Timer() as timer:
            for batch_idx, batch in enumerate(tqdm(batches, desc="Embedding batches")):
                batch_start = time.time()
                
                try:
                    # Get strings for batch
                    batch_strings = {
                        hash_value: unique_strings[hash_value]
                        for hash_value in batch
                    }
                    
                    # Generate embeddings for batch
                    batch_embeddings, token_count = self._generate_embeddings(
                        list(batch_strings.keys()),
                        list(batch_strings.values())
                    )
                    
                    # Update embeddings dictionary
                    self.embeddings.update(batch_embeddings)
                    
                    # Update processed hashes
                    processed_hashes.update(batch)
                    
                    # Update counters
                    total_embedded += len(batch)
                    total_tokens += token_count
                    total_requests += 1
                    
                    # Record batch duration
                    batch_duration = time.time() - batch_start
                    batch_durations.append(batch_duration)
                    
                    # Log progress
                    if (batch_idx + 1) % 10 == 0:
                        logger.info("Embedded %d/%d batches, %d strings, %.2f seconds/batch", 
                                   batch_idx + 1, len(batches), total_embedded, 
                                   sum(batch_durations[-10:]) / 10)
                        logger.info("Memory usage: %.2f GB", get_memory_usage())
                    
                    # Save checkpoint periodically
                    if self.config['data']['checkpoints_enabled'] and (batch_idx + 1) % 50 == 0:
                        checkpoint_path = self.checkpoint_dir / f"embedding_{batch_idx + 1}.ckpt"
                        
                        # For large embeddings, save to memory-mapped file
                        if len(self.embeddings) > 100000:
                            embeddings_mmap = MMapDict(self.temp_dir / "embeddings.mmap", mode='w+')
                            for h, v in self.embeddings.items():
                                embeddings_mmap[h] = v
                            
                            save_checkpoint({
                                'embeddings_location': str(self.temp_dir / "embeddings.mmap"),
                                'processed_hashes': list(processed_hashes)
                            }, checkpoint_path)
                        else:
                            save_checkpoint({
                                'embeddings': self.embeddings,
                                'processed_hashes': list(processed_hashes)
                            }, checkpoint_path)
                
                except Exception as e:
                    logger.error("Error embedding batch %d: %s", batch_idx, str(e))
                    
                    # Save checkpoint on error
                    error_checkpoint = self.checkpoint_dir / f"embedding_error_{batch_idx}.ckpt"
                    save_checkpoint({
                        'embeddings': self.embeddings,
                        'processed_hashes': list(processed_hashes)
                    }, error_checkpoint)
                    
                    # Continue with next batch after a short delay
                    time.sleep(5)
                    continue
        
        # Save final results
        self._save_results(list(processed_hashes))
        
        results = {
            'total_embeddings': len(processed_hashes),
            'new_embeddings': total_embedded,
            'batch_count': len(batches),
            'token_count': total_tokens,
            'request_count': total_requests,
            'duration': timer.duration,
            'batch_durations': batch_durations
        }
        
        logger.info("Embedding completed: %d total embeddings, %d new, %.2f seconds",
                   results['total_embeddings'], results['new_embeddings'], timer.duration)
        
        return results
    
    def _load_data(self):
        """
        Load unique strings and field hash mapping.
        
        Returns:
            tuple: (unique_strings, field_hash_mapping)
        """
        unique_strings = {}
        field_hash_mapping = {}
        
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
            if unique_strings_path.exists():
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
            if field_hash_path.exists():
                with open(field_hash_path, 'r') as f:
                    field_hash_mapping = json.load(f)
                logger.info(f"Loaded field hash mapping from JSON: {len(field_hash_mapping)} mappings")
        
        return unique_strings, field_hash_mapping
    
    @retry(
        retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        before_sleep=lambda retry_state: logger.info(f"Retrying after error: {retry_state.outcome.exception()}")
    )
    def _generate_embeddings(self, hash_values, strings):
        """
        Generate embeddings for a batch of strings with rate limiting.
        
        Args:
            hash_values (list): List of hash values
            strings (list): List of strings to embed
            
        Returns:
            tuple: (embeddings_dict, token_count)
        """
        # Check rate limits
        self._check_rate_limits()
        
        # Generate embeddings
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=strings,
                timeout=self.request_timeout
            )
            
            # Extract embeddings
            embeddings = {}
            for i, embedding_data in enumerate(response.data):
                hash_value = hash_values[i]
                embeddings[hash_value] = np.array(embedding_data.embedding, dtype=np.float32)
            
            # Update rate limit counters
            token_count = response.usage.total_tokens
            self.tokens_per_minute += token_count
            self.requests_per_minute += 1
            
            # Check if minute has elapsed for rate limiting
            current_time = time.time()
            if current_time - self.minute_start >= 60:
                self.tokens_per_minute = 0
                self.requests_per_minute = 0
                self.minute_start = current_time
            
            return embeddings, token_count
        
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def _check_rate_limits(self):
        """
        Check and enforce API rate limits.
        """
        # Reset counters if minute has elapsed
        current_time = time.time()
        if current_time - self.minute_start >= 60:
            self.tokens_per_minute = 0
            self.requests_per_minute = 0
            self.minute_start = current_time
            return
        
        # Check if we're approaching rate limits
        if self.tokens_per_minute > self.tpm_limit * 0.95:
            # Sleep until the next minute starts
            sleep_time = 60 - (current_time - self.minute_start)
            logger.info(f"Approaching token rate limit, sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time + 1)  # Add 1 second buffer
            
            # Reset counters
            self.tokens_per_minute = 0
            self.requests_per_minute = 0
            self.minute_start = time.time()
        
        if self.requests_per_minute > self.rpm_limit * 0.95:
            # Sleep until the next minute starts
            sleep_time = 60 - (current_time - self.minute_start)
            logger.info(f"Approaching request rate limit, sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time + 1)  # Add 1 second buffer
            
            # Reset counters
            self.tokens_per_minute = 0
            self.requests_per_minute = 0
            self.minute_start = time.time()
    
    def _save_results(self, processed_hashes):
        """
        Save embedding results.
        
        Args:
            processed_hashes (list): List of processed hash values
        """
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # For large embeddings, use memory-mapped files
        if len(self.embeddings) > 100000:
            # Create memory-mapped file
            embeddings_mmap = MMapDict(self.temp_dir / "embeddings.mmap", mode='w+')
            
            # Save embeddings
            for hash_value, embedding in self.embeddings.items():
                embeddings_mmap[hash_value] = embedding
            
            # Save index
            with open(self.output_dir / "embedding_index.json", 'w') as f:
                json.dump({
                    'location': str(self.temp_dir / "embeddings.mmap"),
                    'count': len(embeddings_mmap),
                    'dimensions': self.dimensions
                }, f, indent=2)
        else:
            # For small datasets, save as JSON (actually pickle due to numpy arrays)
            with open(self.temp_dir / "embeddings.pkl", 'wb') as f:
                import pickle
                pickle.dump(self.embeddings, f)
            
            # Save index
            with open(self.output_dir / "embedding_index.json", 'w') as f:
                json.dump({
                    'location': str(self.temp_dir / "embeddings.pkl"),
                    'count': len(self.embeddings),
                    'dimensions': self.dimensions,
                    'format': 'pickle'
                }, f, indent=2)
        
        # Save list of embedded hashes
        with open(self.output_dir / "embedded_hashes.json", 'w') as f:
            json.dump(processed_hashes, f)
        
        # Save sample of embeddings
        sample_size = min(100, len(self.embeddings))
        sample_hashes = list(self.embeddings.keys())[:sample_size]
        sample_embeddings = {
            hash_value: self.embeddings[hash_value].tolist()
            for hash_value in sample_hashes
        }
        
        with open(self.output_dir / "embeddings_sample.json", 'w') as f:
            json.dump(sample_embeddings, f, indent=2)
        
        # Save embedding statistics
        embedding_stats = {
            'count': len(processed_hashes),
            'dimensions': self.dimensions,
            'model': self.model
        }
        
        with open(self.output_dir / "embedding_statistics.json", 'w') as f:
            json.dump(embedding_stats, f, indent=2)
        
        # Save final checkpoint
        checkpoint_path = self.checkpoint_dir / "embedding_final.ckpt"
        save_checkpoint({
            'embeddings_location': str(self.temp_dir / "embeddings.mmap"),
            'processed_hashes': processed_hashes
        }, checkpoint_path)
        
        logger.info(f"Embedding results saved to {self.output_dir}")
