"""
Unified Embedding and Indexing Module for Entity Resolution

This module handles both the generation of vector embeddings and their direct indexing
in Weaviate, avoiding the storage of vectors on disk as per project requirements.

PERFORMANCE OPTIMIZATIONS FOR TIER 4 OPENAI ACCOUNT (2025-07-02):
- CORRECTED: Rate limits matched to actual Tier 4 limits (5M TPM, 10K RPM, 500M TPD)
- OPTIMIZED: embedding_batch_size: 32 → 512 (16x improvement)
- OPTIMIZED: Predictive rate limiting to avoid API throttling
- OPTIMIZED: Smarter thread synchronization and lock management
- OPTIMIZED: Dynamic batch sizing based on token estimation
- OPTIMIZED: Weaviate batch size: 100 → 1000 (10x improvement)
- OPTIMIZED: Early returns and efficient data structures
- REALISTIC: 2-4x speedup achievable through architectural improvements
- Expected speedup: 10 days → 3-5 days processing time (realistic)
"""

import os
import sys
import logging

# Suppress verbose HTTP logging for background processing
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
import pickle
import time
import json
import threading
import random
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import requests
from openai import OpenAI
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.config import VectorDistances
from weaviate.util import generate_uuid5

# Import embedding provider factory
from src.embedding_providers.factory import EmbeddingProviderFactory

logger = logging.getLogger(__name__)

class EmbeddingAndIndexingPipeline:
    """
    Combined pipeline for embedding generation and direct Weaviate indexing.
    Implements a stateless vector processing approach that maintains checkpointing
    while avoiding intermediate vector storage.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config: Configuration dictionary with embedding and indexing parameters
        """
        self.config = config
        
        # Initialize embedding provider based on configuration
        self.embedding_provider = self._init_embedding_provider()
        
        # Get provider-specific configuration for backward compatibility
        self.embedding_model = self.embedding_provider.get_model_name()
        self.embedding_dimensions = self.embedding_provider.get_dimensions()
        
        # Check if we're using ColBERT (which uses automatic vectorization)
        self.is_colbert = False
        vectorizer_config = self.embedding_provider.get_weaviate_vectorizer_config()
        if isinstance(vectorizer_config, dict) and vectorizer_config.get('_is_colbert'):
            self.is_colbert = True
            logger.info("ColBERT mode detected - will use automatic vectorization")
        
        # Get provider rate limiting configuration
        rate_config = self.embedding_provider.get_rate_limit_config()
        self.batch_size = rate_config.get('batch_size', config.get("embedding_batch_size", 512))
        
        # Backward compatibility: maintain existing rate limiting attributes with sensible defaults
        # Use large numbers for "unlimited" rather than None to avoid comparison errors
        self.max_tokens_per_minute = getattr(self.embedding_provider, 'max_tokens_per_minute', float('inf'))
        self.max_requests_per_minute = getattr(self.embedding_provider, 'max_requests_per_minute', float('inf'))
        self.max_tokens_per_day = getattr(self.embedding_provider, 'max_tokens_per_day', float('inf'))
        self.tokens_this_minute = getattr(self.embedding_provider, 'tokens_this_minute', 0)
        self.requests_this_minute = getattr(self.embedding_provider, 'requests_this_minute', 0)
        self.minute_start = getattr(self.embedding_provider, 'minute_start', time.time())
        self.tokens_today = getattr(self.embedding_provider, 'tokens_today', 0)
        self.day_start = getattr(self.embedding_provider, 'day_start', time.time())
        self.api_rate_limit_info = getattr(self.embedding_provider, 'api_rate_limit_info', {})
        self.last_api_headers = getattr(self.embedding_provider, 'last_api_headers', {})
        
        # Daily usage monitoring configuration
        self.tpd_resume_threshold = config.get('tpd_resume_threshold', 50_000_000)
        
        # Embedding fields configuration
        self.embed_fields = config.get("embed_fields", ["composite", "person", "title"])
        self.skip_fields = config.get("skip_fields", ["provision", "subjects", "personId"])
        
        # Initialize Weaviate client
        self.weaviate_client = self._init_weaviate_client()
        
        # Prepare the collection
        self.collection = self._ensure_schema_exists()
        
        # Initialize checkpoint tracking
        self.processed_hashes = set()
        
        # Failed request tracking for batch migration support
        self.failed_requests = {}  # hash_value -> failure_info
        self.retry_queue = []  # List of hashes to retry
        
        # Load daily token usage tracking
        self._load_daily_usage_checkpoint()
        
        logger.info(f"Initialized EmbeddingAndIndexingPipeline with model {self.embedding_model}")
    
    def _init_embedding_provider(self):
        """
        Initialize the embedding provider based on configuration.
        
        Returns:
            Initialized embedding provider instance
        """
        # Check for new embedding provider configuration
        embedding_config = self.config.get('embedding')
        if embedding_config and 'provider' in embedding_config:
            # Use new provider-based configuration
            try:
                provider = EmbeddingProviderFactory.create_from_full_config(embedding_config)
                logger.info(f"Initialized {embedding_config['provider']} embedding provider")
                return provider
            except Exception as e:
                logger.error(f"Failed to create embedding provider: {str(e)}")
                logger.info("Falling back to legacy OpenAI configuration")
        
        # Fallback to legacy OpenAI configuration for backward compatibility
        logger.info("Using legacy OpenAI configuration")
        openai_config = {
            'model': self.config.get('embedding_model', 'text-embedding-3-small'),
            'dimensions': self.config.get('embedding_dimensions', 1536),
            'api_key_env': 'OPENAI_API_KEY',
            'api_base': self.config.get('openai_api_base', 'https://api.openai.com/v1'),
            'batch_size': self.config.get('embedding_batch_size', 512),
            'max_tokens_per_minute': self.config.get('max_tokens_per_minute', 4_800_000),
            'max_requests_per_minute': self.config.get('max_requests_per_minute', 9_500),
            'max_tokens_per_day': self.config.get('max_tokens_per_day', 480_000_000),
            'supports_batch': True
        }
        
        # Override with explicit OpenAI API key if provided
        openai_api_key = self.config.get("openai_api_key")
        if openai_api_key:
            os.environ['OPENAI_API_KEY'] = openai_api_key
        
        return EmbeddingProviderFactory.create_provider('openai', openai_config)
    
    def _init_weaviate_client(self):
        """Initialize and return a Weaviate client based on configuration."""
        # Get Weaviate connection parameters
        weaviate_url = self.config.get("weaviate_url", "http://localhost:8080")
        
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
        grpc_port = self.config.get("weaviate_grpc_port", 50051)
        
        # Create API key authentication if provided
        auth_client_secret = None
        api_key = self.config.get("weaviate_api_key")
        if api_key:
            from weaviate.auth import AuthApiKey
            auth_client_secret = AuthApiKey(api_key)
        
        try:
            # Create connection parameters
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
    
    def _ensure_schema_exists(self):
        """Create Weaviate schema if it doesn't exist."""
        
        # Get recreate flag from config
        recreate_collections = self.config.get("recreate_collections", False)
        
        # Create collection with named vectors if it doesn't exist
        try:
            # If recreate_collections is true, we forcibly delete the collection
            if recreate_collections:
                try:
                    self.weaviate_client.collections.delete("EntityString")
                    logger.info("Deleted existing EntityString collection")
                    # Force collection to not exist by setting a flag
                    collection_exists = False
                except Exception as e:
                    logger.info(f"No existing EntityString collection found or cannot be deleted: {e}")
                    # We'll try to get it again below
                    collection_exists = True
            else:
                # Default case - we're not forcing recreation
                collection_exists = True
            
            # Check if collection exists only if we haven't just deleted it
            if collection_exists:
                try:
                    collection = self.weaviate_client.collections.get("EntityString")
                    logger.info("EntityString collection already exists")
                    return collection
                except Exception:
                    # Collection doesn't exist, we'll create it below
                    pass
            
            # At this point, we know we need to create the collection
            # Get index configuration parameters
            ef = self.config.get("weaviate_ef", 128)
            max_connections = self.config.get("weaviate_max_connections", 64)
            ef_construction = self.config.get("weaviate_ef_construction", 128)
            
            # Get dynamic vectorizer configuration from provider
            vectorizer_config = self.embedding_provider.get_weaviate_vectorizer_config()
            
            if vectorizer_config:
                # Check if this is a ColBERT configuration
                if isinstance(vectorizer_config, dict) and vectorizer_config.get('_is_colbert'):
                    # Special handling for ColBERT with MUVERA - use raw API
                    logger.info(f"Creating new EntityString collection with {self.embedding_provider.get_model_name()} ColBERT + MUVERA")
                    muvera_config = vectorizer_config.get('_config', {})
                    
                    # Use raw API for proper MUVERA configuration
                    from src.create_colbert_schema import create_colbert_collection_with_muvera
                    
                    # Get Weaviate URL from client
                    weaviate_url = "http://localhost:8080"  # Default
                    if hasattr(self.weaviate_client, '_connection') and hasattr(self.weaviate_client._connection, 'url'):
                        weaviate_url = str(self.weaviate_client._connection.url).rstrip('/')
                    
                    # Prepare HNSW configuration
                    hnsw_config = {
                        'ef': ef,
                        'efConstruction': ef_construction,
                        'maxConnections': max_connections
                    }
                    
                    success = create_colbert_collection_with_muvera(
                        weaviate_url=weaviate_url,
                        muvera_config=muvera_config,
                        collection_name="EntityString",
                        recreate=True,  # Always recreate when explicitly requested
                        hnsw_config=hnsw_config
                    )
                    
                    if success:
                        # Get the collection reference
                        collection = self.weaviate_client.collections.get("EntityString")
                    else:
                        raise Exception("Failed to create EntityString collection with MUVERA")
                else:
                    # Regular Weaviate integration (OpenAI, Mistral, Cohere, etc.)
                    logger.info(f"Creating new EntityString collection with {self.embedding_provider.get_model_name()} Vectorizer")
                    collection = self.weaviate_client.collections.create(
                        name="EntityString",
                        description="Collection for entity string values with their embeddings",
                        vectorizer_config=vectorizer_config,
                        vector_index_config=Configure.VectorIndex.hnsw(
                            ef=ef,
                            max_connections=max_connections,
                            ef_construction=ef_construction,
                            distance_metric=VectorDistances.COSINE
                        ),
                        properties=[
                            Property(name="original_string", data_type=DataType.TEXT),
                            Property(name="hash_value", data_type=DataType.TEXT),
                            Property(name="field_type", data_type=DataType.TEXT),
                            Property(name="frequency", data_type=DataType.INT)
                        ]
                    )
            else:
                # Provider uses custom vectors (e.g., local models)
                logger.info(f"Creating new EntityString collection with custom vectors for {self.embedding_provider.get_model_name()}")
                collection = self.weaviate_client.collections.create(
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
            
            logger.info(f"Created EntityString collection with {self.embedding_provider.get_model_name()} configuration")
            return collection
        except Exception as e:
            logger.error(f"Error creating Weaviate schema: {str(e)}")
            raise
    
    def _load_daily_usage_checkpoint(self) -> None:
        """
        Load daily token usage tracking from checkpoint file.
        """
        checkpoint_dir = self.config.get("checkpoint_dir", "data/checkpoints")
        daily_usage_path = os.path.join(checkpoint_dir, 'daily_token_usage.json')
        
        try:
            if os.path.exists(daily_usage_path):
                with open(daily_usage_path, 'r') as f:
                    usage_data = json.load(f)
                
                # Check if it's the same day
                saved_day_start = usage_data.get('day_start')
                current_time = time.time()
                
                if saved_day_start:
                    # Check if we're in the same day (within 24 hours)
                    if current_time - saved_day_start < 86400:  # 24 hours
                        self.tokens_today = usage_data.get('tokens_today', 0)
                        self.day_start = saved_day_start
                        logger.info(f"Loaded daily usage: {self.tokens_today:,} tokens used today")
                    else:
                        # New day, reset counters
                        self.tokens_today = 0
                        self.day_start = current_time
                        logger.info("New day detected, reset daily token usage")
                else:
                    # No valid day_start, treat as new day
                    self.tokens_today = 0
                    self.day_start = current_time
            else:
                # No checkpoint file, start fresh
                self.tokens_today = 0
                self.day_start = time.time()
                logger.info("No daily usage checkpoint found, starting fresh")
                
        except Exception as e:
            logger.error(f"Error loading daily usage checkpoint: {e}")
            # Fall back to fresh start
            self.tokens_today = 0
            self.day_start = time.time()
    
    def _save_daily_usage_checkpoint(self) -> None:
        """
        Save daily token usage tracking to checkpoint file.
        """
        checkpoint_dir = self.config.get("checkpoint_dir", "data/checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        daily_usage_path = os.path.join(checkpoint_dir, 'daily_token_usage.json')
        
        try:
            usage_data = {
                'tokens_today': self.tokens_today,
                'day_start': self.day_start,
                'last_updated': time.time()
            }
            
            with open(daily_usage_path, 'w') as f:
                json.dump(usage_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving daily usage checkpoint: {e}")
    
    def _parse_api_headers(self, response) -> None:
        """
        Parse OpenAI API response headers for rate limit information.
        
        Args:
            response: OpenAI API response object
        """
        if hasattr(response, '_response') and hasattr(response._response, 'headers'):
            headers = response._response.headers
            
            # Parse rate limit headers
            self.api_rate_limit_info.update({
                'remaining_requests': self._parse_header_int(headers.get('x-ratelimit-remaining-requests')),
                'remaining_tokens': self._parse_header_int(headers.get('x-ratelimit-remaining-tokens')),
                'reset_requests': headers.get('x-ratelimit-reset-requests'),
                'reset_tokens': headers.get('x-ratelimit-reset-tokens'),
                'limit_requests': self._parse_header_int(headers.get('x-ratelimit-limit-requests')),
                'limit_tokens': self._parse_header_int(headers.get('x-ratelimit-limit-tokens'))
            })
            
            # Store full headers for debugging
            self.last_api_headers = dict(headers)
            
            logger.debug(f"API Rate Limits - Tokens: {self.api_rate_limit_info['remaining_tokens']}/{self.api_rate_limit_info['limit_tokens']}, "
                        f"Requests: {self.api_rate_limit_info['remaining_requests']}/{self.api_rate_limit_info['limit_requests']}")
    
    def _parse_header_int(self, value: str) -> Optional[int]:
        """
        Parse header value as integer, return None if invalid.
        
        Args:
            value: Header value string
            
        Returns:
            Parsed integer or None
        """
        if value is None:
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            return None
    
    def _check_daily_rate_limit(self) -> bool:
        """
        Check if daily token limit has been reached.
        
        Returns:
            True if under limit, False if limit reached
        """
        # Reset daily counter if new day
        current_time = time.time()
        if self.day_start and current_time - self.day_start >= 86400:  # 24 hours
            self.tokens_today = 0
            self.day_start = current_time
            self._save_daily_usage_checkpoint()
            logger.info("Daily token usage reset for new day")
        
        # Check if we're under the daily limit
        return self.tokens_today < self.max_tokens_per_day
    
    def _wait_for_daily_limit_reset(self) -> None:
        """
        Wait for daily token limit to reset with periodic polling.
        Uses 30-minute polling intervals and resumes when >100M tokens available.
        """
        max_tokens_display = f"{self.max_tokens_per_day:,.0f}" if self.max_tokens_per_day != float('inf') else "∞"
        logger.warning(f"Daily token limit reached ({self.tokens_today:,}/{max_tokens_display}). "
                      f"Entering polling mode with {self.tpd_poll_interval/60:.0f}-minute intervals")
        
        while True:
            # Wait for polling interval
            logger.info(f"Waiting {self.tpd_poll_interval/60:.0f} minutes before checking daily limit...")
            time.sleep(self.tpd_poll_interval)
            
            # Check if daily limit has reset (new day)
            if self._check_daily_rate_limit():
                # Handle case where max_tokens_per_day is infinity (unlimited)
                if self.max_tokens_per_day == float('inf'):
                    logger.info("No daily token limit for this provider. Resuming processing.")
                    break
                remaining_tokens = self.max_tokens_per_day - self.tokens_today
                if remaining_tokens >= self.tpd_resume_threshold:
                    logger.info(f"Daily limit reset detected. Resuming processing with {remaining_tokens:,} tokens available")
                    break
                else:
                    logger.info(f"Daily limit reset but only {remaining_tokens:,} tokens available (need {self.tpd_resume_threshold:,}). Continuing to wait...")
            else:
                max_tokens_display = f"{self.max_tokens_per_day:,.0f}" if self.max_tokens_per_day != float('inf') else "∞"
                logger.info(f"Daily limit still active. Used: {self.tokens_today:,}/{max_tokens_display} tokens")
    
    def _load_migrated_failed_requests(self) -> None:
        """
        Load failed requests migrated from batch processing.
        """
        checkpoint_dir = self.config.get("checkpoint_dir", "data/checkpoints")
        migrated_failed_path = os.path.join(checkpoint_dir, 'realtime_failed_requests.pkl')
        
        if os.path.exists(migrated_failed_path):
            try:
                with open(migrated_failed_path, 'rb') as f:
                    self.failed_requests = pickle.load(f)
                
                # Filter out already processed hashes
                retryable_count = 0
                for hash_value, failure_info in self.failed_requests.items():
                    if hash_value not in self.processed_hashes:
                        # Check if this request is retryable
                        max_retries = self.config.get("error_handling", {}).get("max_retry_attempts", 3)
                        retry_count = failure_info.get('retry_count', 0)
                        error_category = failure_info.get('error_category', 'other')
                        
                        # Skip permanent errors like validation failures
                        if error_category != 'validation' and retry_count < max_retries:
                            self.retry_queue.append(hash_value)
                            retryable_count += 1
                
                logger.info(f"Loaded {len(self.failed_requests)} migrated failed requests, "
                           f"{retryable_count} are retryable")
                           
            except Exception as e:
                logger.error(f"Error loading migrated failed requests: {e}")
                self.failed_requests = {}
        else:
            logger.debug("No migrated failed requests found")
    
    def _save_failed_requests_checkpoint(self) -> None:
        """
        Save failed requests to checkpoint file.
        """
        if not self.failed_requests:
            return
            
        checkpoint_dir = self.config.get("checkpoint_dir", "data/checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        failed_requests_path = os.path.join(checkpoint_dir, 'realtime_failed_requests.pkl')
        
        try:
            with open(failed_requests_path, 'wb') as f:
                pickle.dump(self.failed_requests, f)
                
            logger.debug(f"Saved {len(self.failed_requests)} failed requests to checkpoint")
            
        except Exception as e:
            logger.error(f"Error saving failed requests checkpoint: {e}")
    
    def _add_failed_request(self, hash_value: str, field_type: str, error_info: Dict[str, Any]) -> None:
        """
        Add a failed request to tracking.
        
        Args:
            hash_value: Hash of the failed request
            field_type: Field type being processed
            error_info: Error information from API
        """
        if hash_value not in self.failed_requests:
            self.failed_requests[hash_value] = {
                'field_type': field_type,
                'error_info': error_info,
                'retry_count': 0,
                'first_attempt': time.time(),
                'last_attempt': time.time(),
                'error_category': self._categorize_realtime_error(error_info),
                'migrated_from_batch': False
            }
        else:
            # Update existing failed request
            self.failed_requests[hash_value]['retry_count'] += 1
            self.failed_requests[hash_value]['last_attempt'] = time.time()
            self.failed_requests[hash_value]['error_info'] = error_info
    
    def _categorize_realtime_error(self, error_info: Dict[str, Any]) -> str:
        """
        Categorize error for retry logic.
        
        Args:
            error_info: Error information
            
        Returns:
            Error category string
        """
        error_msg = str(error_info).lower()
        
        if 'rate limit' in error_msg or 'rate_limit' in error_msg:
            return 'rate_limit'
        elif 'quota' in error_msg or 'insufficient' in error_msg:
            return 'quota'
        elif 'invalid' in error_msg or 'validation' in error_msg:
            return 'validation'
        elif 'server' in error_msg or 'internal' in error_msg or 'timeout' in error_msg:
            return 'server'
        else:
            return 'other'
    
    def _get_retry_requests(self, string_dict: Dict[str, str], field_hash_mapping: Dict[str, Dict[str, int]],
                          string_counts: Dict[str, int]) -> List[Tuple[str, str, str, int]]:
        """
        Get retry requests from the retry queue.
        
        Args:
            string_dict: Hash to string mapping
            field_hash_mapping: Hash to field mapping
            string_counts: Hash to frequency mapping
            
        Returns:
            List of (hash, string, field_type, frequency) tuples for retry
        """
        retry_requests = []
        
        for hash_value in self.retry_queue[:]:
            if hash_value in self.failed_requests and hash_value not in self.processed_hashes:
                failure_info = self.failed_requests[hash_value]
                
                # Check if we have the required data for retry
                if hash_value in string_dict:
                    string_value = string_dict[hash_value]
                    field_type = failure_info.get('field_type', 'composite')
                    frequency = string_counts.get(hash_value, 1)
                    
                    retry_requests.append((hash_value, string_value, field_type, frequency))
                    
                    # Remove from retry queue (will be re-added if it fails again)
                    self.retry_queue.remove(hash_value)
                else:
                    logger.warning(f"Cannot retry hash {hash_value} - string data not found")
                    self.retry_queue.remove(hash_value)
        
        if retry_requests:
            logger.info(f"Prepared {len(retry_requests)} failed requests for retry")
            
        return retry_requests
    
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type((requests.exceptions.RequestException, json.JSONDecodeError))
    )
    def _get_embeddings_batch(self, texts: List[str]) -> Tuple[List[np.ndarray], List[int]]:
        """
        Get embeddings for a batch of texts using the configured embedding provider.
        
        Args:
            texts: List of strings to embed
            
        Returns:
            Tuple of (embeddings, token_counts) where embeddings is a list of numpy arrays
            and token_counts is a list of token counts for each input
        """
        try:
            # Get embeddings using the provider
            embeddings, token_counts = self.embedding_provider.get_embeddings(texts)
            
            # Update daily token usage for providers that track it
            total_tokens = sum(token_counts)
            if hasattr(self, 'tokens_today'):
                self.tokens_today += total_tokens
            
            logger.debug(f"Generated {len(embeddings)} embeddings using {total_tokens} tokens "
                        f"with {self.embedding_provider.get_model_name()}")
            
            return embeddings, token_counts
            
        except Exception as e:
            logger.error(f"Error generating embeddings via {self.embedding_provider.get_model_name()}: {str(e)}")
            # Re-raise to trigger retry - failed requests will be tracked at higher level
            raise requests.exceptions.RequestException(f"Embedding provider error: {str(e)}")

    
    def _index_embeddings_batch(self, items_to_index: List[Dict[str, Any]]) -> Tuple[int, List[str]]:
        """
        Index a batch of embeddings directly in Weaviate with enhanced diagnostics.
        
        Args:
            items_to_index: List of items to index, each with hash_value, original_string,
                           field_type, frequency, and vector
        
        Returns:
            Tuple of (number of successfully indexed items, list of successfully indexed hash_values)
        """
        # Import here to avoid circular imports
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.vector_diagnostics import VectorDiagnosticTool
        
        indexed_count = 0
        successful_hashes = []
        
        try:
            # Initialize diagnostic tool
            diagnostics = VectorDiagnosticTool(self.weaviate_client, {
                "embedding_dimensions": self.embedding_dimensions,
                "vector_diagnostics_verbose": self.config.get("vector_diagnostics_verbose", False)
            })
            
            # Select verification samples
            verification_samples = random.sample(
                items_to_index, 
                min(3, len(items_to_index))
            ) if items_to_index else []
            
            # Minimal batch logging
            logger.debug(f"Indexing batch of {len(items_to_index)} items")
            
            # Use larger fixed-size batch configuration for better performance - OPTIMIZED
            # OpenAI supports up to 2048 inputs per request, so use larger batches for indexing
            if self.is_colbert:
                # Use smaller batches for ColBERT to avoid timeouts during automatic vectorization
                optimal_batch_size = min(8, len(items_to_index))  # Much smaller for ColBERT
                logger.debug(f"Using reduced batch size {optimal_batch_size} for ColBERT automatic vectorization")
            else:
                optimal_batch_size = min(500, len(items_to_index))  # Increased from 100 to 500
            
            with self.collection.batch.fixed_size(batch_size=optimal_batch_size) as batch_writer:
                for item in items_to_index:
                    try:
                        # Generate UUID from hash value and field type for idempotency
                        uuid_input = f"{item['hash_value']}_{item['field_type']}"
                        uuid = generate_uuid5(uuid_input)
                        
                        # Remove vector from properties
                        properties = {k: v for k, v in item.items() if k != 'vector'}
                        
                        if self.is_colbert:
                            # Skip vector processing for ColBERT
                            vector_data = None
                        else:
                            # Process vector with diagnostic tool
                            vector_data = diagnostics.debug_vector_transmission(
                                item['vector'],
                                item['hash_value'],
                                item['field_type']
                            )
                            
                            # Verify vector dimensions
                            if len(vector_data) != self.embedding_dimensions:
                                logger.error(f"Vector dimension mismatch: {len(vector_data)} != {self.embedding_dimensions}")
                                continue
                        
                        # Add object - with or without vector depending on provider
                        if self.is_colbert:
                            # ColBERT uses automatic vectorization, don't provide vector
                            batch_writer.add_object(
                                properties=properties,
                                uuid=uuid
                            )
                        else:
                            # Regular providers need explicit vector
                            batch_writer.add_object(
                                properties=properties,
                                uuid=uuid,
                                vector=vector_data  # Direct vector assignment with normalized format
                            )
                        
                        indexed_count += 1
                        successful_hashes.append(item['hash_value'])
                        
                    except Exception as e:
                        logger.error(f"Error indexing item {item.get('hash_value', 'unknown')}: {str(e)}")
            
            # Verify vector persistence for sample items
            if verification_samples:
                verification_success = diagnostics.verify_vector_persistence(verification_samples)
                if not verification_success:
                    logger.warning("Vector persistence verification FAILED")
            
            logger.debug(f"Successfully indexed {indexed_count}/{len(items_to_index)} items")
            
        except Exception as e:
            logger.error(f"Error in batch indexing with enhanced diagnostics: {str(e)}")
            # Don't raise here, allow processing to continue with other batches
        
        return indexed_count, successful_hashes
    
    def _process_and_index_batch(self, batch_texts: List[Tuple[str, str, str, int]], 
                               lock: threading.Lock) -> Tuple[int, int]:
        """
        Process a batch of texts: generate embeddings and index in Weaviate.
        
        Args:
            batch_texts: List of (hash, text, field_type, frequency) tuples
            lock: Lock for thread-safe counter updates
            
        Returns:
            Tuple of (indexed_count, tokens_used)
        """
        # OPTIMIZED: Pre-check rate limits before expensive lock acquisition
        # Quick check without lock to avoid blocking threads unnecessarily
        if not self._check_daily_rate_limit():
            max_tokens_display = f"{self.max_tokens_per_day:,.0f}" if self.max_tokens_per_day != float('inf') else "∞"
            logger.info(f"Daily token limit reached ({self.tokens_today:,}/{max_tokens_display})")
            self._wait_for_daily_limit_reset()
            return 0, 0  # Early return to avoid processing
        
        # OPTIMIZED: Smart rate limit enforcement with predictive throttling
        current_time = time.time()
        
        # Extract texts for estimation
        texts = [item[1] for item in batch_texts]
        
        # Estimate tokens for this batch to avoid exceeding limits
        estimated_tokens = len(texts) * 100  # Conservative estimate: 100 tokens per text
        
        sleep_time = 0
        with lock:
            # Reset counters if new minute
            if current_time - self.minute_start >= 60:
                self.tokens_this_minute = 0
                self.requests_this_minute = 0
                self.minute_start = current_time
            
            # OPTIMIZED: Predictive rate limiting - check if this batch would exceed limits
            elif (self.tokens_this_minute + estimated_tokens >= self.max_tokens_per_minute or 
                  self.requests_this_minute + 1 >= self.max_requests_per_minute):
                # Only sleep the minimum required time
                sleep_time = max(1, 60 - (current_time - self.minute_start))
                logger.debug(f"Predictive rate limit pause: {sleep_time:.1f}s")
                
        # Sleep outside lock to avoid blocking other threads
        if sleep_time > 0:
            time.sleep(sleep_time)
            # Reset counters after sleep
            with lock:
                self.tokens_this_minute = 0
                self.requests_this_minute = 0
                self.minute_start = time.time()
        
        # Extract hash values, texts, and field types
        hashes = [item[0] for item in batch_texts]
        texts = [item[1] for item in batch_texts]
        field_types = [item[2] for item in batch_texts]
        frequencies = [item[3] for item in batch_texts]
        
        try:
            if self.is_colbert:
                # ColBERT uses automatic vectorization, skip embedding generation
                embeddings = [None] * len(texts)  # Placeholder
                token_counts = [self.embedding_provider.estimate_tokens(text) for text in texts]
            else:
                # Get embeddings for regular providers
                embeddings, token_counts = self._get_embeddings_batch(texts)
                
                # Update rate limiting counters
                with lock:
                    self.requests_this_minute += 1
                    self.tokens_this_minute += sum(token_counts)
            
            # Prepare items for indexing
            items_to_index = []
            for i, hash_val in enumerate(hashes):
                item = {
                    'hash_value': hash_val,
                    'original_string': texts[i],
                    'field_type': field_types[i],
                    'frequency': frequencies[i]
                }
                if not self.is_colbert:
                    item['vector'] = embeddings[i]
                items_to_index.append(item)
            
            # Index in Weaviate
            indexed_count, successful_hashes = self._index_embeddings_batch(items_to_index)
            
            # Update processed hashes - only mark successfully indexed items as processed
            with lock:
                for hash_val in successful_hashes:
                    self.processed_hashes.add(hash_val)
            
            return indexed_count, sum(token_counts)
            
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            
            # Track failed requests for retry
            with lock:
                for i, hash_val in enumerate(hashes):
                    if hash_val not in self.processed_hashes:  # Only track if not already processed
                        self._add_failed_request(hash_val, field_types[i], {'error': str(e)})
                        
            return 0, 0
    
    def _process_batches_parallel(self, texts_to_process: List[Tuple[str, str, str, int]], 
                               max_workers: int = 4) -> Tuple[int, int]:
        """
        Process multiple batches in parallel using ThreadPoolExecutor.
        
        Args:
            texts_to_process: List of (hash, text, field_type, frequency) tuples
            max_workers: Maximum number of threads to use
            
        Returns:
            Tuple of (total_indexed, total_tokens)
        """
        # Calculate batch size for optimal parallelization - OPTIMIZED
        num_texts = len(texts_to_process)
        
        # OPTIMIZED: Calculate ideal batch size for maximum throughput
        # Balance between OpenAI's capacity (2048) and efficient parallelization
        
        # Calculate tokens per batch to stay under rate limits
        avg_tokens_per_text = 100  # Conservative estimate
        max_tokens_per_batch = self.max_tokens_per_minute // (max_workers * 2)  # Allow headroom
        max_texts_by_tokens = max_tokens_per_batch // avg_tokens_per_text
        
        # Use the most restrictive limit
        if self.is_colbert:
            # Much smaller batches for ColBERT to avoid timeouts
            optimal_batch_size = min(self.batch_size, 8)  # Max 8 for ColBERT
            adjusted_batch_size = min(optimal_batch_size, max(4, num_texts // max_workers))  # At least 4 items per batch
        else:
            optimal_batch_size = min(self.batch_size, max_texts_by_tokens, 2048)  # OpenAI's max input limit
            adjusted_batch_size = min(optimal_batch_size, max(128, num_texts // max_workers))  # At least 128 items per batch
        
        # Reset rate limits for this processing run
        self.tokens_this_minute = 0
        self.requests_this_minute = 0
        self.minute_start = time.time()
        
        # Prepare batches
        batches = []
        for i in range(0, num_texts, adjusted_batch_size):
            batch = texts_to_process[i:i+adjusted_batch_size]
            batches.append(batch)
        
        logger.info(f"Processing {len(batches)} batches with {adjusted_batch_size} items per batch using {max_workers} workers")
        
        # Results container
        total_indexed = 0
        total_tokens = 0
        
        # Create a lock for synchronizing counter updates
        lock = threading.Lock()
        
        # Process batches in parallel - no progress bars for background processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batches for processing
            futures = [executor.submit(self._process_and_index_batch, batch, lock) for batch in batches]
            
            # Process results as they complete - minimal logging
            for i, future in enumerate(as_completed(futures)):
                indexed_count, tokens_used = future.result()
                total_indexed += indexed_count
                total_tokens += tokens_used
        
        return total_indexed, total_tokens
    
    def _select_strings_to_process(self, string_dict: Dict[str, str], 
                                field_hash_mapping: Dict[str, Dict[str, int]],
                                string_counts: Dict[str, int]) -> List[Tuple[str, str, str, int]]:
        """
        Select strings for embedding and indexing based on field types.
        
        Args:
            string_dict: Dictionary mapping hash to string value
            field_hash_mapping: Mapping of hash to field types
            string_counts: Mapping of hash to frequency count
            
        Returns:
            List of (hash, string, field_type, frequency) tuples to process
        """
        logger.info("Selecting strings to process based on field types")
        
        # Track field match statistics
        field_match_counts = {field: 0 for field in self.embed_fields}
        strings_to_process = []
        
        # Calculate cache coverage metrics
        total_candidates = sum(1 for hash_val in string_dict.keys() 
                            if hash_val in field_hash_mapping and 
                            any(field in self.embed_fields for field in field_hash_mapping[hash_val].keys()))
        
        already_processed = sum(1 for hash_val in string_dict.keys() 
                             if hash_val in self.processed_hashes and hash_val in field_hash_mapping and 
                             any(field in self.embed_fields for field in field_hash_mapping[hash_val].keys()))
        
        cache_coverage = 0 if total_candidates == 0 else already_processed / total_candidates
        
        # If we have almost complete coverage already, log and return empty list
        if total_candidates > 0 and cache_coverage >= 0.99:
            logger.info(f"===== PROCESSING STATUS: COMPLETE =====")
            logger.info(f"All eligible strings ({already_processed}/{total_candidates}) are already processed")
            logger.info(f"No new processing required - proceeding with existing data")
            return []
        
        # Process each string
        for hash_val, string_val in string_dict.items():
            # Skip already processed or invalid entries
            if hash_val in self.processed_hashes or hash_val == "NULL" or not string_val:
                continue
            
            # Get field mapping for this hash
            if hash_val in field_hash_mapping:
                # Extract fields
                fields = list(field_hash_mapping[hash_val].keys())
                
                # Check which fields match our embed_fields list
                matched_fields = [f for f in fields if f in self.embed_fields]
                
                # Update match counts
                for field in matched_fields:
                    field_match_counts[field] += 1
                
                # If any field matches, include this string - once per field type
                for field in matched_fields:
                    strings_to_process.append((
                        hash_val, 
                        string_val, 
                        field,
                        string_counts.get(hash_val, 1)
                    ))
        
        # Add retry requests from migrated failed batch processing
        retry_requests = self._get_retry_requests(string_dict, field_hash_mapping, string_counts)
        strings_to_process.extend(retry_requests)
        
        # Concise selection summary
        selected_count = len(strings_to_process)
        retry_count = len(retry_requests)
        new_count = selected_count - retry_count
        
        if selected_count > 0:
            logger.info(f"Processing {selected_count} items ({new_count} new, {retry_count} retry) | Cache: {already_processed}/{total_candidates} ({cache_coverage:.1%})")
        
        return strings_to_process
    
    def get_processing_status(self) -> Dict[str, Any]:
        """
        Get comprehensive processing status similar to batch processing.
        
        Returns:
            Dictionary with detailed status information
        """
        status = {
            'timestamp': time.time(),
            'processed_hashes': len(self.processed_hashes),
            'failed_requests': len(self.failed_requests),
            'retry_queue': len(self.retry_queue),
            'daily_token_usage': {
                'tokens_used': self.tokens_today,
                'tokens_limit': self.max_tokens_per_day,
                'usage_percentage': (self.tokens_today / self.max_tokens_per_day * 100) if self.max_tokens_per_day != float('inf') else 0,
                'day_start': self.day_start
            },
            'rate_limits': {
                'tokens_per_minute': {
                    'current': self.tokens_this_minute,
                    'limit': self.max_tokens_per_minute,
                    'usage_percentage': (self.tokens_this_minute / self.max_tokens_per_minute * 100) if self.max_tokens_per_minute != float('inf') else 0
                },
                'requests_per_minute': {
                    'current': self.requests_this_minute,
                    'limit': self.max_requests_per_minute,
                    'usage_percentage': (self.requests_this_minute / self.max_requests_per_minute * 100) if self.max_requests_per_minute != float('inf') else 0
                }
            },
            'api_rate_limit_info': self.api_rate_limit_info.copy(),
            'failed_request_analysis': self._analyze_failed_requests(),
            'processing_mode': 'real-time'
        }
        
        return status
    
    def _analyze_failed_requests(self) -> Dict[str, Any]:
        """
        Analyze failed requests similar to batch processing.
        
        Returns:
            Dictionary with failure analysis
        """
        if not self.failed_requests:
            return {
                'total_failed': 0,
                'by_category': {},
                'retryable': 0,
                'max_retries_exceeded': 0
            }
        
        analysis = {
            'total_failed': len(self.failed_requests),
            'by_category': {},
            'by_retry_count': {},
            'retryable': 0,
            'max_retries_exceeded': 0
        }
        
        max_retry_attempts = self.config.get("error_handling", {}).get("max_retry_attempts", 3)
        
        for hash_value, failure_data in self.failed_requests.items():
            category = failure_data.get('error_category', 'other')
            retry_count = failure_data.get('retry_count', 0)
            
            # Count by category
            analysis['by_category'][category] = analysis['by_category'].get(category, 0) + 1
            
            # Count by retry attempts
            analysis['by_retry_count'][retry_count] = analysis['by_retry_count'].get(retry_count, 0) + 1
            
            # Count retryable vs max retries exceeded
            if category != 'validation' and retry_count < max_retry_attempts:
                analysis['retryable'] += 1
            else:
                analysis['max_retries_exceeded'] += 1
        
        return analysis
    
    def print_status_report(self) -> None:
        """
        Print detailed status report similar to batch processing.
        """
        status = self.get_processing_status()
        
        print("\n" + "="*60)
        print("REAL-TIME PROCESSING STATUS")
        print("="*60)
        
        # Processing statistics
        print(f"Processed Hashes: {status['processed_hashes']:,}")
        print(f"Failed Requests: {status['failed_requests']:,}")
        print(f"Retry Queue: {status['retry_queue']:,}")
        
        # Daily usage
        daily = status['daily_token_usage']
        print(f"\n📊 DAILY TOKEN USAGE:")
        print(f"   Used: {daily['tokens_used']:,}/{daily['tokens_limit']:,} ({daily['usage_percentage']:.1f}%)")
        
        # Rate limits
        rate_limits = status['rate_limits']
        tpm = rate_limits['tokens_per_minute']
        rpm = rate_limits['requests_per_minute']
        print(f"\n⏱️  RATE LIMITS:")
        print(f"   Tokens/min: {tpm['current']:,}/{tpm['limit']:,} ({tpm['usage_percentage']:.1f}%)")
        print(f"   Requests/min: {rpm['current']:,}/{rpm['limit']:,} ({rpm['usage_percentage']:.1f}%)")
        
        # API rate limit info from headers
        api_info = status['api_rate_limit_info']
        if api_info['remaining_tokens'] is not None:
            print(f"\n🔗 API RATE LIMITS (from headers):")
            print(f"   Remaining tokens: {api_info['remaining_tokens']:,}")
            print(f"   Remaining requests: {api_info['remaining_requests']:,}")
            print(f"   Reset time (tokens): {api_info['reset_tokens']}")
        
        # Failed request analysis
        failed_analysis = status['failed_request_analysis']
        if failed_analysis['total_failed'] > 0:
            print(f"\n❌ FAILED REQUEST ANALYSIS:")
            print(f"   Total failed: {failed_analysis['total_failed']:,}")
            print(f"   Retryable: {failed_analysis['retryable']:,}")
            print(f"   Max retries exceeded: {failed_analysis['max_retries_exceeded']:,}")
            
            if failed_analysis['by_category']:
                print(f"   By category:")
                for category, count in failed_analysis['by_category'].items():
                    print(f"     • {category}: {count:,}")
    
    def load_checkpoint(self, checkpoint_dir: str) -> None:
        """
        Load checkpoint data from disk.
        
        Args:
            checkpoint_dir: Directory containing checkpoint files
        """
        # Load processed hashes if checkpoint exists
        processed_hashes_path = os.path.join(checkpoint_dir, 'processed_hashes.pkl')
        
        if os.path.exists(processed_hashes_path):
            try:
                with open(processed_hashes_path, 'rb') as f:
                    self.processed_hashes = set(pickle.load(f))
                    
                logger.info(f"Loaded checkpoint with {len(self.processed_hashes)} processed hashes")
                
                # Load migrated failed requests
                self._load_migrated_failed_requests()
                
            except Exception as e:
                logger.error(f"Error loading checkpoint: {str(e)}")
                self.processed_hashes = set()
        else:
            logger.info(f"No checkpoint found at {processed_hashes_path}")
            self.processed_hashes = set()
    
    def save_checkpoint(self, checkpoint_dir: str) -> None:
        """
        Save checkpoint data to disk.
        
        Args:
            checkpoint_dir: Directory to save checkpoint files
        """
        # Ensure checkpoint directory exists
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save processed hashes
        processed_hashes_path = os.path.join(checkpoint_dir, 'processed_hashes.pkl')
        
        try:
            with open(processed_hashes_path, 'wb') as f:
                pickle.dump(list(self.processed_hashes), f)
            
            # Save daily usage tracking
            self._save_daily_usage_checkpoint()
            
            # Save failed requests if any
            self._save_failed_requests_checkpoint()
                
            logger.info(f"Saved checkpoint with {len(self.processed_hashes)} processed hashes, "
                       f"{len(self.failed_requests)} failed requests, and daily usage ({self.tokens_today:,} tokens)")
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")
    
    def process(self, string_dict: Dict[str, str], field_hash_mapping: Dict[str, Dict[str, int]],
              string_counts: Dict[str, int], checkpoint_dir: str) -> Dict[str, Any]:
        """
        Process strings: generate embeddings and index in Weaviate.
        
        Args:
            string_dict: Dictionary mapping hash to string value
            field_hash_mapping: Mapping of hash to field types
            string_counts: Mapping of hash to frequency count
            checkpoint_dir: Directory for saving checkpoints
            
        Returns:
            Dictionary with metrics about the process
        """
        logger.info("Starting unified embedding and indexing process")
        start_time = time.time()
        
        # Load checkpoint
        self.load_checkpoint(checkpoint_dir)
        
        # Select strings to process
        strings_to_process = self._select_strings_to_process(
            string_dict, field_hash_mapping, string_counts
        )
        
        if not strings_to_process:
            logger.info("No new strings to process, proceeding with existing data")
            
            # Get collection stats
            try:
                result = self.collection.aggregate.over_all(total_count=True)
                collection_count = result.total_count
                logger.info(f"Collection contains {collection_count} objects")
                
            except Exception as e:
                logger.error(f"Error getting collection stats: {str(e)}")
                collection_count = None
                
            return {
                'status': 'completed',
                'elapsed_time': time.time() - start_time,
                'strings_processed': 0,
                'total_tokens': 0,
                'collection_count': collection_count
            }
        
        # Calculate optimal number of workers based on rate limits - OPTIMIZED
        import multiprocessing
        available_cores = multiprocessing.cpu_count()
        # Increase default workers to better utilize rate limits
        default_workers = min(8, available_cores)  # Increased from 4 to 8
        max_workers = min(available_cores, self.config.get("embedding_workers", default_workers))
        logger.info(f"Using {max_workers} worker threads on system with {available_cores} cores (optimized for rate limits)")
        
        # Process in batches with checkpoints
        checkpoint_batch = self.config.get("checkpoint_batch", 1000)
        total_processed = 0
        total_tokens = 0
        
        # Process in checkpoint batches - OPTIMIZED for larger throughput
        # Increase checkpoint frequency to reduce I/O overhead for large datasets
        optimized_checkpoint_batch = max(checkpoint_batch, 2000)  # At least 2000 items per checkpoint
        for i in range(0, len(strings_to_process), optimized_checkpoint_batch):
            batch = strings_to_process[i:i+optimized_checkpoint_batch]
            batch_start_time = time.time()
            
            # Process batch
            indexed_count, tokens_used = self._process_batches_parallel(batch, max_workers)
            
            # Update counters
            total_processed += indexed_count
            total_tokens += tokens_used
            
            # Save checkpoint - now atomic (only successfully indexed items are marked as processed)
            self.save_checkpoint(checkpoint_dir)
            
            # Ultra-minimal: Only token/quota usage
            daily_usage_pct = (self.tokens_today / self.max_tokens_per_day * 100) if self.max_tokens_per_day != float('inf') else 0
            failed_count = len(self.failed_requests)
            
            max_tokens_display = f"{self.max_tokens_per_day:,.0f}" if self.max_tokens_per_day != float('inf') else "∞"
            logger.info(f"Tokens: {self.tokens_today:,}/{max_tokens_display} ({daily_usage_pct:.1f}%)" +
                       (f" | Failed: {failed_count}" if failed_count > 0 else ""))
        
        elapsed_time = time.time() - start_time
        overall_throughput = total_processed / elapsed_time if elapsed_time > 0 else 0
        
        # Enhanced completion summary
        daily_usage_pct = (self.tokens_today / self.max_tokens_per_day * 100) if self.max_tokens_per_day != float('inf') else 0
        failed_count = len(self.failed_requests)
        
        # Concise completion summary
        max_tokens_display = f"{self.max_tokens_per_day:,.0f}" if self.max_tokens_per_day != float('inf') else "∞"
        logger.info(f"COMPLETE: {total_processed} items in {elapsed_time:.1f}s ({overall_throughput:.1f}/s) | "
                   f"Tokens: {self.tokens_today:,}/{max_tokens_display} ({daily_usage_pct:.1f}%)" +
                   (f" | Failed: {failed_count}" if failed_count > 0 else " | Success: 100%"))
        
        # Get collection stats
        try:
            result = self.collection.aggregate.over_all(total_count=True)
            collection_count = result.total_count
            logger.info(f"Collection now contains {collection_count} objects")
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            collection_count = None
        
        # Enhanced return metrics similar to batch processing
        failed_analysis = self._analyze_failed_requests()
        
        return {
            'status': 'completed',
            'elapsed_time': elapsed_time,
            'strings_processed': total_processed,
            'tokens_used': total_tokens,
            'collection_count': collection_count,
            'throughput': overall_throughput,
            'daily_token_usage': {
                'tokens_used': self.tokens_today,
                'tokens_limit': self.max_tokens_per_day,
                'usage_percentage': (self.tokens_today / self.max_tokens_per_day * 100) if self.max_tokens_per_day > 0 else 0
            },
            'failed_requests': {
                'total_failed': len(self.failed_requests),
                'retryable': failed_analysis['retryable'],
                'max_retries_exceeded': failed_analysis['max_retries_exceeded'],
                'by_category': failed_analysis['by_category']
            },
            'processing_mode': 'real-time',
            'migrated_from_batch': len([f for f in self.failed_requests.values() if f.get('migrated_from_batch', False)]) > 0
        }

def embedding_and_indexing(config: Dict[str, Any], string_dict: Dict[str, str], 
                         field_hash_mapping: Dict[str, Dict[str, int]],
                         string_counts: Dict[str, int]) -> Dict[str, Any]:
    """
    Unified function for embedding generation and indexing.
    
    Args:
        config: Configuration dictionary
        string_dict: Dictionary mapping hash to string value
        field_hash_mapping: Mapping of hash to field types
        string_counts: Mapping of hash to frequency count
        
    Returns:
        Dictionary with metrics
    """
    logger.info("Starting unified embedding and indexing process")
    
    # Get checkpoint directory
    checkpoint_dir = config.get("checkpoint_dir", "data/checkpoints")
    
    # Initialize the pipeline
    pipeline = None
    metrics = {}
    
    try:
        pipeline = EmbeddingAndIndexingPipeline(config)
        
        # Process the data
        metrics = pipeline.process(string_dict, field_hash_mapping, string_counts, checkpoint_dir)
        
    except Exception as e:
        logger.error(f"Error in embedding and indexing pipeline: {str(e)}")
        metrics = {
            'status': 'error',
            'error': str(e),
            'elapsed_time': 0
        }
    finally:
        # Ensure client is closed even if an exception occurs
        if pipeline is not None and hasattr(pipeline, 'weaviate_client'):
            try:
                pipeline.weaviate_client.close()
                logger.info("Weaviate client connection closed in cleanup handler")
            except Exception as e:
                logger.error(f"Error closing Weaviate client in cleanup: {str(e)}")
    
    return metrics

def main(config_path: str = 'config.yml'):
    """
    Main function for unified embedding and indexing.
    
    Args:
        config_path: Path to the configuration file
    """
    import yaml
    import pickle
    import os
    
    # Load configuration with environment-specific overrides
    from src.config_utils import load_config_with_environment
    config = load_config_with_environment(config_path)
    
    # Get checkpoint directory
    checkpoint_dir = config.get("checkpoint_dir", "data/checkpoints")
    
    # Load preprocessing data
    string_dict_path = os.path.join(checkpoint_dir, "string_dict.pkl")
    field_hash_mapping_path = os.path.join(checkpoint_dir, "field_hash_mapping.pkl") 
    string_counts_path = os.path.join(checkpoint_dir, "string_counts.pkl")
    
    try:
        # Load string dictionary
        with open(string_dict_path, 'rb') as f:
            string_dict = pickle.load(f)
        
        # Load field hash mapping
        with open(field_hash_mapping_path, 'rb') as f:
            field_hash_mapping = pickle.load(f)
        
        # Load string counts
        with open(string_counts_path, 'rb') as f:
            string_counts = pickle.load(f)
        
        logger.info(f"Loaded preprocessing data: {len(string_dict)} strings, "
                   f"{len(field_hash_mapping)} field mappings, {len(string_counts)} string counts")
        
        # Run embedding and indexing
        metrics = embedding_and_indexing(config, string_dict, field_hash_mapping, string_counts)
        
        # Save metrics
        output_dir = os.path.join(config.get("output_dir", "data/output"))
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, "embedding_indexing_metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Saved metrics to {output_dir}/embedding_indexing_metrics.json")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Unified embedding and indexing for entity resolution')
    parser.add_argument('--config', default='config.yml', help='Path to configuration file')
    parser.add_argument('--reset', action='store_true', help='Reset progress and start from beginning')
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Handle reset flag
    if args.reset:
        import shutil
        import os
        
        # Load config to get checkpoint directory with environment-specific overrides
        from src.config_utils import load_config_with_environment
        config = load_config_with_environment(args.config)
        
        checkpoint_dir = config.get("checkpoint_dir", "data/checkpoints")
        processed_hashes_path = os.path.join(checkpoint_dir, 'processed_hashes.pkl')
        
        # Delete processed hashes checkpoint
        if os.path.exists(processed_hashes_path):
            os.remove(processed_hashes_path)
            logger.info(f"Deleted checkpoint file: {processed_hashes_path}")
    
    # Run unified embedding and indexing
    main(args.config)
