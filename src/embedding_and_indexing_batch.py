"""
Batch Embedding and Indexing Module for Entity Resolution

This module implements OpenAI's Batch API for embedding generation, providing
cost-effective processing at 50% lower cost with 24-hour turnaround time.

Features:
- Automated queue management: Maintains 16 active batches, automatically submitting
  new ones as slots free up with 30-minute polling intervals
- OPTIMIZED quota usage: Aggressively maintains close to 800K enqueued requests
- Automatic state persistence and recovery
- Manual override options for direct control

PERFORMANCE OPTIMIZATIONS (2025-07-02):
- CORRECTED: 800K IS the conservative limit - don't throttle it further!
- FIXED: Quota usage now maintains 99% of 800K = ~792K requests  
- FIXED: Post-submission threshold at 99% of 800K
- CORRECTED: No additional safety margins beyond 1% 
- REMOVED: All conservative buffers - 800K is already 80% of 1M limit
- TARGET: Maintain exactly 800K enqueued requests as intended

The automated queue system eliminates manual batch management while maintaining
the full 800K conservative request quota without additional throttling.
"""

import os
import sys
import logging
import pickle
import time
import json
import uuid
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from tqdm import tqdm
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.config import VectorDistances
from weaviate.util import generate_uuid5
from openai import OpenAI
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

logger = logging.getLogger(__name__)

# Import tiktoken for accurate token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not available - using fallback token estimation")

class BatchJobStatus:
    """Enum-like class for batch job statuses."""
    PENDING = "pending"
    VALIDATING = "validating"
    IN_PROGRESS = "in_progress"
    FINALIZING = "finalizing"
    CANCELLING = "cancelling"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"

class BatchEmbeddingPipeline:
    """
    Batch-based pipeline for embedding generation using OpenAI's Batch API.
    Provides cost-effective processing with 50% lower cost and 24-hour turnaround.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the batch pipeline with configuration.
        
        Args:
            config: Configuration dictionary with batch processing parameters
        """
        self.config = config
        self.weaviate_client = None  # Initialize as None first
        self.collection = None
        
        # OpenAI API configuration
        self.api_key = config.get("openai_api_key", os.environ.get("OPENAI_API_KEY"))
        if not self.api_key:
            raise ValueError("OpenAI API key not found in config or environment variables")
            
        self.embedding_model = config.get("embedding_model", "text-embedding-3-small")
        self.embedding_dimensions = config.get("embedding_dimensions", 1536)
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=self.api_key)
        
        # Batch processing parameters
        self.batch_size = config.get("batch_embedding_size", 50000)  # Larger batches for cost efficiency
        self.max_requests_per_file = config.get("max_requests_per_file", 50000)
        self.manual_polling = config.get("batch_manual_polling", True)  # Manual polling by default
        self.poll_interval = config.get("batch_poll_interval", 300)  # 5 minutes (only used in auto mode)
        self.max_wait_time = config.get("batch_max_wait_time", 86400)  # 24 hours
        
        # Token quota management - 800K IS ALREADY CONSERVATIVE  
        self.token_quota_limit = config.get("token_quota_limit", 500_000_000)  # 500M default OpenAI limit
        self.quota_safety_margin = config.get("quota_safety_margin", 0.01)  # CORRECTED: 1% safety margin - 800K is the target!
        self.max_concurrent_jobs = config.get("max_concurrent_jobs", 50)  # Limit concurrent jobs
        
        # Request quota management (1M enqueued requests limit)
        # CORRECTED: Set to 102% of 800K to ensure we ALWAYS hit exactly 800K!
        self.request_quota_target = 800_000  # Target exactly 800K requests
        self.request_quota_limit = config.get("request_quota_limit", 816_000)  # 102% of 800K for safety buffer
        self.max_requests_per_job = config.get("max_requests_per_job", 50_000)  # 50K requests per job
        
        # Automated queue management - maintain 16 active batches with auto-submission
        self.max_active_batches = config.get("max_active_batches", 16)  # Max concurrent batches in queue
        self.queue_poll_interval = config.get("queue_poll_interval", 1800)  # 30 minutes between status checks
        self.use_automated_queue = config.get("use_automated_queue", True)  # Enable automated queue management
        
        # Queue state tracking
        self.active_batch_queue = []  # List of currently active batch job IDs
        self.pending_batches = []  # List of batch data waiting for submission
        self.completed_batches = []  # List of completed batch job IDs
        
        # Embedding fields configuration
        self.embed_fields = config.get("embed_fields", ["composite", "person", "title"])
        self.skip_fields = config.get("skip_fields", ["provision", "subjects", "personId"])
        
        # Initialize Weaviate client (optional - only needed for download/indexing)
        self.weaviate_client = None
        self.collection = None
        self._weaviate_init_attempted = False
        
        # Try to initialize Weaviate, but don't fail if it's not available
        try:
            self.weaviate_client = self._init_weaviate_client()
            self.collection = self._ensure_schema_exists()
            logger.info("Weaviate connection established - ready for both batch creation and download")
        except Exception as e:
            logger.warning(f"Weaviate not available during initialization: {e}")
            logger.info("Batch job creation will work without Weaviate (download will require Weaviate later)")
        
        # Initialize batch job tracking
        self.batch_jobs = {}  # job_id -> job_info
        self.processed_hashes = set()
        self.blacklisted_files = set()  # input_file_ids to avoid reprocessing
        
        # Failed request tracking and retry system
        self.failed_requests = {}  # custom_id -> {error_info, retry_count, last_attempt}
        
        # Error handling configuration
        error_config = config.get("error_handling", {})
        self.max_retry_attempts = error_config.get("max_retry_attempts", 3)
        self.enable_automatic_retries = error_config.get("enable_automatic_retries", True)
        self.retry_batch_creation = error_config.get("retry_batch_creation", True)
        self.initial_retry_delay = error_config.get("initial_retry_delay", 300)
        self.max_retry_delay = error_config.get("max_retry_delay", 3600)
        self.backoff_multiplier = error_config.get("backoff_multiplier", 2)
        self.enable_detailed_error_reporting = error_config.get("enable_detailed_error_reporting", True)
        self.max_sample_errors = error_config.get("max_sample_errors", 10)
        self.cleanup_successful_retries_enabled = error_config.get("cleanup_successful_retries", True)
        
        # Error categorization with configurable retry behavior
        retryable_types = error_config.get("retryable_error_types", ["rate_limit", "server", "quota"])
        permanent_types = error_config.get("permanent_error_types", ["validation"])
        
        self.error_categories = {
            'rate_limit': ['rate_limit_exceeded', 'rate_limit'],
            'quota': ['quota_exceeded', 'insufficient_quota'],
            'validation': ['invalid_request', 'validation_error'],
            'server': ['internal_error', 'server_error', 'timeout'],
            'other': []  # catch-all for uncategorized errors
        }
        
        # Store retryable vs permanent categorization
        self.retryable_categories = set(retryable_types)
        self.permanent_categories = set(permanent_types)
        
        logger.info(f"Initialized BatchEmbeddingPipeline with model {self.embedding_model}")
    
    def _ensure_weaviate_connection(self):
        """
        Ensure Weaviate connection is established when needed for download/indexing operations.
        """
        if self.weaviate_client is None and not self._weaviate_init_attempted:
            self._weaviate_init_attempted = True
            try:
                logger.info("Initializing Weaviate connection for download/indexing operations...")
                self.weaviate_client = self._init_weaviate_client()
                self.collection = self._ensure_schema_exists()
                logger.info("Weaviate connection established successfully")
            except Exception as e:
                error_msg = f"Failed to connect to Weaviate for download/indexing: {e}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
        elif self.weaviate_client is None:
            raise RuntimeError("Weaviate connection required but failed during previous attempt")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with proper cleanup."""
        self.close()
    
    def close(self):
        """Properly close all connections."""
        if self.weaviate_client is not None:
            try:
                self.weaviate_client.close()
                logger.debug("Weaviate client connection closed")
            except Exception as e:
                logger.debug(f"Error closing Weaviate client: {e}")
            finally:
                self.weaviate_client = None
                self.collection = None
    
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
            # Create connection parameters with extended timeouts for batch operations
            from weaviate.connect import ConnectionParams
            
            # Get timeout settings from config with production defaults
            grpc_timeout = self.config.get('grpc_keepalive_timeout_ms', 60000)  # 60 seconds default
            write_timeout = self.config.get('weaviate_write_timeout', 600)  # 10 minutes default
            read_timeout = self.config.get('weaviate_read_timeout', 600)  # 10 minutes default
            
            connection_params = ConnectionParams.from_params(
                http_host=host,
                http_port=port,
                http_secure=secure,
                grpc_host=host,  # Using same host for gRPC
                grpc_port=grpc_port,
                grpc_secure=secure,  # Using same security setting for gRPC
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
                    collection_exists = False
                except Exception as e:
                    logger.info(f"No existing EntityString collection found or cannot be deleted: {e}")
                    collection_exists = True
            else:
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
            
            # Create collection with dynamic embedding model configuration
            # Check if new embedding config exists, otherwise use legacy
            embedding_config = self.config.get('embedding', {})
            if 'providers' in embedding_config and 'openai' in embedding_config['providers']:
                openai_config = embedding_config['providers']['openai']
                model = openai_config.get('model', 'text-embedding-3-small')
                dimensions = openai_config.get('dimensions', 1536)
            else:
                # Fallback to legacy configuration
                model = self.config.get('embedding_model', 'text-embedding-3-small')
                dimensions = self.config.get('embedding_dimensions', 1536)
            
            logger.info(f"Creating new EntityString collection with OpenAI Vectorizer ({model}, {dimensions}D)")
            collection = self.weaviate_client.collections.create(
                name="EntityString",
                description="Collection for entity string values with their embeddings",
                vectorizer_config=Configure.Vectorizer.text2vec_openai(
                    model=model,
                    dimensions=dimensions
                ),
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
            
            logger.info(f"Created EntityString collection with OpenAI Vectorizer configuration")
            return collection
        except Exception as e:
            logger.error(f"Error creating Weaviate schema: {str(e)}")
            raise
    
    def _create_batch_requests_file(self, strings_to_process: List[Tuple[str, str, str, int]], 
                                  output_path: str) -> Dict[str, Any]:
        """
        Create a JSONL file with batch embedding requests.
        
        Args:
            strings_to_process: List of (hash, text, field_type, frequency) tuples
            output_path: Path to save the JSONL file
            
        Returns:
            Dictionary with file metadata
        """
        logger.info(f"Creating batch requests file with {len(strings_to_process)} requests")
        
        # Create custom ID mapping for tracking
        custom_id_mapping = {}
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, (hash_val, text, field_type, frequency) in enumerate(strings_to_process):
                # Create unique custom ID
                custom_id = f"{hash_val}_{field_type}_{i}"
                custom_id_mapping[custom_id] = {
                    'hash_value': hash_val,
                    'original_string': text,
                    'field_type': field_type,
                    'frequency': frequency,
                    'index': i
                }
                
                # Create batch request
                request = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/embeddings",
                    "body": {
                        "model": self.embedding_model,
                        "input": text,
                        "encoding_format": "float"
                    }
                }
                
                # Write as JSONL
                f.write(json.dumps(request) + '\n')
        
        file_metadata = {
            'path': output_path,
            'request_count': len(strings_to_process),
            'custom_id_mapping': custom_id_mapping,
            'created_at': time.time()
        }
        
        logger.info(f"Created batch requests file: {output_path} ({len(strings_to_process)} requests)")
        return file_metadata
    
    def _is_file_content_blacklisted(self, file_path: str) -> bool:
        """
        Check if the content of a JSONL file matches any blacklisted content.
        This prevents reprocessing of files with identical content to previous attempts.
        """
        if not self.blacklisted_files:
            return False
        
        try:
            # For now, we'll use a simple approach: check if any existing OpenAI files
            # would conflict with this content. Since we can't predict file IDs,
            # we'll rely on the blacklist of actual file IDs from OpenAI.
            # The real protection happens when we check quota and existing jobs.
            
            # This is a placeholder - the main blacklist protection happens at the
            # job creation level where we check against actual OpenAI file IDs
            return False
            
        except Exception as e:
            logger.debug(f"Error checking blacklist for {file_path}: {e}")
            return False
    
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5)
    )
    def _upload_batch_file(self, file_path: str) -> str:
        """
        Upload batch requests file to OpenAI.
        
        Args:
            file_path: Path to the JSONL file
            
        Returns:
            File ID from OpenAI
        """
        logger.info(f"Uploading batch file: {file_path}")
        
        try:
            with open(file_path, 'rb') as f:
                file_response = self.openai_client.files.create(
                    file=f,
                    purpose='batch'
                )
            
            logger.info(f"Successfully uploaded file, ID: {file_response.id}")
            return file_response.id
            
        except Exception as e:
            logger.error(f"Error uploading batch file: {str(e)}")
            raise
    
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5)
    )
    def _create_batch_job(self, input_file_id: str, description: str = None) -> str:
        """
        Create a batch job for processing embeddings.
        
        Args:
            input_file_id: File ID from OpenAI upload
            description: Optional job description
            
        Returns:
            Batch job ID
        """
        logger.info(f"Creating batch job for file: {input_file_id}")
        
        try:
            batch_response = self.openai_client.batches.create(
                input_file_id=input_file_id,
                endpoint="/v1/embeddings",
                completion_window="24h",
                metadata={
                    "description": description or "Entity resolution embeddings batch job",
                    "created_by": "embedding_and_indexing_batch"
                }
            )
            
            logger.info(f"Created batch job: {batch_response.id}")
            return batch_response.id
            
        except Exception as e:
            logger.error(f"Error creating batch job: {str(e)}")
            raise
    
    def _poll_batch_job(self, batch_id: str) -> Dict[str, Any]:
        """
        Poll batch job status until completion.
        
        Args:
            batch_id: Batch job ID
            
        Returns:
            Final batch job status
        """
        logger.info(f"Polling batch job: {batch_id}")
        
        start_time = time.time()
        last_status = None
        
        with tqdm(desc=f"Waiting for batch job {batch_id[:8]}...", 
                 bar_format="{l_bar}{bar}| {elapsed}<{remaining}") as pbar:
            
            while time.time() - start_time < self.max_wait_time:
                try:
                    batch_status = self.openai_client.batches.retrieve(batch_id)
                    current_status = batch_status.status
                    
                    # Update progress if status changed
                    if current_status != last_status:
                        logger.info(f"Batch job {batch_id} status: {current_status}")
                        last_status = current_status
                        pbar.set_description(f"Batch job {batch_id[:8]} - {current_status}")
                    
                    # Check if job is complete
                    if current_status == BatchJobStatus.COMPLETED:
                        logger.info(f"Batch job {batch_id} completed successfully")
                        return {
                            'status': current_status,
                            'output_file_id': batch_status.output_file_id,
                            'request_counts': batch_status.request_counts,
                            'batch_data': batch_status
                        }
                    elif current_status in [BatchJobStatus.FAILED, BatchJobStatus.EXPIRED, BatchJobStatus.CANCELLED]:
                        logger.error(f"Batch job {batch_id} failed with status: {current_status}")
                        return {
                            'status': current_status,
                            'error': f"Job failed with status: {current_status}",
                            'batch_data': batch_status
                        }
                    
                    # Wait before next poll
                    time.sleep(self.poll_interval)
                    pbar.update(1)
                    
                except Exception as e:
                    logger.error(f"Error polling batch job {batch_id}: {str(e)}")
                    time.sleep(self.poll_interval)
        
        # Timeout reached
        logger.error(f"Batch job {batch_id} timed out after {self.max_wait_time} seconds")
        return {
            'status': 'timeout',
            'error': f"Job timed out after {self.max_wait_time} seconds"
        }
    
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(3)
    )
    def _download_batch_results(self, output_file_id: str, output_path: str) -> str:
        """
        Download batch job results.
        
        Args:
            output_file_id: Output file ID from completed batch job
            output_path: Local path to save results
            
        Returns:
            Path to downloaded results file
        """
        logger.info(f"Downloading batch results from file: {output_file_id}")
        
        try:
            # Get file content with extended timeout
            import httpx
            client = self.openai_client._client
            old_timeout = client.timeout
            try:
                # Extend timeout to 5 minutes for large file downloads
                client.timeout = httpx.Timeout(300.0)
                file_response = self.openai_client.files.content(output_file_id)
            finally:
                # Restore original timeout
                client.timeout = old_timeout
            
            # Save to local file
            with open(output_path, 'wb') as f:
                f.write(file_response.content)
            
            logger.info(f"Downloaded batch results to: {output_path}")
            return output_path
            
        except Exception as e:
            error_msg = str(e)
            # Check if it's a gateway timeout
            if "504" in error_msg or "Gateway time-out" in error_msg:
                logger.error(f"OpenAI API gateway timeout downloading {output_file_id}. This is an OpenAI server issue.")
                logger.info("You can try again later when OpenAI's servers are less busy.")
                raise Exception(f"OpenAI gateway timeout (server issue): {output_file_id}")
            else:
                logger.error(f"Error downloading batch results: {error_msg}")
                raise
    
    def _process_batch_results(self, results_path: str, custom_id_mapping: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process downloaded batch results and prepare for indexing.
        
        Args:
            results_path: Path to downloaded results file
            custom_id_mapping: Mapping of custom IDs to original data
            
        Returns:
            List of items ready for indexing
        """
        logger.info(f"Processing batch results from: {results_path}")
        
        items_to_index = []
        successful_requests = 0
        failed_requests = 0
        
        with open(results_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    result = json.loads(line.strip())
                    custom_id = result.get('custom_id')
                    
                    if custom_id not in custom_id_mapping:
                        logger.warning(f"Unknown custom_id in results: {custom_id}")
                        continue
                    
                    # Check if request was successful
                    response = result.get('response')
                    if response and response.get('status_code') == 200:
                        # Extract embedding
                        body = response.get('body', {})
                        data = body.get('data', [])
                        
                        if data and len(data) > 0:
                            embedding = data[0].get('embedding')
                            if embedding:
                                # Get original item data
                                item_data = custom_id_mapping[custom_id]
                                
                                # Prepare for indexing
                                items_to_index.append({
                                    'hash_value': item_data['hash_value'],
                                    'original_string': item_data['original_string'],
                                    'field_type': item_data['field_type'],
                                    'frequency': item_data['frequency'],
                                    'vector': np.array(embedding, dtype=np.float32)
                                })
                                successful_requests += 1
                            else:
                                logger.warning(f"No embedding found for custom_id: {custom_id}")
                                failed_requests += 1
                        else:
                            logger.warning(f"No data in response for custom_id: {custom_id}")
                            failed_requests += 1
                    else:
                        # Request failed
                        error_info = result.get('error', {})
                        logger.warning(f"Failed request for custom_id {custom_id}: {error_info}")
                        
                        # Track failed request for retry
                        self._track_failed_request(custom_id, error_info)
                        failed_requests += 1
                        
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing result line: {str(e)}")
                    failed_requests += 1
                except Exception as e:
                    logger.error(f"Error processing result: {str(e)}")
                    failed_requests += 1
        
        logger.info(f"Processed batch results: {successful_requests} successful, {failed_requests} failed")
        return items_to_index
    
    def _process_batch_results_without_mapping(self, results_path: str) -> List[Dict[str, Any]]:
        """
        Process downloaded batch results for recovered jobs without custom_id_mapping.
        This creates a simplified mapping based on the custom_id format.
        
        Args:
            results_path: Path to downloaded results file
            
        Returns:
            List of items ready for indexing
        """
        logger.info(f"Processing batch results (recovered job) from: {results_path}")
        
        items_to_index = []
        successful_requests = 0
        failed_requests = 0
        
        with open(results_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    result = json.loads(line.strip())
                    custom_id = result.get('custom_id')
                    
                    if not custom_id:
                        logger.warning("No custom_id in result")
                        continue
                    
                    # Check if request was successful
                    response = result.get('response')
                    if response and response.get('status_code') == 200:
                        # Extract embedding
                        body = response.get('body', {})
                        data = body.get('data', [])
                        
                        if data and len(data) > 0:
                            embedding = data[0].get('embedding')
                            if embedding:
                                # Parse custom_id to extract info: format is "hash_field_index"
                                try:
                                    parts = custom_id.split('_')
                                    if len(parts) >= 3:
                                        hash_value = '_'.join(parts[:-2])  # Everything except last 2 parts
                                        field_type = parts[-2]
                                        index = parts[-1]
                                        
                                        # We don't have the original string, but we can still index
                                        items_to_index.append({
                                            'hash_value': hash_value,
                                            'original_string': f"Recovered string {hash_value}",  # Placeholder
                                            'field_type': field_type,
                                            'frequency': 1,  # Default frequency
                                            'vector': np.array(embedding, dtype=np.float32)
                                        })
                                        successful_requests += 1
                                    else:
                                        logger.warning(f"Cannot parse custom_id format: {custom_id}")
                                        failed_requests += 1
                                except Exception as parse_error:
                                    logger.warning(f"Error parsing custom_id {custom_id}: {parse_error}")
                                    failed_requests += 1
                            else:
                                logger.warning(f"No embedding found for custom_id: {custom_id}")
                                failed_requests += 1
                        else:
                            logger.warning(f"No data in response for custom_id: {custom_id}")
                            failed_requests += 1
                    else:
                        # Request failed
                        error_info = result.get('error', {})
                        logger.warning(f"Failed request for custom_id {custom_id}: {error_info}")
                        
                        # Track failed request for retry
                        self._track_failed_request(custom_id, error_info)
                        failed_requests += 1
                        
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing result line: {str(e)}")
                    failed_requests += 1
                except Exception as e:
                    logger.error(f"Error processing result: {str(e)}")
                    failed_requests += 1
        
        logger.info(f"Processed batch results (recovered): {successful_requests} successful, {failed_requests} failed")
        logger.warning("Note: Original string values not available for recovered jobs - using placeholders")
        return items_to_index
    
    def _track_failed_request(self, custom_id: str, error_info: Dict[str, Any]) -> None:
        """
        Track a failed request for retry handling.
        
        Args:
            custom_id: The custom ID of the failed request
            error_info: Error information from OpenAI response
        """
        if custom_id not in self.failed_requests:
            self.failed_requests[custom_id] = {
                'error_info': error_info,
                'retry_count': 0,
                'first_attempt': time.time(),
                'last_attempt': time.time(),
                'error_category': self._categorize_error(error_info)
            }
        else:
            # Update existing failed request
            self.failed_requests[custom_id]['retry_count'] += 1
            self.failed_requests[custom_id]['last_attempt'] = time.time()
            self.failed_requests[custom_id]['error_info'] = error_info  # Update with latest error
    
    def _categorize_error(self, error_info: Dict[str, Any]) -> str:
        """
        Categorize error based on error information.
        
        Args:
            error_info: Error information from OpenAI
            
        Returns:
            Error category string
        """
        error_code = error_info.get('code', '').lower()
        error_message = error_info.get('message', '').lower()
        
        for category, keywords in self.error_categories.items():
            if category == 'other':
                continue
            for keyword in keywords:
                if keyword in error_code or keyword in error_message:
                    return category
        
        return 'other'
    
    def _process_error_file(self, error_file_id: str, job_id: str) -> Dict[str, Any]:
        """
        Download and process error file from OpenAI batch job.
        
        Args:
            error_file_id: OpenAI file ID for the error file
            job_id: Batch job ID for context
            
        Returns:
            Dictionary with error analysis
        """
        logger.info(f"Processing error file {error_file_id} for job {job_id}")
        
        error_analysis = {
            'total_errors': 0,
            'errors_by_category': {},
            'failed_custom_ids': [],
            'sample_errors': []
        }
        
        try:
            # Download error file content
            error_content = self.openai_client.files.content(error_file_id)
            error_text = error_content.content.decode('utf-8')
            
            # Process each error line
            for line_num, line in enumerate(error_text.split('\n'), 1):
                if not line.strip():
                    continue
                    
                try:
                    error_data = json.loads(line.strip())
                    custom_id = error_data.get('custom_id')
                    error_info = error_data.get('error', {})
                    
                    if custom_id:
                        # Track this failed request
                        self._track_failed_request(custom_id, error_info)
                        error_analysis['failed_custom_ids'].append(custom_id)
                        
                        # Categorize error
                        category = self._categorize_error(error_info)
                        error_analysis['errors_by_category'][category] = error_analysis['errors_by_category'].get(category, 0) + 1
                        
                        # Keep sample errors (configurable limit)
                        if len(error_analysis['sample_errors']) < self.max_sample_errors:
                            error_analysis['sample_errors'].append({
                                'custom_id': custom_id,
                                'error': error_info,
                                'category': category,
                                'line_number': line_num
                            })
                        
                        error_analysis['total_errors'] += 1
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"Error parsing error file line {line_num}: {e}")
                    continue
                    
            logger.info(f"Processed error file: {error_analysis['total_errors']} errors, "
                       f"categories: {error_analysis['errors_by_category']}")
                       
        except Exception as e:
            logger.error(f"Error processing error file {error_file_id}: {e}")
            
        return error_analysis
    
    def get_failed_requests_summary(self) -> Dict[str, Any]:
        """
        Get summary of failed requests for reporting.
        
        Returns:
            Dictionary with failure statistics
        """
        summary = {
            'total_failed': len(self.failed_requests),
            'by_category': {},
            'by_retry_count': {},
            'retryable_requests': 0,
            'max_retries_exceeded': 0
        }
        
        for custom_id, failure_data in self.failed_requests.items():
            category = failure_data['error_category']
            retry_count = failure_data['retry_count']
            
            # Count by category
            summary['by_category'][category] = summary['by_category'].get(category, 0) + 1
            
            # Count by retry attempts
            summary['by_retry_count'][retry_count] = summary['by_retry_count'].get(retry_count, 0) + 1
            
            # Count retryable vs max retries exceeded
            if retry_count < self.max_retry_attempts:
                summary['retryable_requests'] += 1
            else:
                summary['max_retries_exceeded'] += 1
                
        return summary
    
    def get_retryable_failed_requests(self) -> List[Dict[str, Any]]:
        """
        Get failed requests that can be retried (haven't exceeded max attempts).
        
        Returns:
            List of failed request data ready for retry
        """
        retryable_requests = []
        
        for custom_id, failure_data in self.failed_requests.items():
            # Check if this error type is retryable
            error_category = failure_data.get('error_category', 'other')
            if error_category in self.permanent_categories:
                continue  # Skip permanent errors
                
            if failure_data['retry_count'] < self.max_retry_attempts:
                # Check if enough time has passed for retry (exponential backoff)
                time_since_last_attempt = time.time() - failure_data['last_attempt']
                min_wait_time = min(
                    self.initial_retry_delay * (self.backoff_multiplier ** failure_data['retry_count']), 
                    self.max_retry_delay
                )
                
                if time_since_last_attempt >= min_wait_time:
                    retryable_requests.append({
                        'custom_id': custom_id,
                        'failure_data': failure_data
                    })
        
        return retryable_requests
    
    def create_retry_batch(self, hash_lookup: Dict[str, str], string_dict: Dict[str, str], 
                          checkpoint_dir: str) -> Optional[str]:
        """
        Create a new batch file containing only failed requests that can be retried.
        
        Args:
            hash_lookup: Hash to string mapping
            string_dict: String dictionary for text lookup
            checkpoint_dir: Directory for saving batch files
            
        Returns:
            Batch job ID if successful, None otherwise
        """
        retryable_requests = self.get_retryable_failed_requests()
        
        if not retryable_requests:
            logger.info("No failed requests are ready for retry")
            return None
            
        logger.info(f"Creating retry batch with {len(retryable_requests)} failed requests")
        
        # Prepare batch data for retry
        retry_batch_data = []
        custom_id_mapping = {}
        
        for request_data in retryable_requests:
            custom_id = request_data['custom_id']
            
            # Parse custom_id to get hash and field info
            # Format: "hash_{hash_value}_field_{field_type}"
            try:
                parts = custom_id.split('_')
                if len(parts) >= 4 and parts[0] == 'hash' and parts[2] == 'field':
                    hash_value = parts[1]
                    field_type = parts[3]
                    
                    # Get original text from hash_lookup and string_dict
                    string_hash = hash_lookup.get(hash_value)
                    if string_hash and string_hash in string_dict:
                        original_text = string_dict[string_hash]
                        
                        # Add to retry batch
                        retry_batch_data.append((hash_value, original_text, field_type, 1))  # frequency=1 for retry
                        
                        # Create mapping for processing results
                        custom_id_mapping[custom_id] = {
                            'hash_value': hash_value,
                            'original_string': original_text,
                            'field_type': field_type,
                            'frequency': 1,
                            'is_retry': True,
                            'retry_count': request_data['failure_data']['retry_count'] + 1
                        }
                    else:
                        logger.warning(f"Could not find original text for retry custom_id {custom_id}")
                else:
                    logger.warning(f"Invalid custom_id format for retry: {custom_id}")
                    
            except Exception as e:
                logger.error(f"Error parsing custom_id {custom_id} for retry: {e}")
                continue
        
        if not retry_batch_data:
            logger.warning("No valid retry requests found after parsing custom_ids")
            return None
            
        # Submit retry batch
        batch_idx = len(self.batch_jobs)  # Use next available batch index
        job_id = self._submit_single_batch(retry_batch_data, batch_idx, checkpoint_dir)
        
        if job_id:
            # Mark this as a retry batch in job metadata
            self.batch_jobs[job_id]['is_retry_batch'] = True
            self.batch_jobs[job_id]['retry_request_count'] = len(retry_batch_data)
            self.batch_jobs[job_id]['file_metadata']['custom_id_mapping'] = custom_id_mapping
            
            logger.info(f"✅ Created retry batch {job_id} with {len(retry_batch_data)} requests")
            
            # Update retry attempts for these requests
            for request_data in retryable_requests:
                custom_id = request_data['custom_id']
                if custom_id in self.failed_requests:
                    self.failed_requests[custom_id]['retry_count'] += 1
                    self.failed_requests[custom_id]['last_attempt'] = time.time()
                    
        return job_id
    
    def cleanup_successful_retries(self) -> int:
        """
        Remove successfully processed requests from failed_requests tracking.
        
        Returns:
            Number of failed requests cleaned up
        """
        cleaned_count = 0
        
        # Get list of custom_ids to remove (can't modify dict while iterating)
        to_remove = []
        
        for custom_id, failure_data in self.failed_requests.items():
            # Parse custom_id to get hash
            try:
                parts = custom_id.split('_')
                if len(parts) >= 2 and parts[0] == 'hash':
                    hash_value = parts[1]
                    
                    # If this hash is now in processed_hashes, the retry succeeded
                    if hash_value in self.processed_hashes:
                        to_remove.append(custom_id)
                        cleaned_count += 1
                        
            except Exception as e:
                logger.debug(f"Error parsing custom_id {custom_id} for cleanup: {e}")
                continue
        
        # Remove successful retries
        for custom_id in to_remove:
            del self.failed_requests[custom_id]
            
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} successfully retried requests")
            
        return cleaned_count
    
    def _index_embeddings_batch(self, items_to_index: List[Dict[str, Any]]) -> int:
        """
        Index a batch of embeddings in Weaviate.
        
        Args:
            items_to_index: List of items to index with vectors
            
        Returns:
            Number of successfully indexed items
        """
        logger.info(f"Indexing batch of {len(items_to_index)} items in Weaviate")
        
        # Ensure Weaviate connection for indexing operations
        self._ensure_weaviate_connection()
        
        indexed_count = 0
        
        try:
            # FIXED: Reduce batch size to prevent gRPC timeout with large datasets
            # For production with 50K items, use smaller batches to avoid deadline exceeded
            optimal_batch_size = min(100, len(items_to_index))  # REDUCED: 1000 → 100 to prevent gRPC timeout
            
            logger.info(f"Using batch size {optimal_batch_size} for {len(items_to_index)} items")
            with self.collection.batch.fixed_size(batch_size=optimal_batch_size) as batch_writer:
                for i, item in enumerate(items_to_index):
                    # Progress logging every 5000 items
                    if i > 0 and i % 5000 == 0:
                        logger.info(f"Progress: {i}/{len(items_to_index)} items processed ({i/len(items_to_index)*100:.1f}%)")
                    try:
                        # Generate UUID from hash value and field type for idempotency
                        uuid_input = f"{item['hash_value']}_{item['field_type']}"
                        uuid = generate_uuid5(uuid_input)
                        
                        # Remove vector from properties
                        properties = {k: v for k, v in item.items() if k != 'vector'}
                        
                        # Verify vector dimensions
                        vector_data = item['vector']
                        if len(vector_data) != self.embedding_dimensions:
                            logger.error(f"Vector dimension mismatch: {len(vector_data)} != {self.embedding_dimensions}")
                            continue
                        
                        # Add object with explicit vector format
                        batch_writer.add_object(
                            properties=properties,
                            uuid=uuid,
                            vector=vector_data.tolist()  # Convert numpy array to list
                        )
                        
                        indexed_count += 1
                        
                    except Exception as e:
                        logger.error(f"Error indexing item {item.get('hash_value', 'unknown')}: {str(e)}")
            
            logger.info(f"Successfully indexed {indexed_count}/{len(items_to_index)} items")
            
        except Exception as e:
            logger.error(f"Error in batch indexing: {str(e)}")
        
        return indexed_count
    
    def _select_strings_to_process(self, string_dict: Dict[str, str], 
                                field_hash_mapping: Dict[str, Dict[str, int]],
                                string_counts: Dict[str, int]) -> List[Tuple[str, str, str, int]]:
        """
        Select strings for embedding based on field types.
        
        Args:
            string_dict: Dictionary mapping hash to string value
            field_hash_mapping: Mapping of hash to field types
            string_counts: Mapping of hash to frequency count
            
        Returns:
            List of (hash, string, field_type, frequency) tuples to process
        """
        logger.info("Selecting strings for batch processing")
        
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
        
        # Log selection statistics
        selected_count = len(strings_to_process)
        logger.info(f"Selected {selected_count} string-field pairs for batch processing")
        logger.info(f"Processing status: {already_processed}/{total_candidates} strings already processed ({cache_coverage:.1%})")
        
        # Log field type distribution 
        for field, count in field_match_counts.items():
            logger.info(f"  Field '{field}': {count} strings selected")
        
        return strings_to_process
    
    def load_checkpoint(self, checkpoint_dir: str) -> None:
        """
        Load checkpoint data from disk.
        
        Args:
            checkpoint_dir: Directory containing checkpoint files
        """
        # Load processed hashes - check for consolidated file first, then batch-specific
        consolidated_hashes_path = os.path.join(checkpoint_dir, 'processed_hashes.pkl')
        batch_hashes_path = os.path.join(checkpoint_dir, 'batch_processed_hashes.pkl')
        batch_jobs_path = os.path.join(checkpoint_dir, 'batch_jobs.pkl')
        blacklisted_files_path = os.path.join(checkpoint_dir, 'batch_blacklisted_files.pkl')
        queue_state_path = os.path.join(checkpoint_dir, 'batch_queue_state.pkl')
        failed_requests_path = os.path.join(checkpoint_dir, 'batch_failed_requests.pkl')
        
        # Load processed hashes - prioritize consolidated state from transition
        if os.path.exists(consolidated_hashes_path):
            try:
                with open(consolidated_hashes_path, 'rb') as f:
                    self.processed_hashes = set(pickle.load(f))
                logger.info(f"Loaded {len(self.processed_hashes)} consolidated processed hashes from transition")
            except Exception as e:
                logger.error(f"Error loading consolidated processed hashes: {str(e)}")
        elif os.path.exists(batch_hashes_path):
            try:
                with open(batch_hashes_path, 'rb') as f:
                    self.processed_hashes = set(pickle.load(f))
                logger.info(f"Loaded {len(self.processed_hashes)} batch-only processed hashes from checkpoint")
            except Exception as e:
                logger.error(f"Error loading batch processed hashes: {str(e)}")
                self.processed_hashes = set()
        
        # Load blacklisted files to avoid reprocessing
        if os.path.exists(blacklisted_files_path):
            try:
                with open(blacklisted_files_path, 'rb') as f:
                    self.blacklisted_files = set(pickle.load(f))
                logger.info(f"Loaded {len(self.blacklisted_files)} blacklisted files - these will be skipped")
            except Exception as e:
                logger.error(f"Error loading blacklisted files: {str(e)}")
                self.blacklisted_files = set()
        
        # Load failed requests for retry tracking - check consolidated file first
        consolidated_failed_path = os.path.join(checkpoint_dir, 'realtime_failed_requests.pkl')
        if os.path.exists(consolidated_failed_path):
            try:
                with open(consolidated_failed_path, 'rb') as f:
                    self.failed_requests = pickle.load(f)
                logger.info(f"Loaded {len(self.failed_requests)} consolidated failed requests for retry tracking")
            except Exception as e:
                logger.error(f"Error loading consolidated failed requests: {str(e)}")
                self.failed_requests = {}
        elif os.path.exists(failed_requests_path):
            try:
                with open(failed_requests_path, 'rb') as f:
                    self.failed_requests = pickle.load(f)
                logger.info(f"Loaded {len(self.failed_requests)} batch-only failed requests for retry tracking")
            except Exception as e:
                logger.error(f"Error loading batch failed requests: {str(e)}")
                self.failed_requests = {}
        
        if os.path.exists(batch_jobs_path):
            try:
                with open(batch_jobs_path, 'rb') as f:
                    self.batch_jobs = pickle.load(f)
                logger.info(f"Loaded {len(self.batch_jobs)} batch jobs from checkpoint")
                
                # Extract hashes from existing batch jobs to avoid reprocessing
                extracted_hashes = 0
                for job_info in self.batch_jobs.values():
                    if 'file_metadata' in job_info and 'custom_id_mapping' in job_info['file_metadata']:
                        for custom_id, item_data in job_info['file_metadata']['custom_id_mapping'].items():
                            hash_value = item_data['hash_value']
                            self.processed_hashes.add(hash_value)
                            extracted_hashes += 1
                
                if extracted_hashes > 0:
                    logger.info(f"Extracted {extracted_hashes} submitted hashes from existing batch jobs")
                    
            except Exception as e:
                logger.error(f"Error loading batch jobs: {str(e)}")
                logger.warning("Batch jobs checkpoint appears corrupted - attempting to recover")
                
                # Try to recover by querying OpenAI for existing batch jobs
                self.batch_jobs = {}
                try:
                    self._recover_batch_jobs_from_api(checkpoint_dir)
                except Exception as recovery_error:
                    logger.error(f"Failed to recover batch jobs from API: {recovery_error}")
                    logger.info("Starting with empty batch jobs - existing jobs will be rediscovered on status check")
        
        # Load queue state
        if os.path.exists(queue_state_path):
            try:
                with open(queue_state_path, 'rb') as f:
                    queue_state = pickle.load(f)
                self.active_batch_queue = queue_state.get('active_batch_queue', [])
                self.pending_batches = queue_state.get('pending_batches', [])
                self.completed_batches = queue_state.get('completed_batches', [])
                logger.info(f"Loaded queue state: {len(self.active_batch_queue)} active, "
                           f"{len(self.pending_batches)} pending, {len(self.completed_batches)} completed")
            except Exception as e:
                logger.error(f"Error loading queue state: {str(e)}")
                self.active_batch_queue = []
                self.pending_batches = []
                self.completed_batches = []
        
        # Recovery: If we have batch jobs but no queue state, rebuild queue from existing jobs
        if self.batch_jobs and not self.active_batch_queue and not self.completed_batches:
            logger.info("🔄 Rebuilding queue state from existing batch jobs...")
            for batch_job_id, job_info in self.batch_jobs.items():
                if job_info.get('auto_queue', False):  # Only auto-queue managed jobs
                    job_status = job_info.get('status', 'unknown')
                    if job_status in ['submitted', 'pending', 'validating', 'in_progress', 'finalizing']:
                        self.active_batch_queue.append(batch_job_id)
                        logger.debug(f"Added job {batch_job_id[:8]}... to active queue (status: {job_status})")
                    elif job_status in ['completed', 'failed', 'expired', 'cancelled']:
                        self.completed_batches.append(batch_job_id)
                        logger.debug(f"Added job {batch_job_id[:8]}... to completed (status: {job_status})")
            
            if self.active_batch_queue or self.completed_batches:
                logger.info(f"🔄 Rebuilt queue: {len(self.active_batch_queue)} active, {len(self.completed_batches)} completed")
                # Save the rebuilt state
                try:
                    self.save_checkpoint(checkpoint_dir)
                    logger.info("✅ Saved rebuilt queue state")
                except Exception as e:
                    logger.warning(f"Could not save rebuilt queue state: {e}")
    
    def save_checkpoint(self, checkpoint_dir: str) -> None:
        """
        Save checkpoint data to disk.
        
        Args:
            checkpoint_dir: Directory to save checkpoint files
        """
        # Ensure checkpoint directory exists
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save processed hashes
        processed_hashes_path = os.path.join(checkpoint_dir, 'batch_processed_hashes.pkl')
        batch_jobs_path = os.path.join(checkpoint_dir, 'batch_jobs.pkl')
        queue_state_path = os.path.join(checkpoint_dir, 'batch_queue_state.pkl')
        failed_requests_path = os.path.join(checkpoint_dir, 'batch_failed_requests.pkl')
        blacklisted_files_path = os.path.join(checkpoint_dir, 'batch_blacklisted_files.pkl')
        
        try:
            with open(processed_hashes_path, 'wb') as f:
                pickle.dump(list(self.processed_hashes), f)
            
            with open(batch_jobs_path, 'wb') as f:
                pickle.dump(self.batch_jobs, f)
            
            # Save queue state
            queue_state = {
                'active_batch_queue': self.active_batch_queue,
                'pending_batches': self.pending_batches,
                'completed_batches': self.completed_batches,
                'saved_at': time.time()
            }
            with open(queue_state_path, 'wb') as f:
                pickle.dump(queue_state, f)
            
            # Save failed requests for retry tracking
            with open(failed_requests_path, 'wb') as f:
                pickle.dump(self.failed_requests, f)
            
            # Save blacklisted files
            with open(blacklisted_files_path, 'wb') as f:
                pickle.dump(list(self.blacklisted_files), f)
                
            logger.info(f"Saved checkpoint: {len(self.processed_hashes)} processed hashes, "
                       f"{len(self.batch_jobs)} batch jobs, {len(self.failed_requests)} failed requests, "
                       f"queue state: {len(self.active_batch_queue)} active")
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")
    
    def _submit_single_batch(self, batch_data: List[Tuple[str, str, str, int]], 
                           batch_idx: int, checkpoint_dir: str) -> Optional[str]:
        """
        Submit a single batch to OpenAI and return the batch job ID.
        
        Args:
            batch_data: List of (hash, text, field_type, frequency) tuples
            batch_idx: Batch index for file naming
            checkpoint_dir: Directory for saving batch files
            
        Returns:
            Batch job ID if successful, None if failed
        """
        try:
            # Create batch requests file
            batch_file_path = os.path.join(checkpoint_dir, f'batch_requests_{batch_idx}.jsonl')
            file_metadata = self._create_batch_requests_file(batch_data, batch_file_path)
            
            # Upload file
            input_file_id = self._upload_batch_file(batch_file_path)
            
            # Check quota capacity before creating batch job
            batch_requests = []
            with open(batch_file_path, 'r') as f:
                for line in f:
                    if line.strip():
                        import json
                        batch_requests.append(json.loads(line))
            
            quota_check = self._check_quota_capacity(batch_requests)
            
            if not quota_check['can_submit']:
                logger.warning(f"🛑 Quota check failed for batch {batch_idx + 1} - will retry later")
                # Clean up uploaded file
                try:
                    self.openai_client.files.delete(input_file_id)
                    logger.debug(f"Cleaned up unused uploaded file {input_file_id}")
                except Exception:
                    pass
                return None
            
            # Create batch job
            batch_job_id = self._create_batch_job(
                input_file_id, 
                f"Entity resolution batch {batch_idx + 1} (auto-queue)"
            )
            
            # Track batch job
            self.batch_jobs[batch_job_id] = {
                'batch_idx': batch_idx,
                'input_file_id': input_file_id,
                'file_metadata': file_metadata,
                'request_count': file_metadata['request_count'],
                'created_at': time.time(),
                'status': 'submitted',
                'auto_queue': True  # Mark as auto-queue managed
            }
            
            # CRITICAL: Immediately verify batch status after submission
            # This catches quota exceeded errors that happen after submission
            try:
                time.sleep(1)  # Brief delay to allow OpenAI to process the submission
                batch_status = self.openai_client.batches.retrieve(batch_job_id)
                
                # Update local status
                self.batch_jobs[batch_job_id]['status'] = batch_status.status
                
                # Check if job failed immediately (quota exceeded, etc.)
                if batch_status.status == 'failed':
                    logger.error(f"❌ Batch job {batch_job_id} failed immediately after submission")
                    if hasattr(batch_status, 'errors') and batch_status.errors:
                        for error in batch_status.errors.data:
                            logger.error(f"   Error: {error.code} - {error.message}")
                            if error.code == 'request_limit_exceeded':
                                logger.warning(f"🛑 QUOTA EXCEEDED: OpenAI's 1M request limit reached!")
                                logger.warning(f"   Failed job {batch_job_id} should be cancelled to free quota")
                                # Return special indicator for quota exceeded
                                return 'QUOTA_EXCEEDED'
                    return None  # Return None for other types of failures
                else:
                    logger.info(f"✅ Submitted batch job {batch_job_id} for batch {batch_idx + 1} (status: {batch_status.status})")
                    
            except Exception as status_check_error:
                logger.warning(f"⚠️  Could not verify status of submitted job {batch_job_id}: {status_check_error}")
                logger.warning(f"🛑 Stopping submissions due to status verification failure (may be quota limit)")
                return None  # Return None to stop further submissions when we can't verify status
            
            return batch_job_id
            
        except Exception as e:
            logger.error(f"Error submitting batch {batch_idx + 1}: {str(e)}")
            return None
    
    def _check_and_update_queue_status(self) -> int:
        """
        Check status of active batches and move completed ones out of the queue.
        Enhanced with robust error handling for API unavailability.
        
        Returns:
            Number of slots freed up
        """
        if not self.active_batch_queue:
            return 0
            
        completed_job_ids = []
        api_errors = 0
        max_api_errors = len(self.active_batch_queue) // 2  # Allow up to 50% failures
        
        for batch_job_id in self.active_batch_queue:
            try:
                batch_status = self.openai_client.batches.retrieve(batch_job_id)
                current_status = batch_status.status
                
                # Update local status
                if batch_job_id in self.batch_jobs:
                    self.batch_jobs[batch_job_id]['status'] = current_status
                
                # Check if batch is completed (success or failure)
                if current_status in [BatchJobStatus.COMPLETED, BatchJobStatus.FAILED, 
                                    BatchJobStatus.EXPIRED, BatchJobStatus.CANCELLED]:
                    completed_job_ids.append(batch_job_id)
                    
                    if current_status == BatchJobStatus.COMPLETED:
                        logger.info(f"✅ Batch job {batch_job_id[:8]}... completed")
                    elif current_status == BatchJobStatus.FAILED:
                        logger.warning(f"❌ Batch job {batch_job_id[:8]}... failed")
                        
                        # CRITICAL: Auto-cleanup failed jobs that consume quota
                        if hasattr(batch_status, 'errors') and batch_status.errors:
                            for error in batch_status.errors.data:
                                if error.code == 'request_limit_exceeded':
                                    logger.warning(f"🧹 Auto-cancelling quota-exceeded job {batch_job_id[:8]}... to free quota space")
                                    try:
                                        # Cancel the failed job to free up quota
                                        self.openai_client.batches.cancel(batch_job_id)
                                        logger.info(f"✅ Successfully cancelled failed job {batch_job_id[:8]}...")
                                    except Exception as cancel_error:
                                        logger.error(f"❌ Failed to cancel job {batch_job_id[:8]}...: {cancel_error}")
                                    break
                    else:
                        logger.warning(f"❌ Batch job {batch_job_id[:8]}... terminated with status: {current_status}")
                        
            except Exception as e:
                api_errors += 1
                error_msg = str(e).lower()
                
                # Categorize different types of errors
                if any(term in error_msg for term in ['timeout', 'connection', 'network', '502', '503', '504']):
                    logger.warning(f"⚠️  Network/timeout error for batch job {batch_job_id[:8]}...: {str(e)}")
                elif 'rate limit' in error_msg or '429' in error_msg:
                    logger.warning(f"⚠️  Rate limit error for batch job {batch_job_id[:8]}...: {str(e)}")
                    # Add a small delay to respect rate limits
                    time.sleep(1)
                elif 'not found' in error_msg or '404' in error_msg:
                    logger.error(f"❌ Batch job {batch_job_id[:8]}... not found - may have been deleted")
                    # Remove from queue since it no longer exists
                    completed_job_ids.append(batch_job_id)
                else:
                    logger.error(f"❌ API error checking batch job {batch_job_id[:8]}...: {str(e)}")
                
                # If too many API errors, stop checking and try again later
                if api_errors > max_api_errors:
                    logger.warning(f"⚠️  Too many API errors ({api_errors}/{len(self.active_batch_queue)}) - "
                                 f"stopping status check to avoid overwhelming API")
                    break
        
        # Move completed jobs out of active queue
        slots_freed = 0
        for batch_job_id in completed_job_ids:
            if batch_job_id in self.active_batch_queue:
                self.active_batch_queue.remove(batch_job_id)
                self.completed_batches.append(batch_job_id)
                slots_freed += 1
        
        if slots_freed > 0:
            logger.info(f"🔓 Freed up {slots_freed} queue slots ({len(self.active_batch_queue)}/{self.max_active_batches} active)")
        
        # Log API error summary if any occurred
        if api_errors > 0:
            logger.info(f"📊 Status check completed with {api_errors} API errors (transient issues)")
            
        return slots_freed
    
    def _verify_actual_completion(self) -> Dict[str, Any]:
        """
        Verify actual completion by checking ALL embedding jobs on OpenAI API.
        This prevents false completion when local tracking becomes inconsistent.
        
        Returns:
            Dictionary with completion verification results
        """
        logger.info("🔍 Verifying actual completion status with OpenAI API...")
        
        try:
            # Query OpenAI API for ALL embedding batch jobs
            active_embedding_jobs = []
            other_status_jobs = {'completed': 0, 'failed': 0, 'cancelled': 0, 'expired': 0}
            total_embedding_jobs = 0
            
            after = None
            while True:
                if after:
                    batches = self.openai_client.batches.list(limit=100, after=after)
                else:
                    batches = self.openai_client.batches.list(limit=100)
                
                for batch in batches.data:
                    # Filter for embedding jobs created by our pipeline
                    endpoint = getattr(batch, 'endpoint', '')
                    metadata = getattr(batch, 'metadata', {})
                    
                    if (('/embeddings' in endpoint or endpoint == '/v1/embeddings') and
                        metadata and metadata.get('created_by') == 'embedding_and_indexing_batch'):
                        
                        total_embedding_jobs += 1
                        
                        # Check status and categorize
                        if batch.status in ['pending', 'validating', 'in_progress', 'finalizing', 'cancelling']:
                            active_embedding_jobs.append({
                                'id': batch.id,
                                'status': batch.status,
                                'created_at': batch.created_at,
                                'endpoint': endpoint
                            })
                        elif batch.status in ['completed', 'failed', 'cancelled', 'expired']:
                            other_status_jobs[batch.status] = other_status_jobs.get(batch.status, 0) + 1
                
                if not batches.has_more:
                    break
                    
                if batches.data:
                    after = batches.data[-1].id
                else:
                    break
            
            # Determine if truly complete
            active_jobs_found = len(active_embedding_jobs)
            truly_complete = (active_jobs_found == 0 and total_embedding_jobs > 0)
            
            # Create status summary
            status_summary = {
                'active': active_jobs_found,
                'completed': other_status_jobs['completed'],
                'failed': other_status_jobs['failed'],
                'cancelled': other_status_jobs['cancelled'],
                'expired': other_status_jobs['expired']
            }
            
            logger.info(f"📊 OpenAI API verification found {total_embedding_jobs} total embedding jobs")
            logger.info(f"   Active jobs: {active_jobs_found}")
            logger.info(f"   Completed: {other_status_jobs['completed']}, Failed: {other_status_jobs['failed']}")
            logger.info(f"   Cancelled: {other_status_jobs['cancelled']}, Expired: {other_status_jobs['expired']}")
            
            return {
                'truly_complete': truly_complete,
                'active_jobs_found': active_jobs_found,
                'total_jobs': total_embedding_jobs,
                'status_summary': status_summary,
                'api_jobs': active_embedding_jobs
            }
            
        except Exception as e:
            logger.error(f"❌ Error verifying completion with OpenAI API: {e}")
            # Conservative fallback: assume NOT complete if we can't verify
            return {
                'truly_complete': False,
                'active_jobs_found': -1,  # Unknown
                'total_jobs': -1,  # Unknown
                'status_summary': {'error': str(e)},
                'api_jobs': []
            }
    
    def _rebuild_tracking_from_api(self, api_jobs: List[Dict[str, Any]]) -> None:
        """
        Rebuild local tracking state from actual OpenAI API job status.
        
        Args:
            api_jobs: List of active jobs from OpenAI API
        """
        logger.info(f"🔧 Rebuilding local tracking from {len(api_jobs)} active API jobs...")
        
        # Clear current tracking
        original_active = len(self.active_batch_queue)
        original_completed = len(self.completed_batches)
        
        self.active_batch_queue.clear()
        
        # Rebuild from API state
        for job in api_jobs:
            job_id = job['id']
            
            # Add to active queue
            self.active_batch_queue.append(job_id)
            
            # Update or create job tracking
            if job_id not in self.batch_jobs:
                self.batch_jobs[job_id] = {
                    'status': job['status'],
                    'created_at': job.get('created_at'),
                    'recovered': True,
                    'rebuild_recovery': True
                }
            else:
                self.batch_jobs[job_id]['status'] = job['status']
        
        logger.info(f"✅ Rebuilt tracking: {len(self.active_batch_queue)} active jobs recovered")
        logger.info(f"   Previous tracking: {original_active} active, {original_completed} completed")
        logger.info(f"   Corrected tracking: {len(self.active_batch_queue)} active, {len(self.completed_batches)} completed")
    
    def _submit_pending_batches(self, checkpoint_dir: str) -> int:
        """
        Submit pending batches ONE AT A TIME with real-time quota verification between each.
        This conservative approach ensures we never exceed quota limits.
        
        Args:
            checkpoint_dir: Directory for saving batch files
            
        Returns:
            Number of new batches submitted
        """
        available_slots = self.max_active_batches - len(self.active_batch_queue)
        if available_slots <= 0 or not self.pending_batches:
            return 0
        
        submitted_count = 0
        max_to_attempt = min(available_slots, len(self.pending_batches))
        
        logger.info(f"📤 Attempting to submit up to {max_to_attempt} pending batches (ONE AT A TIME with quota verification)")
        
        # Submit ONE batch at a time with quota verification between each
        for attempt in range(max_to_attempt):
            if not self.pending_batches:
                break
            
            # Check quota availability before each submission
            try:
                logger.debug(f"🔍 Checking quota before submission attempt {attempt + 1}")
                current_usage = self._get_current_usage()
                
                # CORRECTED: Stop when we hit exactly 800K requests (the actual target)
                current_requests = current_usage['total_active_requests']
                if current_requests >= self.request_quota_target:  # Stop at exactly 800K
                    logger.warning(f"🛑 Quota usage at {current_requests:,} requests (target: {self.request_quota_target:,}) - attempting cleanup and recovery")
                    
                    # Try to clean up failed jobs first
                    cleaned_up = self._cleanup_failed_quota_jobs()
                    if cleaned_up > 0:
                        logger.info(f"✅ Cleaned up {cleaned_up} failed jobs - rechecking quota")
                        # Recheck quota after cleanup
                        updated_usage = self._get_current_usage()
                        updated_requests = updated_usage['total_active_requests']
                        if updated_requests < self.request_quota_target:  # Below 800K after cleanup
                            logger.info(f"🎉 Quota recovered to {updated_requests:,} requests after cleanup - continuing")
                        else:
                            logger.warning(f"🚨 Still at {updated_requests:,} requests after cleanup - stopping submissions")
                            break
                    else:
                        logger.warning(f"🚨 No failed jobs to clean up - quota genuinely full at {current_requests:,} requests")
                        break
                    
            except Exception as quota_check_error:
                logger.warning(f"⚠️  Could not verify quota status before submission: {quota_check_error}")
                # Continue with submission but be extra cautious
            
            # Submit single batch
            batch_data, batch_idx = self.pending_batches.pop(0)
            batch_job_id = self._submit_single_batch(batch_data, batch_idx, checkpoint_dir)
            
            if batch_job_id and batch_job_id != 'QUOTA_EXCEEDED':
                self.active_batch_queue.append(batch_job_id)
                submitted_count += 1
                logger.info(f"✅ Successfully submitted batch {batch_job_id[:8]}... ({submitted_count}/{max_to_attempt})")
                logger.info(f"📋 Active queue: {len(self.active_batch_queue)}/{self.max_active_batches}")
                
                # CRITICAL: Verify quota status after EACH successful submission
                try:
                    time.sleep(2)  # Brief delay to allow OpenAI to update quota
                    post_submission_usage = self._get_current_usage()
                    post_requests = post_submission_usage['total_active_requests']
                    
                    logger.info(f"📊 Post-submission quota: {post_requests:,}/{self.request_quota_target:,} "
                               f"(target: 800K) - {post_submission_usage['active_jobs']} active jobs")
                    
                    # CORRECTED: Stop when we reach exactly 800K requests
                    if post_requests >= self.request_quota_target:  # Stop at exactly 800K
                        logger.warning(f"🚨 Quota usage now at {post_requests:,} requests after submission - "
                                     f"reached target of {self.request_quota_target:,}, stopping further submissions")
                        break
                        
                except Exception as post_check_error:
                    logger.warning(f"⚠️  Could not verify quota after submission: {post_check_error}")
                    # Continue but be cautious
                
            elif batch_job_id == 'QUOTA_EXCEEDED':
                # Put the batch back at the front and enter polling mode
                self.pending_batches.insert(0, (batch_data, batch_idx))
                logger.warning(f"🔄 Quota exceeded - entering polling mode for batch {batch_idx + 1}")
                
                # Enter polling loop - check every 30 minutes for quota availability
                if self._wait_for_quota_recovery(checkpoint_dir):
                    logger.info(f"🎉 Quota recovered - resuming submissions from batch {batch_idx + 1}")
                    continue  # Try this batch again
                else:
                    logger.warning(f"⏰ Quota polling timed out or failed - stopping submissions")
                    break
                
            else:
                # If submission failed for other reasons, put the batch back at the front
                self.pending_batches.insert(0, (batch_data, batch_idx))
                logger.warning(f"❌ Failed to submit batch {batch_idx + 1} - stopping further attempts")
                break  # Stop trying if any submission fails
        
        if submitted_count > 0:
            logger.info(f"📤 Successfully submitted {submitted_count} batches with real-time quota verification")
        
        return submitted_count
    
    def create_batch_jobs_with_automated_queue(self, string_dict: Dict[str, str], 
                                             field_hash_mapping: Dict[str, Dict[str, int]],
                                             string_counts: Dict[str, int], 
                                             checkpoint_dir: str) -> Dict[str, Any]:
        """
        Create and manage batch jobs using an automated queue system.
        Maintains 16 active batches, automatically submitting new ones as slots free up.
        
        Args:
            string_dict: Dictionary mapping hash to string value
            field_hash_mapping: Mapping of hash to field types
            string_counts: Mapping of hash to frequency count
            checkpoint_dir: Directory for saving checkpoints
            
        Returns:
            Dictionary with job creation and processing metrics
        """
        logger.info("🚀 Starting automated batch queue management")
        logger.info(f"🔧 Configuration: Max active batches: {self.max_active_batches}")
        logger.info(f"🔧 Configuration: Poll interval: {self.queue_poll_interval / 60:.0f} minutes")
        logger.info(f"🔧 Configuration: Conservative request limit: {self.request_quota_limit:,}")
        start_time = time.time()
        
        # Check initial quota status
        try:
            initial_usage = self._get_current_usage()
            logger.info(f"📊 Initial quota status: {initial_usage['total_active_requests']:,}/{self.request_quota_limit:,} requests "
                       f"({initial_usage['total_active_requests']/self.request_quota_limit*100:.1f}%), "
                       f"{initial_usage['active_jobs']} active jobs")
            
            if initial_usage['total_active_requests'] > self.request_quota_limit * 0.95:
                logger.warning(f"⚠️  Starting with high quota usage ({initial_usage['total_active_requests']/self.request_quota_limit*100:.1f}%) - "
                             f"limited capacity for new jobs")
        except Exception as quota_error:
            logger.warning(f"⚠️  Could not check initial quota status: {quota_error}")
        
        # Load checkpoint
        self.load_checkpoint(checkpoint_dir)
        
        # Select strings to process
        strings_to_process = self._select_strings_to_process(
            string_dict, field_hash_mapping, string_counts
        )
        
        if not strings_to_process:
            logger.info("No new strings to process")
            return {
                'status': 'no_work',
                'message': 'All eligible strings already processed',
                'elapsed_time': time.time() - start_time
            }
        
        # Split into batches
        all_batches = []
        for i in range(0, len(strings_to_process), self.max_requests_per_file):
            batch = strings_to_process[i:i+self.max_requests_per_file]
            all_batches.append(batch)
        
        logger.info(f"📊 Split {len(strings_to_process)} requests into {len(all_batches)} batch files")
        
        # Initialize pending batches (if not resuming)
        if not self.pending_batches and not self.active_batch_queue:
            # Starting fresh - add all batches to pending queue
            self.pending_batches = [(batch, idx) for idx, batch in enumerate(all_batches)]
            logger.info(f"📋 Added {len(self.pending_batches)} batches to pending queue")
        elif self.pending_batches or self.active_batch_queue:
            # Resuming from checkpoint
            logger.info(f"🔄 Resuming: {len(self.active_batch_queue)} active, "
                       f"{len(self.pending_batches)} pending, {len(self.completed_batches)} completed")
        
        # Fill initial queue
        initial_submitted = self._submit_pending_batches(checkpoint_dir)
        if initial_submitted > 0:
            logger.info(f"🚀 Initial submission: {initial_submitted} batches added to queue")
        
        # Save checkpoint after initial submission
        self.save_checkpoint(checkpoint_dir)
        
        # Main processing loop
        total_cycles = 0
        last_status_time = time.time()
        
        logger.info(f"🔄 Starting automated queue processing...")
        logger.info(f"📊 Queue status: {len(self.active_batch_queue)} active, "
                   f"{len(self.pending_batches)} pending, {len(self.completed_batches)} completed")
        
        consecutive_errors = 0
        max_consecutive_errors = 3  # Allow up to 3 consecutive cycles with errors
        
        try:
            while self.active_batch_queue or self.pending_batches:
                total_cycles += 1
                cycle_had_errors = False
                
                try:
                    # Check and update queue status
                    slots_freed = self._check_and_update_queue_status()
                    
                    # Submit new batches to fill freed slots
                    if slots_freed > 0 or (self.pending_batches and len(self.active_batch_queue) < self.max_active_batches):
                        submitted = self._submit_pending_batches(checkpoint_dir)
                        if submitted > 0:
                            logger.info(f"📤 Submitted {submitted} new batches to queue")
                    
                    # Check for retryable failed requests if we have available slots and no pending batches
                    if (self.enable_automatic_retries and 
                        self.retry_batch_creation and
                        len(self.active_batch_queue) < self.max_active_batches and 
                        not self.pending_batches and 
                        len(self.failed_requests) > 0):
                        
                        retryable_count = len(self.get_retryable_failed_requests())
                        if retryable_count > 0:
                            logger.info(f"🔄 Attempting to create retry batch for {retryable_count} failed requests")
                            
                            # Need hash_lookup and string_dict for retry batch creation
                            # These should be loaded from checkpoints
                            hash_lookup_path = os.path.join(checkpoint_dir, 'hash_lookup.pkl')
                            string_dict_path = os.path.join(checkpoint_dir, 'string_dict.pkl')
                            
                            if os.path.exists(hash_lookup_path) and os.path.exists(string_dict_path):
                                try:
                                    with open(hash_lookup_path, 'rb') as f:
                                        hash_lookup = pickle.load(f)
                                    with open(string_dict_path, 'rb') as f:
                                        string_dict = pickle.load(f)
                                    
                                    retry_job_id = self.create_retry_batch(hash_lookup, string_dict, checkpoint_dir)
                                    if retry_job_id:
                                        logger.info(f"✅ Created retry batch {retry_job_id}")
                                        self.active_batch_queue.append(retry_job_id)
                                        
                                except Exception as e:
                                    logger.error(f"Error creating retry batch: {e}")
                            else:
                                logger.warning("Cannot create retry batch - hash_lookup.pkl or string_dict.pkl not found")
                    
                    # Save checkpoint after any changes
                    self.save_checkpoint(checkpoint_dir)
                    
                    # Reset error counter on successful cycle
                    consecutive_errors = 0
                    
                except Exception as cycle_error:
                    cycle_had_errors = True
                    consecutive_errors += 1
                    logger.error(f"❌ Error in processing cycle {total_cycles}: {str(cycle_error)}")
                    
                    # If too many consecutive errors, add longer delay
                    if consecutive_errors >= max_consecutive_errors:
                        extended_wait = min(self.queue_poll_interval * 2, 3600)  # Max 1 hour
                        logger.warning(f"⚠️  {consecutive_errors} consecutive errors - "
                                     f"extending wait to {extended_wait / 60:.0f} minutes")
                        time.sleep(extended_wait)
                        consecutive_errors = 0  # Reset after extended wait
                    else:
                        # Short delay before retrying
                        time.sleep(60)  # 1 minute
                
                # Enhanced status update every 5 minutes or when significant changes occur
                current_time = time.time()
                if (current_time - last_status_time >= 300 or slots_freed > 0 or 
                    cycle_had_errors or total_cycles == 1):  # 5 minutes or significant events
                    
                    # Get detailed queue status with quota information
                    try:
                        current_usage = self._get_current_usage()
                        available_slots = self.max_active_batches - len(self.active_batch_queue)
                        
                        status_msg = f"📊 Cycle {total_cycles}: Queue: " \
                                   f"{len(self.active_batch_queue)}/{self.max_active_batches} active, " \
                                   f"{len(self.pending_batches)} pending, " \
                                   f"{len(self.completed_batches)} completed"
                        
                        # Add quota status
                        quota_msg = f"🔢 Quota: {current_usage['total_active_requests']:,}/{self.request_quota_limit:,} requests " \
                                   f"({current_usage['total_active_requests']/self.request_quota_limit*100:.1f}%), " \
                                   f"{current_usage['active_jobs']} active jobs"
                        
                        if cycle_had_errors:
                            status_msg += f" (cycle had errors: {consecutive_errors}/{max_consecutive_errors})"
                        
                        logger.info(status_msg)
                        logger.info(quota_msg)
                        
                        # Log available capacity
                        if available_slots > 0 and self.pending_batches:
                            logger.info(f"💡 {available_slots} queue slots available, {len(self.pending_batches)} batches pending")
                        elif len(self.active_batch_queue) >= self.max_active_batches:
                            logger.info(f"⏸️  Queue full ({self.max_active_batches}/{self.max_active_batches}) - waiting for completion")
                        
                        # Warn if quota is getting high and handle quota recovery
                        current_requests = current_usage['total_active_requests']
                        if current_requests >= self.request_quota_target:  # At or above 800K
                            logger.warning(f"🚨 Critical quota usage: {current_requests:,} requests (target: 800K) - attempting recovery")
                            
                            # Try proactive cleanup
                            cleaned_up = self._cleanup_failed_quota_jobs()
                            if cleaned_up > 0:
                                logger.info(f"🧹 Quota recovery: Cleaned up {cleaned_up} failed jobs")
                                # Force recheck on next cycle
                                last_status_time = 0  # Trigger immediate status update
                                
                        elif current_requests > 750_000:  # Above 750K but below 800K
                            logger.warning(f"⚠️  High quota usage: {current_requests:,} requests approaching target of 800K")
                        elif current_requests > 640_000:  # Above 640K (80% of 800K)
                            logger.info(f"📈 Moderate quota usage: {current_requests:,} requests (target: 800K)")
                            
                    except Exception as status_error:
                        # Fallback to basic status if quota check fails
                        status_msg = f"📊 Cycle {total_cycles}: Queue status: " \
                                   f"{len(self.active_batch_queue)} active, " \
                                   f"{len(self.pending_batches)} pending, " \
                                   f"{len(self.completed_batches)} completed"
                        
                        if cycle_had_errors:
                            status_msg += f" (cycle had errors: {consecutive_errors}/{max_consecutive_errors})"
                        
                        logger.info(status_msg)
                        logger.warning(f"⚠️  Could not get quota status: {status_error}")
                    
                    last_status_time = current_time
                
                # Exit condition: comprehensive completion verification
                if not self.active_batch_queue and not self.pending_batches:
                    # CRITICAL: Don't exit just because local queues are empty!
                    # Verify actual completion with OpenAI API to prevent false completion
                    logger.info(f"🔍 Local queues empty - verifying actual job completion with OpenAI API...")
                    
                    actual_completion = self._verify_actual_completion()
                    
                    if actual_completion['truly_complete']:
                        logger.info(f"✅ Verified completion: All {actual_completion['total_jobs']} jobs are actually finished")
                        logger.info(f"📊 Final status breakdown: {actual_completion['status_summary']}")
                        break
                    else:
                        logger.warning(f"⚠️  FALSE COMPLETION DETECTED!")
                        logger.warning(f"   Local tracking showed completion but {actual_completion['active_jobs_found']} jobs are still active")
                        logger.warning(f"   Status breakdown: {actual_completion['status_summary']}")
                        logger.warning(f"   Continuing monitoring to prevent data loss...")
                        
                        # Rebuild local tracking from API state
                        self._rebuild_tracking_from_api(actual_completion['api_jobs'])
                        
                        # Continue processing - don't exit!
                
                # Intelligent wait strategy based on queue state and quota status
                if self.active_batch_queue and not cycle_had_errors:  # Normal operation
                    logger.info(f"⏳ Normal operation - waiting {self.queue_poll_interval / 60:.0f} minutes before next status check...")
                    time.sleep(self.queue_poll_interval)
                elif not self.active_batch_queue and self.pending_batches:
                    # No active batches but pending work - likely quota issue or system recovery needed
                    logger.warning(f"🔄 No active batches but {len(self.pending_batches)} pending - attempting intelligent recovery")
                    
                    # Try quota recovery first
                    try:
                        logger.info("🧹 Attempting quota recovery through failed job cleanup...")
                        cleaned_up = self._cleanup_failed_quota_jobs()
                        
                        if cleaned_up > 0:
                            logger.info(f"✅ Cleaned up {cleaned_up} failed jobs - attempting immediate retry")
                            # Short delay then retry immediately
                            time.sleep(30)  # 30 seconds
                            continue
                        else:
                            # No failed jobs to clean up - check quota with probe
                            logger.info("🔍 No failed jobs found - probing quota availability...")
                            probe_result = self._probe_quota_availability()
                            
                            if probe_result.get('quota_available'):
                                logger.info("✅ Quota probe successful - quota space available, retrying immediately")
                                time.sleep(10)  # Brief delay then retry
                                continue
                            elif probe_result.get('quota_exceeded'):
                                logger.warning("🚨 Quota probe confirmed quota exceeded - waiting for quota recovery")
                                # Longer wait for quota to free up as jobs complete
                                quota_wait = min(self.queue_poll_interval // 2, 1800)  # 30 minutes max
                                logger.info(f"⏳ Waiting {quota_wait / 60:.0f} minutes for quota to free up...")
                                time.sleep(quota_wait)
                            else:
                                # Unknown quota status - conservative wait
                                short_wait = min(self.queue_poll_interval // 4, 900)  # 15 minutes max
                                logger.info(f"⏳ Unknown quota status - waiting {short_wait / 60:.0f} minutes before retry...")
                                time.sleep(short_wait)
                                
                    except Exception as recovery_error:
                        logger.error(f"❌ Error during intelligent recovery: {recovery_error}")
                        # Fallback to normal short wait
                        short_wait = min(self.queue_poll_interval // 4, 900)  # 15 minutes max
                        logger.info(f"⏳ Recovery failed - waiting {short_wait / 60:.0f} minutes before retry...")
                        time.sleep(short_wait)
                else:
                    # Some other state - normal wait
                    logger.info(f"⏳ Standard wait - {self.queue_poll_interval / 60:.0f} minutes before next cycle...")
                    time.sleep(self.queue_poll_interval)
                
        except KeyboardInterrupt:
            logger.info(f"🛑 Process interrupted by user")
            logger.info(f"📊 Current state: {len(self.active_batch_queue)} active, "
                       f"{len(self.pending_batches)} pending, {len(self.completed_batches)} completed")
            self.save_checkpoint(checkpoint_dir)
            return {
                'status': 'interrupted',
                'active_batches': len(self.active_batch_queue),
                'pending_batches': len(self.pending_batches),
                'completed_batches': len(self.completed_batches),
                'elapsed_time': time.time() - start_time,
                'message': 'Process interrupted. State saved, can be resumed.'
            }
        except Exception as e:
            logger.error(f"❌ Error in automated queue processing: {str(e)}")
            self.save_checkpoint(checkpoint_dir)
            raise
        
        elapsed_time = time.time() - start_time
        
        logger.info(f"✅ Automated queue processing completed in {elapsed_time / 60:.1f} minutes")
        logger.info(f"📊 Final stats: {len(self.completed_batches)} batches completed")
        logger.info(f"📋 Total processing cycles: {total_cycles}")
        logger.info(f"💰 Estimated cost savings: ${len(strings_to_process) * 0.00001 * 0.5:.4f}")
        
        return {
            'status': 'completed',
            'total_batches': len(all_batches),
            'completed_batches': len(self.completed_batches),
            'total_requests': len(strings_to_process),
            'processing_cycles': total_cycles,
            'estimated_cost_savings': len(strings_to_process) * 0.00001 * 0.5,
            'elapsed_time': elapsed_time,
            'message': f'Automated queue completed {len(self.completed_batches)} batches in {total_cycles} cycles.'
        }
    
    def _recover_batch_jobs_from_api_readonly(self) -> None:
        """
        Read-only recovery of batch jobs from OpenAI API for status checking.
        Does not save any files or checkpoints - purely for status display.
        """
        logger.info("Attempting to recover batch jobs from OpenAI API (read-only)")
        
        try:
            recovered_jobs = 0
            after = None
            total_batches_checked = 0
            
            # Paginate through ALL batch jobs to find ours
            while True:
                # Get batch of results (up to 100 per request)
                if after:
                    batches = self.openai_client.batches.list(limit=100, after=after)
                else:
                    batches = self.openai_client.batches.list(limit=100)
                
                total_batches_checked += len(batches.data)
                
                # Process this page of results
                for batch in batches.data:
                    # Look for our batch jobs (filter by metadata or other identifiers)
                    metadata = getattr(batch, 'metadata', {})
                    if (metadata and 
                        metadata.get('created_by') == 'embedding_and_indexing_batch'):
                        
                        # Include ALL statuses - even failed ones so we can track them
                        logger.debug(f"Found batch {batch.id}: {batch.status}")
                        self.batch_jobs[batch.id] = {
                            'batch_idx': recovered_jobs,  # Sequential numbering for recovered jobs
                            'input_file_id': batch.input_file_id,
                            'status': batch.status,
                            'created_at': batch.created_at,
                            'recovered': True,  # Mark as recovered
                            'original_description': getattr(batch, 'metadata', {}).get('description', ''),
                            'readonly_recovery': True  # Mark as read-only recovery
                        }
                        
                        if hasattr(batch, 'output_file_id') and batch.output_file_id:
                            self.batch_jobs[batch.id]['output_file_id'] = batch.output_file_id
                        
                        # For read-only recovery, assume no files are downloaded
                        self.batch_jobs[batch.id]['results_downloaded'] = False
                        self.batch_jobs[batch.id]['results_processed'] = False
                        
                        recovered_jobs += 1
                
                # Check if there are more results
                if not batches.has_more:
                    break
                    
                # Get the last batch ID for pagination
                if batches.data:
                    after = batches.data[-1].id
                else:
                    break
            
            logger.info(f"Scanned {total_batches_checked} total batches from OpenAI API")
            if recovered_jobs > 0:
                logger.info(f"Recovered {recovered_jobs} entity resolution batch jobs (read-only)")
            else:
                logger.info("No existing entity resolution batch jobs found in OpenAI API")
                
        except Exception as e:
            logger.error(f"Error querying OpenAI API for batch recovery: {str(e)}")
            raise

    def _recover_batch_jobs_from_api(self, checkpoint_dir: str = None) -> None:
        """
        Attempt to recover ALL batch jobs by querying OpenAI API with pagination.
        This helps when checkpoint files are corrupted.
        Also checks for existing downloaded files to update status accordingly.
        """
        logger.info("Attempting to recover ALL batch jobs from OpenAI API")
        
        try:
            recovered_jobs = 0
            after = None
            total_batches_checked = 0
            
            # Paginate through ALL batch jobs to find ours
            while True:
                # Get batch of results (up to 100 per request)
                if after:
                    batches = self.openai_client.batches.list(limit=100, after=after)
                else:
                    batches = self.openai_client.batches.list(limit=100)
                
                total_batches_checked += len(batches.data)
                
                # Process this page of results
                for batch in batches.data:
                    # Look for our batch jobs (filter by metadata or other identifiers)
                    metadata = getattr(batch, 'metadata', {})
                    if (metadata and 
                        metadata.get('created_by') == 'embedding_and_indexing_batch'):
                        
                        # Include ALL statuses - even failed ones so we can track them
                        logger.debug(f"Recovering batch {batch.id} with input_file_id: {batch.input_file_id}")
                        self.batch_jobs[batch.id] = {
                            'batch_idx': recovered_jobs,  # Sequential numbering for recovered jobs
                            'input_file_id': batch.input_file_id,
                            'status': batch.status,
                            'created_at': batch.created_at,
                            'recovered': True,  # Mark as recovered
                            'original_description': getattr(batch, 'metadata', {}).get('description', '')
                        }
                        
                        if hasattr(batch, 'output_file_id') and batch.output_file_id:
                            self.batch_jobs[batch.id]['output_file_id'] = batch.output_file_id
                        
                        # Initialize download/processing status for all jobs
                        if batch.status == 'completed':
                            # Check for existing downloaded files if checkpoint_dir is available
                            results_downloaded = False
                            if checkpoint_dir:
                                logger.debug(f"Checking for downloaded files for job {batch.id[:8]}... in directory: {checkpoint_dir}")
                                
                                # Get all existing batch_results_*.jsonl files
                                import glob
                                existing_result_files = glob.glob(os.path.join(checkpoint_dir, 'batch_results_*.jsonl'))
                                
                                if existing_result_files:
                                    logger.debug(f"  Found {len(existing_result_files)} existing result files")
                                    # For now, mark all completed jobs as downloaded if ANY result files exist
                                    # This is a simple fix - we could get more sophisticated later
                                    results_downloaded = True
                                    logger.info(f"✅ Found existing results files - marking job {batch.id[:8]}... as downloaded")
                                else:
                                    logger.debug(f"  No result files found in checkpoint directory")
                            
                            # Set download/processing status
                            self.batch_jobs[batch.id]['results_downloaded'] = results_downloaded
                            self.batch_jobs[batch.id]['results_processed'] = False  # Conservative: assume not processed until confirmed
                        else:
                            # For non-completed jobs, these flags don't apply
                            self.batch_jobs[batch.id]['results_downloaded'] = False
                            self.batch_jobs[batch.id]['results_processed'] = False
                        
                        recovered_jobs += 1
                        
                        logger.debug(f"Recovered batch {batch.id}: {batch.status} (created: {batch.created_at})")
                
                # Check if there are more results
                if not batches.has_more:
                    break
                    
                # Get the last batch ID for pagination
                if batches.data:
                    after = batches.data[-1].id
                else:
                    break
            
            logger.info(f"Scanned {total_batches_checked} total batches from OpenAI API")
            if recovered_jobs > 0:
                logger.info(f"Recovered {recovered_jobs} entity resolution batch jobs")
                
                # Group by status for summary
                status_counts = {}
                downloaded_count = 0
                for job_info in self.batch_jobs.values():
                    status = job_info['status']
                    status_counts[status] = status_counts.get(status, 0) + 1
                    if job_info.get('results_downloaded', False):
                        downloaded_count += 1
                
                logger.info("Recovered job status summary:")
                for status, count in status_counts.items():
                    logger.info(f"  {status}: {count} jobs")
                
                if downloaded_count > 0:
                    logger.info(f"Found {downloaded_count} jobs with existing downloaded results")
                    
            else:
                logger.info("No existing entity resolution batch jobs found in OpenAI API")
                
        except Exception as e:
            logger.error(f"Error querying OpenAI API for batch recovery: {str(e)}")
            raise
    
    def _estimate_tokens_for_requests(self, requests: List[Dict[str, Any]]) -> int:
        """
        Estimate total tokens needed for a list of batch requests.
        Uses tiktoken for accurate counting when available, fallback to character-based estimation.
        
        Args:
            requests: List of embedding requests
            
        Returns:
            Estimated total token count
        """
        total_tokens = 0
        
        # Initialize tokenizer if available
        encoding = None
        if TIKTOKEN_AVAILABLE:
            try:
                encoding = tiktoken.encoding_for_model(self.embedding_model)
            except Exception as e:
                logger.debug(f"Could not load tiktoken encoding for {self.embedding_model}: {e}")
                encoding = None
        
        for request in requests:
            # Get the input text from the request
            if 'body' in request and 'input' in request['body']:
                text = request['body']['input']
                
                if encoding is not None:
                    # Accurate token counting with tiktoken
                    try:
                        estimated_tokens = len(encoding.encode(text))
                    except Exception:
                        # Fallback to character-based estimation
                        estimated_tokens = len(text) // 4 + 10
                else:
                    # Fallback: ~4 characters per token (GPT tokenizer approximation)
                    estimated_tokens = len(text) // 4 + 10  # Add small buffer
                
                total_tokens += estimated_tokens
                
        return total_tokens
    
    def _get_current_usage(self) -> Dict[str, int]:
        """
        Get current token and request usage from active batch jobs.
        Uses exact request counts from our stored data and live API status checks.
        
        Returns:
            Dictionary with current usage statistics including both tokens and requests
        """
        logger.info("Checking current usage from active batch jobs...")
        
        active_jobs = 0
        estimated_active_tokens = 0
        total_active_requests = 0
        
        try:
            # Get all current batch jobs from OpenAI API to check status
            api_batch_statuses = {}
            after = None
            while True:
                if after:
                    batches = self.openai_client.batches.list(limit=100, after=after)
                else:
                    batches = self.openai_client.batches.list(limit=100)
                
                for batch in batches.data:
                    # Only count batches created by our pipeline
                    endpoint = getattr(batch, 'endpoint', '')
                    metadata = getattr(batch, 'metadata', {})
                    
                    if (('/embeddings' in endpoint or endpoint == '/v1/embeddings') and
                        metadata and metadata.get('created_by') == 'embedding_and_indexing_batch'):
                        # Track all jobs for visibility, but only count quota for non-failed jobs
                        if batch.status in ['pending', 'validating', 'in_progress', 'finalizing', 'failed', 'cancelled', 'cancelling', 'completed']:
                            api_batch_statuses[batch.id] = batch.status
                            
                            # Only count quota for jobs that actually consume quota (not failed/cancelled/completed)
                            if batch.status in ['pending', 'validating', 'in_progress', 'finalizing', 'cancelling']:
                                # Get request count from OpenAI for accurate tracking
                                if hasattr(batch, 'request_counts') and batch.request_counts:
                                    request_count = getattr(batch.request_counts, 'total', 0)
                                    total_active_requests += request_count
                                    estimated_active_tokens += request_count * 100  # 100 tokens/request estimate
                                    active_jobs += 1
                                else:
                                    # Fallback: estimate from file size
                                    fallback_requests = self.max_requests_per_file
                                    total_active_requests += fallback_requests
                                    estimated_active_tokens += fallback_requests * 100
                                    active_jobs += 1
                
                if not batches.has_more:
                    break
                    
                if batches.data:
                    after = batches.data[-1].id
                else:
                    break
            
            logger.info(f"Found {active_jobs} active embedding batch jobs using ~{estimated_active_tokens:,} tokens and {total_active_requests:,} requests (tracked {len(api_batch_statuses)} total jobs)")
            
            # Count failed jobs for visibility (they don't consume quota but good to track)
            failed_jobs = len([batch_id for batch_id, status in api_batch_statuses.items() if status == 'failed'])
            if failed_jobs > 0:
                logger.info(f"ℹ️  {failed_jobs} failed embedding jobs found (these do not consume quota)")
            
            return {
                'active_jobs': active_jobs,
                'estimated_active_tokens': estimated_active_tokens,
                'total_active_requests': total_active_requests,
                'token_quota_limit': self.token_quota_limit,
                'request_quota_limit': self.request_quota_limit,
                'available_token_quota': max(0, self.token_quota_limit - estimated_active_tokens),
                'available_request_quota': max(0, self.request_quota_limit - total_active_requests)
            }
            
        except Exception as e:
            logger.error(f"Error checking current usage: {e}")
            # Return conservative estimates on error
            return {
                'active_jobs': 0,
                'estimated_active_tokens': 0,
                'total_active_requests': 0,
                'token_quota_limit': self.token_quota_limit,
                'request_quota_limit': self.request_quota_limit,
                'available_token_quota': self.token_quota_limit // 2,  # Conservative fallback
                'available_request_quota': self.request_quota_limit // 2  # Conservative fallback
            }
    
    def _wait_for_quota_recovery(self, checkpoint_dir: str, max_wait_hours: int = 32) -> bool:
        """
        Wait for quota to recover by polling every 30 minutes.
        
        Args:
            checkpoint_dir: Directory for checkpoints
            max_wait_hours: Maximum hours to wait (default 32 hours)
            
        Returns:
            True if quota recovered, False if timed out or failed
        """
        poll_interval_minutes = 30
        poll_interval_seconds = poll_interval_minutes * 60
        max_polls = (max_wait_hours * 60) // poll_interval_minutes  # Calculate max polling attempts
        
        logger.info(f"⏰ Starting quota recovery polling (every {poll_interval_minutes} minutes, max {max_wait_hours} hours)")
        
        for poll_attempt in range(max_polls):
            try:
                # Check if any active jobs have completed to free up quota
                logger.info(f"🔍 Checking for completed jobs that may have freed quota (attempt {poll_attempt + 1}/{max_polls})")
                # Note: Failed jobs should automatically free quota once they reach 'failed' status
                # We don't need to "clean up" failed jobs - they don't consume quota
                
                # Check current quota usage
                current_usage = self._get_current_usage()
                current_requests = current_usage['total_active_requests']
                
                logger.info(f"📊 Quota check (poll {poll_attempt + 1}): {current_requests:,}/{self.request_quota_target:,} "
                           f"(target: 800K) - {current_usage['active_jobs']} active jobs")
                
                # Check if we're below the 800K target to submit new batches
                if current_requests < self.request_quota_target:
                    logger.info(f"🎉 Quota recovered to {current_requests:,} requests - ready to resume submissions!")
                    return True
                
                # If this is the last attempt, don't wait
                if poll_attempt + 1 >= max_polls:
                    logger.warning(f"⏰ Reached maximum polling time ({max_wait_hours} hours) - quota still at {current_requests:,} requests")
                    break
                
                # Wait for next poll
                logger.info(f"⏳ Quota still at {current_requests:,} requests - waiting {poll_interval_minutes} minutes for next check...")
                
                # Save checkpoint before long wait
                try:
                    self.save_checkpoint(checkpoint_dir)
                    logger.debug(f"💾 Saved checkpoint during polling wait")
                except Exception as save_error:
                    logger.warning(f"⚠️  Could not save checkpoint during polling: {save_error}")
                
                time.sleep(poll_interval_seconds)
                
            except KeyboardInterrupt:
                logger.info(f"⏹️  Quota polling interrupted by user")
                return False
            except Exception as poll_error:
                logger.error(f"❌ Error during quota polling (attempt {poll_attempt + 1}): {poll_error}")
                if poll_attempt + 1 >= max_polls:
                    break
                logger.info(f"🔄 Continuing to next poll attempt in {poll_interval_minutes} minutes...")
                time.sleep(poll_interval_seconds)
        
        logger.warning(f"⏰ Quota recovery polling timed out after {max_wait_hours} hours")
        return False
    
    def _probe_quota_availability(self) -> Dict[str, Any]:
        """
        Probe quota availability by attempting a small test submission.
        This is the most reliable way to check if quota is available since OpenAI
        doesn't provide a direct quota usage API.
        
        Returns:
            Dictionary with quota probe results
        """
        logger.debug("🔍 Probing quota availability with test submission")
        
        try:
            # Create a minimal test request
            test_request = {
                "custom_id": f"quota_probe_{int(time.time())}",
                "method": "POST",
                "url": "/v1/embeddings",
                "body": {
                    "model": self.embedding_model,
                    "input": "test",
                    "dimensions": self.embedding_dimensions
                }
            }
            
            # Create temporary JSONL file for probe
            probe_file_path = os.path.join(self.config.get("checkpoint_dir", "data/checkpoints"), 
                                         f"quota_probe_{int(time.time())}.jsonl")
            
            with open(probe_file_path, 'w') as f:
                f.write(json.dumps(test_request) + '\n')
            
            # Upload probe file
            try:
                with open(probe_file_path, 'rb') as f:
                    file_response = self.openai_client.files.create(
                        file=f,
                        purpose='batch'
                    )
                probe_file_id = file_response.id
                logger.debug(f"📤 Uploaded quota probe file: {probe_file_id}")
                
                # Attempt to create batch job (this will fail if quota exceeded)
                try:
                    batch_response = self.openai_client.batches.create(
                        input_file_id=probe_file_id,
                        endpoint="/v1/embeddings",
                        completion_window="24h",
                        metadata={
                            "description": "Quota availability probe",
                            "created_by": "embedding_and_indexing_batch",
                            "probe": True
                        }
                    )
                    
                    # Success! Quota is available
                    probe_job_id = batch_response.id
                    logger.debug(f"✅ Quota probe successful: {probe_job_id}")
                    
                    # Immediately cancel the probe job to free quota
                    try:
                        self.openai_client.batches.cancel(probe_job_id)
                        logger.debug(f"🧹 Cancelled quota probe job: {probe_job_id}")
                    except Exception as cancel_error:
                        logger.warning(f"⚠️  Could not cancel probe job {probe_job_id}: {cancel_error}")
                    
                    # Clean up probe file
                    try:
                        self.openai_client.files.delete(probe_file_id)
                        logger.debug(f"🧹 Deleted quota probe file: {probe_file_id}")
                    except Exception as file_cleanup_error:
                        logger.warning(f"⚠️  Could not delete probe file {probe_file_id}: {file_cleanup_error}")
                    
                    return {
                        'quota_available': True,
                        'probe_successful': True,
                        'probe_job_id': probe_job_id,
                        'message': 'Quota probe successful - quota space available'
                    }
                    
                except Exception as batch_create_error:
                    # Check if this is a quota exceeded error
                    error_msg = str(batch_create_error).lower()
                    if 'request_limit_exceeded' in error_msg or 'quota' in error_msg or 'limit' in error_msg:
                        logger.warning(f"🚨 Quota probe detected quota exceeded: {batch_create_error}")
                        
                        # Clean up probe file
                        try:
                            self.openai_client.files.delete(probe_file_id)
                            logger.debug(f"🧹 Deleted quota probe file after quota exceeded: {probe_file_id}")
                        except Exception:
                            pass
                        
                        return {
                            'quota_available': False,
                            'probe_successful': True,
                            'quota_exceeded': True,
                            'error': str(batch_create_error),
                            'message': 'Quota probe detected quota exceeded - no space available'
                        }
                    else:
                        # Some other error
                        logger.error(f"❌ Quota probe failed with unexpected error: {batch_create_error}")
                        
                        # Clean up probe file
                        try:
                            self.openai_client.files.delete(probe_file_id)
                        except Exception:
                            pass
                        
                        return {
                            'quota_available': None,  # Unknown
                            'probe_successful': False,
                            'error': str(batch_create_error),
                            'message': 'Quota probe failed with unexpected error'
                        }
                        
            except Exception as file_upload_error:
                logger.error(f"❌ Failed to upload quota probe file: {file_upload_error}")
                
                # Clean up local file
                try:
                    os.remove(probe_file_path)
                except Exception:
                    pass
                
                return {
                    'quota_available': None,  # Unknown
                    'probe_successful': False,
                    'error': str(file_upload_error),
                    'message': 'Failed to upload quota probe file'
                }
            
            finally:
                # Clean up local probe file
                try:
                    if os.path.exists(probe_file_path):
                        os.remove(probe_file_path)
                        logger.debug(f"🧹 Cleaned up local probe file: {probe_file_path}")
                except Exception as cleanup_error:
                    logger.warning(f"⚠️  Could not clean up local probe file: {cleanup_error}")
                    
        except Exception as e:
            logger.error(f"❌ Quota probe failed with exception: {e}")
            return {
                'quota_available': None,  # Unknown
                'probe_successful': False,
                'error': str(e),
                'message': 'Quota probe failed with exception'
            }
    
    def _cleanup_failed_quota_jobs(self) -> int:
        """
        Proactively find and cleanup failed jobs that are consuming quota.
        This is more aggressive than the reactive cleanup in _check_and_update_queue_status.
        
        Returns:
            Number of failed jobs cleaned up
        """
        logger.info("🧹 Starting proactive cleanup of failed quota-consuming jobs")
        cleaned_up_count = 0
        
        try:
            # Get all embedding batch jobs from OpenAI API
            after = None
            total_checked = 0
            
            while True:
                if after:
                    batches = self.openai_client.batches.list(limit=100, after=after)
                else:
                    batches = self.openai_client.batches.list(limit=100)
                
                for batch in batches.data:
                    endpoint = getattr(batch, 'endpoint', '')
                    if '/embeddings' in endpoint or endpoint == '/v1/embeddings':
                        total_checked += 1
                        
                        # Check if job failed due to quota exceeded
                        if batch.status == 'failed':
                            if hasattr(batch, 'errors') and batch.errors:
                                for error in batch.errors.data:
                                    if error.code == 'request_limit_exceeded':
                                        logger.warning(f"🚨 Found quota-exceeded job {batch.id[:8]}... consuming quota")
                                        
                                        try:
                                            # Cancel the failed job to free up quota
                                            cancelled_batch = self.openai_client.batches.cancel(batch.id)
                                            logger.info(f"✅ Successfully cancelled quota-exceeded job {batch.id[:8]}...")
                                            cleaned_up_count += 1
                                            
                                            # Update local tracking if we have it
                                            if batch.id in self.batch_jobs:
                                                self.batch_jobs[batch.id]['status'] = 'cancelled'
                                                
                                        except Exception as cancel_error:
                                            # Check if already cancelled
                                            if 'already' in str(cancel_error).lower() and 'cancel' in str(cancel_error).lower():
                                                logger.info(f"ℹ️  Job {batch.id[:8]}... already cancelled")
                                                cleaned_up_count += 1
                                            else:
                                                logger.error(f"❌ Failed to cancel quota-exceeded job {batch.id[:8]}...: {cancel_error}")
                                        break
                
                if not batches.has_more:
                    break
                    
                if batches.data:
                    after = batches.data[-1].id
                else:
                    break
            
            logger.info(f"🧹 Cleanup complete: Checked {total_checked} embedding jobs, cleaned up {cleaned_up_count} quota-exceeded jobs")
            
            if cleaned_up_count > 0:
                logger.info(f"💡 Cleaned up {cleaned_up_count} failed jobs - quota space should now be available")
                
        except Exception as e:
            logger.error(f"❌ Error during failed job cleanup: {e}")
            
        return cleaned_up_count
    
    def _check_quota_capacity(self, new_requests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Check if we have enough quota capacity for new batch requests.
        Validates both token limits (500M) and request limits (1M).
        
        Args:
            new_requests: List of new embedding requests to submit
            
        Returns:
            Dictionary with quota check results
        """
        # Get current usage
        usage = self._get_current_usage()
        
        # Estimate tokens needed for new requests
        estimated_new_tokens = self._estimate_tokens_for_requests(new_requests)
        new_request_count = len(new_requests)
        
        # Calculate total usage with new requests
        total_estimated_tokens = usage['estimated_active_tokens'] + estimated_new_tokens
        total_estimated_requests = usage['total_active_requests'] + new_request_count
        
        # CORRECTED: No safety margins - aim for exactly 800K requests
        # Use the full token limit with minimal safety margin
        optimized_safety_margin = 0.01  # Just 1% safety margin for tokens
        safe_token_limit = int(self.token_quota_limit * (1 - optimized_safety_margin))
        
        # CORRECTED: Use exactly 800K as the target, no buffers or reductions
        safe_request_limit = self.request_quota_target  # Exactly 800K requests
        
        # Check if we're within all limits - allow exactly at the target (<=)
        within_token_quota = total_estimated_tokens <= safe_token_limit
        within_request_quota = total_estimated_requests <= safe_request_limit
        within_job_limit = usage['active_jobs'] < self.max_concurrent_jobs
        
        # Ensure individual request batch doesn't exceed per-job limit
        within_job_request_limit = new_request_count <= self.max_requests_per_job
        
        can_submit = (within_token_quota and within_request_quota and 
                     within_job_limit and within_job_request_limit)
        
        result = {
            'can_submit': can_submit,
            
            # Token quota info
            'current_token_usage': usage['estimated_active_tokens'],
            'new_request_tokens': estimated_new_tokens,
            'total_estimated_tokens': total_estimated_tokens,
            'token_quota_limit': self.token_quota_limit,
            'safe_token_limit': safe_token_limit,
            'token_quota_exceeded': not within_token_quota,
            
            # Request quota info
            'current_request_usage': usage['total_active_requests'],
            'new_request_count': new_request_count,
            'total_estimated_requests': total_estimated_requests,
            'request_quota_limit': self.request_quota_limit,
            'safe_request_limit': safe_request_limit,
            'request_quota_exceeded': not within_request_quota,
            
            # Job limits
            'active_jobs': usage['active_jobs'],
            'max_concurrent_jobs': self.max_concurrent_jobs,
            'job_limit_exceeded': not within_job_limit,
            
            # Per-job request limit
            'max_requests_per_job': self.max_requests_per_job,
            'job_request_limit_exceeded': not within_job_request_limit
        }
        
        logger.info(f"Quota check: {estimated_new_tokens:,} new tokens, {new_request_count:,} new requests, "
                   f"{usage['estimated_active_tokens']:,} active tokens, {usage['total_active_requests']:,} active requests, "
                   f"{total_estimated_tokens:,}/{safe_token_limit:,} total tokens, "
                   f"{total_estimated_requests:,}/{safe_request_limit:,} total requests "
                   f"(conservative limit: {safe_request_limit:,}/{self.request_quota_limit:,}) "
                   f"(can_submit: {can_submit})")
        
        return result
    
    def create_batch_jobs_only(self, string_dict: Dict[str, str], field_hash_mapping: Dict[str, Dict[str, int]],
                             string_counts: Dict[str, int], checkpoint_dir: str) -> Dict[str, Any]:
        """
        Create batch jobs without waiting for completion (manual polling mode).
        
        Args:
            string_dict: Dictionary mapping hash to string value
            field_hash_mapping: Mapping of hash to field types
            string_counts: Mapping of hash to frequency count
            checkpoint_dir: Directory for saving checkpoints
            
        Returns:
            Dictionary with job creation metrics
        """
        logger.info("Creating batch jobs for manual polling (without automated queue)")
        logger.info(f"🔧 Configuration: Conservative request limit: {self.request_quota_limit:,} (80% of 1M)")
        logger.info(f"🔧 Configuration: Max jobs per batch: {self.max_active_batches} (will pause after this many)")
        logger.info(f"🔧 Configuration: Safety margin: {self.quota_safety_margin:.1%}")
        start_time = time.time()
        
        # Load checkpoint
        self.load_checkpoint(checkpoint_dir)
        
        # Select strings to process
        strings_to_process = self._select_strings_to_process(
            string_dict, field_hash_mapping, string_counts
        )
        
        if not strings_to_process:
            logger.info("No new strings to process")
            return {
                'status': 'no_work',
                'message': 'All eligible strings already processed',
                'elapsed_time': time.time() - start_time
            }
        
        # Split into batches if needed (due to API limits)
        batches = []
        for i in range(0, len(strings_to_process), self.max_requests_per_file):
            batch = strings_to_process[i:i+self.max_requests_per_file]
            batches.append(batch)
        
        logger.info(f"Split {len(strings_to_process)} requests into {len(batches)} batch files")
        
        jobs_created = 0
        jobs_created_this_session = 0  # Track jobs created in this session
        
        # Create each batch job
        for batch_idx, batch in enumerate(batches):
            logger.info(f"Creating batch {batch_idx + 1}/{len(batches)} with {len(batch)} requests")
            
            try:
                # Create batch requests file
                batch_file_path = os.path.join(checkpoint_dir, f'batch_requests_{batch_idx}.jsonl')
                file_metadata = self._create_batch_requests_file(batch, batch_file_path)
                
                # Upload file
                input_file_id = self._upload_batch_file(batch_file_path)
                
                # Check quota capacity before creating batch job
                batch_requests = []
                with open(batch_file_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            import json
                            batch_requests.append(json.loads(line))
                
                quota_check = self._check_quota_capacity(batch_requests)
                
                if not quota_check['can_submit']:
                    logger.warning(f"🛑 Quota check failed for batch {batch_idx + 1}:")
                    logger.warning(f"  Current token usage: {quota_check['current_token_usage']:,} tokens")
                    logger.warning(f"  Current request usage: {quota_check['current_request_usage']:,} requests")
                    logger.warning(f"  New request tokens: {quota_check['new_request_tokens']:,}")
                    logger.warning(f"  New request count: {quota_check['new_request_count']:,}")
                    logger.warning(f"  Total tokens would be: {quota_check['total_estimated_tokens']:,}/{quota_check['safe_token_limit']:,}")
                    logger.warning(f"  Total requests would be: {quota_check['total_estimated_requests']:,}/{quota_check['safe_request_limit']:,}")
                    logger.warning(f"  Active jobs: {quota_check['active_jobs']}/{quota_check['max_concurrent_jobs']}")
                    
                    if quota_check['token_quota_exceeded']:
                        logger.warning(f"  ⚠️  Token quota would be exceeded")
                    if quota_check['request_quota_exceeded']:
                        logger.warning(f"  ⚠️  Request quota would be exceeded")
                    if quota_check['job_limit_exceeded']:
                        logger.warning(f"  ⚠️  Concurrent job limit would be exceeded")
                    if quota_check['job_request_limit_exceeded']:
                        logger.warning(f"  ⚠️  Per-job request limit would be exceeded ({quota_check['new_request_count']:,} > {quota_check['max_requests_per_job']:,})")
                    
                    logger.info(f"💡 Pausing batch creation after {jobs_created} jobs to respect quota limits")
                    logger.info(f"📊 Progress: {batch_idx}/{len(batches)} batches processed")
                    logger.info(f"💡 Wait for active jobs to complete, then run again to continue")
                    
                    # Clean up uploaded file since we won't use it
                    try:
                        self.openai_client.files.delete(input_file_id)
                        logger.debug(f"Cleaned up unused uploaded file {input_file_id}")
                    except Exception as cleanup_error:
                        logger.debug(f"Could not clean up file {input_file_id}: {cleanup_error}")
                    
                    # Return current progress
                    elapsed_time = time.time() - start_time
                    return {
                        'status': 'quota_paused',
                        'jobs_created': jobs_created,
                        'batches_processed': batch_idx,
                        'total_batches': len(batches),
                        'total_requests': sum(len(b) for b in batches[:batch_idx]),
                        'remaining_requests': sum(len(b) for b in batches[batch_idx:]),
                        'estimated_cost_savings': sum(len(b) for b in batches[:batch_idx]) * 0.00001 * 0.5,
                        'elapsed_time': elapsed_time,
                        'quota_info': quota_check,
                        'message': f'Created {jobs_created} batch jobs then paused due to quota limits. Wait for jobs to complete, then re-run to continue.'
                    }
                
                # Quota check passed - create batch job
                logger.info(f"✅ Quota check passed - creating batch job {batch_idx + 1}")
                # Create batch job
                batch_job_id = self._create_batch_job(
                    input_file_id, 
                    f"Entity resolution batch {batch_idx + 1}/{len(batches)}"
                )
                
                # Track batch job
                self.batch_jobs[batch_job_id] = {
                    'batch_idx': batch_idx,
                    'input_file_id': input_file_id,
                    'file_metadata': file_metadata,
                    'request_count': file_metadata['request_count'],  # Store exact request count
                    'created_at': time.time(),
                    'status': 'submitted'
                }
                
                jobs_created += 1
                jobs_created_this_session += 1
                logger.info(f"Created batch job {batch_job_id} for batch {batch_idx + 1}")
                
                # Save checkpoint after each job creation
                self.save_checkpoint(checkpoint_dir)
                
                # Check if we've reached the batch limit for this session
                if jobs_created_this_session >= self.max_active_batches:
                    logger.info(f"🛑 Reached batch limit of {self.max_active_batches} jobs created in this session")
                    logger.info(f"📊 Progress: {batch_idx + 1}/{len(batches)} batches processed")
                    logger.info(f"💡 Created {jobs_created_this_session} jobs, pausing for manual resumption")
                    logger.info(f"📋 Run the same command again to continue creating the remaining {len(batches) - (batch_idx + 1)} batches")
                    
                    elapsed_time = time.time() - start_time
                    return {
                        'status': 'batch_limit_reached',
                        'jobs_created': jobs_created,
                        'jobs_created_this_session': jobs_created_this_session,
                        'batches_processed': batch_idx + 1,
                        'total_batches': len(batches),
                        'total_requests': sum(len(b) for b in batches[:batch_idx + 1]),
                        'remaining_requests': sum(len(b) for b in batches[batch_idx + 1:]),
                        'estimated_cost_savings': sum(len(b) for b in batches[:batch_idx + 1]) * 0.00001 * 0.5,
                        'elapsed_time': elapsed_time,
                        'message': f'Created {jobs_created_this_session} batch jobs then paused due to batch limit. Re-run to continue with remaining {len(batches) - (batch_idx + 1)} batches.'
                    }
                
            except Exception as e:
                logger.error(f"Error creating batch job for batch {batch_idx}: {str(e)}")
                continue
        
        elapsed_time = time.time() - start_time
        
        logger.info(f"✅ Created {jobs_created} batch jobs in {elapsed_time:.2f} seconds")
        logger.info(f"📋 Jobs will process {len(strings_to_process)} embedding requests")
        logger.info(f"⏰ Check job status manually using: python batch_manager.py --status")
        logger.info(f"📥 Download results when ready using: python batch_manager.py --download")
        
        # If we completed all batches without hitting the job limit
        if jobs_created_this_session < self.max_active_batches:
            logger.info(f"🎉 All {len(batches)} batches completed in this session")
        
        return {
            'status': 'jobs_created',
            'jobs_created': jobs_created,
            'jobs_created_this_session': jobs_created_this_session,
            'total_requests': len(strings_to_process),
            'estimated_cost_savings': len(strings_to_process) * 0.00001 * 0.5,  # Rough estimate
            'elapsed_time': elapsed_time,
            'message': f'Created {jobs_created} batch jobs ({jobs_created_this_session} this session). Use --batch-status to check progress.'
        }
    
    def check_batch_status(self, checkpoint_dir: str) -> Dict[str, Any]:
        """
        Check status of all batch jobs without waiting.
        
        Args:
            checkpoint_dir: Directory containing checkpoint files
            
        Returns:
            Dictionary with status information for all jobs
        """
        logger.info("Checking status of batch jobs")
        
        # Load checkpoint
        self.load_checkpoint(checkpoint_dir)
        
        # If no batch jobs found locally, try to recover from API (READ-ONLY)
        if not self.batch_jobs:
            logger.info("No local batch jobs found - attempting to recover from OpenAI API (read-only)")
            try:
                self._recover_batch_jobs_from_api_readonly()
                if self.batch_jobs:
                    logger.info(f"Recovered {len(self.batch_jobs)} batch jobs from API (read-only)")
                    # DO NOT save checkpoint during status check - keep it read-only
                    logger.debug("Status check is read-only - not saving checkpoint")
                else:
                    logger.info("No batch jobs found in OpenAI API either")
                    return {
                        'status': 'no_jobs',
                        'message': 'No batch jobs found locally or in OpenAI API.'
                    }
            except Exception as e:
                logger.error(f"Failed to recover batch jobs from API: {str(e)}")
                return {
                    'status': 'no_jobs',
                    'message': 'No batch jobs found. Create jobs first with embedding_and_indexing stage.'
                }
        
        job_statuses = {}
        # Detailed status counters for maximum granularity
        status_counts = {
            BatchJobStatus.PENDING: 0,
            BatchJobStatus.VALIDATING: 0, 
            BatchJobStatus.IN_PROGRESS: 0,
            BatchJobStatus.FINALIZING: 0,
            BatchJobStatus.CANCELLING: 0,
            BatchJobStatus.COMPLETED: 0,
            BatchJobStatus.FAILED: 0,
            BatchJobStatus.EXPIRED: 0,
            BatchJobStatus.CANCELLED: 0,
            'error': 0  # For jobs we couldn't check
        }
        
        logger.info(f"Checking status of {len(self.batch_jobs)} batch jobs...")
        
        for batch_job_id, job_info in self.batch_jobs.items():
            try:
                batch_status = self.openai_client.batches.retrieve(batch_job_id)
                current_status = batch_status.status
                
                # Handle both recovered jobs (missing file_metadata) and original jobs
                request_count = None
                if 'file_metadata' in job_info:
                    request_count = job_info['file_metadata'].get('request_count')
                elif hasattr(batch_status, 'request_counts') and batch_status.request_counts:
                    # For recovered jobs, try to get request count from OpenAI batch status
                    request_count = getattr(batch_status.request_counts, 'total', None)
                
                # Enhanced status tracking with download/processing states
                download_status = 'not_applicable'
                processing_status = 'not_applicable'
                
                if current_status == BatchJobStatus.COMPLETED:
                    # Check if results have been downloaded
                    if job_info.get('results_downloaded', False):
                        download_status = 'downloaded'
                        # Check if results have been processed/indexed
                        if job_info.get('results_processed', False):
                            processing_status = 'processed'
                        else:
                            processing_status = 'pending_processing'
                    else:
                        download_status = 'pending_download'
                        processing_status = 'pending_download'
                
                job_statuses[batch_job_id] = {
                    'batch_idx': job_info['batch_idx'],
                    'status': current_status,
                    'download_status': download_status,
                    'processing_status': processing_status,
                    'created_at': job_info['created_at'],
                    'request_count': request_count,
                    'output_file_id': getattr(batch_status, 'output_file_id', None),
                    'recovered': job_info.get('recovered', False)
                }
                
                # Update local status
                self.batch_jobs[batch_job_id]['status'] = current_status
                if hasattr(batch_status, 'output_file_id'):
                    self.batch_jobs[batch_job_id]['output_file_id'] = batch_status.output_file_id
                
                # Count detailed statuses
                if current_status in status_counts:
                    status_counts[current_status] += 1
                else:
                    # Unknown status - count as failed for backwards compatibility
                    status_counts[BatchJobStatus.FAILED] += 1
                
                batch_display = job_info.get('batch_idx', 'unknown')
                if isinstance(batch_display, int):
                    batch_display += 1
                
                # Log failure details for failed jobs
                if current_status == BatchJobStatus.FAILED:
                    error_details = ""
                    if hasattr(batch_status, 'errors') and batch_status.errors:
                        error_details = f" - Errors: {batch_status.errors}"
                    elif hasattr(batch_status, 'request_counts'):
                        counts = batch_status.request_counts
                        error_details = f" - Requests: {getattr(counts, 'total', 0)} total, {getattr(counts, 'completed', 0)} completed, {getattr(counts, 'failed', 0)} failed"
                    logger.warning(f"Job {batch_job_id[:8]}... (batch {batch_display}): {current_status}{error_details}")
                else:
                    logger.info(f"Job {batch_job_id[:8]}... (batch {batch_display}): {current_status}")
                
            except Exception as e:
                logger.error(f"Error checking job {batch_job_id}: {str(e)}")
                job_statuses[batch_job_id] = {
                    'batch_idx': job_info.get('batch_idx', 'unknown'),
                    'status': 'error',
                    'error': str(e),
                    'recovered': job_info.get('recovered', False)
                }
                status_counts['error'] += 1
        
        # Do NOT save checkpoint during status check - keep it read-only
        logger.debug("Status check complete - no files were created or modified")
        
        # Detailed Summary
        total_jobs = len(self.batch_jobs)
        completed_count = status_counts[BatchJobStatus.COMPLETED]
        
        logger.info(f"\n📊 DETAILED BATCH JOB STATUS:")
        logger.info(f"   Total jobs: {total_jobs}")
        
        # Show each status with appropriate emoji and only if count > 0
        status_display = [
            (BatchJobStatus.PENDING, "⏳ Pending", status_counts[BatchJobStatus.PENDING]),
            (BatchJobStatus.VALIDATING, "🔍 Validating", status_counts[BatchJobStatus.VALIDATING]), 
            (BatchJobStatus.IN_PROGRESS, "🔄 In Progress", status_counts[BatchJobStatus.IN_PROGRESS]),
            (BatchJobStatus.FINALIZING, "🏁 Finalizing", status_counts[BatchJobStatus.FINALIZING]),
            (BatchJobStatus.COMPLETED, "✅ Completed", status_counts[BatchJobStatus.COMPLETED]),
            (BatchJobStatus.FAILED, "❌ Failed", status_counts[BatchJobStatus.FAILED]),
            (BatchJobStatus.EXPIRED, "⏰ Expired", status_counts[BatchJobStatus.EXPIRED]),
            (BatchJobStatus.CANCELLED, "🚫 Cancelled", status_counts[BatchJobStatus.CANCELLED]),
            ('error', "⚠️  Error", status_counts['error'])
        ]
        
        for status_key, label, count in status_display:
            if count > 0:
                logger.info(f"   {label}: {count}")
        
        # Show current status vs new conservative limits
        if total_jobs > 0:
            active_count = (status_counts[BatchJobStatus.PENDING] + 
                          status_counts[BatchJobStatus.VALIDATING] + 
                          status_counts[BatchJobStatus.IN_PROGRESS] + 
                          status_counts[BatchJobStatus.FINALIZING])
            
            logger.info(f"")
            logger.info(f"🔍 GRANULAR STATUS BREAKDOWN:")
            
            # Define status display order and emojis
            status_display = [
                (BatchJobStatus.PENDING, "⏳ Pending"),
                (BatchJobStatus.VALIDATING, "🔍 Validating"), 
                (BatchJobStatus.IN_PROGRESS, "🔄 In Progress"),
                (BatchJobStatus.FINALIZING, "🏁 Finalizing"),
                (BatchJobStatus.CANCELLING, "🛑 Cancelling"),
                (BatchJobStatus.COMPLETED, "✅ Completed"),
                (BatchJobStatus.FAILED, "❌ Failed"),
                (BatchJobStatus.EXPIRED, "⏰ Expired"),
                (BatchJobStatus.CANCELLED, "🚫 Cancelled"),
                ('error', "⚠️  Error")
            ]
            
            # Show each status if count > 0
            for status_key, label in status_display:
                count = status_counts.get(status_key, 0)
                if count > 0:
                    logger.info(f"   {label}: {count}")
            
            # Add failed request reporting
            if len(self.failed_requests) > 0 and self.enable_detailed_error_reporting:
                logger.info(f"")
                logger.info(f"🔍 FAILED REQUEST ANALYSIS:")
                
                failed_summary = self.get_failed_requests_summary()
                logger.info(f"   Total failed requests: {failed_summary['total_failed']}")
                
                if failed_summary['by_category']:
                    logger.info(f"   Failures by category:")
                    for category, count in failed_summary['by_category'].items():
                        logger.info(f"     • {category}: {count}")
                
                if failed_summary['retryable_requests'] > 0:
                    logger.info(f"   🔄 Retryable requests: {failed_summary['retryable_requests']}")
                if failed_summary['max_retries_exceeded'] > 0:
                    logger.info(f"   ❌ Max retries exceeded: {failed_summary['max_retries_exceeded']}")
            
            if active_count > 0:
                logger.info(f"")
                logger.info(f"⏳ Jobs are still processing. Check again later.")
            elif completed_count == total_jobs:
                logger.info(f"")
                logger.info(f"✅ All jobs completed! Ready to download and process results.")
        
        return {
            'status': 'checked',
            'total_jobs': total_jobs,
            'status_counts': status_counts,  # Detailed breakdown
            'completed': completed_count,
            'job_statuses': job_statuses,
            'ready_for_download': completed_count > 0,
            # Backwards compatibility - grouped counts
            'pending': status_counts[BatchJobStatus.PENDING] + status_counts[BatchJobStatus.VALIDATING],
            'in_progress': status_counts[BatchJobStatus.IN_PROGRESS] + status_counts[BatchJobStatus.FINALIZING],
            'failed': status_counts[BatchJobStatus.FAILED] + status_counts[BatchJobStatus.EXPIRED] + 
                     status_counts[BatchJobStatus.CANCELLED] + status_counts['error']
        }
    
    def process_completed_jobs(self, checkpoint_dir: str) -> Dict[str, Any]:
        """
        Download and process results from completed batch jobs.
        
        Args:
            checkpoint_dir: Directory containing checkpoint files
            
        Returns:
            Dictionary with processing metrics
        """
        logger.info("Processing completed batch jobs")
        start_time = time.time()
        
        # Ensure Weaviate connection for downloading and indexing
        self._ensure_weaviate_connection()
        
        # Load checkpoint
        self.load_checkpoint(checkpoint_dir)
        
        if not self.batch_jobs:
            logger.info("No batch jobs found")
            return {
                'status': 'no_jobs',
                'message': 'No batch jobs found'
            }
        
        # Find completed jobs that haven't been processed yet
        # We need to check actual status from OpenAI API, not just local cache
        completed_jobs = []
        for batch_job_id, job_info in self.batch_jobs.items():
            try:
                # Get real-time status from OpenAI API
                batch_status = self.openai_client.batches.retrieve(batch_job_id)
                current_status = batch_status.status
                
                if (current_status == BatchJobStatus.COMPLETED and 
                    not job_info.get('results_processed', False)):
                    completed_jobs.append((batch_job_id, job_info))
                    
            except Exception as e:
                logger.debug(f"Could not check status for job {batch_job_id}: {e}")
                # Fall back to cached status for this job
                if (job_info.get('status') == BatchJobStatus.COMPLETED and 
                    not job_info.get('results_processed', False)):
                    completed_jobs.append((batch_job_id, job_info))
        
        if not completed_jobs:
            logger.info("No new completed jobs to process")
            return {
                'status': 'no_completed_jobs',
                'message': 'No new completed jobs found. Check status first with --batch-status.'
            }
        
        logger.info(f"Found {len(completed_jobs)} completed jobs to process")
        
        total_processed = 0
        successful_jobs = 0
        failed_jobs = 0
        
        # Process each completed job
        for batch_job_id, job_info in completed_jobs:
            # Check if this batch job ID is blacklisted (skip previous attempts)
            if batch_job_id in self.blacklisted_files:
                logger.warning(f"⚫ Skipping blacklisted batch job {batch_job_id[:8]}... (batch {job_info['batch_idx'] + 1})")
                logger.info(f"📋 This job was from previous attempts with flawed source data and will be ignored")
                continue
                
            logger.info(f"Processing batch job {batch_job_id} (batch {job_info['batch_idx'] + 1})")
            
            try:
                # Get current job status to get output file ID
                batch_status = self.openai_client.batches.retrieve(batch_job_id)
                
                if not hasattr(batch_status, 'output_file_id') or not batch_status.output_file_id:
                    logger.error(f"No output file ID for job {batch_job_id}")
                    failed_jobs += 1
                    continue
                
                # Download results
                results_file_path = os.path.join(checkpoint_dir, f'batch_results_{job_info["batch_idx"]}.jsonl')
                self._download_batch_results(batch_status.output_file_id, results_file_path)
                
                # Check for and process error file if available
                if hasattr(batch_status, 'error_file_id') and batch_status.error_file_id:
                    logger.info(f"Error file found for job {batch_job_id}: {batch_status.error_file_id}")
                    error_analysis = self._process_error_file(batch_status.error_file_id, batch_job_id)
                    
                    # Store error analysis in job info
                    self.batch_jobs[batch_job_id]['error_analysis'] = error_analysis
                    logger.warning(f"Job {batch_job_id} had {error_analysis['total_errors']} failed requests. "
                                 f"Categories: {error_analysis['errors_by_category']}")
                else:
                    logger.debug(f"No error file for job {batch_job_id} - all requests succeeded")
                
                # Mark results as downloaded
                self.batch_jobs[batch_job_id]['results_downloaded'] = True                
                # Process results - handle recovered jobs without file_metadata
                if 'file_metadata' in job_info and 'custom_id_mapping' in job_info['file_metadata']:
                    # Original job with full metadata
                    custom_id_mapping = job_info['file_metadata']['custom_id_mapping']
                    items_to_index = self._process_batch_results(results_file_path, custom_id_mapping)
                else:
                    # Recovered job - process without custom_id_mapping
                    logger.warning(f"Job {batch_job_id} is recovered - processing without custom_id_mapping")
                    items_to_index = self._process_batch_results_without_mapping(results_file_path)
                
                # Index in Weaviate
                indexed_count = self._index_embeddings_batch(items_to_index)
                total_processed += indexed_count
                
                # Update processed hashes
                for item in items_to_index:
                    self.processed_hashes.add(item['hash_value'])
                
                # Clean up any failed requests that were successfully retried
                if self.cleanup_successful_retries_enabled:
                    cleaned_retries = self.cleanup_successful_retries()
                    if cleaned_retries > 0:
                        logger.info(f"Cleaned up {cleaned_retries} successfully retried requests")
                
                # Mark job as processed
                self.batch_jobs[batch_job_id]['results_processed'] = True
                self.batch_jobs[batch_job_id]['indexed_count'] = indexed_count
                self.batch_jobs[batch_job_id]['processed_at'] = time.time()
                
                successful_jobs += 1
                logger.info(f"✅ Processed job {batch_job_id}: indexed {indexed_count} items")
                
                # Save checkpoint after each job
                self.save_checkpoint(checkpoint_dir)
                
            except Exception as e:
                logger.error(f"Error processing job {batch_job_id}: {str(e)}")
                failed_jobs += 1
        
        elapsed_time = time.time() - start_time
        
        logger.info(f"\n📊 BATCH PROCESSING COMPLETE:")
        logger.info(f"   ✅ Successful jobs: {successful_jobs}")
        logger.info(f"   ❌ Failed jobs: {failed_jobs}")
        logger.info(f"   📊 Total items indexed: {total_processed}")
        logger.info(f"   ⏱️  Processing time: {elapsed_time:.2f} seconds")
        
        # Get collection stats
        try:
            result = self.collection.aggregate.over_all(total_count=True)
            collection_count = result.total_count
            logger.info(f"   🗄️  Collection now contains: {collection_count} objects")
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            collection_count = None
        
        return {
            'status': 'completed',
            'successful_jobs': successful_jobs,
            'failed_jobs': failed_jobs,
            'total_processed': total_processed,
            'elapsed_time': elapsed_time,
            'collection_count': collection_count
        }

    def process(self, string_dict: Dict[str, str], field_hash_mapping: Dict[str, Dict[str, int]],
              string_counts: Dict[str, int], checkpoint_dir: str) -> Dict[str, Any]:
        """
        Process strings using batch API: create jobs, wait for completion, and index results.
        
        Args:
            string_dict: Dictionary mapping hash to string value
            field_hash_mapping: Mapping of hash to field types
            string_counts: Mapping of hash to frequency count
            checkpoint_dir: Directory for saving checkpoints
            
        Returns:
            Dictionary with metrics about the process
        """
        logger.info("Starting batch embedding and indexing process")
        logger.info(f"🔧 Config flags: manual_polling={self.manual_polling}, use_automated_queue={self.use_automated_queue}")
        start_time = time.time()
        
        # Check if manual polling and automated queue are enabled
        if self.manual_polling and self.use_automated_queue:
            logger.info("✅ Manual polling mode with automated queue management enabled")
            return self.create_batch_jobs_with_automated_queue(string_dict, field_hash_mapping, string_counts, checkpoint_dir)
        elif self.manual_polling:
            logger.info("⚠️  Manual polling mode enabled - creating jobs without automated queue")
            return self.create_batch_jobs_only(string_dict, field_hash_mapping, string_counts, checkpoint_dir)
        
        # Continue with automatic polling (original behavior)
        logger.info("Automatic polling mode enabled - will wait for job completion")
        
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
                'total_cost_savings': 0,
                'collection_count': collection_count
            }
        
        # Split into batches if needed (due to API limits)
        batches = []
        for i in range(0, len(strings_to_process), self.max_requests_per_file):
            batch = strings_to_process[i:i+self.max_requests_per_file]
            batches.append(batch)
        
        logger.info(f"Split {len(strings_to_process)} requests into {len(batches)} batch files")
        
        total_processed = 0
        total_jobs_created = 0
        
        # Process each batch
        for batch_idx, batch in enumerate(batches):
            logger.info(f"Processing batch {batch_idx + 1}/{len(batches)} with {len(batch)} requests")
            
            try:
                # Create batch requests file
                batch_file_path = os.path.join(checkpoint_dir, f'batch_requests_{batch_idx}.jsonl')
                file_metadata = self._create_batch_requests_file(batch, batch_file_path)
                
                # Upload file
                input_file_id = self._upload_batch_file(batch_file_path)
                
                # Create batch job
                batch_job_id = self._create_batch_job(
                    input_file_id, 
                    f"Entity resolution batch {batch_idx + 1}/{len(batches)}"
                )
                
                # Track batch job
                self.batch_jobs[batch_job_id] = {
                    'batch_idx': batch_idx,
                    'input_file_id': input_file_id,
                    'file_metadata': file_metadata,
                    'request_count': file_metadata['request_count'],  # Store exact request count
                    'created_at': time.time(),
                    'status': 'created'
                }
                
                total_jobs_created += 1
                
                # Save checkpoint after creating job
                self.save_checkpoint(checkpoint_dir)
                
            except Exception as e:
                logger.error(f"Error creating batch job for batch {batch_idx}: {str(e)}")
                continue
        
        logger.info(f"Created {total_jobs_created} batch jobs, now waiting for completion...")
        
        # Wait for all jobs to complete and process results
        completed_jobs = 0
        failed_jobs = 0
        
        for batch_job_id, job_info in self.batch_jobs.items():
            if job_info.get('status') == 'completed':
                completed_jobs += 1
                continue
                
            logger.info(f"Waiting for batch job {batch_job_id} (batch {job_info['batch_idx'] + 1})")
            
            # Poll for completion
            job_result = self._poll_batch_job(batch_job_id)
            
            if job_result['status'] == BatchJobStatus.COMPLETED:
                try:
                    # Download results
                    results_file_path = os.path.join(checkpoint_dir, f'batch_results_{job_info["batch_idx"]}.jsonl')
                    self._download_batch_results(job_result['output_file_id'], results_file_path)
                    
                    # Process results - handle recovered jobs without file_metadata
                    if 'file_metadata' in job_info and 'custom_id_mapping' in job_info['file_metadata']:
                        # Original job with full metadata
                        custom_id_mapping = job_info['file_metadata']['custom_id_mapping']
                        items_to_index = self._process_batch_results(results_file_path, custom_id_mapping)
                    else:
                        # Recovered job - process without custom_id_mapping
                        logger.warning(f"Job {batch_job_id} is recovered - processing without custom_id_mapping")
                        items_to_index = self._process_batch_results_without_mapping(results_file_path)
                    
                    # Index in Weaviate
                    indexed_count = self._index_embeddings_batch(items_to_index)
                    total_processed += indexed_count
                    
                    # Update processed hashes
                    for item in items_to_index:
                        self.processed_hashes.add(item['hash_value'])
                    
                    # Update job status
                    self.batch_jobs[batch_job_id]['status'] = 'completed'
                    self.batch_jobs[batch_job_id]['indexed_count'] = indexed_count
                    
                    completed_jobs += 1
                    logger.info(f"Completed batch job {batch_job_id}: indexed {indexed_count} items")
                    
                    # Save checkpoint
                    self.save_checkpoint(checkpoint_dir)
                    
                except Exception as e:
                    logger.error(f"Error processing results for batch job {batch_job_id}: {str(e)}")
                    failed_jobs += 1
            else:
                logger.error(f"Batch job {batch_job_id} failed: {job_result.get('error', 'Unknown error')}")
                failed_jobs += 1
        
        elapsed_time = time.time() - start_time
        
        # Calculate cost savings (50% savings with batch API)
        estimated_standard_cost = len(strings_to_process) * 0.00001  # Rough estimate
        estimated_batch_cost = estimated_standard_cost * 0.5
        cost_savings = estimated_standard_cost - estimated_batch_cost
        
        logger.info(f"Batch processing completed in {elapsed_time:.2f} seconds")
        logger.info(f"Processed {total_processed} strings across {completed_jobs} completed jobs")
        logger.info(f"Failed jobs: {failed_jobs}")
        logger.info(f"Estimated cost savings: ${cost_savings:.4f} (50% reduction)")
        
        # Get collection stats
        try:
            result = self.collection.aggregate.over_all(total_count=True)
            collection_count = result.total_count
            logger.info(f"Collection now contains {collection_count} objects")
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            collection_count = None
        
        return {
            'status': 'completed',
            'elapsed_time': elapsed_time,
            'strings_processed': total_processed,
            'jobs_created': total_jobs_created,
            'jobs_completed': completed_jobs,
            'jobs_failed': failed_jobs,
            'collection_count': collection_count,
            'estimated_cost_savings': cost_savings
        }


def embedding_and_indexing_batch(config: Dict[str, Any], string_dict: Dict[str, str], 
                               field_hash_mapping: Dict[str, Dict[str, int]],
                               string_counts: Dict[str, int]) -> Dict[str, Any]:
    """
    Batch-based function for embedding generation and indexing.
    
    Args:
        config: Configuration dictionary
        string_dict: Dictionary mapping hash to string value
        field_hash_mapping: Mapping of hash to field types
        string_counts: Mapping of hash to frequency count
        
    Returns:
        Dictionary with metrics
    """
    logger.info("Starting batch embedding and indexing process")
    
    # Check if ColBERT provider is configured
    embedding_config = config.get('embedding', {})
    if embedding_config.get('provider') == 'jina_colbert':
        logger.warning("=" * 80)
        logger.warning("BATCH PROCESSING NOT SUPPORTED WITH COLBERT")
        logger.warning("=" * 80)
        logger.warning("Jina ColBERT uses automatic vectorization during data insertion.")
        logger.warning("Batch processing is not applicable for ColBERT embeddings.")
        logger.warning("Please set 'use_batch_embeddings: false' in your configuration")
        logger.warning("and use the standard embedding_and_indexing pipeline instead.")
        logger.warning("=" * 80)
        
        return {
            'status': 'error',
            'error': 'Batch processing not supported with ColBERT provider',
            'message': 'ColBERT uses automatic vectorization - disable batch mode',
            'elapsed_time': 0
        }
    
    # Get checkpoint directory
    checkpoint_dir = config.get("checkpoint_dir", "data/checkpoints")
    
    # Initialize the pipeline
    pipeline = None
    metrics = {}
    
    try:
        pipeline = BatchEmbeddingPipeline(config)
        
        # Process the data
        metrics = pipeline.process(string_dict, field_hash_mapping, string_counts, checkpoint_dir)
        
    except Exception as e:
        logger.error(f"Error in batch embedding and indexing pipeline: {str(e)}")
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


if __name__ == "__main__":
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description='Batch embedding and indexing for entity resolution')
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
        # Load config to get checkpoint directory with environment-specific overrides
        from src.config_utils import load_config_with_environment
        config = load_config_with_environment(args.config)
        
        checkpoint_dir = config.get("checkpoint_dir", "data/checkpoints")
        batch_files = [
            os.path.join(checkpoint_dir, 'batch_processed_hashes.pkl'),
            os.path.join(checkpoint_dir, 'batch_jobs.pkl')
        ]
        
        # Delete batch checkpoint files
        for file_path in batch_files:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Deleted checkpoint file: {file_path}")
    
    # Load configuration and run batch processing with environment-specific overrides
    from src.config_utils import load_config_with_environment
    config = load_config_with_environment(args.config)
    
    # Load preprocessing data
    checkpoint_dir = config.get("checkpoint_dir", "data/checkpoints")
    
    try:
        # Load required data
        with open(os.path.join(checkpoint_dir, "string_dict.pkl"), 'rb') as f:
            string_dict = pickle.load(f)
        
        with open(os.path.join(checkpoint_dir, "field_hash_mapping.pkl"), 'rb') as f:
            field_hash_mapping = pickle.load(f)
        
        with open(os.path.join(checkpoint_dir, "string_counts.pkl"), 'rb') as f:
            string_counts = pickle.load(f)
        
        logger.info(f"Loaded preprocessing data: {len(string_dict)} strings")
        
        # Run batch embedding and indexing
        metrics = embedding_and_indexing_batch(config, string_dict, field_hash_mapping, string_counts)
        
        # Save metrics
        output_dir = config.get("output_dir", "data/output")
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, "batch_embedding_metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Batch processing completed: {metrics}")
        
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        raise