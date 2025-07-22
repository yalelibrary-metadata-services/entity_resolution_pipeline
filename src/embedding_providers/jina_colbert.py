"""
Jina ColBERT Multi-Vector Embedding Provider with MUVERA Compression

This module implements the Jina ColBERT v2 provider for multi-vector embeddings
in the entity resolution pipeline. It supports two modes:

1. Automatic Vectorization Mode (Recommended): 
   - Weaviate automatically calls Jina API during data insertion
   - No manual embedding generation needed
   - MUVERA compression happens transparently in Weaviate
   
2. Manual Mode (Not recommended):
   - Manually call Jina API to get embeddings
   - Insert pre-computed embeddings into Weaviate
   - Loses benefits of automatic vectorization

The automatic mode is preferred as it simplifies the pipeline and ensures
proper MUVERA compression within Weaviate.
"""

import os
import time
import logging
from typing import List, Tuple, Dict, Any, Optional, Union

import numpy as np
import requests

from .base import EmbeddingProvider

try:
    from weaviate.classes.config import VectorDistances
except ImportError:
    VectorDistances = None

logger = logging.getLogger(__name__)


class JinaColBERTProvider(EmbeddingProvider):
    """Jina ColBERT v2 multi-vector embedding provider implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Jina ColBERT provider with configuration.
        
        Args:
            config: Provider configuration dictionary
        """
        # API configuration
        self.api_key_env = config.get('api_key_env', 'JINA_API_KEY')
        self.api_key = os.environ.get(self.api_key_env)
        if not self.api_key:
            raise ValueError(f"Jina API key not found in environment variable: {self.api_key_env}")
        
        self.api_base = config.get('api_base', 'https://api.jina.ai/v1')
        self.model = config.get('model', 'jina-colbert-v2')
        
        # Automatic vectorization mode (recommended)
        self.use_automatic_vectorization = config.get('use_automatic_vectorization', True)
        
        # ColBERT specific settings
        # Note: 'dimensions' field is the per-token embedding size (typically 128 for ColBERT)
        self.token_dimensions = config.get('dimensions', 128)
        # Jina ColBERT v2 supports up to 8,192 tokens (extendable to 12,288)
        self.max_tokens_per_text = config.get('max_tokens_per_text', 8192)
        
        # MUVERA compression configuration
        muvera_config = config.get('muvera', {})
        self.muvera_config = {
            'enabled': muvera_config.get('enabled', True),
            'ksim': muvera_config.get('ksim', 8),  # Number of Gaussian buckets (2^8 = 256)
            'dprojections': muvera_config.get('dprojections', 16),  # Dimensionality of sub-vectors
            'repetitions': muvera_config.get('repetitions', 1)  # Number of encoding iterations
        }
        
        # Calculate MUVERA compressed dimensions (for index, not storage)
        # This is what the index uses internally, but fetched vectors will be multi-vectors
        self.muvera_dimensions = (2 ** self.muvera_config['ksim']) * self.muvera_config['dprojections'] * self.muvera_config['repetitions']
        
        # Rate limiting configuration (only used in manual mode)
        self.batch_size = config.get('batch_size', 32)  # Smaller batch for multi-vectors
        self.rate_limit_delay = config.get('rate_limit_delay', 0.2)
        self.max_requests_per_minute = config.get('max_requests_per_minute', 100)
        self.max_tokens_per_minute = config.get('max_tokens_per_minute', 10_000_000)
        self.max_tokens_per_day = float('inf')
        
        # Rate limiting state
        self.requests_this_minute = 0
        self.tokens_this_minute = 0
        self.minute_start = time.time()
        
        # Batch API support - disabled for ColBERT in automatic mode
        self.supports_batch = False if self.use_automatic_vectorization else config.get('supports_batch', True)
        
        # Set JINAAI_APIKEY for Weaviate compatibility
        if self.api_key:
            os.environ['JINAAI_APIKEY'] = self.api_key
        
        mode = "automatic vectorization" if self.use_automatic_vectorization else "manual"
        logger.info(f"Initialized Jina ColBERT provider with model {self.model} in {mode} mode")
    
    def get_embeddings(self, texts: List[str]) -> Tuple[List[Union[np.ndarray, List[np.ndarray]]], List[int]]:
        """
        Generate embeddings for a list of texts.
        
        In automatic vectorization mode (recommended), this returns placeholder embeddings
        since Weaviate will handle the actual vectorization during data insertion.
        
        In manual mode, this calls the Jina ColBERT API to get multi-vector embeddings.
        
        Args:
            texts: List of strings to embed
            
        Returns:
            Tuple of (embeddings, token_counts)
            - Automatic mode: Returns None embeddings as placeholders
            - Manual mode: Returns list of multi-vector embeddings
        """
        # Validate inputs
        validated_texts = self.validate_texts(texts)
        
        # In automatic vectorization mode, return placeholders
        if self.use_automatic_vectorization:
            # Return None embeddings to signal automatic vectorization
            # The pipeline should detect this and skip embedding insertion
            embeddings = [None for _ in validated_texts]
            token_counts = [0 for _ in validated_texts]
            logger.debug(f"Automatic vectorization mode: returning {len(embeddings)} placeholder embeddings")
            return embeddings, token_counts
        
        # Manual mode: Generate embeddings via API
        # Enforce rate limits
        self._enforce_rate_limits(len(validated_texts))
        
        try:
            # Prepare request
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            # Build request data for ColBERT
            data = {
                'model': self.model,
                'input': validated_texts,
                'encoding_type': 'float',
                'input_type': 'document',  # ColBERT uses document encoding
            }
            
            # Make API request
            response = requests.post(
                f"{self.api_base}/embeddings",
                headers=headers,
                json=data,
                timeout=120  # Longer timeout for multi-vector processing
            )
            
            # Check for errors
            if response.status_code != 200:
                error_msg = f"Jina ColBERT API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise requests.exceptions.RequestException(error_msg)
            
            # Parse response
            result = response.json()
            
            # Extract multi-vector embeddings
            embeddings = []
            token_counts = []
            
            for item in result['data']:
                # ColBERT returns multiple embeddings per text (one per token)
                if 'embeddings' in item:  # Multi-vector response
                    token_embeddings = [np.array(emb, dtype=np.float32) for emb in item['embeddings']]
                    embeddings.append(token_embeddings)
                    token_counts.append(len(token_embeddings))
                else:  # Fallback to single embedding
                    embedding = np.array(item['embedding'], dtype=np.float32)
                    embeddings.append([embedding])  # Wrap in list for consistency
                    token_counts.append(1)
            
            # Update rate limiting counters
            self.requests_this_minute += 1
            total_tokens = sum(token_counts)
            self.tokens_this_minute += total_tokens
            
            logger.debug(f"Generated {len(embeddings)} Jina ColBERT multi-vector embeddings (manual mode)")
            
            # Add configured delay between requests
            if self.rate_limit_delay > 0:
                time.sleep(self.rate_limit_delay)
            
            return embeddings, token_counts
            
        except Exception as e:
            logger.error(f"Error generating embeddings via Jina ColBERT: {str(e)}")
            raise requests.exceptions.RequestException(f"Jina ColBERT API error: {str(e)}")
    
    def get_dimensions(self) -> int:
        """
        Return the MUVERA compressed dimensions.
        
        Note: While ColBERT generates multi-vector embeddings (one vector per token),
        MUVERA compresses these for the index. This method returns the compressed
        dimensions used by the index for compatibility with the pipeline.
        """
        return self.muvera_dimensions
    
    def get_model_name(self) -> str:
        """Return the Jina ColBERT model name."""
        return self.model
    
    def get_weaviate_vectorizer_config(self) -> Optional[Union[Dict[str, Any], List[Any]]]:
        """
        Return Weaviate vectorizer configuration for Jina ColBERT.
        
        In automatic vectorization mode, returns a special marker that tells the pipeline
        to use text2colbert-jinaai with MUVERA compression.
        
        In manual mode, returns None to use custom vectors.
        
        Returns:
            Configuration for Jina ColBERT
        """
        if self.use_automatic_vectorization:
            # Return special marker for automatic vectorization with MUVERA
            logger.debug(f"Configuring text2colbert_jinaai with MUVERA for model {self.model}")
            return {
                '_is_colbert': True,
                '_config': self.muvera_config,
                'model': self.model,
                'use_automatic_vectorization': True
            }
        else:
            # Manual mode - use custom vectors
            logger.debug(f"Manual mode: no Weaviate vectorizer configuration needed")
            return None
    
    def supports_batch_api(self) -> bool:
        """Whether Jina ColBERT supports batch processing."""
        return self.supports_batch
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for ColBERT models.
        
        Args:
            text: Input text to estimate
            
        Returns:
            Estimated number of tokens
        """
        # ColBERT typically uses wordpiece tokenization
        # Estimate ~1.3 tokens per word
        words = len(text.split())
        return min(int(words * 1.3), self.max_tokens_per_text)
    
    def get_rate_limit_config(self) -> Dict[str, Any]:
        """Get Jina ColBERT rate limiting configuration."""
        return {
            'max_requests_per_minute': self.max_requests_per_minute,
            'max_tokens_per_minute': self.max_tokens_per_minute,
            'rate_limit_delay': self.rate_limit_delay,
            'batch_size': self.batch_size
        }
    
    def _enforce_rate_limits(self, num_texts: int) -> None:
        """
        Enforce per-minute rate limits for Jina API.
        
        Args:
            num_texts: Number of texts being processed
        """
        current_time = time.time()
        
        # Reset minute counters if new minute
        if current_time - self.minute_start >= 60:
            self.requests_this_minute = 0
            self.tokens_this_minute = 0
            self.minute_start = current_time
        
        # Estimate tokens for this request (more for ColBERT)
        estimated_tokens = num_texts * 200  # Higher estimate for multi-vectors
        
        # Check per-minute limits
        if (self.requests_this_minute >= self.max_requests_per_minute or 
            self.tokens_this_minute + estimated_tokens >= self.max_tokens_per_minute):
            
            # Sleep until next minute
            sleep_time = 60 - (current_time - self.minute_start)
            if sleep_time > 0:
                logger.debug(f"Jina rate limit reached, sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
                
                # Reset counters after sleep
                self.requests_this_minute = 0
                self.tokens_this_minute = 0
                self.minute_start = time.time()