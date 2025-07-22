"""
Cohere Embedding Provider

This module implements the Cohere embedding provider for the entity resolution pipeline.
It supports Cohere's embedding API with various multilingual and English models.
"""

import os
import time
import logging
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import requests

from .base import EmbeddingProvider

logger = logging.getLogger(__name__)


class CohereEmbeddingProvider(EmbeddingProvider):
    """Cohere embedding provider implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Cohere provider with configuration.
        
        Args:
            config: Provider configuration dictionary
        """
        # API configuration
        self.api_key_env = config.get('api_key_env', 'COHERE_API_KEY')
        self.api_key = os.environ.get(self.api_key_env)
        if not self.api_key:
            raise ValueError(f"Cohere API key not found in environment variable: {self.api_key_env}")
        
        self.api_base = config.get('api_base', 'https://api.cohere.ai/v1')
        self.model = config.get('model', 'embed-multilingual-v3.0')
        
        # Model-specific dimensions
        self.dimensions = self._get_model_dimensions(config)
        
        # Input type for embeddings
        self.input_type = config.get('input_type', 'search_document')
        
        # Truncation setting
        self.truncate = config.get('truncate', 'END')
        
        # Rate limiting configuration (Cohere's limits)
        self.batch_size = config.get('batch_size', 96)  # Cohere supports up to 96 texts per request
        self.rate_limit_delay = config.get('rate_limit_delay', 0.1)
        self.max_requests_per_minute = config.get('max_requests_per_minute', 100)
        self.max_tokens_per_minute = config.get('max_tokens_per_minute', 10_000_000)
        self.max_tokens_per_day = float('inf')  # No daily limit documented
        
        # Rate limiting state
        self.requests_this_minute = 0
        self.tokens_this_minute = 0
        self.minute_start = time.time()
        
        # Batch API support
        self.supports_batch = config.get('supports_batch', True)
        
        # Set COHERE_APIKEY for Weaviate compatibility
        if self.api_key:
            os.environ['COHERE_APIKEY'] = self.api_key
        
        logger.info(f"Initialized Cohere provider with model {self.model} ({self.dimensions}D)")
    
    def _get_model_dimensions(self, config: Dict[str, Any]) -> int:
        """Get dimensions based on model."""
        if 'dimensions' in config:
            return config['dimensions']
        
        # Default dimensions for known models
        model_dimensions = {
            'embed-multilingual-v3.0': 1024,
            'embed-multilingual-light-v3.0': 384,
            'embed-english-v3.0': 1024,
            'embed-english-light-v3.0': 384,
            'embed-v4.0': 1024,  # Latest model
        }
        
        return model_dimensions.get(self.model, 1024)
    
    def get_embeddings(self, texts: List[str]) -> Tuple[List[np.ndarray], List[int]]:
        """
        Generate embeddings for a list of texts using Cohere API.
        
        Args:
            texts: List of strings to embed
            
        Returns:
            Tuple of (embeddings, token_counts)
        """
        # Validate inputs
        validated_texts = self.validate_texts(texts)
        
        # Enforce rate limits
        self._enforce_rate_limits(len(validated_texts))
        
        try:
            # Prepare request
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
                'X-Client-Name': 'entity-resolver'
            }
            
            # Build request data
            data = {
                'model': self.model,
                'texts': validated_texts,
                'input_type': self.input_type,
                'truncate': self.truncate,
            }
            
            # Make API request
            response = requests.post(
                f"{self.api_base}/embed",
                headers=headers,
                json=data,
                timeout=60
            )
            
            # Check for errors
            if response.status_code != 200:
                error_msg = f"Cohere API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise requests.exceptions.RequestException(error_msg)
            
            # Parse response
            result = response.json()
            
            # Extract embeddings
            embeddings = []
            for embedding in result['embeddings']:
                embedding_array = np.array(embedding, dtype=np.float32)
                embeddings.append(embedding_array)
            
            # Estimate token counts (Cohere doesn't provide exact counts)
            token_counts = [self.estimate_tokens(text) for text in validated_texts]
            
            # Update rate limiting counters
            self.requests_this_minute += 1
            total_tokens = sum(token_counts)
            self.tokens_this_minute += total_tokens
            
            # Store metadata if provided
            if 'meta' in result:
                self.last_meta = result['meta']
            
            logger.debug(f"Generated {len(embeddings)} Cohere embeddings")
            
            # Add configured delay between requests
            if self.rate_limit_delay > 0:
                time.sleep(self.rate_limit_delay)
            
            return embeddings, token_counts
            
        except Exception as e:
            logger.error(f"Error generating embeddings via Cohere: {str(e)}")
            raise requests.exceptions.RequestException(f"Cohere API error: {str(e)}")
    
    def get_dimensions(self) -> int:
        """Return the embedding dimensions for Cohere."""
        return self.dimensions
    
    def get_model_name(self) -> str:
        """Return the Cohere model name."""
        return self.model
    
    def get_weaviate_vectorizer_config(self) -> Optional[Dict[str, Any]]:
        """
        Return Weaviate vectorizer configuration for Cohere.
        
        Returns:
            Dict with text2vec_cohere configuration for proper Weaviate integration
        """
        try:
            from weaviate.classes.config import Configure
            
            # Return proper Cohere vectorizer configuration
            logger.debug(f"Configuring text2vec_cohere for model {self.model}")
            
            # Build configuration
            config_args = {
                'model': self.model,
                # Weaviate will use environment variable COHERE_APIKEY
            }
            
            # Add optional parameters if not using defaults
            if self.truncate != 'END':
                config_args['truncate'] = self.truncate
            
            return Configure.Vectorizer.text2vec_cohere(**config_args)
            
        except (ImportError, AttributeError) as e:
            logger.warning(f"Could not configure Cohere vectorizer: {e}. Using custom vectors.")
            return None  # Fall back to custom vectors
    
    def supports_batch_api(self) -> bool:
        """Whether Cohere supports batch processing."""
        return self.supports_batch
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for Cohere models.
        
        Args:
            text: Input text to estimate
            
        Returns:
            Estimated number of tokens
        """
        # Cohere uses BPE tokenization similar to other models
        # Estimate ~4 characters per token
        return max(1, len(text) // 4)
    
    def get_rate_limit_config(self) -> Dict[str, Any]:
        """Get Cohere rate limiting configuration."""
        return {
            'max_requests_per_minute': self.max_requests_per_minute,
            'max_tokens_per_minute': self.max_tokens_per_minute,
            'rate_limit_delay': self.rate_limit_delay,
            'batch_size': self.batch_size
        }
    
    def _enforce_rate_limits(self, num_texts: int) -> None:
        """
        Enforce per-minute rate limits for Cohere API.
        
        Args:
            num_texts: Number of texts being processed
        """
        current_time = time.time()
        
        # Reset minute counters if new minute
        if current_time - self.minute_start >= 60:
            self.requests_this_minute = 0
            self.tokens_this_minute = 0
            self.minute_start = current_time
        
        # Estimate tokens for this request
        estimated_tokens = num_texts * 100  # Conservative estimate
        
        # Check per-minute limits
        if (self.requests_this_minute >= self.max_requests_per_minute or 
            self.tokens_this_minute + estimated_tokens >= self.max_tokens_per_minute):
            
            # Sleep until next minute
            sleep_time = 60 - (current_time - self.minute_start)
            if sleep_time > 0:
                logger.debug(f"Cohere rate limit reached, sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
                
                # Reset counters after sleep
                self.requests_this_minute = 0
                self.tokens_this_minute = 0
                self.minute_start = time.time()