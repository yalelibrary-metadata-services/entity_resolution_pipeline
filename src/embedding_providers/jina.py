"""
Jina AI Embedding Provider

This module implements the Jina AI embedding provider for the entity resolution pipeline.
It supports Jina's embedding API with various models.
"""

import os
import time
import logging
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import requests

from .base import EmbeddingProvider

logger = logging.getLogger(__name__)


class JinaEmbeddingProvider(EmbeddingProvider):
    """Jina AI embedding provider implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Jina provider with configuration.
        
        Args:
            config: Provider configuration dictionary
        """
        # API configuration
        self.api_key_env = config.get('api_key_env', 'JINA_API_KEY')
        self.api_key = os.environ.get(self.api_key_env)
        if not self.api_key:
            raise ValueError(f"Jina API key not found in environment variable: {self.api_key_env}")
        
        self.api_base = config.get('api_base', 'https://api.jina.ai/v1')
        self.model = config.get('model', 'jina-embeddings-v3')
        
        # Model-specific dimensions
        self.dimensions = self._get_model_dimensions(config)
        
        # Rate limiting configuration (Jina's limits are more generous)
        self.batch_size = config.get('batch_size', 100)
        self.rate_limit_delay = config.get('rate_limit_delay', 0.1)
        self.max_requests_per_minute = config.get('max_requests_per_minute', 500)
        self.max_tokens_per_minute = config.get('max_tokens_per_minute', 10_000_000)
        self.max_tokens_per_day = float('inf')  # No daily limit documented
        
        # Rate limiting state
        self.requests_this_minute = 0
        self.tokens_this_minute = 0
        self.minute_start = time.time()
        
        # Batch API support (Jina supports batch embeddings natively)
        self.supports_batch = config.get('supports_batch', True)
        
        # Task type for v3 models
        self.task_type = config.get('task_type', 'retrieval.passage')
        
        # Set JINAAI_APIKEY for Weaviate compatibility
        if self.api_key:
            os.environ['JINAAI_APIKEY'] = self.api_key
        
        logger.info(f"Initialized Jina provider with model {self.model} ({self.dimensions}D)")
    
    def _get_model_dimensions(self, config: Dict[str, Any]) -> int:
        """Get dimensions based on model."""
        if 'dimensions' in config:
            return config['dimensions']
        
        # Default dimensions for known models
        model_dimensions = {
            'jina-embeddings-v3': 1024,  # Can be customized
            'jina-embeddings-v2-base-en': 768,
            'jina-embeddings-v2-small-en': 512,
        }
        
        return model_dimensions.get(self.model, 1024)
    
    def get_embeddings(self, texts: List[str]) -> Tuple[List[np.ndarray], List[int]]:
        """
        Generate embeddings for a list of texts using Jina API.
        
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
                'Content-Type': 'application/json'
            }
            
            # Build request data
            data = {
                'model': self.model,
                'input': validated_texts,
                'encoding_type': 'float'
            }
            
            # Add task type for v3 models
            if 'v3' in self.model:
                data['task'] = self.task_type
                # Optionally specify dimensions for v3
                if hasattr(self, 'dimensions') and self.dimensions != 1024:
                    data['dimensions'] = self.dimensions
            
            # Make API request
            response = requests.post(
                f"{self.api_base}/embeddings",
                headers=headers,
                json=data,
                timeout=60
            )
            
            # Check for errors
            if response.status_code != 200:
                error_msg = f"Jina API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise requests.exceptions.RequestException(error_msg)
            
            # Parse response
            result = response.json()
            
            # Extract embeddings
            embeddings = []
            token_counts = []
            
            for item in result['data']:
                embedding = np.array(item['embedding'], dtype=np.float32)
                embeddings.append(embedding)
                # Jina provides usage information
                token_counts.append(item.get('usage', {}).get('total_tokens', self.estimate_tokens(texts[item['index']])))
            
            # Update rate limiting counters
            self.requests_this_minute += 1
            total_tokens = sum(token_counts)
            self.tokens_this_minute += total_tokens
            
            # Store usage info if provided
            if 'usage' in result:
                self.last_usage = result['usage']
            
            logger.debug(f"Generated {len(embeddings)} Jina embeddings")
            
            # Add configured delay between requests
            if self.rate_limit_delay > 0:
                time.sleep(self.rate_limit_delay)
            
            return embeddings, token_counts
            
        except Exception as e:
            logger.error(f"Error generating embeddings via Jina: {str(e)}")
            raise requests.exceptions.RequestException(f"Jina API error: {str(e)}")
    
    def get_dimensions(self) -> int:
        """Return the embedding dimensions for Jina."""
        return self.dimensions
    
    def get_model_name(self) -> str:
        """Return the Jina model name."""
        return self.model
    
    def get_weaviate_vectorizer_config(self) -> Optional[Dict[str, Any]]:
        """
        Return Weaviate vectorizer configuration for Jina.
        
        Returns:
            Dict with text2vec_jinaai configuration for proper Weaviate integration
        """
        try:
            from weaviate.classes.config import Configure
            
            # Return proper Jina vectorizer configuration
            logger.debug(f"Configuring text2vec_jinaai for model {self.model}")
            return Configure.Vectorizer.text2vec_jinaai(
                model=self.model,
                # Weaviate will use environment variable JINAAI_APIKEY
            )
            
        except (ImportError, AttributeError) as e:
            logger.warning(f"Could not configure Jina vectorizer: {e}. Using custom vectors.")
            return None  # Fall back to custom vectors
    
    def supports_batch_api(self) -> bool:
        """Whether Jina supports batch processing."""
        return self.supports_batch
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for Jina models.
        
        Args:
            text: Input text to estimate
            
        Returns:
            Estimated number of tokens
        """
        # Jina uses similar tokenization
        # Estimate ~4 characters per token
        return max(1, len(text) // 4)
    
    def get_rate_limit_config(self) -> Dict[str, Any]:
        """Get Jina rate limiting configuration."""
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
        
        # Estimate tokens for this request
        estimated_tokens = num_texts * 100  # Conservative estimate
        
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