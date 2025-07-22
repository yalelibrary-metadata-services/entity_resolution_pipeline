"""
Mistral Embedding Provider

This module implements the Mistral.AI embedding provider for the entity resolution pipeline.
It supports Mistral's embedding API with the mistral-embed model.
"""

import os
import time
import logging
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import requests

from .base import EmbeddingProvider

logger = logging.getLogger(__name__)


class MistralEmbeddingProvider(EmbeddingProvider):
    """Mistral embedding provider implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Mistral provider with configuration.
        
        Args:
            config: Provider configuration dictionary
        """
        # API configuration
        self.api_key_env = config.get('api_key_env', 'MISTRAL_API_KEY')
        self.api_key = os.environ.get(self.api_key_env)
        if not self.api_key:
            raise ValueError(f"Mistral API key not found in environment variable: {self.api_key_env}")
        
        self.api_base = config.get('api_base', 'https://api.mistral.ai/v1')
        self.model = config.get('model', 'mistral-embed')
        self.dimensions = config.get('dimensions', 1024)
        
        # Rate limiting configuration
        self.batch_size = config.get('batch_size', 25)
        self.rate_limit_delay = config.get('rate_limit_delay', 0.5)
        self.max_requests_per_minute = config.get('max_requests_per_minute', 360)
        self.max_tokens_per_minute = config.get('max_tokens_per_minute', 19_000_000)
        self.max_tokens_per_month = config.get('max_tokens_per_month', 190_000_000_000)
        # Calculate daily limit from monthly (monthly / 30 days)
        self.max_tokens_per_day = self.max_tokens_per_month // 30
        
        # Rate limiting state
        self.requests_this_minute = 0
        self.tokens_this_minute = 0
        self.minute_start = time.time()
        self.tokens_this_month = 0
        self.month_start = time.time()
        
        # Batch API support (Mistral currently doesn't support batch embeddings)
        self.supports_batch = config.get('supports_batch', False)
        
        # Initialize Mistral client
        try:
            from mistralai import Mistral
            self.client = Mistral(api_key=self.api_key)
        except ImportError as e:
            raise ImportError("Mistral client not available. Install with: pip install mistralai") from e
        
        # Set MISTRAL_APIKEY for Weaviate compatibility if using our default variable
        if self.api_key_env == 'MISTRAL_API_KEY' and self.api_key:
            os.environ['MISTRAL_APIKEY'] = self.api_key
        
        logger.info(f"Initialized Mistral provider with model {self.model} ({self.dimensions}D)")
    
    def get_embeddings(self, texts: List[str]) -> Tuple[List[np.ndarray], List[int]]:
        """
        Generate embeddings for a list of texts using Mistral API.
        
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
            # Use Mistral client to get embeddings (new API)
            response = self.client.embeddings.create(
                model=self.model,
                inputs=validated_texts
            )
            
            # Extract embeddings
            embeddings = []
            for embedding_data in response.data:
                embedding = np.array(embedding_data.embedding, dtype=np.float32)
                embeddings.append(embedding)
            
            # Estimate token counts (Mistral doesn't provide detailed token usage)
            token_counts = [self.estimate_tokens(text) for text in validated_texts]
            
            # Update rate limiting counters
            self.requests_this_minute += 1
            total_tokens = sum(token_counts)
            self.tokens_this_minute += total_tokens
            self.tokens_this_month += total_tokens
            
            logger.debug(f"Generated {len(embeddings)} Mistral embeddings")
            
            # Add configured delay between requests
            if self.rate_limit_delay > 0:
                time.sleep(self.rate_limit_delay)
            
            return embeddings, token_counts
            
        except Exception as e:
            logger.error(f"Error generating embeddings via Mistral: {str(e)}")
            raise requests.exceptions.RequestException(f"Mistral API error: {str(e)}")
    
    def get_dimensions(self) -> int:
        """Return the embedding dimensions for Mistral."""
        return self.dimensions
    
    def get_model_name(self) -> str:
        """Return the Mistral model name."""
        return self.model
    
    def get_weaviate_vectorizer_config(self) -> Optional[Dict[str, Any]]:
        """
        Return Weaviate vectorizer configuration for Mistral.
        
        Returns:
            Dict with text2vec_mistral configuration for proper Weaviate integration
        """
        try:
            from weaviate.classes.config import Configure
            
            # Return proper Mistral vectorizer configuration
            # This uses the text2vec-mistral module in Weaviate
            logger.debug(f"Configuring text2vec_mistral for model {self.model}")
            return Configure.Vectorizer.text2vec_mistral(
                model=self.model,
                # Weaviate will use environment variable MISTRAL_APIKEY
                # or API key can be passed at runtime via headers
            )
            
        except (ImportError, AttributeError) as e:
            logger.warning(f"Could not configure Mistral vectorizer: {e}. Using custom vectors.")
            return None  # Fall back to custom vectors
    
    def supports_batch_api(self) -> bool:
        """Whether Mistral supports batch processing."""
        return self.supports_batch
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for Mistral models.
        
        Args:
            text: Input text to estimate
            
        Returns:
            Estimated number of tokens
        """
        # Mistral uses similar tokenization to other models
        # Estimate ~4 characters per token for most languages
        return max(1, len(text) // 4)
    
    def get_rate_limit_config(self) -> Dict[str, Any]:
        """Get Mistral rate limiting configuration."""
        return {
            'max_requests_per_minute': self.max_requests_per_minute,
            'max_tokens_per_minute': self.max_tokens_per_minute,
            'max_tokens_per_month': self.max_tokens_per_month,
            'rate_limit_delay': self.rate_limit_delay,
            'batch_size': self.batch_size
        }
    
    def _enforce_rate_limits(self, num_texts: int) -> None:
        """
        Enforce per-minute and monthly rate limits for Mistral API.
        
        Args:
            num_texts: Number of texts being processed
        """
        current_time = time.time()
        
        # Reset minute counters if new minute
        if current_time - self.minute_start >= 60:
            self.requests_this_minute = 0
            self.tokens_this_minute = 0
            self.minute_start = current_time
        
        # Reset monthly counters if new month (30 days)
        if current_time - self.month_start >= 2592000:  # 30 days in seconds
            self.tokens_this_month = 0
            self.month_start = current_time
            logger.info("Monthly token usage reset for Mistral")
        
        # Estimate tokens for this request
        estimated_tokens = num_texts * 100  # Conservative estimate
        
        # Check monthly limits first
        if self.tokens_this_month + estimated_tokens >= self.max_tokens_per_month:
            logger.error(f"Mistral monthly token limit would be exceeded: {self.tokens_this_month + estimated_tokens} >= {self.max_tokens_per_month}")
            raise requests.exceptions.RequestException("Monthly token limit exceeded")
        
        # Check per-minute limits
        if (self.requests_this_minute >= self.max_requests_per_minute or 
            self.tokens_this_minute + estimated_tokens >= self.max_tokens_per_minute):
            
            # Sleep until next minute
            sleep_time = 60 - (current_time - self.minute_start)
            if sleep_time > 0:
                logger.debug(f"Mistral rate limit reached, sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
                
                # Reset counters after sleep
                self.requests_this_minute = 0
                self.tokens_this_minute = 0
                self.minute_start = time.time()