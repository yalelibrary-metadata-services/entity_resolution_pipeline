"""
OpenAI Embedding Provider

This module implements the OpenAI embedding provider for the entity resolution pipeline.
It extracts the existing OpenAI logic into a clean provider interface.
"""

import os
import time
import json
import logging
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from openai import OpenAI
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import requests

from .base import EmbeddingProvider

logger = logging.getLogger(__name__)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize OpenAI provider with configuration.
        
        Args:
            config: Provider configuration dictionary
        """
        # API configuration
        self.api_key_env = config.get('api_key_env', 'OPENAI_API_KEY')
        self.api_key = os.environ.get(self.api_key_env)
        if not self.api_key:
            raise ValueError(f"OpenAI API key not found in environment variable: {self.api_key_env}")
        
        self.api_base = config.get('api_base', 'https://api.openai.com/v1')
        self.model = config.get('model', 'text-embedding-3-small')
        self.dimensions = config.get('dimensions', 1536)
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        
        # Rate limiting configuration
        self.batch_size = config.get('batch_size', 512)
        self.max_tokens_per_minute = config.get('max_tokens_per_minute', 4_800_000)
        self.max_requests_per_minute = config.get('max_requests_per_minute', 9_500)
        self.max_tokens_per_day = config.get('max_tokens_per_day', 480_000_000)
        
        # Rate limiting state
        self.tokens_this_minute = 0
        self.requests_this_minute = 0
        self.minute_start = time.time()
        self.tokens_today = 0
        self.day_start = time.time()
        
        # API monitoring
        self.api_rate_limit_info = {
            'remaining_requests': None,
            'remaining_tokens': None,
            'reset_requests': None,
            'reset_tokens': None,
            'limit_requests': None,
            'limit_tokens': None
        }
        self.last_api_headers = {}
        
        # Batch API support
        self.supports_batch = config.get('supports_batch', True)
        self.batch_endpoint = config.get('batch_endpoint', '/v1/embeddings/batch')
        self.max_batch_size = config.get('max_batch_size', 50000)
        
        logger.info(f"Initialized OpenAI provider with model {self.model} ({self.dimensions}D)")
    
    def get_embeddings(self, texts: List[str]) -> Tuple[List[np.ndarray], List[int]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of strings to embed
            
        Returns:
            Tuple of (embeddings, token_counts)
        """
        # Validate inputs
        validated_texts = self.validate_texts(texts)
        
        # Check daily rate limit
        if not self._check_daily_rate_limit():
            logger.warning("Daily token limit reached")
            raise requests.exceptions.RequestException("Daily token limit exceeded")
        
        # Check and enforce per-minute rate limits
        self._enforce_rate_limits(len(validated_texts))
        
        try:
            # Get embeddings using the OpenAI client
            response = self.client.embeddings.create(
                model=self.model,
                input=validated_texts
            )
            
            # Parse API headers for rate limit monitoring
            self._parse_api_headers(response)
            
            # Extract embeddings
            embeddings = []
            for embedding_data in response.data:
                embedding = np.array(embedding_data.embedding, dtype=np.float32)
                embeddings.append(embedding)
            
            # Get token count
            token_count = response.usage.total_tokens
            token_counts = [token_count // len(validated_texts)] * len(validated_texts)  # Distribute evenly
            
            # Update rate limiting counters
            self.tokens_this_minute += token_count
            self.requests_this_minute += 1
            self.tokens_today += token_count
            
            logger.debug(f"Generated {len(embeddings)} embeddings using {token_count} tokens")
            
            return embeddings, token_counts
            
        except Exception as e:
            logger.error(f"Error generating embeddings via OpenAI: {str(e)}")
            raise requests.exceptions.RequestException(f"OpenAI API error: {str(e)}")
    
    def get_dimensions(self) -> int:
        """Return the embedding dimensions for OpenAI."""
        return self.dimensions
    
    def get_model_name(self) -> str:
        """Return the OpenAI model name."""
        return self.model
    
    def get_weaviate_vectorizer_config(self) -> Optional[Dict[str, Any]]:
        """
        Return Weaviate vectorizer configuration for OpenAI.
        
        Returns:
            Dictionary with OpenAI vectorizer configuration
        """
        from weaviate.classes.config import Configure
        
        return Configure.Vectorizer.text2vec_openai(
            model=self.model,
            dimensions=self.dimensions
        )
    
    def supports_batch_api(self) -> bool:
        """Whether OpenAI supports batch processing."""
        return self.supports_batch
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for OpenAI models.
        
        Args:
            text: Input text to estimate
            
        Returns:
            Estimated number of tokens
        """
        # Try using tiktoken for accurate estimation
        try:
            import tiktoken
            
            # Get encoding for the model
            if "text-embedding-3" in self.model:
                encoding = tiktoken.get_encoding("cl100k_base")
            else:
                encoding = tiktoken.get_encoding("cl100k_base")  # Default
            
            return len(encoding.encode(text))
            
        except ImportError:
            # Fallback estimation: ~4 chars per token for English
            return max(1, len(text) // 4)
    
    def get_rate_limit_config(self) -> Dict[str, Any]:
        """Get OpenAI rate limiting configuration."""
        return {
            'max_requests_per_minute': self.max_requests_per_minute,
            'max_tokens_per_minute': self.max_tokens_per_minute,
            'max_tokens_per_day': self.max_tokens_per_day,
            'rate_limit_delay': 0.0,  # Dynamic based on limits
            'batch_size': self.batch_size
        }
    
    def _enforce_rate_limits(self, num_texts: int) -> None:
        """
        Enforce per-minute rate limits.
        
        Args:
            num_texts: Number of texts being processed
        """
        current_time = time.time()
        
        # Reset counters if new minute
        if current_time - self.minute_start >= 60:
            self.tokens_this_minute = 0
            self.requests_this_minute = 0
            self.minute_start = current_time
            return
        
        # Estimate tokens for this request
        estimated_tokens = num_texts * 100  # Conservative estimate
        
        # Check if this request would exceed limits
        if (self.tokens_this_minute + estimated_tokens >= self.max_tokens_per_minute or 
            self.requests_this_minute + 1 >= self.max_requests_per_minute):
            
            # Sleep until next minute
            sleep_time = 60 - (current_time - self.minute_start)
            if sleep_time > 0:
                logger.debug(f"Rate limit reached, sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
                
                # Reset counters after sleep
                self.tokens_this_minute = 0
                self.requests_this_minute = 0
                self.minute_start = time.time()
    
    def _check_daily_rate_limit(self) -> bool:
        """
        Check if daily token limit has been reached.
        
        Returns:
            True if under limit, False if limit reached
        """
        current_time = time.time()
        
        # Reset daily counter if new day
        if current_time - self.day_start >= 86400:  # 24 hours
            self.tokens_today = 0
            self.day_start = current_time
            logger.info("Daily token usage reset for new day")
        
        return self.tokens_today < self.max_tokens_per_day
    
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
            
            logger.debug(f"OpenAI Rate Limits - Tokens: {self.api_rate_limit_info['remaining_tokens']}/{self.api_rate_limit_info['limit_tokens']}, "
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