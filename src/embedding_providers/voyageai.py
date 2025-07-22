"""
Voyage AI Embedding Provider

This module implements the Voyage AI embedding provider for the entity resolution pipeline.
It supports all Voyage AI embedding models via their API, with a focus on voyage-3-large.
"""

import os
import logging
import time
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import requests

from .base import EmbeddingProvider

logger = logging.getLogger(__name__)


class VoyageAIEmbeddingProvider(EmbeddingProvider):
    """Voyage AI embedding provider implementation."""
    
    # Model dimensions mapping - voyage-3-large supports multiple dimensions
    MODEL_DIMENSIONS = {
        'voyage-3-large': 2048,  # Default to maximum dimensions
        'voyage-3.5': 1024,
        'voyage-3.5-lite': 1024,
        'voyage-3': 1024,
        'voyage-3-lite': 512,
        'voyage-large-2': 1536,
        'voyage-code-2': 1536,
        'voyage-2': 1024,
        'voyage-law-2': 1024,
        'voyage-finance-2': 1024,
        'voyage-multilingual-2': 1024,
        'voyage-large-2-instruct': 1024,
        'voyage-code-3': 1024,
    }
    
    # Token limits per request
    MODEL_TOKEN_LIMITS = {
        'voyage-3-large': 32000,      # 32K token context length
        'voyage-3.5-lite': 1000000,   # 1M tokens
        'voyage-3.5': 320000,         # 320K tokens
        'voyage-2': 320000,           # 320K tokens
        'voyage-code-3': 120000,      # 120K tokens
        'voyage-large-2-instruct': 120000,  # 120K tokens
        'voyage-finance-2': 120000,   # 120K tokens
        'voyage-multilingual-2': 120000,  # 120K tokens
        'voyage-law-2': 120000,       # 120K tokens
    }
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Voyage AI provider with configuration.
        
        Args:
            config: Provider configuration dictionary
        """
        self.model = config.get('model', 'voyage-3-large')
        self.api_key = os.environ.get(config.get('api_key_env', 'VOYAGEAI_APIKEY'))
        
        if not self.api_key:
            raise ValueError("Voyage AI API key not found in environment variable: VOYAGEAI_APIKEY")
        
        # Handle dimensions - voyage-3-large supports multiple options
        if self.model == 'voyage-3-large':
            # Support custom dimensions for voyage-3-large (2048, 1024, 512, 256)
            self.dimensions = config.get('dimensions', 2048)
            if self.dimensions not in [2048, 1024, 512, 256]:
                logger.warning(f"voyage-3-large dimensions {self.dimensions} not in supported set [2048, 1024, 512, 256], defaulting to 2048")
                self.dimensions = 2048
        elif self.model in self.MODEL_DIMENSIONS:
            self.dimensions = self.MODEL_DIMENSIONS[self.model]
        else:
            self.dimensions = config.get('dimensions', 1024)
            logger.warning(f"Unknown model {self.model}, using configured dimensions: {self.dimensions}")
        
        # API configuration
        self.api_base = config.get('api_base', 'https://api.voyageai.com')
        
        # Optional parameters for voyage-3-large
        self.output_dimension = config.get('output_dimension', self.dimensions)
        self.output_dtype = config.get('output_dtype', 'float')  # Options: float, int8, binary
        
        # Batch configuration
        self.batch_size = min(config.get('batch_size', 128), 128)  # Max 128 per request
        self.supports_batch = config.get('supports_batch', False)
        
        # Rate limiting - Default to Tier 1 limits
        self.max_requests_per_minute = config.get('max_requests_per_minute', 2000)
        self.max_tokens_per_minute = config.get('max_tokens_per_minute', 8000000)
        self.rate_limit_delay = config.get('rate_limit_delay', 0.03)  # 30ms between requests
        
        # Token limit for this model
        self.max_tokens_per_request = self.MODEL_TOKEN_LIMITS.get(self.model, 120000)
        
        # Track rate limiting
        self.tokens_this_minute = 0
        self.requests_this_minute = 0
        self.minute_start = time.time()
        
        logger.info(f"Initialized Voyage AI provider with model {self.model} ({self.dimensions}D, dtype={self.output_dtype})")
    
    def get_embeddings(self, texts: List[str]) -> Tuple[List[np.ndarray], List[int]]:
        """
        Generate embeddings for a list of texts using Voyage AI API.
        
        Args:
            texts: List of strings to embed
            
        Returns:
            Tuple of (embeddings, token_counts)
        """
        # Validate inputs
        validated_texts = self.validate_texts(texts)
        
        # Check rate limits
        self._check_and_wait_for_rate_limits(len(validated_texts))
        
        try:
            # Prepare request
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            data = {
                'input': validated_texts,
                'model': self.model
            }
            
            # Add voyage-3-large specific parameters
            if self.model == 'voyage-3-large':
                if self.output_dimension != self.dimensions:
                    data['output_dimension'] = self.output_dimension
                if self.output_dtype != 'float':
                    data['output_dtype'] = self.output_dtype
            
            # Make API request
            response = requests.post(
                f'{self.api_base}/v1/embeddings',
                headers=headers,
                json=data,
                timeout=60
            )
            
            # Handle response
            if response.status_code == 200:
                result = response.json()
                
                # Extract embeddings
                embeddings_data = result.get('data', [])
                embeddings = []
                
                for item in embeddings_data:
                    embedding = item.get('embedding', [])
                    
                    # Handle different output dtypes
                    if self.output_dtype == 'int8':
                        embeddings.append(np.array(embedding, dtype=np.int8))
                    elif self.output_dtype == 'binary':
                        embeddings.append(np.array(embedding, dtype=np.uint8))
                    else:
                        embeddings.append(np.array(embedding, dtype=np.float32))
                
                # Extract usage information
                usage = result.get('usage', {})
                total_tokens = usage.get('total_tokens', 0)
                
                # Update rate limit tracking
                self._update_rate_limits(total_tokens)
                
                # Estimate tokens per text
                tokens_per_text = total_tokens // len(validated_texts) if validated_texts else 0
                token_counts = [tokens_per_text] * len(validated_texts)
                
                logger.debug(f"Generated {len(embeddings)} Voyage AI embeddings using {total_tokens} tokens")
                
                return embeddings, token_counts
                
            else:
                error_msg = f"Voyage AI API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Voyage AI API: {str(e)}")
            raise RuntimeError(f"Voyage AI API request failed: {str(e)}")
        except Exception as e:
            logger.error(f"Error generating embeddings via Voyage AI: {str(e)}")
            raise RuntimeError(f"Voyage AI embedding error: {str(e)}")
    
    def _check_and_wait_for_rate_limits(self, num_texts: int) -> None:
        """
        Check rate limits and wait if necessary.
        
        Args:
            num_texts: Number of texts to process
        """
        current_time = time.time()
        
        # Reset counters if new minute
        if current_time - self.minute_start >= 60:
            self.tokens_this_minute = 0
            self.requests_this_minute = 0
            self.minute_start = current_time
        
        # Estimate tokens (conservative estimate)
        estimated_tokens = num_texts * 100
        
        # Check if we would exceed rate limits
        if (self.requests_this_minute + 1 >= self.max_requests_per_minute or
            self.tokens_this_minute + estimated_tokens >= self.max_tokens_per_minute):
            # Wait until next minute
            sleep_time = 60 - (current_time - self.minute_start)
            if sleep_time > 0:
                logger.info(f"Rate limit approaching, waiting {sleep_time:.1f}s")
                time.sleep(sleep_time)
                # Reset after sleep
                self.tokens_this_minute = 0
                self.requests_this_minute = 0
                self.minute_start = time.time()
        
        # Add small delay between requests
        time.sleep(self.rate_limit_delay)
    
    def _update_rate_limits(self, tokens_used: int) -> None:
        """
        Update rate limit tracking.
        
        Args:
            tokens_used: Number of tokens used in the request
        """
        self.requests_this_minute += 1
        self.tokens_this_minute += tokens_used
    
    def get_dimensions(self) -> int:
        """Return the embedding dimensions for Voyage AI model."""
        return self.dimensions
    
    def get_model_name(self) -> str:
        """Return the Voyage AI model name."""
        return self.model
    
    def get_weaviate_vectorizer_config(self) -> Optional[Dict[str, Any]]:
        """
        Return Weaviate vectorizer configuration for Voyage AI.
        
        Returns:
            Configuration for text2vec-voyageai module
        """
        # Voyage AI doesn't have a built-in Weaviate vectorizer module
        # We'll use custom vectors instead
        return None
    
    def supports_batch_api(self) -> bool:
        """Whether Voyage AI supports batch processing."""
        return self.supports_batch
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for Voyage AI models.
        
        Args:
            text: Input text to estimate
            
        Returns:
            Estimated number of tokens
        """
        # Voyage AI uses similar tokenization to OpenAI
        # Estimate ~4 characters per token on average
        return max(1, len(text) // 4)
    
    def get_rate_limit_config(self) -> Dict[str, Any]:
        """Get Voyage AI rate limiting configuration."""
        return {
            'max_requests_per_minute': self.max_requests_per_minute,
            'max_tokens_per_minute': self.max_tokens_per_minute,
            'max_tokens_per_day': None,  # No daily limit specified
            'rate_limit_delay': self.rate_limit_delay,
            'batch_size': self.batch_size,
            'max_tokens_per_request': self.max_tokens_per_request
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the Voyage AI model.
        
        Returns:
            Dictionary with model information
        """
        info = {
            'provider': 'voyageai',
            'model_name': self.model,
            'dimensions': self.dimensions,
            'output_dimension': self.output_dimension,
            'output_dtype': self.output_dtype,
            'max_tokens_per_request': self.max_tokens_per_request,
            'rate_limits': {
                'tier': 'Tier 1 (default)',
                'max_rpm': self.max_requests_per_minute,
                'max_tpm': self.max_tokens_per_minute
            }
        }
        
        # Add voyage-3-large specific info
        if self.model == 'voyage-3-large':
            info['supported_dimensions'] = [2048, 1024, 512, 256]
            info['supported_dtypes'] = ['float', 'int8', 'binary']
            info['features'] = ['Matryoshka learning', 'Quantization-aware training', 'Multilingual']
        
        return info