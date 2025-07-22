"""
Base Embedding Provider Interface

This module defines the abstract base class for all embedding providers in the
entity resolution pipeline. All providers must implement this interface to ensure
consistent behavior and compatibility.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the provider with configuration.
        
        Args:
            config: Provider-specific configuration dictionary
        """
        pass
    
    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> Tuple[List[np.ndarray], List[int]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of strings to embed
            
        Returns:
            Tuple of (embeddings, token_counts)
            - embeddings: List of numpy arrays, one per input text
            - token_counts: List of integers representing token usage per text
        """
        pass
    
    @abstractmethod
    def get_dimensions(self) -> int:
        """
        Return the embedding dimensions for this provider.
        
        Returns:
            Integer representing the embedding vector size
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """
        Return the model name/identifier.
        
        Returns:
            String identifying the specific model being used
        """
        pass
    
    @abstractmethod
    def get_weaviate_vectorizer_config(self) -> Optional[Union[Dict[str, Any], List[Any]]]:
        """
        Return Weaviate vectorizer configuration if applicable.
        
        Returns:
            Dictionary with Weaviate vectorizer config, list of named vector configs,
            or None for custom/local models
        """
        pass
    
    @abstractmethod
    def supports_batch_api(self) -> bool:
        """
        Whether this provider supports batch processing.
        
        Returns:
            Boolean indicating if batch API is available
        """
        pass
    
    @abstractmethod
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for rate limiting.
        
        Args:
            text: Input text to estimate
            
        Returns:
            Estimated number of tokens
        """
        pass
    
    # Common utility methods that can be inherited by all providers
    
    def validate_texts(self, texts: List[str]) -> List[str]:
        """
        Validate and clean input texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of validated texts
            
        Raises:
            ValueError: If texts list is empty or contains invalid entries
        """
        if not texts:
            raise ValueError("Input texts list cannot be empty")
        
        validated = []
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise ValueError(f"Text at index {i} must be a string, got {type(text)}")
            if not text.strip():
                raise ValueError(f"Text at index {i} cannot be empty or whitespace-only")
            validated.append(text.strip())
        
        return validated
    
    def chunk_texts(self, texts: List[str], chunk_size: int) -> List[List[str]]:
        """
        Split texts into chunks for batch processing.
        
        Args:
            texts: List of texts to chunk
            chunk_size: Maximum size of each chunk
            
        Returns:
            List of text chunks
        """
        if chunk_size <= 0:
            raise ValueError("Chunk size must be positive")
        
        chunks = []
        for i in range(0, len(texts), chunk_size):
            chunks.append(texts[i:i + chunk_size])
        
        return chunks
    
    def get_rate_limit_config(self) -> Dict[str, Any]:
        """
        Get rate limiting configuration for this provider.
        
        Returns:
            Dictionary with rate limiting parameters
        """
        # Default implementation - providers can override
        return {
            'max_requests_per_minute': None,
            'max_tokens_per_minute': None,
            'max_tokens_per_day': None,
            'rate_limit_delay': 0.0
        }