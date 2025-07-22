"""
HuggingFace Embedding Provider

This module implements the HuggingFace embedding provider for the entity resolution pipeline.
It supports local sentence-transformers models and HuggingFace Hub models.
"""

import os
import logging
import threading
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import requests

from .base import EmbeddingProvider

logger = logging.getLogger(__name__)


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """HuggingFace embedding provider implementation for local models."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize HuggingFace provider with configuration.
        
        Args:
            config: Provider configuration dictionary
        """
        self.model = config.get('model', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        self.dimensions = config.get('dimensions', 384)
        self.device = config.get('device', 'cpu')  # 'cuda' or 'cpu'
        
        # Local model settings
        self.batch_size = config.get('batch_size', 256)
        self.max_length = config.get('max_length', 512)
        self.normalize_embeddings = config.get('normalize_embeddings', True)
        
        # No API rate limits for local models
        self.supports_batch = config.get('supports_batch', False)
        
        # Optional HuggingFace token for private models
        self.hf_token = os.environ.get('HF_TOKEN')
        
        # Thread safety lock for model access
        self._model_lock = threading.Lock()
        
        # Initialize the model
        self._load_model()
        
        logger.info(f"Initialized HuggingFace provider with model {self.model} ({self.dimensions}D) on {self.device}")
    
    def _load_model(self):
        """Load the HuggingFace model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Load model with optional authentication token
            kwargs = {}
            if self.hf_token:
                kwargs['use_auth_token'] = self.hf_token
            
            self.model_instance = SentenceTransformer(
                self.model,
                device=self.device,
                **kwargs
            )
            
            # Verify dimensions match configuration
            test_embedding = self.model_instance.encode(["test"], convert_to_numpy=True)
            actual_dimensions = test_embedding.shape[1]
            
            if actual_dimensions != self.dimensions:
                logger.warning(f"Model dimensions ({actual_dimensions}) don't match config ({self.dimensions}). "
                              f"Updating config to match model.")
                self.dimensions = actual_dimensions
            
        except ImportError as e:
            raise ImportError("sentence-transformers not available. Install with: pip install sentence-transformers") from e
        except Exception as e:
            logger.error(f"Error loading HuggingFace model {self.model}: {str(e)}")
            raise RuntimeError(f"Failed to load model {self.model}: {str(e)}")
    
    def get_embeddings(self, texts: List[str]) -> Tuple[List[np.ndarray], List[int]]:
        """
        Generate embeddings for a list of texts using HuggingFace model.
        
        Args:
            texts: List of strings to embed
            
        Returns:
            Tuple of (embeddings, token_counts)
        """
        # Validate inputs
        validated_texts = self.validate_texts(texts)
        
        try:
            # Thread-safe model access
            with self._model_lock:
                # Generate embeddings using sentence-transformers
                embeddings_array = self.model_instance.encode(
                    validated_texts,
                    batch_size=self.batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=self.normalize_embeddings,
                    show_progress_bar=False  # Disable progress bar for cleaner logs
                )
            
            # Convert to list of numpy arrays
            embeddings = [np.array(emb, dtype=np.float32) for emb in embeddings_array]
            
            # Estimate token counts (sentence-transformers doesn't provide exact counts)
            token_counts = [self.estimate_tokens(text) for text in validated_texts]
            
            logger.debug(f"Generated {len(embeddings)} HuggingFace embeddings")
            
            return embeddings, token_counts
            
        except Exception as e:
            logger.error(f"Error generating embeddings via HuggingFace: {str(e)}")
            raise RuntimeError(f"HuggingFace model error: {str(e)}")
    
    def get_dimensions(self) -> int:
        """Return the embedding dimensions for HuggingFace model."""
        return self.dimensions
    
    def get_model_name(self) -> str:
        """Return the HuggingFace model name."""
        return self.model
    
    def get_weaviate_vectorizer_config(self) -> Optional[Dict[str, Any]]:
        """
        Return Weaviate vectorizer configuration for HuggingFace.
        
        Returns:
            None since we're using local models (custom vectors)
        """
        # Local models don't integrate with Weaviate vectorizers
        # We'll provide custom vectors instead
        return None
    
    def supports_batch_api(self) -> bool:
        """Whether HuggingFace supports batch processing."""
        return self.supports_batch
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for HuggingFace models.
        
        Args:
            text: Input text to estimate
            
        Returns:
            Estimated number of tokens
        """
        try:
            # Try to get tokenizer for more accurate estimation (thread-safe)
            with self._model_lock:
                tokenizer = self.model_instance.tokenizer
                if tokenizer:
                    tokens = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=self.max_length)
                    return len(tokens)
        except:
            pass
        
        # Fallback: estimate based on whitespace and punctuation
        # Most BERT-like models use subword tokenization
        words = text.split()
        # Estimate 1.3 tokens per word on average for subword tokenization
        return max(1, int(len(words) * 1.3))
    
    def get_rate_limit_config(self) -> Dict[str, Any]:
        """Get HuggingFace rate limiting configuration."""
        return {
            'max_requests_per_minute': None,  # No API limits for local models
            'max_tokens_per_minute': None,
            'max_tokens_per_day': None,
            'rate_limit_delay': 0.0,  # No delay needed
            'batch_size': self.batch_size
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        try:
            info = {
                'model_name': self.model,
                'dimensions': self.dimensions,
                'device': self.device,
                'max_length': self.max_length,
                'normalize_embeddings': self.normalize_embeddings
            }
            
            # Add tokenizer info if available
            if hasattr(self.model_instance, 'tokenizer') and self.model_instance.tokenizer:
                tokenizer = self.model_instance.tokenizer
                info['tokenizer_vocab_size'] = getattr(tokenizer, 'vocab_size', 'unknown')
                info['tokenizer_model_max_length'] = getattr(tokenizer, 'model_max_length', 'unknown')
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return {'error': str(e)}
    
    def warm_up(self, sample_texts: Optional[List[str]] = None) -> None:
        """
        Warm up the model with sample texts to improve first-inference performance.
        
        Args:
            sample_texts: Optional list of sample texts for warmup
        """
        if sample_texts is None:
            sample_texts = ["This is a sample text for model warmup."]
        
        try:
            logger.info("Warming up HuggingFace model...")
            self.get_embeddings(sample_texts)
            logger.info("Model warmup completed")
        except Exception as e:
            logger.warning(f"Model warmup failed: {str(e)}")
    
    def set_device(self, device: str) -> None:
        """
        Change the device for the model.
        
        Args:
            device: Target device ('cuda' or 'cpu')
        """
        try:
            with self._model_lock:
                self.model_instance.to(device)
                self.device = device
            logger.info(f"Moved model to device: {device}")
        except Exception as e:
            logger.error(f"Failed to move model to device {device}: {str(e)}")
            raise RuntimeError(f"Device change failed: {str(e)}")