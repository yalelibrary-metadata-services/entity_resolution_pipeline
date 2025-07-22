"""
Embedding Providers Module

This module provides a pluggable architecture for different embedding providers
in the entity resolution pipeline, supporting OpenAI, Mistral, HuggingFace, and others.
"""

from .base import EmbeddingProvider
from .factory import EmbeddingProviderFactory

# Provider imports (lazy loaded by factory)
__all__ = [
    'EmbeddingProvider',
    'EmbeddingProviderFactory',
]