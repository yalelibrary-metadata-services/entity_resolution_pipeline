"""
Embedding Provider Factory

This module provides a factory for creating embedding provider instances based on
configuration. It supports dynamic loading of providers and validation of configurations.
"""

import importlib
import logging
from typing import Dict, Any, List

from .base import EmbeddingProvider

logger = logging.getLogger(__name__)


class EmbeddingProviderFactory:
    """Factory for creating embedding providers."""
    
    # Registry of available providers
    _providers = {
        'openai': 'OpenAIEmbeddingProvider',
        'mistral': 'MistralEmbeddingProvider', 
        'huggingface': 'HuggingFaceEmbeddingProvider',
        'jina': 'JinaEmbeddingProvider',
        'cohere': 'CohereEmbeddingProvider',
        'jina_colbert': 'JinaColBERTProvider',
        'voyageai': 'VoyageAIEmbeddingProvider',
    }
    
    @classmethod
    def create_provider(cls, provider_type: str, config: Dict[str, Any]) -> EmbeddingProvider:
        """
        Create an embedding provider instance.
        
        Args:
            provider_type: Type of provider to create (e.g., 'openai', 'mistral')
            config: Provider-specific configuration dictionary
            
        Returns:
            Initialized embedding provider instance
            
        Raises:
            ValueError: If provider type is unknown
            ImportError: If provider module cannot be imported
            Exception: If provider initialization fails
        """
        if provider_type not in cls._providers:
            available = ', '.join(cls._providers.keys())
            raise ValueError(f"Unknown provider type: '{provider_type}'. Available providers: {available}")
        
        provider_class_name = cls._providers[provider_type]
        
        try:
            # Dynamic import of the provider module
            module = importlib.import_module('src.embedding_providers')
            
            # Import the specific provider class
            if provider_type == 'openai':
                from .openai import OpenAIEmbeddingProvider
                provider_class = OpenAIEmbeddingProvider
            elif provider_type == 'mistral':
                from .mistral import MistralEmbeddingProvider
                provider_class = MistralEmbeddingProvider
            elif provider_type == 'huggingface':
                from .huggingface import HuggingFaceEmbeddingProvider
                provider_class = HuggingFaceEmbeddingProvider
            elif provider_type == 'jina':
                from .jina import JinaEmbeddingProvider
                provider_class = JinaEmbeddingProvider
            elif provider_type == 'cohere':
                from .cohere import CohereEmbeddingProvider
                provider_class = CohereEmbeddingProvider
            elif provider_type == 'jina_colbert':
                from .jina_colbert import JinaColBERTProvider
                provider_class = JinaColBERTProvider
            elif provider_type == 'voyageai':
                from .voyageai import VoyageAIEmbeddingProvider
                provider_class = VoyageAIEmbeddingProvider
            else:
                # Fallback to dynamic lookup
                provider_class = getattr(module, provider_class_name)
            
            logger.info(f"Creating {provider_type} embedding provider")
            
            # Validate configuration before creating provider
            cls._validate_config(provider_type, config)
            
            # Create and return provider instance
            return provider_class(config)
            
        except ImportError as e:
            logger.error(f"Failed to import {provider_type} provider: {str(e)}")
            raise ImportError(f"Could not import provider '{provider_type}': {str(e)}")
        except Exception as e:
            logger.error(f"Failed to create {provider_type} provider: {str(e)}")
            raise Exception(f"Failed to initialize {provider_type} provider: {str(e)}")
    
    @classmethod
    def _validate_config(cls, provider_type: str, config: Dict[str, Any]) -> None:
        """
        Validate provider configuration.
        
        Args:
            provider_type: Type of provider
            config: Configuration to validate
            
        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(config, dict):
            raise ValueError(f"Configuration for {provider_type} must be a dictionary")
        
        # Common required fields
        required_fields = ['model', 'dimensions']
        
        # Provider-specific validation
        if provider_type == 'openai':
            required_fields.extend(['api_key_env'])
        elif provider_type == 'mistral':
            required_fields.extend(['api_key_env'])
        elif provider_type == 'jina':
            required_fields.extend(['api_key_env'])
        elif provider_type == 'cohere':
            required_fields.extend(['api_key_env'])
        elif provider_type == 'jina_colbert':
            required_fields.extend(['api_key_env'])
        elif provider_type == 'voyageai':
            required_fields.extend(['api_key_env'])
        elif provider_type == 'huggingface':
            # HuggingFace may not need API key for local models
            pass
        
        # Check required fields
        missing_fields = [field for field in required_fields if field not in config]
        if missing_fields:
            raise ValueError(f"Missing required configuration fields for {provider_type}: {missing_fields}")
        
        # Validate dimensions
        dimensions = config.get('dimensions')
        if not isinstance(dimensions, int) or dimensions <= 0:
            raise ValueError(f"Dimensions must be a positive integer, got: {dimensions}")
        
        logger.debug(f"Configuration validated for {provider_type} provider")
    
    @classmethod
    def get_available_providers(cls) -> List[str]:
        """
        Get list of available provider types.
        
        Returns:
            List of provider type strings
        """
        return list(cls._providers.keys())
    
    @classmethod
    def register_provider(cls, provider_type: str, provider_class_name: str) -> None:
        """
        Register a new provider type.
        
        Args:
            provider_type: Unique identifier for the provider
            provider_class_name: Name of the provider class
        """
        if provider_type in cls._providers:
            logger.warning(f"Overriding existing provider registration for '{provider_type}'")
        
        cls._providers[provider_type] = provider_class_name
        logger.info(f"Registered provider '{provider_type}' -> {provider_class_name}")
    
    @classmethod
    def create_from_full_config(cls, embedding_config: Dict[str, Any]) -> EmbeddingProvider:
        """
        Create provider from full embedding configuration section.
        
        Args:
            embedding_config: Full embedding configuration with provider selection
            
        Returns:
            Initialized embedding provider
            
        Raises:
            ValueError: If configuration is invalid
        """
        if 'provider' not in embedding_config:
            raise ValueError("Configuration must specify 'provider' field")
        
        if 'providers' not in embedding_config:
            raise ValueError("Configuration must include 'providers' section")
        
        provider_type = embedding_config['provider']
        providers_config = embedding_config['providers']
        
        if provider_type not in providers_config:
            raise ValueError(f"No configuration found for provider '{provider_type}'")
        
        provider_config = providers_config[provider_type]
        
        return cls.create_provider(provider_type, provider_config)