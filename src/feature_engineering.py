import hashlib
import json
import logging
import math
import os
import re
import threading
import time
import unicodedata
from collections import defaultdict, Counter
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast
import heapq

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from weaviate.classes.query import Filter

from src.birth_death_regexes import BirthDeathYearExtractor
from src.checkpoint_manager import get_checkpoint_manager
from src.taxonomy_feature import TaxonomyDissimilarity, create_taxonomy_feature

# Import determinism monitor if available
_DETERMINISM_MONITOR_AVAILABLE = False
try:
    from src.determinism_monitor import get_determinism_monitor, setup_deterministic_behavior
    _DETERMINISM_MONITOR_AVAILABLE = True
except ImportError:
    # Define fallback function for setting random seed
    def setup_deterministic_behavior(seed=42, context=None):
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        logging.info(f"Initialized deterministic behavior with seed {seed} in context '{context or 'global'}'")

# Configure logging
logger = logging.getLogger(__name__)

# Version information to help with debugging 
_VERSION = "3.5.2"  # Updated version with birth_death_match fix
_BUILD_DATE = "20250331"

class VersionedCache:
    """
    Cache that supports versioning for deterministic similarity feature calculation.
    Ensures that results are consistent even when run in different environments.
    """
    def __init__(self, name, max_size=100000):
        """
        Initialize versioned cache.
        
        Args:
            name: Cache identifier string
            max_size: Maximum size of the cache (items)
        """
        self.name = name
        self.max_size = max_size
        self.data = {}
        self.cache_version = str(int(time.time()))
        
        # Tracking metrics
        self.hits = 0
        self.misses = 0
        self.lock = threading.RLock()
        
    def __contains__(self, key):
        """
        Check if key exists in cache.
        
        Args:
            key: Cache key
            
        Returns:
            bool: True if key exists in cache
        """
        return key in self.data
        
    def __getitem__(self, key):
        """
        Get item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Any: Cached value
            
        Raises:
            KeyError: If key not in cache
        """
        with self.lock:
            if key in self.data:
                self.hits += 1
                return self.data[key]
            self.misses += 1
            raise KeyError(f"Key {key} not found in cache {self.name}")
        
    def __setitem__(self, key, value):
        """
        Set item in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        with self.lock:
            # Check if cache is full
            if len(self.data) >= self.max_size and key not in self.data:
                # Remove a random key to make space
                remove_key = next(iter(self.data))
                del self.data[remove_key]
            self.data[key] = value
            
    def get(self, key, default=None):
        """
        Get item from cache with default fallback.
        
        Args:
            key: Cache key
            default: Default value if key not found
            
        Returns:
            Any: Cached value or default
        """
        with self.lock:
            try:
                return self[key]
            except KeyError:
                return default
                
    def clear(self):
        """Clear the cache."""
        with self.lock:
            self.data.clear()
            self.hits = 0
            self.misses = 0
    
    def get_stats(self):
        """
        Get cache statistics.
        
        Returns:
            dict: Cache statistics
        """
        with self.lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0
            return {
                "name": self.name,
                "size": len(self.data),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "version": self.cache_version
            }


class FeatureEngineering:
    """
    Feature engineering module for entity resolution.
    
    This class handles the extraction and calculation of features for entity matching,
    including string similarity metrics, vector-based features, and composite features.
    """
    
    def __init__(self, config: Dict[str, Any], weaviate_client: Any, hash_lookup: Dict[str, Dict[str, str]]):
        """
        Initialize the feature engineering module.
        
        Args:
            config: Configuration dictionary with feature parameters
            weaviate_client: Initialized Weaviate client for querying vectors
            hash_lookup: Dictionary mapping personId to field hashes
        """
        # Set up deterministic behavior with enhanced logging
        seed = config.get('random_seed', 42)
        setup_deterministic_behavior(seed)
        self.random_seed = seed
        logger.info(f"Initialized FeatureEngineering with random seed: {seed}")
        
        # Initialize determinism monitoring if available
        self.determinism_monitor = None
        if _DETERMINISM_MONITOR_AVAILABLE and config.get("enable_determinism_monitoring", False):
            self.determinism_monitor = get_determinism_monitor()
            logger.info("Initialized determinism monitoring")
        
        self.config = config
        self.weaviate_client = weaviate_client
        self.hash_lookup = hash_lookup
        
        # Initialize caches
        self.string_cache = {}
        self.vector_cache = {}
        self.similarity_cache = {}
        self.disable_caching = self.config.get("disable_feature_caching", False)
        
        # Synchronize cache versions using a class-level attribute for consistency
        # This ensures all instances created during the same run share the same cache version
        if not hasattr(FeatureEngineering, '_global_cache_version'):
            FeatureEngineering._global_cache_version = str(int(time.time()))
        self.cache_version = FeatureEngineering._global_cache_version

        # CRITICAL FIX: Use a single shared class-level cache for similarity to ensure
        # all instances access the same cache - this fixes issues with multithreaded access
        if not hasattr(FeatureEngineering, '_shared_similarity_cache'):
            FeatureEngineering._shared_similarity_cache = {}
        self.similarity_cache = FeatureEngineering._shared_similarity_cache
            
        # Initialize other caches as versioned caches
        self.vector_cache = VersionedCache("vector_cache")
        self.string_cache = VersionedCache("string_cache")
        
        # Strictly enforce dictionary-based caching for key features
        self.disable_caching = False  # Never disable caching for critical features
        
        # Global cache lock for thread-safe access to shared cache
        if not hasattr(FeatureEngineering, '_shared_cache_lock'):
            FeatureEngineering._shared_cache_lock = threading.RLock()
        self.cache_lock = FeatureEngineering._shared_cache_lock
            
        logger.info(f"Initialized cache system with version: {self.cache_version}")
        
        self.birth_death_extractor = BirthDeathYearExtractor()
        self.substitution_mapping = {}  # Maps custom features to the features they replace
        self.component_registry = {}    # Stores feature functions for components of composite features
        self._debug_composite = {}      # Tracks which composite features have been debugged
        
        # Initialize taxonomy dissimilarity calculator if paths are provided
        self.taxonomy_dissimilarity = None
        if 'taxonomy_path' in config or 'classified_data_path' in config:
            try:
                self.taxonomy_dissimilarity = TaxonomyDissimilarity(config)
                logger.info("Initialized taxonomy dissimilarity feature")
            except Exception as e:
                logger.warning(f"Could not initialize taxonomy dissimilarity: {e}")
        
        # Track feature calculation issues for diagnostic purposes
        self.calculation_issues = []
        self.binary_features = ['person_low_levenshtein_indicator', 
                            'person_low_jaro_winkler_indicator', 
                            'person_low_cosine_indicator',
                            'birth_death_match']
        
        # Initialize synchronization primitives for thread safety
        self.registry_lock = RLock()
        
        # Initialize feature_names to ensure it's always available
        self.feature_names = []
        
        # Configure feature parameters from config
        self.batch_size = config.get("feature_batch_size", 1000)
        self.num_workers = config.get("feature_workers", 4)
        
        # Enable debug mode for identifying issues with custom features
        self.debug_custom_features = config.get("debug_custom_features", False)
        # Load feature configuration
        feature_config = config.get("features", {})
        similarity_config = feature_config.get("similarity_metrics", {})
        self.use_binary_indicators = similarity_config.get("use_binary_indicators", True)
        self.include_both_metrics = similarity_config.get("include_both_metrics", False)
        # Get base enabled features from config
        base_enabled_features = feature_config.get("enabled", [
            'person_low_levenshtein_indicator',
            'person_title_squared',
            'composite_cosine', 
            'birth_death_match',
            'title_cosine_squared',
            'title_role_adjusted'
        ])
        
        # Filter enabled features based on similarity metric configuration
        self.enabled_features = []
        for feature in base_enabled_features:
            # Handle similarity metric features based on configuration
            is_binary_indicator = feature in ['person_low_levenshtein_indicator', 'person_low_jaro_winkler_indicator']
            is_direct_similarity = feature in ['person_levenshtein_similarity', 'person_jaro_winkler_similarity']
            
            # Include feature if it's not a similarity metric or if it matches the chosen metric type
            if (not is_binary_indicator and not is_direct_similarity) or \
            (is_binary_indicator and (self.use_binary_indicators or self.include_both_metrics)) or \
            (is_direct_similarity and (not self.use_binary_indicators or self.include_both_metrics)):
                self.enabled_features.append(feature)
        
        # Log the final feature selection with detailed information
        logger.info(f"Similarity metric configuration:")
        logger.info(f"  - Using {'binary indicators' if self.use_binary_indicators else 'direct similarities'} as primary metric")
        logger.info(f"  - {'Including' if self.include_both_metrics else 'Not including'} both metric types")
        
        # Log which specific similarity features are enabled
        binary_metrics = [f for f in self.enabled_features if f in ['person_low_levenshtein_indicator', 'person_low_jaro_winkler_indicator']]
        direct_metrics = [f for f in self.enabled_features if f in ['person_levenshtein_similarity', 'person_jaro_winkler_similarity']]
        logger.info(f"Enabled binary similarity metrics: {binary_metrics if binary_metrics else 'None'}")
        logger.info(f"Enabled direct similarity metrics: {direct_metrics if direct_metrics else 'None'}")
        
        # Log all enabled features for clarity
        logger.info(f"All enabled features: {self.enabled_features}")
        
        self.feature_params = feature_config.get("parameters", {})
        
        # Initialize feature registry with all available features
        all_feature_methods = {
            # Binary indicator features
            'person_low_levenshtein_indicator': self._calc_person_low_levenshtein_indicator,
            'person_low_jaro_winkler_indicator': self._calc_person_low_jaro_winkler_indicator,
            'person_low_cosine_indicator': self._calc_person_low_cosine_indicator,
            
            # Raw similarity features
            'person_levenshtein_similarity': self._calc_person_levenshtein_similarity,
            'person_jaro_winkler_similarity': self._calc_person_jaro_winkler_similarity,
            
            # Other features
            'person_cosine': self._calc_person_cosine,
            'person_title_squared': self._calc_person_title_squared,
            'composite_cosine': self._calc_composite_cosine, 
            'composite_cosine_squared': self._calc_composite_cosine_squared,
            'birth_death_match': self._calc_birth_death_match,
            'title_cosine_squared': self._calc_title_cosine_squared,
            'title_role_adjusted': self._calc_title_role_adjusted,
            'person_title_adjusted_squared': self._calc_person_title_adjusted_squared,
            'roles_cosine': self._calc_role_cosine,
            'roles_cosine_weighted': self._calc_roles_cosine_weighted,
            'marcKey_cosine': self._calc_marcKey_cosine,
            'marcKey_title_squared': self._calc_marcKey_title_squared,
            'person_role_squared': self._calc_person_role_squared,
            'taxonomy_dissimilarity': self._calc_taxonomy_dissimilarity,
            'has_related_work': self._calc_has_related_work,
            
            # Non-linear feature transformations
            'confidence_amplifier_exponential': self._calc_confidence_amplifier_exponential,
            'confidence_amplifier_polynomial': self._calc_confidence_amplifier_polynomial,
            'person_title_interaction_cubic': self._calc_person_title_interaction_cubic,
            'marc_roles_polynomial_interaction': self._calc_marc_roles_polynomial_interaction,
            'confidence_cascade_indicator': self._calc_confidence_cascade_indicator,
            'evidence_strength_weighted': self._calc_evidence_strength_weighted,
            'harmonic_mean_primary_features': self._calc_harmonic_mean_primary_features,
            'geometric_confidence_scaling': self._calc_geometric_confidence_scaling
        }
        
        # Filter to only enabled features
        self.feature_registry = {}
        for feature_name in self.enabled_features:
            if feature_name in all_feature_methods:
                self.feature_registry[feature_name] = all_feature_methods[feature_name]
            else:
                logger.warning(f"Feature '{feature_name}' specified in config but not implemented")
                
        # Load role configuration for title_role_adjusted feature
        self.role_weights = config.get("role_weights", {})
        self.role_compatibility = config.get("role_compatibility", {})
        
        # Feature normalization
        self.scaler = MinMaxScaler()
        self.is_fitted = False
        
        # Track field to hash mapping
        self.field_hash_mapping = {}  # Will be populated as needed
        
        logger.info(f"Initialized FeatureEngineering with {len(self.feature_registry)} features")
    
    def substitute_features(self, substitution_mapping: Dict[str, str]) -> None:
        """
        Register feature substitutions for custom implementations.
        
        Args:
            substitution_mapping: Dictionary mapping feature name to feature it replaces
        """
        with self.registry_lock:
            self.substitution_mapping = substitution_mapping.copy()
            
            # Create reverse mapping for reporting purposes
            self.reverse_substitution = {v: k for k, v in substitution_mapping.items()}
            
            if substitution_mapping:
                logger.info(f"Registered feature substitutions: {substitution_mapping}")
    
    def register_component(self, feature_name: str, component_func) -> None:
        """
        Register a component feature calculator function.
        
        Args:
            feature_name: Name of the feature
            component_func: Function that calculates the feature
        """
        with self.registry_lock:
            self.component_registry[feature_name] = component_func
            logger.info(f"Registered component feature: {feature_name}")
    
    def register_feature(self, feature_name: str, feature_func) -> None:
        """
        Register a new feature calculation function.
        
        Args:
            feature_name: Name of the feature
            feature_func: Function that calculates the feature
        """
        with self.registry_lock:
            self.feature_registry[feature_name] = feature_func
            
            # Add to enabled features if not already there
            if feature_name not in self.enabled_features:
                self.enabled_features.append(feature_name)
            
            logger.info(f"Registered feature: {feature_name}")
    
    def get_available_features(self) -> List[str]:
        """
        Get list of all available features.
        
        Returns:
            List of feature names
        """
        # Start with enabled features list
        features = list(self.feature_registry.keys())
        
        # Append any substituted features
        for feature in self.substitution_mapping:
            if feature not in features:
                features.append(feature)
                
        return features
        
    def get_substitution_mapping(self) -> Dict[str, str]:
        """
        Get the current feature substitution mapping.
        
        Returns:
            Dictionary mapping custom feature names to original feature names
        """
        return self.substitution_mapping.copy()
    
    def _generate_transaction_id(self, left_id: str, right_id: str) -> str:
        """
        Generate a transaction ID for logging and debugging.
        
        Args:
            left_id: ID of first entity
            right_id: ID of second entity
            
        Returns:
            String transaction ID
        """
        thread_id = id(threading.current_thread())
        return f"{left_id[:6]}-{right_id[:6]}-{thread_id}"
        
    def _normalize_string(self, text: str) -> str:
        """
        Normalize a string for consistent matching.
        
        Args:
            text: String to normalize
            
        Returns:
            Normalized string
        """
        if text is None:
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Normalize Unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Remove punctuation and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def _normalize_role_category(self, role_value: str) -> str:
        """
        Normalize role values to canonical categories.
        
        Args:
            role_value: Raw role string from data
            
        Returns:
            Canonical role category
        """
        if not role_value or not role_value.strip():
            return "CreativeRole"  # Default fallback
            
        role_clean = role_value.strip()
        
        # Map to four canonical categories
        if role_clean == "Contributor":
            return "Contributor"
        elif role_clean == "Subject":
            return "Subject"
        elif role_clean == "Associated person":
            return "Associated person"
        else:
            # All other specific roles (Author, Editor, etc.) map to CreativeRole
            return "CreativeRole"

    def _get_string_value(self, entity_id: str, field: str, warn_if_missing: bool = True) -> str:
        """
        Get the original string value for an entity field.
        
        Args:
            entity_id: Entity ID
            field: Field name
            warn_if_missing: Whether to log warnings for missing fields (default True)
            
        Returns:
            Original string value or empty string if not found
        """
        # Check if entity_id is in hash lookup
        if entity_id not in self.hash_lookup:
            if warn_if_missing:
                logger.warning(f"Entity ID {entity_id} not found in hash lookup")
            return ""
        
        # Get the hash for this field
        field_hash = self.hash_lookup[entity_id].get(field)
        if not field_hash:
            # Only warn for core fields, not optional ones like relatedWork
            if warn_if_missing and field not in ['relatedWork', 'attribution', 'provision', 'genres']:
                logger.warning(f"Field {field} not found for entity {entity_id}")
            return ""
        
        # Handle NULL sentinel value - return empty string without querying Weaviate
        if field_hash == "NULL":
            return ""
        
        # Check string cache first
        if field_hash in self.string_cache:
            return self.string_cache[field_hash]
        
        # Query Weaviate for the string value using the hash and field type
        try:
            collection = self.weaviate_client.collections.get("EntityString")
            
            # Create a filter for hash and field type using Filter class
            hash_filter = Filter.by_property("hash_value").equal(field_hash)
            field_filter = Filter.by_property("field_type").equal(field)
            combined_filter = Filter.all_of([hash_filter, field_filter])
            
            # Use fetch_objects with the filter - proper v4 syntax
            query_result = collection.query.fetch_objects(
                filters=combined_filter,
                return_properties=["original_string", "hash_value", "field_type"]
            )
            
            # Check if we found any matching objects
            if query_result.objects and len(query_result.objects) > 0:
                obj = query_result.objects[0]
                string_value = obj.properties.get("original_string", "")
                self.string_cache[field_hash] = string_value
                return string_value
                
        except Exception as e:
            logger.warning(f"Error retrieving string for hash {field_hash}: {str(e)}")
        
        return ""
        
    def _get_vector(self, entity_id: str, field: str) -> Optional[np.ndarray]:
        """
        Get the vector for an entity field with standardized 'default' key access pattern.
        
        Args:
            entity_id: Entity ID
            field: Field name
            
        Returns:
            Vector as numpy array or None if not found
        """
        # Check vector cache first
        if entity_id in self.vector_cache and field in self.vector_cache[entity_id]:
            return self.vector_cache[entity_id][field]
        
        # Determine if we should suppress warnings
        # Vectors won't exist during preprocessing initialization
        suppress_warnings = self.config.get("suppress_vector_warnings", False)
        
        # If not in cache, try to get from hash lookup and query Weaviate
        if entity_id in self.hash_lookup and field in self.hash_lookup[entity_id]:
            field_hash = self.hash_lookup[entity_id][field]
            
            try:
                collection = self.weaviate_client.collections.get("EntityString")
                
                # Query using the proper v4 client method - fetch_objects with Filter class
                hash_filter = Filter.by_property("hash_value").equal(field_hash)
                field_filter = Filter.by_property("field_type").equal(field)
                combined_filter = Filter.all_of([hash_filter, field_filter])
                
                # Use proper v4 client syntax for vector retrieval
                query_result = collection.query.fetch_objects(
                    filters=combined_filter,
                    include_vector=True
                )
                
                # Check if we found any matching objects
                if query_result.objects and len(query_result.objects) > 0:
                    obj = query_result.objects[0]
                    if hasattr(obj, 'vector'):
                        # Handle vector retrieval for different configurations
                        vector_data = None
                        if isinstance(obj.vector, dict):
                            # Check for jina_colbert first, then fall back to default
                            if 'jina_colbert' in obj.vector:
                                vector_data = obj.vector['jina_colbert']
                            elif 'default' in obj.vector:
                                vector_data = obj.vector['default']
                            else:
                                # Try to get any available vector
                                available_keys = list(obj.vector.keys())
                                if available_keys:
                                    vector_data = obj.vector[available_keys[0]]
                        elif isinstance(obj.vector, list):
                            # Backward compatibility for direct vector access
                            vector_data = obj.vector
                        
                        if vector_data is not None:
                            # Handle both single vectors and multi-vectors (ColBERT)
                            if isinstance(vector_data, list) and len(vector_data) > 0:
                                if isinstance(vector_data[0], list):
                                    # Multi-vector format (ColBERT): aggregate to single vector
                                    # Use mean pooling to create a single representation
                                    multi_vectors = np.array(vector_data, dtype=np.float32)
                                    vector_array = np.mean(multi_vectors, axis=0)
                                    logger.debug(f"Aggregated {len(vector_data)} token vectors to single vector for {entity_id}.{field}")
                                else:
                                    # Single vector format
                                    vector_array = np.array(vector_data, dtype=np.float32)
                            else:
                                # Direct array format
                                vector_array = np.array(vector_data, dtype=np.float32)
                            
                            if entity_id not in self.vector_cache:
                                self.vector_cache[entity_id] = {}
                            self.vector_cache[entity_id][field] = vector_array
                            return self.vector_cache[entity_id][field]
            except Exception as e:
                if not suppress_warnings:
                    logger.warning(f"Error retrieving vector for {entity_id}.{field}: {str(e)}")
        
        # Log warning unless suppressed, and only for fields we care about
        important_fields = ['person', 'title']
        if not suppress_warnings and field in important_fields:
            logger.warning(f"Vector not found for {entity_id}.{field}")
        return None
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors with robust error handling and bounds enforcement.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (0-1), properly bounded
        """
        # Thorough input validation
        if vec1 is None or vec2 is None:
            logger.debug("Null vector detected in cosine similarity calculation")
            return 0.0
            
        # Ensure vectors are numpy arrays and have the same shape
        try:
            if not isinstance(vec1, np.ndarray):
                vec1 = np.array(vec1, dtype=np.float32)
            if not isinstance(vec2, np.ndarray):
                vec2 = np.array(vec2, dtype=np.float32)
                
            # Check for consistent dimensions
            if vec1.shape != vec2.shape:
                logger.error(f"Vector dimension mismatch in cosine similarity: {vec1.shape} vs {vec2.shape}")
                return 0.0
                
            # Check for NaN or infinite values
            if np.isnan(vec1).any() or np.isnan(vec2).any() or np.isinf(vec1).any() or np.isinf(vec2).any():
                logger.error("Vector contains NaN or infinite values in cosine similarity calculation")
                return 0.0
                
        except Exception as e:
            logger.error(f"Error preprocessing vectors for cosine similarity: {str(e)}")
            return 0.0
            
        # Calculate vector norms with numerical stability
        try:
            # Use safe computation with better numerical stability
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            # Handle zero vectors properly
            if norm1 < 1e-10 or norm2 < 1e-10:  # Use small epsilon instead of exact zero
                logger.debug("Near-zero norm detected in cosine similarity calculation")
                # If both vectors are zero-like, they're technically identical
                if norm1 < 1e-10 and norm2 < 1e-10:
                    return 1.0
                return 0.0
                
            # Compute cosine similarity
            dot_product = np.dot(vec1, vec2)
            similarity = dot_product / (norm1 * norm2)
            
            # Ensure result is within [-1.0, 1.0] range (theoretical bounds for cosine similarity)
            # Handle numerical precision issues
            if similarity < -1.0:
                if similarity < -1.01:  # Only log significant deviation
                    logger.debug(f"Cosine similarity below -1.0 encountered: {similarity}")
                similarity = -1.0
            elif similarity > 1.0:
                if similarity > 1.01:  # Only log significant deviation
                    logger.debug(f"Cosine similarity above 1.0 encountered: {similarity}")
                similarity = 1.0
            
            # Check for NaN (can happen with numerical instability)
            if np.isnan(similarity):
                logger.error("NaN result in cosine similarity calculation")
                return 0.0
                
            # For entity resolution, we typically want [0,1] range
            return max(0.0, similarity)
            
        except Exception as e:
            logger.error(f"Error computing cosine similarity: {str(e)}")
            return 0.0
    
    def _calc_levenshtein_distance(self, s1: str, s2: str) -> int:
        """
        Calculate Levenshtein distance between two strings.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Levenshtein distance
        """
        if s1 == s2:
            return 0
            
        # Ensure s1 is the shorter string
        if len(s1) > len(s2):
            s1, s2 = s2, s1
            
        # Special cases for empty strings
        if not s1:
            return len(s2)
            
        # Initialize previous row of distances
        previous_row = range(len(s2) + 1)
        
        # Calculate edit distance
        for i, c1 in enumerate(s1):
            # Initialize current row with i+1 (distance from empty string)
            current_row = [i + 1]
            
            for j, c2 in enumerate(s2):
                # Calculate insertions, deletions, and substitutions
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                
                # Get minimum of the three operations
                current_row.append(min(insertions, deletions, substitutions))
                
            # Update previous row
            previous_row = current_row
            
        return previous_row[-1]
            
    def _calc_levenshtein_similarity(self, s1: str, s2: str) -> float:
        """
        Calculate normalized Levenshtein similarity between two strings.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Similarity score between 0 and 1
        """
        if s1 == s2:
            return 1.0
            
        # Handle empty strings
        if not s1 or not s2:
            return 0.0 if s1 or s2 else 1.0  # Empty strings match each other
            
        # Calculate Levenshtein distance
        distance = self._calc_levenshtein_distance(s1, s2)
        
        # Normalize by maximum possible distance
        max_len = max(len(s1), len(s2))
        return 1.0 - (distance / max_len)
            
    def _calc_jaro_winkler_similarity(self, s1: str, s2: str, p: float = 0.1) -> float:
        """
        Calculate Jaro-Winkler similarity between two strings.
        
        Args:
            s1: First string
            s2: Second string
            p: Scaling factor for prefix matches (default 0.1)
            
        Returns:
            Similarity score between 0 and 1
        """
        # Fast path for identical strings
        if s1 == s2:
            return 1.0
            
        # Handle empty strings
        if not s1 or not s2:
            return 0.0
            
        # Calculate Jaro similarity first
        len1, len2 = len(s1), len(s2)
        
        # Maximum distance for matches
        match_distance = max(len1, len2) // 2 - 1
        match_distance = max(0, match_distance)  # Ensure non-negative
        
        # Track matches
        s1_matches = [False] * len1
        s2_matches = [False] * len2
        
        # Count matching characters
        matches = 0
        for i in range(len1):
            start = max(0, i - match_distance)
            end = min(i + match_distance + 1, len2)
            
            for j in range(start, end):
                if not s2_matches[j] and s1[i] == s2[j]:
                    s1_matches[i] = True
                    s2_matches[j] = True
                    matches += 1
                    break
        
        # If no matches, return 0
        if matches == 0:
            return 0.0
            
        # Count transpositions
        transpositions = 0
        k = 0
        for i in range(len1):
            if s1_matches[i]:
                while not s2_matches[k]:
                    k += 1
                if s1[i] != s2[k]:
                    transpositions += 1
                k += 1
                
        # Calculate Jaro similarity
        jaro = (matches / len1 + matches / len2 + (matches - transpositions / 2) / matches) / 3
        
        # Calculate common prefix length (for Winkler modification)
        l = 0
        max_l = min(4, min(len1, len2))  # Maximum 4 characters as per original algorithm
        while l < max_l and s1[l] == s2[l]:
            l += 1
            
        # Apply Winkler modification
        return jaro + l * p * (1 - jaro)
            
    def _calc_person_levenshtein_similarity(self, left_id: str, right_id: str) -> float:
        """
        Calculate Levenshtein similarity between person names.
        
        Args:
            left_id: ID of the first entity
            right_id: ID of the second entity
            
        Returns:
            Similarity score between 0 and 1
        """
        # Get parameters from config
        params = self.feature_params.get('person_levenshtein_similarity', {})
        
        # Get normalized person strings
        left_person = self._normalize_string(self._get_string_value(left_id, 'person'))
        right_person = self._normalize_string(self._get_string_value(right_id, 'person'))
        
        # Calculate similarity
        return self._calc_levenshtein_similarity(left_person, right_person)
    
    def _calc_person_jaro_winkler_similarity(self, left_id: str, right_id: str) -> float:
        """
        Calculate Jaro-Winkler similarity between person names.
        
        Args:
            left_id: ID of the first entity
            right_id: ID of the second entity
            
        Returns:
            Similarity score between 0 and 1
        """
        # Get parameters from config
        params = self.feature_params.get('person_jaro_winkler_similarity', {})
        
        # Get normalized person strings
        left_person = self._normalize_string(self._get_string_value(left_id, 'person'))
        right_person = self._normalize_string(self._get_string_value(right_id, 'person'))
        
        # Calculate similarity
        return self._calc_jaro_winkler_similarity(left_person, right_person)
    
    def _calc_person_low_levenshtein_indicator(self, left_id: str, right_id: str) -> float:
        """
        Calculate binary indicator for low Levenshtein similarity of person names.
        Returns 1.0 if similarity is below threshold, 0.0 otherwise.
        
        Args:
            left_id: ID of the first entity
            right_id: ID of the second entity
            
        Returns:
            1.0 if similarity is below threshold, 0.0 otherwise
        """
        # Get parameters from config
        params = self.feature_params.get('person_low_levenshtein_indicator', {})
        threshold = params.get('threshold', 0.8)  # Default threshold
        
        # Get similarity score
        similarity = self._calc_person_levenshtein_similarity(left_id, right_id)
        
        # Return binary indicator (1.0 if below threshold)
        return 1.0 if similarity < threshold else 0.0
    
    def _calc_person_low_jaro_winkler_indicator(self, left_id: str, right_id: str) -> float:
        """
        Calculate binary indicator for low Jaro-Winkler similarity of person names.
        Returns 1.0 if similarity is below threshold, 0.0 otherwise.
        
        Args:
            left_id: ID of the first entity
            right_id: ID of the second entity
            
        Returns:
            1.0 if similarity is below threshold, 0.0 otherwise
        """
        # Get parameters from config
        params = self.feature_params.get('person_low_jaro_winkler_indicator', {})
        threshold = params.get('threshold', 0.8)  # Default threshold
        
        # Get similarity score
        similarity = self._calc_person_jaro_winkler_similarity(left_id, right_id)
        
        # Return binary indicator (1.0 if below threshold)
        return 1.0 if similarity < threshold else 0.0
    
    def _calc_person_cosine(self, left_id: str, right_id: str) -> float:
        """
        Calculate cosine similarity between person field vectors.
        
        Args:
            left_id: ID of the first entity
            right_id: ID of the second entity
            
        Returns:
            Cosine similarity between person vectors
        """
        # Get parameters from config
        params = self.feature_params.get('person_cosine', {})
        weight = params.get('weight', 1.0)
        fallback_value = params.get('fallback_value', 0.5)
        
        # Check if person field exists in hash_lookup before trying to get vector
        # to avoid unnecessary warnings in logs
        left_has_person = left_id in self.hash_lookup and 'person' in self.hash_lookup[left_id]
        right_has_person = right_id in self.hash_lookup and 'person' in self.hash_lookup[right_id]
        
        # If either entity doesn't have a person field, return fallback value
        if not left_has_person or not right_has_person:
            return fallback_value * weight
        
        # Get person vectors only if both entities have person fields
        left_vec = self._get_vector(left_id, 'person')
        right_vec = self._get_vector(right_id, 'person')
        
        # If either vector is missing, try string comparison instead
        if left_vec is None or right_vec is None:
            # Get person strings directly from hash lookup
            left_person_hash = self.hash_lookup[left_id]['person']
            right_person_hash = self.hash_lookup[right_id]['person']
            
            # Check if the person hashes are identical, which is a quick way to check
            # if the original strings are identical without doing string lookups
            if left_person_hash == right_person_hash:
                return 1.0 * weight
            
            # Fall back to string value comparison
            left_person = None
            right_person = None
            
            # Only look up strings if we need to
            if left_person_hash in self.string_cache:
                left_person = self.string_cache[left_person_hash]
            else:
                left_person = self._get_string_value(left_id, 'person')
                
            if right_person_hash in self.string_cache:
                right_person = self.string_cache[right_person_hash]
            else:
                right_person = self._get_string_value(right_id, 'person')
            
            # Check if person values are identical strings
            if left_person and right_person and left_person == right_person:
                return 1.0 * weight
                
            # Otherwise return neutral value
            return fallback_value * weight
        
        # Calculate cosine similarity
        similarity = self._cosine_similarity(left_vec, right_vec)
        return similarity * weight
        
    def _calc_person_low_cosine_indicator(self, left_id: str, right_id: str) -> float:
        """
        Calculate binary indicator for low cosine similarity of person name embeddings.
        Returns 1.0 if similarity is below threshold, 0.0 otherwise.
        This complements the person_low_levenshtein_indicator for improved classifier training.
        
        Args:
            left_id: ID of the first entity
            right_id: ID of the second entity
            
        Returns:
            1.0 if similarity is below threshold, 0.0 otherwise
        """
        # Create a globally ordered key to prevent thread race conditions
        # Always use the same order regardless of which ID is passed first
        # This ensures symmetric caching and thread safety
        if left_id > right_id:
            left_id, right_id = right_id, left_id
        
        # Use class-level cache version for consistency across instances
        cache_key = f"pci:{self.cache_version}:{left_id}:{right_id}"
        
        # Generate transaction ID for better logging trackability
        transaction_id = self._generate_transaction_id(left_id, right_id)
        
        # Critical thread safety - use proper locking
        # First check if result is already in cache
        with self.cache_lock:
            if cache_key in self.similarity_cache:
                cached_value = self.similarity_cache[cache_key]
                return cached_value
        
        # Get parameters from config
        params = self.feature_params.get('person_low_cosine_indicator', {})
        threshold = params.get('threshold', 0.7)  # Default threshold
        
        # Get normalized person strings for checking exact matches
        try:
            left_person = self._normalize_string(self._get_string_value(left_id, 'person'))
            right_person = self._normalize_string(self._get_string_value(right_id, 'person'))
            
            # Enable detailed logging at the debug level
            debug_mode = self.config.get("debug_binary_indicators", False)
            if debug_mode:
                logger.info(f"[{transaction_id}] COSINE INDICATOR: Comparing '{self._get_string_value(left_id, 'person')}' vs '{self._get_string_value(right_id, 'person')}'")
            
            # CRITICAL FIX: Check for identical strings first and always return 0.0
            # This is the most important part of the fix
            if left_person and right_person and left_person == right_person:
                # Check for invisible character differences using byte representation
                left_bytes = left_person.encode('utf-8')
                right_bytes = right_person.encode('utf-8')
                
                # For debugging encoding issues
                if debug_mode and left_bytes != right_bytes:
                    logger.info(f"[{transaction_id}] ENCODING ANOMALY DETECTED: '{left_person}' != '{right_person}' as bytes")
                    logger.info(f"[{transaction_id}] Left bytes: {[b for b in left_bytes]}")
                    logger.info(f"[{transaction_id}] Right bytes: {[b for b in right_bytes]}")
                
                if debug_mode:
                    logger.info(f"[{transaction_id}] EXACT MATCH: Identical strings '{left_person}'")
                
                # Store in cache with lock
                with self.cache_lock:
                    self.similarity_cache[cache_key] = 0.0
                return 0.0  # Identical strings are definite matches
                
            # Get vectors for cosine similarity
            left_vec = self._get_vector(left_id, 'person')
            right_vec = self._get_vector(right_id, 'person')
            
            # If vectors are missing, fall back to string-based comparison
            if left_vec is None or right_vec is None:
                # Use string similarity fallback
                if left_person and right_person:
                    # Try Jaro-Winkler first as it's generally better for names
                    jw_sim = self._calc_jaro_winkler_similarity(left_person, right_person)
                    
                    # Establish threshold based on the same default for cosine
                    jw_threshold = params.get('string_fallback_threshold', threshold)
                    result = 1.0 if jw_sim < jw_threshold else 0.0
                    
                    if debug_mode:
                        logger.info(f"[{transaction_id}] VECTOR MISSING: Using string similarity fallback: {jw_sim:.4f}")
                        
                    # Store in cache with lock
                    with self.cache_lock:
                        self.similarity_cache[cache_key] = result
                    return result
                else:
                    # Default to "different" if we can't compare
                    if debug_mode:
                        logger.info(f"[{transaction_id}] NO COMPARABLE DATA: Defaulting to different (1.0)")
                        
                    # Store in cache with lock
                    with self.cache_lock:
                        self.similarity_cache[cache_key] = 1.0
                    return 1.0
            
            # Calculate cosine similarity
            similarity = np.dot(left_vec, right_vec) / (np.linalg.norm(left_vec) * np.linalg.norm(right_vec))
            
            # Log the calculated similarity for debugging
            if debug_mode:
                logger.info(f"[{transaction_id}] Cosine similarity: {similarity:.4f}, threshold: {threshold}, indicator: {1.0 if similarity < threshold else 0.0}")
                
            # Create result
            result = 1.0 if similarity < threshold else 0.0
            
            # Store in cache with lock
            with self.cache_lock:
                self.similarity_cache[cache_key] = result
                
            return result
        except Exception as e:
            logger.error(f"Error calculating person_low_cosine_indicator: {e}")
            # Store error case in cache too to avoid repeated failures
            with self.cache_lock:
                self.similarity_cache[cache_key] = 1.0  # Default to "different" on error
            return 1.0
    
    def _calc_person_title_squared(self, left_id: str, right_id: str) -> float:
        """
        Calculate person cosine similarity with title cosine similarity squared.
        Emphasizes title divergence to disambiguate similar names with semantically distant titles.
        
        Args:
            left_id: ID of the first entity
            right_id: ID of the second entity
            
        Returns:
            person_cosine * title_cosine^2
        """
        # Get feature parameters from config
        params = self.feature_params.get('person_title_squared', {})
        weight = params.get('weight', 1.0)
        
        try:
            # Get vectors - with null checking
            left_person_vec = self._get_vector(left_id, 'person')
            right_person_vec = self._get_vector(right_id, 'person')
            left_title_vec = self._get_vector(left_id, 'title')
            right_title_vec = self._get_vector(right_id, 'title')
            
            # Validate vectors - use safe fallback values when needed
            if left_person_vec is None or right_person_vec is None:
                logger.warning(f"Missing person vectors for {left_id} or {right_id}")
                person_sim = 0.5  # Use middle value as fallback
            else:
                person_sim = self._cosine_similarity(left_person_vec, right_person_vec)
            
            if left_title_vec is None or right_title_vec is None:
                logger.warning(f"Missing title vectors for {left_id} or {right_id}")
                title_sim = 0.5  # Use middle value as fallback
            else:
                title_sim = self._cosine_similarity(left_title_vec, right_title_vec)
            
            # Ensure similarity values are valid
            if not isinstance(person_sim, (int, float, np.number)):
                logger.warning(f"Non-numeric person similarity value: {type(person_sim)} for {left_id} - {right_id}")
                person_sim = 0.5
            elif np.isnan(person_sim) or np.isinf(person_sim):
                logger.warning(f"Invalid numeric person similarity value: {person_sim} for {left_id} - {right_id}")
                person_sim = 0.5
                
            if not isinstance(title_sim, (int, float, np.number)):
                logger.warning(f"Non-numeric title similarity value: {type(title_sim)} for {left_id} - {right_id}")
                title_sim = 0.5
            elif np.isnan(title_sim) or np.isinf(title_sim):
                logger.warning(f"Invalid numeric title similarity value: {title_sim} for {left_id} - {right_id}")
                title_sim = 0.5
            
            # Calculate composite similarity with improved error handling
            # Average to make more symmetrical
            avg_sim = (person_sim + title_sim) / 2
            squared = avg_sim * avg_sim
            result = squared * weight
            
            # Check for numeric issues
            if not isinstance(result, (int, float, np.number)):
                logger.warning(f"Non-numeric result: {type(result)} for person_title_squared, using fallback value")
                return 0.5 * weight
            elif np.isnan(result) or np.isinf(result):
                logger.warning(f"Invalid numeric result: {result} for person_title_squared, using fallback value")
                return 0.5 * weight
            
            # Explicitly clamp result to [0,1] range
            return max(0.0, min(1.0, result))
        except Exception as e:
            logger.error(f"Error computing features for {left_id} - {right_id}: {e}")
            return 0.5 * weight  # Use safe fallback value
    
    def _calc_composite_cosine(self, left_id: str, right_id: str) -> float:
        """
        Calculate a composite feature based on cosine similarity of the composite field.
        
        Args:
            left_id: ID of the first entity
            right_id: ID of the second entity
            
        Returns:
            Composite similarity score
        """
        # Get parameters from config
        params = self.feature_params.get('composite_cosine', {})
        weight = params.get('weight', 0.6)
        
        # Get the composite vectors
        left_vec = self._get_vector(left_id, 'composite')
        right_vec = self._get_vector(right_id, 'composite')
        
        # If either vector is missing, return default value
        if left_vec is None or right_vec is None:
            return 0.5  # Neutral value when no comparison possible
        
        # Calculate cosine similarity
        try:
            similarity = self._cosine_similarity(left_vec, right_vec)
            return similarity * weight
        except Exception as e:
            logger.warning(f"Error calculating composite cosine similarity: {e}")
            return 0.5  # Default value on error
    
    def _calc_birth_death_match(self, left_id: str, right_id: str) -> float:
        """
        Calculate binary indicator for matching birth/death years.
        
        CRITICAL: If both birth AND death years are present for both entities,
        then BOTH must match (within tolerance) for a positive match.
        If only one type of year is available, then that one must match.
        
        Args:
            left_id: ID of the first entity
            right_id: ID of the second entity
            
        Returns:
            1.0 if years match according to above rules (within tolerance), 0.0 otherwise
        """
        # Get feature parameters from config
        params = self.feature_params.get('birth_death_match', {})
        tolerance = params.get('tolerance', 2)
        weight = params.get('weight', 1.0)
        
        # Get person name strings with potential birth/death years
        left_person = self._get_string_value(left_id, 'person')
        right_person = self._get_string_value(right_id, 'person')
        
        # Extract birth/death years using the provided extractor
        left_birth, left_death = self.birth_death_extractor.parse(left_person)
        right_birth, right_death = self.birth_death_extractor.parse(right_person)
        
        # Track if we have years to compare and if they match
        birth_years_available = left_birth is not None and right_birth is not None
        death_years_available = left_death is not None and right_death is not None
        both_years_available = birth_years_available and death_years_available
        no_years_available = left_birth is None and right_birth is None and left_death is None and right_death is None
        
        # For detailed debugging when needed
        debug_mode = self.config.get("debug_features", False)
        if debug_mode:
            logger.info(f"===== BIRTH/DEATH MATCH DIAGNOSTIC =====")
            logger.info(f"Left person ({left_id}): '{left_person}'")
            logger.info(f"  Birth: {left_birth}, Death: {left_death}")
            logger.info(f"Right person ({right_id}): '{right_person}'")
            logger.info(f"  Birth: {right_birth}, Death: {right_death}")
            logger.info(f"Tolerance: {tolerance}")
        
        # NEW CASE: If neither entity has any birth/death years, return 1.0
        # This means "no evidence of conflict" rather than "confirmed match"
        if no_years_available:
            if debug_mode:
                logger.info(f"Match found: Neither entity has birth/death years - no temporal conflict")
                logger.info(f"========================================")
            return 0.0 * weight
        
        # Check for birth year match if both entities have birth years
        birth_year_match = False
        if birth_years_available:
            birth_year_match = abs(left_birth - right_birth) <= tolerance
            if debug_mode:
                logger.info(f"Birth year difference: {abs(left_birth - right_birth)} years")
                logger.info(f"Birth year match: {birth_year_match}")
                
        # Check for death year match if both entities have death years
        death_year_match = False
        if death_years_available:
            death_year_match = abs(left_death - right_death) <= tolerance
            if debug_mode:
                logger.info(f"Death year difference: {abs(left_death - right_death)} years")
                logger.info(f"Death year match: {death_year_match}")
        
        # CRITICAL FIX: If both birth and death years are available for both entities,
        # then BOTH must match for a positive result
        if both_years_available:
            # Both birth and death must match when both are available
            if birth_year_match and death_year_match:
                if debug_mode:
                    logger.info(f"Match found: Both birth AND death years match")
                    logger.info(f"========================================")
                return 1.0 * weight
        else:
            # Handle cases where only one year type (birth or death) is available
            
            # Case 1: Only birth years are available for both entities
            if birth_years_available and birth_year_match:
                if debug_mode:
                    logger.info(f"Match found on birth year (only available year type)")
                    logger.info(f"========================================")
                return 1.0 * weight
                
            # Case 2: Only death years are available for both entities
            if death_years_available and death_year_match:
                if debug_mode:
                    logger.info(f"Match found on death year (only available year type)")
                    logger.info(f"========================================")
                return 1.0 * weight
                
            # Case 3: One entity has only birth, the other has both birth and death
            if left_birth is not None and right_birth is not None and (left_death is None or right_death is None):
                if birth_year_match:
                    if debug_mode:
                        logger.info(f"Match found on birth year (one entity has only birth)")
                        logger.info(f"========================================")
                    return 1.0 * weight
                    
            # Case 4: One entity has only death, the other has both birth and death
            if left_death is not None and right_death is not None and (left_birth is None or right_birth is None):
                if death_year_match:
                    if debug_mode:
                        logger.info(f"Match found on death year (one entity has only death)")
                        logger.info(f"========================================")
                    return 1.0 * weight
        
        # Log lack of match for debugging
        if debug_mode:
            reasons = []
            if not no_years_available and not birth_years_available and not death_years_available:
                reasons.append("Partial temporal data available but incomplete for comparison")
            elif both_years_available:
                if not birth_year_match:
                    reasons.append(f"Birth years differ by more than {tolerance} years")
                if not death_year_match:
                    reasons.append(f"Death years differ by more than {tolerance} years")
            elif birth_years_available and not birth_year_match:
                reasons.append(f"Birth years differ by more than {tolerance} years")
            elif death_years_available and not death_year_match:
                reasons.append(f"Death years differ by more than {tolerance} years")
            logger.info(f"No match: {', '.join(reasons)}")
            logger.info(f"========================================")
                
        # No match found
        return 0.0 * weight
    
    def _calc_title_cosine_squared(self, left_id: str, right_id: str) -> float:
        """
        Calculate title cosine similarity squared.
        
        Args:
            left_id: ID of the first entity
            right_id: ID of the second entity
            
        Returns:
            Squared cosine similarity of title embeddings
        """
        # Get title vectors
        left_vec = self._get_vector(left_id, 'title')
        right_vec = self._get_vector(right_id, 'title')
        
        if left_vec is None or right_vec is None:
            # Default to 0.25 (mid-range squared) if we can't compare
            return 0.25
        
        # Calculate cosine similarity
        similarity = np.dot(left_vec, right_vec) / (np.linalg.norm(left_vec) * np.linalg.norm(right_vec))
        
        # Return squared value
        return similarity ** 2
    
    def _calc_title_role_adjusted(self, left_id: str, right_id: str) -> float:
        """
        Calculate title similarity adjusted by role compatibility.
        
        Args:
            left_id: ID of the first entity
            right_id: ID of the second entity
            
        Returns:
            Role-adjusted title similarity
        """
        # Get parameters from config
        params = self.feature_params.get('title_role_adjusted', {})
        
        # Extract roles from titles
        left_title = self._get_string_value(left_id, 'title')
        right_title = self._get_string_value(right_id, 'title')
        
        # Get and normalize embedded roles from the hash lookup
        left_role_raw = self._get_string_value(left_id, 'roles') if left_id in self.hash_lookup and 'roles' in self.hash_lookup[left_id] else ""
        right_role_raw = self._get_string_value(right_id, 'roles') if right_id in self.hash_lookup and 'roles' in self.hash_lookup[right_id] else ""
        
        # Normalize to canonical categories
        left_role = self._normalize_role_category(left_role_raw)
        right_role = self._normalize_role_category(right_role_raw)
            
        # Calculate role compatibility score
        role_score = 1.0  # Default to full compatibility
        
        # Check role compatibility matrix with correct key format
        role_key = f"{left_role}-{right_role}"
        reverse_key = f"{right_role}-{left_role}"
        
        if role_key in self.role_compatibility:
            role_score = self.role_compatibility[role_key]
        elif reverse_key in self.role_compatibility:
            role_score = self.role_compatibility[reverse_key]
        
        # Get title vectors for cosine similarity
        left_vec = self._get_vector(left_id, 'title')
        right_vec = self._get_vector(right_id, 'title')
        
        if left_vec is None or right_vec is None:
            # Default to 0.5 (neutral) if we can't compare
            return 0.5 * role_score
        
        # Calculate cosine similarity
        similarity = np.dot(left_vec, right_vec) / (np.linalg.norm(left_vec) * np.linalg.norm(right_vec))
        
        # Adjust by role compatibility
        return similarity * role_score
    
    def _calc_person_title_adjusted_squared(self, left_id: str, right_id: str) -> float:
        """
        Calculate a composite feature incorporating person, title, and role information.
        
        Args:
            left_id: ID of the first entity
            right_id: ID of the second entity
            
        Returns:
            Squared composite value
        """
        # Get parameters from config
        params = self.feature_params.get('person_title_adjusted_squared', {})
        person_weight = params.get('person_weight', 2.0)
        title_weight = params.get('title_weight', 1.0)
        
        # Get component features
        person_indicator = self._calc_person_low_cosine_indicator(left_id, right_id)
        title_value = self._calc_title_role_adjusted(left_id, right_id)
        
        # Convert title value to indicator (low similarity = 1, high = 0)
        title_indicator = 1.0 if title_value < 0.6 else 0.0
        
        # Calculate weighted value
        value = (person_indicator * person_weight + title_indicator * title_weight) / (person_weight + title_weight)
        
        # Return squared value for better class separation
        return value ** 2
        
    def _calc_role_cosine(self, left_id: str, right_id: str) -> float:
        """
        Calculate cosine similarity between roles field vectors.
        
        Args:
            left_id: ID of the first entity
            right_id: ID of the second entity
            
        Returns:
            Cosine similarity between roles vectors
        """
        # Get parameters from config
        params = self.feature_params.get('roles_cosine', {})
        weight = params.get('weight', 1.0)
        fallback_value = params.get('fallback_value', 0.5)
        
        # Check if roles field exists in hash_lookup before trying to get vector
        # to avoid unnecessary warnings in logs
        left_has_roles = left_id in self.hash_lookup and 'roles' in self.hash_lookup[left_id]
        right_has_roles = right_id in self.hash_lookup and 'roles' in self.hash_lookup[right_id]
        
        # If either entity doesn't have a roles field, return fallback value
        if not left_has_roles or not right_has_roles:
            return fallback_value * weight
        
        # Get roles vectors only if both entities have roles fields
        left_vec = self._get_vector(left_id, 'roles')
        right_vec = self._get_vector(right_id, 'roles')
        
        # If either vector is missing, try string comparison instead
        if left_vec is None or right_vec is None:
            # Get roles strings directly from hash lookup
            left_roles_hash = self.hash_lookup[left_id]['roles']
            right_roles_hash = self.hash_lookup[right_id]['roles']
            
            # Check if the roles hashes are identical, which is a quick way to check
            # if the original strings are identical without doing string lookups
            if left_roles_hash == right_roles_hash:
                return 1.0 * weight
            
            # Fall back to string value comparison
            left_roles = None
            right_roles = None
            
            # Only look up strings if we need to
            if left_roles_hash in self.string_cache:
                left_roles = self.string_cache[left_roles_hash]
            else:
                left_roles = self._get_string_value(left_id, 'roles')
                
            if right_roles_hash in self.string_cache:
                right_roles = self.string_cache[right_roles_hash]
            else:
                right_roles = self._get_string_value(right_id, 'roles')
            
            # Check if roles are identical strings
            if left_roles and right_roles and left_roles == right_roles:
                return 1.0 * weight
                
            # Otherwise return neutral value
            return fallback_value * weight
        
        # Calculate cosine similarity
        similarity = self._cosine_similarity(left_vec, right_vec)
        return similarity * weight
    
    def _calc_roles_cosine_weighted(self, left_id: str, right_id: str) -> float:
        """
        Calculate role cosine similarity weighted by role compatibility matrix.
        
        This feature extends the base roles_cosine by applying role compatibility
        weights from the configuration to account for role relationships.
        
        Args:
            left_id: ID of the first entity
            right_id: ID of the second entity
            
        Returns:
            Role cosine similarity weighted by role compatibility
        """
        # Get parameters from config
        params = self.feature_params.get('roles_cosine_weighted', {})
        weight = params.get('weight', 1.0)
        fallback_value = params.get('fallback_value', 0.5)
        
        # Get base role cosine similarity
        base_similarity = self._calc_role_cosine(left_id, right_id)
        
        # Ensure base_similarity is a float
        try:
            base_similarity = float(base_similarity) if base_similarity is not None else 0.0
        except (ValueError, TypeError) as e:
            logger.warning(f"Type error in roles_cosine_weighted base similarity for {left_id}, {right_id}: {e}")
            logger.warning(f"base_similarity value: {base_similarity} (type: {type(base_similarity)})")
            base_similarity = 0.0
        
        # Extract and normalize role strings for compatibility weighting
        left_role_raw = self._get_string_value(left_id, 'roles') if left_id in self.hash_lookup and 'roles' in self.hash_lookup[left_id] else ""
        right_role_raw = self._get_string_value(right_id, 'roles') if right_id in self.hash_lookup and 'roles' in self.hash_lookup[right_id] else ""
        
        # Normalize to canonical categories
        left_role = self._normalize_role_category(left_role_raw)
        right_role = self._normalize_role_category(right_role_raw)
        
        # Apply role compatibility matrix weights
        role_key = f"{left_role}-{right_role}"
        reverse_key = f"{right_role}-{left_role}"
        
        # Look up compatibility score
        compatibility = 1.0  # Default to full compatibility
        if role_key in self.role_compatibility:
            compatibility = self.role_compatibility[role_key]
        elif reverse_key in self.role_compatibility:
            compatibility = self.role_compatibility[reverse_key]
        
        # Ensure compatibility is a float
        try:
            compatibility = float(compatibility) if compatibility is not None else 1.0
        except (ValueError, TypeError) as e:
            logger.warning(f"Type error in role compatibility value for {role_key}/{reverse_key}: {e}")
            logger.warning(f"compatibility value: {compatibility} (type: {type(compatibility)})")
            compatibility = 1.0
        
        # Weight the base similarity by role compatibility
        weighted_similarity = base_similarity * compatibility
        
        # Ensure final result is float
        try:
            result = float(weighted_similarity * weight)
            return result
        except (ValueError, TypeError) as e:
            logger.warning(f"Type error in final roles_cosine_weighted calculation: {e}")
            logger.warning(f"weighted_similarity: {weighted_similarity}, weight: {weight}")
            return 0.0
    
    def _calc_marcKey_cosine(self, left_id: str, right_id: str) -> float:
        """
        Calculate cosine similarity between marcKey field vectors.
        
        Args:
            left_id: ID of the first entity
            right_id: ID of the second entity
            
        Returns:
            Cosine similarity between marcKey vectors
        """
        # Get parameters from config
        params = self.feature_params.get('marcKey_cosine', {})
        weight = params.get('weight', 1.0)
        fallback_value = params.get('fallback_value', 0.5)
        
        # Check if marcKey field exists in hash_lookup before trying to get vector
        # to avoid unnecessary warnings in logs
        left_has_marcKey = left_id in self.hash_lookup and 'marcKey' in self.hash_lookup[left_id]
        right_has_marcKey = right_id in self.hash_lookup and 'marcKey' in self.hash_lookup[right_id]
        
        # If either entity doesn't have a marcKey field, return fallback value
        if not left_has_marcKey or not right_has_marcKey:
            return fallback_value * weight
        
        # Get marcKey vectors only if both entities have marcKey fields
        left_vec = self._get_vector(left_id, 'marcKey')
        right_vec = self._get_vector(right_id, 'marcKey')
        
        # If either vector is missing, try string comparison instead
        if left_vec is None or right_vec is None:
            # Get marcKey strings directly from hash lookup
            left_marcKey_hash = self.hash_lookup[left_id]['marcKey']
            right_marcKey_hash = self.hash_lookup[right_id]['marcKey']
            
            # Check if the marcKey hashes are identical, which is a quick way to check
            # if the original strings are identical without doing string lookups
            if left_marcKey_hash == right_marcKey_hash:
                return 1.0 * weight
            
            # Fall back to string value comparison
            left_marcKey = None
            right_marcKey = None
            
            # Only look up strings if we need to
            if left_marcKey_hash in self.string_cache:
                left_marcKey = self.string_cache[left_marcKey_hash]
            else:
                left_marcKey = self._get_string_value(left_id, 'marcKey')
                
            if right_marcKey_hash in self.string_cache:
                right_marcKey = self.string_cache[right_marcKey_hash]
            else:
                right_marcKey = self._get_string_value(right_id, 'marcKey')
            
            # Check if marcKey values are identical strings
            if left_marcKey and right_marcKey and left_marcKey == right_marcKey:
                return 1.0 * weight
                
            # Otherwise return neutral value
            return fallback_value * weight
        
        # Calculate cosine similarity
        similarity = self._cosine_similarity(left_vec, right_vec)
        return similarity * weight
        
    def _calc_person_role_squared(self, left_id: str, right_id: str) -> float:
        """
        Calculate person cosine similarity with roles cosine similarity squared.
        Emphasizes roles divergence to disambiguate similar names with semantically different roles.
        
        Args:
            left_id: ID of the first entity
            right_id: ID of the second entity
            
        Returns:
            Harmonic mean squared of person_cosine and roles_cosine
        """
        # Get feature parameters from config
        params = self.feature_params.get('person_role_squared', {})
        weight = params.get('weight', 1.0)
        
        try:
            # Get vectors - with null checking
            left_person_vec = self._get_vector(left_id, 'person')
            right_person_vec = self._get_vector(right_id, 'person')
            left_roles_vec = self._get_vector(left_id, 'roles')
            right_roles_vec = self._get_vector(right_id, 'roles')
            
            # Validate vectors - use safe fallback values when needed
            if left_person_vec is None or right_person_vec is None:
                logger.debug(f"Missing person vectors for {left_id} or {right_id}")
                person_sim = 0.5  # Use middle value as fallback
            else:
                person_sim = self._cosine_similarity(left_person_vec, right_person_vec)
            
            if left_roles_vec is None or right_roles_vec is None:
                logger.debug(f"Missing roles vectors for {left_id} or {right_id}")
                roles_sim = 0.5  # Use middle value as fallback
            else:
                roles_sim = self._cosine_similarity(left_roles_vec, right_roles_vec)
            
            # Ensure similarity values are valid
            if not isinstance(person_sim, (int, float, np.number)):
                logger.warning(f"Non-numeric person similarity value: {type(person_sim)} for {left_id} - {right_id}")
                person_sim = 0.5
            elif np.isnan(person_sim) or np.isinf(person_sim):
                logger.warning(f"Invalid numeric person similarity value: {person_sim} for {left_id} - {right_id}")
                person_sim = 0.5
                
            if not isinstance(roles_sim, (int, float, np.number)):
                logger.warning(f"Non-numeric roles similarity value: {type(roles_sim)} for {left_id} - {right_id}")
                roles_sim = 0.5
            elif np.isnan(roles_sim) or np.isinf(roles_sim):
                logger.warning(f"Invalid numeric roles similarity value: {roles_sim} for {left_id} - {right_id}")
                roles_sim = 0.5
            
            # Calculate composite similarity with improved error handling
            # Average to make more symmetrical
            avg_sim = (person_sim + roles_sim) / 2
            squared = avg_sim * avg_sim
            result = squared * weight
            
            # Check for numeric issues
            if not isinstance(result, (int, float, np.number)):
                logger.warning(f"Non-numeric result: {type(result)} for person_role_squared, using fallback value")
                return 0.5 * weight
            elif np.isnan(result) or np.isinf(result):
                logger.warning(f"Invalid numeric result: {result} for person_role_squared, using fallback value")
                return 0.5 * weight
            
            # Explicitly clamp result to [0,1] range
            return max(0.0, min(1.0, result))
        except Exception as e:
            logger.error(f"Error computing features for {left_id} - {right_id}: {e}")
            return 0.5 * weight  # Use safe fallback value
    
    def _calc_marcKey_title_squared(self, left_id: str, right_id: str) -> float:
        """
        Calculate marcKey cosine similarity with title cosine similarity squared.
        Follows the same pattern as person_title_squared, using harmonic mean of
        marcKey and title cosine similarities, squared.
        
        Args:
            left_id: ID of the first entity
            right_id: ID of the second entity
            
        Returns:
            marcKey_cosine * title_cosine^2
        """
        # Get feature parameters from config
        params = self.feature_params.get('marcKey_title_squared', {})
        weight = params.get('weight', 1.0)
        
        try:
            # Get vectors - with null checking
            left_marcKey_vec = self._get_vector(left_id, 'marcKey')
            right_marcKey_vec = self._get_vector(right_id, 'marcKey')
            left_title_vec = self._get_vector(left_id, 'title')
            right_title_vec = self._get_vector(right_id, 'title')
            
            # Validate vectors - use safe fallback values when needed
            if left_marcKey_vec is None or right_marcKey_vec is None:
                logger.debug(f"Missing marcKey vectors for {left_id} or {right_id}")
                marcKey_sim = 0.5  # Use middle value as fallback
            else:
                marcKey_sim = self._cosine_similarity(left_marcKey_vec, right_marcKey_vec)
            
            if left_title_vec is None or right_title_vec is None:
                logger.debug(f"Missing title vectors for {left_id} or {right_id}")
                title_sim = 0.5  # Use middle value as fallback
            else:
                title_sim = self._cosine_similarity(left_title_vec, right_title_vec)
            
            # Ensure similarity values are valid
            if not isinstance(marcKey_sim, (int, float, np.number)):
                logger.warning(f"Non-numeric marcKey similarity value: {type(marcKey_sim)} for {left_id} - {right_id}")
                marcKey_sim = 0.5
            elif np.isnan(marcKey_sim) or np.isinf(marcKey_sim):
                logger.warning(f"Invalid numeric marcKey similarity value: {marcKey_sim} for {left_id} - {right_id}")
                marcKey_sim = 0.5
                
            if not isinstance(title_sim, (int, float, np.number)):
                logger.warning(f"Non-numeric title similarity value: {type(title_sim)} for {left_id} - {right_id}")
                title_sim = 0.5
            elif np.isnan(title_sim) or np.isinf(title_sim):
                logger.warning(f"Invalid numeric title similarity value: {title_sim} for {left_id} - {right_id}")
                title_sim = 0.5
            
            # Calculate composite similarity with improved error handling
            # Average to make more symmetrical
            avg_sim = (marcKey_sim + title_sim) / 2
            squared = avg_sim * avg_sim
            result = squared * weight
            
            # Check for numeric issues
            if not isinstance(result, (int, float, np.number)):
                logger.warning(f"Non-numeric result: {type(result)} for marcKey_title_squared, using fallback value")
                return 0.5 * weight
            elif np.isnan(result) or np.isinf(result):
                logger.warning(f"Invalid numeric result: {result} for marcKey_title_squared, using fallback value")
                return 0.5 * weight
            
            # Explicitly clamp result to [0,1] range
            return max(0.0, min(1.0, result))
        except Exception as e:
            logger.error(f"Error computing features for {left_id} - {right_id}: {e}")
            return 0.5 * weight  # Use safe fallback value
    
    def _calc_composite_cosine_squared(self, left_id: str, right_id: str) -> float:
        """
        Calculate the squared value of the composite cosine similarity.
        This feature emphasizes the differences in composite similarity values,
        enhancing the separation between matching and non-matching entities.
        
        Args:
            left_id: ID of the first entity
            right_id: ID of the second entity
            
        Returns:
            composite_cosine^2
        """
        # Get parameters from config
        params = self.feature_params.get('composite_cosine_squared', {})
        weight = params.get('weight', 1.0)
        
        # Get the composite vectors
        left_vec = self._get_vector(left_id, 'composite')
        right_vec = self._get_vector(right_id, 'composite')
        
        # If either vector is missing, return default value
        if left_vec is None or right_vec is None:
            return 0.25 * weight  # Neutral squared value when no comparison possible
        
        # Calculate cosine similarity
        try:
            similarity = self._cosine_similarity(left_vec, right_vec)
            # Square the similarity value to enhance separation
            squared_similarity = similarity * similarity
            return squared_similarity * weight
        except Exception as e:
            logger.warning(f"Error calculating composite cosine squared similarity: {e}")
            return 0.25 * weight  # Default squared value on error
    
    def _calc_taxonomy_dissimilarity(self, left_id: str, right_id: str) -> float:
        """
        Calculate taxonomy-based dissimilarity between two entities.
        
        Uses SetFit classification results to determine how different the domains
        of two entities are based on the hierarchical taxonomy.
        
        Args:
            left_id: ID of the first entity
            right_id: ID of the second entity
            
        Returns:
            Dissimilarity score between 0.0 (same domain) and 1.0 (unrelated domains)
        """
        # Check if taxonomy dissimilarity is available
        if self.taxonomy_dissimilarity is None:
            logger.debug("Taxonomy dissimilarity not initialized, returning neutral value")
            return 0.5
            
        # Get parameters from config
        params = self.feature_params.get('taxonomy_dissimilarity', {})
        weight = params.get('weight', 1.0)
        
        try:
            # Use the taxonomy feature function
            return create_taxonomy_feature(left_id, right_id, 
                                         self.taxonomy_dissimilarity, 
                                         weight)
        except Exception as e:
            logger.warning(f"Error calculating taxonomy dissimilarity: {e}")
            return 0.5 * weight  # Return neutral value on error
    
    def _calc_has_related_work(self, left_id: str, right_id: str) -> float:
        """
        Calculate binary indicator for both entities having non-null relatedWork.
        
        This feature captures the intuition that well-known people are more likely
        to have related works documented, and matches between such entities are 
        more likely to be true positives.
        
        Args:
            left_id: ID of the first entity
            right_id: ID of the second entity
            
        Returns:
            1.0 if both entities have non-null relatedWork, 0.0 otherwise
        """
        # Get parameters from config
        params = self.feature_params.get('has_related_work', {})
        weight = params.get('weight', 1.0)
        
        try:
            # Get relatedWork values for both entities (quietly, no warnings for missing field)
            left_related_work = self._get_string_value(left_id, 'relatedWork', warn_if_missing=False)
            right_related_work = self._get_string_value(right_id, 'relatedWork', warn_if_missing=False)
            
            # Check if both have non-null, non-empty relatedWork
            left_has_related = bool(left_related_work and left_related_work.strip())
            right_has_related = bool(right_related_work and right_related_work.strip())
            
            # Return 1.0 if both have related work, 0.0 otherwise
            binary_value = 1.0 if (left_has_related and right_has_related) else 0.0
            
            return binary_value * weight
            
        except Exception as e:
            logger.warning(f"Error calculating has_related_work feature: {e}")
            return 0.0  # Return 0.0 on error (conservative)
    
    # ========================================
    # Non-Linear Feature Transformations
    # ========================================
    
    def _calc_confidence_amplifier_exponential(self, left_id: str, right_id: str) -> float:
        """
        Calculate exponentially amplified confidence for high-similarity pairs.
        
        Uses exponential transformation to amplify high-confidence signals while
        dampening low-confidence signals, creating better class separation.
        
        Args:
            left_id: ID of the first entity
            right_id: ID of the second entity
            
        Returns:
            Exponentially amplified confidence score
        """
        # Get parameters from config
        params = self.feature_params.get('confidence_amplifier_exponential', {})
        amplification_factor = params.get('amplification_factor', 2.0)
        threshold = params.get('threshold', 0.8)
        dampen_factor = params.get('dampen_factor', 0.5)
        weight = params.get('weight', 1.0)
        
        # Get strongest base signals
        person_sim = self._calc_person_cosine(left_id, right_id)
        composite_sim = self._calc_composite_cosine_squared(left_id, right_id)
        
        # Take the maximum as the base signal
        base_signal = max(person_sim, composite_sim)
        
        # Exponential amplification for high-confidence cases
        if base_signal > threshold:
            # exp(k * (x - threshold)) for x > threshold
            amplified = np.exp(amplification_factor * (base_signal - threshold))
            return min(1.0, amplified) * weight  # Cap at 1.0
        else:
            # Dampen low signals
            return base_signal * dampen_factor * weight
    
    def _calc_confidence_amplifier_polynomial(self, left_id: str, right_id: str) -> float:
        """
        Calculate polynomial-transformed confidence using multiple features.
        
        Uses polynomial transformation with multiple features to create
        non-linear decision boundaries.
        
        Args:
            left_id: ID of the first entity
            right_id: ID of the second entity
            
        Returns:
            Polynomial-transformed confidence score
        """
        # Get parameters from config
        params = self.feature_params.get('confidence_amplifier_polynomial', {})
        coeff_a = params.get('coeff_a', 0.5)  # Cubic coefficient
        coeff_b = params.get('coeff_b', -0.3)  # Quadratic coefficient  
        coeff_c = params.get('coeff_c', 0.8)   # Linear coefficient
        weight = params.get('weight', 1.0)
        
        # Get base features
        person_sim = self._calc_person_cosine(left_id, right_id)
        composite_sim = self._calc_composite_cosine_squared(left_id, right_id)
        
        # Geometric mean for stable combination
        epsilon = 1e-8  # Avoid log(0)
        geometric_mean = np.sqrt((person_sim + epsilon) * (composite_sim + epsilon))
        
        # Polynomial transformation: a*x^3 + b*x^2 + c*x
        result = coeff_a * (geometric_mean ** 3) + coeff_b * (geometric_mean ** 2) + coeff_c * geometric_mean
        
        # Ensure result is in [0,1] range
        result = max(0.0, min(1.0, result))
        
        return result * weight
    
    def _calc_person_title_interaction_cubic(self, left_id: str, right_id: str) -> float:
        """
        Calculate cubic interaction between person and title features.
        
        Emphasizes cases where both person and title similarities are high
        through cubic interaction terms.
        
        Args:
            left_id: ID of the first entity
            right_id: ID of the second entity
            
        Returns:
            Cubic interaction score
        """
        # Get parameters from config
        params = self.feature_params.get('person_title_interaction_cubic', {})
        interaction_weight = params.get('interaction_weight', 0.7)
        individual_weight = params.get('individual_weight', 0.3)
        weight = params.get('weight', 1.0)
        
        # Get base features
        person_sim = self._calc_person_cosine(left_id, right_id)
        title_sim = self._calc_title_role_adjusted(left_id, right_id)
        
        # Cubic interaction: emphasizes cases where both are high
        interaction_term = (person_sim * title_sim) ** 3
        individual_terms = 0.5 * (person_sim ** 3 + title_sim ** 3)
        
        result = interaction_weight * interaction_term + individual_weight * individual_terms
        
        return result * weight
    
    def _calc_marc_roles_polynomial_interaction(self, left_id: str, right_id: str) -> float:
        """
        Calculate polynomial interaction between MARC and roles features.
        
        Uses quadratic interaction with cross-terms optimized for
        bibliographic structure analysis.
        
        Args:
            left_id: ID of the first entity
            right_id: ID of the second entity
            
        Returns:
            Polynomial interaction score
        """
        # Get parameters from config
        params = self.feature_params.get('marc_roles_polynomial_interaction', {})
        weight = params.get('weight', 1.0)
        
        # Get base features
        marc_sim = self._calc_marcKey_cosine(left_id, right_id)
        roles_sim = self._calc_roles_cosine_weighted(left_id, right_id)
        
        # Quadratic interaction: f(m,r) = m^2 + r^2 + 2*m*r
        result = marc_sim**2 + roles_sim**2 + 2 * marc_sim * roles_sim
        
        # Normalize to [0,1] range (max possible value is 4 when m=r=1)
        result = min(1.0, result / 4.0)
        
        return result * weight
    
    def _calc_confidence_cascade_indicator(self, left_id: str, right_id: str) -> float:
        """
        Calculate multi-level confidence cascade indicator.
        
        Provides discrete confidence levels based on combined scores
        for more nuanced decision making.
        
        Args:
            left_id: ID of the first entity
            right_id: ID of the second entity
            
        Returns:
            Cascaded confidence indicator
        """
        # Get parameters from config
        params = self.feature_params.get('confidence_cascade_indicator', {})
        thresholds = params.get('thresholds', [0.95, 0.85, 0.70])
        levels = params.get('confidence_levels', [1.0, 0.8, 0.6, 0.2])
        person_weight = params.get('person_weight', 0.6)
        composite_weight = params.get('composite_weight', 0.4)
        weight = params.get('weight', 1.0)
        
        # Get base features
        person_sim = self._calc_person_cosine(left_id, right_id)
        composite_sim = self._calc_composite_cosine_squared(left_id, right_id)
        
        # Calculate combined score
        combined_score = person_weight * person_sim + composite_weight * composite_sim
        
        # Apply cascade thresholds
        if combined_score > thresholds[0]:
            return levels[0] * weight  # Very high confidence
        elif combined_score > thresholds[1]:
            return levels[1] * weight  # High confidence
        elif combined_score > thresholds[2]:
            return levels[2] * weight  # Medium confidence
        else:
            return levels[3] * weight  # Low confidence
    
    def _calc_evidence_strength_weighted(self, left_id: str, right_id: str) -> float:
        """
        Calculate evidence-strength weighted feature combination.
        
        Weights features by their typical reliability and coverage,
        giving more weight to features with higher discriminative power.
        
        Args:
            left_id: ID of the first entity
            right_id: ID of the second entity
            
        Returns:
            Evidence-weighted score
        """
        # Get parameters from config
        params = self.feature_params.get('evidence_strength_weighted', {})
        person_reliability = params.get('person_reliability', 0.5)
        marc_reliability = params.get('marc_reliability', 0.8)
        birth_death_reliability = params.get('birth_death_reliability', 1.0)
        weight = params.get('weight', 1.0)
        
        # Get base features
        person_sim = self._calc_person_cosine(left_id, right_id)
        marc_sim = self._calc_marcKey_cosine(left_id, right_id)
        birth_death = self._calc_birth_death_match(left_id, right_id)
        
        # Calculate total reliability weight
        total_weight = person_reliability + marc_reliability + birth_death_reliability
        
        # Weight features by reliability
        weighted_score = (person_reliability * person_sim + 
                         marc_reliability * marc_sim + 
                         birth_death_reliability * birth_death) / total_weight
        
        return weighted_score * weight
    
    def _calc_harmonic_mean_primary_features(self, left_id: str, right_id: str) -> float:
        """
        Calculate harmonic mean of primary features.
        
        Harmonic mean emphasizes cases where ALL features are high,
        providing conservative but high-confidence matches.
        
        Args:
            left_id: ID of the first entity
            right_id: ID of the second entity
            
        Returns:
            Harmonic mean of primary features
        """
        # Get parameters from config with proper type conversion
        params = self.feature_params.get('harmonic_mean_primary_features', {})
        
        # Ensure config parameters are properly typed (with fallback for safety)
        try:
            weight = float(params.get('weight', 1.0))
            epsilon = float(params.get('epsilon', 1e-8))
        except (ValueError, TypeError) as e:
            logger.warning(f"Error converting config parameters for harmonic_mean_primary_features: {e}")
            weight = 1.0
            epsilon = 1e-8
        
        # Get primary features with type checking
        person_sim = self._calc_person_cosine(left_id, right_id)
        composite_sim = self._calc_composite_cosine_squared(left_id, right_id)
        roles_sim = self._calc_roles_cosine_weighted(left_id, right_id)
        
        # Ensure all values are float and handle any type issues
        try:
            person_sim = float(person_sim) if person_sim is not None else 0.0
            composite_sim = float(composite_sim) if composite_sim is not None else 0.0
            roles_sim = float(roles_sim) if roles_sim is not None else 0.0
        except (ValueError, TypeError) as e:
            logger.warning(f"Type conversion error in harmonic_mean_primary_features for {left_id}, {right_id}: {e}")
            logger.warning(f"Values: person_sim={person_sim}, composite_sim={composite_sim}, roles_sim={roles_sim}")
            return 0.0
        
        # Harmonic mean: 3 / (1/p + 1/c + 1/r)
        # Add epsilon to avoid division by zero
        harmonic_mean = 3.0 / (1.0/(person_sim + epsilon) + 
                              1.0/(composite_sim + epsilon) + 
                              1.0/(roles_sim + epsilon))
        
        return harmonic_mean * weight
    
    def _calc_geometric_confidence_scaling(self, left_id: str, right_id: str) -> float:
        """
        Calculate geometric mean with confidence-based scaling.
        
        Uses geometric mean with variance-based confidence scaling
        to penalize inconsistent feature values.
        
        Args:
            left_id: ID of the first entity
            right_id: ID of the second entity
            
        Returns:
            Confidence-scaled geometric mean
        """
        # Get parameters from config
        params = self.feature_params.get('geometric_confidence_scaling', {})
        weight = params.get('weight', 1.0)
        variance_penalty = params.get('variance_penalty', 1.0)
        
        # Get primary features
        person_sim = self._calc_person_cosine(left_id, right_id)
        title_sim = self._calc_title_role_adjusted(left_id, right_id)
        marc_sim = self._calc_marcKey_cosine(left_id, right_id)
        
        # Geometric mean
        epsilon = 1e-8  # Avoid log(0)
        geometric_mean = ((person_sim + epsilon) * (title_sim + epsilon) * (marc_sim + epsilon)) ** (1/3)
        
        # Calculate confidence factor based on variance
        feature_values = [person_sim, title_sim, marc_sim]
        variance = np.var(feature_values)
        confidence_factor = 1.0 / (1.0 + variance_penalty * variance)
        
        return geometric_mean * confidence_factor * weight
    
    def _calculate_feature(self, feature_name: str, entity_pair: Tuple[str, str]) -> float:
        """
        Calculate a single feature for a pair of entities.
        
        Args:
            feature_name: Name of feature to calculate
            entity_pair: Tuple of (left_id, right_id)
            
        Returns:
            Feature value
        """
        left_id, right_id = entity_pair
        
        # Check if feature is in substitution mapping
        if isinstance(feature_name, str) and feature_name in self.substitution_mapping:
            # Get the feature this substitutes
            original_feature = self.substitution_mapping[feature_name]
            
            # Generate a composite key for caching
            cache_key = f"{self.cache_version}:{left_id}:{right_id}:{feature_name}"
            
            # Check cache
            if not self.disable_caching and cache_key in self.similarity_cache:
                return self.similarity_cache[cache_key]
                
            # Use custom implementation if available in component registry
            if feature_name in self.component_registry:
                # Call custom implementation with additional debug flag
                debug_mode = self.debug_custom_features and feature_name not in self._debug_composite
                if debug_mode:
                    self._debug_composite[feature_name] = True
                    
                try:
                    value = self.component_registry[feature_name](left_id, right_id, debug=debug_mode)
                    
                    # Cache result
                    if not self.disable_caching:
                        self.similarity_cache[cache_key] = value
                        
                    return value
                except Exception as e:
                    # Log error and fall back to original feature
                    logger.error(f"Error calculating custom feature {feature_name}: {e}")
                    self.calculation_issues.append({
                        "feature": feature_name,
                        "entity_pair": (left_id, right_id),
                        "error": str(e),
                        "action": "falling back to original feature"
                    })
                    
                    # Fall back to original feature
                    feature_name = original_feature
        
        # Calculate standard feature
        if isinstance(feature_name, str) and feature_name in self.feature_registry:
            return self.feature_registry[feature_name](left_id, right_id)
        else:
            logger.error(f"Feature {feature_name} not found in registry")
            return 0.0
    
    def calculate_features(self, entity_pairs: List[Tuple[str, str]], string_dict: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Calculate all enabled features for a list of entity pairs.
        
        Args:
            entity_pairs: List of (left_id, right_id) tuples
            string_dict: Optional dictionary mapping IDs to string values (not used in this implementation)
            
        Returns:
            Array of feature values (rows = pairs, columns = features)
        """
        # Get list of features to calculate
        self.feature_names = list(self.feature_registry.keys())
        feature_count = len(self.feature_names)
        
        # Prepare results array
        result = np.zeros((len(entity_pairs), feature_count))
        
        # Calculate features for each pair
        for i, pair in enumerate(entity_pairs):
            for j, feature_name in enumerate(self.feature_names):
                result[i, j] = self._calculate_feature(feature_name, pair)
                
        return result
        
    def compute_features(self, entity_pairs: List[Tuple[str, str, Any]], string_dict: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute features for training.py, which uses a different format than calculate_features.
        
        Args:
            entity_pairs: List of (left_id, right_id, match_label) tuples
            string_dict: Optional dictionary mapping IDs to string values
            
        Returns:
            Tuple of (features, labels) where features is a numpy array of feature values
            and labels is a numpy array of binary labels (1 for match, 0 for non-match)
        """
        # Extract entity IDs and labels
        pairs = [(left_id, right_id) for left_id, right_id, _ in entity_pairs]
        labels = np.array([1 if str(label).lower() == 'true' else 0 
                         for _, _, label in entity_pairs])
        
        # Calculate features
        features = self.calculate_features(pairs, string_dict)
        
        return features, labels
    
    def calculate_features_batch(self, entity_pairs: List[Tuple[str, str]], string_dict: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Calculate features in batches for better performance.
        
        Args:
            entity_pairs: List of (left_id, right_id) tuples
            string_dict: Optional dictionary mapping IDs to string values
            
        Returns:
            Array of feature values (rows = pairs, columns = features)
        """
        # Get list of features to calculate
        self.feature_names = list(self.feature_registry.keys())
        feature_count = len(self.feature_names)
        
        # Prepare results array
        result = np.zeros((len(entity_pairs), feature_count))
        
        # Process in batches
        batch_size = min(self.batch_size, len(entity_pairs))
        num_batches = math.ceil(len(entity_pairs) / batch_size)
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(entity_pairs))
            batch = entity_pairs[start_idx:end_idx]
            
            # Calculate features for this batch
            batch_result = self.calculate_features(batch, string_dict)
            
            # Store in result array
            result[start_idx:end_idx, :] = batch_result
            
        return result
        
    def compute_features_batch(self, entity_pairs: List[Tuple[str, str, Any]], string_dict: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch version of compute_features for training.py.
        
        Args:
            entity_pairs: List of (left_id, right_id, match_label) tuples
            string_dict: Optional dictionary mapping IDs to string values
            
        Returns:
            Tuple of (features, labels) where features is a numpy array of feature values
            and labels is a numpy array of binary labels (1 for match, 0 for non-match)
        """
        # Extract entity IDs and labels
        pairs = [(left_id, right_id) for left_id, right_id, _ in entity_pairs]
        labels = np.array([1 if str(label).lower() == 'true' else 0 
                         for _, _, label in entity_pairs])
        
        # Calculate features in batches
        features = self.calculate_features_batch(pairs, string_dict)
        
        return features, labels
    
    def normalize_features(self, features: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Conditionally normalize features based on configuration.
        
        When enable_feature_scaling=true: Uses LibraryCatalogScaler with domain-specific 
        scaling strategies, feature group-based scaling, and binary feature preservation.
        
        When enable_feature_scaling=false: Returns raw features without any normalization.
        This is the original approach that worked well due to features being naturally 
        well-normalized (mostly 0-1 range).
        
        Args:
            features: Feature array to normalize
            fit: Whether to fit the scaler on this data
            
        Returns:
            Normalized feature array (if scaling enabled) or raw features (if disabled)
        """
        # Early exit for empty input
        if features.size == 0:
            logger.warning("Empty feature array provided to normalize_features")
            return features.copy()
            
        # Check if feature scaling is enabled in configuration
        enable_scaling = self.config.get("use_enhanced_scaling", False)
        if not enable_scaling:
            logger.info("Feature scaling disabled by configuration - returning raw features")
            return features.copy()
            
        # Lazy initialization of scaling bridge
        if not hasattr(self, '_scaling_bridge'):
            try:
                from src.scaling_bridge import ScalingBridge
                # Use scaling_config.yml as the default configuration file
                config_path = "scaling_config.yml"
                self._scaling_bridge = ScalingBridge(config_path)
                self._scaling_bridge.connect(self)
                
                # Try to load a checkpoint in production mode (when fit=False)
                if not fit and self.has_scaler_checkpoint():
                    if self.load_scaler_checkpoint():
                        logger.info("Loaded fitted scaler checkpoint for production consistency")
                    else:
                        logger.warning("Failed to load scaler checkpoint, will fit new scaler")
                        
                logger.info("Initialized ScalingBridge with LibraryCatalogScaler")
            except Exception as e:
                logger.error(f"Failed to initialize ScalingBridge: {e}")
                # Fallback to returning raw features if initialization fails
                logger.warning("Falling back to raw features due to ScalingBridge initialization failure")
                return features.copy()
        
        # Ensure we have valid feature names
        if not hasattr(self, 'feature_names') or not self.feature_names:
            self.feature_names = list(self.feature_registry.keys()) if hasattr(self, 'feature_registry') else []
            
        if not self.feature_names:
            logger.warning("No feature names available for scaling, returning raw features")
            return features.copy()
            
        try:
            # Validate input dimensions
            if len(features.shape) != 2:
                logger.error(f"Expected 2D feature array, got {len(features.shape)}D")
                return features.copy()
                
            if features.shape[1] != len(self.feature_names):
                logger.warning(f"Feature count mismatch: array has {features.shape[1]} columns, "
                              f"expected {len(self.feature_names)} features")
                # Adjust feature names if necessary
                if features.shape[1] < len(self.feature_names):
                    self.feature_names = self.feature_names[:features.shape[1]]
                else:
                    # Pad with generic names if we have more columns than names
                    additional_names = [f"feature_{i}" for i in range(len(self.feature_names), features.shape[1])]
                    self.feature_names.extend(additional_names)
            
            # Initialize or get the LibraryCatalogScaler
            if not hasattr(self._scaling_bridge, 'scaler') or self._scaling_bridge.scaler is None:
                from src.robust_scaler import LibraryCatalogScaler
                self._scaling_bridge.scaler = LibraryCatalogScaler(self.config)
                logger.debug("Initialized LibraryCatalogScaler")
            
            # Apply scaling based on fit parameter
            if fit:
                # Fit and transform
                scaled_features = self._scaling_bridge.scaler.fit_transform(features, self.feature_names)
                logger.debug(f"Fitted and transformed {features.shape[0]} samples with {features.shape[1]} features")
                
                # Save checkpoint after successful fitting for training/production consistency
                checkpoint_path = self.save_scaler_checkpoint()
                if checkpoint_path:
                    logger.info(f"Saved scaler checkpoint for training/production consistency: {checkpoint_path}")
                    
            else:
                # Transform only (production mode)
                if hasattr(self._scaling_bridge.scaler, 'scalers') and self._scaling_bridge.scaler.scalers is not None:
                    # Check if scalers has elements (handle different data types safely)
                    try:
                        has_scalers = len(self._scaling_bridge.scaler.scalers) > 0
                    except (TypeError, AttributeError):
                        # Handle cases where scalers might not support len() or be None
                        has_scalers = bool(self._scaling_bridge.scaler.scalers)
                    
                    if has_scalers:
                        scaled_features = self._scaling_bridge.scaler.transform(features)
                        logger.debug(f"Transformed {features.shape[0]} samples with {features.shape[1]} features")
                    else:
                        # Scaler not fitted yet, need to fit first
                        logger.info("Scaler not fitted yet, performing fit_transform")
                        scaled_features = self._scaling_bridge.scaler.fit_transform(features, self.feature_names)
                else:
                    # Scaler not fitted yet, need to fit first
                    logger.info("Scaler not fitted yet, performing fit_transform")
                    scaled_features = self._scaling_bridge.scaler.fit_transform(features, self.feature_names)
                    
                    # Save checkpoint after successful fitting
                    checkpoint_path = self.save_scaler_checkpoint()
                    if checkpoint_path:
                        logger.info(f"Saved scaler checkpoint after fit_transform: {checkpoint_path}")
            
            # Validate output
            if scaled_features.shape != features.shape:
                logger.error(f"Output shape mismatch: expected {features.shape}, got {scaled_features.shape}")
                return features.copy()
                
            # Additional validation for NaN/inf values
            if np.isnan(scaled_features).any() or np.isinf(scaled_features).any():
                logger.warning("Scaled features contain NaN or infinite values, applying correction")
                scaled_features = np.nan_to_num(scaled_features, nan=0.5, posinf=1.0, neginf=0.0)
            
            # Ensure values are in expected range [0,1] for most features
            scaled_features = np.clip(scaled_features, 0.0, 1.0)
            
            logger.debug(f"Successfully normalized features using LibraryCatalogScaler")
            return scaled_features
            
        except Exception as e:
            logger.error(f"Error in feature normalization: {str(e)}")
            logger.warning("Falling back to raw features due to normalization error")
            return features.copy()
    
    def save_scaler_checkpoint(self, scaler_name: str = "fitted_scaler") -> Optional[str]:
        """
        Save the current fitted scaler to checkpoint for training/production consistency.
        
        Args:
            scaler_name: Name for the scaler checkpoint
            
        Returns:
            Path where scaler was saved, or None if failed
        """
        try:
            # Ensure we have a fitted scaler
            if not hasattr(self, '_scaling_bridge') or not hasattr(self._scaling_bridge, 'scaler'):
                logger.warning("No fitted scaler available to save")
                return None
                
            if not hasattr(self._scaling_bridge.scaler, 'scalers') or not self._scaling_bridge.scaler.scalers:
                logger.warning("Scaler is not fitted, cannot save checkpoint")
                return None
            
            # Get checkpoint manager
            from src.checkpoint_manager import get_checkpoint_manager
            checkpoint_manager = get_checkpoint_manager(self.config)
            
            # Save scaler using checkpoint manager
            saved_path = checkpoint_manager.save_scaler(self._scaling_bridge.scaler, scaler_name)
            logger.info(f"Saved scaler checkpoint for training/production consistency: {saved_path}")
            return saved_path
            
        except Exception as e:
            logger.error(f"Failed to save scaler checkpoint: {str(e)}")
            return None
    
    def load_scaler_checkpoint(self, scaler_name: str = "fitted_scaler") -> bool:
        """
        Load a fitted scaler from checkpoint for production use.
        
        Args:
            scaler_name: Name of the scaler checkpoint to load
            
        Returns:
            True if scaler was successfully loaded
        """
        try:
            # Get checkpoint manager
            from src.checkpoint_manager import get_checkpoint_manager
            checkpoint_manager = get_checkpoint_manager(self.config)
            
            # Check if checkpoint exists
            if not checkpoint_manager.has_scaler_checkpoint(scaler_name):
                logger.warning(f"No scaler checkpoint found: {scaler_name}")
                return False
            
            # Load the scaler
            scaler = checkpoint_manager.load_scaler(self.config, scaler_name)
            
            # Initialize scaling bridge if needed
            if not hasattr(self, '_scaling_bridge'):
                from src.scaling_bridge import ScalingBridge
                config_path = "scaling_config.yml"
                self._scaling_bridge = ScalingBridge(config_path)
                self._scaling_bridge.connect(self)
            
            # Set the loaded scaler
            self._scaling_bridge.scaler = scaler
            
            logger.info(f"Loaded fitted scaler from checkpoint: {scaler_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load scaler checkpoint: {str(e)}")
            return False
    
    def has_scaler_checkpoint(self, scaler_name: str = "fitted_scaler") -> bool:
        """
        Check if a scaler checkpoint exists.
        
        Args:
            scaler_name: Name of the scaler checkpoint
            
        Returns:
            True if checkpoint exists
        """
        try:
            from src.checkpoint_manager import get_checkpoint_manager
            checkpoint_manager = get_checkpoint_manager(self.config)
            return checkpoint_manager.has_scaler_checkpoint(scaler_name)
        except Exception:
            return False
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names.
        
        Returns:
            List of feature names
        """
        return self.feature_names
        
    def compute_features_for_pair(self, left_id: str, right_id: str) -> Dict[str, float]:
        """
        Compute all enabled features for a single entity pair.
        
        Args:
            left_id: ID of the first entity
            right_id: ID of the second entity
            
        Returns:
            Dictionary of feature names to feature values
        """
        # Make sure feature names are initialized
        if not hasattr(self, 'feature_names') or not self.feature_names:
            self.feature_names = list(self.feature_registry.keys())
            
        # Calculate each feature and store in dictionary
        feature_values = {}
        
        try:
            for feature_name in self.feature_names:
                feature_values[feature_name] = self._calculate_feature(feature_name, (left_id, right_id))
                
            return feature_values
        except Exception as e:
            logger.error(f"Error computing features for pair {left_id} - {right_id}: {e}")
            # Return empty dict on error
            return {}
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary of cache statistics
        """
        return {
            "string_cache": self.string_cache.get_stats(),
            "vector_cache": self.vector_cache.get_stats(),
            "similarity_cache": {"size": len(self.similarity_cache)},
            "disable_caching": self.disable_caching,
            "cache_version": self.cache_version
        }
    
    def save_checkpoint(self, file_path: str) -> None:
        """
        Save feature engineering state to checkpoint.
        
        Args:
            file_path: Path to save checkpoint file
        """
        # Create checkpoint manager if needed
        checkpoint_manager = get_checkpoint_manager(self.config)
        
        # Prepare state for saving
        state = {
            "feature_names": self.feature_names,
            "scaler": self.scaler if self.is_fitted else None,
            "is_fitted": self.is_fitted,
            "cache_version": self.cache_version,
            "calculation_issues": self.calculation_issues,
            "field_hash_mapping": self.field_hash_mapping,
            "version_info": {
                "version": _VERSION,
                "build_date": _BUILD_DATE
            }
        }
        
        # Create a simple JSON file to save the state
        import json
        import os
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save state to JSON file
        try:
            with open(file_path, 'w') as f:
                json.dump(state, f, default=lambda o: str(o) if isinstance(o, (np.ndarray, np.number)) else o)
            logger.info(f"Saved feature engineering checkpoint to {file_path}")
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
    
    def load_checkpoint(self, file_path: str) -> None:
        """
        Load feature engineering state from checkpoint.
        
        Args:
            file_path: Path to checkpoint file
        """
        # Create checkpoint manager if needed
        checkpoint_manager = get_checkpoint_manager(self.config)
        
        # Load checkpoint from JSON file
        import json
        try:
            with open(file_path, 'r') as f:
                state = json.load(f)
            logger.info(f"Loaded feature engineering checkpoint from {file_path}")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            state = {}
        
        # Restore state
        self.feature_names = state.get("feature_names", [])
        
        # Restore scaler if present
        scaler = state.get("scaler")
        if scaler is not None:
            self.scaler = scaler
            self.is_fitted = state.get("is_fitted", False)
        
        # Restore cache version if present
        cache_version = state.get("cache_version")
        if cache_version:
            self.cache_version = cache_version
            
        # Restore field hash mapping
        field_hash_mapping = state.get("field_hash_mapping")
        if field_hash_mapping:
            self.field_hash_mapping = field_hash_mapping
            
        # Check version compatibility
        version_info = state.get("version_info", {})
        if version_info.get("version") != _VERSION:
            logger.warning(f"Checkpoint version ({version_info.get('version')}) does not match current version ({_VERSION})")
    
    def get_feature_importance_factors(self) -> Dict[str, float]:
        """
        Get feature importance factors for different features.
        
        These factors are used to give certain features more weight in the model.
        A factor > 1.0 increases importance, < 1.0 decreases importance.
        
        Returns:
            Dictionary mapping feature names to importance factors
        """
        importance_factors = {}
        
        # Taxonomy dissimilarity should be weighted carefully
        # Same domain (dissimilarity=0) doesn't strongly indicate same person
        # Different domains (dissimilarity>0) can indicate different people
        if 'taxonomy_dissimilarity' in self.enabled_features:
            importance_factors['taxonomy_dissimilarity'] = 0.8  # Reduced importance
            
        # Birth/death matches are definitive when they occur
        # Life dates are the most reliable disambiguation feature
        if 'birth_death_match' in self.enabled_features:
            importance_factors['birth_death_match'] = 2.5
            
        logger.info(f"Using feature importance factors: {importance_factors}")
        return importance_factors