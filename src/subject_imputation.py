"""
Subject Imputation Module for Entity Resolution

This module fills missing subject fields using composite field vector join strategy with centroid calculation.
It leverages semantic similarity of composite field content to find appropriate subject values for records
that are missing subject information.
"""

import os
import logging
import pickle
import json
import time
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import defaultdict, Counter
from dataclasses import dataclass

import weaviate
from weaviate.classes.query import MetadataQuery, Filter

from src.preprocessing import load_hash_lookup, load_composite_subject_mapping

logger = logging.getLogger(__name__)

@dataclass
class ImputationResult:
    """Data class for subject imputation results"""
    record_id: str
    composite_hash: str
    imputed_subject_hash: str
    confidence_score: float
    candidate_count: int
    centroid_similarity: float
    frequency_score: float
    alternative_subjects: List[Tuple[str, float]]

class SubjectImputation:
    """
    Fills missing subject fields using composite field semantic similarity and vector join strategy.
    
    This class uses the composite-subject mapping to find semantically similar composite fields,
    then calculates a weighted centroid of their associated subjects to determine the best
    subject value for records missing subject information.
    """
    
    def __init__(self, config: Dict[str, Any], weaviate_client):
        """
        Initialize the Subject Imputation system.
        
        Args:
            config: Configuration dictionary containing thresholds and parameters
            weaviate_client: Connected Weaviate client for vector operations
        """
        self.config = config
        self.weaviate_client = weaviate_client
        
        # Verify Weaviate collection exists
        try:
            self.collection = weaviate_client.collections.get("EntityString")
            # Test collection accessibility
            test_result = self.collection.query.fetch_objects(limit=1)
            logger.debug("Successfully connected to Weaviate EntityString collection")
        except Exception as e:
            logger.error(f"Failed to connect to Weaviate EntityString collection: {e}")
            raise RuntimeError(f"Subject imputation requires Weaviate collection 'EntityString' to be available. "
                             f"Please run embedding_and_indexing stage first. Error: {e}")
        
        # Imputation configuration with defaults and validation
        imputation_config = config.get('subject_imputation', {})
        self.similarity_threshold = max(0.0, min(1.0, imputation_config.get('similarity_threshold', 0.65)))
        self.confidence_threshold = max(0.0, min(1.0, imputation_config.get('confidence_threshold', 0.70)))
        self.min_candidates = max(1, imputation_config.get('min_candidates', 3))
        self.max_candidates = max(10, min(1000, imputation_config.get('max_candidates', 150)))
        self.frequency_weight = max(0.0, min(1.0, imputation_config.get('frequency_weight', 0.3)))
        self.centroid_weight = max(0.0, min(1.0, imputation_config.get('centroid_weight', 0.7)))
        self.use_caching = imputation_config.get('use_caching', True)
        self.cache_size_limit = max(100, min(100000, imputation_config.get('cache_size_limit', 10000)))
        
        # Ensure weights sum to reasonable value
        total_weight = self.frequency_weight + self.centroid_weight
        if total_weight <= 0:
            logger.warning("Invalid weight configuration, using defaults")
            self.frequency_weight = 0.3
            self.centroid_weight = 0.7
        
        # Data storage
        self.hash_lookup = None
        self.composite_subject_mapping = None
        self.string_dict = None
        self.string_counts = None
        self.imputation_results = {}
        self.imputation_cache = {}  # Cache for imputation results
        self.vector_cache = {}      # Cache for vectors
        
        logger.info(f"Initialized SubjectImputation with similarity_threshold={self.similarity_threshold}")
    
    def _load_data(self) -> None:
        """Load required data structures from preprocessing checkpoints."""
        checkpoint_dir = self.config.get('checkpoint_dir', 'data/checkpoints')
        
        # Load hash lookup
        hash_lookup_path = os.path.join(checkpoint_dir, 'hash_lookup.pkl')
        if not os.path.exists(hash_lookup_path):
            raise FileNotFoundError(f"Hash lookup not found at {hash_lookup_path}")
        
        self.hash_lookup = load_hash_lookup(hash_lookup_path)
        logger.info(f"Loaded hash_lookup with {len(self.hash_lookup)} entities")
        
        # Load composite-subject mapping
        mapping_path = os.path.join(checkpoint_dir, 'composite_subject_mapping.pkl')
        if not os.path.exists(mapping_path):
            raise FileNotFoundError(f"Composite-subject mapping not found at {mapping_path}")
        
        self.composite_subject_mapping = load_composite_subject_mapping(mapping_path)
        logger.info(f"Loaded composite_subject_mapping with {len(self.composite_subject_mapping)} entries")
        
        # Load string dictionary and counts for frequency scoring
        try:
            string_dict_path = os.path.join(checkpoint_dir, 'string_dict.pkl')
            with open(string_dict_path, 'rb') as f:
                self.string_dict = pickle.load(f)
            logger.info(f"Loaded string_dict with {len(self.string_dict)} entries")
            
            string_counts_path = os.path.join(checkpoint_dir, 'string_counts.pkl')
            with open(string_counts_path, 'rb') as f:
                self.string_counts = pickle.load(f)
            logger.info(f"Loaded string_counts with {len(self.string_counts)} entries")
            
        except Exception as e:
            logger.warning(f"Could not load string dictionary/counts: {e}")
            self.string_dict = {}
            self.string_counts = {}
        
        # Load existing imputation cache if available
        if self.use_caching:
            self._load_imputation_cache()
    
    def _load_imputation_cache(self) -> None:
        """Load imputation cache from checkpoint if available."""
        try:
            checkpoint_dir = self.config.get('checkpoint_dir', 'data/checkpoints')
            cache_dir = os.path.join(checkpoint_dir, 'cache')
            cache_path = os.path.join(cache_dir, 'imputation_cache.pkl')
            
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    self.imputation_cache = pickle.load(f)
                logger.info(f"Loaded imputation cache with {len(self.imputation_cache)} entries")
            else:
                logger.info("No existing imputation cache found, starting fresh")
                
        except Exception as e:
            logger.warning(f"Error loading imputation cache: {e}")
            self.imputation_cache = {}
    
    def _save_imputation_cache(self) -> None:
        """Save imputation cache to checkpoint."""
        if not self.use_caching:
            return
        
        try:
            checkpoint_dir = self.config.get('checkpoint_dir', 'data/checkpoints')
            cache_dir = os.path.join(checkpoint_dir, 'cache')
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, 'imputation_cache.pkl')
            
            # Limit cache size to prevent excessive memory usage
            if len(self.imputation_cache) > self.cache_size_limit:
                # Keep only the most recent entries
                cache_items = list(self.imputation_cache.items())
                self.imputation_cache = dict(cache_items[-self.cache_size_limit:])
            
            with open(cache_path, 'wb') as f:
                pickle.dump(self.imputation_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Saved imputation cache with {len(self.imputation_cache)} entries")
            
        except Exception as e:
            logger.error(f"Error saving imputation cache: {e}")
    
    def get_vector(self, hash_value: str, field_type: str) -> Optional[np.ndarray]:
        """
        Retrieve vector from Weaviate for given hash and field type with caching.
        
        Args:
            hash_value: Hash value of the string
            field_type: Type of field (e.g., 'composite', 'subjects')
            
        Returns:
            Vector as numpy array, or None if not found
        """
        if not hash_value or hash_value == "NULL":
            return None
        
        # Check cache first
        cache_key = f"{hash_value}_{field_type}"
        if cache_key in self.vector_cache:
            return self.vector_cache[cache_key]
        
        try:
            # Query Weaviate for the specific hash and field type
            result = self.collection.query.fetch_objects(
                where=Filter.by_property("hash_value").equal(hash_value) & 
                      Filter.by_property("field_type").equal(field_type),
                limit=1,
                include_vector=True
            )
            
            vector = None
            if result.objects:
                vector = np.array(result.objects[0].vector['default'])
            
            # Cache the result (even if None)
            if len(self.vector_cache) < 5000:  # Limit vector cache size
                self.vector_cache[cache_key] = vector
            
            return vector
                
        except Exception as e:
            logger.error(f"Error retrieving vector for {hash_value} ({field_type}): {e}")
            return None
    
    def _identify_records_for_imputation(self) -> List[Tuple[str, Dict[str, str]]]:
        """
        Identify records that need subject imputation.
        
        Returns:
            List of (record_id, record_data) tuples for records needing imputation
        """
        records_for_imputation = []
        
        for record_id, record_data in self.hash_lookup.items():
            composite_hash = record_data.get('composite')
            subject_hash = record_data.get('subjects')
            
            # Include records that have composite but missing or NULL subjects
            # Skip records that have been subject to quality audit remediation
            if (composite_hash and composite_hash != "NULL" and 
                (not subject_hash or subject_hash == "NULL") and
                not record_data.get('subject_remediated', False)):
                
                records_for_imputation.append((record_id, record_data))
        
        logger.info(f"Identified {len(records_for_imputation)} records needing subject imputation")
        return records_for_imputation
    
    def impute_subject_for_record(self, record_id: str, record_data: Dict[str, str]) -> Optional[ImputationResult]:
        """
        Impute subject for a single record using composite field vector join strategy.
        
        Args:
            record_id: Unique identifier for the record
            record_data: Dictionary containing field hashes for the record
            
        Returns:
            ImputationResult object with imputation details, or None if imputation not possible
        """
        composite_hash = record_data.get('composite')
        
        if not composite_hash or composite_hash == "NULL":
            return None
        
        # Check cache first
        if self.use_caching and composite_hash in self.imputation_cache:
            cached_result = self.imputation_cache[composite_hash]
            logger.debug(f"Using cached imputation for composite {composite_hash}")
            return ImputationResult(
                record_id=record_id,
                composite_hash=composite_hash,
                imputed_subject_hash=cached_result['subject_hash'],
                confidence_score=cached_result['confidence'],
                candidate_count=cached_result['candidate_count'],
                centroid_similarity=cached_result['centroid_similarity'],
                frequency_score=cached_result['frequency_score'],
                alternative_subjects=cached_result['alternatives']
            )
        
        try:
            # Get vector for the composite field
            composite_vector = self.get_vector(composite_hash, 'composite')
            if composite_vector is None:
                logger.debug(f"No vector found for composite hash {composite_hash}")
                return None
            
            # Query Weaviate for similar composite fields
            similar_results = self.collection.query.near_vector(
                near_vector=composite_vector,
                limit=self.max_candidates,
                where=Filter.by_property("field_type").equal("composite"),
                return_metadata=MetadataQuery(distance=True),
                include_vector=False
            )
            
            # Collect candidate subjects from similar composites
            candidate_subjects = []
            composite_similarities = []
            
            for result in similar_results.objects:
                if not hasattr(result, 'metadata') or not result.metadata.distance:
                    continue
                
                # Convert distance to similarity (cosine distance -> cosine similarity)
                similarity = 1.0 - result.metadata.distance
                
                # Skip if below threshold or if it's the same composite
                if similarity < self.similarity_threshold:
                    continue
                
                similar_composite_hash = result.properties.get('hash_value')
                if similar_composite_hash == composite_hash:
                    continue  # Skip the original composite
                
                # Look up associated subject using preprocessed mapping
                if similar_composite_hash in self.composite_subject_mapping:
                    subject_hash = self.composite_subject_mapping[similar_composite_hash]
                    
                    # Only include records that have subjects
                    if subject_hash and subject_hash != "NULL":
                        candidate_subjects.append(subject_hash)
                        composite_similarities.append(similarity)
            
            if len(candidate_subjects) < self.min_candidates:
                logger.debug(f"Not enough candidate subjects ({len(candidate_subjects)}) for record {record_id}")
                return None
            
            # Get vectors for all candidate subjects
            subject_vectors = []
            valid_similarities = []
            valid_subjects = []
            
            for i, subject_hash in enumerate(candidate_subjects):
                subject_vector = self.get_vector(subject_hash, 'subjects')
                if subject_vector is not None:
                    subject_vectors.append(subject_vector)
                    valid_similarities.append(composite_similarities[i])
                    valid_subjects.append(subject_hash)
            
            if not subject_vectors:
                logger.debug(f"No valid subject vectors found for record {record_id}")
                return None
            
            # Calculate weighted centroid based on composite similarity scores
            weights = np.array(valid_similarities)
            weights = weights / np.sum(weights)  # Normalize weights
            
            subject_vectors_array = np.array(subject_vectors)
            centroid_vector = np.average(subject_vectors_array, axis=0, weights=weights)
            
            # Find subject vector closest to centroid
            best_subject_hash = None
            best_similarity = -1.0
            subject_similarities = []
            
            for i, subject_vector in enumerate(subject_vectors):
                centroid_similarity = np.dot(subject_vector, centroid_vector) / (
                    np.linalg.norm(subject_vector) * np.linalg.norm(centroid_vector)
                )
                
                subject_similarities.append((valid_subjects[i], centroid_similarity))
                
                if centroid_similarity > best_similarity:
                    best_similarity = centroid_similarity
                    best_subject_hash = valid_subjects[i]
            
            if best_subject_hash is None:
                return None
            
            # Calculate frequency score for the best subject
            frequency_score = 0.0
            if best_subject_hash in self.string_counts:
                max_frequency = max(self.string_counts.values()) if self.string_counts else 1
                frequency_score = np.log(self.string_counts[best_subject_hash] + 1) / np.log(max_frequency + 1)
            
            # Calculate overall confidence score
            confidence_score = (
                self.centroid_weight * best_similarity + 
                self.frequency_weight * frequency_score
            )
            
            # Sort alternative subjects by similarity to centroid
            subject_similarities.sort(key=lambda x: x[1], reverse=True)
            
            result = ImputationResult(
                record_id=record_id,
                composite_hash=composite_hash,
                imputed_subject_hash=best_subject_hash,
                confidence_score=confidence_score,
                candidate_count=len(candidate_subjects),
                centroid_similarity=best_similarity,
                frequency_score=frequency_score,
                alternative_subjects=subject_similarities[:10]  # Keep top 10 alternatives
            )
            
            # Cache the result if caching is enabled
            if self.use_caching:
                self.imputation_cache[composite_hash] = {
                    'subject_hash': best_subject_hash,
                    'confidence': confidence_score,
                    'candidate_count': len(candidate_subjects),
                    'centroid_similarity': best_similarity,
                    'frequency_score': frequency_score,
                    'alternatives': subject_similarities[:10]
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error imputing subject for record {record_id}: {e}")
            return None
    
    def execute(self) -> Dict[str, Any]:
        """
        Execute subject imputation for all records needing subjects.
        
        Returns:
            Dictionary with imputation results and statistics
        """
        logger.info("Starting subject imputation")
        start_time = time.time()
        
        # Load required data
        self._load_data()
        
        # Identify records needing imputation
        records_for_imputation = self._identify_records_for_imputation()
        
        if not records_for_imputation:
            logger.info("No records found needing subject imputation")
            return {
                'status': 'completed',
                'total_candidates': 0,
                'successful_imputations': 0,
                'high_confidence_imputations': 0,
                'elapsed_time': time.time() - start_time
            }
        
        # Track statistics
        successful_imputations = 0
        high_confidence_imputations = 0
        total_processed = 0
        
        # Process each record
        for record_id, record_data in records_for_imputation:
            imputation_result = self.impute_subject_for_record(record_id, record_data)
            
            if imputation_result is not None:
                successful_imputations += 1
                self.imputation_results[record_id] = imputation_result
                
                # Update hash_lookup with imputed subject if confidence is high enough
                if imputation_result.confidence_score >= self.confidence_threshold:
                    high_confidence_imputations += 1
                    
                    # Update the record data
                    record_data['subjects'] = imputation_result.imputed_subject_hash
                    record_data['imputed_subject'] = True
                    record_data['imputation_confidence'] = imputation_result.confidence_score
                    record_data['imputation_alternatives'] = len(imputation_result.alternative_subjects)
                    
                    logger.debug(f"Imputed subject for record {record_id}: "
                               f"{imputation_result.imputed_subject_hash} "
                               f"(confidence: {imputation_result.confidence_score:.3f})")
                else:
                    # Mark as attempted but low confidence
                    record_data['imputation_attempted'] = True
                    record_data['imputation_confidence'] = imputation_result.confidence_score
            
            total_processed += 1
            
            if total_processed % 1000 == 0:
                logger.info(f"Processed {total_processed}/{len(records_for_imputation)} records, "
                           f"{successful_imputations} successful, {high_confidence_imputations} high-confidence")
        
        # Save updated hash_lookup and composite-subject mapping if we made changes
        if high_confidence_imputations > 0:
            checkpoint_dir = self.config.get('checkpoint_dir', 'data/checkpoints')
            hash_lookup_path = os.path.join(checkpoint_dir, 'hash_lookup.pkl')
            
            try:
                with open(hash_lookup_path, 'wb') as f:
                    pickle.dump(self.hash_lookup, f, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(f"Saved updated hash_lookup with {high_confidence_imputations} imputed subjects")
                
                # Update composite-subject mapping with newly imputed subjects
                for record_id, imputation_result in self.imputation_results.items():
                    if (imputation_result.confidence_score >= self.confidence_threshold and 
                        imputation_result.composite_hash in self.composite_subject_mapping):
                        self.composite_subject_mapping[imputation_result.composite_hash] = imputation_result.imputed_subject_hash
                
                # Save updated composite-subject mapping
                from src.preprocessing import save_composite_subject_mapping
                save_composite_subject_mapping(self.composite_subject_mapping, checkpoint_dir)
                logger.info(f"Updated composite-subject mapping with {high_confidence_imputations} new mappings")
                
            except Exception as e:
                logger.error(f"Error saving updated data structures: {e}")
        
        # Save imputation cache
        self._save_imputation_cache()
        
        elapsed_time = time.time() - start_time
        
        # Generate statistics and save results
        statistics = self.generate_statistics()
        output_results = self._save_imputation_results(statistics)
        
        results = {
            'status': 'completed',
            'elapsed_time': elapsed_time,
            'total_candidates': len(records_for_imputation),
            'successful_imputations': successful_imputations,
            'high_confidence_imputations': high_confidence_imputations,
            'success_rate': successful_imputations / len(records_for_imputation),
            'high_confidence_rate': high_confidence_imputations / len(records_for_imputation),
            'statistics': statistics,
            'output_saved': output_results
        }
        
        logger.info(f"Subject imputation completed in {elapsed_time:.2f} seconds")
        logger.info(f"Successfully imputed {successful_imputations}/{len(records_for_imputation)} subjects")
        logger.info(f"High-confidence imputations: {high_confidence_imputations} ({100*high_confidence_imputations/len(records_for_imputation):.1f}%)")
        
        return results
    
    def generate_statistics(self) -> Dict[str, Any]:
        """
        Generate performance statistics for the imputation process.
        
        Returns:
            Dictionary with detailed statistics
        """
        if not self.imputation_results:
            return {'no_results': True}
        
        # Extract metrics
        confidence_scores = [result.confidence_score for result in self.imputation_results.values()]
        centroid_similarities = [result.centroid_similarity for result in self.imputation_results.values()]
        frequency_scores = [result.frequency_score for result in self.imputation_results.values()]
        candidate_counts = [result.candidate_count for result in self.imputation_results.values()]
        
        # Subject frequency analysis
        imputed_subjects = [result.imputed_subject_hash for result in self.imputation_results.values()]
        subject_frequency = Counter(imputed_subjects)
        
        statistics = {
            'total_imputations': len(self.imputation_results),
            'confidence_score_stats': {
                'mean': float(np.mean(confidence_scores)),
                'std': float(np.std(confidence_scores)),
                'min': float(np.min(confidence_scores)),
                'max': float(np.max(confidence_scores)),
                'median': float(np.median(confidence_scores))
            },
            'centroid_similarity_stats': {
                'mean': float(np.mean(centroid_similarities)),
                'std': float(np.std(centroid_similarities)),
                'min': float(np.min(centroid_similarities)),
                'max': float(np.max(centroid_similarities)),
                'median': float(np.median(centroid_similarities))
            },
            'frequency_score_stats': {
                'mean': float(np.mean(frequency_scores)),
                'std': float(np.std(frequency_scores)),
                'min': float(np.min(frequency_scores)),
                'max': float(np.max(frequency_scores)),
                'median': float(np.median(frequency_scores))
            },
            'candidate_count_stats': {
                'mean': float(np.mean(candidate_counts)),
                'std': float(np.std(candidate_counts)),
                'min': int(np.min(candidate_counts)),
                'max': int(np.max(candidate_counts)),
                'median': float(np.median(candidate_counts))
            },
            'most_common_subjects': subject_frequency.most_common(10),
            'high_confidence_count': sum(1 for score in confidence_scores if score >= self.confidence_threshold),
            'cache_hit_rate': len(self.imputation_cache) / max(len(self.imputation_results), 1) if self.use_caching else 0.0
        }
        
        return statistics
    
    def _save_imputation_results(self, statistics: Dict[str, Any]) -> str:
        """Save detailed imputation results to file."""
        try:
            output_dir = self.config.get('output_dir', 'data/output')
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            results_path = os.path.join(output_dir, f'subject_imputation_results_{timestamp}.json')
            
            # Prepare results for JSON serialization
            serializable_results = {}
            for record_id, result in self.imputation_results.items():
                serializable_results[record_id] = {
                    'composite_hash': result.composite_hash,
                    'imputed_subject_hash': result.imputed_subject_hash,
                    'confidence_score': result.confidence_score,
                    'candidate_count': result.candidate_count,
                    'centroid_similarity': result.centroid_similarity,
                    'frequency_score': result.frequency_score,
                    'alternative_subjects': [(subj, float(sim)) for subj, sim in result.alternative_subjects[:5]]
                }
            
            # Configuration used
            config_info = {
                'similarity_threshold': self.similarity_threshold,
                'confidence_threshold': self.confidence_threshold,
                'min_candidates': self.min_candidates,
                'max_candidates': self.max_candidates,
                'frequency_weight': self.frequency_weight,
                'centroid_weight': self.centroid_weight
            }
            
            output_data = {
                'timestamp': timestamp,
                'configuration': config_info,
                'statistics': statistics,
                'imputation_results': serializable_results
            }
            
            with open(results_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            logger.info(f"Subject imputation results saved to {results_path}")
            return results_path
            
        except Exception as e:
            logger.error(f"Error saving imputation results: {e}")
            return ""