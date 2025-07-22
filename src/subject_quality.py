"""
Subject Quality Audit Module for Entity Resolution

This module evaluates existing subject field quality using composite field vector similarity analysis.
It automatically identifies and corrects low-quality subject assignments by finding better alternatives
through semantic similarity of composite field content.
"""

import os
import logging
import pickle
import json
import time
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import defaultdict
from dataclasses import dataclass

import weaviate
from weaviate.classes.query import MetadataQuery, Filter

from src.preprocessing import load_hash_lookup, load_composite_subject_mapping

logger = logging.getLogger(__name__)

@dataclass
class QualityScore:
    """Data class for subject quality assessment results"""
    original_subject_hash: str
    alternative_subject_hash: Optional[str]
    similarity_score: float
    frequency_score: float
    composite_score: float
    overall_score: float
    needs_remediation: bool
    confidence: float

class SubjectQualityAudit:
    """
    Evaluates and improves subject field quality using composite field semantic similarity.
    
    This class automatically identifies subject assignments that could be improved by finding
    semantically similar composite fields and their associated subjects.
    """
    
    def __init__(self, config: Dict[str, Any], weaviate_client):
        """
        Initialize the Subject Quality Audit system.
        
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
            raise RuntimeError(f"Subject quality audit requires Weaviate collection 'EntityString' to be available. "
                             f"Please run embedding_and_indexing stage first. Error: {e}")
        
        # Quality audit configuration with defaults and validation
        quality_config = config.get('subject_quality_audit', {})
        self.similarity_threshold = max(0.0, min(1.0, quality_config.get('similarity_threshold', 0.70)))
        self.remediation_threshold = max(0.0, min(1.0, quality_config.get('remediation_threshold', 0.60)))
        self.min_alternatives = max(1, quality_config.get('min_alternatives', 3))
        self.max_candidates = max(10, min(1000, quality_config.get('max_candidates', 100)))
        self.frequency_weight = max(0.0, min(1.0, quality_config.get('frequency_weight', 0.3)))
        self.similarity_weight = max(0.0, min(1.0, quality_config.get('similarity_weight', 0.7)))
        self.auto_remediate = quality_config.get('auto_remediate', True)
        self.confidence_threshold = max(0.0, min(1.0, quality_config.get('confidence_threshold', 0.80)))
        
        # Ensure weights sum to reasonable value
        total_weight = self.frequency_weight + self.similarity_weight
        if total_weight <= 0:
            logger.warning("Invalid weight configuration, using defaults")
            self.frequency_weight = 0.3
            self.similarity_weight = 0.7
        
        # Data storage
        self.hash_lookup = None
        self.composite_subject_mapping = None
        self.string_dict = None
        self.string_counts = None
        self.quality_results = {}
        self.remediation_count = 0
        
        logger.info(f"Initialized SubjectQualityAudit with similarity_threshold={self.similarity_threshold}")
    
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
    
    def get_vector(self, hash_value: str, field_type: str) -> Optional[np.ndarray]:
        """
        Retrieve vector from Weaviate for given hash and field type.
        
        Args:
            hash_value: Hash value of the string
            field_type: Type of field (e.g., 'composite', 'subjects')
            
        Returns:
            Vector as numpy array, or None if not found
        """
        if not hash_value or hash_value == "NULL":
            return None
        
        try:
            # Query Weaviate for the specific hash and field type
            result = self.collection.query.fetch_objects(
                where=Filter.by_property("hash_value").equal(hash_value) & 
                      Filter.by_property("field_type").equal(field_type),
                limit=1,
                include_vector=True
            )
            
            if result.objects:
                vector = result.objects[0].vector['default']
                return np.array(vector)
            else:
                logger.debug(f"No vector found for hash {hash_value} with field_type {field_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving vector for {hash_value} ({field_type}): {e}")
            return None
    
    def _find_alternative_subjects(self, composite_hash: str) -> List[Tuple[str, float]]:
        """
        Find alternative subjects using composite field similarity and preprocessed mapping.
        
        Args:
            composite_hash: Hash of the composite field to find alternatives for
            
        Returns:
            List of (subject_hash, similarity_score) tuples for alternative subjects
        """
        if not composite_hash or composite_hash == "NULL":
            return []
        
        try:
            # Get vector for the composite field
            composite_vector = self.get_vector(composite_hash, 'composite')
            if composite_vector is None:
                logger.debug(f"No vector found for composite hash {composite_hash}")
                return []
            
            # Query Weaviate for similar composite fields
            similar_results = self.collection.query.near_vector(
                near_vector=composite_vector,
                limit=self.max_candidates,
                where=Filter.by_property("field_type").equal("composite"),
                return_metadata=MetadataQuery(distance=True),
                include_vector=False
            )
            
            alternative_subjects = []
            
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
                        alternative_subjects.append((subject_hash, similarity))
            
            # Sort by similarity score descending
            alternative_subjects.sort(key=lambda x: x[1], reverse=True)
            
            logger.debug(f"Found {len(alternative_subjects)} alternative subjects for composite {composite_hash}")
            return alternative_subjects
            
        except Exception as e:
            logger.error(f"Error finding alternative subjects for {composite_hash}: {e}")
            return []
    
    def evaluate_subject_quality(self, record_id: str, record_data: Dict[str, str]) -> Optional[QualityScore]:
        """
        Evaluate quality of existing subject assignment for a record.
        
        Args:
            record_id: Unique identifier for the record
            record_data: Dictionary containing field hashes for the record
            
        Returns:
            QualityScore object with assessment results, or None if evaluation not possible
        """
        composite_hash = record_data.get('composite')
        original_subject_hash = record_data.get('subjects')
        
        # Only evaluate records that have both composite and subjects
        if not composite_hash or composite_hash == "NULL":
            return None
        if not original_subject_hash or original_subject_hash == "NULL":
            return None
        
        try:
            # Find alternative subjects based on composite similarity
            alternatives = self._find_alternative_subjects(composite_hash)
            
            if len(alternatives) < self.min_alternatives:
                logger.debug(f"Not enough alternatives ({len(alternatives)}) for record {record_id}")
                return None
            
            # Get vector for original subject
            original_subject_vector = self.get_vector(original_subject_hash, 'subjects')
            if original_subject_vector is None:
                logger.debug(f"No vector found for original subject {original_subject_hash}")
                return None
            
            # Calculate weighted centroid of alternative subjects
            alternative_vectors = []
            weights = []
            
            for subject_hash, similarity in alternatives:
                subject_vector = self.get_vector(subject_hash, 'subjects')
                if subject_vector is not None:
                    alternative_vectors.append(subject_vector)
                    weights.append(similarity)
            
            if not alternative_vectors:
                logger.debug(f"No alternative subject vectors found for record {record_id}")
                return None
            
            # Calculate weighted centroid
            weights = np.array(weights)
            weights = weights / np.sum(weights)  # Normalize weights
            
            alternative_vectors_array = np.array(alternative_vectors)
            centroid_vector = np.average(alternative_vectors_array, axis=0, weights=weights)
            
            # Calculate similarity between original subject and centroid
            similarity_score = np.dot(original_subject_vector, centroid_vector) / (
                np.linalg.norm(original_subject_vector) * np.linalg.norm(centroid_vector)
            )
            
            # Calculate frequency score (higher frequency = better)
            frequency_score = 0.0
            if original_subject_hash in self.string_counts:
                # Normalize frequency score (log scale to prevent dominance)
                max_frequency = max(self.string_counts.values()) if self.string_counts else 1
                frequency_score = np.log(self.string_counts[original_subject_hash] + 1) / np.log(max_frequency + 1)
            
            # Calculate composite score of alternatives (average similarity to composite)
            composite_score = np.mean([sim for _, sim in alternatives])
            
            # Calculate overall quality score
            overall_score = (
                self.similarity_weight * similarity_score + 
                self.frequency_weight * frequency_score
            )
            
            # Determine if remediation is needed and find best alternative
            needs_remediation = overall_score < self.remediation_threshold
            best_alternative = None
            confidence = overall_score
            
            if needs_remediation:
                # Find the most frequent alternative subject that's similar to centroid
                alternative_scores = []
                
                for subject_hash, _ in alternatives:
                    subject_vector = self.get_vector(subject_hash, 'subjects')
                    if subject_vector is not None:
                        alt_similarity = np.dot(subject_vector, centroid_vector) / (
                            np.linalg.norm(subject_vector) * np.linalg.norm(centroid_vector)
                        )
                        alt_frequency = 0.0
                        if subject_hash in self.string_counts:
                            alt_frequency = np.log(self.string_counts[subject_hash] + 1) / np.log(max_frequency + 1)
                        
                        alt_score = self.similarity_weight * alt_similarity + self.frequency_weight * alt_frequency
                        alternative_scores.append((subject_hash, alt_score))
                
                if alternative_scores:
                    alternative_scores.sort(key=lambda x: x[1], reverse=True)
                    best_alternative = alternative_scores[0][0]
                    confidence = alternative_scores[0][1]
            
            return QualityScore(
                original_subject_hash=original_subject_hash,
                alternative_subject_hash=best_alternative,
                similarity_score=similarity_score,
                frequency_score=frequency_score,
                composite_score=composite_score,
                overall_score=overall_score,
                needs_remediation=needs_remediation,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error evaluating subject quality for record {record_id}: {e}")
            return None
    
    def execute(self) -> Dict[str, Any]:
        """
        Execute subject quality audit for all records with existing subjects.
        
        Returns:
            Dictionary with audit results and statistics
        """
        logger.info("Starting subject quality audit")
        start_time = time.time()
        
        # Load required data
        self._load_data()
        
        # Track statistics
        total_evaluated = 0
        needs_remediation = 0
        successful_evaluations = 0
        
        # Process each record with subjects
        for record_id, record_data in self.hash_lookup.items():
            subject_hash = record_data.get('subjects')
            
            # Only evaluate records that have subjects
            if not subject_hash or subject_hash == "NULL":
                continue
            
            quality_score = self.evaluate_subject_quality(record_id, record_data)
            
            if quality_score is not None:
                successful_evaluations += 1
                self.quality_results[record_id] = quality_score
                
                if quality_score.needs_remediation:
                    needs_remediation += 1
                    
                    # Mark for remediation in hash_lookup
                    if 'subject_evaluated' not in record_data:
                        record_data['subject_evaluated'] = True
                    record_data['subject_remediation_required'] = True
                    if quality_score.alternative_subject_hash:
                        record_data['alternative_subject_hash'] = quality_score.alternative_subject_hash
                else:
                    # Mark as evaluated but not needing remediation
                    if 'subject_evaluated' not in record_data:
                        record_data['subject_evaluated'] = True
                    record_data['subject_remediation_required'] = False
            
            total_evaluated += 1
            
            if total_evaluated % 1000 == 0:
                logger.info(f"Evaluated {total_evaluated} records, {needs_remediation} need remediation")
        
        # Perform automatic remediation if enabled
        if self.auto_remediate:
            remediation_results = self.perform_remediation()
        else:
            remediation_results = {'auto_remediation_disabled': True}
        
        elapsed_time = time.time() - start_time
        
        # Generate audit report
        report = self._generate_audit_report()
        
        results = {
            'status': 'completed',
            'elapsed_time': elapsed_time,
            'total_evaluated': total_evaluated,
            'successful_evaluations': successful_evaluations,
            'needs_remediation': needs_remediation,
            'remediation_rate': needs_remediation / max(successful_evaluations, 1),
            'remediation_results': remediation_results,
            'report_saved': report
        }
        
        logger.info(f"Subject quality audit completed in {elapsed_time:.2f} seconds")
        logger.info(f"Evaluated {successful_evaluations}/{total_evaluated} records")
        logger.info(f"Found {needs_remediation} records needing remediation ({100*needs_remediation/max(successful_evaluations,1):.1f}%)")
        
        return results
    
    def perform_remediation(self) -> Dict[str, Any]:
        """
        Automatically apply high-confidence subject improvements.
        
        Returns:
            Dictionary with remediation results
        """
        logger.info("Starting automatic subject remediation")
        
        remediated_count = 0
        high_confidence_count = 0
        
        for record_id, quality_score in self.quality_results.items():
            if (quality_score.needs_remediation and 
                quality_score.alternative_subject_hash and 
                quality_score.confidence >= self.confidence_threshold):
                
                high_confidence_count += 1
                
                # Update hash_lookup with better subject
                if record_id in self.hash_lookup:
                    original_subject = self.hash_lookup[record_id]['subjects']
                    self.hash_lookup[record_id]['subjects'] = quality_score.alternative_subject_hash
                    self.hash_lookup[record_id]['subject_remediated'] = True
                    self.hash_lookup[record_id]['original_subject_hash'] = original_subject
                    
                    remediated_count += 1
                    
                    logger.debug(f"Remediated subject for record {record_id}: "
                               f"{original_subject} -> {quality_score.alternative_subject_hash} "
                               f"(confidence: {quality_score.confidence:.3f})")
        
        # Save updated hash_lookup and composite-subject mapping
        if remediated_count > 0:
            checkpoint_dir = self.config.get('checkpoint_dir', 'data/checkpoints')
            hash_lookup_path = os.path.join(checkpoint_dir, 'hash_lookup.pkl')
            
            try:
                with open(hash_lookup_path, 'wb') as f:
                    pickle.dump(self.hash_lookup, f, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(f"Saved updated hash_lookup with {remediated_count} remediated subjects")
                
                # Update composite-subject mapping with remediated subjects
                for record_id, quality_score in self.quality_results.items():
                    if (quality_score.needs_remediation and quality_score.alternative_subject_hash and 
                        quality_score.confidence >= self.confidence_threshold and record_id in self.hash_lookup):
                        
                        composite_hash = self.hash_lookup[record_id].get('composite')
                        if composite_hash and composite_hash in self.composite_subject_mapping:
                            self.composite_subject_mapping[composite_hash] = quality_score.alternative_subject_hash
                
                # Save updated composite-subject mapping
                from src.preprocessing import save_composite_subject_mapping
                save_composite_subject_mapping(self.composite_subject_mapping, checkpoint_dir)
                logger.info(f"Updated composite-subject mapping with {remediated_count} remediated mappings")
                
            except Exception as e:
                logger.error(f"Error saving updated data structures: {e}")
        
        self.remediation_count = remediated_count
        
        logger.info(f"Automatic remediation completed: {remediated_count}/{high_confidence_count} high-confidence subjects improved")
        
        return {
            'high_confidence_candidates': high_confidence_count,
            'remediated_subjects': remediated_count,
            'success_rate': remediated_count / max(high_confidence_count, 1)
        }
    
    def _generate_audit_report(self) -> str:
        """Generate detailed audit report and save to file."""
        try:
            output_dir = self.config.get('output_dir', 'data/output')
            reports_dir = os.path.join(output_dir, 'reports')
            os.makedirs(reports_dir, exist_ok=True)
            
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            report_path = os.path.join(reports_dir, f'subject_quality_audit_{timestamp}.json')
            
            # Aggregate statistics
            if self.quality_results:
                similarity_scores = [qs.similarity_score for qs in self.quality_results.values()]
                frequency_scores = [qs.frequency_score for qs in self.quality_results.values()]
                overall_scores = [qs.overall_score for qs in self.quality_results.values()]
                
                statistics = {
                    'total_records_evaluated': len(self.quality_results),
                    'records_needing_remediation': sum(1 for qs in self.quality_results.values() if qs.needs_remediation),
                    'similarity_score_stats': {
                        'mean': float(np.mean(similarity_scores)),
                        'std': float(np.std(similarity_scores)),
                        'min': float(np.min(similarity_scores)),
                        'max': float(np.max(similarity_scores))
                    },
                    'frequency_score_stats': {
                        'mean': float(np.mean(frequency_scores)),
                        'std': float(np.std(frequency_scores)),
                        'min': float(np.min(frequency_scores)),
                        'max': float(np.max(frequency_scores))
                    },
                    'overall_score_stats': {
                        'mean': float(np.mean(overall_scores)),
                        'std': float(np.std(overall_scores)),
                        'min': float(np.min(overall_scores)),
                        'max': float(np.max(overall_scores))
                    }
                }
            else:
                statistics = {'total_records_evaluated': 0}
            
            # Configuration used
            config_info = {
                'similarity_threshold': self.similarity_threshold,
                'remediation_threshold': self.remediation_threshold,
                'min_alternatives': self.min_alternatives,
                'max_candidates': self.max_candidates,
                'frequency_weight': self.frequency_weight,
                'similarity_weight': self.similarity_weight,
                'confidence_threshold': self.confidence_threshold,
                'auto_remediate': self.auto_remediate
            }
            
            report_data = {
                'timestamp': timestamp,
                'configuration': config_info,
                'statistics': statistics,
                'remediation_count': self.remediation_count
            }
            
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            logger.info(f"Subject quality audit report saved to {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Error generating audit report: {e}")
            return ""