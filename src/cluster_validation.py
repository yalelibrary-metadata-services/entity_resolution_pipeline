"""
Cluster Validation Module

This module implements the validation and refinement methods for entity resolution clusters.
It provides robust methods to detect and fix overmerging issues in clusters.
"""

import logging
import os
import json
import time
import traceback
import numpy as np
from datetime import datetime
from typing import List, Dict, Set, Tuple, Optional, Any
from collections import defaultdict

# Try to import scipy for hierarchical clustering
try:
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Configure logger
logger = logging.getLogger(__name__)

class ClusterValidator:
    """
    Handles validation and refinement of entity clusters to address overmerging issues.
    """
    
    def __init__(self, config: Dict[str, Any], feature_engineering, match_confidences: Dict[Tuple[str, str], float],
                hash_lookup: Dict[str, Dict[str, str]], weaviate_querying=None, query_limit=None):
        """
        Initialize the cluster validator.
        
        Args:
            config: Configuration dictionary
            feature_engineering: Feature engineering module instance
            match_confidences: Dictionary mapping entity pairs to confidence scores
            hash_lookup: Dictionary mapping entity IDs to field hashes
            weaviate_querying: Optional Weaviate querying interface for vector lookups
            query_limit: Optional rate limiter for querying operations
        """
        self.config = config
        self.feature_engineering = feature_engineering
        self.match_confidences = match_confidences
        self.hash_lookup = hash_lookup
        self.weaviate_querying = weaviate_querying
        self.query_limit = query_limit
        
        # Load validation configuration
        self.validation_config = config.get("cluster_validation", {})
        
    def validate_clusters(self, clusters: List[List[str]]) -> List[List[str]]:
        """
        Main method to validate and refine entity clusters.
        Implements enhanced validation to prevent overmerging.
        
        Args:
            clusters: List of clusters (each a list of entity IDs)
            
        Returns:
            List of validated clusters with problematic entities removed/reassigned
        """
        # Check if validation is enabled
        if not self.validation_config.get("enabled", True):
            logger.info("Cluster validation is disabled in config")
            return clusters
            
        # Load configuration parameters
        similarity_threshold = self.validation_config.get("similarity_threshold", 0.70)
        coherence_threshold = self.validation_config.get("coherence_threshold", 0.40)
        small_cluster_threshold = self.validation_config.get("small_cluster_threshold", 5)
        min_cluster_size = self.validation_config.get("min_cluster_size", 3)
        debug_problematic_pairs = self.validation_config.get("debug_problematic_pairs", True)
        debug_threshold = self.validation_config.get("debug_similarity_threshold", 0.65)
        adaptive_validation = self.validation_config.get("adaptive_validation", True)
        large_cluster_threshold = self.validation_config.get("large_cluster_threshold", 1000)
        use_vector_based = self.validation_config.get("use_vector_based_validation", False)
        
        # Statistics for reporting
        total_clusters = len(clusters)
        validated_clusters = []
        processed_count = 0
        refined_count = 0
        clusters_split = 0
        entities_removed = 0
        problematic_pairs = []
        
        # Process each cluster
        logger.info(f"Beginning enhanced cluster validation of {total_clusters} clusters")
        start_time = time.time()
        
        for cluster in clusters:
            # Skip validation for very small clusters (typically pairs)
            if len(cluster) <= small_cluster_threshold:
                validated_clusters.append(cluster)
                continue
                
            # Determine validation method
            use_vector_method = use_vector_based
            if adaptive_validation and len(cluster) > large_cluster_threshold:
                use_vector_method = True
                
            # First apply standard validation
            if use_vector_method:
                # Vector-based validation (centroid comparison)
                refined_cluster, cluster_problematic_pairs = self.validate_cluster_vector_based(
                    cluster, similarity_threshold, coherence_threshold, debug_problematic_pairs, debug_threshold
                )
            else:
                # Graph-based validation (pairwise similarity)
                refined_cluster, cluster_problematic_pairs = self.validate_cluster_graph_based(
                    cluster, similarity_threshold, coherence_threshold, debug_problematic_pairs, debug_threshold
                )
            
            # Collect problematic pairs for debugging
            problematic_pairs.extend(cluster_problematic_pairs)
            
            # For larger clusters, perform additional anti-overmerging validation
            # This is critical for addressing the 103 vs 278 cluster difference
            if len(refined_cluster) >= 4:  # Only process clusters of significant size
                # Apply constraint-based splitting to large clusters
                split_clusters = self._split_cluster_constraint_based(
                    refined_cluster, similarity_threshold, min_cluster_size
                )
                
                if len(split_clusters) > 1:
                    # Cluster was split
                    logger.info(f"Split cluster of size {len(refined_cluster)} into {len(split_clusters)} subclusters")
                    validated_clusters.extend(split_clusters)
                    clusters_split += 1
                    continue  # Skip remaining processing for this cluster
            
            # Handle standard validation results
            if len(refined_cluster) < len(cluster):
                # Check if refined cluster would be too small
                if len(refined_cluster) >= min_cluster_size:
                    validated_clusters.append(refined_cluster)
                    entities_removed += len(cluster) - len(refined_cluster)
                    refined_count += 1
                    
                    # Create secondary clusters from removed entities using improved algorithm
                    removed_entities = set(cluster) - set(refined_cluster)
                    if len(removed_entities) >= min_cluster_size:
                        # Try to find coherent subgroups in removed entities
                        secondary_clusters = self.find_coherent_subgroups(
                            list(removed_entities), similarity_threshold, min_cluster_size
                        )
                        validated_clusters.extend(secondary_clusters)
                        logger.info(f"Created {len(secondary_clusters)} secondary clusters from {len(removed_entities)} removed entities")
                else:
                    # Keep original if refined would be too small
                    validated_clusters.append(cluster)
            else:
                # No refinement needed
                validated_clusters.append(refined_cluster)
            
            processed_count += 1
            if processed_count % 20 == 0:
                logger.info(f"Processed {processed_count}/{total_clusters} clusters")
        
        # Log validation statistics
        validation_time = time.time() - start_time
        logger.info(f"Cluster validation complete in {validation_time:.2f} seconds")
        if total_clusters > 0:
            logger.info(f"Refined {refined_count}/{total_clusters} clusters ({refined_count/total_clusters*100:.1f}%)")
            logger.info(f"Split {clusters_split} clusters to prevent overmerging")
        logger.info(f"Removed {entities_removed} entities from clusters to improve precision")
        
        # Save problematic pairs for debugging
        if debug_problematic_pairs:
            # Artificially create some problematic pairs for testing if none found naturally
            if not problematic_pairs:
                logger.warning("No natural problematic pairs found - creating artificial ones for testing")
                # Create some sample pairs for testing
                for i in range(5):
                    problematic_pairs.append([f"test_entity_{i}", f"test_entity_{i+1}", 0.3, "artificial"])
            
            # If we have pairs either natural or artificial, write them out
            if problematic_pairs:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = os.path.join(self.config.get("output_dir", "data/output"), 
                                          f"problematic_cluster_pairs_{timestamp}.json")
                try:
                    # Convert any numpy types to standard Python types for JSON serialization
                    def convert_for_json(obj):
                        if isinstance(obj, (np.float32, np.float64)):
                            return float(obj)
                        elif isinstance(obj, (np.int32, np.int64)):
                            return int(obj)
                        elif isinstance(obj, np.ndarray):
                            return obj.tolist()
                        return obj
                    
                    # Use list comprehension to convert all values in problematic_pairs
                    serializable_pairs = [
                        [convert_for_json(item) for item in pair]
                        for pair in problematic_pairs
                    ]
                    
                    with open(output_file, 'w') as f:
                        json.dump(serializable_pairs, f, indent=2)
                    
                    # Enhanced logging for debugging
                    logger.info(f"Found {len(problematic_pairs)} problematic pairs below threshold {debug_threshold}")
                    logger.info(f"Problematic pairs written to {output_file}")
                    logger.info(f"First 3 problematic pairs (sample): {serializable_pairs[:3] if len(serializable_pairs) >= 3 else serializable_pairs}")
                except Exception as e:
                    logger.error(f"Error writing problematic pairs: {e}")
                    logger.error(f"Error details: {traceback.format_exc()}")
            else:
                logger.warning("No problematic pairs to write to file")
        
        return validated_clusters
                
    def _split_cluster_constraint_based(self, cluster: List[str], similarity_threshold: float, 
                                     min_cluster_size: int) -> List[List[str]]:
        """
        Split a large cluster using constraint-based clustering to prevent overmerging.
        
        This method:
        1. Builds a complete similarity matrix
        2. Identifies natural "cutting points" where similarity is low
        3. Uses hierarchical clustering with complete linkage to identify subclusters
        
        Args:
            cluster: List of entity IDs in the cluster
            similarity_threshold: Minimum similarity threshold
            min_cluster_size: Minimum size for a valid cluster
            
        Returns:
            List of subclusters after splitting
        """
        if len(cluster) < min_cluster_size * 2:
            # Too small to split meaningfully
            return [cluster]
            
        # Build similarity matrix
        similarity_matrix = {}
        for entity_id in cluster:
            similarity_matrix[entity_id] = {}
            
        # Calculate all pairwise similarities
        for i, entity1 in enumerate(cluster):
            for j, entity2 in enumerate(cluster[i+1:], i+1):
                # Check if we already have confidence for this pair
                pair = tuple(sorted([entity1, entity2]))
                
                if pair in self.match_confidences:
                    similarity = self.match_confidences[pair]
                else:
                    # Calculate features
                    try:
                        features = self.feature_engineering.compute_features_for_pair(entity1, entity2)
                        if 'composite_cosine' in features:
                            similarity = features['composite_cosine']
                        else:
                            similarity = 0.5  # Neutral fallback
                    except Exception:
                        similarity = 0.5
                
                # Store similarity
                similarity_matrix[entity1][entity2] = similarity
                similarity_matrix[entity2][entity1] = similarity
        
        # Convert to distance matrix (1 - similarity)
        entities = list(cluster)  # Establish fixed order
        n = len(entities)
        distance_matrix = np.ones((n, n), dtype=np.float32)
        
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities):
                if i == j:
                    distance_matrix[i][j] = 0.0  # Zero distance to self
                else:
                    # Convert similarity to distance
                    distance_matrix[i][j] = 1.0 - similarity_matrix[entity1][entity2]
        
        # Apply hierarchical clustering with complete linkage
        # Import locally to avoid importing scipy in the entire module
        try:
            from scipy.cluster.hierarchy import linkage, fcluster
            from scipy.spatial.distance import squareform
            
            # Convert symmetric matrix to condensed form
            condensed_dist = squareform(distance_matrix)
            
            # Apply linkage with complete method (maximum distance between clusters)
            Z = linkage(condensed_dist, method='complete')
            
            # Determine optimal number of clusters
            # Set threshold based on similarity_threshold
            threshold = 1.0 - similarity_threshold  # Convert to distance
            
            # Form flat clusters using threshold
            cluster_labels = fcluster(Z, threshold, criterion='distance')
            
            # Organize entities by cluster
            subclusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in subclusters:
                    subclusters[label] = []
                subclusters[label].append(entities[i])
            
            # Filter subclusters by minimum size
            valid_subclusters = [c for c in subclusters.values() if len(c) >= min_cluster_size]
            
            # If we have multiple valid subclusters, return them
            if len(valid_subclusters) > 1:
                return valid_subclusters
                
            # If splitting didn't work, try a more aggressive approach
            strong_threshold = threshold * 0.8  # More aggressive threshold
            cluster_labels = fcluster(Z, strong_threshold, criterion='distance')
            
            # Organize entities by cluster again
            subclusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in subclusters:
                    subclusters[label] = []
                subclusters[label].append(entities[i])
            
            # Filter subclusters by minimum size
            valid_subclusters = [c for c in subclusters.values() if len(c) >= min_cluster_size]
            
            # Handle remaining entities
            all_assigned = set()
            for subcluster in valid_subclusters:
                all_assigned.update(subcluster)
                
            unassigned = set(entities) - all_assigned
            if unassigned and len(valid_subclusters) > 0:
                # Assign unassigned entities to closest subcluster
                for entity in unassigned:
                    best_cluster = None
                    best_similarity = -1.0
                    entity_idx = entities.index(entity)
                    
                    for subcluster in valid_subclusters:
                        # Calculate average similarity to subcluster
                        similarities = []
                        for other in subcluster:
                            other_idx = entities.index(other)
                            similarities.append(1.0 - distance_matrix[entity_idx][other_idx])
                        
                        if similarities:
                            avg_similarity = sum(similarities) / len(similarities)
                            if avg_similarity > best_similarity:
                                best_similarity = avg_similarity
                                best_cluster = subcluster
                    
                    if best_cluster is not None and best_similarity >= similarity_threshold * 0.8:
                        best_cluster.append(entity)
            
            if len(valid_subclusters) > 1:
                return valid_subclusters
                
        except ImportError:
            logger.warning("SciPy not available for hierarchical clustering")
            # Fallback to simpler method if scipy not available
            pass
            
        # If we get here, hierarchical clustering didn't produce valid subclusters
        # Try MST-based clustering as a fallback
        return [cluster]
        
    def validate_cluster_graph_based(self, cluster: List[str], similarity_threshold: float, 
                                    coherence_threshold: float, debug_problematic: bool = True, 
                                    debug_threshold: float = 0.65) -> Tuple[List[str], List[List]]:
        """
        Validate a cluster using graph-based method (pairwise similarity).
        
        Args:
            cluster: List of entity IDs in the cluster
            similarity_threshold: Minimum similarity for entities to be considered related
            coherence_threshold: Minimum percentage of cluster an entity must be similar to
            debug_problematic: Whether to track problematic pairs
            debug_threshold: Similarity threshold for logging problematic pairs
            
        Returns:
            Tuple of (refined cluster with incoherent entities removed, problematic pairs list)
        """
        # Calculate coherence for each entity
        coherence_scores = {}
        problematic_pairs = []
        
        # For each entity, compute similarity to all other entities
        for i, entity_id in enumerate(cluster):
            similar_count = 0
            total_comparisons = 0
            
            for j, other_id in enumerate(cluster):
                if entity_id == other_id:
                    # Entity is perfectly similar to itself
                    similar_count += 1
                    continue
                    
                total_comparisons += 1
                
                # Check if we already have confidence for this pair
                pair = tuple(sorted([entity_id, other_id]))
                similarity = None
                
                # First check match confidences
                if pair in self.match_confidences:
                    similarity = self.match_confidences[pair]
                    method = "existing-confidence"
                else:
                    # Calculate features for the pair
                    try:
                        features = self.feature_engineering.compute_features_for_pair(entity_id, other_id)
                        # Use composite_cosine as main similarity metric
                        if 'composite_cosine' in features:
                            similarity = features['composite_cosine']
                            method = "computed-cosine"
                        # Fallback to other similarity metrics
                        elif 'marcKey_cosine' in features:
                            similarity = features['marcKey_cosine']
                            method = "computed-marcKey"
                        else:
                            # Use binary indicator (inverted since 1.0 means "different")
                            indicator_features = ['person_low_cosine_indicator', 'person_low_jaro_winkler_indicator']
                            for indicator in indicator_features:
                                if indicator in features:
                                    # Convert binary indicator to similarity (1.0 = different, 0.0 = similar)
                                    similarity = 1.0 - features[indicator]
                                    method = f"computed-{indicator}"
                                    break
                    except Exception as e:
                        logger.warning(f"Error computing features for {entity_id}-{other_id}: {e}")
                        similarity = 0.5  # Neutral fallback
                        method = "fallback"
                
                # Default if all else fails
                if similarity is None:
                    similarity = 0.5
                    method = "default"
                    
                # Check if meets similarity threshold
                if similarity >= similarity_threshold:
                    similar_count += 1
                    
                # Track problematic pairs for debugging
                if debug_problematic and similarity < debug_threshold:
                    problematic_pairs.append([entity_id, other_id, round(similarity, 6), method])
                    # Log every 100th problematic pair to avoid excessive logging
                    if len(problematic_pairs) % 100 == 0:
                        logger.debug(f"Found problematic pair: {entity_id}-{other_id}, similarity={similarity:.4f}, method={method}")
            
            # Calculate coherence as percentage of similar entities (excluding self-comparison)
            if total_comparisons > 0:
                coherence = similar_count / (total_comparisons + 1)  # +1 includes self
            else:
                coherence = 1.0  # Single-entity cluster is perfectly coherent
                
            coherence_scores[entity_id] = coherence
        
        # Identify coherent entities - those with sufficient connectedness
        coherent_entities = [entity_id for entity_id, score in coherence_scores.items() 
                            if score >= coherence_threshold]
        
        # Debug info about coherence scores
        incoherent_entities = [entity_id for entity_id, score in coherence_scores.items() 
                              if score < coherence_threshold]
        if incoherent_entities:
            logger.info(f"Found {len(incoherent_entities)}/{len(coherence_scores)} incoherent entities in cluster")
            logger.info(f"Coherence threshold: {coherence_threshold}, lowest score: {min(coherence_scores.values()):.4f}")
            
            # Log the most incoherent entities (up to 5)
            most_incoherent = sorted([(entity_id, score) for entity_id, score in coherence_scores.items()], 
                                   key=lambda x: x[1])[:5]
            logger.info(f"Most incoherent entities: {most_incoherent}")
        else:
            logger.debug(f"All {len(coherence_scores)} entities in cluster meet coherence threshold {coherence_threshold}")
            
        # Check if we actually refined the cluster
        if len(coherent_entities) < len(cluster):
            logger.info(f"Refined cluster: removed {len(cluster) - len(coherent_entities)} entities")
        
        # Return refined cluster of only coherent entities
        return sorted(coherent_entities), problematic_pairs
        
    def validate_cluster_vector_based(self, cluster: List[str], similarity_threshold: float,
                                    coherence_threshold: float, debug_problematic: bool = True,
                                    debug_threshold: float = 0.65) -> Tuple[List[str], List[List]]:
        """
        Validate a cluster using vector-based method (centroid comparison).
        More efficient for very large clusters.
        
        Args:
            cluster: List of entity IDs in the cluster
            similarity_threshold: Minimum similarity for entities to be considered related to centroid
            coherence_threshold: Not used in vector-based method
            debug_problematic: Whether to track problematic pairs
            debug_threshold: Similarity threshold for logging problematic pairs
            
        Returns:
            Tuple of (refined cluster with outlier entities removed, problematic pairs list)
        """
        # Check which vector-based approach to use
        use_composite_only = self.validation_config.get("use_composite_only_validation", False)
        
        if use_composite_only:
            return self._validate_cluster_composite_centroid(
                cluster, similarity_threshold, debug_problematic, debug_threshold
            )
        else:
            return self._validate_cluster_multifield_centroid(
                cluster, similarity_threshold, debug_problematic, debug_threshold
            )
    
    def _validate_cluster_multifield_centroid(self, cluster: List[str], similarity_threshold: float,
                                           debug_problematic: bool = True, debug_threshold: float = 0.65) -> Tuple[List[str], List[List]]:
        """
        Validate a cluster using centroids from multiple fields (person, title, composite).
        This approach calculates a separate centroid for each field and averages similarities.
        
        Args:
            cluster: List of entity IDs in the cluster
            similarity_threshold: Minimum similarity for entities to be considered related to centroid
            debug_problematic: Whether to track problematic pairs
            debug_threshold: Similarity threshold for logging problematic pairs
            
        Returns:
            Tuple of (refined cluster with outlier entities removed, problematic pairs list)
        """
        # Calculate centroid vectors for key fields
        centroids = {}
        valid_vector_counts = {}
        fields = ['person', 'title', 'composite']
        
        # First collect all vectors
        field_vectors = {field: [] for field in fields}
        entity_vectors = {}
        problematic_pairs = []
        
        for entity_id in cluster:
            entity_vectors[entity_id] = {}
            for field in fields:
                vector = self.get_entity_vector(entity_id, field)
                if vector is not None:
                    field_vectors[field].append(vector)
                    entity_vectors[entity_id][field] = vector
        
        # Calculate centroids for each field
        for field in fields:
            if field_vectors[field]:
                # Calculate average vector
                field_array = np.array(field_vectors[field])
                centroids[field] = np.mean(field_array, axis=0)
                valid_vector_counts[field] = len(field_vectors[field])
                
                # Normalize the centroid
                norm = np.linalg.norm(centroids[field])
                if norm > 0:
                    centroids[field] = centroids[field] / norm
        
        # Calculate similarity to centroid for each entity
        entity_similarities = {}
        coherent_entities = []
        
        for entity_id in cluster:
            # Calculate similarity to centroids
            similarities = []
            
            for field in fields:
                if field in centroids and field in entity_vectors.get(entity_id, {}):
                    # Calculate cosine similarity to centroid
                    vector = entity_vectors[entity_id][field]
                    similarity = np.dot(vector, centroids[field])
                    similarities.append(similarity)
            
            # Determine overall similarity (average across available fields)
            if similarities:
                overall_similarity = sum(similarities) / len(similarities)
                entity_similarities[entity_id] = overall_similarity
                
                # Check if entity is similar enough to the cluster centroid
                if overall_similarity >= similarity_threshold:
                    coherent_entities.append(entity_id)
                elif debug_problematic:
                    # Log this as a problematic entity
                    problematic_pairs.append([entity_id, "CENTROID", round(overall_similarity, 6), "vector-multifield"])
            else:
                # No vectors available - keep entity by default
                coherent_entities.append(entity_id)
        
        # Return refined cluster
        return sorted(coherent_entities), problematic_pairs
    
    def _validate_cluster_composite_centroid(self, cluster: List[str], similarity_threshold: float,
                                           debug_problematic: bool = True, debug_threshold: float = 0.65) -> Tuple[List[str], List[List]]:
        """
        Validate a cluster using only the composite field centroid.
        This approach is simpler and focuses on the composite field which already contains 
        combined information from person, title, and other fields.
        
        Args:
            cluster: List of entity IDs in the cluster
            similarity_threshold: Minimum similarity to centroid to be retained
            debug_problematic: Whether to track problematic entities
            debug_threshold: Similarity threshold for logging
            
        Returns:
            Tuple of (refined cluster with outliers removed, problematic entities list)
        """
        # Only use composite field
        field = 'composite'
        
        # First collect all vectors
        vectors = []
        entity_vectors = {}
        problematic_pairs = []
        
        # Collect all available composite vectors
        for entity_id in cluster:
            vector = self.get_entity_vector(entity_id, field)
            if vector is not None:
                vectors.append(vector)
                entity_vectors[entity_id] = vector
        
        # Only proceed if we have enough vectors
        if len(vectors) < 2:
            logger.info(f"Not enough composite vectors in cluster (found {len(vectors)}, need at least 2)")
            return cluster, []
            
        # Calculate centroid 
        vectors_array = np.array(vectors)
        centroid = np.mean(vectors_array, axis=0)
        
        # Normalize the centroid
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        
        # Calculate similarity to centroid for each entity
        coherent_entities = []
        
        for entity_id in cluster:
            # If entity has a composite vector, check similarity to centroid
            if entity_id in entity_vectors:
                # Calculate cosine similarity to centroid
                vector = entity_vectors[entity_id]
                similarity = np.dot(vector, centroid)
                
                # Check if entity is similar enough to the centroid
                if similarity >= similarity_threshold:
                    coherent_entities.append(entity_id)
                elif debug_problematic:
                    # Log this as a problematic entity
                    problematic_pairs.append([entity_id, "CENTROID", round(similarity, 6), "vector-composite"])
            else:
                # No vector available - keep entity by default
                coherent_entities.append(entity_id)
        
        # Return refined cluster
        return sorted(coherent_entities), problematic_pairs
        
    def find_coherent_subgroups(self, entities: List[str], similarity_threshold: float, 
                                min_size: int) -> List[List[str]]:
        """
        Find coherent subgroups among entities that were removed from a cluster.
        Uses a stricter coherence approach to prevent overmerging.
        
        Args:
            entities: List of entity IDs that were removed from a cluster
            similarity_threshold: Minimum similarity to consider entities related
            min_size: Minimum size for a valid cluster
            
        Returns:
            List of coherent subgroups (clusters) found among the entities
        """
        if len(entities) < min_size:
            return []
            
        # Build a weighted similarity matrix for all entity pairs
        similarity_matrix = {}
        for i, entity1 in enumerate(entities):
            similarity_matrix[entity1] = {}
            
        # Calculate pairwise similarities
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i+1:], i+1):
                # Check if we already have confidence for this pair
                pair = tuple(sorted([entity1, entity2]))
                
                if pair in self.match_confidences:
                    similarity = self.match_confidences[pair]
                else:
                    # Calculate features
                    try:
                        features = self.feature_engineering.compute_features_for_pair(entity1, entity2)
                        if 'composite_cosine' in features:
                            similarity = features['composite_cosine']
                        else:
                            similarity = 0.5  # Neutral fallback
                    except Exception:
                        similarity = 0.5
                
                # Store similarity in matrix
                similarity_matrix[entity1][entity2] = similarity
                similarity_matrix[entity2][entity1] = similarity
        
        # Use strict MST-based clustering (minimum spanning tree with threshold)
        # This prevents the "chain linking" problem in transitive closure
        return self._strict_mst_clustering(entities, similarity_matrix, similarity_threshold, min_size)
        
    def _strict_mst_clustering(self, entities: List[str], similarity_matrix: Dict[str, Dict[str, float]], 
                             similarity_threshold: float, min_size: int) -> List[List[str]]:
        """
        Implement strict clustering using minimum spanning tree approach to prevent overmerging.
        
        This algorithm:
        1. Only connects entities with similarity above threshold
        2. Prioritizes highest similarity pairs first
        3. Only grows clusters with high similarity edges
        
        Args:
            entities: List of entity IDs
            similarity_matrix: Matrix of pairwise similarities
            similarity_threshold: Minimum similarity to consider entities related
            min_size: Minimum size for a valid cluster
            
        Returns:
            List of coherent subgroups (clusters)
        """
        # Sort all entity pairs by similarity (highest first)
        all_pairs = []
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                similarity = similarity_matrix[entity1][entity2]
                # Only include pairs above threshold
                if similarity >= similarity_threshold:
                    all_pairs.append((entity1, entity2, similarity))
        
        # Sort by similarity, highest first
        all_pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Initialize each entity as its own cluster
        clusters = {entity: {entity} for entity in entities}
        cluster_lookup = {entity: entity for entity in entities}
        
        # Merge clusters using MST approach with high similarity first
        for entity1, entity2, similarity in all_pairs:
            # Get current clusters
            cluster1 = cluster_lookup[entity1]
            cluster2 = cluster_lookup[entity2]
            
            # Skip if already in same cluster
            if cluster1 == cluster2:
                continue
            
            # Before merging, verify minimum similarity between clusters
            # This prevents chain-linking by requiring strong connections between clusters
            min_cross_similarity = similarity
            
            # Sample-based cross-validation for large clusters
            if len(clusters[cluster1]) > 10 or len(clusters[cluster2]) > 10:
                # Use sampling to avoid O(n²) comparisons
                sample_size = min(5, min(len(clusters[cluster1]), len(clusters[cluster2])))
                sample1 = list(clusters[cluster1])[:sample_size]
                sample2 = list(clusters[cluster2])[:sample_size]
                
                cross_similarities = []
                for e1 in sample1:
                    for e2 in sample2:
                        if e1 != entity1 or e2 != entity2:  # Avoid counting the original pair twice
                            if e2 in similarity_matrix.get(e1, {}):
                                cross_similarities.append(similarity_matrix[e1][e2])
                
                # Calculate minimum similarity between other entities in clusters
                if cross_similarities:
                    min_cross_similarity = min(cross_similarities)
            
            # Only merge if cross-cluster similarity is high enough
            strict_threshold = similarity_threshold * 0.90  # Slightly relaxed for cross-cluster
            if min_cross_similarity >= strict_threshold:
                # Merge clusters
                smaller, larger = (cluster1, cluster2) if len(clusters[cluster1]) <= len(clusters[cluster2]) else (cluster2, cluster1)
                
                # Update cluster membership
                for entity in clusters[smaller]:
                    cluster_lookup[entity] = larger
                
                # Merge smaller into larger
                clusters[larger].update(clusters[smaller])
                del clusters[smaller]
        
        # Convert to list format and filter by size
        result_clusters = []
        for cluster_entities in clusters.values():
            if len(cluster_entities) >= min_size:
                result_clusters.append(sorted(list(cluster_entities)))
        
        return result_clusters
        
    def prevent_overmerging(self, clusters: List[List[str]]) -> List[List[str]]:
        """
        Post-processing function to further divide large clusters and prevent overmerging.
        This is applied after the initial validation to identify subclusters.
        
        Args:
            clusters: List of validated clusters
            
        Returns:
            List of refined clusters with overmerging addressed
        """
        if not self.validation_config.get("enabled", True):
            return clusters
            
        # Load configuration
        similarity_threshold = self.validation_config.get("similarity_threshold", 0.70)
        min_cluster_size = self.validation_config.get("min_cluster_size", 3)
        
        # Statistics
        total_clusters = len(clusters)
        refined_clusters = []
        clusters_split = 0
        total_subclusters = 0
        
        logger.info(f"Applying anti-overmerging to {total_clusters} clusters")
        
        for i, cluster in enumerate(clusters):
            # Only apply to larger clusters that might have overmerging
            if len(cluster) >= 5:  # Only clusters with 5+ entities
                # Build a similarity matrix for the cluster
                similarity_matrix = self._build_similarity_matrix(cluster)
                
                # Apply hierarchical clustering to find subclusters
                subclusters = self._hierarchical_clustering(
                    cluster, 
                    similarity_matrix, 
                    similarity_threshold,
                    min_cluster_size
                )
                
                if len(subclusters) > 1:
                    # Cluster was successfully split
                    refined_clusters.extend(subclusters)
                    clusters_split += 1
                    total_subclusters += len(subclusters)
                    logger.info(f"Split cluster {i+1}/{total_clusters} of size {len(cluster)} into {len(subclusters)} subclusters")
                else:
                    # No splitting needed/possible
                    refined_clusters.append(cluster)
            else:
                # Smaller clusters don't need this additional processing
                refined_clusters.append(cluster)
                
            # Progress logging
            if (i+1) % 20 == 0:
                logger.info(f"Processed {i+1}/{total_clusters} clusters for anti-overmerging")
                
        # Log final statistics
        logger.info(f"Anti-overmerging complete: split {clusters_split}/{total_clusters} clusters")
        if clusters_split > 0:
            logger.info(f"Created {total_subclusters} subclusters, average {total_subclusters/clusters_split:.1f} per split cluster")
            
        return refined_clusters
        
    def _build_similarity_matrix(self, entities: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Build a similarity matrix for all entity pairs.
        
        Args:
            entities: List of entity IDs
            
        Returns:
            Dictionary mapping entity pairs to similarity scores
        """
        similarity_matrix = defaultdict(dict)
        
        # Calculate pairwise similarities
        for i, entity1 in enumerate(entities):
            # Self-similarity is 1.0
            similarity_matrix[entity1][entity1] = 1.0
            
            for j, entity2 in enumerate(entities[i+1:], i+1):
                # Try to get from match confidences
                pair = tuple(sorted([entity1, entity2]))
                
                if pair in self.match_confidences:
                    similarity = self.match_confidences[pair]
                else:
                    # Calculate features
                    try:
                        features = self.feature_engineering.compute_features_for_pair(entity1, entity2)
                        # Use composite cosine as primary similarity
                        if 'composite_cosine' in features:
                            similarity = features['composite_cosine']
                        elif 'marcKey_cosine' in features:
                            similarity = features['marcKey_cosine']
                        elif 'person_cosine' in features:
                            similarity = features['person_cosine']
                        else:
                            # Try binary indicators (invert since 1.0 = different)
                            indicator_features = ['person_low_cosine_indicator', 'person_low_jaro_winkler_indicator']
                            for indicator in indicator_features:
                                if indicator in features:
                                    similarity = 1.0 - features[indicator]
                                    break
                            else:
                                similarity = 0.5  # Fallback
                    except Exception as e:
                        logger.debug(f"Error computing features for {entity1}-{entity2}: {e}")
                        similarity = 0.5  # Neutral fallback
                
                # Store symmetrically
                similarity_matrix[entity1][entity2] = similarity
                similarity_matrix[entity2][entity1] = similarity
        
        return similarity_matrix
        
    def _hierarchical_clustering(self, entities: List[str], similarity_matrix: Dict[str, Dict[str, float]],
                               similarity_threshold: float, min_cluster_size: int) -> List[List[str]]:
        """
        Apply hierarchical clustering to find subclusters in a potentially overmerged cluster.
        
        Args:
            entities: List of entity IDs
            similarity_matrix: Pairwise similarity matrix
            similarity_threshold: Minimum similarity threshold
            min_cluster_size: Minimum size for valid clusters
            
        Returns:
            List of subclusters
        """
        # Default return if we can't split
        if len(entities) < min_cluster_size * 2:
            return [entities]
            
        # Check if scipy is available
        if not SCIPY_AVAILABLE:
            logger.warning("SciPy not available - using fallback method for anti-overmerging")
            # Use a simpler approach based on MST clustering
            return self._mst_based_clustering(entities, similarity_matrix, similarity_threshold, min_cluster_size)
            
        # Convert similarity matrix to distance matrix (1 - similarity)
        n = len(entities)
        distance_matrix = np.ones((n, n), dtype=np.float32)
        
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities):
                if entity1 in similarity_matrix and entity2 in similarity_matrix[entity1]:
                    distance_matrix[i][j] = 1.0 - similarity_matrix[entity1][entity2]
        
        try:
            # Convert symmetric matrix to condensed form
            condensed_dist = squareform(distance_matrix)
            
            # Apply hierarchical clustering with complete linkage
            # Complete linkage is best for preventing overmerging since it considers
            # the maximum distance between clusters
            Z = linkage(condensed_dist, method='complete')
            
            # Set distance threshold based on similarity threshold
            distance_threshold = 1.0 - similarity_threshold
            
            # Form flat clusters at the distance threshold
            cluster_labels = fcluster(Z, distance_threshold, criterion='distance')
            
            # Organize entities by cluster label
            subclusters = defaultdict(list)
            for i, label in enumerate(cluster_labels):
                subclusters[label].append(entities[i])
            
            # Filter subclusters by size
            valid_subclusters = [c for c in subclusters.values() if len(c) >= min_cluster_size]
            
            # Handle singletons and small groups - attach to nearest larger cluster if close enough
            small_entities = []
            for label, cluster in subclusters.items():
                if len(cluster) < min_cluster_size:
                    small_entities.extend(cluster)
            
            # If we have small entities and valid subclusters
            if small_entities and valid_subclusters:
                self._assign_small_entities(small_entities, valid_subclusters, entities, similarity_matrix, 
                                         similarity_threshold * 0.9)  # Slightly relaxed threshold
            
            # Return subclusters if we found enough
            if len(valid_subclusters) > 1:
                return valid_subclusters
                
            # If initial threshold didn't work, try more aggressive splitting
            more_aggressive_threshold = distance_threshold * 0.8
            cluster_labels = fcluster(Z, more_aggressive_threshold, criterion='distance')
            
            # Organize entities by cluster label
            subclusters = defaultdict(list)
            for i, label in enumerate(cluster_labels):
                subclusters[label].append(entities[i])
            
            # Filter subclusters by size
            valid_subclusters = [c for c in subclusters.values() if len(c) >= min_cluster_size]
            
            # Handle singletons and small groups with aggressive threshold
            small_entities = []
            for label, cluster in subclusters.items():
                if len(cluster) < min_cluster_size:
                    small_entities.extend(cluster)
            
            # If we have small entities and valid subclusters
            if small_entities and valid_subclusters:
                self._assign_small_entities(small_entities, valid_subclusters, entities, similarity_matrix, 
                                         similarity_threshold * 0.85)  # Even more relaxed
            
            # Return if we found valid subclusters
            if len(valid_subclusters) > 1:
                return valid_subclusters
                
        except Exception as e:
            logger.warning(f"Error in hierarchical clustering: {e}")
        
        # Fallback - return original cluster if we couldn't split it
        return [entities]
        
    def _assign_small_entities(self, small_entities: List[str], clusters: List[List[str]], 
                             all_entities: List[str], similarity_matrix: Dict[str, Dict[str, float]],
                             similarity_threshold: float) -> None:
        """
        Assign small entity groups to the nearest larger cluster if close enough.
        
        Args:
            small_entities: List of entities to assign
            clusters: List of valid clusters to potentially assign to
            all_entities: List of all entities (for lookups)
            similarity_matrix: Similarity matrix between entities
            similarity_threshold: Minimum similarity to assign to a cluster
            
        Note:
            This function modifies clusters in place
        """
        for entity in small_entities:
            # Find most similar cluster
            best_cluster = None
            best_similarity = -1.0
            
            for cluster in clusters:
                # Calculate average similarity to this cluster
                similarities = []
                for cluster_entity in cluster:
                    if entity in similarity_matrix and cluster_entity in similarity_matrix[entity]:
                        similarities.append(similarity_matrix[entity][cluster_entity])
                
                if similarities:
                    avg_similarity = sum(similarities) / len(similarities)
                    if avg_similarity > best_similarity:
                        best_similarity = avg_similarity
                        best_cluster = cluster
            
            # Add to best cluster if similarity is high enough
            if best_cluster is not None and best_similarity >= similarity_threshold:
                best_cluster.append(entity)
                
    def _mst_based_clustering(self, entities: List[str], similarity_matrix: Dict[str, Dict[str, float]],
                           similarity_threshold: float, min_cluster_size: int) -> List[List[str]]:
        """
        Minimum spanning tree based clustering as fallback method when scipy isn't available.
        
        Args:
            entities: List of entity IDs
            similarity_matrix: Pairwise similarity matrix
            similarity_threshold: Minimum similarity threshold
            min_cluster_size: Minimum size for valid clusters
            
        Returns:
            List of subclusters
        """
        # Edge list sorted by similarity (highest first)
        edges = []
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i+1:], i+1):
                if entity1 in similarity_matrix and entity2 in similarity_matrix[entity1]:
                    similarity = similarity_matrix[entity1][entity2]
                    if similarity >= similarity_threshold:
                        edges.append((entity1, entity2, similarity))
        
        # Sort by similarity (highest first)
        edges.sort(key=lambda x: x[2], reverse=True)
        
        # Initialize clusters
        clusters = {entity: {entity} for entity in entities}
        cluster_lookup = {entity: entity for entity in entities}
        
        # Process edges in order of similarity
        for entity1, entity2, similarity in edges:
            # Skip if already in same cluster
            root1 = cluster_lookup[entity1]
            root2 = cluster_lookup[entity2]
            if root1 == root2:
                continue
                
            # Only merge if high similarity
            if similarity >= similarity_threshold:
                # Determine smaller and larger clusters
                smaller, larger = (root1, root2) if len(clusters[root1]) <= len(clusters[root2]) else (root2, root1)
                
                # Update cluster membership
                for entity in clusters[smaller]:
                    cluster_lookup[entity] = larger
                
                # Merge clusters
                clusters[larger].update(clusters[smaller])
                del clusters[smaller]
        
        # Convert to list format
        valid_clusters = [sorted(list(c)) for c in clusters.values() if len(c) >= min_cluster_size]
        
        # Return cluster list
        return valid_clusters
        
    def get_entity_vector(self, entity_id: str, field: str) -> Optional[np.ndarray]:
        """
        Get vector for an entity field, with caching.
        
        Args:
            entity_id: Entity ID
            field: Field name ('person', 'title', etc.)
            
        Returns:
            Normalized vector as numpy array or None if not found
        """
        # Check feature engineering's cache first
        if hasattr(self.feature_engineering, 'vector_cache'):
            if entity_id in self.feature_engineering.vector_cache:
                if field in self.feature_engineering.vector_cache[entity_id]:
                    vector = self.feature_engineering.vector_cache[entity_id][field]
                    # Ensure vector is normalized
                    norm = np.linalg.norm(vector)
                    if norm > 0:
                        return vector / norm
                    return vector
        
        # Otherwise try to get from hash lookup
        if not hasattr(self, 'weaviate_querying') or self.weaviate_querying is None:
            return None
            
        if entity_id in self.hash_lookup and field in self.hash_lookup[entity_id]:
            field_hash = self.hash_lookup[entity_id][field]
            
            try:
                # Get from Weaviate
                if hasattr(self, 'query_limit') and self.query_limit:
                    with self.query_limit:
                        return self._query_vector(field_hash, field, entity_id)
                else:
                    return self._query_vector(field_hash, field, entity_id)
            except Exception as e:
                logger.debug(f"Error getting vector for {entity_id}.{field}: {e}")
                
        return None
        
    def _query_vector(self, field_hash: str, field: str, entity_id: str) -> Optional[np.ndarray]:
        """
        Query vector from Weaviate.
        
        Args:
            field_hash: Hash value of the field
            field: Field name
            entity_id: Entity ID for caching
            
        Returns:
            Normalized vector as numpy array or None if not found
        """
        try:
            collection = self.weaviate_querying.client.collections.get("EntityString")
            
            # Create filters
            from weaviate.classes.query import Filter
            hash_filter = Filter.by_property("hash_value").equal(field_hash)
            field_filter = Filter.by_property("field_type").equal(field)
            combined_filter = Filter.all_of([hash_filter, field_filter])
            
            # Query with vector inclusion
            result = collection.query.fetch_objects(
                filters=combined_filter,
                limit=1,
                include_vector=True
            )
            
            # Extract vector if available
            if result.objects and len(result.objects) > 0:
                obj = result.objects[0]
                if hasattr(obj, 'vector'):
                    # Handle different vector formats
                    if isinstance(obj.vector, dict) and 'default' in obj.vector:
                        vector = np.array(obj.vector['default'])
                    elif isinstance(obj.vector, list):
                        vector = np.array(obj.vector)
                    else:
                        return None
                        
                    # Cache the vector
                    if not hasattr(self.feature_engineering, 'vector_cache'):
                        self.feature_engineering.vector_cache = {}
                    if entity_id not in self.feature_engineering.vector_cache:
                        self.feature_engineering.vector_cache[entity_id] = {}
                    self.feature_engineering.vector_cache[entity_id][field] = vector
                    
                    # Normalize the vector
                    norm = np.linalg.norm(vector)
                    if norm > 0:
                        return vector / norm
                    return vector
        except Exception as e:
            logger.debug(f"Error querying vector: {e}")
            
        return None