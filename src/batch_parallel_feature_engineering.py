"""
Feature engineering module for entity resolution pipeline.

This module computes features for record pairs based on vector similarity
and other similarity metrics with batch and parallel processing.
"""

import os
import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import jellyfish
import Levenshtein
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

from src.utils import (
    Timer, get_memory_usage, save_checkpoint, load_checkpoint,
    extract_birth_death_years, normalize_name, compute_harmonic_mean
)
from src.mmap_dict import MMapDict

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Handles feature engineering for entity resolution.
    
    Features:
    - Vector similarity features
    - String similarity features
    - Interaction features
    - Feature normalization
    - Recursive feature elimination
    - Batch and parallel processing
    """
    
    def __init__(self, config):
        """
        Initialize the feature engineer with configuration parameters.
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = config
        
        # Feature configuration
        self.cosine_similarities = config['features']['cosine_similarities']
        self.string_similarities = config['features']['string_similarities']
        self.harmonic_means = config['features']['harmonic_means']
        self.additional_interactions = config['features']['additional_interactions']
        self.normalize_features = config['features']['normalize_features']
        self.rfe_enabled = config['features']['rfe_enabled']
        
        # Prefilters
        self.exact_name_birth_death_prefilter = config['features']['prefilters']['exact_name_birth_death_prefilter']
        self.composite_cosine_prefilter = config['features']['prefilters']['composite_cosine_prefilter']
        self.person_cosine_prefilter = config['features']['prefilters']['person_cosine_prefilter']
        
        # Batch processing parameters
        self.batch_size = config['system']['batch_size']
        self.max_workers = config['system']['max_workers']
        
        # Data paths
        self.output_dir = Path(config['system']['output_dir'])
        self.temp_dir = Path(config['system']['temp_dir'])
        self.checkpoint_dir = Path(config['system']['checkpoint_dir'])
        self.ground_truth_file = Path(config['data']['ground_truth_file'])
        
        # Create directories if they don't exist
        for dir_path in [self.output_dir, self.temp_dir, self.checkpoint_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize feature set
        self.feature_names = []
        self.scaler = None
        self.rfe = None
        
        # Load unique strings for string similarity calculations
        self.unique_strings = self._load_unique_strings()
        
        logger.info("FeatureEngineer initialized")
    
    def _load_unique_strings(self):
        """
        Load unique strings for string similarity calculations.
        
        Returns:
            dict: Dictionary of hash -> string
        """
        # Load unique strings from index
        unique_strings_index_path = self.output_dir / "unique_strings_index.json"
        if unique_strings_index_path.exists():
            with open(unique_strings_index_path, 'r') as f:
                unique_strings_index = json.load(f)
            
            location = unique_strings_index.get('location')
            if location and os.path.exists(location):
                return MMapDict(location)
        
        # Fall back to JSON file
        unique_strings_path = self.output_dir / "unique_strings.json"
        if unique_strings_path.exists():
            with open(unique_strings_path, 'r') as f:
                return json.load(f)
        
        # Fall back to sample
        unique_strings_sample_path = self.output_dir / "unique_strings_sample.json"
        if unique_strings_sample_path.exists():
            with open(unique_strings_sample_path, 'r') as f:
                return json.load(f)
        
        logger.warning("Could not load unique strings, string similarity features may not work correctly")
        return {}
    
    def execute_ground_truth_features(self, checkpoint=None):
        """
        Execute feature engineering for ground truth data.
        
        Args:
            checkpoint (str, optional): Path to checkpoint file. Defaults to None.
            
        Returns:
            dict: Feature engineering results
        """
        # Load checkpoint if provided
        if checkpoint and os.path.exists(checkpoint):
            state = load_checkpoint(checkpoint)
            feature_set = state.get('feature_set', {})
            processed_pairs = set(state.get('processed_pairs', []))
            self.feature_names = state.get('feature_names', [])
            
            if 'scaler' in state:
                self.scaler = state['scaler']
            
            if 'rfe' in state:
                self.rfe = state['rfe']
            
            logger.info(f"Resumed ground truth feature engineering from checkpoint: {checkpoint}")
            logger.info(f"Loaded {len(feature_set)} feature set entries")
            logger.info(f"Loaded {len(processed_pairs)} processed pairs")
        else:
            feature_set = {}
            processed_pairs = set()
        
        # Load ground truth data
        match_pairs = self._load_ground_truth()
        logger.info(f"Loaded {len(match_pairs)} ground truth pairs")
        
        # Load pair vectors
        pair_vectors = self._load_pair_vectors()
        logger.info(f"Loaded {len(pair_vectors)} pair vectors")
        
        # Filter out already processed pairs
        pairs_to_process = [
            (left_id, right_id, match)
            for left_id, right_id, match in match_pairs
            if f"{left_id}_{right_id}" not in processed_pairs
        ]
        
        if not pairs_to_process:
            logger.info("No new pairs to process")
            return {
                'pairs_processed': len(processed_pairs),
                'feature_set_size': len(feature_set),
                'duration': 0.0
            }
        
        logger.info(f"Processing {len(pairs_to_process)} ground truth pairs")
        
        # Create batches of pairs for parallel processing
        pair_batches = self._create_pair_batches(pairs_to_process)
        logger.info(f"Created {len(pair_batches)} pair batches")
        
        # Process pair batches
        with Timer() as timer:
            for batch_idx, batch in enumerate(tqdm(pair_batches, desc="Processing pair batches")):
                try:
                    # Get feature vectors for batch
                    batch_features = self._compute_batch_features(batch, pair_vectors)
                    
                    # Update feature set
                    feature_set.update(batch_features)
                    
                    # Update processed pairs
                    for left_id, right_id, _ in batch:
                        processed_pairs.add(f"{left_id}_{right_id}")
                    
                    # Save checkpoint periodically
                    if (batch_idx + 1) % 10 == 0:
                        logger.info(
                            f"Processed {batch_idx + 1}/{len(pair_batches)} batches, "
                            f"{len(processed_pairs)}/{len(match_pairs)} pairs"
                        )
                        
                        checkpoint_path = self.checkpoint_dir / f"ground_truth_features_{batch_idx + 1}.ckpt"
                        
                        checkpoint_state = {
                            'feature_set': feature_set,
                            'processed_pairs': list(processed_pairs),
                            'feature_names': self.feature_names
                        }
                        
                        if self.scaler:
                            checkpoint_state['scaler'] = self.scaler
                        
                        if self.rfe:
                            checkpoint_state['rfe'] = self.rfe
                        
                        save_checkpoint(checkpoint_state, checkpoint_path)
                
                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    
                    # Save checkpoint on error
                    error_checkpoint = self.checkpoint_dir / f"ground_truth_features_error_{batch_idx}.ckpt"
                    save_checkpoint({
                        'feature_set': feature_set,
                        'processed_pairs': list(processed_pairs),
                        'feature_names': self.feature_names
                    }, error_checkpoint)
                    
                    # Continue with next batch
                    continue
            
            # Normalize features if configured
            if self.normalize_features:
                feature_set = self._normalize_features(feature_set)
            
            # Perform recursive feature elimination if configured
            if self.rfe_enabled:
                feature_set = self._perform_recursive_feature_elimination(feature_set)
        
        # Save final results
        self._save_ground_truth_features(feature_set, processed_pairs)
        
        results = {
            'pairs_processed': len(processed_pairs),
            'feature_set_size': len(feature_set),
            'feature_count': len(self.feature_names),
            'duration': timer.duration
        }
        
        logger.info(
            f"Ground truth feature engineering completed: {results['pairs_processed']} pairs, "
            f"{results['feature_count']} features, "
            f"{timer.duration:.2f} seconds"
        )
        
        return results
    
    def _load_ground_truth(self):
        """
        Load ground truth data from file.
        
        Returns:
            list: List of (left_id, right_id, match) tuples
        """
        match_pairs = []
        
        with open(self.ground_truth_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    left_id = parts[0]
                    right_id = parts[1]
                    match = parts[2].lower() == 'true'
                    match_pairs.append((left_id, right_id, match))
        
        return match_pairs
    
    def _load_pair_vectors(self):
        """
        Load pair vectors from file.
        
        Returns:
            dict: Dictionary of pair ID -> vectors
        """
        # Load metadata
        metadata_path = self.output_dir / "ground_truth_query_metadata.json"
        if not metadata_path.exists():
            logger.warning("Ground truth query metadata not found")
            return {}
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load pair vectors
        pair_vectors_path = metadata.get('pair_vectors_path')
        if not pair_vectors_path:
            logger.warning("Pair vectors path not found in metadata")
            return {}
        
        with open(pair_vectors_path, 'rb') as f:
            import pickle
            pair_vectors = pickle.load(f)
        
        # Convert to dictionary for easier lookup
        pair_dict = {}
        for pair in pair_vectors:
            left_id = pair['left_id']
            right_id = pair['right_id']
            pair_dict[f"{left_id}_{right_id}"] = pair
        
        return pair_dict
    
    def _create_pair_batches(self, pairs):
        """
        Create batches of pairs for parallel processing.
        
        Args:
            pairs (list): List of (left_id, right_id, match) tuples
            
        Returns:
            list: Batches of pairs
        """
        # Create batches
        batches = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i + self.batch_size]
            batches.append(batch)
        
        return batches
    
    def _compute_batch_features(self, batch, pair_vectors):
        """
        Compute features for a batch of pairs.
        
        Args:
            batch (list): Batch of (left_id, right_id, match) tuples
            pair_vectors (dict): Dictionary of pair ID -> vectors
            
        Returns:
            dict: Dictionary of pair ID -> feature vector
        """
        # Create a process pool for parallel processing
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit jobs
            future_to_pair = {
                executor.submit(
                    self._compute_pair_features,
                    left_id,
                    right_id,
                    match,
                    pair_vectors.get(f"{left_id}_{right_id}")
                ): (left_id, right_id, match)
                for left_id, right_id, match in batch
            }
            
            # Process results as they complete
            batch_features = {}
            for future in as_completed(future_to_pair):
                left_id, right_id, match = future_to_pair[future]
                
                try:
                    # UPDATE HERE: Unpack all 5 values returned by _compute_pair_features
                    result = future.result()
                    
                    # Handle both the old 3-value and new 5-value return formats
                    if len(result) == 5:
                        pair_id, features, labels, features_raw, features_norm = result
                    else:
                        pair_id, features, labels = result
                        features_raw = {}
                        features_norm = {}
                    
                    # Store feature vector with additional data
                    if pair_id and features:
                        batch_features[pair_id] = {
                            'features': features,
                            'labels': labels,
                            'features_raw': features_raw,
                            'features_norm': features_norm
                        }
                    
                    # Initialize feature names if not already set
                    if features and not self.feature_names:
                        self.feature_names = list(features.keys())
                
                except Exception as e:
                    logger.error(f"Error computing features for pair {left_id}_{right_id}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
            
            return batch_features
    
    
    def _compute_pair_features(self, left_id, right_id, match, pair_data):
        """
        Compute features for a single pair, with proper representation of features
        in raw, normalized, and standardized forms.
        
        Args:
            left_id (str): Left record ID
            right_id (str): Right record ID
            match (bool): Whether the pair is a match
            pair_data (dict): Pair vector data
            
        Returns:
            tuple: (pair_id, features, labels, features_raw, features_norm)
        """
        pair_id = f"{left_id}_{right_id}"
        
        # Initialize feature vector and labels
        features = {}
        features_raw = {}  # Store raw values
        features_norm = {}  # Store domain-normalized values
        labels = {'match': match} if match is not None else {}
        
        # If pair data is not available, return empty features
        if not pair_data:
            return pair_id, features, labels
        
        # Get vectors for both records
        left_vectors = pair_data.get('left_vectors', {})
        right_vectors = pair_data.get('right_vectors', {})
        
        # Calculate cosine similarity features
        for field in self.cosine_similarities:
            if field in left_vectors and field in right_vectors:
                # Calculate cosine similarity (raw value in [-1,1] range)
                raw_similarity = self._compute_cosine_similarity(
                    left_vectors[field],
                    right_vectors[field]
                )
                
                # Store raw value
                features_raw[f"{field}_cosine"] = raw_similarity
                
                # Normalize cosine similarity from [-1, 1] to [0, 1]
                normalized_similarity = (raw_similarity + 1) / 2
                features_norm[f"{field}_cosine"] = normalized_similarity
                
                # Initial value for features - will be replaced by StandardScaler if enabled
                features[f"{field}_cosine"] = normalized_similarity
                
                # Add title_cosine_squared feature if it's the title field and the feature is enabled
                if field == 'title' and self.config['features'].get('title_cosine_squared', {}).get('enabled', False):
                    # Store all three versions
                    features_raw[f"title_cosine_squared"] = raw_similarity ** 2
                    features_norm[f"title_cosine_squared"] = normalized_similarity ** 2
                    features[f"title_cosine_squared"] = normalized_similarity ** 2  # Will be scaled later
                
                # When processing the composite field, calculate the low_composite_penalty
                if field == 'composite':
                    # Get configuration 
                    low_composite_config = self.config['features'].get('low_composite_penalty', {})
                    enabled = low_composite_config.get('enabled', False)
                    threshold = low_composite_config.get('threshold', 0.50)
                    
                    # Use the normalized value for the penalty calculation
                    if enabled:
                        # Calculate penalty using normalized value
                        penalty_value = 1.0 if normalized_similarity < threshold else 0.0
                        
                        # Store in all representations for consistency
                        features_raw[f"low_composite_penalty"] = penalty_value
                        features_norm[f"low_composite_penalty"] = penalty_value
                        features[f"low_composite_penalty"] = penalty_value
                        
                        # Log when applied
                        if penalty_value == 1.0:
                            logger.info(f"Low composite penalty applied for {pair_id}: normalized={normalized_similarity:.4f} < {threshold}")
        
        # Apply prefilters if configured - BUT ONLY USE FOR LABELING, not for controlling feature calculation
        prefilter_result = self._apply_prefilters(left_vectors, right_vectors, features_norm)
        if prefilter_result:
            labels['prefiltered'] = True
            labels['prefilter_match'] = prefilter_result == 'match'
        
        # Calculate string similarity features - NOW OUTSIDE THE PREFILTER CHECK
        for field_config in self.string_similarities:
            field = field_config['field']
            metrics = field_config['metrics']
            
            # Get left and right field hashes
            left_hash = None
            right_hash = None
            
            if 'hashes' in pair_data:
                left_hash = pair_data['hashes'].get('left', {}).get(field)
                right_hash = pair_data['hashes'].get('right', {}).get(field)
            
            # If hashes are available, get the string values
            left_string = None
            right_string = None
            
            if left_hash and left_hash in self.unique_strings:
                left_string = self.unique_strings[left_hash]
            
            if right_hash and right_hash in self.unique_strings:
                right_string = self.unique_strings[right_hash]
            
            # If we have both strings, calculate string similarity
            if left_string and right_string:
                for metric in metrics:
                    if metric == 'levenshtein':
                        # Calculate Levenshtein distance
                        distance = Levenshtein.distance(left_string, right_string)
                        max_len = max(len(left_string), len(right_string))
                        
                        # Convert to similarity score (0-1 range)
                        similarity = 1.0 - (distance / max_len) if max_len > 0 else 1.0
                        
                        # Levenshtein similarity is already normalized (0-1)
                        features[f"{field}_levenshtein"] = similarity
                        features_norm[f"{field}_levenshtein"] = similarity
                    
                    elif metric == 'jaro_winkler':
                        # Jaro-Winkler similarity is already normalized (0-1)
                        similarity = jellyfish.jaro_winkler_similarity(left_string, right_string)
                        features[f"{field}_jaro_winkler"] = similarity
                        features_norm[f"{field}_jaro_winkler"] = similarity
        
        # Calculate harmonic mean features - NOW OUTSIDE THE PREFILTER CHECK
        for field1, field2 in self.harmonic_means:
            # Get normalized cosine similarities for both fields
            sim1 = features_norm.get(f"{field1}_cosine")
            sim2 = features_norm.get(f"{field2}_cosine")
            
            if sim1 is not None and sim2 is not None:
                # Calculate harmonic mean
                # Since we've normalized the input similarities to 0-1 range,
                # the harmonic mean will also be in 0-1 range
                harmonic_mean = compute_harmonic_mean(sim1, sim2)
                features[f"{field1}_{field2}_harmonic"] = harmonic_mean
                features_norm[f"{field1}_{field2}_harmonic"] = harmonic_mean
        
        # Calculate additional interaction features - NOW OUTSIDE THE PREFILTER CHECK
        for interaction in self.additional_interactions:
            interaction_type = interaction['type']
            fields = interaction['fields']
            
            if len(fields) == 2:
                field1, field2 = fields
                
                # Get normalized cosine similarities for both fields
                sim1 = features_norm.get(f"{field1}_cosine")
                sim2 = features_norm.get(f"{field2}_cosine")
                
                if sim1 is not None and sim2 is not None:
                    if interaction_type == 'product':
                        # Calculate product
                        # Product of two 0-1 values will remain in 0-1 range
                        product = sim1 * sim2
                        features[f"{field1}_{field2}_product"] = product
                        features_norm[f"{field1}_{field2}_product"] = product
                    
                    elif interaction_type == 'ratio':
                        # Calculate ratio
                        if sim2 > 0:
                            ratio = sim1 / sim2
                            
                            # Normalize the ratio to 0-1 range
                            # Use a sigmoid-like function to map arbitrary ratios to 0-1
                            # This ensures very large ratios don't dominate
                            normalized_ratio = 2 / (1 + np.exp(-ratio)) - 1
                            features[f"{field1}_{field2}_ratio"] = normalized_ratio
                            features_norm[f"{field1}_{field2}_ratio"] = normalized_ratio
                        else:
                            # Avoid division by zero
                            features[f"{field1}_{field2}_ratio"] = 0.0
                            features_norm[f"{field1}_{field2}_ratio"] = 0.0

        # Calculate birth/death year features - NOW OUTSIDE THE PREFILTER CHECK
        birth_death_config = self.config['features'].get('birth_death_features', {})
        if birth_death_config.get('enabled', False):
            # Get person strings for both records
            left_person = None
            right_person = None
            
            if 'hashes' in pair_data:
                left_hash = pair_data['hashes'].get('left', {}).get('person')
                right_hash = pair_data['hashes'].get('right', {}).get('person')
                
                if left_hash and left_hash in self.unique_strings:
                    left_person = self.unique_strings[left_hash]
                
                if right_hash and right_hash in self.unique_strings:
                    right_person = self.unique_strings[right_hash]
            
            # Extract birth/death years
            left_birth, left_death = None, None
            right_birth, right_death = None, None
            
            if left_person:
                left_birth, left_death = extract_birth_death_years(left_person)
            
            if right_person:
                right_birth, right_death = extract_birth_death_years(right_person)
            
            # Feature 1: birth_death_left - binary indicator if left record has birth/death info
            features['birth_death_left'] = 1.0 if (left_birth or left_death) else 0.0
            features_norm['birth_death_left'] = 1.0 if (left_birth or left_death) else 0.0
            
            # Feature 2: birth_death_right - binary indicator if right record has birth/death info
            features['birth_death_right'] = 1.0 if (right_birth or right_death) else 0.0
            features_norm['birth_death_right'] = 1.0 if (right_birth or right_death) else 0.0
            
            # Feature 3: birth_death_match - binary indicator if birth/death years match
            birth_match = (left_birth and right_birth and left_birth == right_birth)
            death_match = (left_death and right_death and left_death == right_death)
            features['birth_death_match'] = 1.0 if (birth_match or death_match) else 0.0
            features_norm['birth_death_match'] = 1.0 if (birth_match or death_match) else 0.0

            # Calculate the person_levenshtein_birth_death_match_product feature
            if ('person_levenshtein' in features and 
                'birth_death_match' in features and
                self.config['features'].get('person_levenshtein_birth_death_match_product', {}).get('enabled', False)):
                
                # Get the base Levenshtein similarity (already in [0,1] range)
                levenshtein_sim = features['person_levenshtein']
                
                # Get birth/death match status (1.0 if dates match, 0.0 if not)
                birth_death_match = features['birth_death_match']
                
                # Get dampening factor for non-matching dates (configurable)
                dampening_factor = self.config['features']['person_levenshtein_birth_death_match_product'].get('dampening_factor', 0.25)
                
                # Calculate the composite feature
                if birth_death_match == 1.0:
                    # If birth/death dates match, keep full Levenshtein similarity
                    product_value = levenshtein_sim
                else:
                    # If birth/death dates don't match, dampen the Levenshtein similarity
                    product_value = levenshtein_sim * dampening_factor
                    
                # Store in all representations
                features_raw['person_levenshtein_birth_death_match_product'] = product_value
                features_norm['person_levenshtein_birth_death_match_product'] = product_value
                features['person_levenshtein_birth_death_match_product'] = product_value

            # Calculate the person_cosine_birth_death_match_product feature
            if ('person_cosine' in features_norm and 
                'birth_death_match' in features and
                self.config['features'].get('person_cosine_birth_death_match_product', {}).get('enabled', False)):
                
                # Get the cosine similarity (already in [0,1] range from normalization)
                cosine_sim = features_norm['person_cosine']
                
                # Get birth/death match status (1.0 if dates match, 0.0 if not)
                birth_death_match = features['birth_death_match']
                
                # Get dampening factor for non-matching dates (configurable)
                dampening_factor = self.config['features']['person_cosine_birth_death_match_product'].get('dampening_factor', 0.25)
                
                # Calculate the composite feature
                if birth_death_match == 1.0:
                    # If birth/death dates match, keep full cosine similarity
                    product_value = cosine_sim
                else:
                    # If birth/death dates don't match, dampen the cosine similarity
                    product_value = cosine_sim * dampening_factor
                    
                # Store in all representations
                features_raw['person_cosine_birth_death_match_product'] = product_value
                features_norm['person_cosine_birth_death_match_product'] = product_value
                features['person_cosine_birth_death_match_product'] = product_value
        
        # Always return all data regardless of prefilter status
        return pair_id, features, labels, features_raw, features_norm
    
    def _compute_cosine_similarity(self, vec1, vec2):
        """Compute cosine similarity between two vectors."""
        # Log vector info
        vec1_info = f"type={type(vec1)}, length={len(vec1) if hasattr(vec1, '__len__') else 'N/A'}"
        vec2_info = f"type={type(vec2)}, length={len(vec2) if hasattr(vec2, '__len__') else 'N/A'}"
        print(f"Computing cosine similarity: vec1={vec1_info}, vec2={vec2_info}")
        
        if not vec1 or not vec2:
            print("Empty vector detected - returning 0")
            return 0.0
        
        # Convert to numpy arrays
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        # Compute dot product
        dot_product = np.dot(vec1, vec2)
        
        # Compute magnitudes
        mag1 = np.linalg.norm(vec1)
        mag2 = np.linalg.norm(vec2)
        
        print(f"Dot product: {dot_product}, Magnitudes: mag1={mag1}, mag2={mag2}")
        
        # Compute cosine similarity
        if mag1 == 0.0 or mag2 == 0.0:
            print("Zero magnitude detected - returning 0")
            return 0.0
        
        similarity = dot_product / (mag1 * mag2)
        print(f"Calculated similarity: {similarity}")
        
        return similarity
    
    def _apply_feature_selection(self, feature_set):
        """
        Apply feature selection based on configuration.
        
        Args:
            feature_set (dict): Dictionary of pair ID -> feature vectors
            
        Returns:
            dict: Feature set with selected features
        """
        # Skip if feature selection is disabled
        if not self.config.get('feature_selection', {}).get('enabled', False):
            logger.info("Feature selection disabled - using all features")
            return feature_set

        # Get configuration
        selection_config = self.config['feature_selection']
        selection_mode = selection_config.get('mode', 'include')
        base_features = set(selection_config.get('base_features', []))
        interaction_features = set(selection_config.get('interaction_features', []))
        
        # Handle feature groups
        if selection_config.get('include_all_cosine', False):
            base_features.update([f for f in self.feature_names if f.endswith('_cosine')])
        if selection_config.get('include_all_levenshtein', False):
            base_features.update([f for f in self.feature_names if f.endswith('_levenshtein')])
        if selection_config.get('include_all_harmonic', False):
            interaction_features.update([f for f in self.feature_names if f.endswith('_harmonic')])
        if selection_config.get('include_all_product', False):
            interaction_features.update([f for f in self.feature_names if f.endswith('_product')])
        if selection_config.get('include_all_ratio', False):
            interaction_features.update([f for f in self.feature_names if f.endswith('_ratio')])
        if selection_config.get('include_all_birth_death', False):
            base_features.update([f for f in self.feature_names if f.startswith('birth_death')])
        
        # Custom features are always kept if configured
        if selection_config.get('keep_custom_features', True):
            custom_feature_patterns = [
                'low_composite_penalty',
                #'person_levenshtein_birth_death_match_product',
                #'person_cosine_birth_death_match_product',
                # Add any other custom features here
            ]
            for pattern in custom_feature_patterns:
                base_features.update([f for f in self.feature_names if pattern in f])
        
        # Build the overall feature set
        selected_features = base_features.union(interaction_features)
        
        # Auto-include dependencies if needed
        if selection_config.get('auto_include_dependencies', False):
            base_dependencies = set()
            for interaction in interaction_features:
                # Extract base feature names from interaction features
                parts = interaction.split('_')
                if len(parts) >= 3:  # e.g., "person_title_harmonic"
                    if parts[-1] in ['harmonic', 'product', 'ratio']:
                        # For these interactions, the base features are the field names with "_cosine" suffix
                        field1 = parts[0]
                        field2 = parts[1]
                        base_dependencies.add(f"{field1}_cosine")
                        base_dependencies.add(f"{field2}_cosine")
            
            # Add the dependencies to selected features
            selected_features.update(base_dependencies)
        
        # Log the feature selection
        logger.info(f"Selected {len(selected_features)} features for training")
        logger.info(f"Base features: {sorted(list(base_features))}")
        logger.info(f"Interaction features: {sorted(list(interaction_features))}")
        
        # Update the feature vectors
        for pair_id, data in feature_set.items():
            # Process all feature representations
            for rep in ['features', 'features_norm', 'features_raw', 'features_std']:
                if rep in data:
                    # Apply feature selection
                    if selection_mode == 'include':
                        # Include mode - keep only selected features
                        data[rep] = {
                            feature: value for feature, value in data[rep].items()
                            if feature in selected_features
                        }
                    else:  # exclude mode
                        # Exclude mode - remove specified features
                        data[rep] = {
                            feature: value for feature, value in data[rep].items()
                            if feature not in selected_features
                        }
        
        # Update feature names to reflect selection
        all_remaining_features = set()
        for pair_id, data in list(feature_set.items())[:5]:  # Sample first 5 pairs
            if 'features' in data:
                all_remaining_features.update(data['features'].keys())
        
        self.feature_names = sorted(list(all_remaining_features))
        logger.info(f"Updated feature names list to {len(self.feature_names)} features")
        
        return feature_set

    def _apply_prefilters(self, left_vectors, right_vectors, features):
        """
        Apply prefilters to automatically classify pairs.
        
        Args:
            left_vectors (dict): Left record vectors
            right_vectors (dict): Right record vectors
            features (dict): Feature vector
            
        Returns:
            str: 'match', 'non_match', or None if no prefilter applies
        """
        # Check if birth/death features are enabled and configured as prefilter
        birth_death_config = self.config['features'].get('birth_death_features', {})
        if birth_death_config.get('enabled', False) and birth_death_config.get('use_as_prefilter', False):
            # If we already computed the birth_death_match feature, use it
            if 'birth_death_match' in features and features['birth_death_match'] == 1.0:
                # We need to also check that names are sufficiently similar
                if 'person_cosine' in features and features['person_cosine'] > 0.5:
                    logger.info(f"Birth/death match prefilter applied - classified as match")
                    return 'match'
        
        # Legacy prefilter support - apply direct birth/death year extraction
        if self.config['features']['prefilters'].get('exact_name_birth_death_prefilter', False):
            try:
                if 'person' in left_vectors and 'person' in right_vectors:
                    # Get person field hashes
                    left_hash = None
                    right_hash = None
                    
                    if 'hashes' in pair_data:
                        left_hash = pair_data['hashes'].get('left', {}).get('person')
                        right_hash = pair_data['hashes'].get('right', {}).get('person')
                    
                    # If we have the hashes, get the string values
                    left_person = None
                    right_person = None
                    
                    if left_hash and left_hash in self.unique_strings:
                        left_person = self.unique_strings[left_hash]
                    
                    if right_hash and right_hash in self.unique_strings:
                        right_person = self.unique_strings[right_hash]
                    
                    # If we have both person strings, extract birth/death years
                    if left_person and right_person:
                        # Extract birth/death years
                        left_birth, left_death = extract_birth_death_years(left_person)
                        right_birth, right_death = extract_birth_death_years(right_person)
                        
                        # Normalize names (remove dates)
                        left_normalized = normalize_name(left_person)
                        right_normalized = normalize_name(right_person)
                        
                        # Check if normalized names match and have matching birth/death years
                        if (left_normalized == right_normalized and
                            ((left_birth and right_birth and left_birth == right_birth) or
                            (left_death and right_death and left_death == right_death))):
                            logger.info(f"Exact name birth/death prefilter applied - found match: '{left_person}' and '{right_person}'")
                            return 'match'
            except Exception as e:
                # Add detailed error logging to catch any issues with the birth/death extraction
                logger.error(f"Error in exact name birth/death prefilter: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Apply composite cosine prefilter
        if (self.composite_cosine_prefilter['enabled'] and
            'composite_cosine' in features):
            
            threshold = self.composite_cosine_prefilter['threshold']
            if features['composite_cosine'] >= threshold:
                return 'match'
        
        # Apply person cosine prefilter
        if (self.person_cosine_prefilter['enabled'] and
            'person_cosine' in features):
            
            threshold = self.person_cosine_prefilter['threshold']
            if features['person_cosine'] < threshold:
                return 'non_match'
        
        # No prefilter applies
        return None
    
    def _normalize_features(self, feature_set):
        """
        Normalize feature vectors using StandardScaler while preserving
        the original raw and normalized values.
        
        Args:
            feature_set (dict): Dictionary of pair ID -> feature vector
            
        Returns:
            dict: Normalized feature set with multiple representations
        """
        # Extract feature vectors
        pair_ids = []
        feature_vectors = []
        
        for pair_id, data in feature_set.items():
            features = data['features']
            
            # Skip pairs with empty features
            if not features:
                continue
            
            # Convert to list in consistent order
            feature_vector = [features.get(name, 0.0) for name in self.feature_names]
            
            pair_ids.append(pair_id)
            feature_vectors.append(feature_vector)
        
        # Convert to numpy array
        X = np.array(feature_vectors)
        
        # Before normalization, store raw and normalized features
        for i, pair_id in enumerate(pair_ids):
            # Store normalized features (from the domain-specific normalization)
            if 'features_norm' not in feature_set[pair_id]:
                feature_set[pair_id]['features_norm'] = {}
                
            # Store raw features 
            if 'features_raw' not in feature_set[pair_id]:
                feature_set[pair_id]['features_raw'] = {}
        
        # Only apply StandardScaler if normalize_features is enabled
        if self.normalize_features:
            logger.info("Applying StandardScaler normalization to features")
            
            # Fit scaler if not already fit
            if not self.scaler:
                self.scaler = StandardScaler()
                self.scaler.fit(X)
                logger.info(f"Fitted StandardScaler with means: {self.scaler.mean_}")
                logger.info(f"Fitted StandardScaler with stds: {self.scaler.scale_}")
            
            # Transform feature vectors
            X_scaled = self.scaler.transform(X)
            
            # Update feature set with scaled values
            for i, pair_id in enumerate(pair_ids):
                # Create new features dictionary for StandardScaler values
                scaled_features = {}
                for j, name in enumerate(self.feature_names):
                    scaled_features[name] = X_scaled[i, j]
                
                # Update feature set - 'features' will be used for model training
                feature_set[pair_id]['features'] = scaled_features
                # Also store as 'features_std' for clarity
                feature_set[pair_id]['features_std'] = scaled_features.copy()
            
            logger.info("StandardScaler applied to features")
        else:
            logger.info("StandardScaler normalization disabled - using domain-normalized features")
            # If StandardScaler is disabled, use the normalized features
            for i, pair_id in enumerate(pair_ids):
                # Create a copy of the normalized features for the std version
                feature_set[pair_id]['features_std'] = feature_set[pair_id]['features'].copy()
        
        feature_set = self._apply_feature_selection(feature_set)

        return feature_set
    
    def _perform_recursive_feature_elimination(self, feature_set):
        """
        Perform recursive feature elimination to select best features.
        
        Args:
            feature_set (dict): Dictionary of pair ID -> feature vector
            
        Returns:
            dict: Feature set with selected features
        """
        # Extract feature vectors and labels
        pair_ids = []
        feature_vectors = []
        labels = []
        
        for pair_id, data in feature_set.items():
            features = data['features']
            match = data['labels'].get('match')
            
            # Skip pairs with empty features or no match label
            if not features or match is None:
                continue
            
            # Convert to list in consistent order
            feature_vector = [features.get(name, 0.0) for name in self.feature_names]
            
            pair_ids.append(pair_id)
            feature_vectors.append(feature_vector)
            labels.append(1 if match else 0)
        
        # Convert to numpy arrays
        X = np.array(feature_vectors)
        y = np.array(labels)
        
        # Create and fit RFE model
        if not self.rfe:
            estimator = LogisticRegression(
                penalty=self.config['classification']['regularization'],
                C=self.config['classification']['regularization_strength'],
                max_iter=self.config['classification']['max_iterations'],
                class_weight=self.config['classification']['class_weight']
            )
            
            step_size = self.config['features']['rfe_step_size']
            cv_folds = self.config['features']['rfe_cv_folds']
            
            from sklearn.feature_selection import RFECV
            self.rfe = RFECV(
                estimator=estimator,
                step=step_size,
                cv=cv_folds,
                scoring='f1'
            )
            
            self.rfe.fit(X, y)
            
            # Get selected feature names
            selected_features = [
                name for i, name in enumerate(self.feature_names)
                if self.rfe.support_[i]
            ]
            
            logger.info(f"Selected {len(selected_features)}/{len(self.feature_names)} features")
            logger.info(f"Selected features: {selected_features}")
        
        # Filter features based on RFE selection
        for pair_id in feature_set:
            features = feature_set[pair_id]['features']
            
            # Filter features
            filtered_features = {
                name: value
                for name, value in features.items()
                if self.rfe.support_[self.feature_names.index(name)]
            }
            
            # Update feature set
            feature_set[pair_id]['features'] = filtered_features
        
        return feature_set
    
    def _save_ground_truth_features(self, feature_set, processed_pairs):
        """
        Save ground truth feature engineering results.
        
        Args:
            feature_set (dict): Dictionary of pair ID -> feature vector
            processed_pairs (set): Set of processed pair IDs
        """
        # Save feature set
        feature_set_path = self.temp_dir / "ground_truth_features.pkl"
        with open(feature_set_path, 'wb') as f:
            import pickle
            pickle.dump(feature_set, f)
        
        # Save scaler
        scaler_path = None
        if self.scaler:
            scaler_path = self.temp_dir / "feature_scaler.pkl"
            with open(scaler_path, 'wb') as f:
                import pickle
                pickle.dump(self.scaler, f)
        
        # Save RFE
        rfe_path = None
        if self.rfe:
            rfe_path = self.temp_dir / "feature_rfe.pkl"
            with open(rfe_path, 'wb') as f:
                import pickle
                pickle.dump(self.rfe, f)
        
        # Save feature statistics
        feature_stats = self._compute_feature_statistics(feature_set)
        with open(self.output_dir / "feature_statistics.json", 'w') as f:
            json.dump(feature_stats, f, indent=2)
        
        # Save metadata
        with open(self.output_dir / "ground_truth_features_metadata.json", 'w') as f:
            json.dump({
                'feature_set_path': str(feature_set_path),
                'scaler_path': str(scaler_path) if scaler_path else None,
                'rfe_path': str(rfe_path) if rfe_path else None,
                'feature_count': len(self.feature_names),
                'pair_count': len(feature_set),
                'processed_pairs': len(processed_pairs),
                'feature_names': self.feature_names
            }, f, indent=2)
        
        # Save final checkpoint
        checkpoint_path = self.checkpoint_dir / "ground_truth_features_final.ckpt"
        
        checkpoint_state = {
            'feature_set': feature_set,
            'processed_pairs': list(processed_pairs),
            'feature_names': self.feature_names
        }
        
        if self.scaler:
            checkpoint_state['scaler'] = self.scaler
        
        if self.rfe:
            checkpoint_state['rfe'] = self.rfe
        
        save_checkpoint(checkpoint_state, checkpoint_path)
        
        logger.info(f"Ground truth feature engineering results saved to {self.output_dir}")
    
    def _compute_feature_statistics(self, feature_set):
        """
        Compute statistics for features.
        
        Args:
            feature_set (dict): Dictionary of pair ID -> feature vector
            
        Returns:
            dict: Feature statistics
        """
        # Extract feature vectors and labels
        feature_vectors = []
        match_labels = []
        
        for pair_id, data in feature_set.items():
            features = data['features']
            match = data['labels'].get('match')
            
            # Skip pairs with empty features or no match label
            if not features or match is None:
                continue
            
            # Add to lists
            feature_vectors.append(features)
            match_labels.append(match)
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(feature_vectors)
        df['match'] = match_labels
        
        # Compute statistics for each feature
        feature_stats = {}
        
        for feature in df.columns:
            if feature == 'match':
                continue
            
            # Compute statistics for matches and non-matches
            match_values = df[df['match']][feature]
            non_match_values = df[~df['match']][feature]
            
            feature_stats[feature] = {
                'overall': {
                    'mean': df[feature].mean(),
                    'median': df[feature].median(),
                    'min': df[feature].min(),
                    'max': df[feature].max(),
                    'std': df[feature].std()
                },
                'match': {
                    'mean': match_values.mean(),
                    'median': match_values.median(),
                    'min': match_values.min(),
                    'max': match_values.max(),
                    'std': match_values.std()
                },
                'non_match': {
                    'mean': non_match_values.mean(),
                    'median': non_match_values.median(),
                    'min': non_match_values.min(),
                    'max': non_match_values.max(),
                    'std': non_match_values.std()
                }
            }
        
        return feature_stats
