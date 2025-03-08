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
                    pair_id, features, labels = future.result()
                    
                    # Store feature vector
                    if pair_id and features:
                        batch_features[pair_id] = {
                            'features': features,
                            'labels': labels
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
        Compute features for a single pair.
        
        Args:
            left_id (str): Left record ID
            right_id (str): Right record ID
            match (bool): Whether the pair is a match
            pair_data (dict): Pair vector data
            
        Returns:
            tuple: (pair_id, features, labels)
        """
        pair_id = f"{left_id}_{right_id}"
        
        # Initialize feature vector
        features = {}
        
        # Initialize labels
        labels = {'match': match} if match is not None else {}
        
        # If pair data is not available, return empty features
        if not pair_data:
            # Try to load vectors for this pair
            pass  # TODO: Implement on-demand vector loading if needed
        
        # Get vectors for both records
        if pair_data:
            left_vectors = pair_data.get('left_vectors', {})
            right_vectors = pair_data.get('right_vectors', {})
            
            # Calculate cosine similarity features
            for field in self.cosine_similarities:
                if field in left_vectors and field in right_vectors:
                    # Calculate cosine similarity
                    similarity = self._compute_cosine_similarity(
                        left_vectors[field],
                        right_vectors[field]
                    )
                    
                    features[f"{field}_cosine"] = similarity
            
            # Apply prefilters if configured
            prefilter_result = self._apply_prefilters(left_vectors, right_vectors, features)
            if prefilter_result:
                labels['prefiltered'] = True
                labels['prefilter_match'] = prefilter_result == 'match'
            
            # Calculate string similarity features
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
                            distance = Levenshtein.distance(left_string, right_string)
                            max_len = max(len(left_string), len(right_string))
                            similarity = 1.0 - (distance / max_len) if max_len > 0 else 1.0
                            features[f"{field}_levenshtein"] = similarity
                        
                        elif metric == 'jaro_winkler':
                            similarity = jellyfish.jaro_winkler_similarity(left_string, right_string)
                            features[f"{field}_jaro_winkler"] = similarity
            
            # Calculate harmonic mean features
            for field1, field2 in self.harmonic_means:
                if (field1 in left_vectors and field1 in right_vectors and
                    field2 in left_vectors and field2 in right_vectors):
                    
                    # Get cosine similarities for both fields
                    sim1 = features.get(f"{field1}_cosine")
                    sim2 = features.get(f"{field2}_cosine")
                    
                    if sim1 is not None and sim2 is not None:
                        # Calculate harmonic mean
                        harmonic_mean = compute_harmonic_mean(sim1, sim2)
                        features[f"{field1}_{field2}_harmonic"] = harmonic_mean
            
            # Calculate additional interaction features
            for interaction in self.additional_interactions:
                interaction_type = interaction['type']
                fields = interaction['fields']
                
                if len(fields) == 2:
                    field1, field2 = fields
                    
                    if (field1 in left_vectors and field1 in right_vectors and
                        field2 in left_vectors and field2 in right_vectors):
                        
                        # Get cosine similarities for both fields
                        sim1 = features.get(f"{field1}_cosine")
                        sim2 = features.get(f"{field2}_cosine")
                        
                        if sim1 is not None and sim2 is not None:
                            if interaction_type == 'product':
                                # Calculate product
                                product = sim1 * sim2
                                features[f"{field1}_{field2}_product"] = product
                            
                            elif interaction_type == 'ratio':
                                # Calculate ratio
                                ratio = sim1 / sim2 if sim2 > 0 else 0.0
                                features[f"{field1}_{field2}_ratio"] = ratio
        
        return pair_id, features, labels
    
    def _compute_cosine_similarity(self, vec1, vec2):
        """
        Compute cosine similarity between two vectors.
        
        Args:
            vec1 (list): First vector
            vec2 (list): Second vector
            
        Returns:
            float: Cosine similarity
        """
        if not vec1 or not vec2:
            return 0.0
        
        # Convert to numpy arrays
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        # Compute dot product
        dot_product = np.dot(vec1, vec2)
        
        # Compute magnitudes
        mag1 = np.linalg.norm(vec1)
        mag2 = np.linalg.norm(vec2)
        
        # Compute cosine similarity
        if mag1 == 0.0 or mag2 == 0.0:
            return 0.0
        
        return dot_product / (mag1 * mag2)
    
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
        # Apply exact name birth death prefilter
        if self.exact_name_birth_death_prefilter:
            if 'person' in left_vectors and 'person' in right_vectors:
                # Get person field hashes
                left_hash = None
                right_hash = None
                
                if 'hashes' in left_vectors:
                    left_hash = left_vectors['hashes'].get('person')
                
                if 'hashes' in right_vectors:
                    right_hash = right_vectors['hashes'].get('person')
                
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
                    
                    # Normalize names
                    left_normalized = normalize_name(left_person)
                    right_normalized = normalize_name(right_person)
                    
                    # Check if normalized names match and have birth/death years
                    if (left_normalized == right_normalized and
                        ((left_birth and right_birth and left_birth == right_birth) or
                        (left_death and right_death and left_death == right_death))):
                        return 'match'
        
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
    
    # The rest of the class remains the same...
    # (This includes methods like _normalize_features, _perform_recursive_feature_elimination, etc.)
    
    def _normalize_features(self, feature_set):
        """
        Normalize feature vectors using StandardScaler.
        
        Args:
            feature_set (dict): Dictionary of pair ID -> feature vector
            
        Returns:
            dict: Normalized feature set
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
        
        # Fit scaler if not already fit
        if not self.scaler:
            self.scaler = StandardScaler()
            self.scaler.fit(X)
        
        # Transform feature vectors
        X_scaled = self.scaler.transform(X)
        
        # Update feature set
        for i, pair_id in enumerate(pair_ids):
            # Convert scaled vector back to dictionary
            scaled_features = {}
            for j, name in enumerate(self.feature_names):
                scaled_features[name] = X_scaled[i, j]
            
            # Update feature set
            feature_set[pair_id]['features'] = scaled_features
        
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
