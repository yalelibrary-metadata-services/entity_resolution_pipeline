"""
Taxonomy-based dissimilarity feature for entity resolution.

This module implements a dissimilarity metric based on SetFit taxonomy classifications
to help differentiate between entities with similar names but different domains.
"""

import os
import json
import logging
from typing import Dict, Set, Optional, Tuple
from collections import defaultdict
import pandas as pd

logger = logging.getLogger(__name__)


class TaxonomyDissimilarity:
    """
    Calculate dissimilarity between entities based on their taxonomy classifications.
    Uses hierarchical distance in the 2-level taxonomy structure.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the taxonomy dissimilarity calculator.
        
        Args:
            config: Configuration dictionary containing paths
        """
        self.config = config
        
        # Load paths from config
        taxonomy_path = config.get('taxonomy_path', 'data/input/revised_taxonomy_final.json')
        classified_path = config.get('classified_data_path', 'data/input/training/tmp/training_dataset_classified_2025-07-05.csv')
        
        # Load taxonomy first and build parent-child map
        self.taxonomy = self._load_taxonomy(taxonomy_path)
        self.parent_child_map = self._build_parent_child_map()
        
        # Load classified data which contains personId → identity → categories mapping
        self.person_to_identity, self.identity_categories = self._load_classified_data(classified_path)
        
        # Cache for dissimilarity calculations
        self.dissimilarity_cache = {}
        
        # Log summary statistics
        logger.info(f"Loaded taxonomy with {len(self.parent_child_map)} parent categories")
        logger.info(f"Total categories: {len(self.child_to_parent) + len(self.parent_child_map)}")
        logger.info(f"Loaded {len(self.person_to_identity)} personId mappings")
        logger.info(f"Loaded classifications for {len(self.identity_categories)} unique identities")
        
        # Log sample mappings for debugging
        if logger.isEnabledFor(logging.DEBUG):
            sample_parents = list(self.parent_child_map.keys())[:2]
            for parent in sample_parents:
                children = list(self.parent_child_map[parent])[:3]
                logger.debug(f"Parent '{parent}' has children: {children}")
        
    def _load_taxonomy(self, path: str) -> Dict:
        """Load the taxonomy from JSON file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.info(f"Successfully loaded taxonomy from {path}")
                return data
        except FileNotFoundError:
            logger.error(f"Taxonomy file not found: {path}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing taxonomy JSON: {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error loading taxonomy: {e}")
            return {}
            
    def _build_parent_child_map(self) -> Dict[str, Set[str]]:
        """
        Build a mapping of parent categories to their children.
        Also creates reverse mapping stored as self.child_to_parent.
        """
        parent_child = {}
        self.child_to_parent = {}
        
        if not self.taxonomy:
            logger.warning("Empty taxonomy, cannot build parent-child map")
            return parent_child
            
        if 'skos:hasTopConcept' not in self.taxonomy:
            logger.warning("Taxonomy missing 'skos:hasTopConcept' field")
            return parent_child
            
        for concept in self.taxonomy['skos:hasTopConcept']:
            # Get parent label
            pref_label = concept.get('skos:prefLabel', {})
            parent_label = pref_label.get('@value', '') if isinstance(pref_label, dict) else ''
            
            if not parent_label:
                logger.warning("Found concept without prefLabel, skipping")
                continue
                
            parent_child[parent_label] = set()
            
            # Get narrower concepts (children)
            narrower = concept.get('skos:narrower', [])
            if not isinstance(narrower, list):
                narrower = [narrower] if narrower else []
                
            for child_concept in narrower:
                child_pref_label = child_concept.get('skos:prefLabel', {})
                child_label = child_pref_label.get('@value', '') if isinstance(child_pref_label, dict) else ''
                
                if child_label:
                    parent_child[parent_label].add(child_label)
                    self.child_to_parent[child_label] = parent_label
                    
        return parent_child
        
    def _load_classified_data(self, path: str) -> Tuple[Dict[str, str], Dict[str, Set[str]]]:
        """
        Load classified data and build two mappings:
        1. personId → identity
        2. identity → set of categories
        
        Returns:
            Tuple of (person_to_identity, identity_categories)
        """
        person_to_identity = {}
        identity_categories = defaultdict(set)
        
        try:
            # Read CSV
            df = pd.read_csv(path, encoding='utf-8', dtype={'identity': str})
            
            # Handle potential missing values
            df = df.dropna(subset=['identity', 'personId', 'mistral_prediction'])
            
            # Build mappings
            for _, row in df.iterrows():
                person_id = str(row['personId'])
                identity = str(float(row['identity'])) if '.' in str(row['identity']) else str(row['identity'])
                category = row['mistral_prediction']
                
                # Map personId to identity
                person_to_identity[person_id] = identity
                
                # Add category to identity's set
                all_categories = identity_categories[identity]
                all_categories.add(category)
            
            # Convert defaultdict to regular dict and filter parent categories if children exist
            filtered_identity_categories = {}
            for identity, all_categories in identity_categories.items():
                # Filter out parent categories if child categories exist
                filtered_categories = set()
                for cat in all_categories:
                    # Check if this is a parent category
                    if cat in self.parent_child_map:
                        # Check if any of its children are in the category set
                        has_child = any(child in all_categories 
                                      for child in self.parent_child_map[cat])
                        if not has_child:
                            # No children present, keep the parent
                            filtered_categories.add(cat)
                    else:
                        # This is a child category, always keep it
                        filtered_categories.add(cat)
                
                filtered_identity_categories[identity] = filtered_categories
                
                # Log filtering results for debugging
                if len(all_categories) != len(filtered_categories):
                    logger.debug(f"Identity {identity}: Filtered {all_categories} to {filtered_categories}")
                
            logger.info(f"Loaded {len(person_to_identity)} personId mappings")
            logger.info(f"Loaded {len(filtered_identity_categories)} identity classifications")
            
            # Log sample for debugging
            if person_to_identity and logger.isEnabledFor(logging.DEBUG):
                sample_person = list(person_to_identity.keys())[0]
                sample_identity = person_to_identity[sample_person]
                logger.debug(f"Sample - PersonId '{sample_person}' → Identity '{sample_identity}' → Categories {filtered_identity_categories.get(sample_identity, set())}")
                
            return person_to_identity, dict(filtered_identity_categories)
            
        except FileNotFoundError:
            logger.error(f"Classified data file not found: {path}")
            return {}, {}
        except pd.errors.EmptyDataError:
            logger.error(f"Classified data file is empty: {path}")
            return {}, {}
        except Exception as e:
            logger.error(f"Error loading classified data: {e}")
            return {}, {}
            
    def get_parent_category(self, category: str) -> Optional[str]:
        """Get the parent category for a given category."""
        # Check if it's already a parent
        if category in self.parent_child_map:
            return category
        # Check if it's a child
        return self.child_to_parent.get(category)
        
    def calculate_category_dissimilarity(self, cat1: str, cat2: str) -> float:
        """
        Calculate dissimilarity between two categories.
        
        Dissimilarity scores:
        - 0.0 if same category
        - 0.3 if sibling categories (same parent)
        - 0.4 if parent-child relationship
        - 0.8 if different parent categories
        - 0.9 if category not found in taxonomy
        
        Args:
            cat1: First category
            cat2: Second category
            
        Returns:
            Dissimilarity score between 0.0 and 1.0
        """
        # Check cache
        cache_key = tuple(sorted([cat1, cat2]))
        if cache_key in self.dissimilarity_cache:
            return self.dissimilarity_cache[cache_key]
            
        # Same category
        if cat1 == cat2:
            score = 0.0
        else:
            # Check for parent-child relationship
            is_parent_child = False
            if cat1 in self.parent_child_map and cat2 in self.parent_child_map[cat1]:
                # cat1 is parent of cat2
                is_parent_child = True
            elif cat2 in self.parent_child_map and cat1 in self.parent_child_map[cat2]:
                # cat2 is parent of cat1
                is_parent_child = True
                
            if is_parent_child:
                # Parent-child relationship - moderate dissimilarity
                # This handles cases like "Arts" (parent) vs "Music" (child)
                score = 0.4
            else:
                # Get parent categories
                parent1 = self.get_parent_category(cat1)
                parent2 = self.get_parent_category(cat2)
                
                # If we can't find parents, assume high dissimilarity
                if parent1 is None or parent2 is None:
                    if parent1 is None:
                        logger.debug(f"Category '{cat1}' not found in taxonomy")
                    if parent2 is None:
                        logger.debug(f"Category '{cat2}' not found in taxonomy")
                    score = 0.9
                # Same parent = sibling categories
                elif parent1 == parent2:
                    score = 0.3
                # Different parents
                else:
                    score = 0.8
                
        # Cache result
        self.dissimilarity_cache[cache_key] = score
        return score
        
    def calculate_entity_dissimilarity(self, person_id1: str, person_id2: str) -> float:
        """
        Calculate dissimilarity between two entities based on their category sets.
        
        For entities with multiple categories, uses the minimum dissimilarity
        across all category pairs to avoid false negatives.
        
        Args:
            person_id1: First personId
            person_id2: Second personId
            
        Returns:
            Dissimilarity score between 0.0 and 1.0
        """
        # Get identities for each personId
        identity1 = self.person_to_identity.get(person_id1)
        identity2 = self.person_to_identity.get(person_id2)
        
        # If either personId is not in training data, return moderate dissimilarity
        if identity1 is None or identity2 is None:
            logger.debug(f"PersonId not found in training data - "
                        f"'{person_id1}': {identity1 is not None}, "
                        f"'{person_id2}': {identity2 is not None}")
            return 0.5  # Neutral value when we can't determine similarity
            
        # Get category sets for each identity
        cats1 = self.identity_categories.get(identity1, set())
        cats2 = self.identity_categories.get(identity2, set())
        
        # If either identity has no categories, return moderate dissimilarity
        if not cats1 or not cats2:
            logger.debug(f"Missing categories - Identity1 '{identity1}': {len(cats1)} cats, "
                        f"Identity2 '{identity2}': {len(cats2)} cats")
            return 0.5  # Neutral value when we can't determine similarity
            
        # Calculate all pairwise dissimilarities
        min_dissim = 1.0
        for cat1 in cats1:
            for cat2 in cats2:
                dissim = self.calculate_category_dissimilarity(cat1, cat2)
                min_dissim = min(min_dissim, dissim)
                
                # Early exit if we find identical categories
                if min_dissim == 0.0:
                    return 0.0
                    
        return min_dissim
        
    def get_feature_value(self, left_id: str, right_id: str) -> float:
        """
        Get the taxonomy dissimilarity feature value for a pair of entities.
        
        This is the main interface for the feature engineering module.
        
        Args:
            left_id: First entity personId
            right_id: Second entity personId
            
        Returns:
            Feature value (dissimilarity score between 0.0 and 1.0)
        """
        return self.calculate_entity_dissimilarity(left_id, right_id)
            
    def get_debug_info(self, person_id: str) -> Dict:
        """
        Get debug information for an entity's categories.
        
        Args:
            person_id: Entity personId
            
        Returns:
            Dictionary with debug information
        """
        try:
            identity = self.person_to_identity.get(person_id)
            
            if identity is None:
                return {
                    'person_id': person_id,
                    'identity': None,
                    'categories': [],
                    'category_count': 0,
                    'note': 'PersonId not found in training data'
                }
                
            categories = self.identity_categories.get(identity, set())
            
            return {
                'person_id': person_id,
                'identity': identity,
                'categories': list(categories),
                'category_count': len(categories)
            }
        except Exception as e:
            return {'error': str(e), 'person_id': person_id}


def create_taxonomy_feature(left_id: str, right_id: str, 
                           taxonomy_calc: TaxonomyDissimilarity,
                           weight: float = 1.0) -> float:
    """
    Feature function for integration with FeatureEngineering class.
    
    Args:
        left_id: First entity personId
        right_id: Second entity personId
        taxonomy_calc: TaxonomyDissimilarity instance
        weight: Feature weight
        
    Returns:
        Weighted taxonomy dissimilarity score
    """
    # Calculate dissimilarity
    dissim = taxonomy_calc.get_feature_value(left_id, right_id)
    
    # Apply weight
    return dissim * weight