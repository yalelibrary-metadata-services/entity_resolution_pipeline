"""
Tests for the entity resolution pipeline.

These tests verify the functionality of the pipeline components.
"""

import os
import sys
import unittest
import yaml
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.batch_parallel_preprocessing import Preprocessor
from src.batch_parallel_embedding import Embedder
from src.batch_parallel_indexing import Indexer
from src.batch_parallel_imputation import Imputator
from src.batch_parallel_querying import QueryEngine
from src.batch_parallel_feature_engineering import FeatureEngineer
from src.batch_parallel_classification import Classifier
from src.pipeline import Pipeline
from src.utils import compute_string_hash, compute_harmonic_mean

class TestPipeline(unittest.TestCase):
    """
    Test suite for the entity resolution pipeline.
    """
    
    @classmethod
    def setUpClass(cls):
        """
        Set up test environment.
        """
        # Load configuration
        with open('config.yml', 'r') as f:
            cls.config = yaml.safe_load(f)
        
        # Set test mode
        cls.config['system']['mode'] = 'dev'
        
        # Set up test directories
        cls.test_dir = Path('test_output')
        cls.test_dir.mkdir(exist_ok=True)
        
        cls.config['system']['output_dir'] = str(cls.test_dir / 'output')
        cls.config['system']['checkpoint_dir'] = str(cls.test_dir / 'checkpoints')
        cls.config['system']['temp_dir'] = str(cls.test_dir / 'temp')
        
        # Create test directories
        for dir_path in [
            cls.config['system']['output_dir'],
            cls.config['system']['checkpoint_dir'],
            cls.config['system']['temp_dir']
        ]:
            os.makedirs(dir_path, exist_ok=True)
    
    def test_string_hash(self):
        """
        Test string hash function.
        """
        # Test with various inputs
        self.assertEqual(compute_string_hash('test'), compute_string_hash('test'))
        self.assertNotEqual(compute_string_hash('test'), compute_string_hash('test2'))
        self.assertEqual(compute_string_hash(''), compute_string_hash(''))
    
    def test_harmonic_mean(self):
        """
        Test harmonic mean function.
        """
        # Test with various inputs
        self.assertEqual(compute_harmonic_mean(2, 2), 2)
        self.assertEqual(compute_harmonic_mean(1, 3), 1.5)
        self.assertEqual(compute_harmonic_mean(0, 2), 0)
    
    def test_pipeline_initialization(self):
        """
        Test pipeline initialization.
        """
        pipeline = Pipeline(self.config)
        
        # Check that pipeline components are initialized
        self.assertIsInstance(pipeline.preprocessor, Preprocessor)
        self.assertIsInstance(pipeline.embedder, Embedder)
        self.assertIsInstance(pipeline.indexer, Indexer)
        self.assertIsInstance(pipeline.imputator, Imputator)
        self.assertIsInstance(pipeline.query_engine, QueryEngine)
        self.assertIsInstance(pipeline.feature_engineer, FeatureEngineer)
        self.assertIsInstance(pipeline.classifier, Classifier)
    
    def test_preprocessor_initialization(self):
        """
        Test preprocessor initialization.
        """
        preprocessor = Preprocessor(self.config)
        
        # Check that preprocessor is initialized correctly
        self.assertEqual(preprocessor.input_dir, Path(self.config['data']['input_dir']))
        self.assertEqual(preprocessor.output_dir, Path(self.config['system']['output_dir']))
        self.assertEqual(preprocessor.temp_dir, Path(self.config['system']['temp_dir']))
        self.assertEqual(preprocessor.checkpoint_dir, Path(self.config['system']['checkpoint_dir']))
    
    def test_embedder_initialization(self):
        """
        Test embedder initialization.
        """
        embedder = Embedder(self.config)
        
        # Check that embedder is initialized correctly
        self.assertEqual(embedder.model, self.config['embedding']['model'])
        self.assertEqual(embedder.dimensions, self.config['embedding']['dimensions'])
        self.assertEqual(embedder.batch_size, self.config['embedding']['batch_size'])
    
    def test_indexer_initialization(self):
        """
        Test indexer initialization.
        """
        indexer = Indexer(self.config)
        
        # Check that indexer is initialized correctly
        self.assertEqual(indexer.weaviate_host, self.config['weaviate']['host'])
        self.assertEqual(indexer.weaviate_port, self.config['weaviate']['port'])
        self.assertEqual(indexer.collection_name, self.config['weaviate']['collection_name'])
    
    def test_imputator_initialization(self):
        """
        Test imputator initialization.
        """
        imputator = Imputator(self.config)
        
        # Check that imputator is initialized correctly
        self.assertEqual(imputator.fields_to_impute, self.config['imputation']['fields_to_impute'])
        self.assertEqual(imputator.similarity_threshold, self.config['imputation']['vector_similarity_threshold'])
        self.assertEqual(imputator.max_candidates, self.config['imputation']['max_candidates'])
        self.assertEqual(imputator.imputation_method, self.config['imputation']['imputation_method'])
    
    def test_query_engine_initialization(self):
        """
        Test query engine initialization.
        """
        query_engine = QueryEngine(self.config)
        
        # Check that query engine is initialized correctly
        self.assertEqual(query_engine.weaviate_host, self.config['weaviate']['host'])
        self.assertEqual(query_engine.weaviate_port, self.config['weaviate']['port'])
        self.assertEqual(query_engine.collection_name, self.config['weaviate']['collection_name'])
    
    def test_feature_engineer_initialization(self):
        """
        Test feature engineer initialization.
        """
        feature_engineer = FeatureEngineer(self.config)
        
        # Check that feature engineer is initialized correctly
        self.assertEqual(feature_engineer.cosine_similarities, self.config['features']['cosine_similarities'])
        self.assertEqual(feature_engineer.string_similarities, self.config['features']['string_similarities'])
        self.assertEqual(feature_engineer.harmonic_means, self.config['features']['harmonic_means'])
        self.assertEqual(feature_engineer.additional_interactions, self.config['features']['additional_interactions'])
    
    def test_classifier_initialization(self):
        """
        Test classifier initialization.
        """
        classifier = Classifier(self.config)
        
        # Check that classifier is initialized correctly
        self.assertEqual(classifier.algorithm, self.config['classification']['algorithm'])
        self.assertEqual(classifier.regularization, self.config['classification']['regularization'])
        self.assertEqual(classifier.learning_rate, self.config['classification']['learning_rate'])
        self.assertEqual(classifier.max_iterations, self.config['classification']['max_iterations'])
    
    @classmethod
    def tearDownClass(cls):
        """
        Clean up test environment.
        """
        # Optionally remove test directories
        # import shutil
        # shutil.rmtree(cls.test_dir)
        pass

if __name__ == '__main__':
    unittest.main()
