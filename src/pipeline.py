"""
Pipeline orchestration module for entity resolution.

This module provides the Pipeline class which coordinates all stages
of the entity resolution pipeline.
"""

import os
import logging
import json
from pathlib import Path
import time

from src.batch_parallel_preprocessing import Preprocessor
from src.batch_parallel_embedding import Embedder
from src.batch_parallel_indexing import Indexer
from src.batch_parallel_imputation import Imputator
from src.batch_parallel_querying import QueryEngine
from src.batch_parallel_feature_engineering import FeatureEngineer
from src.batch_parallel_classification import Classifier
from src.reporting import ReportGenerator
from src.utils import Timer

logger = logging.getLogger(__name__)

class Pipeline:
    """
    Orchestrates the entity resolution pipeline.
    
    Features:
    - Modular execution of pipeline stages
    - Checkpoint management for resumable processing
    - Configuration management
    - Progress tracking and reporting
    """
    
    def __init__(self, config):
        """
        Initialize the pipeline with configuration parameters.
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = config
        
        # Data paths
        self.output_dir = Path(config['system']['output_dir'])
        self.checkpoint_dir = Path(config['system']['checkpoint_dir'])
        
        # Create directories if they don't exist
        for dir_path in [self.output_dir, self.checkpoint_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize pipeline stages
        self.preprocessor = Preprocessor(config)
        self.embedder = Embedder(config)
        self.indexer = Indexer(config)
        self.imputator = Imputator(config)
        self.query_engine = QueryEngine(config)
        self.feature_engineer = FeatureEngineer(config)
        self.classifier = Classifier(config)
        self.report_generator = ReportGenerator(config)
        
        logger.info("Pipeline initialized")
    
    def run_all(self, resume=False, checkpoint_path=None):
        """
        Run the complete pipeline.
        
        Args:
            resume (bool, optional): Whether to resume from checkpoint. Defaults to False.
            checkpoint_path (str, optional): Path to checkpoint file. Defaults to None.
            
        Returns:
            dict: Pipeline results
        """
        with Timer() as timer:
            # Stage 1: Preprocessing
            preprocessing_results = self.run_preprocessing(resume, checkpoint_path)
            
            # Stage 2: Embedding
            embedding_results = self.run_embedding(resume, checkpoint_path)
            
            # Stage 3: Indexing
            indexing_results = self.run_indexing(resume, checkpoint_path)
            
            # Stage 4: Imputation
            imputation_results = self.run_imputation(resume, checkpoint_path)
            
            # Stage 5: Ground Truth Queries
            query_results = self.run_ground_truth_queries(resume, checkpoint_path)
            
            # Stage 6: Feature Engineering
            feature_results = self.run_feature_engineering(resume, checkpoint_path)
            
            # Stage 7: Classification
            classification_results = self.run_classification(resume, checkpoint_path)
            
            # Stage 8: Reporting
            reporting_results = self.run_reporting()
        
        # Combine results
        results = {
            'preprocessing': preprocessing_results,
            'embedding': embedding_results,
            'indexing': indexing_results,
            'imputation': imputation_results,
            'ground_truth_queries': query_results,
            'feature_engineering': feature_results,
            'classification': classification_results,
            'reporting': reporting_results,
            'total_duration': timer.duration
        }
        
        # Save pipeline results
        self._save_results(results)
        
        logger.info(f"Pipeline completed in {timer.duration:.2f} seconds")
        
        return results
    
    def run_preprocessing(self, resume=False, checkpoint_path=None):
        """
        Run the preprocessing stage.
        
        Args:
            resume (bool, optional): Whether to resume from checkpoint. Defaults to False.
            checkpoint_path (str, optional): Path to checkpoint file. Defaults to None.
            
        Returns:
            dict: Preprocessing results
        """
        logger.info("==== Running Preprocessing Stage ====")
        
        # Get checkpoint path
        if resume and not checkpoint_path:
            checkpoint_path = self.checkpoint_dir / "preprocessing_final.ckpt"
            if not checkpoint_path.exists():
                checkpoint_path = self._find_latest_checkpoint("preprocessing")
        
        # Execute preprocessing
        results = self.preprocessor.execute(checkpoint_path if resume else None)
        
        logger.info("==== Preprocessing Stage Complete ====")
        
        return results
    
    def run_embedding(self, resume=False, checkpoint_path=None):
        """
        Run the embedding stage.
        
        Args:
            resume (bool, optional): Whether to resume from checkpoint. Defaults to False.
            checkpoint_path (str, optional): Path to checkpoint file. Defaults to None.
            
        Returns:
            dict: Embedding results
        """
        logger.info("==== Running Embedding Stage ====")
        
        # Get checkpoint path
        if resume and not checkpoint_path:
            checkpoint_path = self.checkpoint_dir / "embedding_final.ckpt"
            if not checkpoint_path.exists():
                checkpoint_path = self._find_latest_checkpoint("embedding")
        
        # Execute embedding
        results = self.embedder.execute(checkpoint_path if resume else None)
        
        logger.info("==== Embedding Stage Complete ====")
        
        return results
    
    def run_indexing(self, resume=False, checkpoint_path=None):
        """
        Run the indexing stage.
        
        Args:
            resume (bool, optional): Whether to resume from checkpoint. Defaults to False.
            checkpoint_path (str, optional): Path to checkpoint file. Defaults to None.
            
        Returns:
            dict: Indexing results
        """
        logger.info("==== Running Indexing Stage ====")
        
        # Get checkpoint path
        if resume and not checkpoint_path:
            checkpoint_path = self.checkpoint_dir / "indexing_final.ckpt"
            if not checkpoint_path.exists():
                checkpoint_path = self._find_latest_checkpoint("indexing")
        
        # Execute indexing
        results = self.indexer.execute(checkpoint_path if resume else None)
        
        logger.info("==== Indexing Stage Complete ====")
        
        return results
    
    def run_imputation(self, resume=False, checkpoint_path=None):
        """
        Run the imputation stage.
        
        Args:
            resume (bool, optional): Whether to resume from checkpoint. Defaults to False.
            checkpoint_path (str, optional): Path to checkpoint file. Defaults to None.
            
        Returns:
            dict: Imputation results
        """
        logger.info("==== Running Imputation Stage ====")
        
        # Get checkpoint path
        if resume and not checkpoint_path:
            checkpoint_path = self.checkpoint_dir / "imputation_final.ckpt"
            if not checkpoint_path.exists():
                checkpoint_path = self._find_latest_checkpoint("imputation")
        
        # Execute imputation
        results = self.imputator.execute(checkpoint_path if resume else None)
        
        logger.info("==== Imputation Stage Complete ====")
        
        return results
    
    def run_ground_truth_queries(self, resume=False, checkpoint_path=None):
        """
        Run the ground truth queries stage.
        
        Args:
            resume (bool, optional): Whether to resume from checkpoint. Defaults to False.
            checkpoint_path (str, optional): Path to checkpoint file. Defaults to None.
            
        Returns:
            dict: Query results
        """
        logger.info("==== Running Ground Truth Queries Stage ====")
        
        # Get checkpoint path
        if resume and not checkpoint_path:
            checkpoint_path = self.checkpoint_dir / "ground_truth_queries_final.ckpt"
            if not checkpoint_path.exists():
                checkpoint_path = self._find_latest_checkpoint("ground_truth_queries")
        
        # Execute ground truth queries
        results = self.query_engine.execute_ground_truth_queries(
            checkpoint_path if resume else None
        )
        
        logger.info("==== Ground Truth Queries Stage Complete ====")
        
        return results
    
    def run_feature_engineering(self, resume=False, checkpoint_path=None):
        """
        Run the feature engineering stage.
        
        Args:
            resume (bool, optional): Whether to resume from checkpoint. Defaults to False.
            checkpoint_path (str, optional): Path to checkpoint file. Defaults to None.
            
        Returns:
            dict: Feature engineering results
        """
        logger.info("==== Running Feature Engineering Stage ====")
        
        # Get checkpoint path
        if resume and not checkpoint_path:
            checkpoint_path = self.checkpoint_dir / "ground_truth_features_final.ckpt"
            if not checkpoint_path.exists():
                checkpoint_path = self._find_latest_checkpoint("ground_truth_features")
        
        # Execute feature engineering
        results = self.feature_engineer.execute_ground_truth_features(
            checkpoint_path if resume else None
        )
        
        logger.info("==== Feature Engineering Stage Complete ====")
        
        return results
    
    def run_classification(self, resume=False, checkpoint_path=None):
        """
        Run the classification stage.
        
        Args:
            resume (bool, optional): Whether to resume from checkpoint. Defaults to False.
            checkpoint_path (str, optional): Path to checkpoint file. Defaults to None.
            
        Returns:
            dict: Classification results
        """
        logger.info("==== Running Classification Stage ====")
        
        # Get checkpoint path
        if resume and not checkpoint_path:
            checkpoint_path = self.checkpoint_dir / "classification_final.ckpt"
            if not checkpoint_path.exists():
                checkpoint_path = self._find_latest_checkpoint("classification")
        
        # Execute classification
        results = self.classifier.execute(checkpoint_path if resume else None)
        
        logger.info("==== Classification Stage Complete ====")
        
        return results
    
    def run_prediction(self, resume=False, checkpoint_path=None):
        """
        Run the prediction stage.
        
        Args:
            resume (bool, optional): Whether to resume from checkpoint. Defaults to False.
            checkpoint_path (str, optional): Path to checkpoint file. Defaults to None.
            
        Returns:
            dict: Prediction results
        """
        logger.info("==== Running Prediction Stage ====")
        
        # Get checkpoint path
        if resume and not checkpoint_path:
            checkpoint_path = self.checkpoint_dir / "prediction_final.ckpt"
            if not checkpoint_path.exists():
                checkpoint_path = self._find_latest_checkpoint("prediction")
        
        # Execute prediction
        results = self.classifier.execute_prediction(checkpoint_path if resume else None)
        
        logger.info("==== Prediction Stage Complete ====")
        
        return results
    
    def run_candidate_queries(self, resume=False, checkpoint_path=None):
        """
        Run the candidate queries stage.
        
        Args:
            resume (bool, optional): Whether to resume from checkpoint. Defaults to False.
            checkpoint_path (str, optional): Path to checkpoint file. Defaults to None.
            
        Returns:
            dict: Query results
        """
        logger.info("==== Running Candidate Queries Stage ====")
        
        # Get checkpoint path
        if resume and not checkpoint_path:
            checkpoint_path = self.checkpoint_dir / "candidate_queries_final.ckpt"
            if not checkpoint_path.exists():
                checkpoint_path = self._find_latest_checkpoint("candidate_queries")
        
        # Execute candidate queries
        results = self.query_engine.execute_candidate_queries(
            checkpoint_path if resume else None
        )
        
        logger.info("==== Candidate Queries Stage Complete ====")
        
        return results
    
    def run_candidate_features(self, resume=False, checkpoint_path=None):
        """
        Run the candidate feature engineering stage.
        
        Args:
            resume (bool, optional): Whether to resume from checkpoint. Defaults to False.
            checkpoint_path (str, optional): Path to checkpoint file. Defaults to None.
            
        Returns:
            dict: Feature engineering results
        """
        logger.info("==== Running Candidate Feature Engineering Stage ====")
        
        # Get checkpoint path
        if resume and not checkpoint_path:
            checkpoint_path = self.checkpoint_dir / "candidate_features_final.ckpt"
            if not checkpoint_path.exists():
                checkpoint_path = self._find_latest_checkpoint("candidate_features")
        
        # Execute candidate feature engineering
        results = self.feature_engineer.execute_candidate_features(
            checkpoint_path if resume else None
        )
        
        logger.info("==== Candidate Feature Engineering Stage Complete ====")
        
        return results
    
    def run_reporting(self):
        """
        Run the reporting stage.
        
        Returns:
            dict: Reporting results
        """
        logger.info("==== Running Reporting Stage ====")
        
        # Execute reporting
        results = self.report_generator.execute()
        
        logger.info("==== Reporting Stage Complete ====")
        
        return results
    
    def _find_latest_checkpoint(self, prefix):
        """
        Find the latest checkpoint file for a given prefix.
        
        Args:
            prefix (str): Checkpoint prefix
            
        Returns:
            Path: Path to latest checkpoint file
        """
        checkpoint_files = list(self.checkpoint_dir.glob(f"{prefix}_*.ckpt"))
        
        if not checkpoint_files:
            return None
        
        # Sort by creation time
        checkpoint_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        return checkpoint_files[0]
    
    def _save_results(self, results):
        """
        Save pipeline results.
        
        Args:
            results (dict): Pipeline results
        """
        # Save results
        with open(self.output_dir / "pipeline_results.json", 'w') as f:
            # Convert to JSON-serializable format
            serializable_results = {}
            
            for stage, stage_results in results.items():
                if isinstance(stage_results, dict):
                    # Remove non-serializable objects
                    serializable_stage = {}
                    
                    for key, value in stage_results.items():
                        if isinstance(value, (int, float, str, bool, list, dict)) or value is None:
                            serializable_stage[key] = value
                    
                    serializable_results[stage] = serializable_stage
                else:
                    # Handle non-dict results
                    serializable_results[stage] = str(stage_results)
            
            json.dump(serializable_results, f, indent=2)
        
        # Save metadata
        with open(self.output_dir / "pipeline_metadata.json", 'w') as f:
            json.dump({
                'completion_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'stages': list(results.keys()),
                'total_duration': results.get('total_duration', 0.0)
            }, f, indent=2)
