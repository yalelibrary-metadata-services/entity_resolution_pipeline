"""
Orchestration Module for Entity Resolution

This module orchestrates the entire entity resolution pipeline, coordinating
the execution of individual components for preprocessing, embedding, indexing,
feature engineering, classifier training, and entity classification.
"""

import logging
import os
import time
import json
import pickle
import argparse
import yaml
from typing import Dict, List, Tuple, Any, Optional

# Local imports
from src.preprocessing import process_data, load_hash_lookup, load_string_dict
from src.embedding_and_indexing import embedding_and_indexing
from src.embedding_and_indexing_batch import embedding_and_indexing_batch
# Keep old imports for backward compatibility but mark as deprecated
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
#    from src.embedding import generate_embeddings
    from src.indexing import index_data_in_weaviate, get_weaviate_client, WeaviateClientManager, close_weaviate_client
from src.feature_engineering import FeatureEngineering
from src.querying import create_weaviate_querying
from src.training import train_classifier, EntityClassifier
from src.classifying import EntityClassification
from src.reporting import generate_report
from src.custom_features import register_custom_features
from src.checkpoint_manager import get_checkpoint_manager
# NEW: Subject enhancement modules
from src.subject_quality import SubjectQualityAudit
from src.subject_imputation import SubjectImputation

logger = logging.getLogger(__name__)

class PipelineOrchestrator:
    """
    Orchestrates the execution of the entity resolution pipeline.
    """
    
    def __init__(self, config_path: str = 'config.yml'):
        """
        Initialize the pipeline orchestrator.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Set up logging
        self._setup_logging()
        
        # Initialize state
        self.hash_lookup = None
        self.string_dict = None
        self.feature_engineering = None
        self.weaviate_querying = None
        self.classifier = None
        
        # Pipeline stages with dynamic enabling based on configuration
        quality_enabled = self.config.get('subject_quality_audit', {}).get('enabled', True)
        imputation_enabled = self.config.get('subject_imputation', {}).get('enabled', True)
        
        self.stages = [
            {'name': 'preprocessing', 'enabled': True, 'func': self._run_preprocessing},
            {'name': 'embedding_and_indexing', 'enabled': True, 'func': self._run_embedding_and_indexing},
            # Keep old stages for backward compatibility but mark as deprecated
            {'name': 'embedding', 'enabled': False, 'func': self._run_embedding, 'deprecated': True},
            {'name': 'indexing', 'enabled': False, 'func': self._run_indexing, 'deprecated': True},
            # NEW: Subject enhancement stages (conditionally enabled)
            {'name': 'subject_quality_audit', 'enabled': quality_enabled, 'func': self._run_subject_quality_audit},
            {'name': 'subject_imputation', 'enabled': imputation_enabled, 'func': self._run_subject_imputation},
            # NEW: Subject enhancement reporting (conditionally enabled)
            {'name': 'subject_enhancement_reporting', 'enabled': quality_enabled or imputation_enabled, 'func': self._run_subject_enhancement_reporting},
            {'name': 'training', 'enabled': True, 'func': self._run_training},
            {'name': 'classifying', 'enabled': True, 'func': self._run_classifying},
            {'name': 'reporting', 'enabled': True, 'func': self._run_reporting}
        ]
        
        logger.info(f"Initialized PipelineOrchestrator with config from {config_path}")
    
    def run_pipeline(self, start_stage: str = None, end_stage: str = None,
                    reset_stages: List[str] = None, resume: bool = False) -> Dict[str, Any]:
        """
        Run the entity resolution pipeline.
        
        Args:
            start_stage: Optional stage to start from
            end_stage: Optional stage to end at
            reset_stages: Optional list of stages to reset
            resume: Whether to resume from the last successful stage
            
        Returns:
            Dictionary with pipeline results and statistics
        """
        logger.info(f"Starting pipeline execution")
        start_time = time.time()
        
        # Initialize checkpoint manager
        checkpoint_manager = get_checkpoint_manager(self.config)
        
        # Handle resumption logic
        if resume and not start_stage:
            next_stage = checkpoint_manager.get_next_stage(self.stages)
            if next_stage:
                start_stage = next_stage
                last_stage = checkpoint_manager.get_last_completed_stage()
                logger.info(f"Resuming pipeline from stage '{start_stage}' (after '{last_stage}')")
            else:
                logger.info(f"All stages already completed, starting from beginning")
        
        # Check if this is a preprocessing-only operation
        preprocessing_only = ((start_stage == "preprocessing" and end_stage == "preprocessing") or
                              (reset_stages and "preprocessing" in reset_stages))
        
        # Modify config to suppress vector warnings during preprocessing
        if preprocessing_only:
            self.config["suppress_vector_warnings"] = True
            logger.info("Vector retrieval warnings will be suppressed during preprocessing operations")
        
        try:
            # Reset specified stages if provided
            if reset_stages:
                for stage in reset_stages:
                    self._reset_stage(stage)
                    checkpoint_manager.reset_stage(stage)
            
            # Determine which stages to run
            stages_to_run = self._get_stages_to_run(start_stage, end_stage)
            
            # Store metrics for each stage
            metrics = {}
            
            # Run each stage
            for stage in stages_to_run:
                logger.info(f"Running pipeline stage: {stage['name']}")
                stage_start = time.time()
                
                try:
                    # Run stage function
                    stage_metrics = stage['func']()
                    
                    # Record metrics
                    elapsed = time.time() - stage_start
                    metrics[stage['name']] = {
                        'elapsed_time': elapsed,
                        **stage_metrics
                    }
                    
                    # Update checkpoint status
                    checkpoint_manager.update_stage_completion(
                        stage['name'], 
                        'completed',
                        metrics[stage['name']]
                    )
                    
                    logger.info(f"Completed stage {stage['name']} in {elapsed:.2f} seconds")
                    
                except Exception as e:
                    logger.error(f"Error in stage {stage['name']}: {str(e)}", exc_info=True)
                    metrics[stage['name']] = {
                        'status': 'failed',
                        'error': str(e)
                    }
                    # Update checkpoint with failure status
                    checkpoint_manager.update_stage_completion(stage['name'], 'failed')
                    break
            
            # Calculate total runtime
            total_runtime = time.time() - start_time
            logger.info(f"Pipeline execution completed in {total_runtime:.2f} seconds")
            
            # Generate final metrics
            pipeline_metrics = {
                'total_runtime': total_runtime,
                'stages': metrics,
                'config': {k: v for k, v in self.config.items() if isinstance(v, (str, int, float, bool, list, dict))}
            }
            
            # Save metrics to file
            output_dir = self.config.get('output_dir', 'data/output')
            os.makedirs(output_dir, exist_ok=True)
            metrics_path = os.path.join(output_dir, 'pipeline_metrics.json')
            
            with open(metrics_path, 'w') as f:
                json.dump(pipeline_metrics, f, indent=2)
            
            logger.info(f"Saved pipeline metrics to {metrics_path}")
            
            return pipeline_metrics
        
        finally:
            # Always restore original config
            if preprocessing_only and "suppress_vector_warnings" in self.config:
                self.config.pop("suppress_vector_warnings")
                logger.info("Restored normal vector warning behavior")
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file with environment-specific overrides.
        
        Returns:
            Configuration dictionary
        """
        try:
            # Use shared config utility for consistency with standalone modules
            from src.config_utils import load_config_with_environment
            return load_config_with_environment(self.config_path)
            
        except Exception as e:
            print(f"Error loading configuration: {str(e)}")
            raise
    
    
    def _setup_logging(self) -> None:
        """
        Set up logging configuration.
        """
        log_dir = self.config.get('log_dir', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        log_level = getattr(logging, self.config.get('log_level', 'INFO'))
        log_file = os.path.join(log_dir, 'pipeline.log')
        
        # Configure root logger - use existing handlers to avoid creating new ones repeatedly
        root_logger = logging.getLogger()
        if not root_logger.handlers:
            file_handler = logging.FileHandler(log_file)
            console_handler = logging.StreamHandler()
            
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            root_logger.addHandler(file_handler)
            root_logger.addHandler(console_handler)
            
        root_logger.setLevel(log_level)
    
    def _get_stages_to_run(self, start_stage: str = None, 
                          end_stage: str = None) -> List[Dict[str, Any]]:
        """
        Get stages to run based on start and end points.
        
        Args:
            start_stage: Optional stage to start from
            end_stage: Optional stage to end at
            
        Returns:
            List of stage dictionaries to run
        """
        # Default is to run all enabled stages
        stages_to_run = [stage for stage in self.stages if stage['enabled']]
        
        # Filter by start stage if provided
        if start_stage:
            start_idx = next((i for i, s in enumerate(stages_to_run) 
                             if s['name'] == start_stage), 0)
            stages_to_run = stages_to_run[start_idx:]
        
        # Filter by end stage if provided
        if end_stage:
            end_idx = next((i for i, s in enumerate(stages_to_run) 
                           if s['name'] == end_stage), len(stages_to_run) - 1)
            stages_to_run = stages_to_run[:end_idx + 1]
        
        return stages_to_run
    
    def _reset_stage(self, stage_name: str) -> None:
        """
        Reset a specific pipeline stage.
        
        Args:
            stage_name: Name of the stage to reset
        """
        logger.info(f"Resetting pipeline stage: {stage_name}")
        
        if stage_name == 'preprocessing':
            # Delete checkpointed files
            checkpoint_dir = self.config.get('checkpoint_dir', 'data/checkpoints')
            files_to_delete = [
                os.path.join(checkpoint_dir, 'hash_lookup.pkl'),
                os.path.join(checkpoint_dir, 'string_dict.pkl'),
                os.path.join(checkpoint_dir, 'string_counts.pkl'),
                os.path.join(checkpoint_dir, 'field_hash_mapping.pkl'),
                os.path.join(checkpoint_dir, 'preprocessing_temp.db')  # SQLite database
            ]
            
            # Also delete any preprocessing checkpoint files
            import glob
            checkpoint_pattern = os.path.join(checkpoint_dir, 'preprocessing_checkpoint_*.pkl')
            checkpoint_files = glob.glob(checkpoint_pattern)
            files_to_delete.extend(checkpoint_files)
            
            for file_path in files_to_delete:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"Deleted checkpoint file: {file_path}")
            
        elif stage_name == 'embedding_and_indexing':
            # Delete checkpoint and reset Weaviate collection
            checkpoint_dir = self.config.get('checkpoint_dir', 'data/checkpoints')
            
            # Delete real-time processing checkpoints
            processed_hashes_path = os.path.join(checkpoint_dir, 'processed_hashes.pkl')
            if os.path.exists(processed_hashes_path):
                os.remove(processed_hashes_path)
                logger.info(f"Deleted real-time processed hashes checkpoint: {processed_hashes_path}")
            
            # Delete batch processing checkpoints
            batch_processed_hashes_path = os.path.join(checkpoint_dir, 'batch_processed_hashes.pkl')
            batch_jobs_path = os.path.join(checkpoint_dir, 'batch_jobs.pkl')
            
            if os.path.exists(batch_processed_hashes_path):
                os.remove(batch_processed_hashes_path)
                logger.info(f"Deleted batch processed hashes checkpoint: {batch_processed_hashes_path}")
            
            if os.path.exists(batch_jobs_path):
                os.remove(batch_jobs_path)
                logger.info(f"Deleted batch jobs checkpoint: {batch_jobs_path}")
            
            # Delete any batch request/result files
            import glob
            batch_request_pattern = os.path.join(checkpoint_dir, 'batch_requests_*.jsonl')
            batch_result_pattern = os.path.join(checkpoint_dir, 'batch_results_*.jsonl')
            
            for pattern in [batch_request_pattern, batch_result_pattern]:
                batch_files = glob.glob(pattern)
                for file_path in batch_files:
                    os.remove(file_path)
                    logger.info(f"Deleted batch file: {file_path}")
            
            # Set the recreate_collections flag to ensure the collection is recreated
            # This flag will be checked in both embedding modules
            self.config["recreate_collections"] = True
            logger.info("Set recreate_collections flag to True")
            
            # We'll let the embedding modules handle the collection deletion and recreation
            # since they have more complete error handling for this process
            
        elif stage_name == 'embedding':
            # Legacy stage - redirect to embedding_and_indexing
            logger.warning("The 'embedding' stage is deprecated. Resetting 'embedding_and_indexing' stage instead.")
            self._reset_stage('embedding_and_indexing')
            
        elif stage_name == 'indexing':
            # Legacy stage - redirect to embedding_and_indexing
            logger.warning("The 'indexing' stage is deprecated. Resetting 'embedding_and_indexing' stage instead.")
            self._reset_stage('embedding_and_indexing')
            
        elif stage_name == 'training':
            # Delete trained model
            checkpoint_dir = self.config.get('checkpoint_dir', 'data/checkpoints')
            model_path = os.path.join(checkpoint_dir, 'classifier_model.pkl')
            
            if os.path.exists(model_path):
                os.remove(model_path)
                logger.info(f"Deleted trained model: {model_path}")
            
        elif stage_name == 'classifying':
            # Delete classification checkpoint and output
            checkpoint_dir = self.config.get('checkpoint_dir', 'data/checkpoints')
            output_dir = self.config.get('output_dir', 'data/output')
            
            files_to_delete = [
                os.path.join(checkpoint_dir, 'classification_checkpoint.pkl'),
                os.path.join(output_dir, 'entity_matches.csv'),
                os.path.join(output_dir, 'entity_clusters.json')
            ]
            
            for file_path in files_to_delete:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"Deleted classification file: {file_path}")
            
        elif stage_name == 'subject_quality_audit':
            # Delete quality audit reports and reset subject evaluation flags
            output_dir = self.config.get('output_dir', 'data/output')
            reports_dir = os.path.join(output_dir, 'reports')
            
            # Delete quality audit report files
            import glob
            report_pattern = os.path.join(reports_dir, 'subject_quality_audit_*.json')
            report_files = glob.glob(report_pattern)
            
            for file_path in report_files:
                os.remove(file_path)
                logger.info(f"Deleted quality audit report: {file_path}")
            
            # Reset subject evaluation flags and regenerate composite-subject mapping
            checkpoint_dir = self.config.get('checkpoint_dir', 'data/checkpoints')
            hash_lookup_path = os.path.join(checkpoint_dir, 'hash_lookup.pkl')
            
            if os.path.exists(hash_lookup_path):
                try:
                    hash_lookup = load_hash_lookup(hash_lookup_path)
                    reset_count = 0
                    composite_subject_mapping = {}
                    
                    for record_id, record_data in hash_lookup.items():
                        # Reset quality audit flags
                        if 'subject_evaluated' in record_data:
                            del record_data['subject_evaluated']
                            reset_count += 1
                        if 'subject_remediation_required' in record_data:
                            del record_data['subject_remediation_required']
                        if 'alternative_subject_hash' in record_data:
                            del record_data['alternative_subject_hash']
                        
                        # Reset remediated subjects to original values
                        if 'subject_remediated' in record_data and 'original_subject_hash' in record_data:
                            record_data['subjects'] = record_data['original_subject_hash']
                            del record_data['subject_remediated']
                            del record_data['original_subject_hash']
                        
                        # Rebuild composite-subject mapping
                        composite_hash = record_data.get('composite')
                        subject_hash = record_data.get('subjects')
                        if composite_hash and composite_hash != "NULL":
                            composite_subject_mapping[composite_hash] = (
                                subject_hash if subject_hash and subject_hash != "NULL" else None
                            )
                    
                    # Save updated hash_lookup and mapping
                    with open(hash_lookup_path, 'wb') as f:
                        pickle.dump(hash_lookup, f, protocol=pickle.HIGHEST_PROTOCOL)
                    
                    from src.preprocessing import save_composite_subject_mapping
                    save_composite_subject_mapping(composite_subject_mapping, checkpoint_dir)
                    
                    logger.info(f"Reset subject evaluation flags for {reset_count} records")
                    logger.info(f"Regenerated composite-subject mapping with {len(composite_subject_mapping)} entries")
                    
                except Exception as e:
                    logger.error(f"Error resetting subject evaluation flags: {e}")
        
        elif stage_name == 'subject_imputation':
            # Delete imputation results and reset imputation flags
            output_dir = self.config.get('output_dir', 'data/output')
            checkpoint_dir = self.config.get('checkpoint_dir', 'data/checkpoints')
            
            # Delete imputation result files
            import glob
            result_pattern = os.path.join(output_dir, 'subject_imputation_results_*.json')
            result_files = glob.glob(result_pattern)
            
            for file_path in result_files:
                os.remove(file_path)
                logger.info(f"Deleted imputation results: {file_path}")
            
            # Delete imputation cache
            cache_dir = os.path.join(checkpoint_dir, 'cache')
            cache_path = os.path.join(cache_dir, 'imputation_cache.pkl')
            
            if os.path.exists(cache_path):
                os.remove(cache_path)
                logger.info(f"Deleted imputation cache: {cache_path}")
            
            # Reset imputation flags and regenerate composite-subject mapping
            hash_lookup_path = os.path.join(checkpoint_dir, 'hash_lookup.pkl')
            
            if os.path.exists(hash_lookup_path):
                try:
                    hash_lookup = load_hash_lookup(hash_lookup_path)
                    reset_count = 0
                    composite_subject_mapping = {}
                    
                    for record_id, record_data in hash_lookup.items():
                        # Reset imputed subjects back to NULL
                        if record_data.get('imputed_subject', False):
                            record_data['subjects'] = 'NULL'
                            del record_data['imputed_subject']
                            reset_count += 1
                        
                        # Remove imputation tracking fields
                        for field in ['imputation_confidence', 'imputation_alternatives', 'imputation_attempted']:
                            if field in record_data:
                                del record_data[field]
                        
                        # Rebuild composite-subject mapping
                        composite_hash = record_data.get('composite')
                        subject_hash = record_data.get('subjects')
                        if composite_hash and composite_hash != "NULL":
                            composite_subject_mapping[composite_hash] = (
                                subject_hash if subject_hash and subject_hash != "NULL" else None
                            )
                    
                    # Save updated hash_lookup and mapping
                    with open(hash_lookup_path, 'wb') as f:
                        pickle.dump(hash_lookup, f, protocol=pickle.HIGHEST_PROTOCOL)
                    
                    from src.preprocessing import save_composite_subject_mapping
                    save_composite_subject_mapping(composite_subject_mapping, checkpoint_dir)
                    
                    logger.info(f"Reset imputation for {reset_count} records")
                    logger.info(f"Regenerated composite-subject mapping with {len(composite_subject_mapping)} entries")
                    
                except Exception as e:
                    logger.error(f"Error resetting imputation flags: {e}")
        
        elif stage_name == 'subject_enhancement_reporting':
            # Delete subject enhancement reports
            output_dir = self.config.get('output_dir', 'data/output')
            reports_dir = os.path.join(output_dir, 'reports')
            
            # Delete comprehensive subject enhancement reports
            import glob
            report_pattern = os.path.join(reports_dir, 'subject_enhancement_comprehensive_report_*.json')
            report_files = glob.glob(report_pattern)
            
            for file_path in report_files:
                os.remove(file_path)
                logger.info(f"Deleted subject enhancement report: {file_path}")
        
        elif stage_name == 'reporting':
            # Delete report files
            output_dir = self.config.get('output_dir', 'data/output')
            report_path = os.path.join(output_dir, 'entity_resolution_report.html')
            
            if os.path.exists(report_path):
                os.remove(report_path)
                logger.info(f"Deleted report file: {report_path}")
        
        else:
            logger.warning(f"Unknown stage name for reset: {stage_name}")
    
    def _run_preprocessing(self) -> Dict[str, Any]:
        """
        Run the preprocessing stage.
        
        Returns:
            Dictionary with preprocessing metrics
        """
        # Load or process data
        input_dir = self.config.get('input_dir', 'data/input')
        checkpoint_dir = self.config.get('checkpoint_dir', 'data/checkpoints')
        
        # Check for existing preprocessing results
        hash_lookup_path = os.path.join(checkpoint_dir, 'hash_lookup.pkl')
        string_dict_path = os.path.join(checkpoint_dir, 'string_dict.pkl')
        
        if os.path.exists(hash_lookup_path) and os.path.exists(string_dict_path):
            logger.info("Loading preprocessed data from checkpoints")
            self.hash_lookup = load_hash_lookup(hash_lookup_path)
            self.string_dict = load_string_dict(string_dict_path)
            
            metrics = {
                'status': 'loaded_from_checkpoint',
                'entity_count': len(self.hash_lookup),
                'unique_strings': len(self.string_dict)
            }
        else:
            logger.info("Running data preprocessing")
            self.hash_lookup, self.string_dict, metrics = process_data(
                self.config, input_dir, checkpoint_dir
            )
        
        logger.info(f"Preprocessing complete: {len(self.hash_lookup)} entities, "
                   f"{len(self.string_dict)} unique strings")
        
        return metrics
    
    def _run_embedding_and_indexing(self) -> Dict[str, Any]:
        """
        Run the unified embedding generation and indexing stage.
        
        Returns:
            Dictionary with embedding and indexing metrics
        """
        # Make sure preprocessing data is loaded
        if self.hash_lookup is None or self.string_dict is None:
            self._run_preprocessing()
        
        # Load additional data required for embedding and indexing
        checkpoint_dir = self.config.get('checkpoint_dir', 'data/checkpoints')
        field_hash_mapping_path = os.path.join(checkpoint_dir, 'field_hash_mapping.pkl')
        string_counts_path = os.path.join(checkpoint_dir, 'string_counts.pkl')
        
        field_hash_mapping = {}
        string_counts = {}
        
        # Load field hash mapping
        if os.path.exists(field_hash_mapping_path):
            try:
                with open(field_hash_mapping_path, 'rb') as f:
                    field_hash_mapping = pickle.load(f)
                logger.info(f"Loaded field hash mapping with {len(field_hash_mapping)} entries")
            except Exception as e:
                logger.error(f"Error loading field hash mapping: {str(e)}")
                field_hash_mapping = {}
        
        # Load string counts
        if os.path.exists(string_counts_path):
            try:
                with open(string_counts_path, 'rb') as f:
                    string_counts = pickle.load(f)
                logger.info(f"Loaded string counts with {len(string_counts)} entries")
            except Exception as e:
                logger.error(f"Error loading string counts: {str(e)}")
                string_counts = {}
        
        # Check if batch processing is enabled
        use_batch = self.config.get("use_batch_embeddings", False)
        
        if use_batch:
            logger.info("Starting batch embedding and indexing process (50% cost savings, 24h turnaround)")
            metrics = embedding_and_indexing_batch(
                self.config, 
                self.string_dict, 
                field_hash_mapping, 
                string_counts
            )
        else:
            logger.info("Starting real-time embedding and indexing process")
            metrics = embedding_and_indexing(
                self.config, 
                self.string_dict, 
                field_hash_mapping, 
                string_counts
            )
        
        return metrics
    
    def _run_embedding(self) -> Dict[str, Any]:
        """
        Legacy stage - Run the embedding generation stage.
        
        Returns:
            Dictionary with embedding metrics
        """
        # Emit deprecation warning
        logger.warning("The 'embedding' stage is deprecated. Please use 'embedding_and_indexing' stage instead.")
        
        # Redirect to unified stage
        return self._run_embedding_and_indexing()
    
    def _run_indexing(self) -> Dict[str, Any]:
        """
        Legacy stage - Run the Weaviate indexing stage.
        
        Returns:
            Dictionary with indexing metrics
        """
        # Emit deprecation warning
        logger.warning("The 'indexing' stage is deprecated. Please use 'embedding_and_indexing' stage instead.")
        
        # Redirect to unified stage
        return self._run_embedding_and_indexing()
    
    def _run_training(self) -> Dict[str, Any]:
        """
        Run the classifier training stage.
        
        Returns:
            Dictionary with training metrics
        """
        # Initialize required components if not already done
        if self.hash_lookup is None or self.string_dict is None:
            self._run_preprocessing()
        
        # Use context manager for Weaviate client
        with WeaviateClientManager(self.config) as client:
            # Initialize feature engineering and querying
            self.feature_engineering = FeatureEngineering(
                self.config, client, self.hash_lookup
            )
            
            # Register custom features with enhanced error handling
            try:
                logger.info("Registering custom features for feature engineering module")
                register_custom_features(self.feature_engineering, self.config)
                logger.info(f"Feature engineering has {len(self.feature_engineering.get_feature_names())} features registered")
                logger.info(f"Feature substitution mapping: {self.feature_engineering.get_substitution_mapping()}")
            except Exception as e:
                logger.error(f"Error registering custom features: {str(e)}")
                logger.error(f"Continuing with base features only - custom features may not be available")
            
            self.weaviate_querying = create_weaviate_querying(
                self.config, client, self.hash_lookup
            )
            
            # Train classifier
            logger.info("Training entity classifier")
            self.classifier = train_classifier(
                self.config, self.feature_engineering, self.hash_lookup, self.string_dict
            )
        
        # Get training metrics from the evaluation report
        output_dir = self.config.get('output_dir', 'data/output')
        evaluation_path = os.path.join(output_dir, 'classifier_evaluation.json')
        training_report_path = os.path.join(output_dir, 'training_report.json')
        
        if os.path.exists(training_report_path):
            with open(training_report_path, 'r') as f:
                metrics = json.load(f)
        elif os.path.exists(evaluation_path):
            with open(evaluation_path, 'r') as f:
                metrics = {'evaluation': json.load(f)}
        else:
            metrics = {'status': 'completed', 'details': 'No detailed metrics available'}
        
        return metrics
    
    def _run_classifying(self) -> Dict[str, Any]:
        """
        Run the entity classification stage.
        
        Returns:
            Dictionary with classification metrics
        """
        # Initialize required components if not already done
        if self.hash_lookup is None or self.string_dict is None:
            self._run_preprocessing()
        
        # Initialize or load classifier if needed
        if self.classifier is None:
            # Load trained classifier
            self.classifier = EntityClassifier(self.config)
            model_path = os.path.join(
                self.config.get('checkpoint_dir', 'data/checkpoints'),
                'classifier_model.pkl'
            )
            
            if not os.path.exists(model_path):
                # Run training if model doesn't exist
                self._run_training()
            else:
                self.classifier.load(model_path)
        
        # Use context manager for Weaviate client
        with WeaviateClientManager(self.config) as client:
            # Initialize feature engineering and querying
            self.feature_engineering = FeatureEngineering(
                self.config, client, self.hash_lookup
            )
            
            self.weaviate_querying = create_weaviate_querying(
                self.config, client, self.hash_lookup
            )
            
            # Classify entities
            logger.info("Classifying entities in the dataset")
            
            # Create classification object
            classification = EntityClassification(
                self.config, self.feature_engineering, self.classifier, self.weaviate_querying
            )
            
            # Set hash lookup explicitly
            classification.hash_lookup = self.hash_lookup
            
            # Run classification
            entity_ids = list(self.hash_lookup.keys())
            metrics = classification.classify_entities(entity_ids, self.hash_lookup, self.string_dict)
        
        return metrics
    
    def _run_subject_quality_audit(self) -> Dict[str, Any]:
        """
        Run the subject quality audit stage.
        
        Returns:
            Dictionary with quality audit metrics
        """
        # Check if this stage is enabled in configuration
        quality_config = self.config.get('subject_quality_audit', {})
        if not quality_config.get('enabled', True):
            logger.info("Subject quality audit disabled in configuration")
            return {'status': 'disabled'}
        
        # Initialize required components if not already done
        if self.hash_lookup is None or self.string_dict is None:
            self._run_preprocessing()
        
        # Use context manager for Weaviate client
        with WeaviateClientManager(self.config) as client:
            logger.info("Starting subject quality audit with automatic remediation")
            
            # Create quality audit instance
            quality_audit = SubjectQualityAudit(self.config, client)
            
            # Execute audit and automatic remediation
            results = quality_audit.execute()
            
            # Update hash_lookup if remediation occurred
            if results.get('remediation_results', {}).get('remediated_subjects', 0) > 0:
                logger.info(f"Quality audit automatically improved {results['remediation_results']['remediated_subjects']} subject assignments")
                
                # Reload hash_lookup to get the updated data
                checkpoint_dir = self.config.get('checkpoint_dir', 'data/checkpoints')
                hash_lookup_path = os.path.join(checkpoint_dir, 'hash_lookup.pkl')
                self.hash_lookup = load_hash_lookup(hash_lookup_path)
        
        return results
    
    def _run_subject_imputation(self) -> Dict[str, Any]:
        """
        Run the subject imputation stage.
        
        Returns:
            Dictionary with imputation metrics
        """
        # Check if this stage is enabled in configuration
        imputation_config = self.config.get('subject_imputation', {})
        if not imputation_config.get('enabled', True):
            logger.info("Subject imputation disabled in configuration")
            return {'status': 'disabled'}
        
        # Initialize required components if not already done
        if self.hash_lookup is None or self.string_dict is None:
            self._run_preprocessing()
        
        # Use context manager for Weaviate client
        with WeaviateClientManager(self.config) as client:
            logger.info("Starting subject imputation for missing values")
            
            # Create imputation instance
            imputation = SubjectImputation(self.config, client)
            
            # Execute imputation
            results = imputation.execute()
            
            # Update hash_lookup if imputation occurred
            if results.get('high_confidence_imputations', 0) > 0:
                logger.info(f"Successfully imputed {results['high_confidence_imputations']} missing subjects")
                
                # Reload hash_lookup to get the updated data
                checkpoint_dir = self.config.get('checkpoint_dir', 'data/checkpoints')
                hash_lookup_path = os.path.join(checkpoint_dir, 'hash_lookup.pkl')
                self.hash_lookup = load_hash_lookup(hash_lookup_path)
        
        return results
    
    def _run_subject_enhancement_reporting(self) -> Dict[str, Any]:
        """
        Generate comprehensive subject enhancement report.
        
        Returns:
            Dictionary with reporting metrics
        """
        # Check if either subject enhancement stage was run
        quality_enabled = self.config.get('subject_quality_audit', {}).get('enabled', True)
        imputation_enabled = self.config.get('subject_imputation', {}).get('enabled', True)
        
        if not (quality_enabled or imputation_enabled):
            logger.info("Subject enhancement reporting skipped - no enhancement stages enabled")
            return {'status': 'skipped', 'reason': 'no_enhancement_stages_enabled'}
        
        # Initialize required components if not already done
        if self.hash_lookup is None or self.string_dict is None:
            self._run_preprocessing()
        
        logger.info("Generating comprehensive subject enhancement report")
        
        # Import and run the reporting function
        from src.reporting import generate_subject_enhancement_report
        
        try:
            results = generate_subject_enhancement_report(
                self.config, 
                self.hash_lookup, 
                self.string_dict
            )
            
            if results['status'] == 'completed':
                logger.info(f"Subject enhancement report completed: {results['enhanced_records']} records enhanced")
                logger.info(f"Report saved to: {results['report_path']}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error generating subject enhancement report: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def _run_reporting(self) -> Dict[str, Any]:
        """
        Run the reporting stage.
        
        Returns:
            Dictionary with reporting metrics
        """
        # Generate final report
        output_dir = self.config.get('output_dir', 'data/output')
        
        # Check if required files exist
        matches_path = os.path.join(output_dir, 'entity_matches.csv')
        clusters_path = os.path.join(output_dir, 'entity_clusters.json')
        evaluation_path = os.path.join(output_dir, 'classifier_evaluation.json')
        
        if not (os.path.exists(matches_path) and os.path.exists(clusters_path)):
            logger.warning("Classification results not found, running classification stage")
            self._run_classifying()
        
        # Generate report
        logger.info("Generating final entity resolution report")
        metrics = generate_report(
            self.config, self.hash_lookup, self.string_dict
        )
        
        return metrics

def main():
    """
    Main function for running the pipeline orchestrator.
    """
    parser = argparse.ArgumentParser(description='Entity Resolution Pipeline Orchestrator')
    parser.add_argument('--config', default='config.yml', help='Path to configuration file')
    parser.add_argument('--start', help='Stage to start from')
    parser.add_argument('--end', help='Stage to end at')
    parser.add_argument('--reset', nargs='*', help='Stages to reset')
    
    # Batch management arguments
    batch_group = parser.add_argument_group('batch management', 'Manual batch processing commands')
    batch_group.add_argument('--batch-status', action='store_true',
                            help='Check status of batch jobs (requires batch processing enabled)')
    batch_group.add_argument('--batch-results', action='store_true', 
                            help='Download and process completed batch results')
    
    args = parser.parse_args()
    
    # Handle batch management commands
    if args.batch_status or args.batch_results:
        from src.embedding_and_indexing_batch import BatchEmbeddingPipeline
        import yaml
        
        # Load configuration
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check if batch processing is enabled
        if not config.get('use_batch_embeddings', False):
            print("⚠️  Batch embeddings are not enabled in configuration.")
            print("   Set 'use_batch_embeddings: true' in your config.yml")
            return
        
        checkpoint_dir = config.get("checkpoint_dir", "data/checkpoints")
        
        # Initialize batch pipeline
        pipeline = None
        try:
            pipeline = BatchEmbeddingPipeline(config)
            
            if args.batch_status:
                result = pipeline.check_batch_status(checkpoint_dir)
                if result.get('ready_for_download', False):
                    print(f"\n💡 Ready to download results!")
                    print(f"   Run: python main.py --batch-results")
                    
            elif args.batch_results:
                result = pipeline.process_completed_jobs(checkpoint_dir)
                if result['status'] == 'completed':
                    print(f"\n🎉 Successfully processed batch results!")
                
        except Exception as e:
            logger.error(f"Error in batch management: {str(e)}")
            raise
        finally:
            if pipeline and hasattr(pipeline, 'weaviate_client'):
                try:
                    pipeline.weaviate_client.close()
                except:
                    pass
        return
    
    # Create and run orchestrator for normal pipeline operations
    orchestrator = PipelineOrchestrator(args.config)
    orchestrator.run_pipeline(args.start, args.end, args.reset)

if __name__ == "__main__":
    main()
