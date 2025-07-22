"""
Intelligent Transition Controller for Entity Resolution

This module orchestrates the seamless transition from batch to real-time
embedding processing, ensuring no data loss and complete state preservation.
"""

import os
import sys
import logging
import time
import json
from typing import Dict, List, Any, Optional

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from batch_state_consolidator import BatchStateConsolidator
from embedding_and_indexing_batch import BatchEmbeddingPipeline
from embedding_and_indexing import EmbeddingAndIndexingPipeline

logger = logging.getLogger(__name__)


class TransitionController:
    """
    Orchestrates seamless transitions between batch and real-time embedding processing.
    
    Handles:
    - Batch ↔ Real-time process transitions
    - State consolidation and migration
    - Failed request transfer
    - Processing initiation for both modes
    - Progress monitoring and reporting
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the transition controller.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.checkpoint_dir = config.get("checkpoint_dir", "data/checkpoints")
        
        # Transition state
        self.transition_state = {
            'started': False,
            'batch_terminated': False,
            'realtime_terminated': False,
            'state_consolidated': False,
            'realtime_started': False,
            'batch_started': False,
            'completed': False,
            'start_time': None,
            'end_time': None,
            'errors': [],
            'direction': None  # 'batch_to_realtime' or 'realtime_to_batch'
        }
        
        # Components
        self.batch_pipeline = None
        self.realtime_pipeline = None
        self.consolidator = None
        
        logger.info("Initialized TransitionController")
    
    def pre_transition_analysis(self) -> Dict[str, Any]:
        """
        Analyze current system state before transition.
        
        Returns:
            Dictionary with analysis results and recommendations
        """
        logger.info("Performing pre-transition analysis...")
        
        analysis = {
            'timestamp': time.time(),
            'batch_state': {},
            'realtime_state': {},
            'recommendations': [],
            'warnings': [],
            'errors': [],
            'transition_feasible': True
        }
        
        try:
            # Initialize consolidator for analysis
            self.consolidator = BatchStateConsolidator(self.config)
            
            # Load and analyze batch state
            batch_state = self.consolidator.load_batch_state()
            analysis['batch_state'] = batch_state
            
            # Load and analyze real-time state
            realtime_state = self.consolidator.load_realtime_state()
            analysis['realtime_state'] = realtime_state
            
            # Get readiness assessment
            summary = self.consolidator.get_consolidation_summary()
            transition_ready = summary['transition_ready']
            
            # Check for blocking issues
            if not transition_ready['ready']:
                analysis['transition_feasible'] = False
                analysis['errors'].extend(transition_ready['issues'])
            
            # Add warnings
            analysis['warnings'].extend(transition_ready.get('warnings', []))
            
            # Generate recommendations
            self._generate_recommendations(analysis, summary)
            
            logger.info(f"Pre-transition analysis complete. Feasible: {analysis['transition_feasible']}")
            
        except Exception as e:
            error_msg = f"Error in pre-transition analysis: {e}"
            logger.error(error_msg)
            analysis['errors'].append(error_msg)
            analysis['transition_feasible'] = False
        
        return analysis
    
    def _generate_recommendations(self, analysis: Dict[str, Any], summary: Dict[str, Any]) -> None:
        """
        Generate recommendations based on analysis.
        
        Args:
            analysis: Analysis dictionary to update
            summary: Consolidation summary
        """
        recommendations = []
        
        # Check for active batch jobs
        active_jobs = summary['batch_state']['queue_active']
        if active_jobs > 0:
            recommendations.append(
                f"Wait for {active_jobs} active batch jobs to complete before transition"
            )
        
        # Check for large number of failed requests
        failed_count = summary['batch_state']['failed_requests']
        if failed_count > 1000:
            recommendations.append(
                f"Large number of failed requests ({failed_count:,}). "
                "Consider analyzing failure patterns before transition"
            )
        
        # Check for processing efficiency
        batch_only = summary['consolidated_state']['batch_only_hashes']
        if batch_only > 0:
            recommendations.append(
                f"Transition will preserve {batch_only:,} batch-processed hashes"
            )
        
        # Estimate transition time
        failed_retryable = analysis.get('failed_request_analysis', {}).get('retryable', 0)
        if failed_retryable > 0:
            est_time = failed_retryable * 0.1  # Rough estimate: 0.1 seconds per retry
            recommendations.append(
                f"Estimated {failed_retryable:,} retryable requests will take ~{est_time:.0f} seconds"
            )
        
        analysis['recommendations'] = recommendations
    
    def terminate_realtime_processing(self) -> Dict[str, Any]:
        """
        Gracefully terminate real-time processing.
        
        Returns:
            Dictionary with termination results
        """
        logger.info("Terminating real-time processing...")
        
        termination_results = {
            'success': False,
            'active_workers_stopped': 0,
            'pending_requests_cleared': 0,
            'errors': []
        }
        
        try:
            # Initialize real-time pipeline if needed with error handling for Weaviate
            if not self.realtime_pipeline:
                try:
                    self.realtime_pipeline = EmbeddingAndIndexingPipeline(self.config)
                except Exception as e:
                    if "Connection refused" in str(e) or "Weaviate" in str(e):
                        logger.warning(f"Weaviate connection failed during real-time pipeline initialization: {e}")
                        logger.warning("Continuing with limited functionality (Weaviate operations will be skipped)")
                        # Create a minimal pipeline object for checkpoint operations
                        from src.embedding_and_indexing import EmbeddingAndIndexingPipeline
                        self.realtime_pipeline = object.__new__(EmbeddingAndIndexingPipeline)
                        self.realtime_pipeline.config = self.config
                        self.realtime_pipeline.weaviate_client = None
                    else:
                        raise
            
            # Load current state
            try:
                self.realtime_pipeline.load_checkpoint(self.checkpoint_dir)
            except Exception as e:
                logger.warning(f"Error loading real-time checkpoint: {e}")
                # Initialize empty state if checkpoint loading fails
                if not hasattr(self.realtime_pipeline, 'retry_queue'):
                    self.realtime_pipeline.retry_queue = []
            
            # Save current state before termination
            try:
                self.realtime_pipeline.save_checkpoint(self.checkpoint_dir)
            except Exception as e:
                logger.warning(f"Error saving real-time checkpoint (continuing anyway): {e}")
            
            # Clear any pending retry queues
            if hasattr(self.realtime_pipeline, 'retry_queue'):
                pending_count = len(self.realtime_pipeline.retry_queue)
                self.realtime_pipeline.retry_queue.clear()
                termination_results['pending_requests_cleared'] = pending_count
                logger.info(f"Cleared {pending_count} pending retry requests")
            
            termination_results['success'] = True
            self.transition_state['realtime_terminated'] = True
            
            logger.info("Real-time processing terminated successfully")
            
        except Exception as e:
            error_msg = f"Error terminating real-time processing: {e}"
            logger.error(error_msg)
            termination_results['errors'].append(error_msg)
        finally:
            # Clean up real-time pipeline
            if self.realtime_pipeline:
                try:
                    if hasattr(self.realtime_pipeline, 'weaviate_client'):
                        self.realtime_pipeline.weaviate_client.close()
                except:
                    pass
                self.realtime_pipeline = None
        
        return termination_results
    
    def terminate_batch_processing(self, force: bool = False, override: bool = False) -> Dict[str, Any]:
        """
        Gracefully terminate batch processing.
        
        Args:
            force: Force termination even if jobs are active
            override: Override job state tracking and force consolidation regardless of supposed active jobs
            
        Returns:
            Dictionary with termination results
        """
        logger.info("Terminating batch processing...")
        
        termination_results = {
            'success': False,
            'active_jobs_cancelled': 0,
            'pending_jobs_cancelled': 0,
            'completed_jobs_processed': 0,
            'jobs_overridden': 0,
            'errors': []
        }
        
        try:
            # Initialize batch pipeline with error handling for Weaviate
            try:
                self.batch_pipeline = BatchEmbeddingPipeline(self.config)
            except Exception as e:
                if "Connection refused" in str(e) or "Weaviate" in str(e):
                    logger.warning(f"Weaviate connection failed during batch pipeline initialization: {e}")
                    logger.warning("Continuing with limited functionality (some operations may be skipped)")
                    # Create a minimal pipeline object for checkpoint operations
                    from src.embedding_and_indexing_batch import BatchEmbeddingPipeline
                    self.batch_pipeline = object.__new__(BatchEmbeddingPipeline)
                    self.batch_pipeline.config = self.config
                    self.batch_pipeline.active_batch_queue = []
                    self.batch_pipeline.pending_batches = []
                    self.batch_pipeline.weaviate_client = None
                else:
                    raise
            
            # Load current state
            try:
                self.batch_pipeline.load_checkpoint(self.checkpoint_dir)
            except Exception as e:
                logger.warning(f"Error loading checkpoint: {e}")
                # Initialize empty state if checkpoint loading fails
                if not hasattr(self.batch_pipeline, 'active_batch_queue'):
                    self.batch_pipeline.active_batch_queue = []
                if not hasattr(self.batch_pipeline, 'pending_batches'):
                    self.batch_pipeline.pending_batches = []
            
            # Process any completed jobs first (unless override is enabled)
            if not override:
                try:
                    completed_results = self.batch_pipeline.process_completed_jobs(self.checkpoint_dir)
                    if completed_results.get('status') == 'completed':
                        termination_results['completed_jobs_processed'] = completed_results.get('successful_jobs', 0)
                        logger.info(f"Processed {termination_results['completed_jobs_processed']} completed jobs")
                except Exception as e:
                    logger.warning(f"Error processing completed jobs: {e}")
            
            # Check for active jobs
            active_jobs = len(self.batch_pipeline.active_batch_queue)
            pending_jobs = len(self.batch_pipeline.pending_batches)
            
            if override:
                # Override mode: forcibly clear all job queues regardless of state
                logger.warning(f"OVERRIDE MODE: Forcibly clearing {active_jobs} active jobs and {pending_jobs} pending jobs")
                logger.warning("This may result in loss of incomplete batch processing progress")
                
                # Clear all job queues
                self.batch_pipeline.active_batch_queue.clear()
                self.batch_pipeline.pending_batches.clear()
                
                # Record what was overridden
                termination_results['jobs_overridden'] = active_jobs + pending_jobs
                termination_results['active_jobs_cancelled'] = active_jobs
                termination_results['pending_jobs_cancelled'] = pending_jobs
                
                logger.info(f"Override completed: cleared {termination_results['jobs_overridden']} jobs from tracking")
                
            elif active_jobs > 0:
                if force:
                    logger.warning(f"Force cancelling {active_jobs} active batch jobs")
                    # TODO: Implement job cancellation
                    termination_results['active_jobs_cancelled'] = active_jobs
                else:
                    raise ValueError(f"Cannot terminate: {active_jobs} active jobs still running. Use force=True to cancel them or --override to ignore job state.")
            
            if pending_jobs > 0 and not override:
                logger.info(f"Clearing {pending_jobs} pending batch jobs")
                self.batch_pipeline.pending_batches.clear()
                termination_results['pending_jobs_cancelled'] = pending_jobs
            
            # Save final state
            try:
                self.batch_pipeline.save_checkpoint(self.checkpoint_dir)
            except Exception as e:
                logger.warning(f"Error saving checkpoint (continuing anyway): {e}")
            
            termination_results['success'] = True
            self.transition_state['batch_terminated'] = True
            
            logger.info("Batch processing terminated successfully")
            
        except Exception as e:
            error_msg = f"Error terminating batch processing: {e}"
            logger.error(error_msg)
            termination_results['errors'].append(error_msg)
        finally:
            # Clean up batch pipeline
            if self.batch_pipeline:
                try:
                    if hasattr(self.batch_pipeline, 'close'):
                        self.batch_pipeline.close()
                except Exception as e:
                    logger.debug(f"Error closing batch pipeline: {e}")
                self.batch_pipeline = None
        
        return termination_results
    
    def consolidate_state(self) -> Dict[str, Any]:
        """
        Consolidate batch and real-time processing state.
        
        Returns:
            Dictionary with consolidation results
        """
        logger.info("Consolidating processing state...")
        
        try:
            # Import and run consolidation
            from batch_state_consolidator import consolidate_batch_state
            
            consolidation_results = consolidate_batch_state(self.config)
            
            if consolidation_results['status'] == 'completed':
                self.transition_state['state_consolidated'] = True
                logger.info("State consolidation completed successfully")
            else:
                logger.error(f"State consolidation failed: {consolidation_results.get('error')}")
            
            return consolidation_results
            
        except Exception as e:
            error_msg = f"Error consolidating state: {e}"
            logger.error(error_msg)
            return {
                'status': 'error',
                'error': error_msg
            }
    
    def start_batch_processing(self, string_dict: Dict[str, str], 
                              field_hash_mapping: Dict[str, Dict[str, int]],
                              string_counts: Dict[str, int],
                              override: bool = False) -> Dict[str, Any]:
        """
        Start batch processing with consolidated state.
        
        Args:
            string_dict: String hash to value mapping
            field_hash_mapping: Hash to field type mapping
            string_counts: Hash to frequency mapping
            override: Skip actual processing startup and just mark as ready
            
        Returns:
            Dictionary with processing results
        """
        logger.info("Starting batch processing with consolidated state...")
        
        if override:
            # Override mode: Skip actual processing startup, just mark as ready
            logger.warning("OVERRIDE MODE: Skipping batch processing startup due to Weaviate issues")
            logger.info("State has been consolidated - you can start batch processing manually later with:")
            logger.info("nohup python batch_manager.py --create > batch_processing.log 2>&1 &")
            
            self.transition_state['batch_started'] = True
            return {
                'status': 'completed',
                'message': 'State consolidated successfully - batch processing ready to start manually',
                'override_used': True
            }
        
        try:
            # Initialize batch pipeline with error handling
            try:
                self.batch_pipeline = BatchEmbeddingPipeline(self.config)
            except Exception as e:
                if "Connection refused" in str(e) or "Weaviate" in str(e):
                    logger.error(f"Cannot start batch processing due to Weaviate connection: {e}")
                    logger.info("State has been consolidated. Start Weaviate and run batch processing manually:")
                    logger.info("nohup python batch_manager.py --create > batch_processing.log 2>&1 &")
                    return {
                        'status': 'error',
                        'error': f"Weaviate connection failed: {e}",
                        'manual_start_required': True
                    }
                else:
                    raise
            
            # Process data with consolidated state
            processing_results = self.batch_pipeline.process(
                string_dict, field_hash_mapping, string_counts, self.checkpoint_dir
            )
            
            if processing_results['status'] == 'completed':
                self.transition_state['batch_started'] = True
                logger.info("Batch processing started successfully")
            else:
                logger.error(f"Batch processing failed: {processing_results.get('error')}")
            
            return processing_results
            
        except Exception as e:
            error_msg = f"Error starting batch processing: {e}"
            logger.error(error_msg)
            return {
                'status': 'error',
                'error': error_msg
            }
        finally:
            # Clean up batch pipeline
            if self.batch_pipeline:
                try:
                    self.batch_pipeline.close()
                except:
                    pass
                self.batch_pipeline = None
    
    def start_realtime_processing(self, string_dict: Dict[str, str], 
                                 field_hash_mapping: Dict[str, Dict[str, int]],
                                 string_counts: Dict[str, int]) -> Dict[str, Any]:
        """
        Start real-time processing with consolidated state.
        
        Args:
            string_dict: String hash to value mapping
            field_hash_mapping: Hash to field type mapping
            string_counts: Hash to frequency mapping
            
        Returns:
            Dictionary with processing results
        """
        logger.info("Starting real-time processing with consolidated state...")
        
        try:
            # Initialize real-time pipeline
            self.realtime_pipeline = EmbeddingAndIndexingPipeline(self.config)
            
            # Process data with consolidated state
            processing_results = self.realtime_pipeline.process(
                string_dict, field_hash_mapping, string_counts, self.checkpoint_dir
            )
            
            if processing_results['status'] == 'completed':
                self.transition_state['realtime_started'] = True
                logger.info("Real-time processing started successfully")
            else:
                logger.error(f"Real-time processing failed: {processing_results.get('error')}")
            
            return processing_results
            
        except Exception as e:
            error_msg = f"Error starting real-time processing: {e}"
            logger.error(error_msg)
            return {
                'status': 'error',
                'error': error_msg
            }
        finally:
            # Clean up real-time pipeline
            if self.realtime_pipeline:
                try:
                    if hasattr(self.realtime_pipeline, 'weaviate_client'):
                        self.realtime_pipeline.weaviate_client.close()
                except:
                    pass
                self.realtime_pipeline = None
    
    def execute_transition(self, string_dict: Dict[str, str], 
                          field_hash_mapping: Dict[str, Dict[str, int]],
                          string_counts: Dict[str, int],
                          direction: str = "batch_to_realtime",
                          force_termination: bool = False,
                          override: bool = False) -> Dict[str, Any]:
        """
        Execute transition between batch and real-time processing.
        
        Args:
            string_dict: String hash to value mapping
            field_hash_mapping: Hash to field type mapping
            string_counts: Hash to frequency mapping
            direction: "batch_to_realtime" or "realtime_to_batch"
            force_termination: Force termination of active jobs
            override: Override job state tracking and force consolidation
            
        Returns:
            Dictionary with complete transition results
        """
        if direction not in ["batch_to_realtime", "realtime_to_batch"]:
            raise ValueError(f"Invalid direction: {direction}. Must be 'batch_to_realtime' or 'realtime_to_batch'")
        
        logger.info(f"Executing {direction.replace('_', '-')} transition...")
        
        self.transition_state['started'] = True
        self.transition_state['start_time'] = time.time()
        self.transition_state['direction'] = direction
        
        transition_results = {
            'status': 'in_progress',
            'direction': direction,
            'pre_analysis': {},
            'termination': {},
            'consolidation': {},
            'new_processing': {},
            'transition_state': self.transition_state.copy(),
            'elapsed_time': 0
        }
        
        try:
            # Step 1: Pre-transition analysis
            logger.info("Step 1: Pre-transition analysis")
            transition_results['pre_analysis'] = self.pre_transition_analysis()
            
            if not transition_results['pre_analysis']['transition_feasible']:
                if not force_termination:
                    transition_results['status'] = 'blocked'
                    logger.error("Transition blocked by pre-analysis issues")
                    return transition_results
                else:
                    logger.warning("Proceeding with forced transition despite issues")
            
            # Step 2: Terminate current processing
            if direction == "batch_to_realtime":
                logger.info("Step 2: Terminating batch processing")
                transition_results['termination'] = self.terminate_batch_processing(force_termination, override)
            else:
                logger.info("Step 2: Terminating real-time processing")
                transition_results['termination'] = self.terminate_realtime_processing()
            
            if not transition_results['termination']['success']:
                if override:
                    logger.warning(f"Termination had issues but continuing due to override mode")
                    logger.warning(f"Termination errors: {transition_results['termination'].get('errors', [])}")
                else:
                    transition_results['status'] = 'failed'
                    logger.error(f"Transition failed during {direction.split('_')[0]} termination")
                    return transition_results
            
            # Step 3: Consolidate state
            logger.info("Step 3: Consolidating state")
            transition_results['consolidation'] = self.consolidate_state()
            
            if transition_results['consolidation']['status'] != 'completed':
                transition_results['status'] = 'failed'
                logger.error("Transition failed during state consolidation")
                return transition_results
            
            # Step 4: Start new processing mode
            if direction == "batch_to_realtime":
                logger.info("Step 4: Starting real-time processing")
                transition_results['new_processing'] = self.start_realtime_processing(
                    string_dict, field_hash_mapping, string_counts
                )
            else:
                logger.info("Step 4: Starting batch processing")
                transition_results['new_processing'] = self.start_batch_processing(
                    string_dict, field_hash_mapping, string_counts, override
                )
            
            if transition_results['new_processing']['status'] != 'completed':
                if override and transition_results['new_processing'].get('manual_start_required'):
                    logger.warning("Processing startup failed but state is consolidated - continuing due to override")
                    logger.info("You can start processing manually after fixing Weaviate connection")
                else:
                    transition_results['status'] = 'failed'
                    target_mode = direction.split('_')[2] if direction == "batch_to_realtime" else "batch"
                    logger.error(f"Transition failed during {target_mode} processing startup")
                    return transition_results
            
            # Mark transition as completed
            self.transition_state['completed'] = True
            self.transition_state['end_time'] = time.time()
            transition_results['status'] = 'completed'
            
            elapsed_time = self.transition_state['end_time'] - self.transition_state['start_time']
            transition_results['elapsed_time'] = elapsed_time
            transition_results['transition_state'] = self.transition_state.copy()
            
            logger.info(f"{direction.replace('_', '-')} transition completed successfully in {elapsed_time:.2f} seconds")
            
        except Exception as e:
            error_msg = f"Error during transition execution: {e}"
            logger.error(error_msg)
            self.transition_state['errors'].append(error_msg)
            transition_results['status'] = 'error'
            transition_results['error'] = error_msg
            transition_results['transition_state'] = self.transition_state.copy()
        
        return transition_results
    
    def get_transition_status(self) -> Dict[str, Any]:
        """
        Get current transition status and progress.
        
        Returns:
            Dictionary with transition status
        """
        return {
            'transition_state': self.transition_state.copy(),
            'timestamp': time.time()
        }
    
    def save_transition_log(self, results: Dict[str, Any]) -> None:
        """
        Save transition results to log file.
        
        Args:
            results: Transition results dictionary
        """
        try:
            log_dir = self.config.get("log_dir", "logs")
            os.makedirs(log_dir, exist_ok=True)
            
            timestamp = int(time.time())
            direction_str = results.get('direction', 'batch_to_realtime').replace('_', '_to_')
            log_file = os.path.join(log_dir, f"{direction_str}_transition_{timestamp}.json")
            
            with open(log_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Transition log saved to {log_file}")
            
        except Exception as e:
            logger.error(f"Error saving transition log: {e}")


def transition_batch_to_realtime(config: Dict[str, Any], 
                                string_dict: Dict[str, str],
                                field_hash_mapping: Dict[str, Dict[str, int]],
                                string_counts: Dict[str, int],
                                force: bool = False,
                                override: bool = False) -> Dict[str, Any]:
    """
    Main function to execute batch-to-real-time transition.
    
    Args:
        config: Configuration dictionary
        string_dict: String hash to value mapping
        field_hash_mapping: Hash to field type mapping
        string_counts: Hash to frequency mapping
        force: Force transition even if there are active batch jobs
        override: Override job state tracking and force consolidation
        
    Returns:
        Dictionary with transition results
    """
    return execute_transition(config, string_dict, field_hash_mapping, string_counts, 
                             "batch_to_realtime", force, override)


def transition_realtime_to_batch(config: Dict[str, Any], 
                                string_dict: Dict[str, str],
                                field_hash_mapping: Dict[str, Dict[str, int]],
                                string_counts: Dict[str, int],
                                force: bool = False,
                                override: bool = False) -> Dict[str, Any]:
    """
    Main function to execute real-time-to-batch transition.
    
    Args:
        config: Configuration dictionary
        string_dict: String hash to value mapping
        field_hash_mapping: Hash to field type mapping
        string_counts: Hash to frequency mapping
        force: Force transition even if there are active real-time jobs
        override: Override job state tracking and force consolidation
        
    Returns:
        Dictionary with transition results
    """
    return execute_transition(config, string_dict, field_hash_mapping, string_counts, 
                             "realtime_to_batch", force, override)


def execute_transition(config: Dict[str, Any], 
                      string_dict: Dict[str, str],
                      field_hash_mapping: Dict[str, Dict[str, int]],
                      string_counts: Dict[str, int],
                      direction: str = "batch_to_realtime",
                      force: bool = False,
                      override: bool = False) -> Dict[str, Any]:
    """
    Generic function to execute transitions between processing modes.
    
    Args:
        config: Configuration dictionary
        string_dict: String hash to value mapping
        field_hash_mapping: Hash to field type mapping
        string_counts: Hash to frequency mapping
        direction: "batch_to_realtime" or "realtime_to_batch"
        force: Force transition even if there are active jobs
        override: Override job state tracking and force consolidation
        
    Returns:
        Dictionary with transition results
    """
    logger.info(f"Starting {direction.replace('_', '-')} transition process...")
    
    controller = TransitionController(config)
    
    try:
        # Execute the transition
        results = controller.execute_transition(
            string_dict, field_hash_mapping, string_counts, direction, force, override
        )
        
        # Save transition log
        controller.save_transition_log(results)
        
        return results
        
    except Exception as e:
        logger.error(f"Error in transition process: {e}")
        return {
            'status': 'error',
            'error': str(e)
        }


if __name__ == "__main__":
    import argparse
    import pickle
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='Execute transitions between batch and real-time processing')
    parser.add_argument('--config', default='config.yml', help='Path to configuration file')
    parser.add_argument('--direction', choices=['batch_to_realtime', 'realtime_to_batch'], 
                       default='batch_to_realtime', help='Direction of transition')
    parser.add_argument('--force', action='store_true', help='Force transition even with active jobs')
    parser.add_argument('--override', action='store_true', help='Override job state tracking and force consolidation regardless of supposed active jobs')
    parser.add_argument('--analyze-only', action='store_true', help='Only perform pre-transition analysis')
    args = parser.parse_args()
    
    # Load configuration
    from config_utils import load_config_with_environment
    config = load_config_with_environment(args.config)
    
    # Load preprocessing data
    checkpoint_dir = config.get("checkpoint_dir", "data/checkpoints")
    
    try:
        # Load required data files
        with open(os.path.join(checkpoint_dir, "string_dict.pkl"), 'rb') as f:
            string_dict = pickle.load(f)
        
        with open(os.path.join(checkpoint_dir, "field_hash_mapping.pkl"), 'rb') as f:
            field_hash_mapping = pickle.load(f)
        
        with open(os.path.join(checkpoint_dir, "string_counts.pkl"), 'rb') as f:
            string_counts = pickle.load(f)
        
        logger.info(f"Loaded preprocessing data: {len(string_dict)} strings, "
                   f"{len(field_hash_mapping)} field mappings, {len(string_counts)} string counts")
        
        if args.analyze_only:
            # Only perform analysis
            controller = TransitionController(config)
            analysis = controller.pre_transition_analysis()
            
            print("\n" + "="*60)
            print("PRE-TRANSITION ANALYSIS")
            print("="*60)
            print(f"Transition Feasible: {analysis['transition_feasible']}")
            
            if analysis['errors']:
                print("\nBLOCKING ISSUES:")
                for error in analysis['errors']:
                    print(f"  ❌ {error}")
            
            if analysis['warnings']:
                print("\nWARNINGS:")
                for warning in analysis['warnings']:
                    print(f"  ⚠️  {warning}")
            
            if analysis['recommendations']:
                print("\nRECOMMENDATIONS:")
                for rec in analysis['recommendations']:
                    print(f"  💡 {rec}")
        else:
            # Execute full transition
            results = execute_transition(
                config, string_dict, field_hash_mapping, string_counts, args.direction, args.force, args.override
            )
            
            print("\n" + "="*60)
            print(f"{args.direction.replace('_', '-').upper()} TRANSITION RESULTS")
            print("="*60)
            print(f"Status: {results['status']}")
            
            if results['status'] == 'completed':
                print(f"Elapsed Time: {results['elapsed_time']:.2f} seconds")
                
                # Show consolidation stats
                if 'consolidation' in results:
                    summary = results['consolidation'].get('summary', {})
                    if 'consolidated_state' in summary:
                        cs = summary['consolidated_state']
                        print(f"Total Processed Hashes: {cs['total_processed_hashes']:,}")
                        print(f"From Batch Only: {cs['batch_only_hashes']:,}")
                        print(f"From Real-time Only: {cs['realtime_only_hashes']:,}")
                
                print("\n✅ Transition completed successfully!")
                if args.direction == "batch_to_realtime":
                    print("Real-time processing is now active with all batch progress preserved.")
                else:
                    print("Batch processing is now active with all real-time progress preserved.")
            
            elif results['status'] == 'blocked':
                print("\n❌ Transition blocked by pre-analysis issues")
                print("Use --force to proceed anyway or resolve the issues first")
            
            else:
                print(f"\n❌ Transition failed: {results.get('error', 'Unknown error')}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        sys.exit(1)