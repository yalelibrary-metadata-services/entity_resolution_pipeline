"""
Batch State Consolidation Module for Entity Resolution

This module provides functionality to consolidate batch processing state
into real-time processing compatible format, enabling seamless transition
between processing modes while preserving all progress and failed requests.
"""

import os
import sys
import logging
import pickle
import json
import time
from typing import Dict, List, Set, Tuple, Any, Optional

logger = logging.getLogger(__name__)


class BatchStateConsolidator:
    """
    Consolidates batch processing state for transition to real-time processing.
    
    This class extracts and merges all batch processing state including:
    - Successfully processed hashes from batch jobs
    - Failed requests with retry information  
    - Queue state and job metadata
    - Blacklisted files and error tracking
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the consolidator with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.checkpoint_dir = config.get("checkpoint_dir", "data/checkpoints")
        
        # State containers
        self.batch_processed_hashes = set()
        self.realtime_processed_hashes = set()
        self.consolidated_processed_hashes = set()
        self.failed_requests = {}
        self.batch_jobs = {}
        self.queue_state = {}
        self.blacklisted_files = set()
        
        logger.info(f"Initialized BatchStateConsolidator for checkpoint dir: {self.checkpoint_dir}")
    
    def load_batch_state(self) -> Dict[str, Any]:
        """
        Load all batch processing state from checkpoint files.
        
        Returns:
            Dictionary with loaded state summary
        """
        logger.info("Loading batch processing state from checkpoints...")
        
        state_summary = {
            'batch_processed_hashes': 0,
            'failed_requests': 0,
            'batch_jobs': 0,
            'blacklisted_files': 0,
            'queue_active': 0,
            'queue_completed': 0,
            'errors': []
        }
        
        # Load batch processed hashes
        batch_hashes_path = os.path.join(self.checkpoint_dir, 'batch_processed_hashes.pkl')
        if os.path.exists(batch_hashes_path):
            try:
                with open(batch_hashes_path, 'rb') as f:
                    self.batch_processed_hashes = set(pickle.load(f))
                state_summary['batch_processed_hashes'] = len(self.batch_processed_hashes)
                logger.info(f"Loaded {len(self.batch_processed_hashes)} batch processed hashes")
            except Exception as e:
                error_msg = f"Error loading batch processed hashes: {e}"
                logger.error(error_msg)
                state_summary['errors'].append(error_msg)
        
        # Load failed requests
        failed_requests_path = os.path.join(self.checkpoint_dir, 'batch_failed_requests.pkl')
        if os.path.exists(failed_requests_path):
            try:
                with open(failed_requests_path, 'rb') as f:
                    self.failed_requests = pickle.load(f)
                state_summary['failed_requests'] = len(self.failed_requests)
                logger.info(f"Loaded {len(self.failed_requests)} failed requests")
            except Exception as e:
                error_msg = f"Error loading failed requests: {e}"
                logger.error(error_msg)
                state_summary['errors'].append(error_msg)
        
        # Load batch jobs
        batch_jobs_path = os.path.join(self.checkpoint_dir, 'batch_jobs.pkl')
        if os.path.exists(batch_jobs_path):
            try:
                with open(batch_jobs_path, 'rb') as f:
                    self.batch_jobs = pickle.load(f)
                state_summary['batch_jobs'] = len(self.batch_jobs)
                logger.info(f"Loaded {len(self.batch_jobs)} batch jobs")
                
                # Extract additional hashes from batch job metadata
                additional_hashes = self._extract_hashes_from_batch_jobs()
                if additional_hashes:
                    logger.info(f"Extracted {len(additional_hashes)} additional hashes from batch job metadata")
                    self.batch_processed_hashes.update(additional_hashes)
                    state_summary['batch_processed_hashes'] = len(self.batch_processed_hashes)
                    
            except Exception as e:
                error_msg = f"Error loading batch jobs: {e}"
                logger.error(error_msg)
                state_summary['errors'].append(error_msg)
        
        # Load queue state
        queue_state_path = os.path.join(self.checkpoint_dir, 'batch_queue_state.pkl')
        if os.path.exists(queue_state_path):
            try:
                with open(queue_state_path, 'rb') as f:
                    self.queue_state = pickle.load(f)
                state_summary['queue_active'] = len(self.queue_state.get('active_batch_queue', []))
                state_summary['queue_completed'] = len(self.queue_state.get('completed_batches', []))
                logger.info(f"Loaded queue state: {state_summary['queue_active']} active, {state_summary['queue_completed']} completed")
            except Exception as e:
                error_msg = f"Error loading queue state: {e}"
                logger.error(error_msg)
                state_summary['errors'].append(error_msg)
        
        # Load blacklisted files
        blacklisted_path = os.path.join(self.checkpoint_dir, 'batch_blacklisted_files.pkl')
        if os.path.exists(blacklisted_path):
            try:
                with open(blacklisted_path, 'rb') as f:
                    self.blacklisted_files = set(pickle.load(f))
                state_summary['blacklisted_files'] = len(self.blacklisted_files)
                logger.info(f"Loaded {len(self.blacklisted_files)} blacklisted files")
            except Exception as e:
                error_msg = f"Error loading blacklisted files: {e}"
                logger.error(error_msg)
                state_summary['errors'].append(error_msg)
        
        return state_summary
    
    def load_realtime_state(self) -> Dict[str, Any]:
        """
        Load existing real-time processing state.
        
        Returns:
            Dictionary with loaded state summary
        """
        logger.info("Loading real-time processing state...")
        
        state_summary = {
            'realtime_processed_hashes': 0,
            'errors': []
        }
        
        # Load real-time processed hashes
        realtime_hashes_path = os.path.join(self.checkpoint_dir, 'processed_hashes.pkl')
        if os.path.exists(realtime_hashes_path):
            try:
                with open(realtime_hashes_path, 'rb') as f:
                    self.realtime_processed_hashes = set(pickle.load(f))
                state_summary['realtime_processed_hashes'] = len(self.realtime_processed_hashes)
                logger.info(f"Loaded {len(self.realtime_processed_hashes)} real-time processed hashes")
            except Exception as e:
                error_msg = f"Error loading real-time processed hashes: {e}"
                logger.error(error_msg)
                state_summary['errors'].append(error_msg)
        
        return state_summary
    
    def _extract_hashes_from_batch_jobs(self) -> Set[str]:
        """
        Extract processed hashes from batch job metadata.
        
        Returns:
            Set of hash values from completed batch jobs
        """
        extracted_hashes = set()
        
        for job_id, job_info in self.batch_jobs.items():
            # Only extract from completed and processed jobs
            if (job_info.get('status') == 'completed' and 
                job_info.get('results_processed', False)):
                
                # Extract from custom_id_mapping if available
                if ('file_metadata' in job_info and 
                    'custom_id_mapping' in job_info['file_metadata']):
                    
                    custom_id_mapping = job_info['file_metadata']['custom_id_mapping']
                    for custom_id, item_data in custom_id_mapping.items():
                        hash_value = item_data.get('hash_value')
                        if hash_value:
                            extracted_hashes.add(hash_value)
        
        return extracted_hashes
    
    def consolidate_processed_hashes(self) -> Set[str]:
        """
        Merge batch and real-time processed hashes into consolidated set.
        
        Returns:
            Consolidated set of all processed hashes
        """
        logger.info("Consolidating processed hashes from batch and real-time processing...")
        
        # Merge both sets
        self.consolidated_processed_hashes = self.batch_processed_hashes.union(self.realtime_processed_hashes)
        
        # Log consolidation statistics
        batch_only = len(self.batch_processed_hashes - self.realtime_processed_hashes)
        realtime_only = len(self.realtime_processed_hashes - self.batch_processed_hashes)
        overlap = len(self.batch_processed_hashes.intersection(self.realtime_processed_hashes))
        total = len(self.consolidated_processed_hashes)
        
        logger.info(f"Hash consolidation complete:")
        logger.info(f"  Batch only: {batch_only:,}")
        logger.info(f"  Real-time only: {realtime_only:,}")
        logger.info(f"  Overlap: {overlap:,}")
        logger.info(f"  Total consolidated: {total:,}")
        
        return self.consolidated_processed_hashes
    
    def analyze_failed_requests(self) -> Dict[str, Any]:
        """
        Analyze failed requests for migration to real-time processing.
        
        Returns:
            Analysis of failed requests by category and retry status
        """
        logger.info("Analyzing failed requests for real-time migration...")
        
        analysis = {
            'total_failed': len(self.failed_requests),
            'by_category': {},
            'by_retry_count': {},
            'retryable': 0,
            'max_retries_exceeded': 0,
            'retryable_requests': [],
            'permanent_failures': []
        }
        
        max_retry_attempts = self.config.get("error_handling", {}).get("max_retry_attempts", 3)
        permanent_categories = {"validation"}  # Error types that shouldn't be retried
        
        for custom_id, failure_data in self.failed_requests.items():
            category = failure_data.get('error_category', 'other')
            retry_count = failure_data.get('retry_count', 0)
            
            # Count by category
            analysis['by_category'][category] = analysis['by_category'].get(category, 0) + 1
            
            # Count by retry attempts
            analysis['by_retry_count'][retry_count] = analysis['by_retry_count'].get(retry_count, 0) + 1
            
            # Determine if retryable
            if category in permanent_categories:
                analysis['permanent_failures'].append(custom_id)
            elif retry_count < max_retry_attempts:
                analysis['retryable'] += 1
                analysis['retryable_requests'].append({
                    'custom_id': custom_id,
                    'category': category,
                    'retry_count': retry_count,
                    'error_info': failure_data.get('error_info', {})
                })
            else:
                analysis['max_retries_exceeded'] += 1
        
        logger.info(f"Failed request analysis:")
        logger.info(f"  Total failed: {analysis['total_failed']:,}")
        logger.info(f"  Retryable: {analysis['retryable']:,}")
        logger.info(f"  Max retries exceeded: {analysis['max_retries_exceeded']:,}")
        logger.info(f"  Permanent failures: {len(analysis['permanent_failures']):,}")
        logger.info(f"  By category: {analysis['by_category']}")
        
        return analysis
    
    def save_consolidated_state(self, backup_existing: bool = True) -> Dict[str, Any]:
        """
        Save consolidated state to real-time compatible checkpoint files.
        
        Args:
            backup_existing: Whether to backup existing real-time checkpoints
            
        Returns:
            Dictionary with save operation results
        """
        logger.info("Saving consolidated state to real-time checkpoint format...")
        
        results = {
            'consolidated_hashes_saved': False,
            'failed_requests_migrated': False,
            'backups_created': [],
            'errors': []
        }
        
        # Create backups if requested
        if backup_existing:
            backup_suffix = f"_pre_consolidation_{int(time.time())}"
            self._create_backup_files(backup_suffix, results)
        
        # Save consolidated processed hashes to real-time format
        realtime_hashes_path = os.path.join(self.checkpoint_dir, 'processed_hashes.pkl')
        try:
            with open(realtime_hashes_path, 'wb') as f:
                pickle.dump(list(self.consolidated_processed_hashes), f)
            
            results['consolidated_hashes_saved'] = True
            logger.info(f"Saved {len(self.consolidated_processed_hashes):,} consolidated hashes to {realtime_hashes_path}")
            
        except Exception as e:
            error_msg = f"Error saving consolidated hashes: {e}"
            logger.error(error_msg)
            results['errors'].append(error_msg)
        
        # Save failed requests in real-time compatible format
        realtime_failed_path = os.path.join(self.checkpoint_dir, 'realtime_failed_requests.pkl')
        try:
            # Convert failed requests to real-time format
            realtime_failed_requests = self._convert_failed_requests_for_realtime()
            
            with open(realtime_failed_path, 'wb') as f:
                pickle.dump(realtime_failed_requests, f)
            
            results['failed_requests_migrated'] = True
            logger.info(f"Saved {len(realtime_failed_requests):,} failed requests to {realtime_failed_path}")
            
        except Exception as e:
            error_msg = f"Error saving failed requests: {e}"
            logger.error(error_msg)
            results['errors'].append(error_msg)
        
        return results
    
    def _create_backup_files(self, backup_suffix: str, results: Dict[str, Any]) -> None:
        """
        Create backup copies of existing checkpoint files.
        
        Args:
            backup_suffix: Suffix to append to backup files
            results: Results dictionary to update with backup information
        """
        backup_files = [
            'processed_hashes.pkl',
            'realtime_failed_requests.pkl'
        ]
        
        for filename in backup_files:
            original_path = os.path.join(self.checkpoint_dir, filename)
            if os.path.exists(original_path):
                backup_path = os.path.join(self.checkpoint_dir, f"{filename}{backup_suffix}")
                try:
                    import shutil
                    shutil.copy2(original_path, backup_path)
                    results['backups_created'].append(backup_path)
                    logger.info(f"Created backup: {backup_path}")
                except Exception as e:
                    error_msg = f"Error creating backup {backup_path}: {e}"
                    logger.error(error_msg)
                    results['errors'].append(error_msg)
    
    def _convert_failed_requests_for_realtime(self) -> Dict[str, Any]:
        """
        Convert batch failed requests to real-time compatible format.
        
        Returns:
            Dictionary of failed requests in real-time format
        """
        realtime_failed = {}
        
        for custom_id, failure_data in self.failed_requests.items():
            # Parse custom_id to extract hash and field info
            # Format: "hash_{hash_value}_field_{field_type}"
            try:
                parts = custom_id.split('_')
                if len(parts) >= 4 and parts[0] == 'hash' and parts[2] == 'field':
                    hash_value = parts[1]
                    field_type = parts[3]
                    
                    # Create real-time compatible entry
                    realtime_failed[hash_value] = {
                        'original_custom_id': custom_id,
                        'field_type': field_type,
                        'error_info': failure_data.get('error_info', {}),
                        'retry_count': failure_data.get('retry_count', 0),
                        'error_category': failure_data.get('error_category', 'other'),
                        'first_attempt': failure_data.get('first_attempt'),
                        'last_attempt': failure_data.get('last_attempt'),
                        'migrated_from_batch': True,
                        'migration_timestamp': time.time()
                    }
                    
            except Exception as e:
                logger.warning(f"Error converting failed request {custom_id}: {e}")
                continue
        
        return realtime_failed
    
    def get_consolidation_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of consolidation state.
        
        Returns:
            Dictionary with consolidation statistics and state
        """
        summary = {
            'timestamp': time.time(),
            'batch_state': {
                'processed_hashes': len(self.batch_processed_hashes),
                'failed_requests': len(self.failed_requests),
                'batch_jobs': len(self.batch_jobs),
                'blacklisted_files': len(self.blacklisted_files),
                'queue_active': len(self.queue_state.get('active_batch_queue', [])),
                'queue_completed': len(self.queue_state.get('completed_batches', []))
            },
            'realtime_state': {
                'processed_hashes': len(self.realtime_processed_hashes)
            },
            'consolidated_state': {
                'total_processed_hashes': len(self.consolidated_processed_hashes),
                'batch_only_hashes': len(self.batch_processed_hashes - self.realtime_processed_hashes),
                'realtime_only_hashes': len(self.realtime_processed_hashes - self.batch_processed_hashes),
                'overlap_hashes': len(self.batch_processed_hashes.intersection(self.realtime_processed_hashes))
            },
            'transition_ready': self._check_transition_readiness()
        }
        
        return summary
    
    def _check_transition_readiness(self) -> Dict[str, Any]:
        """
        Check if the system is ready for batch-to-real-time transition.
        
        Returns:
            Dictionary with readiness status and any issues
        """
        readiness = {
            'ready': True,
            'issues': [],
            'warnings': []
        }
        
        # Check for active batch jobs
        active_jobs = len(self.queue_state.get('active_batch_queue', []))
        if active_jobs > 0:
            readiness['ready'] = False
            readiness['issues'].append(f"{active_jobs} active batch jobs still running")
        
        # Check for pending batches
        pending_batches = len(self.queue_state.get('pending_batches', []))
        if pending_batches > 0:
            readiness['warnings'].append(f"{pending_batches} pending batches will be lost")
        
        # Check if consolidation would lose data
        if len(self.batch_processed_hashes) == 0 and len(self.batch_jobs) > 0:
            readiness['warnings'].append("Batch jobs exist but no processed hashes found - may need to process completed jobs first")
        
        return readiness


def consolidate_batch_state(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function to consolidate batch processing state for real-time transition.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with consolidation results and summary
    """
    logger.info("Starting batch state consolidation process...")
    
    try:
        # Initialize consolidator
        consolidator = BatchStateConsolidator(config)
        
        # Load all state
        batch_state = consolidator.load_batch_state()
        realtime_state = consolidator.load_realtime_state()
        
        # Consolidate processed hashes
        consolidated_hashes = consolidator.consolidate_processed_hashes()
        
        # Analyze failed requests
        failed_analysis = consolidator.analyze_failed_requests()
        
        # Save consolidated state
        save_results = consolidator.save_consolidated_state()
        
        # Get comprehensive summary
        summary = consolidator.get_consolidation_summary()
        
        # Combine all results
        results = {
            'status': 'completed',
            'summary': summary,
            'batch_state_loaded': batch_state,
            'realtime_state_loaded': realtime_state,
            'failed_request_analysis': failed_analysis,
            'save_results': save_results,
            'consolidated_hashes_count': len(consolidated_hashes),
            'transition_ready': summary['transition_ready']
        }
        
        logger.info("Batch state consolidation completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"Error in batch state consolidation: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'summary': {}
        }


if __name__ == "__main__":
    import argparse
    import yaml
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='Consolidate batch processing state for real-time transition')
    parser.add_argument('--config', default='config.yml', help='Path to configuration file')
    parser.add_argument('--dry-run', action='store_true', help='Analyze state without saving changes')
    args = parser.parse_args()
    
    # Load configuration
    from config_utils import load_config_with_environment
    config = load_config_with_environment(args.config)
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No files will be modified")
        # TODO: Implement dry-run analysis
    else:
        # Run consolidation
        results = consolidate_batch_state(config)
        
        # Print summary
        print("\n" + "="*50)
        print("BATCH STATE CONSOLIDATION SUMMARY")
        print("="*50)
        
        if results['status'] == 'completed':
            summary = results['summary']
            print(f"Consolidated Hashes: {summary['consolidated_state']['total_processed_hashes']:,}")
            print(f"Failed Requests: {results['failed_request_analysis']['total_failed']:,}")
            print(f"Retryable Failed: {results['failed_request_analysis']['retryable']:,}")
            
            if results['transition_ready']['ready']:
                print("\n✅ System is ready for batch-to-real-time transition")
            else:
                print("\n❌ System is NOT ready for transition:")
                for issue in results['transition_ready']['issues']:
                    print(f"  - {issue}")
        else:
            print(f"❌ Consolidation failed: {results.get('error', 'Unknown error')}")