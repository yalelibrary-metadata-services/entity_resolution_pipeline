#!/usr/bin/env python3
"""
Entity Resolution Pipeline for Yale University Library Catalog

Main entry point for running the entity resolution pipeline.
This script coordinates the execution of all pipeline stages.

Usage:
  python main.py --config config.yml [--start STAGE] [--end STAGE] [--reset STAGE1 STAGE2 ...]
"""

import os
import sys
import logging
import argparse
import yaml
import time
from typing import Dict, List, Any

# Add the project root and src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import pipeline modules
from src.utils import setup_logging, create_directory_structure, get_memory_usage
from src.orchestrating import PipelineOrchestrator

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Entity Resolution Pipeline')
    
    parser.add_argument('--config', default='config.yml',
                      help='Path to configuration file')
    
    parser.add_argument('--start',
                      help='Stage to start from (preprocessing, embedding_and_indexing, training, classifying, reporting)')
    
    parser.add_argument('--end',
                      help='Stage to end at (preprocessing, embedding_and_indexing, training, classifying, reporting)')
    
    parser.add_argument('--reset', nargs='*',
                      help='Stages to reset before running')
    
    parser.add_argument('--docker', action='store_true',
                      help='Run Weaviate in Docker')
    
    parser.add_argument('--resume', action='store_true',
                      help='Resume pipeline from last completed stage')
    
    parser.add_argument('--status', action='store_true',
                      help='Show current pipeline execution status and exit')
                      
    parser.add_argument('--disable-scaling', action='store_true',
                      help='EXPERIMENTAL: Disable all feature scaling and use raw values directly')
    
    # Transition commands
    parser.add_argument('--switch-to-realtime', action='store_true',
                      help='Switch from batch to real-time embedding processing')
    
    parser.add_argument('--force-transition', action='store_true',
                      help='Force transition even with active batch jobs')
    
    parser.add_argument('--analyze-transition', action='store_true',
                      help='Analyze readiness for batch-to-real-time transition')
    
    return parser.parse_args()

def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    except Exception as e:
        print(f"Error loading configuration: {str(e)}")
        sys.exit(1)

def start_weaviate_docker(config):
    """
    Start Weaviate using Docker Compose.
    
    Args:
        config: Configuration dictionary
    """
    try:
        import subprocess
        docker_compose_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'docker-compose.yml')
        
        if not os.path.exists(docker_compose_path):
            print(f"Docker Compose file not found: {docker_compose_path}")
            sys.exit(1)
        
        print("Starting Weaviate using Docker Compose...")
        subprocess.run(['docker-compose', '-f', docker_compose_path, 'up', '-d'], check=True)
        
        # Wait for Weaviate to be ready
        print("Waiting for Weaviate to be ready...")
        time.sleep(10)
        
    except Exception as e:
        print(f"Error starting Weaviate Docker: {str(e)}")
        sys.exit(1)

def load_preprocessing_data(checkpoint_dir: str) -> tuple:
    """Load preprocessing data required for transition operations."""
    try:
        import pickle
        
        # Load string dictionary
        with open(os.path.join(checkpoint_dir, "string_dict.pkl"), 'rb') as f:
            string_dict = pickle.load(f)
        
        # Load field hash mapping
        with open(os.path.join(checkpoint_dir, "field_hash_mapping.pkl"), 'rb') as f:
            field_hash_mapping = pickle.load(f)
        
        # Load string counts
        with open(os.path.join(checkpoint_dir, "string_counts.pkl"), 'rb') as f:
            string_counts = pickle.load(f)
        
        return string_dict, field_hash_mapping, string_counts
        
    except FileNotFoundError as e:
        print(f"❌ Preprocessing data not found: {str(e)}")
        print("💡 Run preprocessing stage first: python main.py --config config.yml --start preprocessing --end preprocessing")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading preprocessing data: {str(e)}")
        sys.exit(1)

def handle_transition_commands(args, config: Dict[str, Any], logger):
    """Handle batch-to-real-time transition commands."""
    # Import transition modules with fallback
    try:
        from src.transition_controller import TransitionController, transition_batch_to_realtime
        from src.batch_state_consolidator import BatchStateConsolidator
    except ImportError as e:
        print(f"❌ Transition modules not available: {e}")
        print("💡 Please ensure all transition modules are properly installed")
        sys.exit(1)
    
    checkpoint_dir = config.get("checkpoint_dir", "data/checkpoints")
    
    # Handle analyze transition command
    if args.analyze_transition:
        print("\n" + "="*60)
        print("BATCH-TO-REAL-TIME TRANSITION ANALYSIS")
        print("="*60)
        
        try:
            controller = TransitionController(config)
            analysis = controller.pre_transition_analysis()
            
            print(f"Transition Feasible: {'✅ YES' if analysis['transition_feasible'] else '❌ NO'}")
            
            if analysis['errors']:
                print("\n🚫 BLOCKING ISSUES:")
                for error in analysis['errors']:
                    print(f"  • {error}")
            
            if analysis['warnings']:
                print("\n⚠️  WARNINGS:")
                for warning in analysis['warnings']:
                    print(f"  • {warning}")
            
            if analysis['recommendations']:
                print("\n💡 RECOMMENDATIONS:")
                for rec in analysis['recommendations']:
                    print(f"  • {rec}")
            
            # Show summary stats
            if 'batch_state' in analysis:
                batch_state = analysis['batch_state']
                print(f"\n📊 BATCH STATE:")
                print(f"  • Processed hashes: {batch_state.get('processed_hashes', 0):,}")
                print(f"  • Failed requests: {batch_state.get('failed_requests', 0):,}")
                print(f"  • Active batch jobs: {batch_state.get('queue_active', 0):,}")
            
            if 'realtime_state' in analysis:
                realtime_state = analysis['realtime_state']
                print(f"\n📊 REAL-TIME STATE:")
                print(f"  • Processed hashes: {realtime_state.get('processed_hashes', 0):,}")
                
        except Exception as e:
            logger.error(f"Error in transition analysis: {e}")
            print(f"❌ Analysis failed: {str(e)}")
            sys.exit(1)
    
    # Handle switch to real-time command
    elif args.switch_to_realtime:
        print("\n" + "="*60)
        print("EXECUTING BATCH-TO-REAL-TIME TRANSITION")
        print("="*60)
        
        try:
            # Load preprocessing data
            string_dict, field_hash_mapping, string_counts = load_preprocessing_data(checkpoint_dir)
            logger.info(f"Loaded preprocessing data: {len(string_dict)} strings, "
                       f"{len(field_hash_mapping)} field mappings, {len(string_counts)} string counts")
            
            # Execute transition
            results = transition_batch_to_realtime(
                config, string_dict, field_hash_mapping, string_counts, args.force_transition
            )
            
            print(f"\n📋 TRANSITION RESULTS")
            print("="*30)
            print(f"Status: {results['status']}")
            
            if results['status'] == 'completed':
                print(f"✅ Transition completed successfully!")
                print(f"⏱️  Elapsed Time: {results['elapsed_time']:.2f} seconds")
                
                # Show consolidation stats
                if 'consolidation' in results:
                    summary = results['consolidation'].get('summary', {})
                    if 'consolidated_state' in summary:
                        cs = summary['consolidated_state']
                        print(f"📊 Total Processed Hashes: {cs['total_processed_hashes']:,}")
                        print(f"📊 From Batch Only: {cs['batch_only_hashes']:,}")
                        print(f"📊 From Real-time Only: {cs['realtime_only_hashes']:,}")
                
                print("\n🚀 Real-time processing is now active with all batch progress preserved.")
                print("💡 You can now run: python main.py --config config.yml")
                
            elif results['status'] == 'blocked':
                print("❌ Transition blocked by pre-analysis issues")
                print("💡 Use --force-transition to proceed anyway or resolve the issues first")
                
                if 'pre_analysis' in results:
                    analysis = results['pre_analysis']
                    if analysis.get('errors'):
                        print("\n🚫 BLOCKING ISSUES:")
                        for error in analysis['errors']:
                            print(f"  • {error}")
                
            else:
                print(f"❌ Transition failed: {results.get('error', 'Unknown error')}")
                if 'transition_state' in results:
                    ts = results['transition_state']
                    if ts.get('errors'):
                        print("\n🔍 ERROR DETAILS:")
                        for error in ts['errors']:
                            print(f"  • {error}")
                sys.exit(1)
                
        except Exception as e:
            logger.error(f"Error in transition execution: {e}")
            print(f"❌ Transition failed: {str(e)}")
            sys.exit(1)

def main():
    """
    Main function to run the entity resolution pipeline.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    # Create directory structure
    create_directory_structure(config)
    
    # Start Weaviate Docker if requested
    if args.docker:
        start_weaviate_docker(config)
    
    # Initialize checkpointing system
    from src.checkpoint_manager import get_checkpoint_manager
    checkpoint_manager = get_checkpoint_manager(config)
    
    # Display status if requested
    if args.status:
        progress = checkpoint_manager.get_pipeline_progress()
        print("\nEntity Resolution Pipeline Status")
        print("===================================\n")
        print(f"Last completed stage: {progress['last_completed_stage'] or 'None'}")
        print(f"Completed stages: {', '.join(progress['completed_stages']) if progress['completed_stages'] else 'None'}")
        
        # Determine next stage to run
        orchestrator = PipelineOrchestrator(args.config)
        next_stage = checkpoint_manager.get_next_stage(orchestrator.stages)
        print(f"Next stage to run: {next_stage or 'All stages completed'}\n")
        
        return
    
    # Handle transition commands
    if args.analyze_transition or args.switch_to_realtime:
        handle_transition_commands(args, config, logger)
        return
    
    # Log initial information
    logger.info("Starting Entity Resolution Pipeline")
    logger.info(f"Configuration loaded from {args.config}")
    logger.info(f"Initial memory usage: {get_memory_usage()}")
    
    # Add disable_scaling flag to config if specified
    if args.disable_scaling:
        config['disable_scaling'] = True
        logger.warning("*** EXPERIMENTAL: Feature scaling completely disabled - will use raw feature values ***")
    
    # Analyze feature configuration
    from src.utils import analyze_feature_configuration
    feature_analysis = analyze_feature_configuration(config)
    logger.info(f"Enabled base features: {feature_analysis['enabled_features']}")
    logger.info(f"Enabled custom features: {feature_analysis['custom_features']}")
    logger.info(f"Active feature substitutions: {feature_analysis['substitutions']}")
    logger.info(f"Effective feature set: {feature_analysis['effective_features']}")
    
    
    # Create and run orchestrator
    orchestrator = PipelineOrchestrator(args.config)
    
    # Set recreate_collections flag if embedding_and_indexing is being reset
    if args.reset and 'embedding_and_indexing' in args.reset:
        orchestrator.config['recreate_collections'] = True
        logger.info("Setting recreate_collections flag to True due to reset flag")
        
    try:
        # Run pipeline
        metrics = orchestrator.run_pipeline(
            args.start, 
            args.end, 
            args.reset,
            args.resume  # Pass resume flag
        )
        
        # Log completion
        logger.info("Entity Resolution Pipeline completed successfully")
        logger.info(f"Total runtime: {metrics['total_runtime']:.2f} seconds")
        logger.info(f"Final memory usage: {get_memory_usage()}")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
