#!/usr/bin/env python3
"""
Entity Resolution Pipeline for Library Catalog Data

Main entry point for running the entity resolution pipeline.
"""

import os
import sys
import argparse
import yaml
import logging
from pathlib import Path
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Entity Resolution Pipeline')
    parser.add_argument(
        '--stage', 
        type=str, 
        default='all',
        choices=[
            'preprocessing', 'embedding', 'indexing', 'imputation', 
            'ground_truth_queries', 'feature_engineering', 'classification',
            'all'
        ],
        help='Pipeline stage to run'
    )
    parser.add_argument(
        '--mode', 
        type=str, 
        default=None,
        choices=['dev', 'prod'],
        help='Operation mode (dev or prod)'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='config.yml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--resume', 
        action='store_true',
        help='Resume from last checkpoint'
    )
    parser.add_argument(
        '--checkpoint', 
        type=str, 
        default=None,
        help='Path to specific checkpoint file'
    )
    
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_directories(config):
    """Create necessary directories."""
    for dir_path in [
        config['system']['checkpoint_dir'],
        config['system']['output_dir'],
        config['system']['temp_dir']
    ]:
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")

def configure_logging(config):
    """Configure logging based on config."""
    log_level = getattr(logging, config['system']['log_level'])
    logging.getLogger().setLevel(log_level)
    
    # Create a file handler for output directory
    log_dir = Path(config['system']['output_dir']) / 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = log_dir / 'pipeline.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logging.getLogger().addHandler(file_handler)
    
    logger.info(f"Logging configured at level {config['system']['log_level']}")
    logger.info(f"Logs will be saved to {log_file}")

def main():
    """Main entry point for the entity resolution pipeline."""
    start_time = time.time()
    
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override mode if specified
    if args.mode:
        config['system']['mode'] = args.mode
    
    # Setup directories
    setup_directories(config)
    
    # Configure logging
    configure_logging(config)
    
    # Log execution information
    logger.info("="*80)
    logger.info(f"Starting Entity Resolution Pipeline")
    logger.info(f"Stage: {args.stage}")
    logger.info(f"Mode: {config['system']['mode']}")
    logger.info(f"Resume: {args.resume}")
    if args.checkpoint:
        logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info("="*80)
    
    # Import pipeline components
    from src.pipeline import Pipeline
    
    # Initialize pipeline
    pipeline = Pipeline(config)
    
    # Run requested stage(s)
    if args.stage == 'all':
        # Run full pipeline
        pipeline.run_all(resume=args.resume, checkpoint_path=args.checkpoint)
    else:
        # Run specific stage
        if args.stage == 'preprocessing':
            pipeline.run_preprocessing(resume=args.resume, checkpoint_path=args.checkpoint)
        elif args.stage == 'embedding':
            pipeline.run_embedding(resume=args.resume, checkpoint_path=args.checkpoint)
        elif args.stage == 'indexing':
            pipeline.run_indexing(resume=args.resume, checkpoint_path=args.checkpoint)
        elif args.stage == 'imputation':
            pipeline.run_imputation(resume=args.resume, checkpoint_path=args.checkpoint)
        elif args.stage == 'ground_truth_queries':
            pipeline.run_ground_truth_queries(resume=args.resume, checkpoint_path=args.checkpoint)
        elif args.stage == 'feature_engineering':
            pipeline.run_feature_engineering(resume=args.resume, checkpoint_path=args.checkpoint)
        elif args.stage == 'classification':
            pipeline.run_classification(resume=args.resume, checkpoint_path=args.checkpoint)
    
    # Log completion
    elapsed_time = time.time() - start_time
    logger.info("="*80)
    logger.info(f"Pipeline execution completed in {elapsed_time:.2f} seconds")
    logger.info("="*80)

if __name__ == "__main__":
    main()
