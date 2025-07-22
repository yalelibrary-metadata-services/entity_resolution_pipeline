#!/usr/bin/env python3
"""
Standalone script to run Recursive Feature Elimination analysis.

This script loads the entity resolution pipeline, performs RFE on the
training data, and generates comprehensive reports on feature importance
and optimal feature subsets.
"""

import argparse
import logging
import os
import sys
import yaml
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_selection_rfe import RecursiveFeatureEliminator
from src.training import EntityClassifier, load_training_data
from src.feature_engineering import FeatureEngineering
from src.utils import setup_deterministic_behavior
from src.custom_features import register_custom_features
from src.preprocessing import load_hash_lookup, load_string_dict
from src.indexing import get_weaviate_client


def setup_logging(config: Dict[str, Any]) -> None:
    """Set up logging configuration."""
    log_level = config.get("log_level", "INFO")
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(config.get("log_dir", "logs"), "rfe_analysis.log"))
        ]
    )


def load_configuration(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_rfe_analysis(config: Dict[str, Any], override_enabled: bool = False) -> None:
    """
    Run the complete RFE analysis pipeline.
    
    Args:
        config: Configuration dictionary
        override_enabled: Force RFE to run even if disabled in config
    """
    logger = logging.getLogger(__name__)
    
    # Check if feature selection is enabled
    feature_selection_config = config.get('feature_selection', {})
    if not feature_selection_config.get('enabled', False) and not override_enabled:
        logger.warning("Feature selection is disabled in configuration. Use --force to override.")
        return
    
    # Set random seed for reproducibility
    seed = config.get('random_seed', 42)
    setup_deterministic_behavior(seed)
    logger.info(f"Running RFE analysis with random seed: {seed}")
    
    # Load hash lookup and string dict
    checkpoint_dir = config.get('checkpoint_dir', 'data/checkpoints')
    hash_lookup_path = os.path.join(checkpoint_dir, 'hash_lookup.pkl')
    string_dict_path = os.path.join(checkpoint_dir, 'string_dict.pkl')
    
    if not os.path.exists(hash_lookup_path):
        logger.error(f"hash_lookup.pkl not found at {hash_lookup_path}. Run preprocessing first.")
        return
        
    logger.info("Loading preprocessed data...")
    hash_lookup = load_hash_lookup(hash_lookup_path)
    string_dict = load_string_dict(string_dict_path) if os.path.exists(string_dict_path) else None
    
    # Initialize Weaviate client
    logger.info("Initializing Weaviate client...")
    weaviate_client = get_weaviate_client(config)
    
    # Initialize feature engineering
    logger.info("Initializing feature engineering module...")
    feature_engineering = FeatureEngineering(config, weaviate_client, hash_lookup)
    
    # Register custom features
    try:
        register_custom_features(feature_engineering, config)
        logger.info(f"Registered {len(feature_engineering.get_feature_names())} features")
    except Exception as e:
        logger.error(f"Error registering custom features: {str(e)}")
    
    # Load ground truth training data
    ground_truth_path = os.path.join(
        config.get("ground_truth_dir", "data/ground_truth"),
        config.get("labeled_matches_file", "labeled_matches.csv")
    )
    
    logger.info(f"Loading training data from {ground_truth_path}...")
    labeled_pairs, label_counts = load_training_data(ground_truth_path)
    logger.info(f"Loaded {label_counts['total']} pairs: "
                f"{label_counts['match']} matches, {label_counts['non_match']} non-matches")
    
    # Compute features for all pairs
    logger.info("Computing features for labeled pairs...")
    X, y = feature_engineering.compute_features(labeled_pairs, string_dict)
    feature_names = feature_engineering.get_feature_names()
    
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Features: {feature_names}")
    
    # Initialize RFE
    logger.info("Initializing Recursive Feature Eliminator...")
    rfe = RecursiveFeatureEliminator(config)
    
    # Run RFE
    logger.info("Starting RFE analysis...")
    rfe.fit(X, y, feature_names, EntityClassifier, config)
    
    # Generate and display report
    report = rfe.generate_report()
    print("\n" + report)
    
    # Save report to file
    report_path = os.path.join(
        config.get("output_dir", "data/output"),
        "rfe_analysis_report.txt"
    )
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"Saved text report to {report_path}")
    
    # Generate visualizations
    try:
        # Performance vs number of features plot
        plot_path = os.path.join(
            config.get("output_dir", "data/output"),
            "rfe_performance_plot.png"
        )
        rfe.plot_scores(save_path=plot_path)
        
        # Feature importance plot
        importance_plot_path = os.path.join(
            config.get("output_dir", "data/output"),
            "rfe_feature_importance.png"
        )
        rfe.plot_feature_importance(save_path=importance_plot_path)
        
    except Exception as e:
        logger.error(f"Error generating plots: {str(e)}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("RFE ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nOptimal number of features: {rfe.n_features_}")
    print(f"Selected features: {', '.join(rfe.get_selected_features())}")
    
    optimal_idx = rfe.scores_[rfe.scoring_metric].index(max(rfe.scores_[rfe.scoring_metric]))
    print(f"\nPerformance with optimal features:")
    print(f"  Precision: {rfe.scores_['precision'][optimal_idx]:.4f}")
    print(f"  Recall: {rfe.scores_['recall'][optimal_idx]:.4f}")
    print(f"  F1-score: {rfe.scores_['f1'][optimal_idx]:.4f}")
    print(f"  Accuracy: {rfe.scores_['accuracy'][optimal_idx]:.4f}")
    
    print(f"\nResults saved to:")
    print(f"  - {rfe.results_path}")
    print(f"  - {report_path}")
    print(f"  - {plot_path}")
    print(f"  - {importance_plot_path}")
    
    # Clean up Weaviate client
    try:
        from src.indexing import close_weaviate_client
        close_weaviate_client(weaviate_client)
        logger.info("Closed Weaviate client connection")
    except Exception as e:
        logger.warning(f"Error closing Weaviate client: {e}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Run Recursive Feature Elimination analysis for entity resolution"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yml",
        help="Path to configuration file (default: config.yml)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force RFE to run even if disabled in configuration"
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["precision", "recall", "f1", "accuracy"],
        help="Override the scoring metric from configuration"
    )
    parser.add_argument(
        "--min-features",
        type=int,
        help="Override minimum features to select"
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        help="Override number of cross-validation folds"
    )
    parser.add_argument(
        "--step",
        type=int,
        help="Override step size (features to remove per iteration)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_configuration(args.config)
    
    # Override configuration with command line arguments
    if args.metric:
        config.setdefault('feature_selection', {}).setdefault('rfe_config', {})['scoring_metric'] = args.metric
    if args.min_features:
        config.setdefault('feature_selection', {}).setdefault('rfe_config', {})['min_features_to_select'] = args.min_features
    if args.cv_folds:
        config.setdefault('feature_selection', {}).setdefault('rfe_config', {})['cv_folds'] = args.cv_folds
    if args.step:
        config.setdefault('feature_selection', {}).setdefault('rfe_config', {})['step'] = args.step
    
    # Setup logging
    os.makedirs(config.get("log_dir", "logs"), exist_ok=True)
    setup_logging(config)
    
    # Run analysis
    try:
        run_rfe_analysis(config, override_enabled=args.force)
    except Exception as e:
        logging.error(f"RFE analysis failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()