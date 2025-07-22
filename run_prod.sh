#!/bin/bash
# Production environment setup for Entity Resolution Pipeline
# This script sets the environment to 'prod' for manual stage management
# 
# Usage: 
#   ./run_prod.sh              (runs in subshell - environment only lasts during script)
#   source ./run_prod.sh       (sets environment in current shell - RECOMMENDED)
#   . ./run_prod.sh            (same as source)

export PIPELINE_ENV=prod

echo "=================================================="
echo "Entity Resolution Pipeline - PRODUCTION ENVIRONMENT"
echo "=================================================="
echo "Environment: $PIPELINE_ENV"
echo "Resource Profile: High-Performance (64 cores, 247GB RAM)"
echo ""
echo "Configuration Summary:"
echo "  - Preprocessing workers: 32 (8x local)"
echo "  - Feature workers: 48 (12x local)"
echo "  - Classification workers: 32 (4x local)"
echo "  - Weaviate batch size: 1,000 (10x local)"
echo "  - String cache: 2M entries (20x local)"
echo "  - Vector cache: 1M entries (20x local)"
echo ""
echo "⚠️  WARNING: Production settings use significant system resources!"
echo "   Monitor CPU and memory usage during pipeline execution."
echo ""
echo "Environment variable PIPELINE_ENV=prod is now set."
echo ""
echo "IMPORTANT: To use production settings, run commands in THIS shell session:"
echo ""
echo "Examples:"
echo "  # Run preprocessing only"
echo "  python -m src.orchestrating --start preprocessing --end preprocessing"
echo ""
echo "  # Or run preprocessing directly (standalone)"
echo "  python -m src.preprocessing"
echo ""
echo "  # Run embedding and indexing with production parallelism"
echo "  python -m src.orchestrating --start embedding_and_indexing --end embedding_and_indexing"
echo ""
echo "  # Run full pipeline with production settings"
echo "  python -m src.orchestrating"
echo ""
echo "  # Reset and run specific stage"
echo "  python -m src.orchestrating --reset preprocessing --start preprocessing"
echo ""
echo "NOTE: If you open a new terminal, run 'source ./run_prod.sh' to set environment again."
echo ""
echo "Available stages: preprocessing, embedding_and_indexing, subject_quality_audit,"
echo "                 subject_imputation, training, classifying, reporting"
echo "=================================================="