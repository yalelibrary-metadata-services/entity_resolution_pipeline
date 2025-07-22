#!/bin/bash
# Local development environment setup for Entity Resolution Pipeline
# This script sets the environment to 'local' for manual stage management
# 
# Usage: 
#   ./run_local.sh             (runs in subshell - environment only lasts during script)
#   source ./run_local.sh      (sets environment in current shell - RECOMMENDED)
#   . ./run_local.sh           (same as source)

export PIPELINE_ENV=local

echo "==============================================="
echo "Entity Resolution Pipeline - LOCAL ENVIRONMENT"
echo "==============================================="
echo "Environment: $PIPELINE_ENV"
echo "Resource Profile: Development (moderate settings)"
echo ""
echo "Configuration Summary:"
echo "  - Preprocessing workers: 4"
echo "  - Feature workers: 4" 
echo "  - Classification workers: 8"
echo "  - Weaviate batch size: 100"
echo "  - String cache: 100K entries"
echo ""
echo "Environment variable PIPELINE_ENV=local is now set."
echo ""
echo "IMPORTANT: To use local settings, run commands in THIS shell session:"
echo ""
echo "Examples:"
echo "  # Run preprocessing only"
echo "  python -m src.orchestrating --start preprocessing --end preprocessing"
echo ""
echo "  # Or run preprocessing directly (standalone)"
echo "  python -m src.preprocessing"
echo ""
echo "  # Run embedding and indexing"
echo "  python -m src.orchestrating --start embedding_and_indexing --end embedding_and_indexing"
echo ""
echo "  # Run full pipeline"
echo "  python -m src.orchestrating"
echo ""
echo "  # Reset and run specific stage"
echo "  python -m src.orchestrating --reset preprocessing --start preprocessing"
echo ""
echo "NOTE: If you open a new terminal, run 'source ./run_local.sh' to set environment again."
echo ""
echo "Available stages: preprocessing, embedding_and_indexing, subject_quality_audit,"
echo "                 subject_imputation, training, classifying, reporting"
echo "==============================================="