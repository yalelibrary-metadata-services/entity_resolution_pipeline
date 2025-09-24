# Entity Resolution Pipeline

A production-ready system for identifying and resolving person entities across MARC 21 library catalog records. Achieves **98.51% precision** and **95.37% recall**.

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Docker and Docker Compose (for Weaviate vector database)
- OpenAI API key (for embeddings)
- Anthropic API key (for individual record classification)

### Installation
```bash
# Clone and setup environment
git clone <repository-url>
cd entity_resolution_pipeline
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Configure API keys
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
echo "ANTHROPIC_API_KEY=your_anthropic_api_key_here" >> .env

# Start Weaviate vector database
docker compose up -d
# Wait ~30 seconds for startup
curl http://localhost:8080/v1/.well-known/ready  # Check readiness
```

### Basic Usage
```bash
# Run complete pipeline
python main.py --config config.yml

# Run with batch embeddings (50% cost savings)
# Set use_batch_embeddings: true in config.yml
python main.py --config config.yml

# Resume from checkpoint
python main.py --resume

# Check status
python main.py --status
```

### Pipeline Stage Commands
```bash
# Run multiple stages
python main.py --start preprocessing --end training
python main.py --start preprocessing --end training --reset

# Run individual stages (one at a time)
python main.py --start preprocessing --end preprocessing
python main.py --start preprocessing --end preprocessing --reset preprocessing

python main.py --start embedding_and_indexing --end embedding_and_indexing
python main.py --start embedding_and_indexing --end embedding_and_indexing --reset embedding_and_indexing

python main.py --start training --end training
python main.py --start training --end training --reset training

python main.py --start classifying --end classifying
python main.py --start classifying --end classifying --reset classifying

python main.py --start reporting --end reporting
python main.py --start reporting --end reporting --reset reporting

# Subject enhancement stages (production only - requires full dataset)
python main.py --start subject_quality --end subject_quality
python main.py --start subject_quality --end subject_quality --reset subject_quality

python main.py --start subject_imputation --end subject_imputation
python main.py --start subject_imputation --end subject_imputation --reset subject_imputation

# Run both subject enhancement stages
python main.py --start subject_quality --end subject_imputation
python main.py --start subject_quality --end subject_imputation --reset subject_quality subject_imputation
```

### Batch Processing Commands
```bash
# Automated batch processing (recommended)
python batch_manager.py --create

# Manual batch operations
python main.py --batch-status     # Check batch job status
python main.py --batch-results    # Download and process results

# Batch manager operations
python batch_manager.py --status
python batch_manager.py --cancel  # Cancel failed jobs
python batch_manager.py --reset   # Clean checkpoint files
```

### Environment Configuration
```bash
# Local development (default)
python main.py --config config.yml

# Production settings (required for subject imputation)
PIPELINE_ENV=prod python main.py --config config.yml
```

## 📊 Performance
- **Precision**: 98.51% (11,507 TP, 174 FP)
- **Recall**: 95.37% (11,507 TP, 559 FN)
- **F1-Score**: 96.91%
- **99.23% reduction** in pairwise comparisons through vector similarity

## 🏗️ System Architecture

### Multi-Layer Architecture
The system implements three complementary processing layers:

1. **Main Entity Resolution Pipeline**: Identifies matching person entities across catalog records
2. **Individual Record Classification**: Classifies individual records into hierarchical taxonomy categories
3. **Subject Enhancement Pipeline**: Automated quality audit and imputation for subject fields using vector similarity *(production only)*

### Technology Stack
- **Vector Database**: Weaviate with HNSW indexing and MUVERA compression
- **Multi-Provider Embeddings**: OpenAI, Jina ColBERT v2, Mistral, Jina AI, Voyage AI, Cohere, HuggingFace
- **ML Framework**: Custom logistic regression with gradient descent
- **Taxonomy Classification**: SetFit (Sentence Transformers + logistic head)
- **Parallel Processing**: asyncio/aiohttp for API rate limit optimization

## 📋 Data Flow & Processing

### Input Data Format
The pipeline processes CSV files extracted from Yale University Library's BIBFRAME catalog:

```csv
composite,person,roles,title,provision,subjects,personId,setfit_prediction,is_parent_category
"Contributor: Bach, Johann Sebastian, 1685-1750
Title: The Well-Tempered Clavier
Attribution: edited by Johann Sebastian Bach
Subjects: Keyboard music; Fugues
Provision information: Leipzig: Breitkopf & Härtel, 1985","Bach, Johann Sebastian, 1685-1750",Composer,The Well-Tempered Clavier,"Leipzig: Breitkopf & Härtel, 1985","Keyboard music; Fugues",12345#Agent700-1,Music and Sound Arts,FALSE
```

### Core Pipeline Flow
```
Input CSV → Preprocessing → Embedding & Indexing → Subject Enhancement* → Training → Classification → Reporting
                                                        ↓
                                               *Production only (requires full dataset)
```

## 🔧 Pipeline Stages Deep Dive

### 1. Preprocessing (`src/preprocessing.py`)
- **Purpose**: Clean and deduplicate input data
- **Key Operations**: 
  - xxHash3-128 based hash deduplication (fastest with lowest collision risk)
  - Pure in-memory processing (15,000-18,000 rows/sec)
  - String frequency analysis and field mapping

### 2. Embedding & Indexing
**Multi-Provider Architecture** (`src/embedding_providers/`):
- **Provider Factory**: Unified system supporting 7 embedding providers
- **Jina ColBERT v2**: Multi-vector embeddings (128D per token) with MUVERA compression
- **OpenAI**: text-embedding-3-small/large with Batch API optimization
- **Automated Batch Processing**: Self-managing queue with 50% cost savings

**Real-time Processing** (`src/embedding_and_indexing.py`):
- Multi-provider support with automatic detection
- Direct Weaviate indexing with MUVERA compression

### 3. Subject Enhancement *(Production Only)*
**Requires full dataset - not available in development environment**

**Quality Audit** (`src/subject_quality.py`):
- Evaluates existing subject field quality using composite field vector similarity
- Automatically identifies low-quality subject assignments
- Applies high-confidence improvements with configurable thresholds

**Subject Imputation** (`src/subject_imputation.py`):
- Fills missing subject fields using vector join strategy
- Calculates weighted centroid from semantically similar composite fields
- Implements confidence scoring for imputation quality

### 4. Training (`src/training.py`)
- **Purpose**: Train logistic regression classifier on labeled entity pairs
- **Algorithm**: Custom gradient descent with L2 regularization
- **Features**: 7 engineered similarity features with domain-specific scaling

### 5. Classification (`src/classifying.py`)
- **Purpose**: Apply trained model to identify entity matches
- **Features**: Batch processing, transitive clustering, confidence scoring

### 6. Reporting (`src/reporting.py`)
- **Purpose**: Generate comprehensive analysis and visualizations
- **Outputs**: Interactive HTML dashboards, CSV exports, performance visualizations

## 📈 Configuration

Key settings in `config.yml`:

```yaml
# Environment (local/prod)
# Subject imputation requires: PIPELINE_ENV=prod

# Embedding provider
embedding:
  provider: "openai"  # openai, jina_colbert, mistral, etc.

# Batch processing (recommended for production)
use_batch_embeddings: true
use_automated_queue: true

# Subject enhancement (production only)
subject_quality_audit:
  enabled: true
subject_imputation:
  enabled: true  # Requires full dataset
```

## 🔍 Troubleshooting

**Weaviate Connection**:
```bash
docker compose ps
docker compose logs weaviate
curl http://localhost:8080/v1/.well-known/ready
```

**Pipeline Failures**:
- Check logs in `data/logs/pipeline.log`
- Use stage-specific execution: `--start <stage> --end <stage>`
- Add `--reset` flag to clear and restart stages

**Subject Enhancement Issues**:
- Subject imputation requires `PIPELINE_ENV=prod` and full dataset
- Quality audit can run in development but imputation cannot

**Batch Processing**:
```bash
python batch_manager.py --status
python batch_manager.py --cancel  # Cancel failed jobs
```

## 📚 Documentation

- `project_structure.md` - Project organization
- `data/output/` - Results and visualizations
- `EXECUTIVE_REPORT.md` - Performance summary
- `FEATURE_ENGINEERING_GUIDE.md` - Advanced feature analysis
