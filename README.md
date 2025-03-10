# Entity Resolution for Library Catalog Data

*This codebase was developed using Anthropic's Claude 3.7 Sonnet model.*
The project instructions and initial prompt used to define the project specifications are in the [prompts](prompts) folder.

## Overview

This system performs entity resolution for personal name entities in library catalog data. It addresses a challenging problem in bibliographic information management: determining when different catalog entries refer to the same real-world person across millions of records.

By integrating vector embeddings with a specialized classification pipeline, the system helps disambiguates personal identities with good precision and recall, aiming to improve metadata quality in library catalogs.

## The Challenge: Identity Resolution in Library Catalogs

### The Problem

Library catalogs contain millions of records with references to people who authored, edited, contributed to, or are subjects of various works. These references frequently appear in inconsistent forms:

- Different name forms: "Schubert, Franz" vs. "Schubert, F." vs. "Franz Schubert"
- Incomplete information: missing birth/death years or other identifiers
- Different languages and transliterations 
- Identical names for different people (e.g., "Samuel Butler" may refer to hundreds of distinct individuals)

Unlike traditional entity resolution problems, library catalog metadata primarily describes works rather than people, making identity resolution particularly challenging. Catalogers follow standards for name formatting ("authority control"), but practices vary across time, institutions, and languages.

### Real-World Impact

Without effective entity resolution:

- Users must manually sift through mixed search results
- Related works by the same person remain scattered across the catalog
- Collection analysis becomes unreliable
- Linked data initiatives are undermined by identity ambiguity
- Digital humanities research is hampered by noisy data

## Technical Approach

### Vector-Based Identity Resolution

This system approaches entity resolution as a machine learning classification problem, using dense vector representations to capture semantic similarities between catalog records. The pipeline:

1. **Vectorizes catalog text fields** using OpenAI's text-embedding-3-small model (1,536 dimensions)
2. **Indexes vectors efficiently** in Weaviate for similarity search
3. **Generates feature vectors** for candidate pairs based on multiple similarity metrics
4. **Trains a logistic regression classifier** to distinguish between matching and non-matching entities

This approach allows the system to learn subtle patterns in how catalog metadata represents personal identities across different contexts, languages, and time periods.

### Key Features and Innovations

- **Sophisticated feature engineering**: Beyond simple string similarity, the system calculates harmonic means between field vectors, birth/death year matching, and domain-specific interaction features
- **Vector-based "hot deck" imputation**: Missing values are filled using vector similarity to complete records
- **Batch and parallel processing**: All pipeline stages support efficient processing of large datasets
- **Perfect precision**: The system (given current training data) is capable of 1.0 precision, minimizing false positives that would incorrectly merge distinct identities
- **Language agnostic design**: Works across multiple languages without language-specific rules
- **Fully configurable pipeline**: All parameters can be tuned through a central configuration file

## Case Study: The "Franz Schubert" Problem

The system successfully disambiguates similar names like Franz Schubert (1797-1828), the Austrian composer, from Franz August Schubert (1806-1893), the German artist.

These cases are particularly challenging because:

- The name strings alone are identical ("Schubert, Franz")
- Their lifetimes partially overlap (1806-1828)
- Both were European and German-speaking
- Not all catalog records include birth/death years
- Domain differences (music vs. visual art) must be inferred from context

Our system correctly separates these identities by:

1. Recognizing distinctive vocabulary patterns in work titles and subjects
2. Leveraging birth/death years when available

## System Architecture

```
entity-resolution/
   ├── README.md               # Project documentation
   ├── config.yml              # Configuration parameters
   ├── docker-compose.yml      # Docker Compose for Weaviate
   ├── requirements.txt        # Python dependencies
   ├── main.py                 # Entry point script
   ├── src/                    # Source code
   │   ├── batch_parallel_preprocessing.py    # Data preprocessing
   │   ├── batch_parallel_embedding.py        # Vector embedding
   │   ├── batch_parallel_indexing.py         # Weaviate integration
   │   ├── batch_parallel_imputation.py       # Null value imputation
   │   ├── batch_parallel_querying.py         # Querying and match candidate retrieval
   │   ├── batch_parallel_feature_engineering.py # Feature engineering
   │   ├── batch_parallel_classification.py   # Classifier training/evaluation
   │   ├── pipeline.py         # Pipeline orchestration
   │   ├── reporting.py        # Reporting and visualization of pipeline results
   │   └── utils.py            # Utility functions
   ├── notebooks/              # Analysis notebooks
   │   └── evaluation.ipynb    # Results evaluation
   └── tests/                  # Testing scripts
       └── test_pipeline.py    # Pipeline verification
```

## Pipeline Components

The pipeline consists of modular, independently executable stages:

### 1. Preprocessing
Efficiently normalizes and deduplicates text fields, tracking frequency of duplicate strings. Maintains memory-efficient mappings between unique strings, embeddings, and person entity IDs using hash-based lookups.

### 2. Vector Embedding
Generates 1,536-dimensional embeddings using OpenAI's text-embedding-3-small model. Implements efficient batch processing, rate limiting, and checkpointing for processing millions of strings.

### 3. Weaviate Indexing
Creates a specialized schema for vector search, using named vectors by field type. Implements idempotent operations for reliable reindexing and efficient query patterns.

### 4. Null Value Imputation
Applies a vector-based hot deck approach to impute missing values. Uses the composite field vector to retrieve near-neighbors for missing fields and calculate average vectors.

### 5. Feature Engineering
Computes feature vectors for candidate pairs including:
- Vector cosine similarities for each field
- Harmonic means between different field similarities
- Birth/death year matching
- Low composite penalties for suspicious matches
- Various interaction features capturing non-linear relationships

### 6. Classification
Trains a logistic regression model using gradient descent on labeled ground truth data. Optimizes for precision while maintaining high recall, with configurable thresholds.

## Performance Metrics

Current performance metrics:

- **Precision**: 1.0 (no false positives)
- **Recall**: 0.90
- **F1 Score**: 0.95
- **ROC AUC**: 0.9999

Confusion matrix:
```
True Negatives: 11,559    False Positives: 0
False Negatives: 1,164    True Positives: 10,545
```

## Setup and Usage

### Requirements
- Python 3.8+
- Docker and Docker Compose
- Weaviate 1.24.x+
- OpenAI API access

### Installation

1. Clone the repository
   ```
   git clone https://github.com/username/entity-resolution.git
   cd entity-resolution
   ```

2. Install dependencies
   ```
   pip install -r requirements.txt
   ```

3. Start Weaviate
   ```
   docker compose up -d
   ```

4. Configure settings in `config.yml`

### Running the Pipeline

The pipeline can be run end-to-end or in stages:

```
# Run the complete pipeline
python main.py --stage all

# Run individual stages
python main.py --stage preprocessing
python main.py --stage embedding
python main.py --stage indexing
python main.py --stage imputation
python main.py --stage ground_truth_queries
python main.py --stage feature_engineering
python main.py --stage classification
python main.py --stage reporting
```

### Development vs. Production Mode

The system supports two operating modes:

```
# Development mode (processes subset of data)
python main.py --mode dev

# Production mode (processes complete dataset)
python main.py --mode prod
```

### Checkpointing

The pipeline maintains checkpoints to allow resuming from interruptions:

```
python main.py --stage classification --resume --checkpoint checkpoints/classification_latest.ckpt
```

## Evaluation and Analysis

After running the pipeline, explore the results:

```
jupyter notebook notebooks/evaluation.ipynb
```

Generated metrics, reports, and visualizations are stored in the `output` directory.

## Scalability and Performance

The system is designed to scale to tens of millions of records:

- Local development: 8 cores, 32GB RAM
- Production deployment: 64+ cores, 256GB+ RAM
- Configurable resource allocation
- Parallelized batch processing
- Memory-mapped file support for large datasets
- Efficient vector indexing and retrieval

## Future Directions

- Active learning for ambiguous cases
- Extended feature set for improved recall
- Integration with authority control and identity management systems

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
