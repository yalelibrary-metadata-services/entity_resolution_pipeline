# Entity Resolution System for Library Catalog Data

This system performs entity resolution for personal identity records in the Yale University Library catalog using vector embeddings and machine learning.

## Overview

The entity resolution pipeline disambiguates and clusters personal name entities in library catalog records. It uses 1,536-dimensional vector embeddings from OpenAI, Weaviate for vector storage and similarity search, and logistic regression for entity disambiguation.

## Key Features

- Scalable data preprocessing with deduplication
- Vector-based representation of entities
- Intelligent null value imputation
- Feature engineering for entity comparison
- Logistic regression classifier for entity disambiguation
- Comprehensive evaluation and reporting

## Project Structure

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
   │   └── exploration.ipynb   # Results exploration
   └── tests/                  # Testing scripts
       └── test_pipeline.py    # Pipeline verification
```

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Start Weaviate:
   ```
   docker compose up -d
   ```

3. Configure settings in `config.yml`

## Pipeline Execution

The pipeline consists of distinct stages that can be executed end-to-end or individually:

1. **Preprocessing**: Efficiently extract and deduplicate fields from the dataset
   ```
   python main.py --stage preprocessing
   ```

2. **Embedding**: Generate vector embeddings for unique strings
   ```
   python main.py --stage embedding
   ```

3. **Indexing**: Store embeddings in Weaviate for efficient retrieval
   ```
   python main.py --stage indexing
   ```

4. **Imputation**: Fill in missing values using vector similarity
   ```
   python main.py --stage imputation
   ```

5. **Classification**: Train and evaluate the entity resolution model
   ```
   python main.py --stage classification
   ```

6. **Full Pipeline**: Run the entire pipeline
   ```
   python main.py --stage all
   ```

## Development vs. Production Mode

The system supports two operating modes:

- **Development Mode**: Processes a subset of data for rapid iteration
  ```
  python main.py --mode dev
  ```

- **Production Mode**: Processes the complete dataset
  ```
  python main.py --mode prod
  ```

## Checkpointing

The pipeline maintains checkpoints to allow resuming from interruptions:

```
python main.py --stage classification --resume --checkpoint checkpoints/classification_latest.ckpt
```

## Results and Evaluation

After running the pipeline, the system generates evaluation metrics and reports in the `output` directory. The results can be visualized using the provided Jupyter notebooks:

```
jupyter notebook notebooks/evaluation.ipynb
```

## Limitations and Future Work

- The current implementation focuses on personal name entities; extending to other entity types would require additional feature engineering.
- The system prioritizes precision over recall, which may result in some valid matches being missed.
- Future work includes integrating active learning for ambiguous cases and extending the feature set for improved accuracy.
