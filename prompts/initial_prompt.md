Develop a robust **entity resolution system** for the **Yale University Library catalog** using **vector embeddings** and **machine learning classification**.

## Approach
1. Generate **1,536-dimensional vector embeddings** for entity representation.
2. Use **Weaviate** for **approximate nearest neighbor (ANN) clustering** and imputation of null values.
3. Construct **feature vectors** for record pairs within each neighborhood.
4. Train a **logistic regression classifier** with **gradient descent**.
5. Evaluate performance using **precision and recall metrics**.

## Dataset and Resources
- **Complete dataset**: 600+ CSV files requiring indexing (~50 million unique strings; ~15GB of data)
- **Labeled training/testing dataset**: 2,000+ records with ground truth matches
- **Development environment**: Local machine (8 cores, 32GB RAM)
- **Production environment**: 64 cores, 256GB RAM

## Key Requirements

### A. **Modular Pipeline Architecture**
- Design a **modular system** with **distinct components**.
- Define **clear interfaces** between modules.
- Support **independent execution** of pipeline stages.
- Implement **configuration management** using YAML.
- Implement **robust checkpointing** to manage processing interruptions.
- Support training and testing in the development environment as well as the **production environment**.

### B. **Efficient String Deduplication & ID Mapping**
- Deduplicate **before embedding** to eliminate redundancy.
- Maintain a **mapping between unique strings and embeddings**.
- Reuse existing embeddings for duplicate strings.
- Track **frequency statistics** of duplicate strings.

### C. **Vector-Based Null Value Imputation**
- For missing values (e.g., `subjects`), perform a **Weaviate vector search** using the `composite` field vector.
- Ensure scalability from **development to production**.

### D. **Feature Engineering & Classification**
1. **Embedding Generation & Indexing**
   - Use **OpenAI text-embedding-3-small**.
   - Index embeddings in **Weaviate (HNSW algorithm)**.
   - Maintain **pointers between `personId` and embeddings**.

2. **Classification & Evaluation**
   - Train **logistic regression classifier**.
   - Evaluate using **precision, recall, and F1-score**.
   - Based on a user-defined parameters (e.g., confidence range, person cosine similarity), support logging of ambiguous cases for review.

### E. **Processing Workflow**
1. **Deduplicate and embed** all unique text strings by field.
2. **Index embeddings** in Weaviate and track references.
3. **Use `person` embeddings** as blocking keys for ANN retrieval and construction of candidate pairs for matching and classification.
4. **Compare record pairs in batch/parallel** using feature vectors.
5. **Impute missing values** using vector averaging.
6. **Train and test classifier** on labeled data.
7. **Apply graph-based clustering with transitivity** to refine entity groups.
8. **Serialize identity clusters** as **JSON-Lines output**.
9. **Evaluate and report results**, logging anomalies.
10. Pipeline modules should cover the following stages: 
   - `preprocessing`
   - `embedding`
   - `indexing`
   - `imputation and training`
   - `classification`
   - `clustering`
11. Design separate modules for:
   - `feature engineering`
   - `querying`
   - `testing and diagnostics`
   - `utility functions`
12. Implement `reporting and analysis` modules corresponding to each stage of the pipeline.
13. Design each module intelligently to be easily extensible and customizable.
14. Include architectural diagrams and design outlines.
15. **Each module** should have its own README.md file to explain and document its interface and usage.

## Testing and Diagnostics
- Include diagnostic scripts to test and verify each stage of the pipeline.
- Complete unit tests are not necessary at this point, but it must be possible to add them later.
- The user must be able to verify that each stage is working as intended. 

## Configuation
- Manage configuration globally using a YAML file.
- Centralize key settings related to feature engineering (e.g., interaction features, recursive feature elimination) and vector indexing and **make them centrally tunable**.

## Additional Requirements
- **Leverage Levenshtein similarity** between `person` names as a feature.
- Use **optimized batch queries and operations** throughout the pipeline for better throughput.
- Implement proper indexing for all fields to ensure fast retrieval.
- Create a sophisticated indexing mechanism that enables efficient lookups.
- **Provide deployment artifacts**:
  - Project-level `README.md` explaining the use case, pipeline, and step-by-step order of execution
  - `requirements.txt` listing dependencies
  - `docker-compose.yml` for Weaviate deployment
  - YAML config file

Summarize the requirements and detail your proposed solution before proceeding. **Explain your work step by step.**

Implement the project as here deployable software artifacts with the following structure. Do not omit any components.

```
   entity-resolution/
   ├── README.md               # Project documentation
   ├── config.yml              # Configuration parameters
   ├── docker-compose.yml      # Docker Compose for Weaviate
   ├── prometheus.yml          # Prometheus monitoring configuration
   ├── requirements.txt        # Python dependencies
   ├── main.py                 # Entry point script
   ├── setup.sh                # Setup script
   ├── src/                    # Source code
   │   ├── batch_preprocessing.py    # Data preprocessing
   │   ├── embedding.py        # Vector embedding
   │   ├── indexing.py         # Weaviate integration
   │   ├── imputation.py       # Null value imputation
   │   ├── batch_querying.py         # Querying and match candidate retrieval
   │   ├── parallel_features.py         # Feature engineering
   │   ├── classification.py   # Classifier training/evaluation
   │   ├── clustering.py       # Entity clustering
   │   ├── pipeline.py         # Pipeline orchestration
   │   ├── analysis.py         # Analysis of pipeline processes and results
   │   ├── reporting.py        # Reporting and visualization of pipeline results
   │   └── utils.py            # Utility functions
   ├── notebooks/              # Analysis notebooks
   │   └── evaluation.ipynb    # Results evaluation
   │   └── exploration.ipynb   # Results exploration
   └── tests/                  # Testing scripts
      └── test_pipeline.py     # Pipeline verification
```