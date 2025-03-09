# Entity Resolution Classification Metrics

Generated: 2025-03-09 08:38:48

## Feature Representations

This system uses multiple feature representations for clarity:

- **StandardScaler Values**: Used for model training (mean=0, std=1)
- **Domain-Normalized Values**: Intuitive [0,1] range for interpretation
- **Raw Values**: Original values (e.g., [-1,1] for cosine similarity)

## Performance Metrics

| Metric | Value |
|--------|-------|
| Precision | 0.9970 |
| Recall | 0.9997 |
| F1 | 0.9984 |
| Accuracy | 0.9984 |
| Roc_auc | 0.9999 |

## Confusion Matrix

| | Predicted Negative | Predicted Positive |
|---------------------|--------------------|
| **Actual Negative** | 11524 | 35 |
| **Actual Positive** | 3 | 11706 |

## Derived Metrics

- **Accuracy**: 0.9984
- **Precision**: 0.9970
- **Recall**: 0.9997
- **F1 Score**: 0.9984
