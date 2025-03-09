# Entity Resolution Classification Metrics

Generated: 2025-03-09 10:10:23

## Feature Representations

This system uses multiple feature representations for clarity:

- **StandardScaler Values**: Used for model training (mean=0, std=1)
- **Domain-Normalized Values**: Intuitive [0,1] range for interpretation
- **Raw Values**: Original values (e.g., [-1,1] for cosine similarity)

## Performance Metrics

| Metric | Value |
|--------|-------|
| Precision | 1.0000 |
| Recall | 0.9004 |
| F1 | 0.9476 |
| Accuracy | 0.9499 |
| Roc_auc | 0.9999 |

## Confusion Matrix

| | Predicted Negative | Predicted Positive |
|---------------------|--------------------|
| **Actual Negative** | 11559 | 0 |
| **Actual Positive** | 1166 | 10543 |

## Derived Metrics

- **Accuracy**: 0.9499
- **Precision**: 1.0000
- **Recall**: 0.9004
- **F1 Score**: 0.9476
