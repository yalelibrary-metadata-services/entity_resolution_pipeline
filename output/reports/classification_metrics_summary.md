# Entity Resolution Classification Metrics

Generated: 2025-03-08 20:40:28

## Feature Representations

This system uses multiple feature representations for clarity:

- **StandardScaler Values**: Used for model training (mean=0, std=1)
- **Domain-Normalized Values**: Intuitive [0,1] range for interpretation
- **Raw Values**: Original values (e.g., [-1,1] for cosine similarity)

## Performance Metrics

| Metric | Value |
|--------|-------|
| Precision | 1.0000 |
| Recall | 0.8850 |
| F1 | 0.9390 |
| Accuracy | 0.9421 |
| Roc_auc | 0.9999 |

## Confusion Matrix

| | Predicted Negative | Predicted Positive |
|---------------------|--------------------|
| **Actual Negative** | 11559 | 0 |
| **Actual Positive** | 1347 | 10362 |

## Derived Metrics

- **Accuracy**: 0.9421
- **Precision**: 1.0000
- **Recall**: 0.8850
- **F1 Score**: 0.9390
