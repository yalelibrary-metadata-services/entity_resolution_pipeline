# Entity Resolution Classification Metrics

Generated: 2025-03-08 22:20:50

## Feature Representations

This system uses multiple feature representations for clarity:

- **StandardScaler Values**: Used for model training (mean=0, std=1)
- **Domain-Normalized Values**: Intuitive [0,1] range for interpretation
- **Raw Values**: Original values (e.g., [-1,1] for cosine similarity)

## Performance Metrics

| Metric | Value |
|--------|-------|
| Precision | 0.9997 |
| Recall | 0.8486 |
| F1 | 0.9180 |
| Accuracy | 0.9237 |
| Roc_auc | 0.9998 |

## Confusion Matrix

| | Predicted Negative | Predicted Positive |
|---------------------|--------------------|
| **Actual Negative** | 11556 | 3 |
| **Actual Positive** | 1773 | 9936 |

## Derived Metrics

- **Accuracy**: 0.9237
- **Precision**: 0.9997
- **Recall**: 0.8486
- **F1 Score**: 0.9180
