# Entity Resolution Feature Importance

Generated: 2025-03-08 20:40:28

## Understanding Feature Values

Feature importance is calculated based on the StandardScaler values used for model training.
These features are normalized to have mean=0 and standard deviation=1, which helps the model
give appropriate weight to each feature regardless of its original scale.

## Feature Importance Ranking

| Feature | Weight | Absolute Weight | Importance |
|---------|--------|----------------|------------|
| person_cosine | 2.9024 | 2.9024 | 0.4890 |
| composite_cosine | 1.5707 | 1.5707 | 0.2646 |
| low_composite_penalty | -0.6653 | 0.6653 | 0.1121 |
| title_cosine | 0.2139 | 0.2139 | 0.0360 |
| title_cosine_squared | 0.2111 | 0.2111 | 0.0356 |
| subjects_cosine | -0.1911 | 0.1911 | 0.0322 |
| provision_cosine | 0.1808 | 0.1808 | 0.0305 |

## Feature Type Importance

| Feature Type | Importance |
|--------------|------------|
| cosine | 0.8523 |
| penalty | 0.1121 |
| squared | 0.0356 |

## Insights

### Top Features
1. **person_cosine** (Importance: 0.4890)
2. **composite_cosine** (Importance: 0.2646)
3. **low_composite_penalty** (Importance: 0.1121)
4. **title_cosine** (Importance: 0.0360)
5. **title_cosine_squared** (Importance: 0.0356)

### Top Feature Types
1. **cosine** (Importance: 0.8523)
2. **penalty** (Importance: 0.1121)
3. **squared** (Importance: 0.0356)
