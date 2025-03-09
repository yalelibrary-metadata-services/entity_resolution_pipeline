# Entity Resolution Feature Importance

Generated: 2025-03-08 22:20:50

## Understanding Feature Values

Feature importance is calculated based on the StandardScaler values used for model training.
These features are normalized to have mean=0 and standard deviation=1, which helps the model
give appropriate weight to each feature regardless of its original scale.

## Feature Importance Ranking

| Feature | Weight | Absolute Weight | Importance |
|---------|--------|----------------|------------|
| person_levenshtein | 2.7353 | 2.7353 | 0.3947 |
| composite_cosine | 1.8542 | 1.8542 | 0.2675 |
| low_composite_penalty | -1.0173 | 1.0173 | 0.1468 |
| title_cosine | 0.2816 | 0.2816 | 0.0406 |
| title_cosine_squared | 0.2493 | 0.2493 | 0.0360 |
| title_subjects_ratio | -0.2378 | 0.2378 | 0.0343 |
| composite_subjects_ratio | 0.1583 | 0.1583 | 0.0228 |
| provision_subjects_harmonic | -0.1174 | 0.1174 | 0.0169 |
| provision_cosine | 0.1065 | 0.1065 | 0.0154 |
| title_provision_harmonic | -0.0642 | 0.0642 | 0.0093 |
| title_subjects_product | 0.0468 | 0.0468 | 0.0068 |
| title_subjects_harmonic | -0.0444 | 0.0444 | 0.0064 |
| subjects_cosine | 0.0174 | 0.0174 | 0.0025 |

## Feature Type Importance

| Feature Type | Importance |
|--------------|------------|
| levenshtein | 0.3947 |
| cosine | 0.3261 |
| penalty | 0.1468 |
| ratio | 0.0572 |
| squared | 0.0360 |
| harmonic | 0.0326 |
| product | 0.0068 |

## Insights

### Top Features
1. **person_levenshtein** (Importance: 0.3947)
2. **composite_cosine** (Importance: 0.2675)
3. **low_composite_penalty** (Importance: 0.1468)
4. **title_cosine** (Importance: 0.0406)
5. **title_cosine_squared** (Importance: 0.0360)

### Top Feature Types
1. **levenshtein** (Importance: 0.3947)
2. **cosine** (Importance: 0.3261)
3. **penalty** (Importance: 0.1468)
