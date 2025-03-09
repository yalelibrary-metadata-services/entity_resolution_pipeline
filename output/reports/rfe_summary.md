# Recursive Feature Elimination Results

Generated: 2025-03-08 20:59:14

## Configuration

- RFE Step Size: 1
- Cross-Validation Folds: 5

## Selected Features

1. person_cosine

## Feature Ranking

| Rank | Feature |
|------|--------|
| 1 | person_cosine |
| 2 | composite_cosine |
| 3 | title_cosine |
| 4 | low_composite_penalty |
| 5 | subjects_cosine |
| 6 | title_cosine_squared |
| 7 | provision_cosine |

## Cross-Validation Performance

The following scores show model performance at each step of feature elimination:

- 7 features: 0.9988
- 6 features: 0.9987
- 5 features: 0.9986
- 4 features: 0.9986
- 3 features: 0.9985
- 2 features: 0.9985
- 1 features: 0.9985

## Eliminated Features

1. title_cosine
2. title_cosine_squared
3. provision_cosine
4. subjects_cosine
5. composite_cosine
6. low_composite_penalty
