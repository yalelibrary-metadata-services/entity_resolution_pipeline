# Entity Resolution Feature Importance

Generated: 2025-03-09 10:10:23

## Understanding Feature Values

Feature importance is calculated based on the StandardScaler values used for model training.
These features are normalized to have mean=0 and standard deviation=1, which helps the model
give appropriate weight to each feature regardless of its original scale.

## Feature Importance Ranking

| Feature | Weight | Absolute Weight | Importance |
|---------|--------|----------------|------------|
| person_cosine | 1.7961 | 1.7961 | 0.2529 |
| person_title_harmonic | 1.2260 | 1.2260 | 0.1726 |
| composite_cosine | 1.1079 | 1.1079 | 0.1560 |
| birth_death_match | 1.0785 | 1.0785 | 0.1519 |
| low_composite_penalty | -0.6786 | 0.6786 | 0.0956 |
| person_provision_harmonic | 0.4572 | 0.4572 | 0.0644 |
| birth_death_right | -0.2618 | 0.2618 | 0.0369 |
| birth_death_left | -0.2248 | 0.2248 | 0.0317 |
| subjects_cosine | -0.1935 | 0.1935 | 0.0272 |
| person_subjects_harmonic | 0.0768 | 0.0768 | 0.0108 |

## Feature Type Importance

| Feature Type | Importance |
|--------------|------------|
| cosine | 0.4362 |
| harmonic | 0.2478 |
| match | 0.1519 |
| penalty | 0.0956 |
| right | 0.0369 |
| left | 0.0317 |

## Insights

### Top Features
1. **person_cosine** (Importance: 0.2529)
2. **person_title_harmonic** (Importance: 0.1726)
3. **composite_cosine** (Importance: 0.1560)
4. **birth_death_match** (Importance: 0.1519)
5. **low_composite_penalty** (Importance: 0.0956)

### Top Feature Types
1. **cosine** (Importance: 0.4362)
2. **harmonic** (Importance: 0.2478)
3. **match** (Importance: 0.1519)
