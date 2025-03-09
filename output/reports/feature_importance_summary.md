# Entity Resolution Feature Importance

Generated: 2025-03-09 11:04:48

## Understanding Feature Values

Feature importance is calculated based on the StandardScaler values used for model training.
These features are normalized to have mean=0 and standard deviation=1, which helps the model
give appropriate weight to each feature regardless of its original scale.

## Feature Importance Ranking

| Feature | Weight | Absolute Weight | Importance |
|---------|--------|----------------|------------|
| person_cosine | 1.7956 | 1.7956 | 0.2528 |
| person_title_harmonic | 1.2257 | 1.2257 | 0.1726 |
| composite_cosine | 1.1079 | 1.1079 | 0.1560 |
| birth_death_match | 1.0780 | 1.0780 | 0.1518 |
| low_composite_penalty | -0.6789 | 0.6789 | 0.0956 |
| person_provision_harmonic | 0.4579 | 0.4579 | 0.0645 |
| birth_death_right | -0.2627 | 0.2627 | 0.0370 |
| birth_death_left | -0.2249 | 0.2249 | 0.0317 |
| subjects_cosine | -0.1935 | 0.1935 | 0.0272 |
| person_subjects_harmonic | 0.0767 | 0.0767 | 0.0108 |

## Feature Type Importance

| Feature Type | Importance |
|--------------|------------|
| cosine | 0.4361 |
| harmonic | 0.2479 |
| match | 0.1518 |
| penalty | 0.0956 |
| right | 0.0370 |
| left | 0.0317 |

## Insights

### Top Features
1. **person_cosine** (Importance: 0.2528)
2. **person_title_harmonic** (Importance: 0.1726)
3. **composite_cosine** (Importance: 0.1560)
4. **birth_death_match** (Importance: 0.1518)
5. **low_composite_penalty** (Importance: 0.0956)

### Top Feature Types
1. **cosine** (Importance: 0.4361)
2. **harmonic** (Importance: 0.2479)
3. **match** (Importance: 0.1518)
