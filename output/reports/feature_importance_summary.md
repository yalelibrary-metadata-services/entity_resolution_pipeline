# Entity Resolution Feature Importance

Generated: 2025-03-09 08:54:25

## Understanding Feature Values

Feature importance is calculated based on the StandardScaler values used for model training.
These features are normalized to have mean=0 and standard deviation=1, which helps the model
give appropriate weight to each feature regardless of its original scale.

## Feature Importance Ranking

| Feature | Weight | Absolute Weight | Importance |
|---------|--------|----------------|------------|
| person_cosine | 1.5797 | 1.5797 | 0.1761 |
| person_title_harmonic | 1.0198 | 1.0198 | 0.1137 |
| composite_cosine | 0.9418 | 0.9418 | 0.1050 |
| person_cosine_birth_death_match_product | 0.8343 | 0.8343 | 0.0930 |
| birth_death_match | 0.7203 | 0.7203 | 0.0803 |
| low_composite_penalty | -0.6533 | 0.6533 | 0.0728 |
| person_provision_harmonic | 0.5755 | 0.5755 | 0.0641 |
| person_subjects_product | 0.4772 | 0.4772 | 0.0532 |
| birth_death_right | -0.3334 | 0.3334 | 0.0372 |
| birth_death_left | -0.2956 | 0.2956 | 0.0329 |
| title_subjects_ratio | -0.2683 | 0.2683 | 0.0299 |
| provision_subjects_harmonic | -0.2115 | 0.2115 | 0.0236 |
| title_provision_harmonic | -0.1936 | 0.1936 | 0.0216 |
| person_subjects_harmonic | 0.1740 | 0.1740 | 0.0194 |
| title_cosine | 0.1431 | 0.1431 | 0.0160 |
| title_subjects_harmonic | -0.1319 | 0.1319 | 0.0147 |
| title_cosine_squared | 0.1268 | 0.1268 | 0.0141 |
| subjects_cosine | -0.0861 | 0.0861 | 0.0096 |
| provision_cosine | -0.0773 | 0.0773 | 0.0086 |
| composite_subjects_ratio | -0.0711 | 0.0711 | 0.0079 |
| title_subjects_product | -0.0573 | 0.0573 | 0.0064 |

## Feature Type Importance

| Feature Type | Importance |
|--------------|------------|
| cosine | 0.3152 |
| harmonic | 0.2571 |
| product | 0.1526 |
| match | 0.0803 |
| penalty | 0.0728 |
| ratio | 0.0378 |
| right | 0.0372 |
| left | 0.0329 |
| squared | 0.0141 |

## Insights

### Top Features
1. **person_cosine** (Importance: 0.1761)
2. **person_title_harmonic** (Importance: 0.1137)
3. **composite_cosine** (Importance: 0.1050)
4. **person_cosine_birth_death_match_product** (Importance: 0.0930)
5. **birth_death_match** (Importance: 0.0803)

### Top Feature Types
1. **cosine** (Importance: 0.3152)
2. **harmonic** (Importance: 0.2571)
3. **product** (Importance: 0.1526)
