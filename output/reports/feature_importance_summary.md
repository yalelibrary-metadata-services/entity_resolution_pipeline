# Entity Resolution Feature Importance

Generated: 2025-03-09 08:38:48

## Understanding Feature Values

Feature importance is calculated based on the StandardScaler values used for model training.
These features are normalized to have mean=0 and standard deviation=1, which helps the model
give appropriate weight to each feature regardless of its original scale.

## Feature Importance Ranking

| Feature | Weight | Absolute Weight | Importance |
|---------|--------|----------------|------------|
| person_cosine | 1.2747 | 1.2747 | 0.1437 |
| person_levenshtein | 0.9708 | 0.9708 | 0.1095 |
| person_title_harmonic | 0.8580 | 0.8580 | 0.0967 |
| composite_cosine | 0.8073 | 0.8073 | 0.0910 |
| low_composite_penalty | -0.6601 | 0.6601 | 0.0744 |
| person_levenshtein_birth_death_match_product | 0.5195 | 0.5195 | 0.0586 |
| person_cosine_birth_death_match_product | 0.5076 | 0.5076 | 0.0572 |
| person_provision_harmonic | 0.4718 | 0.4718 | 0.0532 |
| birth_death_match | 0.3987 | 0.3987 | 0.0450 |
| person_subjects_product | 0.3773 | 0.3773 | 0.0425 |
| birth_death_right | -0.3611 | 0.3611 | 0.0407 |
| birth_death_left | -0.3401 | 0.3401 | 0.0383 |
| title_subjects_ratio | -0.2160 | 0.2160 | 0.0244 |
| title_cosine | 0.1777 | 0.1777 | 0.0200 |
| provision_subjects_harmonic | -0.1772 | 0.1772 | 0.0200 |
| title_cosine_squared | 0.1522 | 0.1522 | 0.0172 |
| person_subjects_harmonic | 0.1350 | 0.1350 | 0.0152 |
| title_provision_harmonic | -0.1228 | 0.1228 | 0.0138 |
| title_subjects_harmonic | -0.1015 | 0.1015 | 0.0114 |
| provision_cosine | -0.0753 | 0.0753 | 0.0085 |
| subjects_cosine | -0.0719 | 0.0719 | 0.0081 |
| composite_subjects_ratio | -0.0593 | 0.0593 | 0.0067 |
| title_subjects_product | -0.0332 | 0.0332 | 0.0037 |

## Feature Type Importance

| Feature Type | Importance |
|--------------|------------|
| cosine | 0.2714 |
| harmonic | 0.2104 |
| product | 0.1621 |
| levenshtein | 0.1095 |
| penalty | 0.0744 |
| match | 0.0450 |
| right | 0.0407 |
| left | 0.0383 |
| ratio | 0.0310 |
| squared | 0.0172 |

## Insights

### Top Features
1. **person_cosine** (Importance: 0.1437)
2. **person_levenshtein** (Importance: 0.1095)
3. **person_title_harmonic** (Importance: 0.0967)
4. **composite_cosine** (Importance: 0.0910)
5. **low_composite_penalty** (Importance: 0.0744)

### Top Feature Types
1. **cosine** (Importance: 0.2714)
2. **harmonic** (Importance: 0.2104)
3. **product** (Importance: 0.1621)
