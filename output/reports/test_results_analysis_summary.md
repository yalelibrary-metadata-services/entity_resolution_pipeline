# Entity Resolution Test Results Analysis

Generated: 2025-03-09 10:10:24

## Feature Representation Guide

This report includes multiple representations of feature values:

- **Standard Features** (e.g., `person_cosine`): StandardScaler values used for model training
- **Raw Features** (e.g., `person_cosine_raw`): Original values in their native range
- **Normalized Features** (e.g., `person_cosine_norm`): Domain-normalized values in [0,1] range

## Overview

- **Total Test Instances**: 23268
- **Correct Predictions**: 22102 (94.99%)
- **Errors**: 1166 (5.01%)

## Error Analysis

- **False Positives**: 0 (0.00% of errors)
- **False Negatives**: 1166 (100.00% of errors)
- **High Confidence Errors** (> 0.8): 1087 (93.22% of errors)
  - High Confidence False Positives: 0
  - High Confidence False Negatives: 1087

## Confidence Statistics

| True Label | Predicted Label | Count | Mean Confidence | Min Confidence | Max Confidence |
|------------|-----------------|-------|-----------------|---------------|---------------|
| 0 | 0 | 11559 | 0.0197 | 0.0001 | 0.9477 |
| 1 | 0 | 1166 | 0.8971 | 0.2358 | 0.9500 |
| 1 | 1 | 10543 | 0.9931 | 0.9500 | 1.0000 |

## Average Feature Values in Errors vs Correct Predictions

| Feature | Correct Predictions | False Positives | False Negatives |
|---------|---------------------|----------------|----------------|
| birth_death_left | 0.0291 | nan | -0.6304 |
| birth_death_match | 0.0329 | nan | -0.5896 |
| birth_death_right | 0.0333 | nan | -0.7145 |
| composite_cosine | -0.0163 | nan | 0.3807 |
| low_composite_penalty | 0.0341 | nan | -0.7932 |
| person_cosine | -0.0360 | nan | 0.7900 |
| person_provision_harmonic | 0.0179 | nan | -0.2958 |
| person_subjects_harmonic | -0.0124 | nan | 0.1639 |
| person_title_harmonic | -0.0166 | nan | 0.3789 |
| subjects_cosine | -0.0040 | nan | -0.0049 |

## Sample Error Cases

### False Positives (Predicted Match, Actually Different)

### False Negatives (Predicted Different, Actually Match)

- **Pair**: 1210643#Agent100-13 - 10972343#Agent600-19
  - Confidence: 0.8948
  - Top Features:
    - birth_death_left: 0.4817
    - birth_death_match: -0.7856
    - birth_death_right: -2.1898

- **Pair**: 9146693#Agent700-43 - 16044224#Agent700-48
  - Confidence: 0.9051
  - Top Features:
    - birth_death_left: 0.4817
    - birth_death_match: -0.7856
    - birth_death_right: -2.1898

- **Pair**: 1287926#Agent100-15 - 1200802#Agent600-17
  - Confidence: 0.9420
  - Top Features:
    - birth_death_left: -2.0759
    - birth_death_match: -0.7856
    - birth_death_right: 0.4567

- **Pair**: 12143179#Agent700-37 - 16044024#Agent700-40
  - Confidence: 0.9321
  - Top Features:
    - birth_death_left: 0.4817
    - birth_death_match: -0.7856
    - birth_death_right: -2.1898

- **Pair**: 5444668#Agent700-45 - 9931651#Agent100-13
  - Confidence: 0.9366
  - Top Features:
    - birth_death_left: -2.0759
    - birth_death_match: -0.7856
    - birth_death_right: 0.4567

