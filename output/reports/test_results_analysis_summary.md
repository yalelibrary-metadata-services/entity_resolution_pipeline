# Entity Resolution Test Results Analysis

Generated: 2025-03-09 11:04:48

## Feature Representation Guide

This report includes multiple representations of feature values:

- **Standard Features** (e.g., `person_cosine`): StandardScaler values used for model training
- **Raw Features** (e.g., `person_cosine_raw`): Original values in their native range
- **Normalized Features** (e.g., `person_cosine_norm`): Domain-normalized values in [0,1] range

## Overview

- **Total Test Instances**: 23268
- **Correct Predictions**: 22104 (95.00%)
- **Errors**: 1164 (5.00%)

## Error Analysis

- **False Positives**: 0 (0.00% of errors)
- **False Negatives**: 1164 (100.00% of errors)
- **High Confidence Errors** (> 0.8): 1086 (93.30% of errors)
  - High Confidence False Positives: 0
  - High Confidence False Negatives: 1086

## Confidence Statistics

| True Label | Predicted Label | Count | Mean Confidence | Min Confidence | Max Confidence |
|------------|-----------------|-------|-----------------|---------------|---------------|
| 0 | 0 | 11559 | 0.0197 | 0.0001 | 0.9478 |
| 1 | 0 | 1164 | 0.8972 | 0.2356 | 0.9499 |
| 1 | 1 | 10545 | 0.9931 | 0.9500 | 1.0000 |

## Average Feature Values in Errors vs Correct Predictions

| Feature | Correct Predictions | False Positives | False Negatives |
|---------|---------------------|----------------|----------------|
| birth_death_left | 0.0298 | nan | -0.6301 |
| birth_death_match | 0.0336 | nan | -0.5893 |
| birth_death_right | 0.0339 | nan | -0.7120 |
| composite_cosine | -0.0157 | nan | 0.3802 |
| low_composite_penalty | 0.0339 | nan | -0.7932 |
| person_cosine | -0.0358 | nan | 0.7902 |
| person_provision_harmonic | 0.0179 | nan | -0.2958 |
| person_subjects_harmonic | -0.0119 | nan | 0.1576 |
| person_title_harmonic | -0.0161 | nan | 0.3781 |
| subjects_cosine | -0.0034 | nan | -0.0116 |

## Sample Error Cases

### False Positives (Predicted Match, Actually Different)

### False Negatives (Predicted Different, Actually Match)

- **Pair**: 1210643#Agent100-13 - 10972343#Agent600-19
  - Confidence: 0.8950
  - Top Features:
    - birth_death_left: 0.4817
    - birth_death_match: -0.7856
    - birth_death_right: -2.1898

- **Pair**: 1287926#Agent100-15 - 1200802#Agent600-17
  - Confidence: 0.9421
  - Top Features:
    - birth_death_left: -2.0759
    - birth_death_match: -0.7856
    - birth_death_right: 0.4567

- **Pair**: 12143179#Agent700-37 - 16044024#Agent700-40
  - Confidence: 0.9322
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

- **Pair**: 12596995-bdb04d84-8ffe-4a38-b0d1-5eb6fdd529a2#http://id.loc.gov/rwo/agents/n50043984 - 1878195#Agent100-12
  - Confidence: 0.8813
  - Top Features:
    - birth_death_left: 0.4817
    - birth_death_match: 1.2730
    - birth_death_right: 0.4567

