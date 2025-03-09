# Entity Resolution Test Results Analysis

Generated: 2025-03-09 08:54:26

## Feature Representation Guide

This report includes multiple representations of feature values:

- **Standard Features** (e.g., `person_cosine`): StandardScaler values used for model training
- **Raw Features** (e.g., `person_cosine_raw`): Original values in their native range
- **Normalized Features** (e.g., `person_cosine_norm`): Domain-normalized values in [0,1] range

## Overview

- **Total Test Instances**: 23268
- **Correct Predictions**: 21867 (93.98%)
- **Errors**: 1401 (6.02%)

## Error Analysis

- **False Positives**: 0 (0.00% of errors)
- **False Negatives**: 1401 (100.00% of errors)
- **High Confidence Errors** (> 0.8): 1317 (94.00% of errors)
  - High Confidence False Positives: 0
  - High Confidence False Negatives: 1317

## Confidence Statistics

| True Label | Predicted Label | Count | Mean Confidence | Min Confidence | Max Confidence |
|------------|-----------------|-------|-----------------|---------------|---------------|
| 0 | 0 | 11559 | 0.0174 | 0.0001 | 0.9567 |
| 1 | 0 | 1401 | 0.9022 | 0.2508 | 0.9600 |
| 1 | 1 | 10308 | 0.9954 | 0.9600 | 1.0000 |

## Average Feature Values in Errors vs Correct Predictions

| Feature | Correct Predictions | False Positives | False Negatives |
|---------|---------------------|----------------|----------------|
| person_cosine | -0.0438 | nan | 0.7739 |
| title_cosine | 0.0033 | nan | -0.0502 |
| title_cosine_squared | 0.0056 | nan | -0.0945 |
| provision_cosine | 0.0156 | nan | -0.2424 |
| subjects_cosine | -0.0047 | nan | 0.0148 |
| composite_cosine | -0.0233 | nan | 0.4298 |
| low_composite_penalty | 0.0434 | nan | -0.7938 |
| person_title_harmonic | -0.0243 | nan | 0.4374 |
| person_provision_harmonic | -0.0126 | nan | 0.2296 |
| person_subjects_harmonic | -0.0152 | nan | 0.1881 |

## Sample Error Cases

### False Positives (Predicted Match, Actually Different)

### False Negatives (Predicted Different, Actually Match)

- **Pair**: 1210643#Agent100-13 - 10972343#Agent600-19
  - Confidence: 0.8758
  - Top Features:
    - person_cosine: 0.7366 (raw: 0.7366, norm: 0.8683)
    - title_cosine: -0.5941 (raw: -0.5941, norm: 0.2030)
    - title_cosine_squared: -0.5711 (raw: -0.5711, norm: 0.2145)

- **Pair**: 9146693#Agent700-43 - 16044224#Agent700-48
  - Confidence: 0.8690
  - Top Features:
    - person_cosine: 0.8349 (raw: 0.8349, norm: 0.9175)
    - title_cosine: -0.1838 (raw: -0.1838, norm: 0.4081)
    - title_cosine_squared: -0.2281 (raw: -0.2281, norm: 0.3859)

- **Pair**: 1287926#Agent100-15 - 1200802#Agent600-17
  - Confidence: 0.9346
  - Top Features:
    - person_cosine: 0.6955 (raw: 0.6955, norm: 0.8478)
    - title_cosine: -0.1467 (raw: -0.1467, norm: 0.4266)
    - title_cosine_squared: -0.1960 (raw: -0.1960, norm: 0.4020)

- **Pair**: 12143179#Agent700-37 - 16044024#Agent700-40
  - Confidence: 0.9218
  - Top Features:
    - person_cosine: 0.8349 (raw: 0.8349, norm: 0.9175)
    - title_cosine: -0.0017 (raw: -0.0017, norm: 0.4992)
    - title_cosine_squared: -0.0686 (raw: -0.0686, norm: 0.4657)

- **Pair**: 5444668#Agent700-45 - 9931651#Agent100-13
  - Confidence: 0.9295
  - Top Features:
    - person_cosine: 0.9074 (raw: 0.9074, norm: 0.9537)
    - title_cosine: -0.2849 (raw: -0.2849, norm: 0.3576)
    - title_cosine_squared: -0.3147 (raw: -0.3147, norm: 0.3426)

