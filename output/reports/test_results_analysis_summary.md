# Entity Resolution Test Results Analysis

Generated: 2025-03-08 22:20:51

## Feature Representation Guide

This report includes multiple representations of feature values:

- **Standard Features** (e.g., `person_cosine`): StandardScaler values used for model training
- **Raw Features** (e.g., `person_cosine_raw`): Original values in their native range
- **Normalized Features** (e.g., `person_cosine_norm`): Domain-normalized values in [0,1] range

## Overview

- **Total Test Instances**: 23268
- **Correct Predictions**: 21492 (92.37%)
- **Errors**: 1776 (7.63%)

## Error Analysis

- **False Positives**: 3 (0.17% of errors)
- **False Negatives**: 1773 (99.83% of errors)
- **High Confidence Errors** (> 0.8): 1147 (64.58% of errors)
  - High Confidence False Positives: 3
  - High Confidence False Negatives: 1144

## Confidence Statistics

| True Label | Predicted Label | Count | Mean Confidence | Min Confidence | Max Confidence |
|------------|-----------------|-------|-----------------|---------------|---------------|
| 0 | 0 | 11556 | 0.0279 | 0.0001 | 0.9579 |
| 0 | 1 | 3 | 0.9703 | 0.9676 | 0.9724 |
| 1 | 0 | 1773 | 0.8260 | 0.0279 | 0.9598 |
| 1 | 1 | 9936 | 0.9948 | 0.9602 | 1.0000 |

## Average Feature Values in Errors vs Correct Predictions

| Feature | Correct Predictions | False Positives | False Negatives |
|---------|---------------------|----------------|----------------|
| title_cosine | -0.0072 | -0.5803 | 0.0818 |
| title_cosine_squared | -0.0034 | -0.5593 | 0.0292 |
| provision_cosine | 0.0102 | -0.5341 | -0.1154 |
| subjects_cosine | 0.0024 | 0.3629 | -0.0636 |
| composite_cosine | -0.0340 | 0.0200 | 0.4614 |
| low_composite_penalty | 0.0583 | -0.7967 | -0.7944 |
| person_levenshtein | 0.0005 | 1.1175 | 0.0369 |
| title_subjects_harmonic | -0.0001 | 0.3131 | -0.0453 |
| title_provision_harmonic | 0.0011 | -0.4620 | -0.0100 |
| provision_subjects_harmonic | 0.0029 | 0.3035 | -0.0720 |

## Sample Error Cases

### False Positives (Predicted Match, Actually Different)

- **Pair**: 16044224#Agent700-39 - 1605973#Agent700-17
  - Confidence: 0.9724
  - Top Features:
    - title_cosine: -0.5088 (raw: -0.5088, norm: 0.2456)
    - title_cosine_squared: -0.5016 (raw: -0.5016, norm: 0.2492)
    - provision_cosine: -0.6659 (raw: -0.6659, norm: 0.1671)

- **Pair**: 16044224#Agent700-39 - 1484712#Agent700-20
  - Confidence: 0.9710
  - Top Features:
    - title_cosine: -0.5088 (raw: -0.5088, norm: 0.2456)
    - title_cosine_squared: -0.5016 (raw: -0.5016, norm: 0.2492)
    - provision_cosine: -0.6659 (raw: -0.6659, norm: 0.1671)

- **Pair**: 16044091#Agent700-32 - 1605973#Agent700-17
  - Confidence: 0.9676
  - Top Features:
    - title_cosine: -0.7234 (raw: -0.7234, norm: 0.1383)
    - title_cosine_squared: -0.6745 (raw: -0.6745, norm: 0.1627)
    - provision_cosine: -0.2707 (raw: -0.2707, norm: 0.3647)

### False Negatives (Predicted Different, Actually Match)

- **Pair**: 1210643#Agent100-13 - 10972343#Agent600-19
  - Confidence: 0.7411
  - Top Features:
    - title_cosine: -0.5941 (raw: -0.5941, norm: 0.2030)
    - title_cosine_squared: -0.5711 (raw: -0.5711, norm: 0.2145)
    - provision_cosine: -0.4507 (raw: -0.4507, norm: 0.2746)

- **Pair**: 12181005#Agent700-31 - 16044363#Agent700-32
  - Confidence: 0.9123
  - Top Features:
    - title_cosine: 0.6660 (raw: 0.6660, norm: 0.8330)
    - title_cosine_squared: 0.5542 (raw: 0.5542, norm: 0.7771)
    - provision_cosine: 0.1249 (raw: 0.1249, norm: 0.5624)

- **Pair**: 15694420#Agent600-31 - 11017966#Agent700-18
  - Confidence: 0.8625
  - Top Features:
    - title_cosine: 0.0569 (raw: 0.0569, norm: 0.5284)
    - title_cosine_squared: -0.0164 (raw: -0.0164, norm: 0.4918)
    - provision_cosine: -0.5238 (raw: -0.5238, norm: 0.2381)

- **Pair**: 1287926#Agent100-15 - 1200802#Agent600-17
  - Confidence: 0.8613
  - Top Features:
    - title_cosine: -0.1467 (raw: -0.1467, norm: 0.4266)
    - title_cosine_squared: -0.1960 (raw: -0.1960, norm: 0.4020)
    - provision_cosine: 1.1831 (raw: 1.1831, norm: 1.0000)

- **Pair**: 12143179#Agent700-37 - 16044024#Agent700-40
  - Confidence: 0.7746
  - Top Features:
    - title_cosine: -0.0017 (raw: -0.0017, norm: 0.4992)
    - title_cosine_squared: -0.0686 (raw: -0.0686, norm: 0.4657)
    - provision_cosine: -0.2444 (raw: -0.2444, norm: 0.3778)

