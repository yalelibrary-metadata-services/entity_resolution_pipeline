# Entity Resolution Test Results Analysis

Generated: 2025-03-09 08:38:50

## Feature Representation Guide

This report includes multiple representations of feature values:

- **Standard Features** (e.g., `person_cosine`): StandardScaler values used for model training
- **Raw Features** (e.g., `person_cosine_raw`): Original values in their native range
- **Normalized Features** (e.g., `person_cosine_norm`): Domain-normalized values in [0,1] range

## Overview

- **Total Test Instances**: 23268
- **Correct Predictions**: 23230 (99.84%)
- **Errors**: 38 (0.16%)

## Error Analysis

- **False Positives**: 35 (92.11% of errors)
- **False Negatives**: 3 (7.89% of errors)
- **High Confidence Errors** (> 0.8): 14 (36.84% of errors)
  - High Confidence False Positives: 14
  - High Confidence False Negatives: 0

## Confidence Statistics

| True Label | Predicted Label | Count | Mean Confidence | Min Confidence | Max Confidence |
|------------|-----------------|-------|-----------------|---------------|---------------|
| 0 | 0 | 11524 | 0.0116 | 0.0001 | 0.4828 |
| 0 | 1 | 35 | 0.8020 | 0.5530 | 0.9855 |
| 1 | 0 | 3 | 0.3594 | 0.2595 | 0.4395 |
| 1 | 1 | 11706 | 0.9847 | 0.5062 | 1.0000 |

## Average Feature Values in Errors vs Correct Predictions

| Feature | Correct Predictions | False Positives | False Negatives |
|---------|---------------------|----------------|----------------|
| person_cosine | 0.0044 | 0.6651 | 0.7198 |
| title_cosine | 0.0014 | -0.5545 | -0.5995 |
| title_cosine_squared | 0.0007 | -0.4865 | -0.5752 |
| provision_cosine | 0.0013 | -0.2062 | -2.0283 |
| subjects_cosine | -0.0035 | 0.4414 | 0.1341 |
| composite_cosine | 0.0042 | 0.0854 | -0.6801 |
| low_composite_penalty | -0.0066 | -0.7967 | 0.5712 |
| person_levenshtein | 0.0034 | 0.1118 | -0.1012 |
| person_title_harmonic | 0.0036 | 0.0123 | 0.0904 |
| person_provision_harmonic | 0.0022 | 0.2325 | -1.4778 |

## Sample Error Cases

### False Positives (Predicted Match, Actually Different)

- **Pair**: 16044091#Agent700-32 - 14704123#Agent700-23
  - Confidence: 0.9766
  - Top Features:
    - person_cosine: 1.0281 (raw: 1.0281, norm: 1.0000)
    - title_cosine: -0.8737 (raw: -0.8737, norm: 0.0632)
    - title_cosine_squared: -0.7918 (raw: -0.7918, norm: 0.1041)

- **Pair**: 16044091#Agent700-32 - 53144#Agent700-22
  - Confidence: 0.9754
  - Top Features:
    - person_cosine: 1.0281 (raw: 1.0281, norm: 1.0000)
    - title_cosine: -0.6371 (raw: -0.6371, norm: 0.1815)
    - title_cosine_squared: -0.6057 (raw: -0.6057, norm: 0.1971)

- **Pair**: 16044091#Agent700-32 - 16044349#Agent700-32
  - Confidence: 0.5678
  - Top Features:
    - person_cosine: -0.3771 (raw: -0.3771, norm: 0.3114)
    - title_cosine: 0.9843 (raw: 0.9843, norm: 0.9922)
    - title_cosine_squared: 0.8721 (raw: 0.8721, norm: 0.9361)

- **Pair**: 16044224#Agent700-39 - 1605973#Agent700-17
  - Confidence: 0.9855
  - Top Features:
    - person_cosine: 1.0281 (raw: 1.0281, norm: 1.0000)
    - title_cosine: -0.5088 (raw: -0.5088, norm: 0.2456)
    - title_cosine_squared: -0.5016 (raw: -0.5016, norm: 0.2492)

- **Pair**: 1119397#Agent100-16 - 14703468#Agent700-23
  - Confidence: 0.7663
  - Top Features:
    - person_cosine: 0.8032 (raw: 0.8032, norm: 0.9016)
    - title_cosine: -1.0525 (raw: -1.0525, norm: 0.0000)
    - title_cosine_squared: -0.9275 (raw: -0.9275, norm: 0.0362)

### False Negatives (Predicted Different, Actually Match)

- **Pair**: 6593516#Agent700-48 - 16043945#Agent700-50
  - Confidence: 0.3793
  - Top Features:
    - person_cosine: 0.8032 (raw: 0.8032, norm: 0.9016)
    - title_cosine: -0.5247 (raw: -0.5247, norm: 0.2377)
    - title_cosine_squared: -0.5146 (raw: -0.5146, norm: 0.2427)

- **Pair**: 8165559#Agent700-27 - 11025388#Agent600-21
  - Confidence: 0.2595
  - Top Features:
    - person_cosine: 0.6214 (raw: 0.6214, norm: 0.8107)
    - title_cosine: -0.6785 (raw: -0.6785, norm: 0.1607)
    - title_cosine_squared: -0.6389 (raw: -0.6389, norm: 0.1806)

- **Pair**: 1931035#Agent100-11 - 10810385#Agent700-168
  - Confidence: 0.4395
  - Top Features:
    - person_cosine: 0.7349 (raw: 0.7349, norm: 0.8674)
    - title_cosine: -0.5952 (raw: -0.5952, norm: 0.2024)
    - title_cosine_squared: -0.5720 (raw: -0.5720, norm: 0.2140)

