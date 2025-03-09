# Entity Resolution Test Results Analysis

Generated: 2025-03-08 20:01:23

## Feature Representation Guide

This report includes multiple representations of feature values:

- **Standard Features** (e.g., `person_cosine`): StandardScaler values used for model training
- **Raw Features** (e.g., `person_cosine_raw`): Original values in their native range
- **Normalized Features** (e.g., `person_cosine_norm`): Domain-normalized values in [0,1] range

## Overview

- **Total Test Instances**: 23268
- **Correct Predictions**: 21921 (94.21%)
- **Errors**: 1347 (5.79%)

## Error Analysis

- **False Positives**: 0 (0.00% of errors)
- **False Negatives**: 1347 (100.00% of errors)
- **High Confidence Errors** (> 0.8): 1327 (98.52% of errors)
  - High Confidence False Positives: 0
  - High Confidence False Negatives: 1327

## Confidence Statistics

| True Label | Predicted Label | Count | Mean Confidence | Min Confidence | Max Confidence |
|------------|-----------------|-------|-----------------|---------------|---------------|
| 0 | 0 | 11559 | 0.0244 | 0.0001 | 0.9537 |
| 1 | 0 | 1347 | 0.9296 | 0.2487 | 0.9600 |
| 1 | 1 | 10362 | 0.9884 | 0.9600 | 0.9999 |

## Average Feature Values in Errors vs Correct Predictions

| Feature | Correct Predictions | False Positives | False Negatives |
|---------|---------------------|----------------|----------------|
| person_cosine | -0.0456 | nan | 0.8381 |
| title_cosine | 0.0161 | nan | -0.2460 |
| title_cosine_squared | 0.0168 | nan | -0.2667 |
| provision_cosine | 0.0347 | nan | -0.5442 |
| subjects_cosine | -0.0135 | nan | 0.1768 |
| composite_cosine | -0.0108 | nan | 0.2511 |
| low_composite_penalty | 0.0412 | nan | -0.7937 |

## Sample Error Cases

### False Positives (Predicted Match, Actually Different)

### False Negatives (Predicted Different, Actually Match)

- **Pair**: 1210643#Agent100-13 - 10972343#Agent600-19
  - Confidence: 0.9354
  - Top Features:
