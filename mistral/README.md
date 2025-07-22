# Mistral Classifier Integration

This directory contains the Mistral.AI classifier integration for the entity resolution pipeline. It provides a drop-in replacement for the SetFit classifier with support for Mistral's fine-tuned domain classification models.

## Overview

The Mistral classifier uses a fine-tuned `ministral-3b-latest` model to classify library catalog records into domain categories. It returns confidence scores for multiple taxonomic levels:
- **Domain**: Specific subject areas (e.g., "Visual Arts and Design", "Natural Sciences")
- **Parent Category**: High-level categories (e.g., "Arts, Culture, and Creative Expression")

## Setup

1. **API Key**: Set your Mistral API key as an environment variable:
   ```bash
   export MISTRAL_API_KEY='your-api-key-here'
   ```

2. **Dependencies**: Install the Mistral client:
   ```bash
   pip install mistralai pandas
   ```

3. **Configuration**: The classifier can be configured in `config.yml`:
   ```yaml
   mistral_classifier:
     enabled: true  # Enable Mistral instead of SetFit
     model_id: "ft:classifier:ministral-3b-latest:2bec22ef:20250702:a2707cf5"
     batch_size: 50
     rate_limit_delay: 0.1
   ```

## Usage

### Command Line Interface

The CLI provides the same interface as the SetFit classifier for easy integration:

#### Single Text Classification
```bash
# Basic classification
python mistral/predict_mistral_classifier.py --text "Title: Organic chemistry\\nSubjects: Chemistry"

# Verbose output with confidence scores
python mistral/predict_mistral_classifier.py --text "Title: Organic chemistry\\nSubjects: Chemistry" --verbose
```

#### CSV Batch Processing
```bash
# Process a CSV file (default: reads 'composite' column)
python mistral/predict_mistral_classifier.py \
  --input data/input/training_dataset.csv \
  --output data/output/training_dataset_mistral.csv

# Custom text column
python mistral/predict_mistral_classifier.py \
  --input data.csv \
  --text_column "description" \
  --output classified.csv
```

#### Custom Model ID
```bash
# Use a different fine-tuned model
python mistral/predict_mistral_classifier.py \
  --model_id "ft:classifier:your-model-id" \
  --text "Your text here"
```

### Output Format

The classifier adds the following columns to CSV files:
- `mistral_prediction`: The highest-scoring domain category
- `mistral_confidence`: Confidence score (0-1) for the prediction
- `mistral_parent_category`: The parent category in the taxonomy
- `is_parent_category`: Boolean indicating if the prediction is a high-level category

### Response Format

The Mistral API returns scores for all possible classes:

```json
{
  "results": [{
    "domain": {
      "scores": {
        "Visual Arts and Design": 0.9502,
        "Literature and Narrative Arts": 0.0016,
        "Natural Sciences": 0.0002,
        ...
      }
    },
    "parent_category": {
      "scores": {
        "Arts, Culture, and Creative Expression": 0.9986,
        "Sciences, Research, and Discovery": 0.0000,
        ...
      }
    }
  }]
}
```

## Integration with Entity Resolution Pipeline

To use Mistral instead of SetFit in the pipeline:

1. Set `mistral_classifier.enabled: true` in `config.yml`
2. The pipeline will automatically use the `mistral_prediction` column instead of `setfit_prediction`
3. The `taxonomy_dissimilarity` feature will calculate based on Mistral's predictions

## Testing

Run the test scripts to verify functionality:

```bash
# Direct API test
python mistral/test_direct.py

# CLI test
python mistral/test_mistral_cli.py
```

## Performance Considerations

- **Rate Limiting**: The classifier includes a 0.1s delay between API calls by default
- **Batch Size**: Process up to 50 texts per batch (configurable)
- **Caching**: Consider implementing result caching for repeated classifications
- **Cost**: Each API call counts against your Mistral API usage quota

## Comparison with SetFit

| Feature | SetFit | Mistral |
|---------|--------|---------|
| Model Type | Local, sentence transformers | API-based, fine-tuned LLM |
| Speed | Fast (local inference) | Slower (API calls) |
| Cost | Free after training | Per-API-call pricing |
| Accuracy | Good for small taxonomies | Excellent for complex taxonomies |
| Multi-label | Supported | Supported (confidence scores) |
| Deployment | Requires model files | API key only |

## Troubleshooting

1. **API Key Error**: Ensure `MISTRAL_API_KEY` is set in your environment
2. **Rate Limit Errors**: Increase `rate_limit_delay` in the config
3. **Model Not Found**: Verify the model ID is correct and accessible
4. **Empty Predictions**: Check that the input text contains meaningful content

## Future Enhancements

- [ ] Implement retry logic for API failures
- [ ] Add result caching to reduce API calls
- [ ] Support for batch API endpoints
- [ ] Confidence threshold filtering
- [ ] Multi-label output format matching SetFit exactly