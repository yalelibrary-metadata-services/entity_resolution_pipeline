# SetFit Entity Classification CLI

1. This package provides command-line tools for training and using SetFit models for hierarchical entity classification with imbalanced classes.

## Features

- **Hierarchical Classification**: Automatically maps rare classes (< 8 examples) to parent categories
- **GPU Support**: Works with CUDA, MPS (Apple Silicon), and CPU
- **Production Ready**: Includes confidence handling and metadata tracking
- **Flexible Input**: Supports CSV files and individual text classification

## Installation

```bash
pip install -r setfit_requirements.txt
```

## Quick Start

### 1. Training a Model

```bash
python train_setfit_classifier.py \
    --csv_path data/input/training_dataset.csv \
    --ground_truth_path data/output/updated_identity_classification_map_v6_pruned.json \
    --output_dir ./my_setfit_model \
    --min_examples 8 \
    --epochs 3
```

### 2. Making Predictions

**Single text:**

```bash
python predict_setfit_classifier.py \
    --model_dir ./my_setfit_model \
    --text "Contributor: Bach, Johann Sebastian\nTitle: The Well-Tempered Clavier" \
    --verbose
```

**CSV file:**

```bash
python predict_setfit_classifier.py \
    --model_dir ./my_setfit_model \
    --input data/new_entities.csv \
    --text_column composite \
    --output predictions.csv \
    --verbose
```

## Data Format

### Input CSV

Must contain:

- `identity`: Unique identifier for each entity
- `composite`: Text description to classify

### Ground Truth JSON

Format:

```json
{
  "identity_id": {
    "label": ["Primary Label"],
    "path": ["Parent Category > Primary Label"],
    "person": "Person Name",
    "rationale": "Classification reasoning"
  }
}
```

## Training Arguments

| Argument                | Default                      | Description                                |
| ----------------------- | ---------------------------- | ------------------------------------------ |
| `--csv_path`          | Required                     | Path to training CSV file                  |
| `--ground_truth_path` | Required                     | Path to ground truth JSON                  |
| `--output_dir`        | `./setfit_model_output`    | Model save directory                       |
| `--model_name`        | `paraphrase-mpnet-base-v2` | Base model from HuggingFace                |
| `--min_examples`      | `8`                        | Minimum examples for direct classification |
| `--test_size`         | `0.2`                      | Test set proportion                        |
| `--val_size`          | `0.2`                      | Validation set proportion                  |
| `--epochs`            | `3`                        | Training epochs                            |
| `--batch_size`        | `16`                       | Training batch size                        |
| `--seed`              | `42`                       | Random seed                                |
| `--device`            | `auto`                     | Device (cuda/mps/cpu/auto)                 |

## Model Output

The trained model saves:

- `setfit_model/`: The trained SetFit model
- `metadata.pkl`: Binary metadata file
- `metadata.json`: Human-readable metadata
- `evaluation_results.json`: Performance metrics

## Performance Metrics

The system reports dual accuracy:

- **Hierarchical Accuracy**: Performance at the training level (with parent category fallbacks)
- **Original Accuracy**: Performance at the desired granular level
- **Granularity Loss**: The cost of using hierarchical classification

## Hierarchical Classification Strategy

Classes with fewer than `min_examples` training samples are automatically mapped to their parent categories:

```
Original: "Music, Sound, and Sonic Arts" (3 examples)
Mapped to: "Arts, Culture, and Creative Expression" (1200+ examples)
```

This provides:

1. More robust training for underrepresented classes
2. Meaningful predictions at a coarser granularity
3. Graceful handling of extreme class imbalance

## GPU Usage

The script automatically detects and uses available GPUs:

- **NVIDIA GPUs**: Uses CUDA if available
- **Apple Silicon**: Uses MPS backend
- **CPU**: Falls back to CPU if no GPU detected

For servers with GPUs, no additional configuration needed.

## Example Output

```
Total number of classes: 10
Total number of examples: 2537
Classes with < 8 examples: 0 (0.0%)

Hierarchical mapping:
Total classes affected: 0

Training with 10 unique labels
Starting training...

DUAL EVALUATION RESULTS
Hierarchical Accuracy: 0.876
Original Accuracy: 0.876
Granularity Loss: 0.000

Final hierarchical accuracy: 0.876
Final original accuracy: 0.876
```

## Production Considerations

- Use `--verbose` for detailed prediction metadata
- Monitor hierarchical vs original accuracy over time
- Consider retraining when more data becomes available for rare classes
- Set up logging for predictions on rare classes
- Implement confidence thresholds based on your use case

## Troubleshooting

**Memory Issues**: Reduce `--batch_size` if running out of GPU memory

**Poor Performance**:

- Try different base models (e.g., `allenai/scibert_scivocab_uncased` for scientific text)
- Adjust `--min_examples` threshold
- Ensure ground truth labels are properly formatted

**Missing Dependencies**: Install with `pip install -r setfit_requirements.txt`
