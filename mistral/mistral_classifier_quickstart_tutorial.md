# Mistral Classifier Factory Quickstart Tutorial: Entity Resolution Taxonomy Classification

## Overview

This tutorial shows how to replace your current SetFit-based taxonomy classification with Mistral's Classifier Factory for your entity resolution pipeline. The approach builds on your existing classified data while simplifying infrastructure and potentially improving classification accuracy.

## Current vs. Mistral Approach

### Your Current SetFit Pipeline
- **Training**: `setfit/train_setfit_simple.py` - Local GPU training with hierarchical classification
- **Prediction**: `setfit/predict_setfit_classifier.py` - Local model serving
- **Feature**: `src/taxonomy_feature.py` - Taxonomy dissimilarity calculation

### Mistral Classifier Factory Benefits
- **Cloud-based training** - No local GPU requirements
- **Production API** - No model serving infrastructure
- **Multi-target classification** - Can classify both parent and child categories
- **Faster iteration** - Minutes vs hours for retraining

## Step 1: Install Dependencies

```bash
pip install mistralai pandas
```

## Step 2: Convert Training Data

Create a script to convert your existing parallel classifications to Mistral's multi-label format:

```python
# scripts/convert_to_mistral_format.py
import json
import pandas as pd
from pathlib import Path

def convert_parallel_classifications_to_mistral(
    entity_csv: str, 
    classifications_json: str, 
    output_jsonl: str
):
    """Convert your parallel classifications to Mistral multi-label format."""
    
    # Load entity data for composite text
    entities_df = pd.read_csv(entity_csv)
    entity_lookup = {}
    for _, row in entities_df.iterrows():
        person_id = str(row['personId'])
        entity_lookup[person_id] = row['composite']
    
    # Load parallel classifications
    with open(classifications_json, 'r') as f:
        classifications = json.load(f)
    
    training_examples = []
    
    for person_id, classification_data in classifications.items():
        # Get composite text for this entity
        composite_text = entity_lookup.get(person_id)
        if not composite_text:
            continue
            
        # Extract labels and paths
        labels_list = classification_data.get('label', [])
        paths_list = classification_data.get('path', [])
        
        if not labels_list:
            continue
        
        # Extract parent categories from paths
        parent_categories = []
        for path in paths_list:
            if " > " in path:
                parent_categories.append(path.split(" > ")[0])
        
        # Create labels for Mistral using native multi-label support
        mistral_labels = {
            "domain": labels_list,  # List of all domains - Mistral's native multi-label format!
            "parent_category": parent_categories  # List of all parent categories
        }
        
        training_examples.append({
            "text": composite_text,
            "labels": mistral_labels
        })
    
    # Write JSONL file
    with open(output_jsonl, 'w') as f:
        for example in training_examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"Converted {len(training_examples)} examples to {output_jsonl}")
    
    # Print statistics about multi-label distribution
    single_count = sum(1 for ex in training_examples if len(ex["labels"]["domain"]) == 1)
    dual_count = sum(1 for ex in training_examples if len(ex["labels"]["domain"]) == 2)
    triple_count = sum(1 for ex in training_examples if len(ex["labels"]["domain"]) == 3)
    
    print(f"Label distribution:")
    print(f"  Single domain: {single_count}")
    print(f"  Dual domain: {dual_count}")
    print(f"  Triple domain: {triple_count}")
    
    return len(training_examples)

# Usage
if __name__ == "__main__":
    convert_parallel_classifications_to_mistral(
        "data/input/training_dataset_classified_2025-06-21.csv",
        "data/input/parallel_classifications.json",
        "data/output/mistral_training_data.jsonl"
    )
```

## Step 3: Create Training and Validation Split

```python
# scripts/split_mistral_data.py
import json
import random

def split_jsonl_data(input_file: str, train_file: str, val_file: str, val_ratio: float = 0.2):
    """Split JSONL data into training and validation sets."""
    
    with open(input_file, 'r') as f:
        data = [json.loads(line) for line in f]
    
    random.shuffle(data)
    split_idx = int(len(data) * (1 - val_ratio))
    
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    with open(train_file, 'w') as f:
        for example in train_data:
            f.write(json.dumps(example) + '\n')
    
    with open(val_file, 'w') as f:
        for example in val_data:
            f.write(json.dumps(example) + '\n')
    
    print(f"Split {len(data)} examples into {len(train_data)} train, {len(val_data)} validation")

# Usage
split_jsonl_data(
    "data/output/mistral_training_data.jsonl",
    "data/output/mistral_train.jsonl", 
    "data/output/mistral_val.jsonl"
)
```

## Step 4: Train Mistral Classifier

```python
# scripts/train_mistral_classifier.py
import os
from mistralai import Mistral

# Set your API key
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise ValueError("Please set MISTRAL_API_KEY environment variable")

client = Mistral(api_key=MISTRAL_API_KEY)

def upload_training_file(file_path: str) -> str:
    """Upload training data to Mistral."""
    
    with open(file_path, "rb") as f:
        uploaded_file = client.files.upload(file=f)
    
    print(f"Uploaded {file_path}, file ID: {uploaded_file.id}")
    return uploaded_file.id

def create_classifier_job(train_file_id: str, val_file_id: str) -> str:
    """Create and start a classifier training job."""
    
    job = client.fine_tuning.jobs.create(
        model="ministral-3b-latest",
        training_files=[{"file_id": train_file_id, "weight": 1}],
        validation_files=[val_file_id],
        hyperparameters={
            "training_steps": 100,
            "learning_rate": 0.0001,
        },
        auto_start=True,
    )
    
    print(f"Created job: {job.id}")
    print(f"Status: {job.status}")
    return job.id

def monitor_job(job_id: str):
    """Monitor training job progress."""
    import time
    
    while True:
        job = client.fine_tuning.jobs.get(job_id)
        print(f"Job {job_id} status: {job.status}")
        
        if job.status == "SUCCESS":
            print(f"Training completed! Model: {job.fine_tuned_model}")
            return job.fine_tuned_model
        elif job.status == "FAILED":
            print(f"Training failed: {job.message}")
            return None
        elif job.status in ["RUNNING", "QUEUED", "VALIDATING"]:
            print("Training in progress...")
            time.sleep(60)  # Check every minute
        else:
            print(f"Unknown status: {job.status}")
            time.sleep(60)

# Main training workflow
if __name__ == "__main__":
    # Upload training and validation data
    train_file_id = upload_training_file("data/output/mistral_train.jsonl")
    val_file_id = upload_training_file("data/output/mistral_val.jsonl")
    
    # Create training job
    job_id = create_classifier_job(train_file_id, val_file_id)
    
    # Monitor progress
    model_id = monitor_job(job_id)
    
    if model_id:
        print(f"🎉 Training completed! Model ID: {model_id}")
        # Save model ID for later use
        with open("data/output/mistral_model_id.txt", "w") as f:
            f.write(model_id)
```

## Step 5: Update Taxonomy Feature Calculator

Replace your existing taxonomy feature with Mistral-based classification:

```python
# src/mistral_taxonomy_feature.py
import os
import json
import logging
from typing import Dict, Optional, Tuple
from functools import lru_cache
import hashlib
from mistralai import Mistral

logger = logging.getLogger(__name__)

class MistralTaxonomyDissimilarity:
    """
    Calculate dissimilarity using Mistral Classifier Factory predictions.
    Uses your actual taxonomy structure from parallel_classifications.json
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Initialize Mistral client
        api_key = config.get('mistral_api_key') or os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("Mistral API key required")
        
        self.client = Mistral(api_key=api_key)
        
        # Load model ID
        model_id_path = config.get('mistral_model_id_path', 'data/output/mistral_model_id.txt')
        with open(model_id_path, 'r') as f:
            self.model_id = f.read().strip()
        
        # Load entity data for composite text lookup
        self.entity_data = self._load_entity_data(config.get('entity_data_path'))
        
        # Cache for classifications
        self.classification_cache = {}
        
        logger.info(f"Initialized Mistral taxonomy with model {self.model_id}")
    
    def _load_entity_data(self, path: str) -> Dict[str, str]:
        """Load entity data to get composite text for personIds."""
        import pandas as pd
        
        try:
            df = pd.read_csv(path)
            entity_data = {}
            for _, row in df.iterrows():
                person_id = str(row['personId'])
                composite = row['composite']
                entity_data[person_id] = composite
            
            logger.info(f"Loaded composite text for {len(entity_data)} entities")
            return entity_data
        except Exception as e:
            logger.error(f"Error loading entity data: {e}")
            return {}
    
    @lru_cache(maxsize=1000)
    def get_classification(self, text_hash: str, text: str) -> Dict:
        """Get classification with caching."""
        
        # Check persistent cache first
        if text_hash in self.classification_cache:
            return self.classification_cache[text_hash]
        
        try:
            # Get classification from Mistral
            response = self.client.classifiers.classify(
                model=self.model_id,
                inputs=[text]
            )
            
            result = response.results[0]
            classification = {}
            
            # Extract predictions for each label
            for label_name, prediction in result.predictions.items():
                if hasattr(prediction, 'value'):
                    classification[label_name] = prediction.value
                else:
                    classification[label_name] = str(prediction)
            
            # Cache result
            self.classification_cache[text_hash] = classification
            
            return classification
            
        except Exception as e:
            logger.error(f"Error getting Mistral classification: {e}")
            return {}
    
    def calculate_dissimilarity(self, class1: Dict, class2: Dict) -> float:
        """
        Calculate dissimilarity between two multi-label classifications.
        Uses Mistral's native multi-label prediction format.
        """
        
        if not class1 or not class2:
            return 0.5  # Neutral when missing data
        
        # Extract domain lists (Mistral returns lists for multi-label)
        domains1 = class1.get('domain', [])
        domains2 = class2.get('domain', [])
        
        # Extract parent category lists
        parents1 = class1.get('parent_category', [])
        parents2 = class2.get('parent_category', [])
        
        # Convert to sets for easier comparison
        domains1_set = set(domains1) if isinstance(domains1, list) else {domains1}
        domains2_set = set(domains2) if isinstance(domains2, list) else {domains2}
        parents1_set = set(parents1) if isinstance(parents1, list) else {parents1}
        parents2_set = set(parents2) if isinstance(parents2, list) else {parents2}
        
        # Calculate domain overlap
        domain_overlap = len(domains1_set & domains2_set)
        total_domains = len(domains1_set | domains2_set)
        
        # Calculate parent category overlap
        parent_overlap = len(parents1_set & parents2_set)
        total_parents = len(parents1_set | parents2_set)
        
        if total_domains == 0:
            return 0.5  # No domains to compare
        
        # Calculate dissimilarity based on overlaps
        # High overlap = low dissimilarity
        domain_dissim = 1.0 - (domain_overlap / total_domains)
        
        # Weight domain dissimilarity more heavily
        if total_parents > 0:
            parent_dissim = 1.0 - (parent_overlap / total_parents)
            # Weighted combination: 70% domain, 30% parent category
            final_dissim = 0.7 * domain_dissim + 0.3 * parent_dissim
        else:
            final_dissim = domain_dissim
        
        return final_dissim
    
    def get_feature_value(self, left_id: str, right_id: str) -> float:
        """Calculate taxonomy dissimilarity between two entities."""
        
        # Get composite text for both entities
        left_text = self.entity_data.get(left_id)
        right_text = self.entity_data.get(right_id)
        
        if not left_text or not right_text:
            logger.debug(f"Missing composite text for {left_id} or {right_id}")
            return 0.5
        
        # Get classifications
        left_hash = hashlib.md5(left_text.encode()).hexdigest()
        right_hash = hashlib.md5(right_text.encode()).hexdigest()
        
        left_class = self.get_classification(left_hash, left_text)
        right_class = self.get_classification(right_hash, right_text)
        
        # Calculate dissimilarity
        return self.calculate_dissimilarity(left_class, right_class)
```

## Step 6: Update Configuration

Update your configuration to use Mistral instead of SetFit:

```python
# config/mistral_config.py
import os

MISTRAL_CONFIG = {
    'mistral_api_key': os.environ.get('MISTRAL_API_KEY'),
    'mistral_model_id_path': 'data/output/mistral_model_id.txt',
    'entity_data_path': 'data/input/training_dataset_classified_2025-06-21.csv',
}

# Update your main config to use Mistral taxonomy
FEATURES_CONFIG = {
    'taxonomy_dissimilarity': {
        'enabled': True,
        'class': 'MistralTaxonomyDissimilarity',
        'config': MISTRAL_CONFIG,
        'weight': 1.0
    }
    # ... other features remain the same
}
```

## Step 7: Batch Prediction Script

Create a script to make predictions on your entity resolution dataset:

```python
# scripts/predict_mistral_batch.py
import pandas as pd
from src.mistral_taxonomy_feature import MistralTaxonomyDissimilarity

def predict_taxonomy_batch(input_csv: str, output_csv: str, config: dict):
    """Make batch predictions using Mistral classifier."""
    
    # Load data
    df = pd.read_csv(input_csv)
    
    # Initialize classifier
    classifier = MistralTaxonomyDissimilarity(config)
    
    # Make predictions
    predictions = []
    for _, row in df.iterrows():
        person_id = str(row['personId'])
        composite = row['composite']
        
        # Get classification
        import hashlib
        text_hash = hashlib.md5(composite.encode()).hexdigest()
        classification = classifier.get_classification(text_hash, composite)
        
        # Handle multi-label predictions (lists)
        domains = classification.get('domain', [])
        parent_categories = classification.get('parent_category', [])
        
        prediction = {
            'personId': person_id,
            'mistral_domains': '|'.join(domains) if isinstance(domains, list) else domains,
            'mistral_parent_categories': '|'.join(parent_categories) if isinstance(parent_categories, list) else parent_categories,
            'mistral_num_domains': len(domains) if isinstance(domains, list) else 1,
            'mistral_num_parents': len(parent_categories) if isinstance(parent_categories, list) else 1,
            'mistral_primary_domain': domains[0] if domains else '',
            'mistral_spans_multiple_parents': len(set(parent_categories)) > 1 if isinstance(parent_categories, list) else False
        }
        predictions.append(prediction)
    
    # Save results
    pred_df = pd.DataFrame(predictions)
    result_df = df.merge(pred_df, on='personId', how='left')
    result_df.to_csv(output_csv, index=False)
    
    print(f"Saved predictions to {output_csv}")

# Usage
if __name__ == "__main__":
    from config.mistral_config import MISTRAL_CONFIG
    
    predict_taxonomy_batch(
        "data/input/training_dataset_classified_2025-06-21.csv",
        "data/output/mistral_predictions.csv",
        MISTRAL_CONFIG
    )
```

## Step 8: Integration Testing

Test the integration with your existing entity resolution pipeline:

```python
# scripts/test_mistral_integration.py
import pandas as pd
from src.mistral_taxonomy_feature import MistralTaxonomyDissimilarity
from config.mistral_config import MISTRAL_CONFIG

def test_taxonomy_feature():
    """Test the Mistral taxonomy feature on sample entity pairs."""
    
    # Load test data
    df = pd.read_csv("data/input/training_dataset_classified_2025-06-21.csv")
    
    # Initialize feature calculator
    taxonomy_calc = MistralTaxonomyDissimilarity(MISTRAL_CONFIG)
    
    # Test on sample pairs
    test_pairs = [
        (df.iloc[0]['personId'], df.iloc[1]['personId']),
        (df.iloc[0]['personId'], df.iloc[10]['personId']),
        (df.iloc[5]['personId'], df.iloc[15]['personId']),
    ]
    
    print("Testing Mistral taxonomy dissimilarity:")
    for left_id, right_id in test_pairs:
        left_id, right_id = str(left_id), str(right_id)
        dissim = taxonomy_calc.get_feature_value(left_id, right_id)
        print(f"  {left_id} vs {right_id}: {dissim:.3f}")

if __name__ == "__main__":
    test_taxonomy_feature()
```

## Usage Summary

1. **Setup**: Install dependencies and set `MISTRAL_API_KEY`
2. **Convert Data**: Run conversion script on your parallel_classifications.json
3. **Train Model**: Upload data and train classifier
4. **Update Pipeline**: Replace SetFit with Mistral taxonomy feature
5. **Test**: Verify integration with existing pipeline

## Your Taxonomy Structure

Based on your `parallel_classifications.json`, the classifier will predict:
- **domain**: List of specific categories (e.g., `["Music, Sound, and Sonic Arts", "Philosophy and Ethics"]`)
- **parent_category**: List of top-level categories (e.g., `["Arts, Culture, and Creative Expression", "Humanities, Thought, and Interpretation"]`)

### Multi-Label Distribution in Your Data:
- **Single domain**: Most entities (your current SetFit approach)
- **Dual domain**: 181 entities with 2 classifications
- **Triple domain**: 9 entities with 3 classifications (like PersonID `3643200#Agent100-13` with Philosophy, Literature, and Music)

### Example Multi-Label Predictions:
```json
{
  "domain": ["Music, Sound, and Sonic Arts"],
  "parent_category": ["Arts, Culture, and Creative Expression"]
}
```

```json
{
  "domain": ["Philosophy and Ethics", "Music, Sound, and Sonic Arts"],
  "parent_category": ["Humanities, Thought, and Interpretation", "Arts, Culture, and Creative Expression"]
}
```

## Key Advantages

- **Reduced Infrastructure**: No local GPU training or model serving
- **Enhanced Multi-label Support**: Better handling of entities with multiple domain classifications
- **Faster Iteration**: Retrain in minutes when adding new classifications
- **Production Ready**: Managed API with automatic scaling
- **Cost Effective**: Pay-per-use pricing vs. infrastructure costs

This approach maintains full compatibility with your existing entity resolution pipeline while upgrading the taxonomy classification component to use Mistral's cloud-based service.