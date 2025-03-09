import pandas as pd
import numpy as np

# Create sample data similar to your original values
sample_data = pd.DataFrame({
    'person_cosine': [-0.762477547, -0.503683474, 0.5112942],
    'title_cosine': [-0.503683474, 0.2, -0.1],
    'provision_cosine': [0.5112942, 0.7, -0.3],
    'pair_id': ['1', '2', '3']  # Non-feature column
})

print("Original sample data:")
print(sample_data)

# Apply normalization logic from reporting.py
def normalize_features(df):
    # Copy to avoid modifying original
    normalized_df = df.copy()
    
    # Identify feature columns
    metadata_cols = ['pair_id']
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    
    print(f"Feature columns: {feature_cols}")
    
    # Process each feature column
    for col in feature_cols:
        # Check if column contains negative values
        if df[col].min() < 0:
            print(f"\nProcessing column: {col}")
            print(f"Before: min={df[col].min()}, max={df[col].max()}")
            
            if df[col].min() >= -1 and df[col].max() <= 1:
                # If values are in [-1, 1] range (like correlations), rescale to [0, 1]
                print(f"Normalizing {col} from [-1, 1] to [0, 1] range")
                old_values = df[col].tolist()
                normalized_df[col] = (df[col] + 1) / 2
                new_values = normalized_df[col].tolist()
                print(f"Sample normalization for {col}:")
                for old, new in zip(old_values, new_values):
                    print(f"  {old} → {new}")
            else:
                # For other ranges, use min-max scaling
                print(f"Applying min-max scaling to {col}")
                min_val = df[col].min()
                max_val = df[col].max()
                range_val = max_val - min_val
                
                # Check for division by zero
                if abs(range_val) < 1e-10:
                    print(f"WARNING: Range is zero for {col}, skipping normalization")
                    continue
                    
                old_values = df[col].tolist()
                normalized_df[col] = (df[col] - min_val) / range_val
                new_values = normalized_df[col].tolist()
                print(f"Sample min-max scaling for {col}:")
                for old, new in zip(old_values, new_values):
                    print(f"  {old} → {new}")
            
            print(f"After: min={normalized_df[col].min()}, max={normalized_df[col].max()}")
    
    return normalized_df

# Apply normalization
normalized_sample = normalize_features(sample_data)

print("\nNormalized sample data:")
print(normalized_sample)