"""
Composite field templating for entity resolution dataset.
Transforms static composite text into a more natural verbalized format.
"""

import csv
import os
from typing import Dict, Optional


def ensure_period(text: str) -> str:
    """Ensure text ends with a period."""
    if text and not text.rstrip().endswith('.'):
        return text.rstrip() + '.'
    return text


def generate_composite(row: Dict[str, str]) -> str:
    """
    Generate a new composite string from row data using natural language template.
    
    Template format:
    "{person}" had the role of {roles} in relation to the work {title}, 
    which has the following provision information: {provision}. 
    This work is about: {subjects}. Its form or genre is: {genres}.
    
    Null fields are omitted with their contextual strings.
    """
    parts = []
    
    # Always include person, roles, and title (these are always present)
    person = row.get('person', '').strip()
    roles = row.get('roles', '').strip()
    title = row.get('title', '').strip()
    
    # Build the main sentence
    main_sentence = f'"{person}" had the role of "{roles}" in relation to the work "{title}"'
    
    # Handle optional provision information
    provision = row.get('provision', '').strip()
    if provision:
        main_sentence += f', which has the following provision information: "{provision}"'
    
    parts.append(ensure_period(main_sentence))
    
    # Handle optional subjects
    subjects = row.get('subjects', '').strip()
    if subjects:
        parts.append(ensure_period(f'This work is about: "{subjects}"'))
    
    # Handle optional genres
    genres = row.get('genres', '').strip()
    if genres:
        parts.append(ensure_period(f'Its form or genre is: "{genres}"'))
    
    # Handle optional relatedWork
    related_work = row.get('relatedWork', '').strip()
    if related_work:
        parts.append(ensure_period(f'It is further related to the work "{related_work}"'))
    
    return ' '.join(parts)


def process_csv_file(input_path: str, output_path: Optional[str] = None) -> None:
    """
    Process CSV file and update composite field with new template.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to output CSV file (if None, overwrites input file)
    """
    if output_path is None:
        output_path = input_path
    
    # Read all rows first
    rows = []
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        for row in reader:
            # Generate new composite
            row['composite'] = generate_composite(row)
            rows.append(row)
    
    # Write updated rows
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Processed {len(rows)} rows")
    print(f"Updated composite fields saved to: {output_path}")


def main():
    """Main entry point for the templating script."""
    input_file = "data/input/training/training_dataset_classified_2025-07-04.csv"
    
    # Check if file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return
    
    # Process the file (overwrites the original)
    process_csv_file(input_file)
    
    # Show a few examples of the transformation
    print("\nExamples of transformed composite fields:")
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i < 3:  # Show first 3 examples
                print(f"\nExample {i+1}:")
                print(f"Person: {row['person']}")
                print(f"New composite: {row['composite']}")
            else:
                break


if __name__ == "__main__":
    main()