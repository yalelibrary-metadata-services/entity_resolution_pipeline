"""
Preprocessing module for entity resolution pipeline.

This module handles data loading, cleaning, and deduplication using batch and
parallel processing for efficient handling of large datasets.
"""

import os
import logging
import json
import pandas as pd
import numpy as np
import hashlib
from pathlib import Path
from tqdm import tqdm
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import csv

from src.utils import (
    Timer, get_memory_usage, save_checkpoint, load_checkpoint,
    compute_string_hash
)
from src.mmap_dict import MMapDict  # Import from the dedicated module

logger = logging.getLogger(__name__)

class Preprocessor:
    """
    Handles efficient preprocessing of large datasets for entity resolution.
    
    Features:
    - Batch processing of CSV files
    - Parallel processing of batches
    - Deduplication of strings
    - Tracking of field usage statistics
    - Memory-efficient data structures
    """
    
    def __init__(self, config):
        """
        Initialize the preprocessor with configuration parameters.
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = config
        
        # Data paths
        self.input_dir = Path(config['data']['input_dir'])
        self.output_dir = Path(config['system']['output_dir'])
        self.temp_dir = Path(config['system']['temp_dir'])
        self.checkpoint_dir = Path(config['system']['checkpoint_dir'])
        
        # Create directories if they don't exist
        for dir_path in [self.output_dir, self.temp_dir, self.checkpoint_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Processing parameters
        self.batch_size = config['system']['batch_size']
        self.csv_chunk_size = config['preprocessing']['csv_chunk_size']
        self.max_workers = config['system']['max_workers']
        self.normalize_strings = config['preprocessing']['normalize_strings']
        self.null_values = config['preprocessing']['null_values']
        
        # Fields to process
        self.fields_to_embed = config['embedding']['fields_to_embed']
        
        # Data structures
        self.unique_strings = {}  # hash -> string
        self.string_counts = {}   # hash -> count
        self.record_field_hashes = {}  # record_id -> field -> hash
        self.field_hash_mapping = {}   # hash -> field -> count
        
        logger.info("Preprocessor initialized")
    
    def execute(self, checkpoint=None):
        """
        Execute preprocessing of input data.
        
        Args:
            checkpoint (str, optional): Path to checkpoint file. Defaults to None.
            
        Returns:
            dict: Preprocessing results
        """
        # Load checkpoint if provided
        if checkpoint and os.path.exists(checkpoint):
            state = load_checkpoint(checkpoint)
            self.unique_strings = state.get('unique_strings', {})
            self.string_counts = state.get('string_counts', {})
            self.record_field_hashes = state.get('record_field_hashes', {})
            self.field_hash_mapping = state.get('field_hash_mapping', {})
            processed_files = set(state.get('processed_files', []))
            logger.info(f"Resumed preprocessing from checkpoint: {checkpoint}")
            logger.info(f"Loaded {len(self.unique_strings)} unique strings")
            logger.info(f"Loaded {len(self.record_field_hashes)} records")
            logger.info(f"Loaded {len(processed_files)} processed files")
        else:
            processed_files = set()
        
        # Find CSV files in input directory
        csv_files = sorted(self.input_dir.glob('*.csv'))
        if self.config['system']['mode'] == 'dev':
            # In dev mode, limit the number of files to process
            dev_sample_size = min(
                10,  # Process at most 10 files in dev mode
                len(csv_files)
            )
            csv_files = csv_files[:dev_sample_size]
            logger.info(f"Dev mode: limited to {len(csv_files)} files")
        
        # Filter out already processed files
        files_to_process = [file for file in csv_files if file.name not in processed_files]
        
        if not files_to_process:
            logger.info("No new files to process")
            # Save results for already processed files
            self._save_results()
            
            return {
                'files_processed': len(processed_files),
                'unique_strings': len(self.unique_strings),
                'records_processed': len(self.record_field_hashes),
                'fields_processed': sum(len(fields) for fields in self.record_field_hashes.values())
            }
        
        logger.info(f"Processing {len(files_to_process)} files")
        
        # Process files in parallel
        total_records = 0
        
        with Timer() as timer:
            # Process each file
            for file_path in tqdm(files_to_process, desc="Processing files"):
                records_in_file = self._process_file(file_path)
                total_records += records_in_file
                
                # Add to processed files
                processed_files.add(file_path.name)
                
                # Save checkpoint periodically
                if len(processed_files) % 5 == 0:
                    checkpoint_path = self.checkpoint_dir / f"preprocessing_{len(processed_files)}.ckpt"
                    save_checkpoint({
                        'unique_strings': self.unique_strings,
                        'string_counts': self.string_counts,
                        'record_field_hashes': self.record_field_hashes,
                        'field_hash_mapping': self.field_hash_mapping,
                        'processed_files': list(processed_files)
                    }, checkpoint_path)
                    
                    # Log progress
                    logger.info(
                        f"Processed {len(processed_files)}/{len(csv_files)} files, "
                        f"{total_records} records, {len(self.unique_strings)} unique strings"
                    )
                    logger.info(f"Memory usage: {get_memory_usage():.2f} GB")
        
        # Save final results
        self._save_results()
        
        # Save final checkpoint
        checkpoint_path = self.checkpoint_dir / "preprocessing_final.ckpt"
        save_checkpoint({
            'unique_strings': self.unique_strings,
            'string_counts': self.string_counts,
            'record_field_hashes': self.record_field_hashes,
            'field_hash_mapping': self.field_hash_mapping,
            'processed_files': list(processed_files)
        }, checkpoint_path)
        
        results = {
            'files_processed': len(processed_files),
            'unique_strings': len(self.unique_strings),
            'records_processed': len(self.record_field_hashes),
            'fields_processed': sum(len(fields) for fields in self.record_field_hashes.values()),
            'duration': timer.duration
        }
        
        logger.info(
            f"Preprocessing completed: {results['files_processed']} files, "
            f"{results['records_processed']} records, "
            f"{results['unique_strings']} unique strings, "
            f"{results['fields_processed']} fields, "
            f"{timer.duration:.2f} seconds"
        )
        
        return results
    
    # The rest of the module remains the same...
    # (This includes methods like _process_file, _process_chunk, etc.)
    
    def _process_file(self, file_path):
        """
        Process a single CSV file.
        
        Args:
            file_path (Path): Path to CSV file
            
        Returns:
            int: Number of records processed
        """
        records_processed = 0
        
        # Use pandas to read CSV in chunks
        for chunk in pd.read_csv(
            file_path,
            chunksize=self.csv_chunk_size,
            low_memory=False,
            dtype=str,  # Read all columns as strings
            na_values=self.null_values,
            keep_default_na=True
        ):
            # Process chunk in parallel
            chunk_records = self._process_chunk(chunk)
            records_processed += chunk_records
        
        return records_processed
    
    def _process_chunk(self, chunk):
        """
        Process a chunk of data in parallel.
        
        Args:
            chunk (pd.DataFrame): DataFrame chunk
            
        Returns:
            int: Number of records processed
        """
        # Fill NaN values with "NULL" string
        chunk = chunk.fillna("NULL")
        
        # Process each record in the chunk in parallel
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Prepare arguments for parallel processing
            args_list = [
                (row.to_dict(), i)
                for i, (_, row) in enumerate(chunk.iterrows())
            ]
            
            # Submit jobs
            futures = {
                executor.submit(self._process_record, args[0], args[1]): args
                for args in args_list
            }
            
            # Process results as they complete
            for future in as_completed(futures):
                try:
                    record_id, field_hashes = future.result()
                    
                    # Store record field hashes
                    self.record_field_hashes[record_id] = field_hashes
                except Exception as e:
                    logger.error(f"Error processing record: {e}")
        
        return len(chunk)
    
    def _process_record(self, record, idx):
        """
        Process a single record.
        
        Args:
            record (dict): Record data
            idx (int): Record index
            
        Returns:
            tuple: (record_id, field_hashes)
        """
        # Extract record ID
        record_id = record.get('personId')
        if not record_id:
            # Generate a unique ID if not present
            record_id = f"record_{idx}"
        
        # Initialize field hashes
        field_hashes = {}
        
        # Process composite field if present
        composite_value = record.get('composite')
        if composite_value and composite_value not in self.null_values:
            composite_hash = self._process_field_value('composite', composite_value)
            field_hashes['composite'] = composite_hash
        
        # Process other fields
        for field in ['person', 'title', 'provision', 'subjects']:
            field_value = record.get(field)
            if field_value and field_value not in self.null_values:
                field_hash = self._process_field_value(field, field_value)
                field_hashes[field] = field_hash
            else:
                field_hashes[field] = "NULL"
        
        # Add 'roles' field if present (not embedded but still tracked)
        roles_value = record.get('roles')
        if roles_value and roles_value not in self.null_values:
            roles_hash = compute_string_hash(roles_value)
            field_hashes['roles'] = roles_hash
            
            # Update unique strings and counts for roles
            if roles_hash not in self.unique_strings:
                self.unique_strings[roles_hash] = roles_value
                self.string_counts[roles_hash] = 0
            
            self.string_counts[roles_hash] += 1
        
        return record_id, field_hashes
    
    def _process_field_value(self, field, value):
        """
        Process a field value.
        
        Args:
            field (str): Field name
            value (str): Field value
            
        Returns:
            str: Hash of field value
        """
        # Normalize string if enabled
        if self.normalize_strings:
            value = self._normalize_string(value)
        
        # Compute hash
        hash_value = compute_string_hash(value)
        
        # Update unique strings
        if hash_value not in self.unique_strings:
            self.unique_strings[hash_value] = value
            self.string_counts[hash_value] = 0
        
        # Update string counts
        self.string_counts[hash_value] += 1
        
        # Update field hash mapping
        if hash_value not in self.field_hash_mapping:
            self.field_hash_mapping[hash_value] = {}
        
        if field not in self.field_hash_mapping[hash_value]:
            self.field_hash_mapping[hash_value][field] = 0
        
        self.field_hash_mapping[hash_value][field] += 1
        
        return hash_value
    
    def _normalize_string(self, text):
        """
        Normalize string value.
        
        Args:
            text (str): Input string
            
        Returns:
            str: Normalized string
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _save_results(self):
        """
        Save preprocessing results to output directory.
        """
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # For large datasets, use memory-mapped files
        if self.config['system']['mode'] == 'prod' or len(self.unique_strings) > 100000:
            # Create memory-mapped files
            unique_strings_mmap = MMapDict(self.temp_dir / "unique_strings.mmap")
            for hash_value, string_value in self.unique_strings.items():
                unique_strings_mmap[hash_value] = string_value
            
            field_hash_mmap = MMapDict(self.temp_dir / "field_hash_mapping.mmap")
            for hash_value, field_counts in self.field_hash_mapping.items():
                field_hash_mmap[hash_value] = field_counts
            
            record_field_mmap = MMapDict(self.temp_dir / "record_field_hashes.mmap")
            for record_id, field_hashes in self.record_field_hashes.items():
                record_field_mmap[record_id] = field_hashes
            
            # Save indexes
            with open(self.output_dir / "unique_strings_index.json", 'w') as f:
                json.dump({
                    'location': str(self.temp_dir / "unique_strings.mmap"),
                    'count': len(unique_strings_mmap)
                }, f, indent=2)
            
            with open(self.output_dir / "field_hash_index.json", 'w') as f:
                json.dump({
                    'location': str(self.temp_dir / "field_hash_mapping.mmap"),
                    'count': len(field_hash_mmap)
                }, f, indent=2)
            
            with open(self.output_dir / "record_index.json", 'w') as f:
                json.dump({
                    'location': str(self.temp_dir / "record_field_hashes.mmap"),
                    'count': len(record_field_mmap)
                }, f, indent=2)
            
            # Make sure to flush all data
            unique_strings_mmap.flush()
            field_hash_mmap.flush()
            record_field_mmap.flush()
            
            # Save string counts
            with open(self.output_dir / "string_counts.json", 'w') as f:
                json.dump(self.string_counts, f, indent=2)
            
            # Save samples for inspection
            sample_unique_strings = dict(list(self.unique_strings.items())[:1000])
            sample_record_field_hashes = dict(list(self.record_field_hashes.items())[:1000])
            sample_field_hash_mapping = dict(list(self.field_hash_mapping.items())[:1000])
            
            with open(self.output_dir / "unique_strings_sample.json", 'w') as f:
                json.dump(sample_unique_strings, f, indent=2)
            
            with open(self.output_dir / "record_field_hashes_sample.json", 'w') as f:
                json.dump(sample_record_field_hashes, f, indent=2)
            
            with open(self.output_dir / "field_hash_mapping_sample.json", 'w') as f:
                json.dump(sample_field_hash_mapping, f, indent=2)
        else:
            # For small datasets, save as JSON
            with open(self.output_dir / "unique_strings.json", 'w') as f:
                json.dump(self.unique_strings, f, indent=2)
            
            with open(self.output_dir / "string_counts.json", 'w') as f:
                json.dump(self.string_counts, f, indent=2)
            
            with open(self.output_dir / "record_field_hashes.json", 'w') as f:
                json.dump(self.record_field_hashes, f, indent=2)
            
            with open(self.output_dir / "field_hash_mapping.json", 'w') as f:
                json.dump(self.field_hash_mapping, f, indent=2)
        
        # Save field statistics
        field_stats = {}
        for field in ['composite', 'person', 'title', 'provision', 'subjects', 'roles']:
            field_stats[field] = {
                'count': sum(1 for record in self.record_field_hashes.values() if field in record),
                'null_count': sum(1 for record in self.record_field_hashes.values() 
                                if field in record and record[field] == "NULL"),
                'unique_count': len(set(record[field] for record in self.record_field_hashes.values() 
                                    if field in record and record[field] != "NULL"))
            }
        
        with open(self.output_dir / "field_statistics.json", 'w') as f:
            json.dump(field_stats, f, indent=2)
        
        logger.info(f"Preprocessing results saved to {self.output_dir}")
