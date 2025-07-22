"""
Preprocessing Module for Entity Resolution

This module handles data preprocessing tasks for entity resolution, including:
- Loading and parsing input data files
- Deduplicating field values
- Creating hash-based data structures for efficient storage
- Maintaining mappings between unique strings and entity identifiers
"""

import os
import csv
import logging
import pickle
import hashlib
import time
import sqlite3
import tempfile
from typing import Dict, List, Tuple, Any, Set, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from tqdm import tqdm
import pandas as pd
import numpy as np
from collections import defaultdict
# zlib no longer needed after switching from CRC32 to xxHash

logger = logging.getLogger(__name__)

def hash_string(string_value: str) -> str:
    """
    Create a deterministic hash for a string value using xxHash3-128 (fastest with lowest collision risk).
    
    Args:
        string_value: String value to hash
        
    Returns:
        xxHash3-128 hash of the string as hex
    """
    if not string_value or string_value == 'nan':
        return "NULL"
    
    # Use xxHash3-128 for optimal speed and collision resistance
    # 128-bit output provides virtually zero collision probability
    # Significantly faster than SHA-256 while maintaining excellent collision resistance
    try:
        import xxhash
        return xxhash.xxh3_128(string_value.encode('utf-8')).hexdigest()
    except ImportError:
        # Fallback to SHA-256 if xxhash not available
        import hashlib
        return hashlib.sha256(string_value.encode('utf-8')).hexdigest()

def process_file(file_path: str) -> Dict[str, Any]:
    """
    Process a single input file and build data structures.
    
    Args:
        file_path: Path to the input CSV file
        
    Returns:
        Dictionary with processing metrics and data structures
    """
    start_time = time.time()
    
    # Initialize local data structures
    local_output_data = {}
    local_string_counts = {}
    local_field_hash_mapping = {}
    
    try:
        # Load the CSV file
        df = pd.read_csv(file_path, na_values='', keep_default_na=False)
        
        # Make sure all expected fields are present
        required_fields = ['composite', 'marcKey', 'person', 'roles', 'title', 'provision', 'subjects', 'genres', 'attribution', 'relatedWork', 'personId']
        missing_fields = [field for field in required_fields if field not in df.columns]
        
        if missing_fields:
            logger.warning(f"File {file_path} is missing fields: {missing_fields}")
            # Fill missing fields with empty strings
            for field in missing_fields:
                df[field] = ''
        
        # Process each row
        total_rows = 0
        for _, row in df.iterrows():
            total_rows += 1
            
            # Get personId
            person_id = str(row['personId'])
            if not person_id:
                continue
                
            # Initialize field hashes for this record
            local_output_data[person_id] = {}
            
            # Process each field
            for field in required_fields:
                if field == 'personId':
                    continue
                    
                # Get field value
                value = str(row[field]) if pd.notna(row[field]) else ''
                
                # Hash the value
                hash_val = hash_string(value)
                
                # Update field hashes
                local_output_data[person_id][field] = hash_val
                
                # Update string counts
                if hash_val != "NULL":
                    local_string_counts[hash_val] = local_string_counts.get(hash_val, 0) + 1
                    
                    # Update field/hash mapping
                    if hash_val not in local_field_hash_mapping:
                        local_field_hash_mapping[hash_val] = {}
                    local_field_hash_mapping[hash_val][field] = local_field_hash_mapping[hash_val].get(field, 0) + 1
        
        elapsed_time = time.time() - start_time
        logger.debug(f"Processed {file_path} with {total_rows} rows in {elapsed_time:.2f} seconds")
        
        return {
            'file_path': file_path,
            'rows_processed': total_rows,
            'elapsed_time': elapsed_time,
            'output_data': local_output_data,
            'string_counts': local_string_counts,
            'field_hash_mapping': local_field_hash_mapping
        }
        
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        return {
            'file_path': file_path,
            'rows_processed': 0,
            'error': str(e),
            'output_data': {},
            'string_counts': {},
            'field_hash_mapping': {}
        }

def create_string_dict(string_counts: Dict[str, int], field_hash_mapping: Dict[str, Dict[str, int]], input_dir: str) -> Dict[str, str]:
    """
    Create dictionary of hash values to original string values.
    
    Args:
        string_counts: Dictionary tracking string occurrence counts
        field_hash_mapping: Dictionary mapping field/hash to usage counts
        input_dir: Directory containing input files
        
    Returns:
        Dictionary mapping hash to original string value
    """
    logger.info("Building string dictionary from hash values")
    string_dict = {}
    
    # Initialize hash-to-string mapping with reverse lookup tables
    hash_to_string_map = {}
    
    # Get list of input files
    csv_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.csv')]
    logger.info(f"Scanning {len(csv_files)} CSV files to extract original string values")
    
    # Process each CSV file to extract original string values
    file_count = 0
    for csv_file in csv_files:
        try:
            file_count += 1
            if file_count % 10 == 0:
                logger.info(f"Processed {file_count} files out of {len(csv_files)}")
                
            # Load the CSV file with proper encoding
            df = pd.read_csv(csv_file, na_values='', keep_default_na=False)
            
            # Make sure all expected fields are present
            required_fields = ['composite', 'marcKey', 'person', 'roles', 'title', 'provision', 'subjects', 'genres', 'relatedWork', 'personId']
            for field in required_fields:
                if field == 'personId':  # Skip personId - not a string we want to record
                    continue
                    
                if field not in df.columns:
                    continue
                    
                # Process each value in this field
                for value in df[field].dropna().astype(str).unique():
                    if not value:  # Skip empty strings
                        continue
                        
                    # Calculate hash
                    hash_val = hash_string(value)
                    
                    # Record the original string value
                    if hash_val != "NULL" and hash_val in string_counts:
                        hash_to_string_map[hash_val] = value
                        
        except Exception as e:
            logger.warning(f"Error extracting strings from {csv_file}: {str(e)}")
    
    # Create the final string dictionary
    for hash_val in string_counts.keys():
        # Use the extracted string if available, otherwise use empty string
        string_dict[hash_val] = hash_to_string_map.get(hash_val, f"Unknown string with hash {hash_val}")
    
    # Log statistics
    filled_count = sum(1 for val in string_dict.values() if val and not val.startswith("Unknown string"))
    logger.info(f"Built string dictionary with {len(string_dict)} entries, {filled_count} with extracted values")
    
    return string_dict

def process_data(config: Dict[str, Any], input_dir: str, checkpoint_dir: str, use_optimized: bool = None) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Any]]:
    """
    Process input data files and create data structures for entity resolution.
    
    Args:
        config: Configuration dictionary
        input_dir: Directory containing input files
        checkpoint_dir: Directory for saving checkpoints
        use_optimized: Whether to use optimized SQLite-based processing for large datasets
                      (if None, will check config['preprocessing_use_optimized'], default True)
        
    Returns:
        Tuple of (hash_lookup, string_dict, metrics)
    """
    # Check if we should use optimized processing
    if use_optimized is None:
        use_optimized = config.get('preprocessing_use_optimized', True)
    
    if use_optimized:
        logger.info("Using optimized SQLite-based preprocessing for large datasets")
        return process_data_optimized(config, input_dir, checkpoint_dir)
    
    # Otherwise use original in-memory processing
    logger.info(f"Starting data preprocessing from {input_dir}")
    start_time = time.time()
    
    # Create output directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize data structures
    output_data = {}  # personId -> field -> hash
    string_counts = {}  # hash -> count
    field_hash_mapping = {}  # hash -> field -> count
    composite_subject_mapping = {}  # NEW: composite_hash -> subject_hash mapping
    
    # Get list of input files
    input_files = []
    for file in os.listdir(input_dir):
        if file.endswith('.csv'):
            input_files.append(os.path.join(input_dir, file))
    
    if not input_files:
        logger.warning(f"No CSV files found in {input_dir}")
        return {}, {}, {'status': 'no_files_found'}
    
    logger.info(f"Found {len(input_files)} input files")
    
    # Process files in parallel
    total_rows = 0
    file_metrics = []
    base_batch_size = config.get("preprocessing_batch_size", 10)
    max_workers = config.get("preprocessing_workers", 4)
    
    # Calculate optimal batch size to fully utilize all workers
    # Ensure each batch has enough files to keep all workers busy
    optimal_batch_size = max(
        base_batch_size,  # Respect configured minimum
        max_workers * 2,  # At least 2 files per worker
        len(input_files) // max(1, len(input_files) // (max_workers * 3))  # Distribute evenly
    )
    
    # But don't exceed total files or create too many small batches
    batch_size = min(optimal_batch_size, len(input_files))
    
    logger.info(f"Processing {len(input_files)} files with {max_workers} workers")
    logger.info(f"Using batch size {batch_size} (base: {base_batch_size}, optimal: {optimal_batch_size})")
    
    # Process in batches to avoid memory issues
    for i in range(0, len(input_files), batch_size):
        batch_files = input_files[i:i+batch_size]
        effective_workers = min(max_workers, len(batch_files))
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(input_files)-1)//batch_size + 1} with {len(batch_files)} files using {effective_workers}/{max_workers} workers")
        
        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            future_to_file = {
                executor.submit(process_file, file_path): file_path
                for file_path in batch_files
            }
            
            # Process results as they complete
            for future in tqdm(as_completed(future_to_file), total=len(batch_files), desc="Processing files"):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    
                    # Merge batch results into main data structures
                    if 'output_data' in result:
                        for person_id, fields in result['output_data'].items():
                            output_data[person_id] = fields
                            
                            # Build composite-subject mapping during processing
                            composite_hash = fields.get('composite')
                            subject_hash = fields.get('subjects')
                            
                            if composite_hash and composite_hash != "NULL":
                                # Map composite to subject (None if subject is NULL or missing)
                                composite_subject_mapping[composite_hash] = (
                                    subject_hash if subject_hash and subject_hash != "NULL" else None
                                )
                            
                    if 'string_counts' in result:
                        for hash_val, count in result['string_counts'].items():
                            string_counts[hash_val] = string_counts.get(hash_val, 0) + count
                            
                    if 'field_hash_mapping' in result:
                        for hash_val, fields in result['field_hash_mapping'].items():
                            if hash_val not in field_hash_mapping:
                                field_hash_mapping[hash_val] = {}
                            for field, count in fields.items():
                                field_hash_mapping[hash_val][field] = field_hash_mapping[hash_val].get(field, 0) + count
                    
                    # Track metrics
                    total_rows += result.get('rows_processed', 0)
                    file_metrics.append(result)
                    
                except Exception as e:
                    logger.error(f"Error processing results for {file_path}: {str(e)}")
        
        # Save checkpoint after each batch
        save_checkpoint(output_data, string_counts, field_hash_mapping, checkpoint_dir, i // batch_size)
    
    # Create string dictionary
    string_dict = create_string_dict(string_counts, field_hash_mapping, input_dir)
    
    # Save final data structures
    hash_lookup_path = os.path.join(checkpoint_dir, 'hash_lookup.pkl')
    string_dict_path = os.path.join(checkpoint_dir, 'string_dict.pkl')
    string_counts_path = os.path.join(checkpoint_dir, 'string_counts.pkl')
    field_hash_mapping_path = os.path.join(checkpoint_dir, 'field_hash_mapping.pkl')
    
    with open(hash_lookup_path, 'wb') as f:
        pickle.dump(output_data, f)
        
    with open(string_dict_path, 'wb') as f:
        pickle.dump(string_dict, f)
        
    with open(string_counts_path, 'wb') as f:
        pickle.dump(string_counts, f)
        
    with open(field_hash_mapping_path, 'wb') as f:
        pickle.dump(field_hash_mapping, f)
    
    # Save composite-subject mapping
    save_composite_subject_mapping(composite_subject_mapping, checkpoint_dir)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Preprocessing completed in {elapsed_time:.2f} seconds")
    logger.info(f"Processed {total_rows} total rows")
    logger.info(f"Found {len(output_data)} unique entities")
    logger.info(f"Found {len(string_counts)} unique strings")
    logger.info(f"Generated {len(composite_subject_mapping)} composite-subject mappings")
    
    # Log subject mapping statistics
    subjects_present = sum(1 for v in composite_subject_mapping.values() if v is not None)
    subjects_missing = sum(1 for v in composite_subject_mapping.values() if v is None)
    logger.info(f"Subject mapping: {subjects_present} records have subjects, {subjects_missing} need imputation")
    
    # Return final data structures and metrics
    metrics = {
        'status': 'completed',
        'elapsed_time': elapsed_time,
        'total_files': len(input_files),
        'total_rows': total_rows,
        'entity_count': len(output_data),
        'unique_strings': len(string_counts),
        'composite_subject_mappings': len(composite_subject_mapping),
        'subjects_present': subjects_present,
        'subjects_missing': subjects_missing,
        'file_metrics': file_metrics
    }
    
    return output_data, string_dict, metrics


class OptimizedPreprocessor:
    """Optimized preprocessor using SQLite for scalable processing"""
    
    def __init__(self, checkpoint_dir: str, batch_size: int = 10, max_workers: int = 4):
        self.checkpoint_dir = checkpoint_dir
        self.batch_size = batch_size
        self.max_workers = max_workers
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize SQLite database for intermediate storage
        self.db_path = os.path.join(checkpoint_dir, 'preprocessing_temp.db')
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with optimized settings"""
        with self._get_connection() as conn:
            # Enable optimizations for bulk loading
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = OFF")  # Much faster for bulk loads
            conn.execute("PRAGMA cache_size = -128000")  # 128MB cache
            conn.execute("PRAGMA temp_store = MEMORY")
            conn.execute("PRAGMA mmap_size = 30000000000")  # 30GB mmap
            conn.execute("PRAGMA page_size = 32768")  # Larger pages
            conn.execute("PRAGMA locking_mode = EXCLUSIVE")  # No concurrent access needed
            
            # Create tables
            conn.execute("""
                CREATE TABLE IF NOT EXISTS entity_hashes (
                    person_id TEXT PRIMARY KEY,
                    composite_hash TEXT,
                    person_hash TEXT,
                    roles_hash TEXT,
                    title_hash TEXT,
                    provision_hash TEXT,
                    subjects_hash TEXT,
                    genres_hash TEXT,
                    marcKey_hash TEXT,
                    relatedWork TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS string_hashes (
                    hash TEXT PRIMARY KEY,
                    original_value TEXT,
                    count INTEGER DEFAULT 1
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS field_hash_mapping (
                    hash TEXT,
                    field TEXT,
                    count INTEGER DEFAULT 1,
                    PRIMARY KEY (hash, field)
                )
            """)
            
            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_string_count ON string_hashes(count)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_field_hash ON field_hash_mapping(hash)")
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with optimized settings"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def process_file_batch(self, file_paths: List[str], batch_num: int) -> Dict[str, Any]:
        """Process a batch of files and write directly to SQLite"""
        start_time = time.time()
        total_rows = 0
        errors = []
        
        # Use all available workers, but don't exceed the number of files
        effective_workers = min(self.max_workers, len(file_paths))
        logger.debug(f"Batch {batch_num}: Processing {len(file_paths)} files with {effective_workers} workers")
        
        # Process files in parallel first to prepare data
        file_data = []
        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            future_to_file = {executor.submit(self._prepare_file_data, fp): fp for fp in file_paths}
            
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        file_data.append(result)
                        total_rows += result['rows_processed']
                except Exception as e:
                    logger.error(f"Error preparing file {file_path}: {str(e)}")
                    errors.append((file_path, str(e)))
        
        # Now write all data to SQLite in one transaction
        if file_data:
            with self._get_connection() as conn:
                conn.execute("BEGIN TRANSACTION")
                
                # Combine all data
                all_entities = []
                all_strings = {}
                all_mappings = {}
                
                for data in file_data:
                    all_entities.extend(data['entities'])
                    all_strings.update(data['strings'])
                    all_mappings.update(data['mappings'])
                
                # Bulk insert
                if all_entities:
                    conn.executemany("""
                        INSERT OR REPLACE INTO entity_hashes 
                        (person_id, composite_hash, person_hash, roles_hash, title_hash, provision_hash, subjects_hash, genres_hash, relatedWork_hash, marcKey_hash)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, all_entities)
                
                if all_strings:
                    string_list = [(h, v) for h, v in all_strings.items()]
                    conn.executemany("""
                        INSERT OR IGNORE INTO string_hashes (hash, original_value, count)
                        VALUES (?, ?, 0)
                    """, string_list)
                    
                    conn.executemany("""
                        UPDATE string_hashes SET count = count + 1 WHERE hash = ?
                    """, [(h,) for h in all_strings.keys()])
                
                if all_mappings:
                    mapping_list = list(all_mappings.values())
                    conn.executemany("""
                        INSERT INTO field_hash_mapping (hash, field, count)
                        VALUES (?, ?, 1)
                        ON CONFLICT(hash, field) DO UPDATE SET count = count + 1
                    """, mapping_list)
                
                conn.execute("COMMIT")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Batch {batch_num}: Processed {len(file_paths)} files, {total_rows} rows in {elapsed_time:.2f}s")
        
        return {
            'batch_num': batch_num,
            'files_processed': len(file_paths),
            'rows_processed': total_rows,
            'elapsed_time': elapsed_time,
            'errors': errors
        }
    
    def _prepare_file_data(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Prepare data from a single file without database access"""
        try:
            # Load CSV
            df = pd.read_csv(file_path, na_values='', keep_default_na=False)
            
            # Expected fields
            required_fields = ['composite', 'marcKey', 'relatedWork', 'person', 'roles', 'title', 'provision', 'subjects', 'genres', 'personId']
            
            # Fill missing fields
            for field in required_fields:
                if field not in df.columns:
                    df[field] = ''
            
            # Filter out rows without personId
            df = df[df['personId'].notna() & (df['personId'] != '')]
            rows_processed = len(df)
            
            if rows_processed == 0:
                return None
            
            # Convert all fields to string arrays for better performance
            str_columns = {}
            for field in required_fields:
                if field != 'personId':
                    str_columns[field] = df[field].fillna('').astype(str).values
            
            # Collect all unique values across all fields
            all_unique_values = set()
            field_to_unique = {}
            
            for field, values in str_columns.items():
                unique_vals = pd.unique(values)
                field_to_unique[field] = unique_vals
                all_unique_values.update(unique_vals)
            
            # Remove empty string
            all_unique_values.discard('')
            
            # Hash all unique values at once - much faster than repeated hashing
            value_to_hash = {}
            for value in all_unique_values:
                if value:
                    value_to_hash[value] = hash_string(value)
            value_to_hash[''] = 'NULL'
            
            # Build string records and field mappings
            string_records = {}
            field_mappings = {}
            
            for field, unique_vals in field_to_unique.items():
                for value in unique_vals:
                    if value and value in value_to_hash:
                        hash_val = value_to_hash[value]
                        if hash_val != "NULL":
                            string_records[hash_val] = value
                            field_key = f"{hash_val}_{field}"
                            field_mappings[field_key] = (hash_val, field)
            
            # Build entity records using pre-computed hashes
            person_ids = df['personId'].astype(str).values
            
            # Create hash arrays by mapping values through our hash dictionary
            entity_records = []
            for i in range(len(person_ids)):
                record = [person_ids[i]]
                for field in required_fields:
                    if field != 'personId':
                        value = str_columns[field][i]
                        record.append(value_to_hash[value])
                entity_records.append(tuple(record))
            
            return {
                'entities': entity_records,
                'strings': string_records,
                'mappings': field_mappings,
                'rows_processed': rows_processed
            }
            
        except Exception as e:
            logger.error(f"Error preparing data from {file_path}: {str(e)}")
            raise
    
    def _process_single_file(self, conn: sqlite3.Connection, file_path: str) -> int:
        """Process a single file and insert data into SQLite"""
        # Load CSV
        df = pd.read_csv(file_path, na_values='', keep_default_na=False)
        
        # Expected fields
        required_fields = ['composite', 'marcKey', 'relatedWork', 'person', 'roles', 'title', 'provision', 'subjects', 'genres', 'personId']
        
        # Fill missing fields
        for field in required_fields:
            if field not in df.columns:
                df[field] = ''
        
        # Filter out rows without personId
        df = df[df['personId'].notna() & (df['personId'] != '')]
        rows_processed = len(df)
        
        if rows_processed == 0:
            return 0
        
        # Vectorized processing - much faster than iterrows
        entity_records = []
        string_records = {}  # Use dict to dedupe
        field_mappings = {}  # Use dict to dedupe
        
        # Process all fields at once
        for field in required_fields:
            if field == 'personId':
                continue
            
            # Vectorized string conversion and hashing
            values = df[field].fillna('').astype(str)
            
            # Process unique values only
            unique_values = values.unique()
            for value in unique_values:
                if value:  # Skip empty strings
                    hash_val = hash_string(value)
                    if hash_val != "NULL":
                        string_records[hash_val] = value
                        field_key = f"{hash_val}_{field}"
                        field_mappings[field_key] = (hash_val, field)
        
        # Build entity records using vectorized operations
        # Pre-compute all hashes for all fields
        field_hashes = {}
        for field in required_fields:
            if field == 'personId':
                continue
            # Vectorized hashing
            values = df[field].fillna('').astype(str)
            field_hashes[field] = values.apply(hash_string)
        
        # Build entity records efficiently using zip
        person_ids = df['personId'].astype(str).tolist()
        
        # Create tuples using list comprehension and zip - much faster than iterrows
        hash_columns = [field_hashes[field].tolist() for field in required_fields if field != 'personId']
        entity_records = [(pid,) + tuple(hashes) for pid, hashes in zip(person_ids, zip(*hash_columns))]
        
        # Bulk insert all records at once
        if entity_records:
            # Use transactions for better performance
            conn.execute("BEGIN TRANSACTION")
            
            # Insert entities
            conn.executemany("""
                INSERT OR REPLACE INTO entity_hashes 
                (person_id, composite_hash, person_hash, roles_hash, title_hash, provision_hash, subjects_hash, genres_hash, marcKey_hash, relatedWork_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, entity_records)
            
            # Bulk insert strings
            if string_records:
                # Convert to list for executemany
                string_list = [(h, v) for h, v in string_records.items()]
                conn.executemany("""
                    INSERT OR IGNORE INTO string_hashes (hash, original_value, count)
                    VALUES (?, ?, 0)
                """, string_list)
                
                # Update counts separately
                conn.executemany("""
                    UPDATE string_hashes SET count = count + 1 WHERE hash = ?
                """, [(h,) for h in string_records.keys()])
            
            # Bulk insert field mappings
            if field_mappings:
                mapping_list = list(field_mappings.values())
                conn.executemany("""
                    INSERT INTO field_hash_mapping (hash, field, count)
                    VALUES (?, ?, 1)
                    ON CONFLICT(hash, field) DO UPDATE SET count = count + 1
                """, mapping_list)
            
            conn.execute("COMMIT")
        
        return rows_processed
    
    
    def process_all_files(self, input_files: List[str]) -> Dict[str, Any]:
        """Process all CSV files"""
        start_time = time.time()
        
        logger.info(f"Processing {len(input_files)} files using optimized SQLite backend")
        
        # Process files in batches
        total_rows = 0
        batch_metrics = []
        
        for i in range(0, len(input_files), self.batch_size):
            batch_files = input_files[i:i+self.batch_size]
            batch_num = i // self.batch_size + 1
            
            logger.info(f"Processing batch {batch_num}/{(len(input_files)-1)//self.batch_size + 1}")
            
            # Process this batch
            metrics = self.process_file_batch(batch_files, batch_num)
            total_rows += metrics['rows_processed']
            batch_metrics.append(metrics)
        
        elapsed_time = time.time() - start_time
        
        return {
            'status': 'completed',
            'elapsed_time': elapsed_time,
            'total_files': len(input_files),
            'total_rows': total_rows,
            'batch_metrics': batch_metrics
        }
    
    def export_to_pickle(self) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, int]]:
        """Export SQLite data to pickle format compatible with existing pipeline"""
        logger.info("Exporting data from SQLite to pickle format")
        
        output_data = {}
        string_dict = {}
        string_counts = {}
        composite_subject_mapping = {}
        
        with self._get_connection() as conn:
            # Export entity hashes
            cursor = conn.execute("""
                SELECT person_id, composite_hash, person_hash, roles_hash, 
                       title_hash, provision_hash, subjects_hash, genres_hash, marcKey_hash, relatedWork_hash
                FROM entity_hashes
            """)
            
            for row in cursor:
                person_id = row['person_id']
                output_data[person_id] = {
                    'composite': row['composite_hash'],
                    'person': row['person_hash'],
                    'roles': row['roles_hash'],
                    'title': row['title_hash'],
                    'provision': row['provision_hash'],
                    'subjects': row['subjects_hash'],
                    'genres': row['genres_hash'],
                    'marcKey': row['marcKey_hash'],
                    'relatedWork': row['relatedWork_hash']
                }
                
                # Build composite-subject mapping during export
                composite_hash = row['composite_hash']
                subject_hash = row['subjects_hash']
                
                if composite_hash and composite_hash != "NULL":
                    # Map composite to subject (None if subject is NULL or missing)
                    composite_subject_mapping[composite_hash] = (
                        subject_hash if subject_hash and subject_hash != "NULL" else None
                    )
            
            # Export string dictionary and counts
            cursor = conn.execute("SELECT hash, original_value, count FROM string_hashes")
            for row in cursor:
                hash_val = row['hash']
                string_dict[hash_val] = row['original_value'] or f"Unknown string with hash {hash_val}"
                string_counts[hash_val] = row['count']
        
        # Save to pickle files
        hash_lookup_path = os.path.join(self.checkpoint_dir, 'hash_lookup.pkl')
        string_dict_path = os.path.join(self.checkpoint_dir, 'string_dict.pkl')
        string_counts_path = os.path.join(self.checkpoint_dir, 'string_counts.pkl')
        
        with open(hash_lookup_path, 'wb') as f:
            pickle.dump(output_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open(string_dict_path, 'wb') as f:
            pickle.dump(string_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open(string_counts_path, 'wb') as f:
            pickle.dump(string_counts, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Also save field_hash_mapping for compatibility
        field_hash_mapping = {}
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT hash, field, count FROM field_hash_mapping")
            for row in cursor:
                hash_val = row['hash']
                if hash_val not in field_hash_mapping:
                    field_hash_mapping[hash_val] = {}
                field_hash_mapping[hash_val][row['field']] = row['count']
        
        field_hash_mapping_path = os.path.join(self.checkpoint_dir, 'field_hash_mapping.pkl')
        with open(field_hash_mapping_path, 'wb') as f:
            pickle.dump(field_hash_mapping, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Save composite-subject mapping
        save_composite_subject_mapping(composite_subject_mapping, self.checkpoint_dir)
        
        # Log composite-subject mapping statistics
        subjects_present = sum(1 for v in composite_subject_mapping.values() if v is not None)
        subjects_missing = sum(1 for v in composite_subject_mapping.values() if v is None)
        logger.info(f"Exported {len(output_data)} entities, {len(string_dict)} unique strings")
        logger.info(f"Generated {len(composite_subject_mapping)} composite-subject mappings")
        logger.info(f"Subject mapping: {subjects_present} records have subjects, {subjects_missing} need imputation")
        
        return output_data, string_dict, string_counts
    
    def cleanup(self):
        """Remove temporary SQLite database"""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
            logger.info("Cleaned up temporary database")


def process_data_optimized(config: Dict[str, Any], input_dir: str, checkpoint_dir: str) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Any]]:
    """
    Optimized preprocessing using memory-efficient batch processing
    
    Args:
        config: Configuration dictionary
        input_dir: Directory containing input files
        checkpoint_dir: Directory for saving checkpoints
        
    Returns:
        Tuple of (hash_lookup, string_dict, metrics)
    """
    logger.info(f"Starting optimized preprocessing from {input_dir}")
    start_time = time.time()
    
    # Get list of input files
    input_files = []
    for file in os.listdir(input_dir):
        if file.endswith('.csv'):
            input_files.append(os.path.join(input_dir, file))
    
    if not input_files:
        logger.warning(f"No CSV files found in {input_dir}")
        return {}, {}, {'status': 'no_files_found'}
    
    logger.info(f"Found {len(input_files)} input files to process")
    
    # Get configuration parameters
    base_batch_size = config.get("preprocessing_batch_size", 50)
    max_workers = config.get("preprocessing_workers", 4)
    
    # Calculate optimal batch size to fully utilize all workers
    # For optimized processing, we can handle larger batches more efficiently
    optimal_batch_size = max(
        base_batch_size,  # Respect configured minimum
        max_workers * 3,  # At least 3 files per worker for better utilization
        len(input_files) // max(1, len(input_files) // (max_workers * 4))  # Distribute evenly
    )
    
    # But don't exceed total files
    batch_size = min(optimal_batch_size, len(input_files))
    
    logger.info(f"Processing {len(input_files)} files with {max_workers} workers")
    logger.info(f"Using batch size {batch_size} (base: {base_batch_size}, optimal: {optimal_batch_size})")
    
    # Initialize global data structures
    output_data = {}
    string_dict = {}
    string_counts = defaultdict(int)  # Use defaultdict for cleaner code
    field_hash_mapping = defaultdict(lambda: defaultdict(int))
    composite_subject_mapping = {}  # NEW: Composite-subject mapping for subject enhancement
    
    # Process files in batches
    total_rows = 0
    total_batches = (len(input_files) - 1) // batch_size + 1
    
    for batch_idx in range(0, len(input_files), batch_size):
        batch_start = time.time()
        batch_files = input_files[batch_idx:batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1
        effective_workers = min(max_workers, len(batch_files))
        
        logger.info(f"Processing batch {batch_num}/{total_batches} with {len(batch_files)} files using {effective_workers}/{max_workers} workers")
        
        # Process files in parallel
        batch_results = []
        batch_worker_start_time = time.time()
        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            # Use the optimized file processing function
            future_to_file = {
                executor.submit(_process_file_optimized, file_path): file_path 
                for file_path in batch_files
            }
            
            active_workers = 0
            completed_files = 0
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                completed_files += 1
                try:
                    result = future.result()
                    if result:
                        batch_results.append(result)
                        total_rows += result['rows_processed']
                        
                    # Log progress every 50 files or at key milestones
                    if completed_files % 50 == 0 or completed_files in [10, 25] or completed_files == len(batch_files):
                        elapsed = time.time() - batch_worker_start_time
                        rate = completed_files / elapsed if elapsed > 0 else 0
                        logger.info(f"  Batch {batch_num}: Completed {completed_files}/{len(batch_files)} files ({rate:.1f} files/sec)")
                        
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {str(e)}")
        
        # Merge batch results more efficiently
        for result in batch_results:
            # Bulk update entities
            output_data.update(result['entities_dict'])
            
            # Merge strings
            string_dict.update(result['strings'])
            for hash_val in result['strings']:
                string_counts[hash_val] += 1
            
            # Merge field mappings
            for hash_val, field in result['mappings'].values():
                field_hash_mapping[hash_val][field] += 1
            
            # Build composite-subject mapping during batch processing
            for person_id, entity_data in result['entities_dict'].items():
                composite_hash = entity_data.get('composite')
                subject_hash = entity_data.get('subjects')
                
                if composite_hash and composite_hash != "NULL":
                    # Map composite to subject (None if subject is NULL or missing)
                    composite_subject_mapping[composite_hash] = (
                        subject_hash if subject_hash and subject_hash != "NULL" else None
                    )
        
        batch_elapsed = time.time() - batch_start
        batch_rows = sum(r['rows_processed'] for r in batch_results)
        logger.info(f"Batch {batch_num}: Processed {len(batch_files)} files, "
                   f"{batch_rows} rows in {batch_elapsed:.2f}s "
                   f"({batch_rows/batch_elapsed:.0f} rows/sec)")
        
        # Skip intermediate checkpoints - they're too expensive with large datasets
        # Only save at the end
    
    # Save final results
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    logger.info("Saving final results...")
    save_start = time.time()
    
    with open(os.path.join(checkpoint_dir, 'hash_lookup.pkl'), 'wb') as f:
        pickle.dump(output_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(os.path.join(checkpoint_dir, 'string_dict.pkl'), 'wb') as f:
        pickle.dump(string_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(os.path.join(checkpoint_dir, 'string_counts.pkl'), 'wb') as f:
        pickle.dump(dict(string_counts), f, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(os.path.join(checkpoint_dir, 'field_hash_mapping.pkl'), 'wb') as f:
        pickle.dump(dict(field_hash_mapping), f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Save composite-subject mapping
    save_composite_subject_mapping(composite_subject_mapping, checkpoint_dir)
    
    save_elapsed = time.time() - save_start
    logger.info(f"Saved results in {save_elapsed:.2f} seconds")
    
    elapsed_time = time.time() - start_time
    
    logger.info(f"Preprocessing completed in {elapsed_time:.2f} seconds")
    logger.info(f"Processed {total_rows} total rows ({total_rows/elapsed_time:.0f} rows/sec)")
    logger.info(f"Found {len(output_data)} unique entities")
    logger.info(f"Found {len(string_dict)} unique strings")
    logger.info(f"Generated {len(composite_subject_mapping)} composite-subject mappings")
    
    # Log subject mapping statistics
    subjects_present = sum(1 for v in composite_subject_mapping.values() if v is not None)
    subjects_missing = sum(1 for v in composite_subject_mapping.values() if v is None)
    logger.info(f"Subject mapping: {subjects_present} records have subjects, {subjects_missing} need imputation")
    
    metrics = {
        'status': 'completed',
        'elapsed_time': elapsed_time,
        'total_files': len(input_files),
        'total_rows': total_rows,
        'entity_count': len(output_data),
        'unique_strings': len(string_dict),
        'composite_subject_mappings': len(composite_subject_mapping),
        'subjects_present': subjects_present,
        'subjects_missing': subjects_missing,
        'rows_per_second': total_rows / elapsed_time
    }
    
    return output_data, string_dict, metrics


def _process_file_optimized(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Optimized file processing with better memory efficiency and speed
    """
    try:
        # Read CSV with optimized settings
        df = pd.read_csv(
            file_path,
            na_values='',
            keep_default_na=False,
            engine='c',  # Faster C engine
            dtype=str,   # Skip type inference
        )
        
        # Expected fields
        required_fields = ['composite', 'marcKey', 'relatedWork', 'person', 'roles', 'title', 'provision', 'subjects', 'genres', 'personId']
        
        # Add missing columns efficiently
        missing_cols = set(required_fields) - set(df.columns)
        if missing_cols:
            for col in missing_cols:
                df[col] = ''
        
        # Filter valid personIds using vectorized operations
        valid_mask = (df['personId'].notna()) & (df['personId'] != '')
        if not valid_mask.any():
            return None
        
        df_valid = df.loc[valid_mask, required_fields]
        rows_processed = len(df_valid)
        
        # Convert to numpy for faster processing - ensure all values are strings
        data_array = df_valid.astype(str).values
        person_ids = data_array[:, required_fields.index('personId')]
        
        # Collect unique values across all fields in one pass
        unique_collector = defaultdict(set)
        field_cols = {}
        
        for col_idx, field in enumerate(required_fields):
            if field != 'personId':
                col_data = data_array[:, col_idx]
                field_cols[field] = col_data
                
                # Get unique values - convert to string to avoid comparison issues
                unique_vals = pd.unique(col_data.astype(str))
                for val in unique_vals:
                    if val and val != 'nan':  # Skip empty strings and 'nan'
                        unique_collector[val].add(field)
        
        # Hash all unique values once
        value_to_hash = {'': 'NULL'}
        string_records = {}
        field_mappings = {}
        
        for value, fields in unique_collector.items():
            hash_val = hash_string(value)
            value_to_hash[value] = hash_val
            string_records[hash_val] = value
            
            # Build field mappings
            for field in fields:
                field_key = f"{hash_val}_{field}"
                field_mappings[field_key] = (hash_val, field)
        
        # Build entity records efficiently
        entities_dict = {}
        
        for i in range(rows_processed):
            person_id = person_ids[i]
            
            # Build entity dict directly
            entity = {}
            for field in required_fields:
                if field != 'personId':
                    value = field_cols[field][i]
                    entity[field] = value_to_hash.get(value, 'NULL')
            
            entities_dict[person_id] = entity
        
        return {
            'entities_dict': entities_dict,
            'strings': string_records,
            'mappings': field_mappings,
            'rows_processed': rows_processed
        }
        
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        raise


def _process_file_simple(file_path: str) -> Optional[Dict[str, Any]]:
    """Ultra-optimized file processing with minimal overhead"""
    try:
        # Use pandas with optimized settings for faster CSV reading
        df = pd.read_csv(
            file_path, 
            na_values='', 
            keep_default_na=False,
            engine='c',  # C engine is faster
            dtype=str,   # Read everything as string to avoid type inference
            memory_map=True  # Memory map large files
        )
        
        # Expected fields
        required_fields = ['composite', 'marcKey', 'relatedWork', 'person', 'roles', 'title', 'provision', 'subjects', 'genres', 'personId']
        field_indices = {field: i for i, field in enumerate(required_fields) if field != 'personId'}
        
        # Add missing columns with empty strings
        for field in required_fields:
            if field not in df.columns:
                df[field] = ''
        
        # Filter valid personIds using numpy for speed
        person_mask = (df['personId'].notna()) & (df['personId'] != '')
        if not person_mask.any():
            return None
        
        # Work with numpy arrays for maximum performance
        data_array = df.loc[person_mask, required_fields].values
        rows_processed = len(data_array)
        
        # Extract person IDs
        person_ids = data_array[:, required_fields.index('personId')]
        
        # Single-pass processing: collect unique values and build hash mapping
        unique_collector = defaultdict(set)
        
        # Process each field column
        for field, col_idx in field_indices.items():
            col_data = data_array[:, col_idx]
            # Get unique values for this column
            unique_vals = np.unique(col_data)
            for val in unique_vals:
                if val:  # Skip empty strings
                    unique_collector[val].add(field)
        
        # Hash all unique values in one pass
        value_to_hash = {'': 'NULL'}
        string_records = {}
        field_mappings = {}
        
        for value, fields in unique_collector.items():
            hash_val = hash_string(value)
            value_to_hash[value] = hash_val
            string_records[hash_val] = value
            
            # Build field mappings for this value
            for field in fields:
                field_key = f"{hash_val}_{field}"
                field_mappings[field_key] = (hash_val, field)
        
        # Vectorized entity record creation
        # Pre-allocate the entity records array
        entity_records = []
        
        # Build records using numpy operations
        for i in range(rows_processed):
            record = [person_ids[i]]
            row = data_array[i]
            
            # Append hashes in field order
            for field in required_fields:
                if field != 'personId':
                    col_idx = required_fields.index(field)
                    value = row[col_idx]
                    record.append(value_to_hash.get(value, 'NULL'))
            
            entity_records.append(tuple(record))
        
        return {
            'entities': entity_records,
            'strings': string_records,
            'mappings': field_mappings,
            'rows_processed': rows_processed
        }
        
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        raise


def save_checkpoint(output_data: Dict[str, Dict[str, str]], string_counts: Dict[str, int],
                  field_hash_mapping: Dict[str, Dict[str, int]], checkpoint_dir: str, batch_num: int) -> None:
    """
    Save checkpoint data during processing.
    
    Args:
        output_data: Dictionary of personId to field hashes
        string_counts: Dictionary of string counts
        field_hash_mapping: Dictionary mapping field/hash to usage counts
        checkpoint_dir: Directory for saving checkpoints
        batch_num: Batch number for checkpoint
    """
    checkpoint_path = os.path.join(checkpoint_dir, f'preprocessing_checkpoint_{batch_num}.pkl')
    
    checkpoint_data = {
        'output_data': output_data,
        'string_counts': string_counts,
        'field_hash_mapping': field_hash_mapping,
        'timestamp': time.time()
    }
    
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint_data, f)
        
    logger.debug(f"Saved preprocessing checkpoint to {checkpoint_path}")

def load_hash_lookup(path: str) -> Dict[str, Dict[str, str]]:
    """
    Load hash lookup data from checkpoint.
    
    Args:
        path: Path to the hash lookup pickle file
        
    Returns:
        Dictionary mapping personId to field hashes
    """
    try:
        with open(path, 'rb') as f:
            hash_lookup = pickle.load(f)
        
        logger.info(f"Loaded hash lookup with {len(hash_lookup)} entities from {path}")
        return hash_lookup
        
    except Exception as e:
        logger.error(f"Error loading hash lookup from {path}: {str(e)}")
        return {}

def load_string_dict(path: str) -> Dict[str, str]:
    """
    Load string dictionary from checkpoint.
    
    Args:
        path: Path to the string dictionary pickle file
        
    Returns:
        Dictionary mapping hash to string value
    """
    try:
        with open(path, 'rb') as f:
            string_dict = pickle.load(f)
        
        logger.info(f"Loaded string dictionary with {len(string_dict)} entries from {path}")
        return string_dict
        
    except Exception as e:
        logger.error(f"Error loading string dictionary from {path}: {str(e)}")
        return {}

def load_string_counts(path: str) -> Dict[str, int]:
    """
    Load string counts from checkpoint.
    
    Args:
        path: Path to the string counts pickle file
        
    Returns:
        Dictionary mapping hash to count
    """
    try:
        with open(path, 'rb') as f:
            string_counts = pickle.load(f)
        
        logger.info(f"Loaded string counts with {len(string_counts)} entries from {path}")
        return string_counts
        
    except Exception as e:
        logger.error(f"Error loading string counts from {path}: {str(e)}")
        return {}

def load_composite_subject_mapping(path: str) -> Dict[str, Optional[str]]:
    """
    Load composite-subject mapping from checkpoint.
    
    Args:
        path: Path to the composite-subject mapping pickle file
        
    Returns:
        Dictionary mapping composite hash to subject hash (or None for missing subjects)
    """
    try:
        with open(path, 'rb') as f:
            composite_subject_mapping = pickle.load(f)
        
        logger.info(f"Loaded composite-subject mapping with {len(composite_subject_mapping)} entries from {path}")
        return composite_subject_mapping
        
    except Exception as e:
        logger.error(f"Error loading composite-subject mapping from {path}: {str(e)}")
        return {}

def save_composite_subject_mapping(mapping: Dict[str, Optional[str]], checkpoint_dir: str) -> None:
    """
    Save composite-subject mapping to checkpoint.
    
    Args:
        mapping: Dictionary mapping composite hash to subject hash
        checkpoint_dir: Directory for saving checkpoint
    """
    try:
        os.makedirs(checkpoint_dir, exist_ok=True)
        mapping_path = os.path.join(checkpoint_dir, 'composite_subject_mapping.pkl')
        
        with open(mapping_path, 'wb') as f:
            pickle.dump(mapping, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        logger.info(f"Saved composite-subject mapping with {len(mapping)} entries to {mapping_path}")
        
    except Exception as e:
        logger.error(f"Error saving composite-subject mapping: {str(e)}")
        raise

def main(config_path: str = 'config.yml'):
    """
    Main function for running preprocessing.
    
    Args:
        config_path: Path to the configuration file
    """
    # This function would typically be called from the orchestrator
    # Implemented here for standalone module testing
    pass

if __name__ == "__main__":
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description='Preprocess data for entity resolution')
    parser.add_argument('--config', default='config.yml', help='Path to configuration file')
    parser.add_argument('--input-dir', help='Directory containing input files')
    parser.add_argument('--checkpoint-dir', help='Directory for saving checkpoints')
    parser.add_argument('--optimized', action='store_true', help='Use optimized SQLite-based preprocessing (recommended for large datasets)')
    args = parser.parse_args()
    
    # Load configuration with environment-specific overrides
    from src.config_utils import load_config_with_environment
    config = load_config_with_environment(args.config)
    
    # Override config with command line arguments
    if args.input_dir:
        config['input_dir'] = args.input_dir
    
    if args.checkpoint_dir:
        config['checkpoint_dir'] = args.checkpoint_dir
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run preprocessing
    input_dir = config.get('input_dir', 'data/input')
    checkpoint_dir = config.get('checkpoint_dir', 'data/checkpoints')
    
    # Use optimized flag if provided, otherwise let process_data check config
    use_optimized = args.optimized if args.optimized else None
    output_data, string_dict, metrics = process_data(config, input_dir, checkpoint_dir, use_optimized=use_optimized)
