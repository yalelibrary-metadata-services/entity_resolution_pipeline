"""
Utility functions for the entity resolution pipeline.
"""

import os
import time
import logging
import json
import pickle
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
import psutil
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import wraps
from concurrent import futures

logger = logging.getLogger(__name__)

class Timer:
    """Simple context manager for timing code blocks."""
    
    def __init__(self, name=None):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.duration = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args, **kwargs):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        if self.name:
            logger.info(f"{self.name} completed in {self.duration:.2f} seconds")

def get_memory_usage():
    """Get current memory usage in GB."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_gb = memory_info.rss / (1024 ** 3)  # Convert bytes to GB
    return memory_gb

def save_checkpoint(state, filepath):
    """
    Save checkpoint to file.
    
    Args:
        state (dict): State to save
        filepath (str or Path): Path to save checkpoint
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save to temporary file first
    temp_filepath = f"{filepath}.tmp"
    with open(temp_filepath, 'wb') as f:
        pickle.dump(state, f)
    
    # Rename to final filename
    os.rename(temp_filepath, filepath)
    
    logger.debug(f"Checkpoint saved to {filepath}")

def load_checkpoint(filepath):
    """
    Load checkpoint from file.
    
    Args:
        filepath (str or Path): Path to checkpoint file
        
    Returns:
        dict: Checkpoint state
    """
    if not os.path.exists(filepath):
        logger.warning(f"Checkpoint file {filepath} does not exist")
        return {}
    
    with open(filepath, 'rb') as f:
        state = pickle.load(f)
    
    logger.debug(f"Checkpoint loaded from {filepath}")
    return state

def compute_string_hash(text):
    """
    Compute deterministic hash for a string.
    
    Args:
        text (str): Input string
        
    Returns:
        str: Hash string
    """
    if not text or text.strip() == '':
        return "132172610905071792854514019103556680276"  # Hash for empty string
    
    # Handle non-string inputs
    if not isinstance(text, str):
        text = str(text)
    
    # Compute hash
    hash_obj = hashlib.md5(text.encode('utf-8'))
    hash_hex = hash_obj.hexdigest()
    
    # Convert to integer for more compact representation
    hash_int = int(hash_hex, 16)
    
    return str(hash_int)

def batch_executor(items, process_func, batch_size=1000, max_workers=None, use_processes=False):
    """
    Process items in batches using parallel execution.
    
    Args:
        items (list): Items to process
        process_func (callable): Function to process each batch
        batch_size (int): Size of each batch
        max_workers (int): Maximum number of workers
        use_processes (bool): Use processes instead of threads
        
    Returns:
        list: Results from processing
    """
    # Create batches
    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    results = []
    
    # Choose executor based on parameter
    executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    
    with executor_class(max_workers=max_workers) as executor:
        # Submit jobs
        future_to_batch = {executor.submit(process_func, batch): i for i, batch in enumerate(batches)}
        
        # Process results as they complete
        for future in futures.as_completed(future_to_batch):
            batch_idx = future_to_batch[future]
            try:
                batch_result = future.result()
                results.extend(batch_result)
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
    
    return results

def save_json(data, filepath):
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(filepath):
    """Load data from JSON file."""
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r') as f:
        return json.load(f)

def compute_harmonic_mean(a, b):
    """
    Compute harmonic mean of two numbers.
    
    Args:
        a (float): First number
        b (float): Second number
        
    Returns:
        float: Harmonic mean
    """
    if a <= 0 or b <= 0:
        return 0
    return 2 * a * b / (a + b)

def extract_birth_death_years(name_string):
    """
    Extract birth and death years from name string.
    Uses the BirthDeathYearExtractor from src.birth_death_regexes.
    
    Args:
        name_string (str): Name string
        
    Returns:
        tuple: (birth_year, death_year)
    """
    from src.birth_death_regexes import BirthDeathYearExtractor
    extractor = BirthDeathYearExtractor()
    return extractor.parse(name_string)

def normalize_name(name_string):
    """
    Normalize name string by removing birth/death years.
    
    Args:
        name_string (str): Name string
        
    Returns:
        str: Normalized name string
    """
    from src.birth_death_regexes import BirthDeathYearExtractor
    extractor = BirthDeathYearExtractor()
    return extractor.normalize_name(name_string)

def timing_decorator(func):
    """Decorator for timing function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logger.debug(f"{func.__name__} completed in {elapsed_time:.2f} seconds")
        return result
    return wrapper
