"""
Custom implementation of a memory-mapped dictionary.

This module provides a simple memory-mapped dictionary for efficient storage
of large data structures with persistence to disk.
"""

import os
import mmap
import json
import pickle
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class MMapDict:
    """
    A simple memory-mapped dictionary implementation.
    
    This class provides a dictionary-like interface backed by a memory-mapped
    file for efficient storage of large data structures.
    """
    
    def __init__(self, filename, max_size=1024*1024*1024):  # Default 1GB
        """
        Initialize a memory-mapped dictionary.
        
        Args:
            filename (str or Path): Path to backing file
            max_size (int, optional): Maximum size in bytes. Defaults to 1GB.
        """
        self.filename = Path(filename)
        self.max_size = max_size
        self.data = {}
        self._dirty = False
        
        # Create directory if it doesn't exist
        self.filename.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing data if file exists
        if self.filename.exists() and self.filename.stat().st_size > 0:
            self._load()
        else:
            # Create empty file
            with open(self.filename, 'wb') as f:
                f.write(b'\0' * 1024)  # Initialize with small empty file
    
    def __getitem__(self, key):
        """Get item by key."""
        return self.data[key]
    
    def __setitem__(self, key, value):
        """Set item by key."""
        self.data[key] = value
        self._dirty = True
    
    def __delitem__(self, key):
        """Delete item by key."""
        del self.data[key]
        self._dirty = True
    
    def __contains__(self, key):
        """Check if key exists."""
        return key in self.data
    
    def __len__(self):
        """Get number of items."""
        return len(self.data)
    
    def __iter__(self):
        """Iterate over keys."""
        return iter(self.data)
    
    def keys(self):
        """Get dictionary keys."""
        return self.data.keys()
    
    def values(self):
        """Get dictionary values."""
        return self.data.values()
    
    def items(self):
        """Get dictionary items."""
        return self.data.items()
    
    def get(self, key, default=None):
        """Get item with default."""
        return self.data.get(key, default)
    
    def update(self, other_dict):
        """Update dictionary with key-value pairs from other_dict."""
        for key, value in other_dict.items():
            self.data[key] = value
        self._dirty = True
    
    def _load(self):
        """Load data from backing file."""
        try:
            with open(self.filename, 'rb') as f:
                self.data = pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading memory-mapped dictionary from {self.filename}: {e}")
            self.data = {}
    
    def flush(self):
        """Flush data to backing file."""
        if not self._dirty:
            return
        
        try:
            with open(self.filename, 'wb') as f:
                pickle.dump(self.data, f)
            self._dirty = False
        except Exception as e:
            logger.error(f"Error flushing memory-mapped dictionary to {self.filename}: {e}")
    
    def to_dict(self):
        """
        Convert MMapDict to regular dictionary for serialization.
        
        Returns:
            dict: Regular dictionary containing the same data
        """
        result = {}
        for key in self.keys():
            value = self[key]
            if hasattr(value, 'to_dict') and callable(getattr(value, 'to_dict')):
                result[key] = value.to_dict()
            elif isinstance(value, dict):
                # Handle nested dictionaries
                result[key] = {k: v for k, v in value.items()}
            else:
                result[key] = value
        return result
    
    def close(self):
        """Flush data and close."""
        self.flush()
    
    def __del__(self):
        """Destructor."""
        self.close()