"""
Regular expressions for matching birth and/or death years in name strings.

This module provides functions and patterns for extracting birth and death years
from person name strings, which is crucial for entity disambiguation.
"""

import re
import logging

logger = logging.getLogger(__name__)

def _compile_birth_death_pattern(patterns=None):
    """
    Compile regular expressions for birth-death year pattern matching.
    
    Args:
        patterns (list, optional): Additional patterns to include. Defaults to None.
        
    Returns:
        list: Compiled regular expression patterns
    """
    # Initialize patterns list if not provided
    if patterns is None:
        patterns = []
    
    # Pattern 1: Birth year with approximate death year - "565 - approximately 665"
    patterns.append(r'(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)\s*[-–—]\s*(?:approximately|ca\.|circa)\s+(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)')
    
    # Pattern 2: Approximate birth and death years
    patterns.append(r'(?:approximately|ca\.|circa)\s+(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)\s*[-–—]\s*(?:approximately|ca\.|circa)\s+(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)')
    
    # Pattern 3: Approximate birth with standard death
    patterns.append(r'(?:approximately|ca\.|circa)\s+(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)\s*[-–—]\s*(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)')
    
    # Pattern 4: Standard birth-death range
    patterns.append(r'(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)\s*[-–—]\s*(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)')
    
    # Pattern 5: Death year only with approximate marker
    patterns.append(r'[-–—]\s*(?:approximately|ca\.|circa)\s+(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)')
    
    # Pattern 6: Death year only (simple)
    patterns.append(r'[-–—]\s*(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)')
    
    # Pattern 7: Birth year only with approximate marker
    patterns.append(r'(?:approximately|ca\.|circa)\s+(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)\s*[-–—]')
    
    # Pattern 8: Birth year only (simple)
    patterns.append(r'(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)\s*[-–—]')
    
    # Pattern 9: Explicit birth/death prefixes
    patterns.append(r'(?:b\.|born)\s+(?:(?:approximately|ca\.|circa)\s+)?(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)|(?:d\.|died)\s+(?:(?:approximately|ca\.|circa)\s+)?(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)')
    
    # Pattern 10: Single approximate year (fallback)
    patterns.append(r'(?:approximately|ca\.|circa)\s+(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)')
    
    # Pattern 11: Years in parentheses - "(1900-1980)"
    patterns.append(r'\(\s*(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)\s*[-–—]\s*(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)\s*\)')
    
    # Pattern 12: Birth year only in parentheses - "(1900-)"
    patterns.append(r'\(\s*(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)\s*[-–—]\s*\)')
    
    # Pattern 13: Death year only in parentheses - "(-1980)"
    patterns.append(r'\(\s*[-–—]\s*(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)\s*\)')
    
    # Pattern 14: Years with fl. (floruit) - "fl. 1500-1550"
    patterns.append(r'(?:fl\.|floruit)\s+(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)\s*[-–—]\s*(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)')
    
    # Pattern 15: Single year with fl. (floruit) - "fl. 1500"
    patterns.append(r'(?:fl\.|floruit)\s+(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)')
    
    # Compile all patterns
    compiled_patterns = [re.compile(pattern) for pattern in patterns]
    
    return compiled_patterns

def extract_birth_death_years(name_string):
    """
    Extract birth and death years from a person name string.
    
    Args:
        name_string (str): Person name string
        
    Returns:
        tuple: (birth_year, death_year) or (None, None) if not found
    """
    if not name_string:
        return None, None
    
    # Compile patterns
    patterns = _compile_birth_death_pattern()
    
    # Try each pattern
    for pattern in patterns:
        match = pattern.search(name_string)
        if match:
            groups = match.groups()
            birth_year = None
            death_year = None
            
            # Extract years from match groups
            if len(groups) >= 2 and groups[0] and groups[1]:
                # Full birth-death range
                birth_year = _clean_year(groups[0])
                death_year = _clean_year(groups[1])
            elif len(groups) >= 1:
                if "born" in name_string.lower() or "b." in name_string.lower():
                    # Birth year only
                    birth_year = _clean_year(groups[0])
                elif "died" in name_string.lower() or "d." in name_string.lower() or "-" in name_string:
                    # Death year only
                    death_year = _clean_year(groups[0])
                elif "fl." in name_string.lower() or "floruit" in name_string.lower():
                    # Floruit (active period) - use as birth year approximation
                    birth_year = _clean_year(groups[0])
            
            return birth_year, death_year
    
    return None, None

def _clean_year(year_str):
    """
    Clean and validate a year string.
    
    Args:
        year_str (str): Year string
        
    Returns:
        int or None: Cleaned year as integer, or None if invalid
    """
    if not year_str:
        return None
    
    # Extract digits only
    digits = re.findall(r'\d+', year_str)
    if not digits:
        return None
    
    # Get first group of digits
    year = int(digits[0])
    
    # Validate year (basic sanity check)
    #if 100 <= year <= 2050:
    if year:
        return year
    
    return None

class BirthDeathYearExtractor:
    """
    Utility class for extracting birth and death years from name strings.
    This is the standardized implementation to be used throughout the codebase.
    """
    
    def __init__(self):
        """Initialize the extractor with compiled patterns."""
        self.patterns = _compile_birth_death_pattern()
    
    def parse(self, name_string):
        """
        Extract birth and death years from a name string.
        
        Args:
            name_string (str): Name string potentially containing birth/death years
            
        Returns:
            tuple: (birth_year, death_year) or (None, None) if not found
        """
        return extract_birth_death_years(name_string)
    
    def has_years(self, name_string):
        """
        Check if a name string contains any birth or death years.
        
        Args:
            name_string (str): Name string to check
            
        Returns:
            bool: True if years are present, False otherwise
        """
        birth_year, death_year = self.parse(name_string)
        return birth_year is not None or death_year is not None
    
    def has_both_years(self, name_string):
        """
        Check if a name string contains both birth and death years.
        
        Args:
            name_string (str): Name string to check
            
        Returns:
            bool: True if both years are present, False otherwise
        """
        birth_year, death_year = self.parse(name_string)
        return birth_year is not None and death_year is not None
    
    def normalize_name(self, name_string):
        """
        Remove birth/death years from a name string.
        
        Args:
            name_string (str): Name string to normalize
            
        Returns:
            str: Normalized name string without birth/death years
        """
        if not name_string:
            return ""
        
        # Remove patterns with birth/death years
        normalized_name = name_string
        
        for pattern in self.patterns:
            normalized_name = pattern.sub('', normalized_name)
        
        # Remove parentheses with years
        normalized_name = re.sub(r'\(\s*\d{2,4}\s*[-–—]?\s*\d{0,4}\s*\)', '', normalized_name)
        
        # Remove trailing commas and whitespace
        normalized_name = re.sub(r',\s*$', '', normalized_name.strip())
        
        # Normalize whitespace
        normalized_name = re.sub(r'\s+', ' ', normalized_name).strip()
        
        return normalized_name
