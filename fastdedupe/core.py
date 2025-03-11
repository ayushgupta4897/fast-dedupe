"""
Core functionality for fast-dedupe.

This module contains the main deduplication function that leverages RapidFuzz
for high-performance fuzzy string matching.
"""

from typing import Dict, List, Tuple, Union
from rapidfuzz import process, fuzz


def dedupe(
    data: List[str], 
    threshold: int = 85, 
    keep_first: bool = True
) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Deduplicate a list of strings using fuzzy matching.
    
    This function identifies and removes duplicate strings from the input list
    based on a similarity threshold. It uses the RapidFuzz library for fast
    fuzzy string matching.
    
    Args:
        data (List[str]): List of strings to deduplicate.
        threshold (int, optional): Similarity threshold (0-100). Higher values
            require more similarity to consider strings as duplicates.
            Default is 85.
        keep_first (bool, optional): If True, keeps the first occurrence of a
            duplicate. If False, keeps the longest string. Default is True.
    
    Returns:
        Tuple[List[str], Dict[str, List[str]]]: A tuple containing:
            - List of deduplicated strings
            - Dictionary mapping each kept string to its duplicates
    
    Examples:
        >>> data = ["Apple iPhone 12", "Apple iPhone12", "Samsung Galaxy"]
        >>> clean, dupes = dedupe(data, threshold=85)
        >>> print(clean)
        ['Apple iPhone 12', 'Samsung Galaxy']
        >>> print(dupes)
        {'Apple iPhone 12': ['Apple iPhone12']}
    """
    if not data:
        return [], {}
    
    # Validate input parameters
    if not isinstance(threshold, int) or not 0 <= threshold <= 100:
        raise ValueError("Threshold must be an integer between 0 and 100")
    
    if not isinstance(keep_first, bool):
        raise ValueError("keep_first must be a boolean")
    
    # Special case for threshold=100 (exact matches only)
    if threshold == 100:
        return _dedupe_exact(data, keep_first)
    
    # Create a copy of the input data to avoid modifying the original
    data_copy = data.copy()
    
    # Dictionary to store deduplicated results
    # Keys are the kept strings, values are lists of their duplicates
    duplicates_map: Dict[str, List[str]] = {}
    
    # List to store deduplicated strings
    clean_data: List[str] = []
    
    # Process each string in the input data
    while data_copy:
        # Get the current string to process
        current = data_copy.pop(0)
        
        # If we're keeping the first occurrence, add it to the clean data
        if keep_first:
            clean_data.append(current)
            duplicates_map[current] = [current]
        
        # Find all strings in the remaining data that match the current string
        # above the threshold
        matches = process.extract(
            current, 
            data_copy, 
            scorer=fuzz.ratio, 
            score_cutoff=threshold
        )
        
        # If we're not keeping the first occurrence, find the longest string
        # among the current string and its matches
        if not keep_first:
            all_matches = [current] + [match[0] for match in matches]
            longest = max(all_matches, key=len)
            clean_data.append(longest)
            duplicates_map[longest] = all_matches
        else:
            # Add the matches to the duplicates map
            duplicates_map[current].extend([match[0] for match in matches])
        
        # Remove the matches from the data copy
        for match in matches:
            data_copy.remove(match[0])
    
    # Remove self-references from the duplicates map
    for key in duplicates_map:
        if key in duplicates_map[key]:
            duplicates_map[key].remove(key)
        
        # If there are no duplicates for this key, remove the empty list
        if not duplicates_map[key]:
            duplicates_map[key] = []
    
    # Remove keys with empty lists from the duplicates map
    duplicates_map = {k: v for k, v in duplicates_map.items() if v}
    
    return clean_data, duplicates_map


def _dedupe_exact(
    data: List[str], 
    keep_first: bool = True
) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Deduplicate a list of strings using exact matching.
    
    This is a helper function for dedupe() when threshold=100.
    
    Args:
        data (List[str]): List of strings to deduplicate.
        keep_first (bool, optional): If True, keeps the first occurrence of a
            duplicate. If False, keeps the longest string. Default is True.
    
    Returns:
        Tuple[List[str], Dict[str, List[str]]]: A tuple containing:
            - List of deduplicated strings
            - Dictionary mapping each kept string to its duplicates
    """
    # Create a copy of the input data to avoid modifying the original
    data_copy = data.copy()
    
    # Dictionary to store deduplicated results
    # Keys are the kept strings, values are lists of their duplicates
    duplicates_map: Dict[str, List[str]] = {}
    
    # List to store deduplicated strings
    clean_data: List[str] = []
    
    # Dictionary to track seen strings
    seen: Dict[str, str] = {}
    
    for item in data_copy:
        if item in seen:
            # This is a duplicate
            duplicates_map.setdefault(seen[item], []).append(item)
        else:
            # This is a new string
            clean_data.append(item)
            seen[item] = item
    
    return clean_data, duplicates_map 