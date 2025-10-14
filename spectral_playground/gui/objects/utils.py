"""Utility functions for fluorophore name/index conversions."""

from typing import List, Callable


def get_fluorophore_list(callback: Callable[[], List[str]]) -> List[str]:
    """Get fluorophore names from callback.
    
    Args:
        callback: Function that returns list of fluorophore names
        
    Returns:
        List of fluorophore names
    """
    try:
        return callback()
    except Exception:
        return []


def fluorophore_name_to_index(name: str, fluor_list: List[str]) -> int:
    """Convert fluorophore name to 0-indexed integer.
    
    Args:
        name: Fluorophore name (e.g., "F1", "Alexa488")
        fluor_list: List of fluorophore names
        
    Returns:
        0-based index of the fluorophore
    """
    try:
        if name in fluor_list:
            return fluor_list.index(name)
        # Fallback: try to extract number from "FX" format
        if name.startswith('F') and name[1:].isdigit():
            return int(name[1:]) - 1
        return 0
    except (ValueError, IndexError):
        return 0


def fluorophore_index_to_name(index: int, fluor_list: List[str]) -> str:
    """Convert fluorophore index to name.
    
    Args:
        index: 0-based fluorophore index
        fluor_list: List of fluorophore names
        
    Returns:
        Fluorophore name or generic "FX" format
    """
    try:
        if 0 <= index < len(fluor_list):
            return fluor_list[index]
        return f'F{index + 1}'
    except Exception:
        return f'F{index + 1}'

