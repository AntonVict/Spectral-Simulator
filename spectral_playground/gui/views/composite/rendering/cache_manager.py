"""RGB caching system for performance optimization."""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ....state import PlaygroundData


class CacheManager:
    """Manages RGB image caching to avoid redundant renders."""
    
    def __init__(self, max_cache_size: int = 20):
        """Initialize cache manager.
        
        Args:
            max_cache_size: Maximum number of cached RGB images
        """
        self.max_cache_size = max_cache_size
        self._cache_dict: dict = {}  # Cache by channel tuple
        self._current_rgb: Optional[np.ndarray] = None  # Current RGB
        self._current_data_id: Optional[int] = None  # Track dataset instance
    
    def get_cached(self, channel_tuple: tuple) -> Optional[np.ndarray]:
        """Get cached RGB image for channel configuration.
        
        Args:
            channel_tuple: Tuple of channel flags
            
        Returns:
            Cached RGB image or None if not cached
        """
        return self._cache_dict.get(channel_tuple)
    
    def store(self, channel_tuple: tuple, rgb: np.ndarray) -> None:
        """Store RGB image in cache.
        
        Args:
            channel_tuple: Tuple of channel flags
            rgb: RGB image to cache
        """
        # Limit cache size to prevent memory issues
        if len(self._cache_dict) >= self.max_cache_size:
            # Remove oldest entry (simple FIFO strategy)
            self._cache_dict.pop(next(iter(self._cache_dict)))
        
        self._cache_dict[channel_tuple] = rgb
        self._current_rgb = rgb
    
    def clear(self) -> None:
        """Clear all cached RGB images."""
        self._cache_dict.clear()
        self._current_rgb = None
    
    def check_dataset_changed(self, data: 'PlaygroundData') -> bool:
        """Check if dataset has changed (requires cache clear).
        
        Args:
            data: Current playground data
            
        Returns:
            True if dataset changed, False otherwise
        """
        current_data_id = id(data.Y) if data.Y is not None else None
        changed = (self._current_data_id != current_data_id)
        
        if changed:
            self.clear()
            self._current_data_id = current_data_id
        
        return changed
    
    @property
    def current_rgb(self) -> Optional[np.ndarray]:
        """Get the current/latest RGB image."""
        return self._current_rgb
    
    def is_cached(self, channel_tuple: tuple) -> bool:
        """Check if a channel configuration is cached.
        
        Args:
            channel_tuple: Tuple of channel flags
            
        Returns:
            True if cached, False otherwise
        """
        return channel_tuple in self._cache_dict

