"""
Streaming hierarchical index builder using sliding windows and counters.

This module implements an efficient streaming approach to building hierarchical
indices using sliding windows of size 4 and counter-based level promotion.
"""

from typing import List, Optional
import numpy as np


class StreamingIndexBuilder:
    """
    Efficient streaming hierarchical index builder.
    
    This class builds hierarchical indices incrementally as values are added,
    using sliding windows of size 4 and counter-based promotion to higher levels.
    Only maintains the last 4 values at each level, making it very memory efficient.
    """
    
    def __init__(self, max_levels: int = 10):
        """
        Initialize streaming index builder.
        
        Args:
            max_levels: Maximum number of hierarchical levels to maintain
        """
        self.max_levels = max_levels
        self.reset()
    
    def reset(self) -> None:
        """Reset the builder to initial state."""
        # Each level maintains a sliding window of up to 4 values
        self.level_windows: List[List[float]] = [[] for _ in range(self.max_levels)]
        
        # Counter for each level to track when to promote to next level
        self.level_counters: List[int] = [0] * self.max_levels
        
        # Store all computed indices for final extraction
        self.all_indices: List[List[float]] = [[] for _ in range(self.max_levels)]
        
        # Track total values processed
        self.total_values = 0
    
    def add_value(self, value: float) -> None:
        """
        Add a value to the streaming index builder.
        
        This method adds the value to level 0 and propagates averages
        up the hierarchy when windows of 4 are completed.
        
        Args:
            value: The value to add
        """
        self.total_values += 1
        self._add_to_level_optimized(0, float(value))
    
    def add_batch(self, values: np.ndarray) -> None:
        """
        Add a batch of values efficiently.
        
        This method processes multiple values at once for better performance.
        
        Args:
            values: Array of values to add
        """
        for value in values:
            self.add_value(float(value))
    
    def _add_to_level_optimized(self, level: int, value: float) -> None:
        """
        Optimized version of add_to_level with reduced overhead.
        
        Args:
            level: The level to add the value to
            value: The value to add
        """
        if level >= self.max_levels:
            return  # Reached maximum depth
        
        # Add value to the sliding window for this level
        window = self.level_windows[level]
        window.append(value)
        self.level_counters[level] += 1
        
        # Store the value in the indices for this level
        self.all_indices[level].append(value)
        
        # Keep only the last 4 values in the sliding window
        if len(window) > 4:
            window.pop(0)
        
        # Check if we have 4 values to promote to next level
        if self.level_counters[level] % 4 == 0 and len(window) == 4:
            # Calculate average of the 4 values (faster than sum/len)
            average = (window[0] + window[1] + window[2] + window[3]) * 0.25
            
            # Promote to next level
            self._add_to_level_optimized(level + 1, average)
            
            # Clear the sliding window for this level (start fresh)
            window.clear()
    
    def _add_to_level(self, level: int, value: float) -> None:
        """
        Add a value to a specific level and handle promotion.
        
        Args:
            level: The level to add the value to
            value: The value to add
        """
        # Delegate to optimized version
        self._add_to_level_optimized(level, value)
    
    def get_indices_by_level(self, level: int) -> List[float]:
        """
        Get all indices computed for a specific level.
        
        Args:
            level: The level to get indices for
            
        Returns:
            List of indices for the specified level
        """
        if 0 <= level < self.max_levels:
            return self.all_indices[level].copy()
        return []
    
    def get_all_indices_flattened(self, max_count: Optional[int] = None) -> np.ndarray:
        """
        Get all indices from all levels flattened into a single array.
        
        Args:
            max_count: Maximum number of indices to return (None for all)
            
        Returns:
            Flattened array of all indices
        """
        all_indices = []
        
        # Collect indices from all levels
        for level in range(self.max_levels):
            all_indices.extend(self.all_indices[level])
        
        # Convert to numpy array
        result = np.array(all_indices)
        
        # Limit to max_count if specified
        if max_count is not None and len(result) > max_count:
            result = result[:max_count]
        
        return result
    
    def get_hierarchical_indices(self, index_space_size: int) -> np.ndarray:
        """
        Get hierarchical indices with proper space allocation.
        
        This method allocates space among different levels and returns
        a properly sized array of hierarchical indices.
        
        Args:
            index_space_size: Total space available for indices
            
        Returns:
            Array of hierarchical indices
        """
        if index_space_size <= 0:
            return np.array([])
        
        # Calculate how much space to allocate to each level
        level_allocations = self._calculate_level_allocations(index_space_size)
        
        # Collect indices from each level according to allocation
        final_indices = []
        
        for level, allocation in enumerate(level_allocations):
            if allocation <= 0 or level >= self.max_levels:
                continue
            
            level_indices = self.all_indices[level]
            
            # Take up to the allocated amount from this level
            if len(level_indices) > 0:
                # Use evenly spaced sampling if we have more indices than allocation
                if len(level_indices) > allocation:
                    # Sample evenly across the available indices
                    step = len(level_indices) / allocation
                    sampled_indices = [level_indices[int(i * step)] for i in range(allocation)]
                    final_indices.extend(sampled_indices)
                else:
                    # Use all available indices from this level
                    final_indices.extend(level_indices)
        
        # Convert to numpy array and ensure proper size
        result = np.array(final_indices)
        
        # Pad or truncate to exact size
        if len(result) < index_space_size:
            padded_result = np.zeros(index_space_size)
            padded_result[:len(result)] = result
            result = padded_result
        elif len(result) > index_space_size:
            result = result[:index_space_size]
        
        return result
    
    def _calculate_level_allocations(self, total_space: int) -> List[int]:
        """
        Calculate space allocation for each level.
        
        Args:
            total_space: Total available space
            
        Returns:
            List of space allocations for each level
        """
        allocations = [0] * self.max_levels
        
        # Count non-empty levels
        non_empty_levels = []
        for level in range(self.max_levels):
            if len(self.all_indices[level]) > 0:
                non_empty_levels.append(level)
        
        if not non_empty_levels:
            return allocations
        
        # Allocate space with preference for finer levels (lower level numbers)
        remaining_space = total_space
        
        for i, level in enumerate(non_empty_levels):
            if i == len(non_empty_levels) - 1:
                # Last level gets remaining space
                allocations[level] = remaining_space
            else:
                # Allocate decreasing fractions: 50%, 25%, 12.5%, etc.
                fraction = 0.5 ** (i + 1)
                allocated = max(1, int(total_space * fraction))
                allocated = min(allocated, remaining_space)
                allocations[level] = allocated
                remaining_space -= allocated
        
        return allocations
    
    def get_statistics(self) -> dict:
        """
        Get statistics about the current state of the builder.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_values_processed': self.total_values,
            'levels_used': 0,
            'indices_per_level': {},
            'current_window_sizes': {},
            'level_counters': self.level_counters.copy()
        }
        
        for level in range(self.max_levels):
            if len(self.all_indices[level]) > 0:
                stats['levels_used'] = level + 1
                stats['indices_per_level'][level] = len(self.all_indices[level])
                stats['current_window_sizes'][level] = len(self.level_windows[level])
        
        return stats
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (f"StreamingIndexBuilder(total_values={self.total_values}, "
                f"levels_used={len([l for l in self.all_indices if len(l) > 0])})")


class StreamingHilbertIndexGenerator:
    """
    Hilbert curve index generator using streaming approach.
    
    This class integrates the streaming index builder with Hilbert curve mapping
    to provide efficient hierarchical index generation during parameter mapping.
    """
    
    def __init__(self):
        """Initialize the streaming Hilbert index generator."""
        from .hilbert_mapper import HilbertCurveMapper
        self.hilbert_mapper = HilbertCurveMapper()
    
    def generate_indices_during_mapping(self, parameters: np.ndarray, 
                                      dimensions: tuple, 
                                      index_space_size: int) -> tuple:
        """
        Generate both 2D mapping and hierarchical indices simultaneously.
        
        Args:
            parameters: 1D array of parameters
            dimensions: Target 2D dimensions (width, height)
            index_space_size: Available space for indices
            
        Returns:
            Tuple of (2D_image, hierarchical_indices, builder_stats)
        """
        # Create streaming index builder
        builder = StreamingIndexBuilder()
        
        # Perform 2D mapping while building indices
        image_2d = self.hilbert_mapper.map_to_2d(parameters, dimensions, builder)
        
        # Extract hierarchical indices
        hierarchical_indices = builder.get_hierarchical_indices(index_space_size)
        
        # Get statistics
        stats = builder.get_statistics()
        
        return image_2d, hierarchical_indices, stats
    
    def generate_optimized_indices(self, image: np.ndarray, index_space_size: int) -> np.ndarray:
        """
        Generate hierarchical indices from 2D image using streaming approach.
        
        Args:
            image: 2D parameter representation
            index_space_size: Available space for indices
            
        Returns:
            Array of hierarchical indices
        """
        height, width = image.shape
        
        # Verify image is suitable for Hilbert curve mapping
        if width != height or width <= 0 or (width & (width - 1)) != 0:
            raise ValueError(f"Image must be square with power-of-2 dimensions, got {width}x{height}")
        
        # Extract parameters in Hilbert order
        parameters = self.hilbert_mapper.map_from_2d(image)
        
        # Create streaming index builder
        builder = StreamingIndexBuilder()
        
        # Add all parameters to the builder
        for param in parameters:
            builder.add_value(float(param))
        
        # Extract hierarchical indices
        return builder.get_hierarchical_indices(index_space_size)