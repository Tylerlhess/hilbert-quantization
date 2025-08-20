"""
Dimension calculation utilities for Hilbert curve parameter mapping.

This module implements the DimensionCalculator interface to find optimal
power-of-4 dimensions for parameter mapping and calculate padding strategies.
"""

import math
from typing import Tuple, List
import numpy as np

from ..interfaces import DimensionCalculator
from ..models import PaddingConfig
from ..config import Constants


class PowerOf4DimensionCalculator(DimensionCalculator):
    """
    Calculator for optimal power-of-4 dimensions and padding strategies.
    
    This implementation finds the smallest power of 4 that can accommodate
    the given parameter count and calculates an efficient padding strategy
    to minimize wasted space.
    """
    
    def __init__(self, min_efficiency_ratio: float = Constants.MIN_EFFICIENCY_RATIO):
        """
        Initialize the dimension calculator.
        
        Args:
            min_efficiency_ratio: Minimum acceptable efficiency ratio for padding
        """
        self.min_efficiency_ratio = min_efficiency_ratio
    
    def calculate_optimal_dimensions(self, param_count: int) -> Tuple[int, int]:
        """
        Calculate nearest power-of-4 dimensions that accommodate parameters.
        
        Finds the smallest power of 4 that is >= param_count and returns
        square dimensions (n x n) where n^2 equals that power of 4.
        
        Args:
            param_count: Number of parameters to accommodate
            
        Returns:
            Tuple of (width, height) dimensions as square grid
            
        Raises:
            ValueError: If param_count is not positive
        """
        if param_count <= 0:
            raise ValueError("Parameter count must be positive")
        
        # Find the smallest power of 4 that accommodates param_count
        target_size = self._find_nearest_power_of_4(param_count)
        
        # Calculate square dimensions (since we want n x n grid)
        dimension = int(math.sqrt(target_size))
        
        return (dimension, dimension)
    
    def calculate_padding_strategy(self, param_count: int, target_dims: Tuple[int, int]) -> PaddingConfig:
        """
        Determine optimal padding to minimize wasted space.
        
        Calculates padding positions and efficiency metrics for the given
        parameter count and target dimensions.
        
        Args:
            param_count: Number of parameters
            target_dims: Target dimensions for mapping
            
        Returns:
            PaddingConfig with padding strategy details
            
        Raises:
            ValueError: If target dimensions are invalid or efficiency too low
        """
        width, height = target_dims
        total_space = width * height
        
        if total_space < param_count:
            raise ValueError(f"Target dimensions {target_dims} cannot accommodate {param_count} parameters")
        
        # Calculate efficiency ratio
        efficiency_ratio = param_count / total_space
        
        if efficiency_ratio < self.min_efficiency_ratio:
            raise ValueError(
                f"Efficiency ratio {efficiency_ratio:.3f} is below minimum {self.min_efficiency_ratio}"
            )
        
        # Calculate padding positions (unused space at the end)
        padding_count = total_space - param_count
        padding_positions = self._calculate_padding_positions(param_count, target_dims)
        
        return PaddingConfig(
            target_dimensions=target_dims,
            padding_value=Constants.DEFAULT_PADDING_VALUE,
            padding_positions=padding_positions,
            efficiency_ratio=efficiency_ratio
        )
    
    def _find_nearest_power_of_4(self, value: int) -> int:
        """
        Find the smallest power of 4 that is >= value.
        
        Args:
            value: Target value
            
        Returns:
            Nearest power of 4 >= value
        """
        if value <= 0:
            return 4
        
        # Check against predefined valid dimensions first
        for size in Constants.VALID_DIMENSIONS:
            if size >= value:
                return size
        
        # If larger than predefined, calculate next power of 4
        power = Constants.VALID_DIMENSIONS[-1]  # Start with largest predefined
        while power < value:
            power *= 4
        
        return power
    
    def _calculate_padding_positions(self, param_count: int, dimensions: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Calculate positions where padding should be applied.
        
        Padding is applied at the end of the parameter space in row-major order.
        
        Args:
            param_count: Number of actual parameters
            dimensions: Target dimensions (width, height)
            
        Returns:
            List of (x, y) positions for padding
        """
        width, height = dimensions
        total_space = width * height
        padding_count = total_space - param_count
        
        padding_positions = []
        
        # Calculate padding positions starting from the end in row-major order
        for i in range(padding_count):
            pos_index = total_space - 1 - i
            y = pos_index // width
            x = pos_index % width
            padding_positions.append((x, y))
        
        return padding_positions
    
    def get_efficiency_metrics(self, param_count: int, dimensions: Tuple[int, int]) -> dict:
        """
        Calculate detailed efficiency metrics for given dimensions.
        
        Args:
            param_count: Number of parameters
            dimensions: Target dimensions
            
        Returns:
            Dictionary with efficiency metrics
        """
        width, height = dimensions
        total_space = width * height
        
        return {
            'total_space': total_space,
            'used_space': param_count,
            'wasted_space': total_space - param_count,
            'efficiency_ratio': param_count / total_space,
            'waste_percentage': ((total_space - param_count) / total_space) * 100,
            'dimensions': dimensions
        }
    
    def find_all_valid_dimensions(self, param_count: int, max_waste_percentage: float = 50.0) -> List[Tuple[int, int]]:
        """
        Find all valid power-of-4 dimensions within waste threshold.
        
        Args:
            param_count: Number of parameters
            max_waste_percentage: Maximum acceptable waste percentage
            
        Returns:
            List of valid (width, height) dimension tuples
        """
        valid_dimensions = []
        
        # Check all valid dimensions from Constants.VALID_DIMENSIONS
        for size in Constants.VALID_DIMENSIONS:
            if size >= param_count:
                dimension = int(math.sqrt(size))
                dims = (dimension, dimension)
                
                metrics = self.get_efficiency_metrics(param_count, dims)
                if metrics['waste_percentage'] <= max_waste_percentage:
                    valid_dimensions.append(dims)
        
        return valid_dimensions


def validate_power_of_4(value: int) -> bool:
    """
    Check if a value is a power of 4.
    
    Args:
        value: Value to check
        
    Returns:
        True if value is a power of 4, False otherwise
    """
    if value <= 0:
        return False
    
    while value > 1:
        if value % 4 != 0:
            return False
        value //= 4
    
    return True


def calculate_dimension_efficiency(param_count: int, dimensions: Tuple[int, int]) -> float:
    """
    Calculate efficiency ratio for given dimensions.
    
    Args:
        param_count: Number of parameters
        dimensions: Target dimensions (width, height)
        
    Returns:
        Efficiency ratio (0.0 to 1.0)
    """
    width, height = dimensions
    total_space = width * height
    
    if total_space == 0:
        return 0.0
    
    return min(1.0, param_count / total_space)