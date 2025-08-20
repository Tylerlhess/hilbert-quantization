"""
Padding configuration utilities for parameter mapping.

This module provides utilities for configuring and managing padding strategies
for parameter mapping to power-of-4 dimensions.
"""

from typing import List, Tuple, Dict, Any
import numpy as np

from ..models import PaddingConfig
from ..config import Constants


class PaddingConfigurationSystem:
    """
    System for managing and optimizing padding configurations.
    
    Provides utilities for creating, validating, and optimizing padding
    strategies for different parameter counts and dimension requirements.
    """
    
    def __init__(self):
        """Initialize the padding configuration system."""
        self.padding_strategies = {}
    
    def create_padding_config(
        self, 
        param_count: int, 
        target_dims: Tuple[int, int],
        padding_value: float = Constants.DEFAULT_PADDING_VALUE,
        strategy: str = "end_fill"
    ) -> PaddingConfig:
        """
        Create a padding configuration for given parameters and dimensions.
        
        Args:
            param_count: Number of parameters to accommodate
            target_dims: Target dimensions (width, height)
            padding_value: Value to use for padding positions
            strategy: Padding strategy ("end_fill", "distributed", "corner_fill")
            
        Returns:
            PaddingConfig with calculated padding strategy
            
        Raises:
            ValueError: If parameters are invalid or strategy unknown
        """
        if param_count <= 0:
            raise ValueError("Parameter count must be positive")
        
        width, height = target_dims
        total_space = width * height
        
        if total_space < param_count:
            raise ValueError(f"Target dimensions {target_dims} cannot accommodate {param_count} parameters")
        
        # Calculate efficiency ratio
        efficiency_ratio = param_count / total_space
        
        # Calculate padding positions based on strategy
        padding_positions = self._calculate_padding_positions_by_strategy(
            param_count, target_dims, strategy
        )
        
        return PaddingConfig(
            target_dimensions=target_dims,
            padding_value=padding_value,
            padding_positions=padding_positions,
            efficiency_ratio=efficiency_ratio
        )
    
    def _calculate_padding_positions_by_strategy(
        self, 
        param_count: int, 
        dimensions: Tuple[int, int], 
        strategy: str
    ) -> List[Tuple[int, int]]:
        """
        Calculate padding positions based on the specified strategy.
        
        Args:
            param_count: Number of actual parameters
            dimensions: Target dimensions (width, height)
            strategy: Padding strategy to use
            
        Returns:
            List of (x, y) positions for padding
        """
        width, height = dimensions
        total_space = width * height
        padding_count = total_space - param_count
        
        if strategy == "end_fill":
            return self._end_fill_strategy(param_count, dimensions)
        elif strategy == "distributed":
            return self._distributed_strategy(param_count, dimensions)
        elif strategy == "corner_fill":
            return self._corner_fill_strategy(param_count, dimensions)
        else:
            raise ValueError(f"Unknown padding strategy: {strategy}")
    
    def _end_fill_strategy(self, param_count: int, dimensions: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Fill padding at the end of the parameter space in row-major order."""
        width, height = dimensions
        total_space = width * height
        padding_count = total_space - param_count
        
        padding_positions = []
        for i in range(padding_count):
            pos_index = total_space - 1 - i
            y = pos_index // width
            x = pos_index % width
            padding_positions.append((x, y))
        
        return padding_positions
    
    def _distributed_strategy(self, param_count: int, dimensions: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Distribute padding evenly throughout the parameter space."""
        width, height = dimensions
        total_space = width * height
        padding_count = total_space - param_count
        
        if padding_count == 0:
            return []
        
        # Use a simple approach: distribute padding positions evenly
        # Start from the end and work backwards with even spacing
        padding_positions = []
        used_positions = set()
        
        # Calculate step size for distribution
        step = max(1, total_space // padding_count)
        
        for i in range(padding_count):
            # Calculate position index with even distribution
            pos_index = total_space - 1 - (i * step)
            
            # Ensure we don't go below parameter space and avoid duplicates
            while pos_index < param_count or pos_index in used_positions:
                pos_index -= 1
                if pos_index < 0:
                    break
            
            if pos_index >= 0:
                y = pos_index // width
                x = pos_index % width
                position = (x, y)
                if position not in used_positions:
                    padding_positions.append(position)
                    used_positions.add(pos_index)
        
        # If we don't have enough positions, fill from the end
        while len(padding_positions) < padding_count:
            for pos_index in range(total_space - 1, param_count - 1, -1):
                if pos_index not in used_positions:
                    y = pos_index // width
                    x = pos_index % width
                    padding_positions.append((x, y))
                    used_positions.add(pos_index)
                    if len(padding_positions) >= padding_count:
                        break
            break
        
        return padding_positions[:padding_count]
    
    def _corner_fill_strategy(self, param_count: int, dimensions: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Fill padding starting from corners of the grid."""
        width, height = dimensions
        total_space = width * height
        padding_count = total_space - param_count
        
        if padding_count == 0:
            return []
        
        # Define corner positions in order: bottom-right, bottom-left, top-right, top-left
        corners = [
            (width - 1, height - 1),  # bottom-right
            (0, height - 1),          # bottom-left
            (width - 1, 0),           # top-right
            (0, 0)                    # top-left
        ]
        
        padding_positions = []
        
        # Add corners first
        for i in range(min(padding_count, len(corners))):
            padding_positions.append(corners[i])
        
        # If we need more positions, use end_fill for the rest
        if padding_count > len(corners):
            remaining_count = padding_count - len(corners)
            
            # Get all end positions and filter out corners
            all_end_positions = self._end_fill_strategy(param_count, dimensions)
            corner_set = set(corners)
            
            # Add non-corner positions
            for pos in all_end_positions:
                if pos not in corner_set and len(padding_positions) < padding_count:
                    padding_positions.append(pos)
        
        return padding_positions
    
    def optimize_padding_for_parameter_count(
        self, 
        param_count: int, 
        max_waste_percentage: float = 50.0
    ) -> List[PaddingConfig]:
        """
        Find optimal padding configurations for a given parameter count.
        
        Args:
            param_count: Number of parameters
            max_waste_percentage: Maximum acceptable waste percentage
            
        Returns:
            List of PaddingConfig options sorted by efficiency
        """
        configs = []
        
        # Check all valid dimensions
        for size in Constants.VALID_DIMENSIONS:
            if size >= param_count:
                dimension = int(np.sqrt(size))
                dims = (dimension, dimension)
                
                efficiency = param_count / size
                waste_percentage = ((size - param_count) / size) * 100
                
                if waste_percentage <= max_waste_percentage:
                    try:
                        config = self.create_padding_config(param_count, dims)
                        configs.append(config)
                    except ValueError:
                        # Skip configurations that can't be created
                        continue
        
        # If no configs found with the threshold, try with a more lenient threshold
        if not configs and max_waste_percentage < 100.0:
            return self.optimize_padding_for_parameter_count(param_count, 100.0)
        
        # Sort by efficiency (highest first)
        configs.sort(key=lambda c: c.efficiency_ratio, reverse=True)
        return configs
    
    def compare_padding_strategies(
        self, 
        param_count: int, 
        target_dims: Tuple[int, int]
    ) -> Dict[str, PaddingConfig]:
        """
        Compare different padding strategies for given parameters.
        
        Args:
            param_count: Number of parameters
            target_dims: Target dimensions
            
        Returns:
            Dictionary mapping strategy names to PaddingConfig objects
        """
        strategies = ["end_fill", "distributed", "corner_fill"]
        results = {}
        
        for strategy in strategies:
            try:
                config = self.create_padding_config(param_count, target_dims, strategy=strategy)
                results[strategy] = config
            except ValueError as e:
                # Strategy might not be applicable for these parameters
                results[strategy] = None
        
        return results
    
    def validate_padding_config(self, config: PaddingConfig) -> Dict[str, Any]:
        """
        Validate a padding configuration and return validation results.
        
        Args:
            config: PaddingConfig to validate
            
        Returns:
            Dictionary with validation results and metrics
        """
        width, height = config.target_dimensions
        total_space = width * height
        
        # Check for position conflicts
        position_conflicts = []
        unique_positions = set(config.padding_positions)
        
        if len(unique_positions) != len(config.padding_positions):
            position_conflicts.append("Duplicate padding positions found")
        
        # Check position bounds
        out_of_bounds = []
        for x, y in config.padding_positions:
            if x < 0 or x >= width or y < 0 or y >= height:
                out_of_bounds.append((x, y))
        
        # Calculate actual efficiency
        used_space = total_space - len(config.padding_positions)
        actual_efficiency = used_space / total_space if total_space > 0 else 0
        
        return {
            "is_valid": len(position_conflicts) == 0 and len(out_of_bounds) == 0,
            "position_conflicts": position_conflicts,
            "out_of_bounds_positions": out_of_bounds,
            "total_space": total_space,
            "used_space": used_space,
            "padding_space": len(config.padding_positions),
            "actual_efficiency": actual_efficiency,
            "declared_efficiency": config.efficiency_ratio,
            "efficiency_match": abs(actual_efficiency - config.efficiency_ratio) < 1e-6
        }
    
    def get_padding_statistics(self, configs: List[PaddingConfig]) -> Dict[str, Any]:
        """
        Calculate statistics for a list of padding configurations.
        
        Args:
            configs: List of PaddingConfig objects
            
        Returns:
            Dictionary with statistical information
        """
        if not configs:
            return {"count": 0}
        
        efficiencies = [c.efficiency_ratio for c in configs]
        padding_counts = [len(c.padding_positions) for c in configs]
        
        return {
            "count": len(configs),
            "efficiency_stats": {
                "min": min(efficiencies),
                "max": max(efficiencies),
                "mean": np.mean(efficiencies),
                "std": np.std(efficiencies)
            },
            "padding_count_stats": {
                "min": min(padding_counts),
                "max": max(padding_counts),
                "mean": np.mean(padding_counts),
                "std": np.std(padding_counts)
            },
            "best_efficiency_config": max(configs, key=lambda c: c.efficiency_ratio),
            "worst_efficiency_config": min(configs, key=lambda c: c.efficiency_ratio)
        }


def create_optimal_padding_config(param_count: int) -> PaddingConfig:
    """
    Create the most efficient padding configuration for a parameter count.
    
    Args:
        param_count: Number of parameters
        
    Returns:
        PaddingConfig with optimal efficiency
    """
    system = PaddingConfigurationSystem()
    configs = system.optimize_padding_for_parameter_count(param_count)
    
    if not configs:
        raise ValueError(f"No valid padding configuration found for {param_count} parameters")
    
    return configs[0]  # Return the most efficient one


def analyze_padding_efficiency(param_counts: List[int]) -> Dict[int, Dict[str, Any]]:
    """
    Analyze padding efficiency for multiple parameter counts.
    
    Args:
        param_counts: List of parameter counts to analyze
        
    Returns:
        Dictionary mapping parameter counts to efficiency analysis
    """
    system = PaddingConfigurationSystem()
    results = {}
    
    for param_count in param_counts:
        try:
            configs = system.optimize_padding_for_parameter_count(param_count)
            stats = system.get_padding_statistics(configs)
            results[param_count] = {
                "configs": configs,
                "statistics": stats,
                "optimal_config": configs[0] if configs else None
            }
        except Exception as e:
            results[param_count] = {
                "error": str(e),
                "configs": [],
                "statistics": {"count": 0}
            }
    
    return results