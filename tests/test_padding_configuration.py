"""
Unit tests for padding configuration system.

Tests the PaddingConfigurationSystem implementation including different
padding strategies, optimization, and validation.
"""

import pytest
import numpy as np
from typing import List, Dict, Any

from hilbert_quantization.utils.padding import (
    PaddingConfigurationSystem,
    create_optimal_padding_config,
    analyze_padding_efficiency
)
from hilbert_quantization.models import PaddingConfig
from hilbert_quantization.config import Constants


class TestPaddingConfigurationSystem:
    """Test cases for PaddingConfigurationSystem."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.system = PaddingConfigurationSystem()
    
    def test_create_padding_config_end_fill(self):
        """Test creating padding config with end_fill strategy."""
        param_count = 10
        target_dims = (4, 4)
        
        config = self.system.create_padding_config(param_count, target_dims, strategy="end_fill")
        
        assert config.target_dimensions == target_dims
        assert config.efficiency_ratio == 10/16
        assert len(config.padding_positions) == 6
        
        # Check that positions are at the end
        expected_positions = [(3, 3), (2, 3), (1, 3), (0, 3), (3, 2), (2, 2)]
        assert config.padding_positions == expected_positions
    
    def test_create_padding_config_distributed(self):
        """Test creating padding config with distributed strategy."""
        param_count = 12
        target_dims = (4, 4)
        
        config = self.system.create_padding_config(param_count, target_dims, strategy="distributed")
        
        assert config.target_dimensions == target_dims
        assert config.efficiency_ratio == 12/16
        assert len(config.padding_positions) == 4
        
        # Positions should be distributed throughout the space
        assert len(set(config.padding_positions)) == len(config.padding_positions)  # No duplicates
    
    def test_create_padding_config_corner_fill(self):
        """Test creating padding config with corner_fill strategy."""
        param_count = 12
        target_dims = (4, 4)
        
        config = self.system.create_padding_config(param_count, target_dims, strategy="corner_fill")
        
        assert config.target_dimensions == target_dims
        assert config.efficiency_ratio == 12/16
        assert len(config.padding_positions) == 4
        
        # Should start with corners
        expected_corners = [(3, 3), (0, 3), (3, 0), (0, 0)]
        assert config.padding_positions == expected_corners
    
    def test_create_padding_config_invalid_strategy(self):
        """Test creating padding config with invalid strategy."""
        with pytest.raises(ValueError, match="Unknown padding strategy"):
            self.system.create_padding_config(10, (4, 4), strategy="invalid_strategy")
    
    def test_create_padding_config_invalid_params(self):
        """Test creating padding config with invalid parameters."""
        # Zero parameter count
        with pytest.raises(ValueError, match="Parameter count must be positive"):
            self.system.create_padding_config(0, (4, 4))
        
        # Negative parameter count
        with pytest.raises(ValueError, match="Parameter count must be positive"):
            self.system.create_padding_config(-5, (4, 4))
        
        # Dimensions too small
        with pytest.raises(ValueError, match="cannot accommodate"):
            self.system.create_padding_config(20, (2, 2))
    
    def test_optimize_padding_for_parameter_count(self):
        """Test optimization of padding configurations."""
        param_count = 200  # Use a parameter count that fits better in 256 (16x16)
        
        configs = self.system.optimize_padding_for_parameter_count(param_count, max_waste_percentage=50.0)
        
        # Should return configurations sorted by efficiency
        assert len(configs) > 0
        
        # Check that efficiencies are in descending order
        efficiencies = [c.efficiency_ratio for c in configs]
        assert efficiencies == sorted(efficiencies, reverse=True)
        
        # All configs should meet waste threshold
        for config in configs:
            width, height = config.target_dimensions
            total_space = width * height
            waste_percentage = ((total_space - param_count) / total_space) * 100
            assert waste_percentage <= 50.0
    
    def test_optimize_padding_strict_threshold(self):
        """Test optimization with strict waste threshold."""
        param_count = 250  # Use a parameter count close to 256 for better efficiency
        
        # Very strict threshold should return fewer or no results
        strict_configs = self.system.optimize_padding_for_parameter_count(param_count, max_waste_percentage=5.0)
        
        # All returned configs should meet the strict threshold
        for config in strict_configs:
            width, height = config.target_dimensions
            total_space = width * height
            waste_percentage = ((total_space - param_count) / total_space) * 100
            assert waste_percentage <= 5.0
    
    def test_compare_padding_strategies(self):
        """Test comparison of different padding strategies."""
        param_count = 10
        target_dims = (4, 4)
        
        results = self.system.compare_padding_strategies(param_count, target_dims)
        
        # Should have results for all strategies
        expected_strategies = ["end_fill", "distributed", "corner_fill"]
        for strategy in expected_strategies:
            assert strategy in results
            assert results[strategy] is not None
            assert isinstance(results[strategy], PaddingConfig)
        
        # All should have same efficiency but different positions
        efficiencies = [results[s].efficiency_ratio for s in expected_strategies]
        assert all(abs(e - efficiencies[0]) < 1e-6 for e in efficiencies)
        
        # Positions should be different
        positions = [set(results[s].padding_positions) for s in expected_strategies]
        assert len(set(tuple(sorted(p)) for p in positions)) > 1  # At least some should be different
    
    def test_validate_padding_config_valid(self):
        """Test validation of a valid padding configuration."""
        config = PaddingConfig(
            target_dimensions=(4, 4),
            padding_value=0.0,
            padding_positions=[(3, 3), (2, 3)],
            efficiency_ratio=14/16
        )
        
        validation = self.system.validate_padding_config(config)
        
        assert validation["is_valid"] == True
        assert len(validation["position_conflicts"]) == 0
        assert len(validation["out_of_bounds_positions"]) == 0
        assert validation["total_space"] == 16
        assert validation["used_space"] == 14
        assert validation["padding_space"] == 2
        assert validation["efficiency_match"] == True
    
    def test_validate_padding_config_invalid(self):
        """Test validation of invalid padding configurations."""
        # Config with out-of-bounds positions
        invalid_config = PaddingConfig(
            target_dimensions=(4, 4),
            padding_value=0.0,
            padding_positions=[(5, 3), (2, 5)],  # Out of bounds
            efficiency_ratio=14/16
        )
        
        validation = self.system.validate_padding_config(invalid_config)
        
        assert validation["is_valid"] == False
        assert len(validation["out_of_bounds_positions"]) == 2
        assert (5, 3) in validation["out_of_bounds_positions"]
        assert (2, 5) in validation["out_of_bounds_positions"]
    
    def test_validate_padding_config_duplicates(self):
        """Test validation with duplicate positions."""
        duplicate_config = PaddingConfig(
            target_dimensions=(4, 4),
            padding_value=0.0,
            padding_positions=[(3, 3), (3, 3), (2, 3)],  # Duplicate (3,3)
            efficiency_ratio=13/16
        )
        
        validation = self.system.validate_padding_config(duplicate_config)
        
        assert validation["is_valid"] == False
        assert "Duplicate padding positions found" in validation["position_conflicts"]
    
    def test_get_padding_statistics(self):
        """Test calculation of padding statistics."""
        configs = [
            PaddingConfig((4, 4), 0.0, [(3, 3)], 15/16),
            PaddingConfig((4, 4), 0.0, [(3, 3), (2, 3)], 14/16),
            PaddingConfig((4, 4), 0.0, [(3, 3), (2, 3), (1, 3)], 13/16)
        ]
        
        stats = self.system.get_padding_statistics(configs)
        
        assert stats["count"] == 3
        assert stats["efficiency_stats"]["min"] == 13/16
        assert stats["efficiency_stats"]["max"] == 15/16
        assert abs(stats["efficiency_stats"]["mean"] - 14/16) < 1e-6
        
        assert stats["padding_count_stats"]["min"] == 1
        assert stats["padding_count_stats"]["max"] == 3
        assert stats["padding_count_stats"]["mean"] == 2.0
        
        assert stats["best_efficiency_config"].efficiency_ratio == 15/16
        assert stats["worst_efficiency_config"].efficiency_ratio == 13/16
    
    def test_get_padding_statistics_empty(self):
        """Test statistics calculation with empty list."""
        stats = self.system.get_padding_statistics([])
        assert stats["count"] == 0


class TestPaddingStrategies:
    """Test specific padding strategy implementations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.system = PaddingConfigurationSystem()
    
    def test_end_fill_strategy_details(self):
        """Test detailed behavior of end_fill strategy."""
        param_count = 6
        dimensions = (3, 3)  # 9 total positions
        
        positions = self.system._end_fill_strategy(param_count, dimensions)
        
        # Should have 3 padding positions (9 - 6 = 3)
        assert len(positions) == 3
        
        # Should be the last 3 positions in row-major order
        # Positions: (0,0)=0, (1,0)=1, (2,0)=2, (0,1)=3, (1,1)=4, (2,1)=5, (0,2)=6, (1,2)=7, (2,2)=8
        # Last 3: positions 8, 7, 6 = (2,2), (1,2), (0,2)
        expected = [(2, 2), (1, 2), (0, 2)]
        assert positions == expected
    
    def test_distributed_strategy_details(self):
        """Test detailed behavior of distributed strategy."""
        param_count = 8
        dimensions = (4, 4)  # 16 total positions
        
        positions = self.system._distributed_strategy(param_count, dimensions)
        
        # Should have 8 padding positions (16 - 8 = 8)
        assert len(positions) == 8
        
        # Positions should be distributed throughout the space
        assert len(set(positions)) == len(positions)  # No duplicates
        
        # All positions should be valid
        for x, y in positions:
            assert 0 <= x < 4
            assert 0 <= y < 4
    
    def test_corner_fill_strategy_details(self):
        """Test detailed behavior of corner_fill strategy."""
        param_count = 12
        dimensions = (4, 4)  # 16 total positions
        
        positions = self.system._corner_fill_strategy(param_count, dimensions)
        
        # Should have 4 padding positions (16 - 12 = 4)
        assert len(positions) == 4
        
        # Should be the 4 corners
        expected_corners = [(3, 3), (0, 3), (3, 0), (0, 0)]
        assert positions == expected_corners
    
    def test_corner_fill_strategy_overflow(self):
        """Test corner_fill strategy when more positions needed than corners."""
        param_count = 4
        dimensions = (4, 4)  # 16 total positions, need 12 padding
        
        positions = self.system._corner_fill_strategy(param_count, dimensions)
        
        # Should have 12 padding positions
        assert len(positions) == 12
        
        # First 4 should be corners
        corners = [(3, 3), (0, 3), (3, 0), (0, 0)]
        assert positions[:4] == corners
        
        # Remaining should be from end_fill strategy
        assert len(positions[4:]) == 8


class TestUtilityFunctions:
    """Test utility functions for padding configuration."""
    
    def test_create_optimal_padding_config(self):
        """Test creation of optimal padding configuration."""
        param_count = 200  # Use a parameter count that has valid configurations
        
        config = create_optimal_padding_config(param_count)
        
        assert isinstance(config, PaddingConfig)
        assert config.efficiency_ratio > 0
        
        # Should be the most efficient option
        system = PaddingConfigurationSystem()
        all_configs = system.optimize_padding_for_parameter_count(param_count)
        assert config.efficiency_ratio == max(c.efficiency_ratio for c in all_configs)
    
    def test_create_optimal_padding_config_no_valid(self):
        """Test optimal config creation when no valid configuration exists."""
        # This should work for any reasonable parameter count
        param_count = 1
        config = create_optimal_padding_config(param_count)
        assert isinstance(config, PaddingConfig)
    
    def test_analyze_padding_efficiency(self):
        """Test analysis of padding efficiency for multiple parameter counts."""
        param_counts = [10, 50, 200, 500]
        
        analysis = analyze_padding_efficiency(param_counts)
        
        assert len(analysis) == len(param_counts)
        
        for param_count in param_counts:
            assert param_count in analysis
            result = analysis[param_count]
            
            if "error" not in result:
                assert "configs" in result
                assert "statistics" in result
                assert "optimal_config" in result
                
                if result["configs"]:
                    assert isinstance(result["optimal_config"], PaddingConfig)
                    assert result["statistics"]["count"] > 0


class TestEdgeCasesAndPerformance:
    """Test edge cases and performance scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.system = PaddingConfigurationSystem()
    
    def test_perfect_fit_scenarios(self):
        """Test scenarios where parameters perfectly fit dimensions."""
        perfect_fits = [
            (4, (2, 2)),
            (16, (4, 4)),
            (64, (8, 8)),
            (256, (16, 16))
        ]
        
        for param_count, dims in perfect_fits:
            config = self.system.create_padding_config(param_count, dims)
            
            assert config.efficiency_ratio == 1.0
            assert len(config.padding_positions) == 0
    
    def test_single_parameter_scenarios(self):
        """Test scenarios with very few parameters."""
        config = self.system.create_padding_config(1, (2, 2))
        
        assert config.efficiency_ratio == 0.25
        assert len(config.padding_positions) == 3
    
    def test_large_parameter_counts(self):
        """Test with large parameter counts."""
        large_param_count = 5000
        
        configs = self.system.optimize_padding_for_parameter_count(large_param_count)
        
        # Should find at least one valid configuration
        assert len(configs) > 0
        
        # All should accommodate the parameter count
        for config in configs:
            width, height = config.target_dimensions
            total_space = width * height
            assert total_space >= large_param_count
    
    def test_strategy_consistency(self):
        """Test that strategies produce consistent results."""
        param_count = 20
        target_dims = (8, 8)
        
        # Run same strategy multiple times
        configs = []
        for _ in range(5):
            config = self.system.create_padding_config(param_count, target_dims, strategy="end_fill")
            configs.append(config)
        
        # All should be identical
        first_config = configs[0]
        for config in configs[1:]:
            assert config.target_dimensions == first_config.target_dimensions
            assert config.efficiency_ratio == first_config.efficiency_ratio
            assert config.padding_positions == first_config.padding_positions


if __name__ == "__main__":
    pytest.main([__file__])