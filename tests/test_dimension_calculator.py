"""
Unit tests for dimension calculation utilities.

Tests the PowerOf4DimensionCalculator implementation including edge cases,
padding strategies, and efficiency calculations.
"""

import pytest
import math
from typing import Tuple

from hilbert_quantization.core.dimension_calculator import (
    PowerOf4DimensionCalculator,
    validate_power_of_4,
    calculate_dimension_efficiency
)
from hilbert_quantization.models import PaddingConfig
from hilbert_quantization.config import Constants


class TestPowerOf4DimensionCalculator:
    """Test cases for PowerOf4DimensionCalculator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = PowerOf4DimensionCalculator()
    
    def test_calculate_optimal_dimensions_basic(self):
        """Test basic dimension calculation for various parameter counts."""
        test_cases = [
            (1, (2, 2)),      # 1 param -> 4 total (2x2)
            (4, (2, 2)),      # 4 params -> 4 total (2x2)
            (5, (4, 4)),      # 5 params -> 16 total (4x4)
            (16, (4, 4)),     # 16 params -> 16 total (4x4)
            (17, (8, 8)),     # 17 params -> 64 total (8x8)
            (64, (8, 8)),     # 64 params -> 64 total (8x8)
            (65, (16, 16)),   # 65 params -> 256 total (16x16)
            (256, (16, 16)),  # 256 params -> 256 total (16x16)
            (1000, (32, 32)), # 1000 params -> 1024 total (32x32)
        ]
        
        for param_count, expected_dims in test_cases:
            result = self.calculator.calculate_optimal_dimensions(param_count)
            assert result == expected_dims, f"Failed for {param_count} parameters"
    
    def test_calculate_optimal_dimensions_edge_cases(self):
        """Test edge cases for dimension calculation."""
        # Test minimum case
        assert self.calculator.calculate_optimal_dimensions(1) == (2, 2)
        
        # Test large numbers
        assert self.calculator.calculate_optimal_dimensions(10000) == (128, 128)  # 16384 is next power of 4 after 10000
        
        # Test exact power of 4 boundaries
        assert self.calculator.calculate_optimal_dimensions(4) == (2, 2)
        assert self.calculator.calculate_optimal_dimensions(16) == (4, 4)
        assert self.calculator.calculate_optimal_dimensions(64) == (8, 8)
        assert self.calculator.calculate_optimal_dimensions(256) == (16, 16)
        assert self.calculator.calculate_optimal_dimensions(1024) == (32, 32)
    
    def test_calculate_optimal_dimensions_invalid_input(self):
        """Test invalid inputs for dimension calculation."""
        with pytest.raises(ValueError, match="Parameter count must be positive"):
            self.calculator.calculate_optimal_dimensions(0)
        
        with pytest.raises(ValueError, match="Parameter count must be positive"):
            self.calculator.calculate_optimal_dimensions(-1)
    
    def test_calculate_padding_strategy_basic(self):
        """Test basic padding strategy calculation."""
        # Test case: 10 parameters in 4x4 grid (16 total)
        param_count = 10
        target_dims = (4, 4)
        
        config = self.calculator.calculate_padding_strategy(param_count, target_dims)
        
        assert config.target_dimensions == target_dims
        assert config.padding_value == Constants.DEFAULT_PADDING_VALUE
        assert config.efficiency_ratio == 10/16  # 0.625
        assert len(config.padding_positions) == 6  # 16 - 10 = 6 padding positions
        
        # Check that padding positions are at the end (row-major order)
        # For 4x4 grid (16 positions), positions 0-9 are used, 10-15 are padding
        # Position 15 = (3,3), 14 = (2,3), 13 = (1,3), 12 = (0,3), 11 = (3,2), 10 = (2,2)
        expected_positions = [(3, 3), (2, 3), (1, 3), (0, 3), (3, 2), (2, 2)]
        assert config.padding_positions == expected_positions
    
    def test_calculate_padding_strategy_perfect_fit(self):
        """Test padding strategy when parameters exactly fit dimensions."""
        param_count = 16
        target_dims = (4, 4)
        
        config = self.calculator.calculate_padding_strategy(param_count, target_dims)
        
        assert config.efficiency_ratio == 1.0
        assert len(config.padding_positions) == 0
    
    def test_calculate_padding_strategy_efficiency_validation(self):
        """Test efficiency ratio validation in padding strategy."""
        # Create calculator with high minimum efficiency
        strict_calculator = PowerOf4DimensionCalculator(min_efficiency_ratio=0.8)
        
        # This should fail: 10 params in 32x32 grid = 10/1024 = 0.0098 efficiency
        with pytest.raises(ValueError, match="Efficiency ratio .* is below minimum"):
            strict_calculator.calculate_padding_strategy(10, (32, 32))
    
    def test_calculate_padding_strategy_invalid_dimensions(self):
        """Test invalid dimension inputs for padding strategy."""
        # Dimensions too small for parameter count
        with pytest.raises(ValueError, match="cannot accommodate"):
            self.calculator.calculate_padding_strategy(20, (2, 2))  # 20 params in 4 total space
    
    def test_padding_positions_calculation(self):
        """Test detailed padding position calculations."""
        # Test 3x3 grid with 5 parameters
        param_count = 5
        target_dims = (3, 3)
        
        config = self.calculator.calculate_padding_strategy(param_count, target_dims)
        
        # Should have 4 padding positions (9 - 5 = 4)
        assert len(config.padding_positions) == 4
        
        # Positions should be at the end in reverse row-major order
        # For 3x3 grid (9 positions), positions 0-4 are used, 5-8 are padding
        # Position 8 = (2,2), 7 = (1,2), 6 = (0,2), 5 = (2,1)
        expected_positions = [(2, 2), (1, 2), (0, 2), (2, 1)]
        assert config.padding_positions == expected_positions
    
    def test_get_efficiency_metrics(self):
        """Test efficiency metrics calculation."""
        param_count = 100
        dimensions = (16, 16)  # 256 total
        
        metrics = self.calculator.get_efficiency_metrics(param_count, dimensions)
        
        expected_metrics = {
            'total_space': 256,
            'used_space': 100,
            'wasted_space': 156,
            'efficiency_ratio': 100/256,
            'waste_percentage': (156/256) * 100,
            'dimensions': (16, 16)
        }
        
        assert metrics == expected_metrics
    
    def test_find_all_valid_dimensions(self):
        """Test finding all valid dimensions within waste threshold."""
        param_count = 100
        
        # With 50% max waste, should include multiple options
        valid_dims = self.calculator.find_all_valid_dimensions(param_count, max_waste_percentage=50.0)
        
        # Should include at least (16, 16) = 256 total, waste = 156/256 = 60.9% > 50%
        # Should include (32, 32) = 1024 total, waste = 924/1024 = 90.2% > 50%
        # Actually, let's check (16, 16): waste = (256-100)/256 = 60.9% > 50%
        # So it should only include larger dimensions if any meet the criteria
        
        # Let's be more specific - with 10% max waste
        valid_dims_strict = self.calculator.find_all_valid_dimensions(param_count, max_waste_percentage=10.0)
        
        # With 10% max waste, 100 params would need at least 100/0.9 = 111.1 total space
        # So (16, 16) = 256 has (256-100)/256 = 60.9% waste > 10%
        # This should return empty list or only very large dimensions
        
        # Test with a parameter count that fits well
        param_count_good = 240  # Close to 256
        valid_dims_good = self.calculator.find_all_valid_dimensions(param_count_good, max_waste_percentage=10.0)
        
        # (16, 16) = 256, waste = (256-240)/256 = 6.25% < 10%
        assert (16, 16) in valid_dims_good


class TestUtilityFunctions:
    """Test utility functions for dimension calculations."""
    
    def test_validate_power_of_4(self):
        """Test power of 4 validation function."""
        # Valid powers of 4
        assert validate_power_of_4(1) == True   # 4^0
        assert validate_power_of_4(4) == True   # 4^1
        assert validate_power_of_4(16) == True  # 4^2
        assert validate_power_of_4(64) == True  # 4^3
        assert validate_power_of_4(256) == True # 4^4
        assert validate_power_of_4(1024) == True # 4^5
        
        # Invalid values
        assert validate_power_of_4(0) == False
        assert validate_power_of_4(-1) == False
        assert validate_power_of_4(2) == False
        assert validate_power_of_4(3) == False
        assert validate_power_of_4(5) == False
        assert validate_power_of_4(8) == False
        assert validate_power_of_4(15) == False
        assert validate_power_of_4(17) == False
    
    def test_calculate_dimension_efficiency(self):
        """Test dimension efficiency calculation function."""
        # Perfect efficiency
        assert calculate_dimension_efficiency(16, (4, 4)) == 1.0
        
        # Partial efficiency
        assert calculate_dimension_efficiency(10, (4, 4)) == 10/16
        
        # Over-capacity (should cap at 1.0)
        assert calculate_dimension_efficiency(20, (4, 4)) == 1.0
        
        # Zero dimensions
        assert calculate_dimension_efficiency(10, (0, 4)) == 0.0
        assert calculate_dimension_efficiency(10, (4, 0)) == 0.0
        
        # Zero parameters
        assert calculate_dimension_efficiency(0, (4, 4)) == 0.0


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = PowerOf4DimensionCalculator()
    
    def test_very_large_parameter_counts(self):
        """Test handling of very large parameter counts."""
        # Test with parameter count larger than largest predefined dimension
        large_param_count = 20000  # Larger than 16384
        
        # Should still work, finding next power of 4
        dims = self.calculator.calculate_optimal_dimensions(large_param_count)
        total_space = dims[0] * dims[1]
        
        # Should be a power of 4 and >= param_count
        assert validate_power_of_4(total_space)
        assert total_space >= large_param_count
    
    def test_custom_efficiency_thresholds(self):
        """Test custom efficiency ratio thresholds."""
        # Very strict efficiency requirement
        strict_calculator = PowerOf4DimensionCalculator(min_efficiency_ratio=0.95)
        
        # Should work for near-perfect fits
        config = strict_calculator.calculate_padding_strategy(16, (4, 4))  # 16/16 = 100% > 95%
        
        # Should fail for poor fits
        with pytest.raises(ValueError):
            strict_calculator.calculate_padding_strategy(5, (4, 4))  # 5/16 = 31.25%
    
    def test_boundary_conditions(self):
        """Test boundary conditions for various calculations."""
        # Test exactly at power of 4 boundaries
        boundary_cases = [4, 16, 64, 256, 1024]
        
        for param_count in boundary_cases:
            dims = self.calculator.calculate_optimal_dimensions(param_count)
            total_space = dims[0] * dims[1]
            assert total_space == param_count  # Should be exact fit
            
            config = self.calculator.calculate_padding_strategy(param_count, dims)
            assert config.efficiency_ratio == 1.0
            assert len(config.padding_positions) == 0
    
    def test_padding_config_validation(self):
        """Test that PaddingConfig validation works correctly."""
        # Valid config should not raise
        config = PaddingConfig(
            target_dimensions=(4, 4),
            padding_value=0.0,
            padding_positions=[(3, 3)],
            efficiency_ratio=0.75
        )
        
        # Invalid efficiency ratio should raise
        with pytest.raises(ValueError, match="Efficiency ratio must be between 0 and 1"):
            PaddingConfig(
                target_dimensions=(4, 4),
                padding_value=0.0,
                padding_positions=[],
                efficiency_ratio=1.5
            )
        
        # Invalid dimensions should raise
        with pytest.raises(ValueError, match="Target dimensions must be a 2-tuple"):
            PaddingConfig(
                target_dimensions=(4, 4, 4),  # 3-tuple instead of 2-tuple
                padding_value=0.0,
                padding_positions=[],
                efficiency_ratio=0.75
            )


if __name__ == "__main__":
    pytest.main([__file__])