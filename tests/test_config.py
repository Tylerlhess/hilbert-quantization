"""
Tests for configuration classes and utilities.
"""

import pytest
from hilbert_quantization.config import (
    QuantizationConfig, CompressionConfig, SearchConfig, SystemConfig,
    validate_power_of_4, get_nearest_power_of_4, calculate_dimension_efficiency
)


def test_validate_power_of_4():
    """Test power of 4 validation."""
    assert validate_power_of_4(4) == True
    assert validate_power_of_4(16) == True
    assert validate_power_of_4(64) == True
    assert validate_power_of_4(256) == True
    assert validate_power_of_4(1024) == True
    
    assert validate_power_of_4(5) == False
    assert validate_power_of_4(15) == False
    assert validate_power_of_4(0) == False
    assert validate_power_of_4(-4) == False


def test_get_nearest_power_of_4():
    """Test nearest power of 4 calculation."""
    assert get_nearest_power_of_4(1) == 4
    assert get_nearest_power_of_4(4) == 4
    assert get_nearest_power_of_4(5) == 16
    assert get_nearest_power_of_4(16) == 16
    assert get_nearest_power_of_4(17) == 64
    assert get_nearest_power_of_4(1000) == 1024


def test_calculate_dimension_efficiency():
    """Test dimension efficiency calculation."""
    assert calculate_dimension_efficiency(64, (8, 8)) == 1.0
    assert calculate_dimension_efficiency(32, (8, 8)) == 0.5
    assert calculate_dimension_efficiency(100, (8, 8)) == 1.0  # Capped at 1.0
    assert calculate_dimension_efficiency(0, (8, 8)) == 0.0


def test_quantization_config_validation():
    """Test QuantizationConfig validation."""
    # Valid configuration
    config = QuantizationConfig(
        target_dimensions=(64, 64),
        min_efficiency_ratio=0.8
    )
    assert config.min_efficiency_ratio == 0.8
    
    # Invalid target dimensions
    with pytest.raises(ValueError):
        QuantizationConfig(target_dimensions=(63, 63))  # Not power of 4


def test_system_config_defaults():
    """Test SystemConfig default initialization."""
    config = SystemConfig()
    assert config.quantization is not None
    assert config.compression is not None
    assert config.search is not None