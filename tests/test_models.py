"""
Tests for core data models.
"""

import pytest
import numpy as np
from hilbert_quantization.models import (
    QuantizedModel, PaddingConfig, SearchResult, ModelMetadata,
    CompressionMetrics, SearchMetrics
)


def test_padding_config_validation():
    """Test PaddingConfig validation."""
    # Valid configuration
    config = PaddingConfig(
        target_dimensions=(64, 64),
        padding_value=0.0,
        padding_positions=[(0, 0), (1, 1)],
        efficiency_ratio=0.8
    )
    assert config.efficiency_ratio == 0.8
    
    # Invalid efficiency ratio
    with pytest.raises(ValueError):
        PaddingConfig(
            target_dimensions=(64, 64),
            padding_value=0.0,
            padding_positions=[],
            efficiency_ratio=1.5
        )


def test_quantized_model_validation():
    """Test QuantizedModel validation."""
    metadata = ModelMetadata(
        model_name="test_model",
        original_size_bytes=1000,
        compressed_size_bytes=500,
        compression_ratio=0.5,
        quantization_timestamp="2024-01-01T00:00:00"
    )
    
    # Valid model
    model = QuantizedModel(
        compressed_data=b"test_data",
        original_dimensions=(64, 64),
        parameter_count=1000,
        compression_quality=0.8,
        hierarchical_indices=np.array([1.0, 2.0, 3.0]),
        metadata=metadata
    )
    assert model.parameter_count == 1000
    
    # Invalid parameter count
    with pytest.raises(ValueError):
        QuantizedModel(
            compressed_data=b"test_data",
            original_dimensions=(64, 64),
            parameter_count=-1,
            compression_quality=0.8,
            hierarchical_indices=np.array([1.0, 2.0, 3.0]),
            metadata=metadata
        )