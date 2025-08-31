"""
Tests for streaming index generator functionality.
"""

import pytest
import numpy as np

from hilbert_quantization.core.streaming_index_builder import (
    StreamingIndexBuilder, StreamingHilbertIndexGenerator
)
from hilbert_quantization.core.index_generator import HierarchicalIndexGeneratorImpl
from hilbert_quantization.config import QuantizationConfig


class TestStreamingIndexBuilder:
    """Test streaming index builder."""
    
    def test_basic_functionality(self):
        """Test basic streaming index builder functionality."""
        builder = StreamingIndexBuilder(max_levels=5)
        
        # Add some values
        for i in range(10):
            builder.add_value(float(i))
        
        stats = builder.get_statistics()
        assert stats['total_values_processed'] == 10
        assert stats['levels_used'] >= 2
        
        # Get indices
        indices = builder.get_hierarchical_indices(50)
        assert len(indices) == 50
        assert not np.any(np.isnan(indices))
    
    def test_hierarchical_promotion(self):
        """Test hierarchical promotion logic."""
        builder = StreamingIndexBuilder(max_levels=3)
        
        # Add exactly 16 values to see promotion pattern
        for i in range(16):
            builder.add_value(float(i))
        
        stats = builder.get_statistics()
        
        # Should have promoted to multiple levels
        assert stats['levels_used'] >= 2
        assert 0 in stats['indices_per_level']
        assert 1 in stats['indices_per_level']
        
        # Level 0 should have all 16 values
        assert stats['indices_per_level'][0] == 16
        # Level 1 should have 4 values (16/4)
        assert stats['indices_per_level'][1] == 4
    
    def test_memory_efficiency(self):
        """Test memory efficiency with large dataset."""
        builder = StreamingIndexBuilder(max_levels=10)
        
        # Add many values
        for i in range(10000):
            builder.add_value(float(i))
        
        # Check that sliding windows are maintained at size 4 or less
        stats = builder.get_statistics()
        for level, window_size in stats['current_window_sizes'].items():
            assert window_size <= 4
        
        # Should be able to extract indices
        indices = builder.get_hierarchical_indices(1000)
        assert len(indices) == 1000
        assert not np.any(np.isnan(indices))
    
    def test_reset_functionality(self):
        """Test reset functionality."""
        builder = StreamingIndexBuilder()
        
        # Add some values
        for i in range(10):
            builder.add_value(float(i))
        
        # Reset
        builder.reset()
        
        stats = builder.get_statistics()
        assert stats['total_values_processed'] == 0
        assert stats['levels_used'] == 0


class TestStreamingHilbertIndexGenerator:
    """Test streaming Hilbert index generator."""
    
    def test_basic_index_generation(self):
        """Test basic index generation."""
        generator = StreamingHilbertIndexGenerator()
        
        # Create test image
        image = np.random.randn(16, 16)
        
        # Generate indices
        indices = generator.generate_optimized_indices(image, 100)
        
        assert len(indices) == 100
        assert not np.any(np.isnan(indices))
        assert not np.any(np.isinf(indices))
    
    def test_integrated_mapping(self):
        """Test integrated mapping functionality."""
        generator = StreamingHilbertIndexGenerator()
        
        # Create test parameters
        parameters = np.random.randn(1000)
        dimensions = (32, 32)
        
        # Test integrated mapping
        image_2d, indices, stats = generator.generate_indices_during_mapping(
            parameters, dimensions, 500
        )
        
        assert image_2d.shape == dimensions
        assert len(indices) == 500
        assert stats['total_values_processed'] == 1000  # Actual parameters processed
        assert stats['levels_used'] >= 5
    
    def test_large_dataset_handling(self):
        """Test handling of large datasets."""
        generator = StreamingHilbertIndexGenerator()
        
        # Create large test image
        image = np.random.randn(128, 128)  # 16,384 parameters
        
        # Generate indices
        indices = generator.generate_optimized_indices(image, 1000)
        
        assert len(indices) == 1000
        assert not np.any(np.isnan(indices))
        assert not np.any(np.isinf(indices))
    
    def test_edge_cases(self):
        """Test edge cases."""
        generator = StreamingHilbertIndexGenerator()
        
        # Empty image - should handle gracefully
        with pytest.raises(ValueError, match="Image must be square with power-of-2 dimensions"):
            generator.generate_optimized_indices(np.array([]).reshape(0, 0), 100)

        
        # Single value
        single_image = np.array([[1.0]])
        single_result = generator.generate_optimized_indices(single_image, 10)
        assert len(single_result) == 10
        
        # Zero index space
        image = np.random.randn(8, 8)
        zero_result = generator.generate_optimized_indices(image, 0)
        assert len(zero_result) == 0


class TestConfigurationIntegration:
    """Test integration with configuration system."""
    
    def test_streaming_configuration(self):
        """Test streaming configuration options."""
        # Test with streaming enabled
        config = QuantizationConfig(
            use_streaming_optimization=True,
            streaming_max_levels=8,
            memory_efficient_mode=True
        )
        
        generator = HierarchicalIndexGeneratorImpl(config)
        
        # Should use streaming approach
        image = np.random.randn(32, 32)
        indices = generator.generate_optimized_indices(image, 100)
        
        assert len(indices) == 100
        assert not np.any(np.isnan(indices))
    
    def test_traditional_configuration(self):
        """Test traditional configuration (streaming disabled)."""
        # Test with streaming disabled
        config = QuantizationConfig(use_streaming_optimization=False)
        
        generator = HierarchicalIndexGeneratorImpl(config)
        
        # Should use traditional approach
        image = np.random.randn(32, 32)
        indices = generator.generate_optimized_indices(image, 100)
        
        assert len(indices) == 100
        assert not np.any(np.isnan(indices))
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        # Valid configuration
        config = QuantizationConfig(
            use_streaming_optimization=True,
            streaming_max_levels=10
        )
        # Should not raise
        
        # Invalid configuration
        with pytest.raises(ValueError, match="Streaming max levels must be at least 1"):
            QuantizationConfig(streaming_max_levels=0)
        
        with pytest.raises(ValueError, match="cannot exceed 15"):
            QuantizationConfig(streaming_max_levels=20)


class TestPerformanceComparison:
    """Test performance comparison between methods."""
    
    def test_method_comparison(self):
        """Test comparison between traditional and streaming methods."""
        # Traditional generator
        trad_config = QuantizationConfig(use_streaming_optimization=False)
        trad_generator = HierarchicalIndexGeneratorImpl(trad_config)
        
        # Streaming generator
        stream_config = QuantizationConfig(use_streaming_optimization=True)
        stream_generator = HierarchicalIndexGeneratorImpl(stream_config)
        
        # Test with same image
        image = np.random.randn(64, 64)
        index_space_size = 500
        
        # Both should produce valid results
        trad_indices = trad_generator.generate_optimized_indices(image, index_space_size)
        stream_indices = stream_generator.generate_optimized_indices(image, index_space_size)
        
        assert len(trad_indices) == index_space_size
        assert len(stream_indices) == index_space_size
        assert not np.any(np.isnan(trad_indices))
        assert not np.any(np.isnan(stream_indices))
        
        # Both should be valid but may differ in values (different algorithms)
        assert not np.any(np.isinf(trad_indices))
        assert not np.any(np.isinf(stream_indices))


if __name__ == "__main__":
    pytest.main([__file__])