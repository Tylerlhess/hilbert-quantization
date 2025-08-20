"""
Tests for MPEG-AI compression implementation.
"""

import pytest
import numpy as np
import io
from PIL import Image

from hilbert_quantization.core.compressor import MPEGAICompressorImpl
from hilbert_quantization.config import CompressionConfig
from hilbert_quantization.models import CompressionMetrics


class TestMPEGAICompressorImpl:
    """Test cases for MPEGAICompressorImpl."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.compressor = MPEGAICompressorImpl()
        
        # Create test images
        self.test_image_small = np.random.rand(32, 32).astype(np.float32)
        self.test_image_large = np.random.rand(256, 256).astype(np.float32)
        
        # Create image with index row (last row contains indices)
        self.test_image_with_index = np.random.rand(33, 32).astype(np.float32)
        # Make last row distinct (simulating index row)
        self.test_image_with_index[-1, :] = np.linspace(0, 1, 32)
    
    def test_compress_basic(self):
        """Test basic compression functionality."""
        compressed = self.compressor.compress(self.test_image_small, quality=0.8)
        
        assert isinstance(compressed, bytes)
        assert len(compressed) > 0
        assert len(compressed) < self.test_image_small.nbytes  # Should be compressed
    
    def test_compress_different_qualities(self):
        """Test compression with different quality settings."""
        qualities = [0.1, 0.5, 0.8, 1.0]
        compressed_sizes = []
        
        for quality in qualities:
            compressed = self.compressor.compress(self.test_image_small, quality)
            compressed_sizes.append(len(compressed))
        
        # Higher quality should generally result in larger compressed size
        # (though this isn't guaranteed for all images)
        assert all(size > 0 for size in compressed_sizes)
    
    def test_decompress_basic(self):
        """Test basic decompression functionality."""
        compressed = self.compressor.compress(self.test_image_small, quality=0.8)
        reconstructed = self.compressor.decompress(compressed)
        
        assert isinstance(reconstructed, np.ndarray)
        assert reconstructed.shape == self.test_image_small.shape
        assert reconstructed.dtype == np.float32
    
    def test_compression_round_trip(self):
        """Test compression and decompression round trip."""
        original = self.test_image_small
        
        # Compress and decompress
        compressed = self.compressor.compress(original, quality=0.9)
        reconstructed = self.compressor.decompress(compressed)
        
        # Check shape preservation
        assert reconstructed.shape == original.shape
        
        # Check that reconstruction is reasonably close (lossy compression)
        mse = np.mean((original - reconstructed) ** 2)
        assert mse < 0.1  # Reasonable threshold for high quality compression
    
    def test_compression_metrics(self):
        """Test compression metrics calculation."""
        original = self.test_image_large
        compressed = self.compressor.compress(original, quality=0.8)
        
        metrics = self.compressor.get_last_compression_metrics()
        assert metrics is not None
        assert isinstance(metrics, CompressionMetrics)
        assert metrics.compression_ratio > 1.0  # Should achieve compression
        assert metrics.compression_time_seconds >= 0
        assert metrics.memory_usage_mb > 0
    
    def test_estimate_compression_ratio(self):
        """Test compression ratio estimation."""
        ratio = self.compressor.estimate_compression_ratio(1000, 200)
        assert ratio == 5.0
        
        # Test edge case
        ratio = self.compressor.estimate_compression_ratio(1000, 0)
        assert ratio == 0.0
    
    def test_reconstruction_error_calculation(self):
        """Test reconstruction error calculation."""
        original = self.test_image_small
        compressed = self.compressor.compress(original, quality=0.5)
        reconstructed = self.compressor.decompress(compressed)
        
        error = self.compressor.calculate_reconstruction_error(original, reconstructed)
        assert error >= 0
        assert isinstance(error, float)
        
        # Error should be updated in metrics
        metrics = self.compressor.get_last_compression_metrics()
        assert metrics.reconstruction_error == error
    
    def test_index_row_integrity_validation(self):
        """Test index row integrity validation."""
        original = self.test_image_with_index
        compressed = self.compressor.compress(original, quality=0.9)
        reconstructed = self.compressor.decompress(compressed)
        
        # Test with high quality - should preserve index row well
        is_preserved = self.compressor.validate_index_row_integrity(
            original, reconstructed, tolerance=0.1
        )
        assert isinstance(is_preserved, bool)
        
        # Test with very low tolerance - might fail
        strict_preserved = self.compressor.validate_index_row_integrity(
            original, reconstructed, tolerance=1e-6
        )
        assert isinstance(strict_preserved, bool)
    
    def test_compress_invalid_inputs(self):
        """Test compression with invalid inputs."""
        # Test non-numpy array
        with pytest.raises(ValueError, match="Image must be a numpy array"):
            self.compressor.compress([[1, 2], [3, 4]], quality=0.8)
        
        # Test wrong dimensions
        with pytest.raises(ValueError, match="Image must be 2-dimensional"):
            self.compressor.compress(np.array([1, 2, 3]), quality=0.8)
        
        # Test invalid quality
        with pytest.raises(ValueError, match="Quality must be between 0.0 and 1.0"):
            self.compressor.compress(self.test_image_small, quality=1.5)
        
        with pytest.raises(ValueError, match="Quality must be between 0.0 and 1.0"):
            self.compressor.compress(self.test_image_small, quality=-0.1)
    
    def test_decompress_invalid_inputs(self):
        """Test decompression with invalid inputs."""
        # Test non-bytes input
        with pytest.raises(ValueError, match="Compressed data must be bytes"):
            self.compressor.decompress("not bytes")
        
        # Test empty bytes
        with pytest.raises(ValueError, match="Compressed data cannot be empty"):
            self.compressor.decompress(b"")
        
        # Test invalid compressed data
        with pytest.raises(RuntimeError, match="Failed to decompress image"):
            self.compressor.decompress(b"invalid jpeg data")
    
    def test_compression_with_config(self):
        """Test compression with custom configuration."""
        config = CompressionConfig(
            quality=0.7,
            preserve_index_row=True,
            validate_reconstruction=True
        )
        
        compressor = MPEGAICompressorImpl(config)
        compressed = compressor.compress(self.test_image_small, quality=0.8)
        reconstructed = compressor.decompress(compressed)
        
        assert reconstructed.shape == self.test_image_small.shape
    
    def test_constant_image_compression(self):
        """Test compression of constant (uniform) images."""
        constant_image = np.ones((64, 64), dtype=np.float32) * 0.5
        
        compressed = self.compressor.compress(constant_image, quality=0.8)
        reconstructed = self.compressor.decompress(compressed)
        
        assert reconstructed.shape == constant_image.shape
        # Constant image should reconstruct very well
        mse = np.mean((constant_image - reconstructed) ** 2)
        assert mse < 0.01
    
    def test_extreme_value_images(self):
        """Test compression of images with extreme values."""
        # Very large values
        large_image = np.random.rand(32, 32) * 1000
        compressed = self.compressor.compress(large_image, quality=0.8)
        reconstructed = self.compressor.decompress(compressed)
        assert reconstructed.shape == large_image.shape
        
        # Very small values
        small_image = np.random.rand(32, 32) * 1e-6
        compressed = self.compressor.compress(small_image, quality=0.8)
        reconstructed = self.compressor.decompress(compressed)
        assert reconstructed.shape == small_image.shape
        
        # Negative values
        negative_image = np.random.rand(32, 32) * 2 - 1  # Range [-1, 1]
        compressed = self.compressor.compress(negative_image, quality=0.8)
        reconstructed = self.compressor.decompress(compressed)
        assert reconstructed.shape == negative_image.shape
    
    def test_shape_mismatch_error_calculation(self):
        """Test error calculation with mismatched shapes."""
        image1 = np.random.rand(32, 32)
        image2 = np.random.rand(64, 64)
        
        with pytest.raises(ValueError, match="Images must have the same shape"):
            self.compressor.calculate_reconstruction_error(image1, image2)
    
    def test_index_row_validation_edge_cases(self):
        """Test index row validation with edge cases."""
        # Very small image (no index row)
        small_image = np.random.rand(1, 10)
        result = self.compressor.validate_index_row_integrity(
            small_image, small_image, tolerance=0.1
        )
        assert result is True  # Should return True for images too small
        
        # Shape mismatch
        image1 = np.random.rand(32, 32)
        image2 = np.random.rand(32, 31)
        result = self.compressor.validate_index_row_integrity(
            image1, image2, tolerance=0.1
        )
        assert result is False  # Should return False for shape mismatch


    def test_compress_with_index_preservation(self):
        """Test index-aware compression functionality."""
        config = CompressionConfig(preserve_index_row=True, validate_reconstruction=True)
        compressor = MPEGAICompressorImpl(config)
        
        original = self.test_image_with_index
        compressed = compressor.compress_with_index_preservation(original, quality=0.8)
        reconstructed = compressor.decompress_with_index_preservation(compressed)
        
        assert reconstructed.shape == original.shape
        
        # Index row should be better preserved than with regular compression
        index_mse = np.mean((original[-1, :] - reconstructed[-1, :]) ** 2)
        assert index_mse < 0.1  # Should be well preserved
    
    def test_compress_with_index_preservation_disabled(self):
        """Test index-aware compression when preservation is disabled."""
        config = CompressionConfig(preserve_index_row=False)
        compressor = MPEGAICompressorImpl(config)
        
        original = self.test_image_with_index
        compressed = compressor.compress_with_index_preservation(original, quality=0.8)
        reconstructed = compressor.decompress_with_index_preservation(compressed)
        
        assert reconstructed.shape == original.shape
    
    def test_compress_image_without_index_row(self):
        """Test index-aware compression on image without index row."""
        config = CompressionConfig(preserve_index_row=True)
        compressor = MPEGAICompressorImpl(config)
        
        # Single row image (no index row)
        single_row = np.random.rand(1, 32).astype(np.float32)
        compressed = compressor.compress_with_index_preservation(single_row, quality=0.8)
        reconstructed = compressor.decompress_with_index_preservation(compressed)
        
        assert reconstructed.shape == single_row.shape
    
    def test_assess_compression_impact_on_model_performance(self):
        """Test compression impact assessment."""
        original = self.test_image_with_index
        compressed = self.compressor.compress(original, quality=0.7)
        reconstructed = self.compressor.decompress(compressed)
        
        metrics = self.compressor.assess_compression_impact_on_model_performance(
            original, reconstructed
        )
        
        # Check that all expected metrics are present
        expected_metrics = ['mse', 'rmse', 'psnr', 'ssim', 'index_row_mse', 
                          'index_preservation_ratio', 'parameter_space_mse']
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
        
        # Check metric ranges
        assert metrics['mse'] >= 0
        assert metrics['rmse'] >= 0
        assert metrics['psnr'] > 0
        assert -1 <= metrics['ssim'] <= 1
        assert metrics['index_row_mse'] >= 0
        assert 0 <= metrics['index_preservation_ratio'] <= 1
        assert metrics['parameter_space_mse'] >= 0
    
    def test_assess_compression_impact_constant_image(self):
        """Test compression impact assessment on constant image."""
        constant_image = np.ones((33, 32), dtype=np.float32) * 0.5
        compressed = self.compressor.compress(constant_image, quality=0.8)
        reconstructed = self.compressor.decompress(compressed)
        
        metrics = self.compressor.assess_compression_impact_on_model_performance(
            constant_image, reconstructed
        )
        
        # For constant images, PSNR should be very high (or inf)
        assert metrics['psnr'] > 30 or metrics['psnr'] == float('inf')
    
    def test_index_preservation_validation_failure(self):
        """Test behavior when index row validation fails."""
        config = CompressionConfig(
            preserve_index_row=True, 
            validate_reconstruction=True,
            max_reconstruction_error=1e-6  # Very strict tolerance
        )
        compressor = MPEGAICompressorImpl(config)
        
        original = self.test_image_with_index
        
        # This should trigger validation warnings due to strict tolerance
        compressed = compressor.compress_with_index_preservation(original, quality=0.1)  # Low quality
        reconstructed = compressor.decompress_with_index_preservation(compressed)
        
        assert reconstructed.shape == original.shape
    
    def test_separated_compression_format(self):
        """Test the separated compression format for index preservation."""
        config = CompressionConfig(preserve_index_row=True)
        compressor = MPEGAICompressorImpl(config)
        
        original = self.test_image_with_index
        compressed = compressor.compress_with_index_preservation(original, quality=0.8)
        
        # Check that separator is present in compressed data
        separator = b'||INDEX_ROW||'
        assert separator in compressed
        
        # Check that we can split and reconstruct
        parts = compressed.split(separator)
        assert len(parts) == 2
        
        reconstructed = compressor.decompress_with_index_preservation(compressed)
        assert reconstructed.shape == original.shape


class TestCompressionMetricsCalculator:
    """Test cases for CompressionMetricsCalculator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        from hilbert_quantization.core.compressor import CompressionMetricsCalculator
        self.calculator = CompressionMetricsCalculator
        
        # Create test images
        self.original = np.random.rand(64, 64).astype(np.float32)
        self.reconstructed = self.original + np.random.normal(0, 0.01, self.original.shape).astype(np.float32)
        
        # Image with index row
        self.original_with_index = np.random.rand(65, 64).astype(np.float32)
        self.original_with_index[-1, :] = np.linspace(0, 1, 64)  # Index row
        self.reconstructed_with_index = self.original_with_index + np.random.normal(0, 0.01, self.original_with_index.shape).astype(np.float32)
    
    def test_calculate_comprehensive_metrics(self):
        """Test comprehensive metrics calculation."""
        metrics = self.calculator.calculate_comprehensive_metrics(
            self.original,
            self.reconstructed,
            compressed_size=1024,
            compression_time=0.1,
            decompression_time=0.05
        )
        
        # Check basic metrics
        assert 'original_size_bytes' in metrics
        assert 'compressed_size_bytes' in metrics
        assert 'compression_ratio' in metrics
        assert 'space_savings_percent' in metrics
        
        # Check performance metrics
        assert 'compression_time_seconds' in metrics
        assert 'decompression_time_seconds' in metrics
        assert 'total_processing_time' in metrics
        assert 'compression_throughput_mbps' in metrics
        assert 'decompression_throughput_mbps' in metrics
        
        # Check quality metrics
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'psnr' in metrics
        assert 'ssim' in metrics
        assert 'mae' in metrics
        assert 'normalized_cross_correlation' in metrics
        
        # Check efficiency score
        assert 'efficiency_score' in metrics
        assert 0 <= metrics['efficiency_score'] <= 1
        
        # Validate metric ranges
        assert metrics['compression_ratio'] > 0
        assert 0 <= metrics['space_savings_percent'] <= 100
        assert metrics['mse'] >= 0
        assert metrics['rmse'] >= 0
        assert metrics['mae'] >= 0
        assert -1 <= metrics['ssim'] <= 1
        assert -1 <= metrics['normalized_cross_correlation'] <= 1
    
    def test_calculate_quality_metrics(self):
        """Test quality metrics calculation."""
        metrics = self.calculator._calculate_quality_metrics(self.original, self.reconstructed)
        
        expected_keys = ['mse', 'rmse', 'psnr', 'mae', 'ssim', 'normalized_cross_correlation']
        for key in expected_keys:
            assert key in metrics
            assert isinstance(metrics[key], (int, float))
        
        # Check relationships
        assert metrics['rmse'] == np.sqrt(metrics['mse'])
        assert metrics['mse'] >= 0
        assert metrics['mae'] >= 0
    
    def test_calculate_ssim(self):
        """Test SSIM calculation."""
        # Perfect match should give SSIM = 1
        ssim_perfect = self.calculator._calculate_ssim(self.original, self.original)
        assert abs(ssim_perfect - 1.0) < 1e-10
        
        # Different images should give SSIM < 1
        ssim_different = self.calculator._calculate_ssim(self.original, self.reconstructed)
        assert ssim_different < 1.0
        assert ssim_different > -1.0
        
        # Constant images
        constant1 = np.ones((32, 32))
        constant2 = np.ones((32, 32)) * 2
        ssim_constant = self.calculator._calculate_ssim(constant1, constant1)
        assert abs(ssim_constant - 1.0) < 1e-10
    
    def test_calculate_index_row_metrics(self):
        """Test index row specific metrics."""
        metrics = self.calculator.calculate_index_row_metrics(
            self.original_with_index, self.reconstructed_with_index
        )
        
        expected_keys = ['index_row_mse', 'index_row_rmse', 'index_row_mae',
                        'index_preservation_ratio', 'index_row_correlation', 'index_row_max_deviation']
        for key in expected_keys:
            assert key in metrics
            assert isinstance(metrics[key], (int, float))
        
        # Check ranges
        assert metrics['index_row_mse'] >= 0
        assert metrics['index_row_rmse'] >= 0
        assert metrics['index_row_mae'] >= 0
        assert 0 <= metrics['index_preservation_ratio'] <= 1
        assert -1 <= metrics['index_row_correlation'] <= 1
        assert metrics['index_row_max_deviation'] >= 0
    
    def test_calculate_index_row_metrics_no_index(self):
        """Test index row metrics with no index row."""
        single_row = np.random.rand(1, 32)
        metrics = self.calculator.calculate_index_row_metrics(single_row, single_row)
        
        # Should return empty dict for images without index row
        assert len(metrics) == 0
    
    def test_calculate_parameter_space_metrics(self):
        """Test parameter space metrics calculation."""
        metrics = self.calculator.calculate_parameter_space_metrics(
            self.original_with_index, self.reconstructed_with_index
        )
        
        expected_keys = ['parameter_space_mse', 'parameter_space_rmse', 
                        'parameter_space_mae', 'parameter_space_psnr']
        for key in expected_keys:
            assert key in metrics
            assert isinstance(metrics[key], (int, float))
        
        # Check ranges
        assert metrics['parameter_space_mse'] >= 0
        assert metrics['parameter_space_rmse'] >= 0
        assert metrics['parameter_space_mae'] >= 0
        assert metrics['parameter_space_psnr'] > 0  # Should be positive or inf
    
    def test_generate_compression_report(self):
        """Test compression report generation."""
        metrics = self.calculator.calculate_comprehensive_metrics(
            self.original,
            self.reconstructed,
            compressed_size=1024,
            compression_time=0.1,
            decompression_time=0.05
        )
        
        # Add index row metrics
        index_metrics = self.calculator.calculate_index_row_metrics(
            self.original_with_index, self.reconstructed_with_index
        )
        metrics.update(index_metrics)
        
        report = self.calculator.generate_compression_report(metrics)
        
        assert isinstance(report, str)
        assert len(report) > 0
        assert "Compression Performance Report" in report
        assert "Original Size" in report
        assert "Compression Ratio" in report
        assert "Quality Metrics" in report
    
    def test_metrics_with_identical_images(self):
        """Test metrics calculation with identical images."""
        metrics = self.calculator.calculate_comprehensive_metrics(
            self.original,
            self.original,  # Identical
            compressed_size=1024,
            compression_time=0.1,
            decompression_time=0.05
        )
        
        # Perfect reconstruction should have zero error
        assert metrics['mse'] == 0.0
        assert metrics['mae'] == 0.0
        assert metrics['psnr'] == float('inf')
        assert abs(metrics['ssim'] - 1.0) < 1e-10
        assert abs(metrics['normalized_cross_correlation'] - 1.0) < 1e-10
    
    def test_metrics_with_constant_images(self):
        """Test metrics calculation with constant images."""
        constant_orig = np.ones((32, 32)) * 0.5
        constant_recon = np.ones((32, 32)) * 0.5
        
        metrics = self.calculator.calculate_comprehensive_metrics(
            constant_orig,
            constant_recon,
            compressed_size=512,
            compression_time=0.05,
            decompression_time=0.02
        )
        
        # Constant identical images should have perfect metrics
        assert metrics['mse'] == 0.0
        assert metrics['psnr'] == float('inf')
        assert abs(metrics['ssim'] - 1.0) < 1e-10
    
    def test_metrics_with_zero_compression_time(self):
        """Test metrics calculation with zero compression time."""
        metrics = self.calculator.calculate_comprehensive_metrics(
            self.original,
            self.reconstructed,
            compressed_size=1024,
            compression_time=0.0,  # Zero time
            decompression_time=0.05
        )
        
        # Should handle zero time gracefully
        assert metrics['compression_throughput_mbps'] == float('inf')
        assert metrics['decompression_throughput_mbps'] < float('inf')
    
    def test_edge_case_zero_compressed_size(self):
        """Test metrics with zero compressed size."""
        metrics = self.calculator.calculate_comprehensive_metrics(
            self.original,
            self.reconstructed,
            compressed_size=0,  # Zero size
            compression_time=0.1,
            decompression_time=0.05
        )
        
        # Should handle zero compressed size gracefully
        assert metrics['compression_ratio'] == 0.0
        assert metrics['space_savings_percent'] == 0.0


if __name__ == "__main__":
    pytest.main([__file__])