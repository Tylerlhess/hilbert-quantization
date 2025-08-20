"""
Integration tests for MPEG-AI compression functionality.
"""

import pytest
import numpy as np
from hilbert_quantization.core.compressor import MPEGAICompressorImpl, CompressionMetricsCalculator
from hilbert_quantization.config import CompressionConfig


class TestCompressionIntegration:
    """Integration tests for compression functionality."""
    
    def test_end_to_end_compression_workflow(self):
        """Test complete compression workflow with metrics."""
        # Create test data
        original_image = np.random.rand(64, 65).astype(np.float32)  # With index row
        original_image[-1, :] = np.linspace(0, 1, 65)  # Index row
        
        # Configure compressor
        config = CompressionConfig(
            quality=0.8,
            preserve_index_row=True,
            validate_reconstruction=True,
            max_reconstruction_error=0.1
        )
        compressor = MPEGAICompressorImpl(config)
        
        # Perform compression with index preservation
        compressed_data = compressor.compress_with_index_preservation(original_image, quality=0.8)
        reconstructed_image = compressor.decompress_with_index_preservation(compressed_data)
        
        # Validate basic properties
        assert reconstructed_image.shape == original_image.shape
        assert isinstance(compressed_data, bytes)
        assert len(compressed_data) > 0
        
        # Get comprehensive metrics
        comprehensive_metrics = compressor.get_comprehensive_metrics(original_image, reconstructed_image)
        
        # Validate metrics
        assert comprehensive_metrics['compression_ratio'] > 1.0
        assert comprehensive_metrics['mse'] >= 0
        assert comprehensive_metrics['psnr'] > 0
        assert -1 <= comprehensive_metrics['ssim'] <= 1
        
        # Get index-specific metrics
        index_metrics = CompressionMetricsCalculator.calculate_index_row_metrics(
            original_image, reconstructed_image
        )
        
        # Validate index preservation
        assert 'index_row_mse' in index_metrics
        assert 'index_preservation_ratio' in index_metrics
        assert index_metrics['index_preservation_ratio'] >= 0
        
        # Generate report
        report = CompressionMetricsCalculator.generate_compression_report(comprehensive_metrics)
        assert isinstance(report, str)
        assert "Compression Performance Report" in report
        
        print(f"Compression successful! Ratio: {comprehensive_metrics['compression_ratio']:.2f}x")
        print(f"PSNR: {comprehensive_metrics['psnr']:.2f} dB")
        print(f"Index preservation: {index_metrics['index_preservation_ratio']:.4f}")
    
    def test_compression_quality_impact(self):
        """Test impact of different compression qualities."""
        original_image = np.random.rand(32, 33).astype(np.float32)
        original_image[-1, :] = np.linspace(0, 1, 33)  # Index row
        
        compressor = MPEGAICompressorImpl()
        qualities = [0.3, 0.6, 0.9]
        results = []
        
        for quality in qualities:
            compressed = compressor.compress(original_image, quality)
            reconstructed = compressor.decompress(compressed)
            
            metrics = compressor.get_comprehensive_metrics(original_image, reconstructed)
            results.append({
                'quality': quality,
                'compression_ratio': metrics['compression_ratio'],
                'psnr': metrics['psnr'],
                'mse': metrics['mse']
            })
        
        # Higher quality should generally result in better reconstruction
        # (though compression ratio might be lower)
        assert len(results) == 3
        for result in results:
            assert result['compression_ratio'] > 0
            assert result['psnr'] > 0
            assert result['mse'] >= 0
    
    def test_model_performance_assessment(self):
        """Test comprehensive model performance assessment."""
        # Create a more realistic parameter-like image
        original_image = np.random.normal(0, 1, (128, 129)).astype(np.float32)
        original_image[-1, :] = np.random.uniform(0, 1, 129)  # Index row
        
        compressor = MPEGAICompressorImpl()
        compressed = compressor.compress(original_image, quality=0.7)
        reconstructed = compressor.decompress(compressed)
        
        # Assess compression impact
        impact_metrics = compressor.assess_compression_impact_on_model_performance(
            original_image, reconstructed
        )
        
        # Validate all expected metrics are present
        expected_keys = [
            'mse', 'rmse', 'psnr', 'ssim', 
            'index_row_mse', 'index_preservation_ratio', 'parameter_space_mse'
        ]
        for key in expected_keys:
            assert key in impact_metrics
        
        # Get parameter space specific metrics
        param_metrics = CompressionMetricsCalculator.calculate_parameter_space_metrics(
            original_image, reconstructed
        )
        
        assert 'parameter_space_mse' in param_metrics
        assert 'parameter_space_psnr' in param_metrics
        
        print(f"Parameter space PSNR: {param_metrics['parameter_space_psnr']:.2f} dB")
        print(f"Index preservation ratio: {impact_metrics['index_preservation_ratio']:.4f}")


if __name__ == "__main__":
    pytest.main([__file__])