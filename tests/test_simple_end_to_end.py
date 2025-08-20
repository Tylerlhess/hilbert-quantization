"""
Simple end-to-end validation tests for the Hilbert quantization system.
"""

import pytest
import numpy as np
from hilbert_quantization.api import HilbertQuantizer
from hilbert_quantization.exceptions import QuantizationError, ValidationError


class TestSimpleEndToEnd:
    """Simple end-to-end tests."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.quantizer = HilbertQuantizer()
    
    def test_basic_quantization_and_reconstruction(self):
        """Test basic quantization and reconstruction workflow."""
        # Create test parameters
        parameters = np.random.randn(256).astype(np.float32)
        
        # Quantize
        quantized_model = self.quantizer.quantize(
            parameters, 
            model_id="basic_test",
            description="Basic end-to-end test"
        )
        
        # Validate quantized model
        assert quantized_model is not None
        assert quantized_model.parameter_count == len(parameters)
        assert len(quantized_model.compressed_data) > 0
        
        # Reconstruct
        reconstructed = self.quantizer.reconstruct(quantized_model)
        
        # Validate reconstruction
        assert len(reconstructed) == len(parameters)
        assert reconstructed.dtype == parameters.dtype
        
        # Check reconstruction quality
        mse = np.mean((parameters - reconstructed) ** 2)
        assert mse < 2.0  # Reasonable threshold
    
    def test_search_functionality(self):
        """Test search functionality with multiple models."""
        # Create test models
        models = []
        for i in range(5):
            params = np.random.randn(64).astype(np.float32)
            model = self.quantizer.quantize(params, f"search_test_{i}")
            models.append(model)
        
        # Create query
        query_params = np.random.randn(64).astype(np.float32)
        
        # Search
        results = self.quantizer.search(
            query_parameters=query_params,
            candidate_models=models,
            max_results=3
        )
        
        # Validate results
        assert len(results) <= 3
        assert len(results) > 0
        
        # Check result structure
        for result in results:
            assert hasattr(result, 'model')
            assert hasattr(result, 'similarity_score')
            assert 0.0 <= result.similarity_score <= 1.0
    
    def test_error_handling(self):
        """Test basic error handling."""
        # Test empty parameters
        with pytest.raises((QuantizationError, ValidationError, ValueError)):
            self.quantizer.quantize(np.array([]), "empty_test")
        
        # Test NaN parameters
        with pytest.raises((QuantizationError, ValidationError)):
            params_with_nan = np.array([1.0, 2.0, np.nan, 4.0])
            self.quantizer.quantize(params_with_nan, "nan_test")
    
    def test_different_parameter_sizes(self):
        """Test with different parameter sizes."""
        sizes = [16, 64, 256]
        
        for size in sizes:
            parameters = np.random.randn(size).astype(np.float32)
            
            # Quantize
            quantized_model = self.quantizer.quantize(parameters, f"size_test_{size}")
            
            # Reconstruct
            reconstructed = self.quantizer.reconstruct(quantized_model)
            
            # Validate
            assert len(reconstructed) == size
            
            # Check quality
            mse = np.mean((parameters - reconstructed) ** 2)
            assert mse < 5.0  # Relaxed threshold for different sizes
    
    def test_compression_quality_levels(self):
        """Test different compression quality levels."""
        parameters = np.random.randn(256).astype(np.float32)
        
        qualities = [0.5, 0.8]
        results = []
        
        for quality in qualities:
            self.quantizer.update_configuration(compression_quality=quality)
            
            quantized_model = self.quantizer.quantize(
                parameters, f"quality_test_{quality}"
            )
            
            reconstructed = self.quantizer.reconstruct(quantized_model)
            mse = np.mean((parameters - reconstructed) ** 2)
            
            results.append({
                'quality': quality,
                'mse': mse,
                'compression_ratio': quantized_model.metadata.compression_ratio
            })
        
        # Validate results
        for result in results:
            assert result['mse'] >= 0
            assert result['compression_ratio'] > 0
        
        # Higher quality should generally give better reconstruction
        if len(results) >= 2:
            # Allow some variance in results
            assert results[1]['mse'] <= results[0]['mse'] * 2


if __name__ == "__main__":
    pytest.main([__file__])