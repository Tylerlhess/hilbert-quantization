"""
Integration tests for the complete quantization pipeline.

Tests the end-to-end workflow including dimension calculation, Hilbert mapping,
index generation, compression, and reconstruction.
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch

from hilbert_quantization.core.pipeline import QuantizationPipeline, ReconstructionPipeline
from hilbert_quantization.models import QuantizedModel, ModelMetadata
from hilbert_quantization.exceptions import HilbertQuantizationError
from hilbert_quantization.config import CompressionConfig


class TestQuantizationPipeline:
    """Test cases for the complete quantization pipeline."""
    
    def test_end_to_end_quantization_small_model(self):
        """Test complete quantization workflow with small model."""
        # Create test parameters (64 parameters -> 8x8 grid)
        parameters = np.random.randn(64).astype(np.float32)
        
        pipeline = QuantizationPipeline()
        
        # Quantize
        quantized_model = pipeline.quantize_model(
            parameters=parameters,
            model_name="test_small_model",
            compression_quality=0.8
        )
        
        # Validate quantized model structure
        assert isinstance(quantized_model, QuantizedModel)
        assert quantized_model.parameter_count == 64
        assert quantized_model.original_dimensions == (8, 8)
        assert quantized_model.compression_quality == 0.8
        assert len(quantized_model.compressed_data) > 0
        assert len(quantized_model.hierarchical_indices) > 0
        
        # Validate metadata
        assert quantized_model.metadata.model_name == "test_small_model"
        assert quantized_model.metadata.original_size_bytes == parameters.nbytes
        assert quantized_model.metadata.compression_ratio > 0
        
        # Reconstruct and validate
        reconstructed = pipeline.reconstruct_parameters(quantized_model)
        
        assert len(reconstructed) == len(parameters)
        assert reconstructed.dtype == parameters.dtype
        
        # Check reconstruction quality (should be reasonable for this quality level)
        mse = np.mean((parameters - reconstructed) ** 2)
        assert mse < 1.0  # Reasonable threshold for 0.8 quality
    
    def test_end_to_end_quantization_medium_model(self):
        """Test complete quantization workflow with medium model."""
        # Create test parameters (1024 parameters -> 32x32 grid)
        parameters = np.random.randn(1024).astype(np.float32)
        
        pipeline = QuantizationPipeline()
        
        # Quantize with high quality
        quantized_model = pipeline.quantize_model(
            parameters=parameters,
            model_name="test_medium_model",
            compression_quality=0.9,
            model_architecture="test_architecture"
        )
        
        # Validate structure
        assert quantized_model.parameter_count == 1024
        assert quantized_model.original_dimensions == (32, 32)
        assert quantized_model.metadata.model_architecture == "test_architecture"
        
        # Reconstruct and validate
        reconstructed = pipeline.reconstruct_parameters(quantized_model)
        
        assert len(reconstructed) == len(parameters)
        
        # Higher quality should give better reconstruction
        mse = np.mean((parameters - reconstructed) ** 2)
        assert mse < 0.5  # Better threshold for higher quality
    
    def test_quantization_with_padding(self):
        """Test quantization with parameters that require padding."""
        # Use 60 parameters (needs padding to reach 64 for 8x8 grid)
        parameters = np.random.randn(60).astype(np.float32)
        
        pipeline = QuantizationPipeline()
        
        quantized_model = pipeline.quantize_model(
            parameters=parameters,
            model_name="test_padded_model",
            compression_quality=0.7
        )
        
        # Should still work correctly
        assert quantized_model.parameter_count == 60
        assert quantized_model.original_dimensions == (8, 8)  # Padded to 64
        
        # Reconstruct should return exactly 60 parameters
        reconstructed = pipeline.reconstruct_parameters(quantized_model)
        assert len(reconstructed) == 60
    
    def test_quantization_validation(self):
        """Test quantization validation functionality."""
        parameters = np.random.randn(256).astype(np.float32)
        
        pipeline = QuantizationPipeline()
        
        quantized_model = pipeline.quantize_model(
            parameters=parameters,
            model_name="test_validation",
            compression_quality=0.8
        )
        
        # Validate quantization
        validation_results = pipeline.validate_quantization(
            original_parameters=parameters,
            quantized_model=quantized_model,
            tolerance=1.0
        )
        
        # Check validation results
        assert 'is_valid' in validation_results
        assert 'mse' in validation_results
        assert 'mae' in validation_results
        assert 'max_error' in validation_results
        assert 'parameter_count_match' in validation_results
        assert 'compression_ratio' in validation_results
        
        assert validation_results['parameter_count_match'] is True
        assert validation_results['compression_ratio'] > 0
        assert validation_results['mse'] >= 0
    
    def test_quantization_with_custom_metadata(self):
        """Test quantization with custom metadata."""
        parameters = np.random.randn(64).astype(np.float32)
        
        custom_metadata = {
            'layer_count': 5,
            'activation_function': 'relu',
            'optimizer': 'adam'
        }
        
        pipeline = QuantizationPipeline()
        
        quantized_model = pipeline.quantize_model(
            parameters=parameters,
            model_name="test_custom_metadata",
            compression_quality=0.8,
            model_architecture="custom_cnn",
            additional_metadata=custom_metadata
        )
        
        # Check metadata preservation
        assert quantized_model.metadata.model_architecture == "custom_cnn"
        assert quantized_model.metadata.additional_info == custom_metadata
        assert quantized_model.metadata.additional_info['layer_count'] == 5
    
    def test_quantization_different_qualities(self):
        """Test quantization with different quality levels."""
        parameters = np.random.randn(256).astype(np.float32)
        
        pipeline = QuantizationPipeline()
        
        qualities = [0.3, 0.5, 0.7, 0.9]
        results = []
        
        for quality in qualities:
            quantized_model = pipeline.quantize_model(
                parameters=parameters,
                model_name=f"test_quality_{quality}",
                compression_quality=quality
            )
            
            reconstructed = pipeline.reconstruct_parameters(quantized_model)
            mse = np.mean((parameters - reconstructed) ** 2)
            
            results.append({
                'quality': quality,
                'compression_ratio': quantized_model.metadata.compression_ratio,
                'mse': mse
            })
        
        # Higher quality should generally give better reconstruction (lower MSE)
        # and lower compression ratio
        for i in range(len(results) - 1):
            current = results[i]
            next_result = results[i + 1]
            
            # Higher quality should have lower or similar MSE
            assert next_result['mse'] <= current['mse'] * 2  # Allow some variance
    
    def test_quantization_error_handling(self):
        """Test error handling in quantization pipeline."""
        pipeline = QuantizationPipeline()
        
        # Test with invalid parameters
        with pytest.raises(HilbertQuantizationError):
            pipeline.quantize_model(
                parameters=np.array([]),  # Empty array
                model_name="test_error",
                compression_quality=0.8
            )
        
        # Test with invalid quality
        with pytest.raises((ValueError, HilbertQuantizationError)):
            pipeline.quantize_model(
                parameters=np.random.randn(64),
                model_name="test_error",
                compression_quality=1.5  # Invalid quality
            )
    
    def test_reconstruction_error_handling(self):
        """Test error handling in reconstruction."""
        pipeline = QuantizationPipeline()
        
        # Create a valid quantized model first
        parameters = np.random.randn(64).astype(np.float32)
        quantized_model = pipeline.quantize_model(
            parameters=parameters,
            model_name="test_reconstruction_error",
            compression_quality=0.8
        )
        
        # Corrupt the compressed data
        corrupted_model = QuantizedModel(
            compressed_data=b"invalid_data",
            original_dimensions=quantized_model.original_dimensions,
            parameter_count=quantized_model.parameter_count,
            compression_quality=quantized_model.compression_quality,
            hierarchical_indices=quantized_model.hierarchical_indices,
            metadata=quantized_model.metadata
        )
        
        # Should raise error on reconstruction
        with pytest.raises(HilbertQuantizationError):
            pipeline.reconstruct_parameters(corrupted_model)
    
    def test_pipeline_info(self):
        """Test pipeline information retrieval."""
        pipeline = QuantizationPipeline()
        
        info = pipeline.get_pipeline_info()
        
        assert 'dimension_calculator' in info
        assert 'hilbert_mapper' in info
        assert 'index_generator' in info
        assert 'compressor' in info
        assert 'compression_config' in info
        
        # Check that component names are reasonable
        assert 'Calculator' in info['dimension_calculator']
        assert 'Mapper' in info['hilbert_mapper']
        assert 'Generator' in info['index_generator']
        assert 'Compressor' in info['compressor']


class TestReconstructionPipeline:
    """Test cases for the specialized reconstruction pipeline."""
    
    def test_reconstruction_with_validation(self):
        """Test reconstruction with comprehensive validation."""
        # Create a quantized model first
        parameters = np.random.randn(256).astype(np.float32)
        
        quantization_pipeline = QuantizationPipeline()
        quantized_model = quantization_pipeline.quantize_model(
            parameters=parameters,
            model_name="test_reconstruction_validation",
            compression_quality=0.8
        )
        
        # Use reconstruction pipeline
        reconstruction_pipeline = ReconstructionPipeline()
        
        reconstructed, validation_metrics = reconstruction_pipeline.reconstruct_with_validation(
            quantized_model=quantized_model,
            validate_indices=True,
            validate_dimensions=True
        )
        
        # Check reconstruction
        assert len(reconstructed) == len(parameters)
        
        # Check validation metrics
        assert 'success' in validation_metrics
        assert 'parameter_count_correct' in validation_metrics
        assert 'reconstruction_time' in validation_metrics
        assert 'dimension_mismatch' in validation_metrics
        assert 'index_integrity_preserved' in validation_metrics
        
        assert validation_metrics['success'] is True
        assert validation_metrics['parameter_count_correct'] is True
        assert validation_metrics['dimension_mismatch'] is False
        assert validation_metrics['reconstruction_time'] > 0
    
    def test_batch_reconstruction(self):
        """Test batch reconstruction of multiple models."""
        # Create multiple quantized models
        quantization_pipeline = QuantizationPipeline()
        quantized_models = []
        
        for i in range(3):
            parameters = np.random.randn(64).astype(np.float32)
            quantized_model = quantization_pipeline.quantize_model(
                parameters=parameters,
                model_name=f"test_batch_{i}",
                compression_quality=0.8
            )
            quantized_models.append(quantized_model)
        
        # Batch reconstruct
        reconstruction_pipeline = ReconstructionPipeline()
        results = reconstruction_pipeline.batch_reconstruct(quantized_models)
        
        # Check results
        assert len(results) == 3
        
        for i, (reconstructed, metrics) in enumerate(results):
            assert len(reconstructed) == 64
            assert metrics['success'] is True
            assert 'reconstruction_time' in metrics
    
    def test_batch_reconstruction_with_errors(self):
        """Test batch reconstruction with some failing models."""
        # Create one valid and one invalid model
        quantization_pipeline = QuantizationPipeline()
        
        # Valid model
        parameters = np.random.randn(64).astype(np.float32)
        valid_model = quantization_pipeline.quantize_model(
            parameters=parameters,
            model_name="test_valid",
            compression_quality=0.8
        )
        
        # Invalid model (corrupted data)
        invalid_model = QuantizedModel(
            compressed_data=b"invalid",
            original_dimensions=(8, 8),
            parameter_count=64,
            compression_quality=0.8,
            hierarchical_indices=np.array([1, 2, 3]),
            metadata=ModelMetadata(
                model_name="test_invalid",
                original_size_bytes=256,
                compressed_size_bytes=100,
                compression_ratio=2.56,
                quantization_timestamp="2024-01-01 00:00:00"
            )
        )
        
        models = [valid_model, invalid_model]
        
        # Batch reconstruct
        reconstruction_pipeline = ReconstructionPipeline()
        results = reconstruction_pipeline.batch_reconstruct(models)
        
        # Check results
        assert len(results) == 2
        
        # First should succeed
        reconstructed_1, metrics_1 = results[0]
        assert len(reconstructed_1) == 64
        assert metrics_1['success'] is True
        
        # Second should fail
        reconstructed_2, metrics_2 = results[1]
        assert len(reconstructed_2) == 0
        assert metrics_2['success'] is False
        assert 'error' in metrics_2


class TestPipelineIntegration:
    """Integration tests for complete pipeline workflows."""
    
    def test_round_trip_consistency(self):
        """Test that multiple round trips maintain consistency."""
        parameters = np.random.randn(256).astype(np.float32)
        
        pipeline = QuantizationPipeline()
        
        # First round trip
        quantized_1 = pipeline.quantize_model(
            parameters=parameters,
            model_name="test_round_trip_1",
            compression_quality=0.9
        )
        reconstructed_1 = pipeline.reconstruct_parameters(quantized_1)
        
        # Second round trip using reconstructed parameters
        quantized_2 = pipeline.quantize_model(
            parameters=reconstructed_1,
            model_name="test_round_trip_2",
            compression_quality=0.9
        )
        reconstructed_2 = pipeline.reconstruct_parameters(quantized_2)
        
        # Should be very similar (within compression error bounds)
        mse_1 = np.mean((parameters - reconstructed_1) ** 2)
        mse_2 = np.mean((reconstructed_1 - reconstructed_2) ** 2)
        
        # Second round trip error should be similar to first
        assert mse_2 <= mse_1 * 2  # Allow some accumulation but not too much
    
    def test_different_parameter_sizes(self):
        """Test pipeline with various parameter sizes."""
        sizes = [16, 64, 256, 1024]  # All powers of 4
        
        pipeline = QuantizationPipeline()
        
        for size in sizes:
            parameters = np.random.randn(size).astype(np.float32)
            
            quantized_model = pipeline.quantize_model(
                parameters=parameters,
                model_name=f"test_size_{size}",
                compression_quality=0.8
            )
            
            reconstructed = pipeline.reconstruct_parameters(quantized_model)
            
            assert len(reconstructed) == size
            
            # Validate dimensions are correct power of 4
            expected_dim = int(np.sqrt(size))
            assert quantized_model.original_dimensions == (expected_dim, expected_dim)
    
    def test_performance_benchmarking(self):
        """Test performance characteristics of the pipeline."""
        parameters = np.random.randn(1024).astype(np.float32)
        
        pipeline = QuantizationPipeline()
        
        # Measure quantization time
        start_time = time.time()
        quantized_model = pipeline.quantize_model(
            parameters=parameters,
            model_name="test_performance",
            compression_quality=0.8
        )
        quantization_time = time.time() - start_time
        
        # Measure reconstruction time
        start_time = time.time()
        reconstructed = pipeline.reconstruct_parameters(quantized_model)
        reconstruction_time = time.time() - start_time
        
        # Performance should be reasonable
        assert quantization_time < 10.0  # Should complete within 10 seconds
        assert reconstruction_time < 5.0   # Reconstruction should be faster
        
        # Check compression effectiveness
        assert quantized_model.metadata.compression_ratio > 1.0
        
        # Check reconstruction quality
        mse = np.mean((parameters - reconstructed) ** 2)
        assert mse < 1.0  # Reasonable quality threshold
    
    @pytest.mark.parametrize("compression_quality", [0.3, 0.5, 0.7, 0.9])
    def test_quality_impact_on_performance(self, compression_quality):
        """Test how compression quality affects performance and accuracy."""
        parameters = np.random.randn(256).astype(np.float32)
        
        pipeline = QuantizationPipeline()
        
        quantized_model = pipeline.quantize_model(
            parameters=parameters,
            model_name=f"test_quality_{compression_quality}",
            compression_quality=compression_quality
        )
        
        reconstructed = pipeline.reconstruct_parameters(quantized_model)
        
        # Validate basic functionality
        assert len(reconstructed) == len(parameters)
        assert quantized_model.compression_quality == compression_quality
        
        # Calculate metrics
        mse = np.mean((parameters - reconstructed) ** 2)
        compression_ratio = quantized_model.metadata.compression_ratio
        
        # Basic sanity checks
        assert mse >= 0
        assert compression_ratio > 0
        
        # Higher quality should generally give better reconstruction
        # (though this is not guaranteed due to compression algorithm specifics)
        if compression_quality >= 0.8:
            assert mse < 2.0  # High quality should have reasonable error