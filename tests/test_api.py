"""
Tests for the high-level API interface.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from hilbert_quantization.api import (
    HilbertQuantizer, BatchQuantizer,
    quantize_model, reconstruct_model, search_similar_models
)
from hilbert_quantization.config import SystemConfig, create_default_config
from hilbert_quantization.models import QuantizedModel, SearchResult, ModelMetadata, CompressionMetrics
from hilbert_quantization.exceptions import (
    QuantizationError, ReconstructionError, SearchError, ValidationError, ConfigurationError
)


class TestHilbertQuantizer:
    """Test HilbertQuantizer main functionality."""
    
    @pytest.fixture
    def quantizer(self):
        """Create a quantizer instance for testing."""
        config = create_default_config()
        config.enable_logging = False  # Disable logging for tests
        return HilbertQuantizer(config)
    
    @pytest.fixture
    def sample_parameters(self):
        """Create sample parameters for testing."""
        return np.random.randn(1000).astype(np.float32)
    
    @pytest.fixture
    def mock_quantized_model(self):
        """Create a mock quantized model."""
        return QuantizedModel(
            compressed_data=b"mock_compressed_data",
            original_dimensions=(32, 32),
            parameter_count=1000,
            compression_quality=0.8,
            hierarchical_indices=np.random.randn(32),
            metadata=ModelMetadata(
                model_name="test_model",
                original_size_bytes=4000,
                compressed_size_bytes=2000,
                compression_ratio=0.5,
                quantization_timestamp="2025-08-29T00:00:00",
                model_architecture="test_architecture",
                additional_info={"description": "Test model"}
            )
        )
    
    def test_quantizer_initialization(self):
        """Test quantizer initialization."""
        # Default initialization
        quantizer = HilbertQuantizer()
        assert quantizer.config is not None
        assert quantizer.config_manager is not None
        assert len(quantizer._model_registry) == 0
        
        # Custom config initialization
        config = create_default_config()
        config.compression.quality = 0.9
        quantizer = HilbertQuantizer(config)
        assert quantizer.config.compression.quality == 0.9
    
    @patch('hilbert_quantization.api.QuantizationPipeline')
    def test_quantize_success(self, mock_pipeline_class, quantizer, sample_parameters, mock_quantized_model):
        """Test successful quantization."""
        # Setup mock
        mock_pipeline = Mock()
        mock_pipeline.quantize_model.return_value = mock_quantized_model
        mock_pipeline_class.return_value = mock_pipeline
        
        # Test quantization
        result = quantizer.quantize(sample_parameters, model_id="test", description="Test model")
        
        assert result == mock_quantized_model
        assert len(quantizer._model_registry) == 1
        assert quantizer._model_registry[0] == mock_quantized_model
        mock_pipeline.quantize_model.assert_called_once()
    
    def test_quantize_with_list_input(self, quantizer):
        """Test quantization with list input."""
        parameters_list = [1.0, 2.0, 3.0, 4.0] * 250  # 1000 parameters
        
        with patch.object(quantizer, '_quantization_pipeline') as mock_pipeline:
            mock_model = Mock()
            mock_pipeline.quantize_model.return_value = mock_model
            
            result = quantizer.quantize(parameters_list)
            
            # Verify numpy array conversion
            call_args = mock_pipeline.quantize_model.call_args[0]
            assert isinstance(call_args[0], np.ndarray)
            assert call_args[0].dtype == np.float32
    
    def test_quantize_validation_errors(self, quantizer):
        """Test quantization validation errors."""
        # Empty parameters
        with pytest.raises(ValidationError, match="Parameters array cannot be empty"):
            quantizer.quantize(np.array([]))
        
        # Non-finite values
        params_with_nan = np.array([1.0, 2.0, np.nan, 4.0])
        with pytest.raises(ValidationError, match="Parameters contain non-finite values"):
            quantizer.quantize(params_with_nan)
        
        params_with_inf = np.array([1.0, 2.0, np.inf, 4.0])
        with pytest.raises(ValidationError, match="Parameters contain non-finite values"):
            quantizer.quantize(params_with_inf)
        
        # Multi-dimensional array
        params_2d = np.random.randn(10, 10)
        with pytest.raises(ValidationError, match="Parameters must be a 1D array"):
            quantizer.quantize(params_2d)
    
    @patch('hilbert_quantization.api.ReconstructionPipeline')
    def test_reconstruct_success(self, mock_pipeline_class, quantizer, mock_quantized_model):
        """Test successful reconstruction."""
        # Setup mock
        mock_pipeline = Mock()
        reconstructed_params = np.random.randn(1000).astype(np.float32)
        mock_pipeline.reconstruct.return_value = reconstructed_params
        mock_pipeline_class.return_value = mock_pipeline
        
        # Test reconstruction
        result = quantizer.reconstruct(mock_quantized_model)
        
        assert np.array_equal(result, reconstructed_params)
        mock_pipeline.reconstruct.assert_called_once_with(mock_quantized_model)
    
    def test_reconstruct_validation_errors(self, quantizer):
        """Test reconstruction validation errors."""
        # Invalid model type
        with pytest.raises(ValidationError, match="Invalid quantized model type"):
            quantizer.reconstruct("not_a_model")
        
        # Model with no compressed data
        invalid_model = Mock()
        invalid_model.compressed_data = None
        with pytest.raises(ValidationError, match="Quantized model has no compressed data"):
            quantizer.reconstruct(invalid_model)
        
        # Model with invalid parameter count
        invalid_model = Mock()
        invalid_model.compressed_data = b"data"
        invalid_model.parameter_count = 0
        with pytest.raises(ValidationError, match="Invalid parameter count"):
            quantizer.reconstruct(invalid_model)
    
    def test_reconstruct_validation_mismatch(self, quantizer, mock_quantized_model):
        """Test reconstruction validation with parameter count mismatch."""
        with patch.object(quantizer, 'reconstruction_pipeline') as mock_pipeline:
            # Return wrong number of parameters
            wrong_params = np.random.randn(500).astype(np.float32)  # Should be 1000
            mock_pipeline.reconstruct.return_value = wrong_params
            
            with pytest.raises(ValidationError, match="Reconstructed parameter count"):
                quantizer.reconstruct(mock_quantized_model)
    
    @patch('hilbert_quantization.api.ProgressiveSimilaritySearchEngine')
    def test_search_success(self, mock_engine_class, quantizer, sample_parameters, mock_quantized_model):
        """Test successful search."""
        # Setup mocks
        mock_engine = Mock()
        search_results = [
            SearchResult(
                model=mock_quantized_model,
                similarity_score=0.9,
                matching_indices={0: 0.9, 1: 0.8},
                reconstruction_error=0.01
            )
        ]
        mock_engine.progressive_search.return_value = search_results
        mock_engine_class.return_value = mock_engine
        
        # Add model to registry
        quantizer._model_registry.append(mock_quantized_model)
        
        with patch.object(quantizer, 'quantize') as mock_quantize:
            query_model = Mock()
            query_model.hierarchical_indices = np.random.randn(32)
            mock_quantize.return_value = query_model
            
            # Test search
            results = quantizer.search(sample_parameters)
            
            assert len(results) == 1
            assert results[0].similarity_score == 0.9
            mock_engine.progressive_search.assert_called_once()
    
    def test_search_with_candidates(self, quantizer, sample_parameters, mock_quantized_model):
        """Test search with provided candidate models."""
        candidates = [mock_quantized_model]
        
        with patch.object(quantizer, 'search_engine') as mock_engine:
            with patch.object(quantizer, 'quantize') as mock_quantize:
                query_model = Mock()
                query_model.hierarchical_indices = np.random.randn(32)
                mock_quantize.return_value = query_model
                
                mock_engine.progressive_search.return_value = []
                
                quantizer.search(sample_parameters, candidates)
                
                # Verify candidates were used
                call_args = mock_engine.progressive_search.call_args[0]
                assert call_args[1] == candidates
    
    def test_search_no_candidates_error(self, quantizer, sample_parameters):
        """Test search error when no candidates available."""
        with pytest.raises(SearchError, match="No candidate models available"):
            quantizer.search(sample_parameters)
    
    def test_search_similarity_threshold_filtering(self, quantizer, sample_parameters, mock_quantized_model):
        """Test search results filtering by similarity threshold."""
        quantizer._model_registry.append(mock_quantized_model)
        
        with patch.object(quantizer, 'search_engine') as mock_engine:
            with patch.object(quantizer, 'quantize') as mock_quantize:
                query_model = Mock()
                query_model.hierarchical_indices = np.random.randn(32)
                mock_quantize.return_value = query_model
                
                # Return results with different similarity scores
                search_results = [
                    SearchResult(mock_quantized_model, 0.9, {}, 0.01),
                    SearchResult(mock_quantized_model, 0.05, {}, 0.01),  # Below threshold
                    SearchResult(mock_quantized_model, 0.8, {}, 0.01)
                ]
                mock_engine.progressive_search.return_value = search_results
                
                # Search with threshold 0.1
                results = quantizer.search(sample_parameters, similarity_threshold=0.1)
                
                # Should filter out the 0.05 result
                assert len(results) == 2
                assert all(r.similarity_score >= 0.1 for r in results)
    
    def test_model_registry_operations(self, quantizer, mock_quantized_model):
        """Test model registry operations."""
        # Add model
        quantizer.add_model_to_registry(mock_quantized_model)
        assert len(quantizer._model_registry) == 1
        
        # Get registry info
        info = quantizer.get_registry_info()
        assert info["total_models"] == 1
        assert mock_quantized_model.model_id in info["model_ids"]
        
        # Remove model
        removed = quantizer.remove_model_from_registry(mock_quantized_model.model_id)
        assert removed is True
        assert len(quantizer._model_registry) == 0
        
        # Try to remove non-existent model
        removed = quantizer.remove_model_from_registry("non_existent")
        assert removed is False
        
        # Clear registry
        quantizer.add_model_to_registry(mock_quantized_model)
        quantizer.clear_registry()
        assert len(quantizer._model_registry) == 0
    
    def test_save_and_load_model(self, quantizer, mock_quantized_model):
        """Test saving and loading models."""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name
        
        try:
            # Test save
            quantizer.save_model(mock_quantized_model, temp_path)
            assert Path(temp_path).exists()
            
            # Test load
            loaded_model = quantizer.load_model(temp_path)
            assert loaded_model.model_id == mock_quantized_model.model_id
            assert loaded_model.parameter_count == mock_quantized_model.parameter_count
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_get_compression_metrics(self, quantizer, mock_quantized_model):
        """Test getting compression metrics."""
        metrics = quantizer.get_compression_metrics(mock_quantized_model)
        assert metrics.compression_ratio == 0.5
        assert metrics.reconstruction_error == 0.01
        
        # Test model without metrics
        model_without_metrics = Mock()
        model_without_metrics.metadata.compression_metrics = None
        with pytest.raises(ValidationError, match="Model does not have compression metrics"):
            quantizer.get_compression_metrics(model_without_metrics)
    
    def test_update_configuration(self, quantizer):
        """Test configuration updates."""
        # Valid updates
        quantizer.update_configuration(
            quantization_padding_value=0.5,
            compression_quality=0.9,
            search_max_results=20
        )
        
        assert quantizer.config.quantization.padding_value == 0.5
        assert quantizer.config.compression.quality == 0.9
        assert quantizer.config.search.max_results == 20
        
        # Invalid parameter
        with pytest.raises(ConfigurationError, match="Unknown configuration parameter"):
            quantizer.update_configuration(invalid_param=123)
    
    def test_get_optimal_configuration(self, quantizer):
        """Test getting optimal configuration."""
        config = quantizer.get_optimal_configuration(5000000)  # Medium model
        assert config.compression.quality == 0.8
        assert config.compression.adaptive_quality is True
    
    def test_benchmark_performance(self, quantizer):
        """Test performance benchmarking."""
        with patch.object(quantizer, 'quantize') as mock_quantize:
            with patch.object(quantizer, 'reconstruct') as mock_reconstruct:
                with patch.object(quantizer, 'search') as mock_search:
                    # Setup mocks
                    mock_model = Mock()
                    mock_model.metadata.compression_metrics.compression_ratio = 0.5
                    mock_quantize.return_value = mock_model
                    mock_reconstruct.return_value = np.random.randn(100)
                    mock_search.return_value = []
                    
                    # Run benchmark
                    results = quantizer.benchmark_performance([100, 200], num_trials=2)
                    
                    assert "parameter_counts" in results
                    assert "quantization_times" in results
                    assert "reconstruction_times" in results
                    assert len(results["parameter_counts"]) == 2


class TestBatchQuantizer:
    """Test BatchQuantizer functionality."""
    
    @pytest.fixture
    def batch_quantizer(self):
        """Create a batch quantizer for testing."""
        config = create_default_config()
        config.enable_logging = False
        return BatchQuantizer(config)
    
    @pytest.fixture
    def sample_parameter_sets(self):
        """Create sample parameter sets for testing."""
        return [
            np.random.randn(100).astype(np.float32),
            np.random.randn(200).astype(np.float32),
            np.random.randn(150).astype(np.float32)
        ]
    
    def test_quantize_batch_success(self, batch_quantizer, sample_parameter_sets):
        """Test successful batch quantization."""
        with patch.object(batch_quantizer.quantizer, 'quantize') as mock_quantize:
            mock_models = [Mock() for _ in sample_parameter_sets]
            mock_quantize.side_effect = mock_models
            
            results = batch_quantizer.quantize_batch(sample_parameter_sets)
            
            assert len(results) == 3
            assert mock_quantize.call_count == 3
    
    def test_quantize_batch_with_ids_and_descriptions(self, batch_quantizer, sample_parameter_sets):
        """Test batch quantization with IDs and descriptions."""
        model_ids = ["model1", "model2", "model3"]
        descriptions = ["desc1", "desc2", "desc3"]
        
        with patch.object(batch_quantizer.quantizer, 'quantize') as mock_quantize:
            mock_quantize.return_value = Mock()
            
            batch_quantizer.quantize_batch(
                sample_parameter_sets, 
                model_ids=model_ids, 
                descriptions=descriptions
            )
            
            # Verify IDs and descriptions were passed
            for i, call in enumerate(mock_quantize.call_args_list):
                assert call[1]['model_id'] == model_ids[i]
                assert call[1]['description'] == descriptions[i]
    
    def test_quantize_batch_validation_errors(self, batch_quantizer, sample_parameter_sets):
        """Test batch quantization validation errors."""
        # Mismatched model IDs
        with pytest.raises(ValueError, match="Number of model IDs must match"):
            batch_quantizer.quantize_batch(sample_parameter_sets, model_ids=["id1", "id2"])
        
        # Mismatched descriptions
        with pytest.raises(ValueError, match="Number of descriptions must match"):
            batch_quantizer.quantize_batch(sample_parameter_sets, descriptions=["desc1"])
    
    def test_search_batch(self, batch_quantizer, sample_parameter_sets):
        """Test batch search functionality."""
        mock_candidates = [Mock(), Mock()]
        
        with patch.object(batch_quantizer.quantizer, 'search') as mock_search:
            mock_results = [
                [Mock(), Mock()],  # Results for query 1
                [Mock()],          # Results for query 2
                []                 # No results for query 3
            ]
            mock_search.side_effect = mock_results
            
            results = batch_quantizer.search_batch(sample_parameter_sets, mock_candidates)
            
            assert len(results) == 3
            assert len(results[0]) == 2
            assert len(results[1]) == 1
            assert len(results[2]) == 0


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_quantize_model_function(self):
        """Test quantize_model convenience function."""
        parameters = np.random.randn(100).astype(np.float32)
        
        with patch('hilbert_quantization.api.HilbertQuantizer') as mock_quantizer_class:
            mock_quantizer = Mock()
            mock_model = Mock()
            mock_quantizer.quantize.return_value = mock_model
            mock_quantizer_class.return_value = mock_quantizer
            
            result = quantize_model(parameters)
            
            assert result == mock_model
            mock_quantizer.quantize.assert_called_once_with(parameters)
    
    def test_reconstruct_model_function(self):
        """Test reconstruct_model convenience function."""
        mock_model = Mock()
        
        with patch('hilbert_quantization.api.HilbertQuantizer') as mock_quantizer_class:
            mock_quantizer = Mock()
            mock_params = np.random.randn(100)
            mock_quantizer.reconstruct.return_value = mock_params
            mock_quantizer_class.return_value = mock_quantizer
            
            result = reconstruct_model(mock_model)
            
            assert np.array_equal(result, mock_params)
            mock_quantizer.reconstruct.assert_called_once_with(mock_model)
    
    def test_search_similar_models_function(self):
        """Test search_similar_models convenience function."""
        query_params = np.random.randn(100).astype(np.float32)
        candidates = [Mock(), Mock()]
        
        with patch('hilbert_quantization.api.HilbertQuantizer') as mock_quantizer_class:
            mock_quantizer = Mock()
            mock_results = [Mock(), Mock()]
            mock_quantizer.search.return_value = mock_results
            mock_quantizer_class.return_value = mock_quantizer
            
            results = search_similar_models(query_params, candidates, max_results=5)
            
            assert results == mock_results
            mock_quantizer.search.assert_called_once_with(query_params, candidates, 5)


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.fixture
    def quantizer(self):
        """Create a quantizer for error testing."""
        config = create_default_config()
        config.enable_logging = False
        return HilbertQuantizer(config)
    
    def test_quantization_pipeline_error(self, quantizer):
        """Test handling of quantization pipeline errors."""
        parameters = np.random.randn(100).astype(np.float32)
        
        with patch.object(quantizer, 'quantization_pipeline') as mock_pipeline:
            mock_pipeline.quantize.side_effect = Exception("Pipeline error")
            
            with pytest.raises(QuantizationError, match="Unexpected error during quantization"):
                quantizer.quantize(parameters)
    
    def test_reconstruction_pipeline_error(self, quantizer):
        """Test handling of reconstruction pipeline errors."""
        mock_model = Mock()
        mock_model.compressed_data = b"data"
        mock_model.parameter_count = 100
        
        with patch.object(quantizer, 'reconstruction_pipeline') as mock_pipeline:
            mock_pipeline.reconstruct.side_effect = Exception("Pipeline error")
            
            with pytest.raises(ReconstructionError, match="Unexpected error during reconstruction"):
                quantizer.reconstruct(mock_model)
    
    def test_search_engine_error(self, quantizer):
        """Test handling of search engine errors."""
        parameters = np.random.randn(100).astype(np.float32)
        quantizer._model_registry.append(Mock())
        
        with patch.object(quantizer, 'quantize') as mock_quantize:
            with patch.object(quantizer, 'search_engine') as mock_engine:
                mock_quantize.return_value = Mock()
                mock_engine.progressive_search.side_effect = Exception("Search error")
                
                with pytest.raises(SearchError, match="Unexpected error during search"):
                    quantizer.search(parameters)
    
    def test_file_operation_errors(self, quantizer):
        """Test file operation error handling."""
        mock_model = Mock()
        
        # Test save error
        with patch('builtins.open', side_effect=IOError("File error")):
            with pytest.raises(QuantizationError, match="Failed to save model"):
                quantizer.save_model(mock_model, "invalid_path")
        
        # Test load error
        with patch('builtins.open', side_effect=IOError("File error")):
            with pytest.raises(QuantizationError, match="Failed to load model"):
                quantizer.load_model("invalid_path")
    
    def test_configuration_update_error(self, quantizer):
        """Test configuration update error handling."""
        with patch.object(quantizer.config_manager, 'update_quantization_config', 
                         side_effect=ValueError("Config error")):
            with pytest.raises(ConfigurationError, match="Failed to update configuration"):
                quantizer.update_configuration(quantization_invalid_param=123)


class TestIntegrationScenarios:
    """Test integration scenarios and workflows."""
    
    @pytest.fixture
    def quantizer(self):
        """Create quantizer for integration tests."""
        config = create_default_config()
        config.enable_logging = False
        return HilbertQuantizer(config)
    
    def test_complete_workflow(self, quantizer):
        """Test complete quantization -> search -> reconstruction workflow."""
        # Create test data
        params1 = np.random.randn(100).astype(np.float32)
        params2 = np.random.randn(100).astype(np.float32)
        query_params = params1 + 0.1 * np.random.randn(100).astype(np.float32)  # Similar to params1
        
        with patch.object(quantizer, 'quantization_pipeline') as mock_quant_pipeline:
            with patch.object(quantizer, 'reconstruction_pipeline') as mock_recon_pipeline:
                with patch.object(quantizer, 'search_engine') as mock_search_engine:
                    # Setup mocks
                    mock_model1 = Mock()
                    mock_model1.model_id = "model1"
                    mock_model1.hierarchical_indices = np.random.randn(32)
                    
                    mock_model2 = Mock()
                    mock_model2.model_id = "model2"
                    mock_model2.hierarchical_indices = np.random.randn(32)
                    
                    mock_query_model = Mock()
                    mock_query_model.hierarchical_indices = np.random.randn(32)
                    
                    mock_quant_pipeline.quantize.side_effect = [mock_model1, mock_model2, mock_query_model]
                    mock_recon_pipeline.reconstruct.return_value = params1
                    
                    search_results = [
                        SearchResult(mock_model1, 0.9, {}, 0.01),
                        SearchResult(mock_model2, 0.3, {}, 0.05)
                    ]
                    mock_search_engine.progressive_search.return_value = search_results
                    
                    # Execute workflow
                    # 1. Quantize models
                    model1 = quantizer.quantize(params1, "model1")
                    model2 = quantizer.quantize(params2, "model2")
                    
                    # 2. Search for similar models
                    results = quantizer.search(query_params, max_results=5)
                    
                    # 3. Reconstruct best match
                    best_match = results[0].model
                    reconstructed = quantizer.reconstruct(best_match)
                    
                    # Verify workflow
                    assert len(results) == 2
                    assert results[0].similarity_score > results[1].similarity_score
                    assert np.array_equal(reconstructed, params1)
    
    def test_registry_persistence_workflow(self, quantizer):
        """Test model registry with save/load workflow."""
        params = np.random.randn(100).astype(np.float32)
        
        with patch.object(quantizer, 'quantization_pipeline') as mock_pipeline:
            mock_model = Mock()
            mock_model.model_id = "persistent_model"
            mock_pipeline.quantize.return_value = mock_model
            
            # Quantize and add to registry
            quantized = quantizer.quantize(params, "persistent_model")
            
            # Verify in registry
            info = quantizer.get_registry_info()
            assert info["total_models"] == 1
            assert "persistent_model" in info["model_ids"]
            
            # Test save/load cycle
            with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
                temp_path = f.name
            
            try:
                quantizer.save_model(quantized, temp_path)
                loaded_model = quantizer.load_model(temp_path)
                
                # Add loaded model to new quantizer
                new_quantizer = HilbertQuantizer()
                new_quantizer.add_model_to_registry(loaded_model)
                
                new_info = new_quantizer.get_registry_info()
                assert new_info["total_models"] == 1
                assert loaded_model.model_id in new_info["model_ids"]
            finally:
                Path(temp_path).unlink(missing_ok=True)