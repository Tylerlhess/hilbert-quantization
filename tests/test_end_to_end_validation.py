"""
End-to-end validation tests for the Hilbert quantization system.

This module implements comprehensive validation tests with real neural network models,
model performance preservation validation, and comprehensive error handling tests.
"""

import pytest
import numpy as np
import time
from typing import List, Dict, Any, Tuple, Optional
from unittest.mock import Mock, patch
import tempfile
import os

from hilbert_quantization.api import HilbertQuantizer, BatchQuantizer
from hilbert_quantization.models import QuantizedModel, SearchResult
from hilbert_quantization.config import create_default_config, SystemConfig
from hilbert_quantization.exceptions import (
    QuantizationError, CompressionError, SearchError, ReconstructionError,
    ConfigurationError, ValidationError, HilbertQuantizationError
)


class MockNeuralNetwork:
    """Mock neural network for testing purposes."""
    
    def __init__(self, layer_sizes: List[int], activation: str = "relu"):
        """Initialize mock neural network."""
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.weights = self._initialize_weights()
        self.biases = self._initialize_biases()
        
    def _initialize_weights(self) -> List[np.ndarray]:
        """Initialize random weights."""
        weights = []
        for i in range(len(self.layer_sizes) - 1):
            w = np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]).astype(np.float32)
            weights.append(w)
        return weights
    
    def _initialize_biases(self) -> List[np.ndarray]:
        """Initialize random biases."""
        biases = []
        for i in range(1, len(self.layer_sizes)):
            b = np.random.randn(self.layer_sizes[i]).astype(np.float32)
            biases.append(b)
        return biases
    
    def get_parameters(self) -> np.ndarray:
        """Get flattened parameters."""
        params = []
        for w in self.weights:
            params.extend(w.flatten())
        for b in self.biases:
            params.extend(b.flatten())
        return np.array(params, dtype=np.float32)
    
    def set_parameters(self, parameters: np.ndarray) -> None:
        """Set parameters from flattened array."""
        idx = 0
        
        # Set weights
        for i, w in enumerate(self.weights):
            size = w.size
            self.weights[i] = parameters[idx:idx+size].reshape(w.shape)
            idx += size
        
        # Set biases
        for i, b in enumerate(self.biases):
            size = b.size
            self.biases[i] = parameters[idx:idx+size].reshape(b.shape)
            idx += size
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the network."""
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = np.dot(x, w) + b
            if i < len(self.weights) - 1:  # Apply activation except for output layer
                if self.activation == "relu":
                    x = np.maximum(0, x)
                elif self.activation == "sigmoid":
                    x = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
                elif self.activation == "tanh":
                    x = np.tanh(x)
        return x
    
    def evaluate_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate classification accuracy."""
        predictions = self.forward(X)
        if predictions.shape[1] > 1:  # Multi-class
            pred_classes = np.argmax(predictions, axis=1)
            true_classes = np.argmax(y, axis=1) if y.shape[1] > 1 else y.astype(int)
        else:  # Binary
            pred_classes = (predictions > 0.5).astype(int).flatten()
            true_classes = y.astype(int).flatten()
        
        return np.mean(pred_classes == true_classes)
    
    def calculate_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate mean squared error loss."""
        predictions = self.forward(X)
        return np.mean((predictions - y) ** 2)


class EndToEndValidationTest:
    """Base class for end-to-end validation tests."""
    
    def __init__(self):
        """Initialize validation test utilities."""
        self.quantizer = HilbertQuantizer()
        self.batch_quantizer = BatchQuantizer()
        
    def create_test_dataset(self, num_samples: int = 100, 
                           input_dim: int = 10, 
                           output_dim: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Create synthetic test dataset."""
        X = np.random.randn(num_samples, input_dim).astype(np.float32)
        
        if output_dim == 1:
            # Binary classification or regression
            y = (np.sum(X, axis=1, keepdims=True) > 0).astype(np.float32)
        else:
            # Multi-class classification
            y = np.zeros((num_samples, output_dim), dtype=np.float32)
            classes = np.random.randint(0, output_dim, num_samples)
            y[np.arange(num_samples), classes] = 1
        
        return X, y
    
    def create_test_networks(self) -> List[MockNeuralNetwork]:
        """Create various test neural networks."""
        networks = [
            # Small network
            MockNeuralNetwork([10, 8, 1], "relu"),
            # Medium network
            MockNeuralNetwork([20, 16, 8, 3], "relu"),
            # Larger network
            MockNeuralNetwork([32, 24, 16, 8, 1], "sigmoid"),
            # Wide network
            MockNeuralNetwork([15, 32, 32, 5], "tanh"),
        ]
        return networks
    
    def measure_performance_degradation(self, original_net: MockNeuralNetwork,
                                      quantized_params: np.ndarray,
                                      test_data: Tuple[np.ndarray, np.ndarray],
                                      tolerance: float = 0.1) -> Dict[str, Any]:
        """Measure performance degradation after quantization."""
        X, y = test_data
        
        # Original performance
        original_accuracy = original_net.evaluate_accuracy(X, y)
        original_loss = original_net.calculate_loss(X, y)
        
        # Quantized performance
        quantized_net = MockNeuralNetwork(original_net.layer_sizes, original_net.activation)
        quantized_net.set_parameters(quantized_params)
        
        quantized_accuracy = quantized_net.evaluate_accuracy(X, y)
        quantized_loss = quantized_net.calculate_loss(X, y)
        
        # Calculate degradation
        accuracy_degradation = original_accuracy - quantized_accuracy
        loss_increase = quantized_loss - original_loss
        
        return {
            'original_accuracy': original_accuracy,
            'quantized_accuracy': quantized_accuracy,
            'accuracy_degradation': accuracy_degradation,
            'original_loss': original_loss,
            'quantized_loss': quantized_loss,
            'loss_increase': loss_increase,
            'within_tolerance': accuracy_degradation <= tolerance,
            'relative_accuracy_loss': accuracy_degradation / max(original_accuracy, 1e-8)
        }


class TestRealNeuralNetworkModels:
    """Test with real neural network models."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.quantizer = HilbertQuantizer()
        self.batch_quantizer = BatchQuantizer()
    
    def create_test_dataset(self, num_samples: int = 100, 
                           input_dim: int = 10, 
                           output_dim: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Create synthetic test dataset."""
        X = np.random.randn(num_samples, input_dim).astype(np.float32)
        
        if output_dim == 1:
            # Binary classification or regression
            y = (np.sum(X, axis=1, keepdims=True) > 0).astype(np.float32)
        else:
            # Multi-class classification
            y = np.zeros((num_samples, output_dim), dtype=np.float32)
            classes = np.random.randint(0, output_dim, num_samples)
            y[np.arange(num_samples), classes] = 1
        
        return X, y
    
    def create_test_networks(self) -> List[MockNeuralNetwork]:
        """Create various test neural networks."""
        networks = [
            # Small network
            MockNeuralNetwork([10, 8, 1], "relu"),
            # Medium network
            MockNeuralNetwork([20, 16, 8, 3], "relu"),
            # Larger network
            MockNeuralNetwork([32, 24, 16, 8, 1], "sigmoid"),
            # Wide network
            MockNeuralNetwork([15, 32, 32, 5], "tanh"),
        ]
        return networks
    
    def measure_performance_degradation(self, original_net: MockNeuralNetwork,
                                      quantized_params: np.ndarray,
                                      test_data: Tuple[np.ndarray, np.ndarray],
                                      tolerance: float = 0.1) -> Dict[str, Any]:
        """Measure performance degradation after quantization."""
        X, y = test_data
        
        # Original performance
        original_accuracy = original_net.evaluate_accuracy(X, y)
        original_loss = original_net.calculate_loss(X, y)
        
        # Quantized performance
        quantized_net = MockNeuralNetwork(original_net.layer_sizes, original_net.activation)
        quantized_net.set_parameters(quantized_params)
        
        quantized_accuracy = quantized_net.evaluate_accuracy(X, y)
        quantized_loss = quantized_net.calculate_loss(X, y)
        
        # Calculate degradation
        accuracy_degradation = original_accuracy - quantized_accuracy
        loss_increase = quantized_loss - original_loss
        
        return {
            'original_accuracy': original_accuracy,
            'quantized_accuracy': quantized_accuracy,
            'accuracy_degradation': accuracy_degradation,
            'original_loss': original_loss,
            'quantized_loss': quantized_loss,
            'loss_increase': loss_increase,
            'within_tolerance': accuracy_degradation <= tolerance,
            'relative_accuracy_loss': accuracy_degradation / max(original_accuracy, 1e-8)
        }
    
    def test_small_network_quantization(self):
        """Test quantization of small neural network."""
        # Create small network
        network = MockNeuralNetwork([8, 6, 1], "relu")
        parameters = network.get_parameters()
        
        # Create test data
        X, y = self.create_test_dataset(50, 8, 1)
        
        # Quantize
        quantized_model = self.quantizer.quantize(
            parameters, 
            model_id="small_network_test",
            description="Small test network"
        )
        
        # Validate quantized model
        assert isinstance(quantized_model, QuantizedModel)
        assert quantized_model.parameter_count == len(parameters)
        assert len(quantized_model.compressed_data) > 0
        
        # Reconstruct and test
        reconstructed_params = self.quantizer.reconstruct(quantized_model)
        
        # Measure performance
        performance = self.measure_performance_degradation(
            network, reconstructed_params, (X, y), tolerance=0.2
        )
        
        # Validate performance preservation
        assert performance['accuracy_degradation'] <= 0.3  # Allow some degradation
        assert performance['relative_accuracy_loss'] <= 0.5  # Max 50% relative loss
        
        # Validate reconstruction quality
        mse = np.mean((parameters - reconstructed_params) ** 2)
        assert mse < 2.0  # Reasonable reconstruction error
    
    def test_medium_network_quantization(self):
        """Test quantization of medium-sized neural network."""
        # Create medium network
        network = MockNeuralNetwork([16, 12, 8, 3], "relu")
        parameters = network.get_parameters()
        
        # Create test data
        X, y = self.create_test_dataset(100, 16, 3)
        
        # Quantize with high quality
        quantized_model = self.quantizer.quantize(
            parameters,
            model_id="medium_network_test",
            description="Medium test network"
        )
        
        # Update configuration for better quality
        self.quantizer.update_configuration(compression_quality=0.9)
        
        quantized_model_hq = self.quantizer.quantize(
            parameters,
            model_id="medium_network_hq_test",
            description="Medium test network (high quality)"
        )
        
        # Test both quality levels
        for model, quality_name in [(quantized_model, "standard"), (quantized_model_hq, "high")]:
            reconstructed_params = self.quantizer.reconstruct(model)
            
            performance = self.measure_performance_degradation(
                network, reconstructed_params, (X, y), tolerance=0.15
            )
            
            # High quality should perform better
            if quality_name == "high":
                assert performance['accuracy_degradation'] <= 0.2
                assert performance['relative_accuracy_loss'] <= 0.3
            else:
                assert performance['accuracy_degradation'] <= 0.25
                assert performance['relative_accuracy_loss'] <= 0.4
    
    def test_large_network_quantization(self):
        """Test quantization of larger neural network."""
        # Create larger network
        network = MockNeuralNetwork([32, 24, 16, 8, 1], "sigmoid")
        parameters = network.get_parameters()
        
        # Create test data
        X, y = self.create_test_dataset(200, 32, 1)
        
        # Quantize
        quantized_model = self.quantizer.quantize(
            parameters,
            model_id="large_network_test",
            description="Large test network"
        )
        
        # Validate compression efficiency
        original_size = parameters.nbytes
        compressed_size = len(quantized_model.compressed_data)
        compression_ratio = original_size / compressed_size
        
        assert compression_ratio > 1.5  # Should achieve reasonable compression
        
        # Reconstruct and validate
        reconstructed_params = self.quantizer.reconstruct(quantized_model)
        
        performance = self.measure_performance_degradation(
            network, reconstructed_params, (X, y), tolerance=0.1
        )
        
        # Larger networks might be more sensitive to quantization
        assert performance['accuracy_degradation'] <= 0.3
        assert performance['relative_accuracy_loss'] <= 0.5
    
    def test_batch_network_quantization(self):
        """Test batch quantization of multiple networks."""
        # Create multiple networks
        networks = self.create_test_networks()
        parameter_sets = [net.get_parameters() for net in networks]
        model_ids = [f"batch_network_{i}" for i in range(len(networks))]
        
        # Batch quantize
        quantized_models = self.batch_quantizer.quantize_batch(
            parameter_sets=parameter_sets,
            model_ids=model_ids,
            descriptions=[f"Batch test network {i}" for i in range(len(networks))]
        )
        
        # Validate all models
        assert len(quantized_models) == len(networks)
        
        for i, (network, quantized_model) in enumerate(zip(networks, quantized_models)):
            # Validate model structure
            assert quantized_model.parameter_count == len(parameter_sets[i])
            assert quantized_model.metadata.model_name == model_ids[i]
            
            # Test reconstruction
            reconstructed = self.quantizer.reconstruct(quantized_model)
            assert len(reconstructed) == len(parameter_sets[i])
            
            # Test performance (with relaxed tolerance for batch processing)
            X, y = self.create_test_dataset(50, networks[i].layer_sizes[0], 
                                          networks[i].layer_sizes[-1])
            
            performance = self.measure_performance_degradation(
                network, reconstructed, (X, y), tolerance=0.25
            )
            
            assert performance['accuracy_degradation'] <= 0.4  # Relaxed for batch
    
    def test_network_similarity_search(self):
        """Test similarity search with neural networks."""
        # Create base network
        base_network = MockNeuralNetwork([16, 12, 8, 1], "relu")
        base_params = base_network.get_parameters()
        
        # Create similar networks (with small variations)
        similar_networks = []
        for i in range(5):
            similar_net = MockNeuralNetwork([16, 12, 8, 1], "relu")
            # Add small noise to make it similar but not identical
            similar_params = base_params + np.random.normal(0, 0.1, len(base_params))
            similar_net.set_parameters(similar_params)
            similar_networks.append(similar_net)
        
        # Create dissimilar networks
        dissimilar_networks = []
        for i in range(3):
            dissimilar_net = MockNeuralNetwork([16, 12, 8, 1], "relu")
            # Keep random initialization (very different)
            dissimilar_networks.append(dissimilar_net)
        
        # Quantize all networks
        all_networks = similar_networks + dissimilar_networks
        quantized_models = []
        
        for i, network in enumerate(all_networks):
            params = network.get_parameters()
            model = self.quantizer.quantize(params, f"search_test_{i}")
            quantized_models.append(model)
        
        # Search for similar networks
        search_results = self.quantizer.search(
            query_parameters=base_params,
            candidate_models=quantized_models,
            max_results=5
        )
        
        # Validate search results
        assert len(search_results) > 0
        assert len(search_results) <= 5
        
        # Results should be sorted by similarity
        similarities = [result.similarity_score for result in search_results]
        assert similarities == sorted(similarities, reverse=True)
        
        # Top results should be from similar networks (probabilistically)
        # This is not guaranteed due to quantization effects, but should be likely
        top_result_indices = []
        for result in search_results[:3]:  # Check top 3
            model_name = result.model.metadata.model_name
            model_idx = int(model_name.split('_')[-1])
            top_result_indices.append(model_idx)
        
        # At least some of the top results should be from similar networks (0-4)
        similar_in_top = sum(1 for idx in top_result_indices if idx < 5)
        assert similar_in_top >= 1  # At least one similar network in top results


class TestModelPerformancePreservation(EndToEndValidationTest):
    """Test model performance preservation after quantization."""
    
    def setup_method(self):
        """Set up test fixtures."""
        super().__init__()
    
    def test_classification_performance_preservation(self):
        """Test classification performance preservation."""
        # Create classification network
        network = MockNeuralNetwork([10, 8, 3], "relu")  # 3-class classification
        parameters = network.get_parameters()
        
        # Create classification dataset
        X, y = self.create_test_dataset(150, 10, 3)
        
        # Test different quality levels
        quality_levels = [0.5, 0.7, 0.9]
        results = {}
        
        for quality in quality_levels:
            self.quantizer.update_configuration(compression_quality=quality)
            
            quantized_model = self.quantizer.quantize(
                parameters,
                model_id=f"classification_test_q{quality}",
                description=f"Classification test at quality {quality}"
            )
            
            reconstructed_params = self.quantizer.reconstruct(quantized_model)
            
            performance = self.measure_performance_degradation(
                network, reconstructed_params, (X, y), tolerance=0.1
            )
            
            results[quality] = performance
        
        # Higher quality should generally preserve performance better
        assert results[0.9]['accuracy_degradation'] <= results[0.7]['accuracy_degradation'] + 0.1
        assert results[0.7]['accuracy_degradation'] <= results[0.5]['accuracy_degradation'] + 0.1
        
        # All should maintain reasonable performance
        for quality, perf in results.items():
            assert perf['accuracy_degradation'] <= 0.3
            assert perf['relative_accuracy_loss'] <= 0.5
    
    def test_regression_performance_preservation(self):
        """Test regression performance preservation."""
        # Create regression network
        network = MockNeuralNetwork([8, 6, 4, 1], "tanh")
        parameters = network.get_parameters()
        
        # Create regression dataset (continuous targets)
        X = np.random.randn(100, 8).astype(np.float32)
        y = (np.sum(X[:, :3], axis=1, keepdims=True) + 
             np.random.normal(0, 0.1, (100, 1))).astype(np.float32)
        
        # Quantize
        quantized_model = self.quantizer.quantize(
            parameters,
            model_id="regression_test",
            description="Regression test network"
        )
        
        reconstructed_params = self.quantizer.reconstruct(quantized_model)
        
        # Measure regression performance
        original_loss = network.calculate_loss(X, y)
        
        quantized_net = MockNeuralNetwork([8, 6, 4, 1], "tanh")
        quantized_net.set_parameters(reconstructed_params)
        quantized_loss = quantized_net.calculate_loss(X, y)
        
        loss_increase = quantized_loss - original_loss
        relative_loss_increase = loss_increase / max(original_loss, 1e-8)
        
        # Validate regression performance preservation
        assert relative_loss_increase <= 1.0  # Loss shouldn't more than double
        assert loss_increase <= 2.0  # Absolute loss increase should be reasonable
    
    def test_performance_across_architectures(self):
        """Test performance preservation across different architectures."""
        architectures = [
            ([5, 3, 1], "relu", "small_deep"),
            ([10, 1], "sigmoid", "single_layer"),
            ([8, 16, 8, 1], "tanh", "bottleneck"),
            ([12, 12, 12, 1], "relu", "uniform_width")
        ]
        
        results = {}
        
        for arch, activation, name in architectures:
            # Create network
            network = MockNeuralNetwork(arch, activation)
            parameters = network.get_parameters()
            
            # Create appropriate test data
            X, y = self.create_test_dataset(80, arch[0], arch[-1])
            
            # Quantize
            quantized_model = self.quantizer.quantize(
                parameters,
                model_id=f"arch_test_{name}",
                description=f"Architecture test: {name}"
            )
            
            reconstructed_params = self.quantizer.reconstruct(quantized_model)
            
            # Measure performance
            performance = self.measure_performance_degradation(
                network, reconstructed_params, (X, y), tolerance=0.2
            )
            
            results[name] = performance
        
        # All architectures should maintain reasonable performance
        for arch_name, perf in results.items():
            assert perf['accuracy_degradation'] <= 0.4, f"Architecture {arch_name} failed"
            assert perf['relative_accuracy_loss'] <= 0.6, f"Architecture {arch_name} failed"
    
    def test_performance_with_different_data_distributions(self):
        """Test performance with different input data distributions."""
        # Create standard network
        network = MockNeuralNetwork([6, 8, 1], "relu")
        parameters = network.get_parameters()
        
        # Quantize once
        quantized_model = self.quantizer.quantize(
            parameters,
            model_id="distribution_test",
            description="Data distribution test"
        )
        
        reconstructed_params = self.quantizer.reconstruct(quantized_model)
        
        # Test with different data distributions
        distributions = {
            'normal': lambda: np.random.randn(60, 6).astype(np.float32),
            'uniform': lambda: np.random.uniform(-2, 2, (60, 6)).astype(np.float32),
            'sparse': lambda: np.random.choice([0, 1, -1], (60, 6)).astype(np.float32),
            'scaled': lambda: np.random.randn(60, 6).astype(np.float32) * 5
        }
        
        results = {}
        
        for dist_name, data_gen in distributions.items():
            X = data_gen()
            y = (np.sum(X, axis=1, keepdims=True) > 0).astype(np.float32)
            
            performance = self.measure_performance_degradation(
                network, reconstructed_params, (X, y), tolerance=0.2
            )
            
            results[dist_name] = performance
        
        # Performance should be reasonable across all distributions
        for dist_name, perf in results.items():
            assert perf['accuracy_degradation'] <= 0.5, f"Distribution {dist_name} failed"


class TestComprehensiveErrorHandling(EndToEndValidationTest):
    """Test comprehensive error handling and edge cases."""
    
    def setup_method(self):
        """Set up test fixtures."""
        super().__init__()
    
    def test_invalid_input_handling(self):
        """Test handling of invalid inputs."""
        # Test empty parameters
        with pytest.raises((QuantizationError, ValidationError, ValueError)):
            self.quantizer.quantize(np.array([]), "empty_test")
        
        # Test NaN parameters
        with pytest.raises((QuantizationError, ValidationError)):
            params_with_nan = np.array([1.0, 2.0, np.nan, 4.0])
            self.quantizer.quantize(params_with_nan, "nan_test")
        
        # Test infinite parameters
        with pytest.raises((QuantizationError, ValidationError)):
            params_with_inf = np.array([1.0, 2.0, np.inf, 4.0])
            self.quantizer.quantize(params_with_inf, "inf_test")
        
        # Test wrong dimensionality
        with pytest.raises((QuantizationError, ValidationError)):
            params_2d = np.random.randn(8, 8)  # Should be 1D
            self.quantizer.quantize(params_2d, "2d_test")
    
    def test_corrupted_model_handling(self):
        """Test handling of corrupted quantized models."""
        # Create valid model first
        parameters = np.random.randn(64).astype(np.float32)
        valid_model = self.quantizer.quantize(parameters, "valid_test")
        
        # Test corrupted compressed data
        from hilbert_quantization.models import QuantizedModel, ModelMetadata
        
        corrupted_model = QuantizedModel(
            compressed_data=b"corrupted_data",
            original_dimensions=valid_model.original_dimensions,
            parameter_count=valid_model.parameter_count,
            compression_quality=valid_model.compression_quality,
            hierarchical_indices=valid_model.hierarchical_indices,
            metadata=valid_model.metadata
        )
        
        with pytest.raises((ReconstructionError, HilbertQuantizationError)):
            self.quantizer.reconstruct(corrupted_model)
        
        # Test corrupted indices
        corrupted_indices_model = QuantizedModel(
            compressed_data=valid_model.compressed_data,
            original_dimensions=valid_model.original_dimensions,
            parameter_count=valid_model.parameter_count,
            compression_quality=valid_model.compression_quality,
            hierarchical_indices=np.array([np.nan, np.inf, 1.0]),  # Corrupted indices
            metadata=valid_model.metadata
        )
        
        # Search should handle corrupted indices gracefully
        try:
            results = self.quantizer.search(
                parameters, [corrupted_indices_model], max_results=1
            )
            # Should either return empty results or handle gracefully
            assert isinstance(results, list)
        except (SearchError, ValidationError):
            # Acceptable to raise error for corrupted data
            pass
    
    def test_configuration_error_handling(self):
        """Test configuration error handling."""
        # Test invalid compression quality
        with pytest.raises((ConfigurationError, ValueError)):
            self.quantizer.update_configuration(compression_quality=1.5)
        
        with pytest.raises((ConfigurationError, ValueError)):
            self.quantizer.update_configuration(compression_quality=-0.1)
        
        # Test invalid search parameters
        with pytest.raises((ConfigurationError, ValueError)):
            self.quantizer.update_configuration(search_max_results=-1)
        
        with pytest.raises((ConfigurationError, ValueError)):
            self.quantizer.update_configuration(search_similarity_threshold=1.5)
    
    def test_memory_limit_handling(self):
        """Test handling of memory-intensive operations."""
        # Test with very large parameter arrays (if memory allows)
        try:
            # Create moderately large array
            large_params = np.random.randn(16384).astype(np.float32)  # 64KB
            
            # Should handle gracefully
            quantized_model = self.quantizer.quantize(large_params, "large_test")
            reconstructed = self.quantizer.reconstruct(quantized_model)
            
            assert len(reconstructed) == len(large_params)
            
        except MemoryError:
            # Acceptable if system doesn't have enough memory
            pytest.skip("Insufficient memory for large parameter test")
        except (QuantizationError, HilbertQuantizationError):
            # Acceptable if system has limits on parameter size
            pass
    
    def test_concurrent_access_handling(self):
        """Test handling of concurrent access scenarios."""
        import threading
        import queue
        
        def quantize_worker(worker_id: int, result_queue: queue.Queue):
            """Worker function for concurrent quantization."""
            try:
                # Create separate quantizer instance
                quantizer = HilbertQuantizer()
                
                # Perform quantization
                params = np.random.randn(256).astype(np.float32)
                model = quantizer.quantize(params, f"concurrent_test_{worker_id}")
                
                # Perform reconstruction
                reconstructed = quantizer.reconstruct(model)
                
                result_queue.put({
                    'worker_id': worker_id,
                    'success': True,
                    'param_count': len(reconstructed),
                    'error': None
                })
                
            except Exception as e:
                result_queue.put({
                    'worker_id': worker_id,
                    'success': False,
                    'param_count': 0,
                    'error': str(e)
                })
        
        # Run concurrent operations
        num_workers = 3
        result_queue = queue.Queue()
        threads = []
        
        for i in range(num_workers):
            thread = threading.Thread(target=quantize_worker, args=(i, result_queue))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Collect results
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())
        
        # Validate results
        assert len(results) == num_workers
        
        # At least some operations should succeed
        successful_ops = sum(1 for r in results if r['success'])
        assert successful_ops >= 1  # At least one should succeed
        
        # All successful operations should have correct parameter counts
        for result in results:
            if result['success']:
                assert result['param_count'] == 256
    
    def test_file_io_error_handling(self):
        """Test file I/O error handling."""
        # Create valid model
        parameters = np.random.randn(64).astype(np.float32)
        model = self.quantizer.quantize(parameters, "file_io_test")
        
        # Test saving to invalid path
        with pytest.raises((QuantizationError, OSError, PermissionError)):
            self.quantizer.save_model(model, "/invalid/path/model.pkl")
        
        # Test loading non-existent file
        with pytest.raises((QuantizationError, FileNotFoundError)):
            self.quantizer.load_model("non_existent_file.pkl")
        
        # Test saving and loading with valid path
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "test_model.pkl")
            
            # Save model
            self.quantizer.save_model(model, file_path)
            assert os.path.exists(file_path)
            
            # Load model
            loaded_model = self.quantizer.load_model(file_path)
            
            # Validate loaded model
            assert loaded_model.parameter_count == model.parameter_count
            assert loaded_model.metadata.model_name == model.metadata.model_name
            
            # Test reconstruction of loaded model
            reconstructed = self.quantizer.reconstruct(loaded_model)
            assert len(reconstructed) == 64


class TestEdgeCases(EndToEndValidationTest):
    """Test edge cases and boundary conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        super().__init__()
    
    def test_minimal_parameter_counts(self):
        """Test with minimal parameter counts."""
        # Test with very small networks
        small_counts = [4, 16, 64]  # Powers of 4
        
        for count in small_counts:
            parameters = np.random.randn(count).astype(np.float32)
            
            # Should handle small networks
            quantized_model = self.quantizer.quantize(parameters, f"minimal_{count}")
            reconstructed = self.quantizer.reconstruct(quantized_model)
            
            assert len(reconstructed) == count
            
            # Reconstruction should be reasonable even for small networks
            mse = np.mean((parameters - reconstructed) ** 2)
            assert mse < 5.0  # Relaxed threshold for small networks
    
    def test_boundary_quality_levels(self):
        """Test with boundary quality levels."""
        parameters = np.random.randn(256).astype(np.float32)
        
        # Test minimum quality
        self.quantizer.update_configuration(compression_quality=0.1)
        model_min = self.quantizer.quantize(parameters, "boundary_min")
        reconstructed_min = self.quantizer.reconstruct(model_min)
        
        # Test maximum quality
        self.quantizer.update_configuration(compression_quality=0.99)
        model_max = self.quantizer.quantize(parameters, "boundary_max")
        reconstructed_max = self.quantizer.reconstruct(model_max)
        
        # Both should work
        assert len(reconstructed_min) == len(parameters)
        assert len(reconstructed_max) == len(parameters)
        
        # Max quality should generally be better
        mse_min = np.mean((parameters - reconstructed_min) ** 2)
        mse_max = np.mean((parameters - reconstructed_max) ** 2)
        
        # Allow some variance, but max quality should generally be better
        assert mse_max <= mse_min * 2
    
    def test_extreme_parameter_values(self):
        """Test with extreme parameter values."""
        # Test with very large values
        large_params = np.random.randn(256).astype(np.float32) * 1000
        model_large = self.quantizer.quantize(large_params, "extreme_large")
        reconstructed_large = self.quantizer.reconstruct(model_large)
        
        assert len(reconstructed_large) == len(large_params)
        
        # Test with very small values
        small_params = np.random.randn(256).astype(np.float32) * 0.001
        model_small = self.quantizer.quantize(small_params, "extreme_small")
        reconstructed_small = self.quantizer.reconstruct(model_small)
        
        assert len(reconstructed_small) == len(small_params)
        
        # Test with mixed extreme values
        mixed_params = np.concatenate([
            np.array([1000.0, -1000.0, 0.001, -0.001] * 64, dtype=np.float32)
        ])
        model_mixed = self.quantizer.quantize(mixed_params, "extreme_mixed")
        reconstructed_mixed = self.quantizer.reconstruct(model_mixed)
        
        assert len(reconstructed_mixed) == len(mixed_params)


if __name__ == "__main__":
    # Run validation tests directly
    print("Running End-to-End Validation Tests...")
    
    # Test with real neural networks
    print("\n=== Neural Network Model Tests ===")
    network_test = TestRealNeuralNetworkModels()
    
    try:
        network_test.test_small_network_quantization()
        print("✓ Small network quantization test passed")
    except Exception as e:
        print(f"✗ Small network test failed: {e}")
    
    try:
        network_test.test_medium_network_quantization()
        print("✓ Medium network quantization test passed")
    except Exception as e:
        print(f"✗ Medium network test failed: {e}")
    
    # Test performance preservation
    print("\n=== Performance Preservation Tests ===")
    perf_test = TestModelPerformancePreservation()
    
    try:
        perf_test.test_classification_performance_preservation()
        print("✓ Classification performance preservation test passed")
    except Exception as e:
        print(f"✗ Classification performance test failed: {e}")
    
    # Test error handling
    print("\n=== Error Handling Tests ===")
    error_test = TestComprehensiveErrorHandling()
    
    try:
        error_test.test_invalid_input_handling()
        print("✓ Invalid input handling test passed")
    except Exception as e:
        print(f"✗ Invalid input handling test failed: {e}")
    
    print("\nEnd-to-end validation tests completed!")