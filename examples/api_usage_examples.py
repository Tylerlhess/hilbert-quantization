"""
Comprehensive API usage examples for the Hilbert quantization system.

This module demonstrates various ways to use the high-level API for quantization,
search, and reconstruction operations.
"""

import numpy as np
from pathlib import Path

from hilbert_quantization.api import (
    HilbertQuantizer, BatchQuantizer,
    quantize_model, reconstruct_model, search_similar_models
)
from hilbert_quantization.config import (
    SystemConfig, create_default_config, create_high_performance_config, create_high_quality_config
)


def basic_usage_example():
    """
    Basic usage example showing quantization, search, and reconstruction.
    """
    print("=== Basic Usage Example ===")
    
    # Create quantizer with default configuration
    quantizer = HilbertQuantizer()
    
    # Generate sample model parameters
    model1_params = np.random.randn(1000).astype(np.float32)
    model2_params = np.random.randn(1000).astype(np.float32)
    query_params = model1_params + 0.1 * np.random.randn(1000).astype(np.float32)  # Similar to model1
    
    print(f"Model 1 parameters: {len(model1_params)} values")
    print(f"Model 2 parameters: {len(model2_params)} values")
    print(f"Query parameters: {len(query_params)} values")
    
    # Quantize models
    print("\n--- Quantization ---")
    quantized_model1 = quantizer.quantize(model1_params, model_id="model_1", description="First test model")
    quantized_model2 = quantizer.quantize(model2_params, model_id="model_2", description="Second test model")
    
    print(f"Quantized model 1: {quantized_model1.model_id}")
    print(f"Compression ratio: {quantized_model1.metadata.compression_metrics.compression_ratio:.3f}")
    print(f"Original size: {quantized_model1.metadata.compression_metrics.original_size} bytes")
    print(f"Compressed size: {quantized_model1.metadata.compression_metrics.compressed_size} bytes")
    
    # Search for similar models
    print("\n--- Similarity Search ---")
    search_results = quantizer.search(query_params, max_results=5, similarity_threshold=0.1)
    
    print(f"Found {len(search_results)} similar models:")
    for i, result in enumerate(search_results):
        print(f"  {i+1}. Model: {result.model.model_id}")
        print(f"     Similarity: {result.similarity_score:.3f}")
        print(f"     Reconstruction error: {result.reconstruction_error:.6f}")
    
    # Reconstruct parameters from best match
    print("\n--- Reconstruction ---")
    if search_results:
        best_match = search_results[0].model
        reconstructed_params = quantizer.reconstruct(best_match)
        
        # Calculate reconstruction accuracy
        original_params = model1_params if best_match.model_id == "model_1" else model2_params
        reconstruction_error = np.mean(np.abs(original_params - reconstructed_params))
        
        print(f"Reconstructed {len(reconstructed_params)} parameters")
        print(f"Reconstruction error: {reconstruction_error:.6f}")
        print(f"Max absolute error: {np.max(np.abs(original_params - reconstructed_params)):.6f}")
    
    # Registry information
    print("\n--- Registry Information ---")
    registry_info = quantizer.get_registry_info()
    print(f"Total models in registry: {registry_info['total_models']}")
    print(f"Model IDs: {registry_info['model_ids']}")
    print(f"Average compression ratio: {np.mean(registry_info['compression_ratios']):.3f}")


def configuration_examples():
    """
    Examples of different configuration options.
    """
    print("\n=== Configuration Examples ===")
    
    # Default configuration
    print("\n--- Default Configuration ---")
    default_quantizer = HilbertQuantizer(create_default_config())
    print(f"Compression quality: {default_quantizer.config.compression.quality}")
    print(f"Search max results: {default_quantizer.config.search.max_results}")
    print(f"Index granularity levels: {default_quantizer.config.quantization.index_granularity_levels}")
    
    # High performance configuration
    print("\n--- High Performance Configuration ---")
    perf_quantizer = HilbertQuantizer(create_high_performance_config())
    print(f"Parallel processing: {perf_quantizer.config.compression.enable_parallel_processing}")
    print(f"Parallel search: {perf_quantizer.config.search.enable_parallel_search}")
    print(f"GPU acceleration: {perf_quantizer.config.enable_gpu_acceleration}")
    print(f"Caching enabled: {perf_quantizer.config.search.enable_caching}")
    
    # High quality configuration
    print("\n--- High Quality Configuration ---")
    quality_quantizer = HilbertQuantizer(create_high_quality_config())
    print(f"Compression quality: {quality_quantizer.config.compression.quality}")
    print(f"Strict validation: {quality_quantizer.config.quantization.strict_validation}")
    print(f"Max reconstruction error: {quality_quantizer.config.compression.max_reconstruction_error}")
    print(f"Similarity threshold: {quality_quantizer.config.search.similarity_threshold}")
    
    # Custom configuration
    print("\n--- Custom Configuration ---")
    custom_config = SystemConfig()
    custom_config.compression.quality = 0.85
    custom_config.search.max_results = 20
    custom_config.quantization.index_granularity_levels = [64, 32, 16, 8]
    
    custom_quantizer = HilbertQuantizer(custom_config)
    print(f"Custom compression quality: {custom_quantizer.config.compression.quality}")
    print(f"Custom max results: {custom_quantizer.config.search.max_results}")
    print(f"Custom granularity levels: {custom_quantizer.config.quantization.index_granularity_levels}")
    
    # Dynamic configuration updates
    print("\n--- Dynamic Configuration Updates ---")
    quantizer = HilbertQuantizer()
    print(f"Original quality: {quantizer.config.compression.quality}")
    
    quantizer.update_configuration(compression_quality=0.95, search_max_results=15)
    print(f"Updated quality: {quantizer.config.compression.quality}")
    print(f"Updated max results: {quantizer.config.search.max_results}")


def batch_processing_example():
    """
    Example of batch processing multiple models.
    """
    print("\n=== Batch Processing Example ===")
    
    # Create batch quantizer
    batch_quantizer = BatchQuantizer()
    
    # Generate multiple parameter sets
    parameter_sets = [
        np.random.randn(500).astype(np.float32),
        np.random.randn(750).astype(np.float32),
        np.random.randn(1000).astype(np.float32),
        np.random.randn(1250).astype(np.float32)
    ]
    
    model_ids = [f"batch_model_{i}" for i in range(len(parameter_sets))]
    descriptions = [f"Batch processed model {i}" for i in range(len(parameter_sets))]
    
    print(f"Processing {len(parameter_sets)} models in batch")
    print(f"Parameter counts: {[len(params) for params in parameter_sets]}")
    
    # Batch quantization
    print("\n--- Batch Quantization ---")
    quantized_models = batch_quantizer.quantize_batch(
        parameter_sets, 
        model_ids=model_ids, 
        descriptions=descriptions
    )
    
    print(f"Successfully quantized {len(quantized_models)} models")
    for model in quantized_models:
        print(f"  {model.model_id}: {model.parameter_count} params, "
              f"ratio: {model.metadata.compression_metrics.compression_ratio:.3f}")
    
    # Batch search
    print("\n--- Batch Search ---")
    query_sets = [
        params + 0.05 * np.random.randn(len(params)).astype(np.float32) 
        for params in parameter_sets[:2]  # Search with first 2 as queries
    ]
    
    search_results = batch_quantizer.search_batch(query_sets, quantized_models, max_results=3)
    
    print(f"Performed {len(search_results)} batch searches")
    for i, results in enumerate(search_results):
        print(f"  Query {i+1}: Found {len(results)} matches")
        if results:
            best_match = results[0]
            print(f"    Best match: {best_match.model.model_id} "
                  f"(similarity: {best_match.similarity_score:.3f})")


def model_persistence_example():
    """
    Example of saving and loading models.
    """
    print("\n=== Model Persistence Example ===")
    
    quantizer = HilbertQuantizer()
    
    # Create and quantize a model
    parameters = np.random.randn(800).astype(np.float32)
    quantized_model = quantizer.quantize(parameters, model_id="persistent_model", 
                                       description="Model for persistence testing")
    
    print(f"Created model: {quantized_model.model_id}")
    print(f"Parameter count: {quantized_model.parameter_count}")
    
    # Save model to file
    save_path = Path("temp_model.pkl")
    quantizer.save_model(quantized_model, save_path)
    print(f"Saved model to: {save_path}")
    
    # Create new quantizer and load model
    new_quantizer = HilbertQuantizer()
    loaded_model = new_quantizer.load_model(save_path)
    
    print(f"Loaded model: {loaded_model.model_id}")
    print(f"Parameter count: {loaded_model.parameter_count}")
    
    # Verify models are equivalent
    original_reconstructed = quantizer.reconstruct(quantized_model)
    loaded_reconstructed = new_quantizer.reconstruct(loaded_model)
    
    reconstruction_diff = np.mean(np.abs(original_reconstructed - loaded_reconstructed))
    print(f"Reconstruction difference: {reconstruction_diff:.10f}")
    
    # Add to new quantizer's registry
    new_quantizer.add_model_to_registry(loaded_model)
    registry_info = new_quantizer.get_registry_info()
    print(f"New quantizer registry: {registry_info['total_models']} models")
    
    # Clean up
    save_path.unlink(missing_ok=True)
    print("Cleaned up temporary file")


def performance_benchmarking_example():
    """
    Example of performance benchmarking.
    """
    print("\n=== Performance Benchmarking Example ===")
    
    quantizer = HilbertQuantizer()
    
    # Benchmark different model sizes
    parameter_counts = [100, 500, 1000, 2000]
    print(f"Benchmarking parameter counts: {parameter_counts}")
    
    results = quantizer.benchmark_performance(parameter_counts, num_trials=3)
    
    print("\n--- Benchmark Results ---")
    print("Param Count | Quant Time | Recon Time | Search Time | Comp Ratio | Recon Error")
    print("-" * 80)
    
    for i, count in enumerate(results["parameter_counts"]):
        print(f"{count:10d} | "
              f"{results['quantization_times'][i]:9.4f}s | "
              f"{results['reconstruction_times'][i]:9.4f}s | "
              f"{results['search_times'][i]:10.4f}s | "
              f"{results['compression_ratios'][i]:9.3f} | "
              f"{results['reconstruction_errors'][i]:10.6f}")


def optimal_configuration_example():
    """
    Example of getting optimal configurations for different model sizes.
    """
    print("\n=== Optimal Configuration Example ===")
    
    quantizer = HilbertQuantizer()
    
    model_sizes = [1000, 10000, 100000, 1000000, 10000000]
    
    print("Model Size | Compression Quality | Adaptive | Index Levels | Cache Size")
    print("-" * 70)
    
    for size in model_sizes:
        optimal_config = quantizer.get_optimal_configuration(size)
        
        print(f"{size:9d} | "
              f"{optimal_config.compression.quality:17.2f} | "
              f"{str(optimal_config.compression.adaptive_quality):8s} | "
              f"{len(optimal_config.quantization.index_granularity_levels):11d} | "
              f"{optimal_config.search.cache_size_limit:9d}")


def convenience_functions_example():
    """
    Example using convenience functions for simple operations.
    """
    print("\n=== Convenience Functions Example ===")
    
    # Generate test data
    parameters = np.random.randn(600).astype(np.float32)
    print(f"Original parameters: {len(parameters)} values")
    
    # Simple quantization
    print("\n--- Simple Quantization ---")
    quantized = quantize_model(parameters)
    print(f"Quantized model ID: {quantized.model_id}")
    print(f"Compression ratio: {quantized.metadata.compression_metrics.compression_ratio:.3f}")
    
    # Simple reconstruction
    print("\n--- Simple Reconstruction ---")
    reconstructed = reconstruct_model(quantized)
    reconstruction_error = np.mean(np.abs(parameters - reconstructed))
    print(f"Reconstructed {len(reconstructed)} parameters")
    print(f"Reconstruction error: {reconstruction_error:.6f}")
    
    # Simple search
    print("\n--- Simple Search ---")
    # Create some candidate models
    candidates = [quantized]
    for i in range(3):
        other_params = np.random.randn(600).astype(np.float32)
        other_quantized = quantize_model(other_params)
        candidates.append(other_quantized)
    
    # Search with query similar to original
    query = parameters + 0.1 * np.random.randn(len(parameters)).astype(np.float32)
    search_results = search_similar_models(query, candidates, max_results=3)
    
    print(f"Found {len(search_results)} similar models:")
    for i, result in enumerate(search_results):
        print(f"  {i+1}. Similarity: {result.similarity_score:.3f}")


def error_handling_example():
    """
    Example demonstrating error handling and validation.
    """
    print("\n=== Error Handling Example ===")
    
    quantizer = HilbertQuantizer()
    
    # Test various error conditions
    print("\n--- Validation Errors ---")
    
    try:
        # Empty parameters
        quantizer.quantize(np.array([]))
    except Exception as e:
        print(f"Empty parameters error: {type(e).__name__}: {e}")
    
    try:
        # Parameters with NaN
        bad_params = np.array([1.0, 2.0, np.nan, 4.0])
        quantizer.quantize(bad_params)
    except Exception as e:
        print(f"NaN parameters error: {type(e).__name__}: {e}")
    
    try:
        # Multi-dimensional parameters
        bad_params = np.random.randn(10, 10)
        quantizer.quantize(bad_params)
    except Exception as e:
        print(f"Multi-dimensional error: {type(e).__name__}: {e}")
    
    try:
        # Search with no candidates
        query = np.random.randn(100).astype(np.float32)
        quantizer.search(query)
    except Exception as e:
        print(f"No candidates error: {type(e).__name__}: {e}")
    
    try:
        # Invalid configuration update
        quantizer.update_configuration(invalid_parameter=123)
    except Exception as e:
        print(f"Invalid config error: {type(e).__name__}: {e}")
    
    print("\nError handling completed successfully!")


def main():
    """
    Run all examples.
    """
    print("Hilbert Quantization API Usage Examples")
    print("=" * 50)
    
    try:
        basic_usage_example()
        configuration_examples()
        batch_processing_example()
        model_persistence_example()
        performance_benchmarking_example()
        optimal_configuration_example()
        convenience_functions_example()
        error_handling_example()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"\nExample failed with error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()