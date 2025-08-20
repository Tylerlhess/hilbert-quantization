# Hilbert Quantization API Guide

This guide provides comprehensive documentation for the Hilbert Quantization system's high-level API.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Classes](#core-classes)
3. [Configuration](#configuration)
4. [Basic Operations](#basic-operations)
5. [Advanced Features](#advanced-features)
6. [Error Handling](#error-handling)
7. [Performance Optimization](#performance-optimization)
8. [Examples](#examples)

## Quick Start

### Installation and Basic Usage

```python
import numpy as np
from hilbert_quantization.api import HilbertQuantizer

# Create quantizer with default settings
quantizer = HilbertQuantizer()

# Quantize model parameters
parameters = np.random.randn(1000).astype(np.float32)
quantized_model = quantizer.quantize(parameters, model_id="my_model")

# Search for similar models
query_params = np.random.randn(1000).astype(np.float32)
results = quantizer.search(query_params, max_results=5)

# Reconstruct parameters
reconstructed = quantizer.reconstruct(quantized_model)
```

### Convenience Functions

For simple operations, use the convenience functions:

```python
from hilbert_quantization.api import quantize_model, reconstruct_model, search_similar_models

# Simple quantization
quantized = quantize_model(parameters)

# Simple reconstruction
reconstructed = reconstruct_model(quantized)

# Simple search
results = search_similar_models(query_params, [quantized])
```

## Core Classes

### HilbertQuantizer

The main API class providing quantization, search, and reconstruction operations.

```python
class HilbertQuantizer:
    def __init__(self, config: Optional[SystemConfig] = None)
    def quantize(self, parameters, model_id=None, description=None, validate=True) -> QuantizedModel
    def reconstruct(self, quantized_model, validate=True) -> np.ndarray
    def search(self, query_parameters, candidate_models=None, max_results=None, similarity_threshold=None) -> List[SearchResult]
```

**Key Features:**
- Automatic model registry management
- Comprehensive error handling and validation
- Configurable compression and search parameters
- Built-in performance monitoring

### BatchQuantizer

Optimized for processing multiple models efficiently.

```python
class BatchQuantizer:
    def quantize_batch(self, parameter_sets, model_ids=None, descriptions=None, parallel=True) -> List[QuantizedModel]
    def search_batch(self, query_sets, candidate_models, max_results=10) -> List[List[SearchResult]]
```

## Configuration

### Configuration Types

The system supports several pre-configured setups:

```python
from hilbert_quantization.config import (
    create_default_config,
    create_high_performance_config,
    create_high_quality_config
)

# Default balanced configuration
quantizer = HilbertQuantizer(create_default_config())

# Optimized for speed
quantizer = HilbertQuantizer(create_high_performance_config())

# Optimized for quality
quantizer = HilbertQuantizer(create_high_quality_config())
```

### Custom Configuration

```python
from hilbert_quantization.config import SystemConfig

config = SystemConfig()
config.compression.quality = 0.85
config.search.max_results = 20
config.quantization.index_granularity_levels = [64, 32, 16, 8]

quantizer = HilbertQuantizer(config)
```

### Dynamic Configuration Updates

```python
# Update configuration at runtime
quantizer.update_configuration(
    compression_quality=0.9,
    search_max_results=15,
    quantization_padding_value=0.1
)
```

### Configuration Parameters

#### Quantization Configuration
- `auto_select_dimensions`: Automatically select optimal dimensions
- `target_dimensions`: Manual dimension specification
- `padding_value`: Value for padding unused space
- `min_efficiency_ratio`: Minimum space utilization efficiency
- `index_granularity_levels`: Hierarchical index granularity levels
- `validate_spatial_locality`: Enable spatial locality validation

#### Compression Configuration
- `quality`: Compression quality (0.0 to 1.0)
- `preserve_index_row`: Preserve hierarchical indices during compression
- `adaptive_quality`: Enable adaptive quality adjustment
- `quality_range`: Valid quality range for adaptive mode
- `enable_parallel_processing`: Enable parallel compression
- `max_reconstruction_error`: Maximum acceptable reconstruction error

#### Search Configuration
- `max_results`: Maximum number of search results
- `similarity_threshold`: Minimum similarity threshold
- `enable_progressive_filtering`: Enable progressive filtering strategy
- `filtering_strategy`: Filtering approach ("progressive", "coarse_to_fine", "fine_to_coarse")
- `enable_parallel_search`: Enable parallel search operations
- `similarity_weights`: Weights for different similarity metrics
- `enable_caching`: Enable search result caching

## Basic Operations

### Quantization

```python
# Basic quantization
quantized = quantizer.quantize(parameters)

# With metadata
quantized = quantizer.quantize(
    parameters,
    model_id="transformer_layer_1",
    description="First transformer layer weights"
)

# With validation disabled for performance
quantized = quantizer.quantize(parameters, validate=False)
```

### Reconstruction

```python
# Basic reconstruction
reconstructed = quantizer.reconstruct(quantized_model)

# With validation disabled
reconstructed = quantizer.reconstruct(quantized_model, validate=False)

# Check reconstruction quality
original_params = get_original_parameters()
error = np.mean(np.abs(original_params - reconstructed))
print(f"Reconstruction error: {error:.6f}")
```

### Similarity Search

```python
# Search in model registry
results = quantizer.search(query_parameters)

# Search specific candidates
results = quantizer.search(query_parameters, candidate_models=[model1, model2])

# Customized search parameters
results = quantizer.search(
    query_parameters,
    max_results=10,
    similarity_threshold=0.2
)

# Process search results
for result in results:
    print(f"Model: {result.model.model_id}")
    print(f"Similarity: {result.similarity_score:.3f}")
    print(f"Reconstruction error: {result.reconstruction_error:.6f}")
```

## Advanced Features

### Model Registry Management

```python
# Add models to registry
quantizer.add_model_to_registry(quantized_model)

# Remove models from registry
removed = quantizer.remove_model_from_registry("model_id")

# Clear entire registry
quantizer.clear_registry()

# Get registry information
info = quantizer.get_registry_info()
print(f"Total models: {info['total_models']}")
print(f"Model IDs: {info['model_ids']}")
print(f"Average compression ratio: {np.mean(info['compression_ratios']):.3f}")
```

### Model Persistence

```python
# Save model to file
quantizer.save_model(quantized_model, "model.pkl")

# Load model from file
loaded_model = quantizer.load_model("model.pkl")

# Save/load with custom paths
from pathlib import Path
model_dir = Path("models")
model_dir.mkdir(exist_ok=True)
quantizer.save_model(quantized_model, model_dir / f"{model.model_id}.pkl")
```

### Compression Metrics

```python
# Get detailed compression metrics
metrics = quantizer.get_compression_metrics(quantized_model)
print(f"Compression ratio: {metrics.compression_ratio:.3f}")
print(f"Original size: {metrics.original_size} bytes")
print(f"Compressed size: {metrics.compressed_size} bytes")
print(f"Reconstruction error: {metrics.reconstruction_error:.6f}")
```

### Optimal Configuration Selection

```python
# Get optimal configuration for model size
param_count = 1000000
optimal_config = quantizer.get_optimal_configuration(param_count)

# Apply optimal configuration
new_quantizer = HilbertQuantizer(optimal_config)
```

### Performance Benchmarking

```python
# Benchmark different model sizes
parameter_counts = [1000, 5000, 10000, 50000]
results = quantizer.benchmark_performance(parameter_counts, num_trials=5)

# Analyze results
for i, count in enumerate(results["parameter_counts"]):
    print(f"Size: {count}, Quantization: {results['quantization_times'][i]:.4f}s")
    print(f"Compression ratio: {results['compression_ratios'][i]:.3f}")
```

## Error Handling

### Exception Types

The API uses specific exception types for different error conditions:

```python
from hilbert_quantization.exceptions import (
    QuantizationError,
    ReconstructionError,
    SearchError,
    ValidationError,
    ConfigurationError
)
```

### Common Error Scenarios

```python
try:
    # Quantization with invalid parameters
    quantizer.quantize(np.array([]))  # Empty array
except ValidationError as e:
    print(f"Validation error: {e}")

try:
    # Search with no candidates
    quantizer.search(query_parameters)  # Empty registry
except SearchError as e:
    print(f"Search error: {e}")

try:
    # Invalid configuration
    quantizer.update_configuration(invalid_param=123)
except ConfigurationError as e:
    print(f"Configuration error: {e}")
```

### Input Validation

The API automatically validates inputs and provides informative error messages:

```python
# These will raise ValidationError with descriptive messages:
quantizer.quantize(np.array([np.nan, 1.0, 2.0]))  # Non-finite values
quantizer.quantize(np.random.randn(10, 10))        # Multi-dimensional array
quantizer.reconstruct("not_a_model")               # Invalid model type
```

## Performance Optimization

### Configuration for Performance

```python
# High-performance configuration
config = create_high_performance_config()
quantizer = HilbertQuantizer(config)

# Key performance settings:
# - enable_parallel_processing = True
# - enable_parallel_search = True
# - enable_caching = True
# - strict_validation = False
```

### Batch Processing

```python
# Process multiple models efficiently
batch_quantizer = BatchQuantizer()

parameter_sets = [model1_params, model2_params, model3_params]
quantized_models = batch_quantizer.quantize_batch(parameter_sets, parallel=True)

# Batch search
query_sets = [query1, query2, query3]
search_results = batch_quantizer.search_batch(query_sets, quantized_models)
```

### Memory Management

```python
# Configure memory limits
config = SystemConfig()
config.max_memory_usage_mb = 4096
config.compression.memory_limit_mb = 2048

quantizer = HilbertQuantizer(config)
```

### Caching and Registry Optimization

```python
# Optimize search caching
quantizer.update_configuration(
    search_enable_caching=True,
    search_cache_size_limit=50000
)

# Manage registry size
if quantizer.get_registry_info()["total_models"] > 1000:
    quantizer.clear_registry()  # Clear old models
```

## Examples

### Complete Workflow Example

```python
import numpy as np
from hilbert_quantization.api import HilbertQuantizer
from hilbert_quantization.config import create_high_quality_config

# Setup
quantizer = HilbertQuantizer(create_high_quality_config())

# Create test models
models = []
for i in range(5):
    params = np.random.randn(1000).astype(np.float32)
    quantized = quantizer.quantize(params, model_id=f"model_{i}")
    models.append((params, quantized))

# Search workflow
query_params = models[0][0] + 0.1 * np.random.randn(1000).astype(np.float32)
search_results = quantizer.search(query_params, max_results=3)

# Analyze results
print("Search Results:")
for i, result in enumerate(search_results):
    original_params = next(params for params, model in models 
                          if model.model_id == result.model.model_id)
    reconstructed = quantizer.reconstruct(result.model)
    
    query_similarity = 1.0 - np.mean(np.abs(query_params - original_params))
    reconstruction_accuracy = 1.0 - np.mean(np.abs(original_params - reconstructed))
    
    print(f"  {i+1}. Model: {result.model.model_id}")
    print(f"     Search similarity: {result.similarity_score:.3f}")
    print(f"     Actual similarity: {query_similarity:.3f}")
    print(f"     Reconstruction accuracy: {reconstruction_accuracy:.3f}")
```

### Model Comparison Example

```python
def compare_models(quantizer, model1_params, model2_params):
    """Compare two models using the quantization system."""
    
    # Quantize both models
    model1 = quantizer.quantize(model1_params, model_id="model_1")
    model2 = quantizer.quantize(model2_params, model_id="model_2")
    
    # Cross-search to find similarity
    results1 = quantizer.search(model1_params, [model2])
    results2 = quantizer.search(model2_params, [model1])
    
    # Calculate bidirectional similarity
    similarity_1_to_2 = results1[0].similarity_score if results1 else 0.0
    similarity_2_to_1 = results2[0].similarity_score if results2 else 0.0
    avg_similarity = (similarity_1_to_2 + similarity_2_to_1) / 2
    
    # Get compression metrics
    metrics1 = quantizer.get_compression_metrics(model1)
    metrics2 = quantizer.get_compression_metrics(model2)
    
    return {
        "similarity": avg_similarity,
        "model1_compression": metrics1.compression_ratio,
        "model2_compression": metrics2.compression_ratio,
        "model1_error": metrics1.reconstruction_error,
        "model2_error": metrics2.reconstruction_error
    }

# Usage
model1_params = np.random.randn(1000).astype(np.float32)
model2_params = model1_params + 0.2 * np.random.randn(1000).astype(np.float32)

comparison = compare_models(quantizer, model1_params, model2_params)
print(f"Model similarity: {comparison['similarity']:.3f}")
print(f"Compression ratios: {comparison['model1_compression']:.3f}, {comparison['model2_compression']:.3f}")
```

## Best Practices

### 1. Configuration Selection
- Use `create_default_config()` for balanced performance
- Use `create_high_performance_config()` for speed-critical applications
- Use `create_high_quality_config()` for accuracy-critical applications
- Use `get_optimal_configuration()` for model-size-specific optimization

### 2. Error Handling
- Always wrap API calls in try-catch blocks for production code
- Use validation flags appropriately (disable for performance, enable for debugging)
- Check configuration warnings and adjust settings accordingly

### 3. Performance Optimization
- Use batch processing for multiple models
- Enable caching for repeated searches
- Configure memory limits appropriately
- Use parallel processing when available

### 4. Model Management
- Use descriptive model IDs and descriptions
- Regularly clean up model registry to manage memory
- Save important models to disk for persistence
- Monitor compression metrics to ensure quality

### 5. Search Optimization
- Set appropriate similarity thresholds
- Use progressive filtering for large candidate pools
- Configure granularity levels based on model characteristics
- Monitor search performance and adjust parameters as needed

## Troubleshooting

### Common Issues

1. **Memory Issues**
   - Reduce batch sizes
   - Configure memory limits
   - Clear model registry regularly
   - Use lower compression quality for large models

2. **Performance Issues**
   - Enable parallel processing
   - Use batch operations
   - Optimize configuration for model size
   - Enable caching for repeated operations

3. **Quality Issues**
   - Increase compression quality
   - Enable strict validation
   - Use high-quality configuration preset
   - Monitor reconstruction errors

4. **Search Issues**
   - Adjust similarity thresholds
   - Increase max results
   - Check candidate pool size
   - Verify index granularity settings

For more detailed examples and advanced usage patterns, see the `examples/api_usage_examples.py` file.