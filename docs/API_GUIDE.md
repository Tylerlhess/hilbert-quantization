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

## Video-Enhanced Features

### VideoHilbertQuantizer

The `VideoHilbertQuantizer` extends the standard `HilbertQuantizer` with video-based storage and search capabilities, providing improved compression ratios and faster similarity search through temporal coherence analysis.

```python
from hilbert_quantization.video_api import VideoHilbertQuantizer

# Initialize with video storage
quantizer = VideoHilbertQuantizer(
    storage_dir="my_video_db",
    frame_rate=30.0,
    video_codec='mp4v',
    max_frames_per_video=10000,
    enable_video_storage=True
)
```

#### Key Methods

**quantize_and_store()** - Quantize and optionally store in video format:
```python
# Quantize and store in video format
quantized_model, frame_metadata = quantizer.quantize_and_store(
    parameters=model_params,
    model_id="transformer_layer_1",
    description="First transformer layer weights",
    store_in_video=True
)

print(f"Stored as frame {frame_metadata.frame_index}")
print(f"Video path: {frame_metadata.video_path}")
```

**video_search()** - Search using video-enhanced algorithms:
```python
# Search with different methods
results = quantizer.video_search(
    query_parameters=query_params,
    max_results=10,
    search_method='hybrid',  # 'video_features', 'hierarchical', or 'hybrid'
    use_temporal_coherence=True,
    similarity_threshold=0.1
)

# Process video search results
for result in results:
    print(f"Model: {result.frame_metadata.model_id}")
    print(f"Overall similarity: {result.similarity_score:.3f}")
    print(f"Video similarity: {result.video_similarity_score:.3f}")
    print(f"Hierarchical similarity: {result.hierarchical_similarity_score:.3f}")
    print(f"Temporal coherence: {result.temporal_coherence_score:.3f}")
    print(f"Search method: {result.search_method}")
```

**compare_search_methods()** - Compare different search approaches:
```python
# Compare all available search methods
comparison = quantizer.compare_search_methods(
    query_parameters=query_params,
    max_results=10
)

# Analyze performance differences
for method, stats in comparison['methods'].items():
    if 'error' not in stats:
        print(f"{method}:")
        print(f"  Search time: {stats['search_time']:.3f}s")
        print(f"  Results found: {stats['result_count']}")
        print(f"  Avg similarity: {stats['avg_similarity']:.3f}")
        print(f"  Method type: {stats['method_type']}")
```

**get_video_storage_info()** - Get comprehensive storage statistics:
```python
info = quantizer.get_video_storage_info()
print(f"Video storage enabled: {info['video_storage_enabled']}")
print(f"Total videos: {info['storage_statistics']['total_videos']}")
print(f"Total frames: {info['storage_statistics']['total_frames']}")
print(f"Average compression ratio: {info['storage_statistics']['avg_compression_ratio']:.2f}")
```

#### Search Method Options

The video-enhanced search engine supports multiple search methods:

1. **'video_features'** - Uses computer vision algorithms:
   - ORB keypoint detection and matching
   - Template matching for structural similarity
   - Histogram comparison for color/intensity patterns
   - Structural similarity (SSIM) for perceptual similarity

2. **'hierarchical'** - Uses embedded hierarchical indices:
   - Fast progressive filtering from coarse to fine granularity
   - Optimized space allocation across multiple levels
   - Spatial locality preservation from Hilbert curve mapping

3. **'hybrid'** - Combines both approaches (recommended):
   - Initial filtering using hierarchical indices (65% weight)
   - Detailed comparison using video features (35% weight)
   - Optional temporal coherence analysis for neighboring frames

#### Temporal Coherence Analysis

When enabled, temporal coherence analysis examines relationships between frames in the same video:

```python
# Enable temporal coherence for improved accuracy
results = quantizer.video_search(
    query_parameters=query_params,
    search_method='hybrid',
    use_temporal_coherence=True
)

# Temporal coherence scores indicate frame neighborhood similarity
for result in results:
    if result.temporal_coherence_score > 0.7:
        print(f"Strong temporal coherence: {result.temporal_coherence_score:.3f}")
```

### VideoBatchQuantizer

For processing multiple models efficiently with video storage:

```python
from hilbert_quantization.video_api import VideoBatchQuantizer

batch_quantizer = VideoBatchQuantizer(
    storage_dir="batch_video_storage",
    enable_video_storage=True
)

# Batch quantization with video storage
parameter_sets = [model1_params, model2_params, model3_params]
model_ids = ["model_1", "model_2", "model_3"]

quantized_models, frame_metadata_list = batch_quantizer.quantize_batch_to_video(
    parameter_sets=parameter_sets,
    model_ids=model_ids,
    store_in_video=True
)
```

### Convenience Functions

```python
from hilbert_quantization.video_api import (
    create_video_quantizer,
    quantize_model_to_video,
    video_search_similar_models
)

# Quick video quantizer creation
quantizer = create_video_quantizer(storage_dir="quick_storage")

# Single model quantization and storage
quantized_model, frame_metadata = quantize_model_to_video(
    parameters=model_params,
    model_id="quick_model"
)

# Direct similarity search
results = video_search_similar_models(
    query_parameters=query_params,
    storage_dir="quick_storage",
    search_method='hybrid'
)
```

## Hugging Face Integration

### HuggingFaceParameterExtractor

Extract parameters from Hugging Face models with stratified sampling and comprehensive metadata:

```python
from hilbert_quantization.huggingface_integration import HuggingFaceParameterExtractor

extractor = HuggingFaceParameterExtractor(cache_dir="./hf_cache")

# Extract parameters with filtering and sampling
result = extractor.extract_model_parameters(
    model_name="bert-base-uncased",
    max_params=100000,
    include_embeddings=True,
    include_attention=True,
    include_mlp=True,
    stratified_sampling=True
)

print(f"Extracted {len(result.parameters)} parameters")
print(f"Model type: {result.metadata.model_type}")
print(f"Architecture: {result.metadata.architecture}")
print(f"Hidden size: {result.metadata.hidden_size}")
print(f"Layers: {result.metadata.num_layers}")
print(f"Sampling applied: {result.sampling_applied}")
```

#### Layer Filtering Options

Control which layers to include in parameter extraction:

```python
# Include only attention layers
attention_result = extractor.extract_model_parameters(
    model_name="gpt2",
    include_embeddings=False,
    include_attention=True,
    include_mlp=False
)

# Include only MLP/feed-forward layers
mlp_result = extractor.extract_model_parameters(
    model_name="gpt2",
    include_embeddings=False,
    include_attention=False,
    include_mlp=True
)
```

#### Stratified Sampling

When parameter counts exceed limits, stratified sampling maintains representativeness:

```python
# Extract with stratified sampling to maintain layer proportions
sampled_result = extractor.extract_model_parameters(
    model_name="microsoft/DialoGPT-large",
    max_params=50000,
    stratified_sampling=True  # Maintains proportional representation
)

# Check sampling information
extraction_info = sampled_result.extraction_info
print(f"Original parameters: {extraction_info['original_parameter_count']}")
print(f"Final parameters: {extraction_info['final_parameter_count']}")
print(f"Layer counts: {extraction_info['layer_counts']}")
```

### HuggingFaceVideoEncoder

Complete workflow for encoding Hugging Face models to video format with registry tracking:

```python
from hilbert_quantization.huggingface_integration import HuggingFaceVideoEncoder

encoder = HuggingFaceVideoEncoder(
    cache_dir="./hf_cache",
    registry_path="hf_models/registry.json",
    video_storage_path="hf_models/videos"
)

# Encode model to video format
encoding_result = encoder.encode_model_to_video(
    model_name="distilbert-base-uncased",
    max_params=100000,
    compression_quality=0.85
)

print(f"Model ID: {encoding_result['model_id']}")
print(f"Encoding time: {encoding_result['encoding_time']:.2f}s")
print(f"Compression ratio: {encoding_result['compression_ratio']:.2f}")
print(f"Video frame: {encoding_result['video_frame_info']['frame_index']}")
```

#### Model Similarity Search

Search for similar models using various methods:

```python
# Search by model ID
similar_models = encoder.search_similar_models(
    query_model="distilbert_base_uncased",
    max_results=5,
    search_method="hybrid",
    architecture_filter="DistilBertModel"
)

# Search by parameter features
query_params = np.random.randn(50000).astype(np.float32)
feature_results = encoder.search_similar_models(
    query_model=query_params,
    max_results=10,
    search_method="features"
)

# Process search results
for result in similar_models:
    print(f"Model: {result['model_name']}")
    print(f"Similarity: {result['similarity_score']:.3f}")
    print(f"Architecture: {result['model_metadata']['architecture']}")
    print(f"Parameters: {result['model_metadata']['total_parameters']:,}")
    print(f"Encoding time: {result['encoding_statistics']['encoding_time']:.2f}s")
```

#### Model Registry Management

```python
# Get model information
model_info = encoder.get_model_info("bert_base_uncased")
if model_info:
    print(f"Model: {model_info['model_name']}")
    print(f"Registration: {model_info['registration_timestamp']}")
    print(f"Access count: {model_info['access_count']}")
    print(f"Tags: {model_info['tags']}")

# List registered models with filtering
models = encoder.list_registered_models(
    architecture_filter="BertModel",
    min_parameters=50000,
    max_parameters=200000
)

for model in models:
    print(f"{model['model_name']}: {model['total_parameters']:,} params")
```

### Convenience Functions

```python
from hilbert_quantization.huggingface_integration import (
    extract_huggingface_parameters,
    get_huggingface_model_info
)

# Quick parameter extraction
result = extract_huggingface_parameters(
    model_name="roberta-base",
    max_params=75000
)

# Quick model info
info = get_huggingface_model_info("gpt2")
print(f"GPT-2 has {info.total_parameters:,} parameters")
```

## Streaming Processing

### MemoryEfficientParameterStreamer

Process large models without loading them entirely into memory:

```python
from hilbert_quantization.core.streaming_processor import (
    MemoryEfficientParameterStreamer,
    StreamingConfig
)

# Configure streaming processor
config = StreamingConfig(
    chunk_size=2048,
    max_memory_mb=2048.0,
    enable_progress=True,
    adaptive_chunk_sizing=True,
    target_layers=['attention', 'mlp'],  # Filter specific layer types
    parallel_processing=True,
    enable_chunk_encoding=True,  # Encode chunks as video frames
    chunk_video_storage_dir="streaming_chunks"
)

streamer = MemoryEfficientParameterStreamer(config)

# Stream model parameters
model_name = "microsoft/DialoGPT-large"
max_params = 1000000

for chunk_array, chunk_metadata, progress in streamer.stream_model_parameters(
    model_name=model_name,
    max_total_params=max_params
):
    # Process each chunk
    print(f"Chunk {chunk_metadata.chunk_id}: {len(chunk_array)} parameters")
    print(f"Layer: {chunk_metadata.layer_name} ({chunk_metadata.layer_type})")
    print(f"Progress: {progress.progress_percent:.1f}%")
    print(f"Rate: {progress.processing_rate:.0f} params/sec")
    print(f"Memory: {progress.memory_usage_mb:.1f}MB")
    
    # Chunk is automatically encoded as video frame if enabled
    if chunk_metadata.video_path:
        print(f"Encoded to: {chunk_metadata.video_path}, frame {chunk_metadata.frame_index}")
```

#### Layer Filtering

Control which layers to process during streaming:

```python
# Process only attention layers
attention_config = StreamingConfig(
    target_layers=['attention'],
    exclude_layers=['embedding', 'output']
)

# Process with custom filter function
def custom_filter(layer_name):
    return 'transformer' in layer_name and 'bias' not in layer_name

custom_config = StreamingConfig(
    target_layers=None,  # Use custom function instead
    exclude_layers=None
)

streamer = MemoryEfficientParameterStreamer(custom_config)
```

#### Memory Management

The streaming processor includes adaptive memory management:

```python
# Configure memory limits and adaptive sizing
memory_config = StreamingConfig(
    chunk_size=1024,
    max_memory_mb=1024.0,
    adaptive_chunk_sizing=True,
    min_chunk_size=256,
    max_chunk_size=4096
)

# Monitor memory usage during streaming
streamer = MemoryEfficientParameterStreamer(memory_config)

# Get streaming statistics
stats = streamer.get_streaming_statistics()
print(f"Memory usage: {stats['memory_usage_mb']:.1f}MB")
print(f"Current chunk size: {stats['chunk_size']}")
print(f"Processing rate: {stats['processing_rate']:.0f} params/sec")
```

#### Chunk Encoding

When enabled, parameter chunks are encoded as video frames for efficient storage:

```python
# Enable chunk encoding
encoding_config = StreamingConfig(
    enable_chunk_encoding=True,
    chunk_video_storage_dir="encoded_chunks",
    chunk_frame_rate=30.0,
    chunk_video_codec='mp4v',
    max_chunks_per_video=1000
)

streamer = MemoryEfficientParameterStreamer(encoding_config)

# Stream with automatic chunk encoding
for chunk_array, chunk_metadata, progress in streamer.stream_model_parameters(
    model_name="gpt2",
    max_total_params=500000
):
    # Each chunk is automatically encoded as a video frame
    if chunk_metadata.video_path:
        print(f"Chunk {chunk_metadata.chunk_id} encoded to video frame")
        print(f"Video: {chunk_metadata.video_path}")
        print(f"Frame: {chunk_metadata.frame_index}")
        print(f"Hierarchical indices: {len(chunk_metadata.hierarchical_indices)} values")
```

#### Error Recovery

The streaming processor includes comprehensive error recovery:

```python
try:
    for chunk_array, chunk_metadata, progress in streamer.stream_model_parameters(
        model_name="very-large-model",
        max_total_params=10000000
    ):
        # Process chunks...
        pass
except Exception as e:
    # Attempt error recovery
    recovery_result = streamer.recover_from_streaming_error(e)
    print(f"Recovery actions: {recovery_result}")
    
    # Retry failed chunk encoding if needed
    if streamer.chunk_encoder:
        retry_result = streamer.retry_failed_chunk_encoding()
        print(f"Retry results: {retry_result}")
```

### Model Size Estimation

Estimate model parameters without loading the full model:

```python
# Estimate model size before streaming
estimated_size = streamer.estimate_model_size("microsoft/DialoGPT-large")
print(f"Estimated parameters: {estimated_size:,}")

# Use estimation for planning
if estimated_size > 1000000:
    # Use smaller chunks for very large models
    config.chunk_size = 512
    config.max_memory_mb = 4096.0
```

### Layer Analysis

Analyze layer composition before streaming:

```python
# Get layer filtering statistics
layer_stats = streamer.get_layer_filtering_statistics("bert-base-uncased")

print(f"Total layers: {layer_stats['total_layers']}")
print(f"Filtered layers: {layer_stats['filtered_layers']}")
print(f"Filter ratio: {layer_stats['filter_ratio']:.2f}")
print(f"Layer types: {layer_stats['all_layer_types']}")
print(f"Filtered types: {layer_stats['filtered_layer_types']}")
```

## Complete Workflow Examples

### Video-Enhanced Model Database

```python
from hilbert_quantization.video_api import VideoHilbertQuantizer
from hilbert_quantization.huggingface_integration import HuggingFaceVideoEncoder

# Create video-enhanced quantizer
quantizer = VideoHilbertQuantizer(
    storage_dir="model_database",
    enable_video_storage=True
)

# Create Hugging Face encoder
hf_encoder = HuggingFaceVideoEncoder(
    video_storage_path="model_database/hf_models"
)

# Encode multiple Hugging Face models
models_to_encode = [
    "bert-base-uncased",
    "distilbert-base-uncased", 
    "roberta-base",
    "gpt2"
]

for model_name in models_to_encode:
    try:
        result = hf_encoder.encode_model_to_video(
            model_name=model_name,
            max_params=100000,
            compression_quality=0.8
        )
        print(f"Encoded {model_name}: {result['compression_ratio']:.2f}x compression")
    except Exception as e:
        print(f"Failed to encode {model_name}: {e}")

# Search for similar models
query_model = "bert-base-uncased"
similar_models = hf_encoder.search_similar_models(
    query_model=query_model,
    max_results=3,
    search_method="hybrid"
)

print(f"\nModels similar to {query_model}:")
for result in similar_models:
    print(f"  {result['model_name']}: {result['similarity_score']:.3f}")
```

### Large Model Streaming Pipeline

```python
from hilbert_quantization.core.streaming_processor import (
    MemoryEfficientParameterStreamer,
    StreamingConfig
)
from hilbert_quantization.video_api import VideoHilbertQuantizer

# Configure for large model processing
streaming_config = StreamingConfig(
    chunk_size=1024,
    max_memory_mb=2048.0,
    adaptive_chunk_sizing=True,
    target_layers=['attention', 'mlp'],
    enable_chunk_encoding=True,
    chunk_video_storage_dir="large_model_chunks"
)

# Initialize components
streamer = MemoryEfficientParameterStreamer(streaming_config)
quantizer = VideoHilbertQuantizer(storage_dir="large_models")

# Process large model in chunks
model_name = "microsoft/DialoGPT-large"
processed_chunks = []

print(f"Processing {model_name} in streaming mode...")

for chunk_array, chunk_metadata, progress in streamer.stream_model_parameters(
    model_name=model_name,
    max_total_params=2000000
):
    # Quantize each chunk
    chunk_quantized = quantizer.quantize(
        chunk_array,
        model_id=f"{model_name}_chunk_{chunk_metadata.chunk_id}",
        validate=False  # Skip validation for performance
    )
    
    processed_chunks.append(chunk_quantized)
    
    # Log progress every 10 chunks
    if chunk_metadata.chunk_id % 10 == 0:
        print(f"  Processed {progress.processed_parameters:,} parameters "
              f"({progress.progress_percent:.1f}%) "
              f"Rate: {progress.processing_rate:.0f} params/sec")

print(f"Completed processing: {len(processed_chunks)} chunks")

# Search across processed chunks
query_chunk = processed_chunks[0]
similar_chunks = quantizer.search(
    query_chunk.hierarchical_indices,
    candidate_models=processed_chunks[1:],
    max_results=5
)

print(f"Found {len(similar_chunks)} similar chunks")
```

For more detailed examples and advanced usage patterns, see the `examples/api_usage_examples.py` file.