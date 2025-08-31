# Hugging Face Integration API Examples

This document provides comprehensive examples for using the Hilbert Quantization system with Hugging Face models, including parameter extraction, video encoding, and similarity search.

## Table of Contents

1. [Basic Parameter Extraction](#basic-parameter-extraction)
2. [Advanced Layer Filtering](#advanced-layer-filtering)
3. [Video Encoding Workflow](#video-encoding-workflow)
4. [Model Similarity Search](#model-similarity-search)
5. [Registry Management](#registry-management)
6. [Streaming Large Models](#streaming-large-models)
7. [Performance Optimization](#performance-optimization)
8. [Error Handling](#error-handling)

## Basic Parameter Extraction

### Simple Parameter Extraction

```python
from hilbert_quantization.huggingface_integration import extract_huggingface_parameters

# Extract parameters from a small model
result = extract_huggingface_parameters(
    model_name="distilbert-base-uncased",
    max_params=50000
)

print(f"Model: {result.metadata.model_name}")
print(f"Architecture: {result.metadata.architecture}")
print(f"Parameters extracted: {len(result.parameters):,}")
print(f"Original parameter count: {result.original_parameter_count:,}")
print(f"Sampling applied: {result.sampling_applied}")

# Access model metadata
metadata = result.metadata
print(f"\nModel Details:")
print(f"  Model type: {metadata.model_type}")
print(f"  Hidden size: {metadata.hidden_size}")
print(f"  Number of layers: {metadata.num_layers}")
print(f"  Attention heads: {metadata.num_attention_heads}")
print(f"  Vocabulary size: {metadata.vocab_size}")
print(f"  Model size: {metadata.model_size_mb:.2f} MB")
```

### Detailed Parameter Extraction with Custom Settings

```python
from hilbert_quantization.huggingface_integration import HuggingFaceParameterExtractor

# Initialize extractor with custom cache directory
extractor = HuggingFaceParameterExtractor(cache_dir="./custom_hf_cache")

# Extract with detailed configuration
result = extractor.extract_model_parameters(
    model_name="bert-base-uncased",
    max_params=100000,
    include_embeddings=True,
    include_attention=True,
    include_mlp=True,
    stratified_sampling=True
)

# Analyze extraction results
extraction_info = result.extraction_info
print(f"Extraction Summary:")
print(f"  Total layers processed: {extraction_info['total_layers_processed']}")
print(f"  Layer type counts: {extraction_info['layer_counts']}")
print(f"  Sampling strategy: {'Stratified' if extraction_info['stratified_sampling'] else 'Sequential'}")

# Examine parameter sources
print(f"\nParameter Sources:")
for source in extraction_info['parameter_sources'][:5]:  # Show first 5
    print(f"  {source['name']}: {source['parameter_count']:,} params ({source['layer_type']})")
```

## Advanced Layer Filtering

### Filter by Layer Types

```python
# Extract only attention parameters
attention_result = extractor.extract_model_parameters(
    model_name="gpt2",
    max_params=75000,
    include_embeddings=False,
    include_attention=True,
    include_mlp=False
)

print(f"Attention-only extraction:")
print(f"  Parameters: {len(attention_result.parameters):,}")
print(f"  Layer counts: {attention_result.extraction_info['layer_counts']}")

# Extract only MLP/feed-forward parameters
mlp_result = extractor.extract_model_parameters(
    model_name="gpt2",
    max_params=75000,
    include_embeddings=False,
    include_attention=False,
    include_mlp=True
)

print(f"\nMLP-only extraction:")
print(f"  Parameters: {len(mlp_result.parameters):,}")
print(f"  Layer counts: {mlp_result.extraction_info['layer_counts']}")

# Compare parameter distributions
print(f"\nParameter Distribution Comparison:")
print(f"  Attention layers: {len(attention_result.parameters):,} parameters")
print(f"  MLP layers: {len(mlp_result.parameters):,} parameters")
print(f"  Ratio (Attention/MLP): {len(attention_result.parameters) / len(mlp_result.parameters):.2f}")
```

### Stratified vs Sequential Sampling

```python
# Compare sampling strategies for large models
model_name = "microsoft/DialoGPT-medium"
max_params = 50000

# Stratified sampling (maintains layer proportions)
stratified_result = extractor.extract_model_parameters(
    model_name=model_name,
    max_params=max_params,
    stratified_sampling=True
)

# Sequential sampling (takes first N parameters)
sequential_result = extractor.extract_model_parameters(
    model_name=model_name,
    max_params=max_params,
    stratified_sampling=False
)

print(f"Sampling Strategy Comparison for {model_name}:")
print(f"\nStratified Sampling:")
print(f"  Layer counts: {stratified_result.extraction_info['layer_counts']}")

print(f"\nSequential Sampling:")
print(f"  Layer counts: {sequential_result.extraction_info['layer_counts']}")

# Analyze representativeness
stratified_types = len(stratified_result.extraction_info['layer_counts'])
sequential_types = len(sequential_result.extraction_info['layer_counts'])
print(f"\nRepresentativeness:")
print(f"  Stratified covers {stratified_types} layer types")
print(f"  Sequential covers {sequential_types} layer types")
```

## Video Encoding Workflow

### Basic Model Encoding

```python
from hilbert_quantization.huggingface_integration import HuggingFaceVideoEncoder

# Initialize encoder with registry
encoder = HuggingFaceVideoEncoder(
    cache_dir="./hf_cache",
    registry_path="model_registry.json",
    video_storage_path="encoded_models"
)

# Encode a model to video format
encoding_result = encoder.encode_model_to_video(
    model_name="distilbert-base-uncased",
    max_params=100000,
    compression_quality=0.85,
    include_embeddings=True,
    include_attention=True,
    include_mlp=True
)

print(f"Encoding Results:")
print(f"  Model ID: {encoding_result['model_id']}")
print(f"  Encoding time: {encoding_result['encoding_time']:.2f} seconds")
print(f"  Parameter count: {encoding_result['parameter_count']:,}")
print(f"  Compression ratio: {encoding_result['compression_ratio']:.2f}x")

# Video frame information
frame_info = encoding_result['video_frame_info']
print(f"\nVideo Storage:")
print(f"  Frame index: {frame_info['frame_index']}")
print(f"  Video path: {frame_info['video_path']}")
print(f"  Frame timestamp: {frame_info['frame_timestamp']}")

# Registry information
print(f"\nRegistry:")
print(f"  Registry entry ID: {encoding_result['registry_entry_id']}")
print(f"  Hierarchical indices: {len(encoding_result['hierarchical_indices'])} values")
```

### Batch Model Encoding

```python
# Encode multiple models with different configurations
models_to_encode = [
    {
        "name": "bert-base-uncased",
        "max_params": 100000,
        "quality": 0.9,
        "layers": {"embeddings": True, "attention": True, "mlp": True}
    },
    {
        "name": "distilbert-base-uncased", 
        "max_params": 75000,
        "quality": 0.85,
        "layers": {"embeddings": True, "attention": True, "mlp": False}
    },
    {
        "name": "roberta-base",
        "max_params": 120000,
        "quality": 0.8,
        "layers": {"embeddings": False, "attention": True, "mlp": True}
    }
]

encoding_results = []

for model_config in models_to_encode:
    try:
        print(f"Encoding {model_config['name']}...")
        
        result = encoder.encode_model_to_video(
            model_name=model_config['name'],
            max_params=model_config['max_params'],
            compression_quality=model_config['quality'],
            **model_config['layers']
        )
        
        encoding_results.append(result)
        
        print(f"  ✓ Success: {result['compression_ratio']:.2f}x compression")
        print(f"    Time: {result['encoding_time']:.2f}s")
        print(f"    Parameters: {result['parameter_count']:,}")
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")

# Analyze batch results
if encoding_results:
    avg_compression = sum(r['compression_ratio'] for r in encoding_results) / len(encoding_results)
    avg_time = sum(r['encoding_time'] for r in encoding_results) / len(encoding_results)
    total_params = sum(r['parameter_count'] for r in encoding_results)
    
    print(f"\nBatch Encoding Summary:")
    print(f"  Models encoded: {len(encoding_results)}")
    print(f"  Average compression ratio: {avg_compression:.2f}x")
    print(f"  Average encoding time: {avg_time:.2f}s")
    print(f"  Total parameters processed: {total_params:,}")
```

## Model Similarity Search

### Search by Model ID

```python
# Search for models similar to a specific encoded model
query_model_id = "distilbert_base_uncased"

similar_models = encoder.search_similar_models(
    query_model=query_model_id,
    max_results=5,
    similarity_threshold=0.1,
    search_method="hybrid"
)

print(f"Models similar to {query_model_id}:")
for i, result in enumerate(similar_models, 1):
    print(f"\n{i}. {result['model_name']}")
    print(f"   Similarity score: {result['similarity_score']:.3f}")
    print(f"   Architecture: {result['model_metadata']['architecture']}")
    print(f"   Parameters: {result['model_metadata']['total_parameters']:,}")
    print(f"   Hidden size: {result['model_metadata']['hidden_size']}")
    print(f"   Layers: {result['model_metadata']['num_layers']}")
    
    # Similarity breakdown
    breakdown = result['similarity_breakdown']
    print(f"   Similarity details: {breakdown}")
```

### Search by Parameter Features

```python
import numpy as np

# Create query parameters (could be from a new model or synthetic)
query_params = np.random.randn(50000).astype(np.float32)

# Search using parameter features
feature_results = encoder.search_similar_models(
    query_model=query_params,
    max_results=10,
    search_method="features",
    similarity_threshold=0.05
)

print(f"Feature-based search results ({len(feature_results)} found):")
for result in feature_results:
    print(f"  {result['model_name']}: {result['similarity_score']:.3f}")
    
    # Show encoding statistics
    stats = result['encoding_statistics']
    print(f"    Encoding time: {stats['encoding_time']:.2f}s")
    print(f"    Compression: {stats['compression_ratio']:.2f}x")
    print(f"    Quality score: {stats['quality_score']:.3f}")
```

### Architecture-Filtered Search

```python
# Search within specific model architectures
bert_models = encoder.search_similar_models(
    query_model="bert_base_uncased",
    max_results=10,
    architecture_filter="BertModel",
    search_method="hybrid"
)

print(f"BERT-family models:")
for result in bert_models:
    metadata = result['model_metadata']
    print(f"  {result['model_name']}")
    print(f"    Similarity: {result['similarity_score']:.3f}")
    print(f"    Hidden size: {metadata['hidden_size']}")
    print(f"    Layers: {metadata['num_layers']}")
    print(f"    Parameters: {metadata['total_parameters']:,}")

# Search for models with similar parameter counts
param_range_models = encoder.list_registered_models(
    min_parameters=80000,
    max_parameters=120000
)

print(f"\nModels with 80K-120K parameters:")
for model in param_range_models:
    print(f"  {model['model_name']}: {model['total_parameters']:,} params")
```

### Search Method Comparison

```python
# Compare different search methods on the same query
query_model = "roberta_base"
search_methods = ["features", "metadata", "hybrid"]

print(f"Search method comparison for {query_model}:")

for method in search_methods:
    try:
        results = encoder.search_similar_models(
            query_model=query_model,
            max_results=3,
            search_method=method
        )
        
        print(f"\n{method.upper()} method:")
        for result in results:
            print(f"  {result['model_name']}: {result['similarity_score']:.3f}")
            
    except Exception as e:
        print(f"\n{method.upper()} method: Failed - {e}")

# Analyze search performance
print(f"\nSearch method characteristics:")
print(f"  features: Uses parameter vector similarity")
print(f"  metadata: Uses model architecture similarity") 
print(f"  hybrid: Combines features + metadata (recommended)")
```

## Registry Management

### Model Information and Statistics

```python
# Get detailed information about a specific model
model_id = "bert_base_uncased"
model_info = encoder.get_model_info(model_id)

if model_info:
    print(f"Model Information for {model_id}:")
    print(f"  Name: {model_info['model_name']}")
    print(f"  Registration: {model_info['registration_timestamp']}")
    print(f"  Last accessed: {model_info['last_accessed']}")
    print(f"  Access count: {model_info['access_count']}")
    print(f"  Storage location: {model_info['storage_location']}")
    print(f"  Tags: {model_info['tags']}")
    print(f"  Notes: {model_info['notes']}")
    
    # Model metadata
    metadata = model_info['model_metadata']
    print(f"\n  Architecture Details:")
    print(f"    Type: {metadata['model_type']}")
    print(f"    Architecture: {metadata['architecture']}")
    print(f"    Hidden size: {metadata['hidden_size']}")
    print(f"    Layers: {metadata['num_layers']}")
    print(f"    Attention heads: {metadata['num_attention_heads']}")
    print(f"    Vocabulary: {metadata['vocab_size']:,}")
    
    # Encoding statistics
    stats = model_info['encoding_statistics']
    print(f"\n  Encoding Statistics:")
    print(f"    Encoding time: {stats['encoding_time']:.2f}s")
    print(f"    Compression ratio: {stats['compression_ratio']:.2f}x")
    print(f"    Parameter count: {stats['parameter_count']:,}")
    print(f"    Quality score: {stats['quality_score']:.3f}")
else:
    print(f"Model {model_id} not found in registry")
```

### Registry Filtering and Analysis

```python
# List models with various filters
print("Registry Analysis:")

# All BERT-family models
bert_models = encoder.list_registered_models(
    architecture_filter="BertModel"
)
print(f"\nBERT models: {len(bert_models)}")
for model in bert_models:
    print(f"  {model['model_name']}: {model['total_parameters']:,} params")

# Models with specific tags
tagged_models = encoder.list_registered_models(
    tag_filter=["huggingface", "transformer"]
)
print(f"\nTagged models: {len(tagged_models)}")

# Large models (>100K parameters)
large_models = encoder.list_registered_models(
    min_parameters=100000
)
print(f"\nLarge models (>100K params): {len(large_models)}")

# Small models (<50K parameters)  
small_models = encoder.list_registered_models(
    max_parameters=50000
)
print(f"Small models (<50K params): {len(small_models)}")

# Registry statistics
all_models = encoder.list_registered_models()
if all_models:
    total_params = sum(m['total_parameters'] for m in all_models)
    avg_params = total_params / len(all_models)
    
    print(f"\nRegistry Statistics:")
    print(f"  Total models: {len(all_models)}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Average parameters per model: {avg_params:,.0f}")
    
    # Architecture distribution
    architectures = {}
    for model in all_models:
        arch = model['architecture']
        architectures[arch] = architectures.get(arch, 0) + 1
    
    print(f"  Architecture distribution:")
    for arch, count in sorted(architectures.items()):
        print(f"    {arch}: {count} models")
```

## Streaming Large Models

### Basic Streaming Setup

```python
from hilbert_quantization.core.streaming_processor import (
    MemoryEfficientParameterStreamer,
    StreamingConfig
)

# Configure for large model streaming
streaming_config = StreamingConfig(
    chunk_size=2048,
    max_memory_mb=2048.0,
    enable_progress=True,
    adaptive_chunk_sizing=True,
    target_layers=['attention', 'mlp'],
    enable_chunk_encoding=True,
    chunk_video_storage_dir="streaming_chunks"
)

streamer = MemoryEfficientParameterStreamer(streaming_config)

# Stream a large model
model_name = "microsoft/DialoGPT-large"
max_params = 1000000

print(f"Streaming {model_name} (max {max_params:,} parameters)...")

chunk_count = 0
total_processed = 0

for chunk_array, chunk_metadata, progress in streamer.stream_model_parameters(
    model_name=model_name,
    max_total_params=max_params
):
    chunk_count += 1
    total_processed += len(chunk_array)
    
    # Log progress every 10 chunks
    if chunk_count % 10 == 0:
        print(f"  Chunk {chunk_count}: {len(chunk_array)} params")
        print(f"    Layer: {chunk_metadata.layer_name} ({chunk_metadata.layer_type})")
        print(f"    Progress: {progress.progress_percent:.1f}%")
        print(f"    Rate: {progress.processing_rate:.0f} params/sec")
        print(f"    Memory: {progress.memory_usage_mb:.1f}MB")
        print(f"    ETA: {progress.estimated_completion_time:.1f}s")
        
        # Check if chunk was encoded as video frame
        if chunk_metadata.video_path:
            print(f"    Encoded to: frame {chunk_metadata.frame_index}")

print(f"\nStreaming completed:")
print(f"  Total chunks: {chunk_count}")
print(f"  Total parameters: {total_processed:,}")
```

### Advanced Streaming with Custom Processing

```python
# Custom processing function for each chunk
def process_chunk(chunk_array, chunk_metadata, progress):
    """Custom processing for each parameter chunk."""
    
    # Analyze chunk statistics
    chunk_stats = {
        'mean': np.mean(chunk_array),
        'std': np.std(chunk_array),
        'min': np.min(chunk_array),
        'max': np.max(chunk_array),
        'zeros': np.sum(chunk_array == 0),
        'sparsity': np.sum(chunk_array == 0) / len(chunk_array)
    }
    
    return chunk_stats

# Stream with custom processing
model_name = "gpt2"
chunk_statistics = []

print(f"Streaming {model_name} with statistical analysis...")

for chunk_array, chunk_metadata, progress in streamer.stream_model_parameters(
    model_name=model_name,
    max_total_params=500000
):
    # Apply custom processing
    stats = process_chunk(chunk_array, chunk_metadata, progress)
    stats['layer_name'] = chunk_metadata.layer_name
    stats['layer_type'] = chunk_metadata.layer_type
    chunk_statistics.append(stats)
    
    # Log interesting findings
    if stats['sparsity'] > 0.5:
        print(f"  High sparsity chunk: {chunk_metadata.layer_name} ({stats['sparsity']:.2f})")
    
    if abs(stats['mean']) > 1.0:
        print(f"  High magnitude chunk: {chunk_metadata.layer_name} (mean: {stats['mean']:.3f})")

# Analyze collected statistics
if chunk_statistics:
    print(f"\nStatistical Analysis ({len(chunk_statistics)} chunks):")
    
    # Overall statistics
    all_means = [s['mean'] for s in chunk_statistics]
    all_stds = [s['std'] for s in chunk_statistics]
    all_sparsities = [s['sparsity'] for s in chunk_statistics]
    
    print(f"  Mean distribution: {np.mean(all_means):.3f} ± {np.std(all_means):.3f}")
    print(f"  Std distribution: {np.mean(all_stds):.3f} ± {np.std(all_stds):.3f}")
    print(f"  Average sparsity: {np.mean(all_sparsities):.3f}")
    
    # Layer type analysis
    layer_types = {}
    for stats in chunk_statistics:
        layer_type = stats['layer_type']
        if layer_type not in layer_types:
            layer_types[layer_type] = []
        layer_types[layer_type].append(stats['sparsity'])
    
    print(f"\n  Sparsity by layer type:")
    for layer_type, sparsities in layer_types.items():
        avg_sparsity = np.mean(sparsities)
        print(f"    {layer_type}: {avg_sparsity:.3f} average sparsity")
```

## Performance Optimization

### Memory Usage Optimization

```python
# Configure for minimal memory usage
memory_optimized_config = StreamingConfig(
    chunk_size=512,  # Smaller chunks
    max_memory_mb=1024.0,  # Lower memory limit
    adaptive_chunk_sizing=True,
    min_chunk_size=128,
    max_chunk_size=1024,
    parallel_processing=False,  # Disable parallel processing
    enable_progress=True,
    progress_interval=500
)

memory_streamer = MemoryEfficientParameterStreamer(memory_optimized_config)

# Monitor memory usage during streaming
model_name = "bert-base-uncased"
memory_samples = []

for chunk_array, chunk_metadata, progress in memory_streamer.stream_model_parameters(
    model_name=model_name,
    max_total_params=200000
):
    memory_samples.append({
        'chunk_id': chunk_metadata.chunk_id,
        'memory_mb': progress.memory_usage_mb,
        'chunk_size': len(chunk_array),
        'processing_rate': progress.processing_rate
    })
    
    # Adaptive chunk size monitoring
    if chunk_metadata.chunk_id % 20 == 0:
        current_config = memory_streamer.config
        print(f"  Chunk {chunk_metadata.chunk_id}: "
              f"size={current_config.chunk_size}, "
              f"memory={progress.memory_usage_mb:.1f}MB")

# Analyze memory usage patterns
if memory_samples:
    max_memory = max(s['memory_mb'] for s in memory_samples)
    avg_memory = sum(s['memory_mb'] for s in memory_samples) / len(memory_samples)
    
    print(f"\nMemory Usage Analysis:")
    print(f"  Peak memory: {max_memory:.1f}MB")
    print(f"  Average memory: {avg_memory:.1f}MB")
    print(f"  Memory efficiency: {avg_memory/memory_optimized_config.max_memory_mb:.2%}")
```

### Parallel Processing Optimization

```python
# Configure for maximum throughput
throughput_config = StreamingConfig(
    chunk_size=4096,  # Larger chunks
    max_memory_mb=4096.0,  # Higher memory limit
    parallel_processing=True,
    max_workers=4,
    adaptive_chunk_sizing=True,
    enable_chunk_encoding=True
)

throughput_streamer = MemoryEfficientParameterStreamer(throughput_config)

# Benchmark processing speed
import time

model_name = "roberta-base"
start_time = time.time()
processed_chunks = 0
processed_params = 0

for chunk_array, chunk_metadata, progress in throughput_streamer.stream_model_parameters(
    model_name=model_name,
    max_total_params=300000
):
    processed_chunks += 1
    processed_params += len(chunk_array)

end_time = time.time()
total_time = end_time - start_time

print(f"Throughput Benchmark Results:")
print(f"  Model: {model_name}")
print(f"  Total time: {total_time:.2f}s")
print(f"  Chunks processed: {processed_chunks}")
print(f"  Parameters processed: {processed_params:,}")
print(f"  Throughput: {processed_params/total_time:.0f} params/sec")
print(f"  Chunk rate: {processed_chunks/total_time:.1f} chunks/sec")
```

## Error Handling

### Robust Model Processing

```python
def robust_model_encoding(encoder, model_names, max_retries=3):
    """Robustly encode multiple models with error handling and retries."""
    
    results = {
        'successful': [],
        'failed': [],
        'retried': []
    }
    
    for model_name in model_names:
        retry_count = 0
        success = False
        
        while retry_count < max_retries and not success:
            try:
                print(f"Encoding {model_name} (attempt {retry_count + 1})...")
                
                # Attempt encoding
                result = encoder.encode_model_to_video(
                    model_name=model_name,
                    max_params=100000,
                    compression_quality=0.8
                )
                
                results['successful'].append({
                    'model_name': model_name,
                    'result': result,
                    'attempts': retry_count + 1
                })
                
                success = True
                print(f"  ✓ Success on attempt {retry_count + 1}")
                
            except Exception as e:
                retry_count += 1
                error_msg = str(e)
                
                print(f"  ✗ Attempt {retry_count} failed: {error_msg}")
                
                if retry_count < max_retries:
                    # Implement retry strategies based on error type
                    if "memory" in error_msg.lower():
                        print(f"    Memory error detected, reducing parameters...")
                        # Could reduce max_params for retry
                    elif "network" in error_msg.lower() or "connection" in error_msg.lower():
                        print(f"    Network error detected, waiting before retry...")
                        time.sleep(5)  # Wait before retry
                    
                    results['retried'].append({
                        'model_name': model_name,
                        'attempt': retry_count,
                        'error': error_msg
                    })
                else:
                    results['failed'].append({
                        'model_name': model_name,
                        'final_error': error_msg,
                        'total_attempts': retry_count
                    })
    
    return results

# Test robust encoding
test_models = [
    "distilbert-base-uncased",
    "bert-base-uncased", 
    "roberta-base",
    "nonexistent-model",  # This will fail
    "gpt2"
]

encoding_results = robust_model_encoding(encoder, test_models)

print(f"\nRobust Encoding Results:")
print(f"  Successful: {len(encoding_results['successful'])}")
print(f"  Failed: {len(encoding_results['failed'])}")
print(f"  Required retries: {len(encoding_results['retried'])}")

# Show details
for success in encoding_results['successful']:
    attempts = success['attempts']
    retry_text = f" (after {attempts} attempts)" if attempts > 1 else ""
    print(f"  ✓ {success['model_name']}{retry_text}")

for failure in encoding_results['failed']:
    print(f"  ✗ {failure['model_name']}: {failure['final_error']}")
```

### Streaming Error Recovery

```python
def robust_streaming_with_recovery(streamer, model_name, max_params):
    """Stream with comprehensive error recovery."""
    
    recovery_stats = {
        'chunk_errors': 0,
        'memory_adjustments': 0,
        'successful_chunks': 0,
        'recovered_errors': []
    }
    
    try:
        for chunk_array, chunk_metadata, progress in streamer.stream_model_parameters(
            model_name=model_name,
            max_total_params=max_params
        ):
            try:
                # Process chunk (could add custom processing here)
                recovery_stats['successful_chunks'] += 1
                
                # Monitor for potential issues
                if progress.memory_usage_mb > streamer.config.max_memory_mb * 0.9:
                    print(f"  Warning: High memory usage ({progress.memory_usage_mb:.1f}MB)")
                
            except Exception as chunk_error:
                recovery_stats['chunk_errors'] += 1
                recovery_stats['recovered_errors'].append({
                    'chunk_id': chunk_metadata.chunk_id,
                    'error': str(chunk_error),
                    'layer': chunk_metadata.layer_name
                })
                
                print(f"  Chunk error in {chunk_metadata.layer_name}: {chunk_error}")
                # Continue processing other chunks
                
    except Exception as streaming_error:
        print(f"Streaming error: {streaming_error}")
        
        # Attempt error recovery
        try:
            recovery_result = streamer.recover_from_streaming_error(streaming_error)
            print(f"Recovery actions: {recovery_result}")
            
            # Update recovery stats
            if 'memory' in str(streaming_error).lower():
                recovery_stats['memory_adjustments'] += 1
                
        except Exception as recovery_error:
            print(f"Recovery failed: {recovery_error}")
    
    return recovery_stats

# Test streaming with recovery
streaming_config = StreamingConfig(
    chunk_size=1024,
    max_memory_mb=1024.0,
    adaptive_chunk_sizing=True
)

recovery_streamer = MemoryEfficientParameterStreamer(streaming_config)

print("Testing streaming with error recovery...")
recovery_stats = robust_streaming_with_recovery(
    recovery_streamer, 
    "microsoft/DialoGPT-medium", 
    500000
)

print(f"\nRecovery Statistics:")
print(f"  Successful chunks: {recovery_stats['successful_chunks']}")
print(f"  Chunk errors: {recovery_stats['chunk_errors']}")
print(f"  Memory adjustments: {recovery_stats['memory_adjustments']}")
print(f"  Recovered errors: {len(recovery_stats['recovered_errors'])}")

for error in recovery_stats['recovered_errors']:
    print(f"    Chunk {error['chunk_id']} ({error['layer']}): {error['error']}")
```

## Complete Integration Example

```python
def complete_huggingface_workflow():
    """Demonstrate a complete workflow with Hugging Face integration."""
    
    print("=== Complete Hugging Face Integration Workflow ===\n")
    
    # 1. Initialize components
    print("1. Initializing components...")
    encoder = HuggingFaceVideoEncoder(
        cache_dir="./workflow_cache",
        registry_path="workflow_registry.json",
        video_storage_path="workflow_videos"
    )
    
    streaming_config = StreamingConfig(
        chunk_size=1024,
        max_memory_mb=2048.0,
        enable_chunk_encoding=True,
        target_layers=['attention', 'mlp']
    )
    streamer = MemoryEfficientParameterStreamer(streaming_config)
    
    # 2. Encode multiple models
    print("\n2. Encoding models to video format...")
    models_to_process = [
        ("distilbert-base-uncased", 75000),
        ("bert-base-uncased", 100000),
        ("roberta-base", 120000)
    ]
    
    encoded_models = []
    for model_name, max_params in models_to_process:
        try:
            result = encoder.encode_model_to_video(
                model_name=model_name,
                max_params=max_params,
                compression_quality=0.85
            )
            encoded_models.append(result)
            print(f"  ✓ {model_name}: {result['compression_ratio']:.2f}x compression")
        except Exception as e:
            print(f"  ✗ {model_name}: {e}")
    
    # 3. Perform similarity searches
    print("\n3. Performing similarity searches...")
    if encoded_models:
        query_model = encoded_models[0]['model_id']
        
        similar_models = encoder.search_similar_models(
            query_model=query_model,
            max_results=5,
            search_method="hybrid"
        )
        
        print(f"Models similar to {query_model}:")
        for result in similar_models:
            print(f"  {result['model_name']}: {result['similarity_score']:.3f}")
    
    # 4. Stream a large model
    print("\n4. Streaming large model processing...")
    large_model = "microsoft/DialoGPT-medium"
    
    chunk_count = 0
    for chunk_array, chunk_metadata, progress in streamer.stream_model_parameters(
        model_name=large_model,
        max_total_params=200000
    ):
        chunk_count += 1
        if chunk_count % 20 == 0:
            print(f"  Processed {progress.processed_parameters:,} parameters "
                  f"({progress.progress_percent:.1f}%)")
    
    # 5. Registry analysis
    print("\n5. Registry analysis...")
    all_models = encoder.list_registered_models()
    
    if all_models:
        total_params = sum(m['total_parameters'] for m in all_models)
        architectures = set(m['architecture'] for m in all_models)
        
        print(f"  Total models in registry: {len(all_models)}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Unique architectures: {len(architectures)}")
        print(f"  Architectures: {', '.join(sorted(architectures))}")
    
    print("\n=== Workflow Complete ===")

# Run the complete workflow
if __name__ == "__main__":
    complete_huggingface_workflow()
```

This comprehensive example demonstrates all major features of the Hugging Face integration, including parameter extraction, video encoding, similarity search, streaming processing, and error handling. The examples show both basic usage and advanced configurations for production use cases.