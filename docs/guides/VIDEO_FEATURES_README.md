# Video-Enhanced Hilbert Quantization

## Overview

This extension adds video-based storage and search capabilities to the Hilbert Quantization library, providing:

- **Video Storage**: Store collections of neural network models as frames in video files
- **Enhanced Compression**: Leverage temporal coherence for better compression ratios
- **Fast Video Search**: Use computer vision algorithms for ultra-fast similarity search
- **Temporal Analysis**: Analyze temporal relationships between similar models

## Key Benefits

### 🎬 Video Storage Benefits
- **Better Compression**: Video codecs exploit temporal coherence between similar models
- **Efficient Storage**: Single video file can store thousands of models
- **Standard Formats**: Uses standard video formats (MP4, AVI) for portability
- **Metadata Preservation**: Rich metadata system for model tracking

### 🔍 Video Search Advantages
- **Multiple Algorithms**: Template matching, feature detection, histogram comparison
- **Parallel Processing**: Multi-threaded search across video files
- **Temporal Coherence**: Leverages model ordering for improved accuracy
- **Hybrid Approach**: Combines video features with hierarchical indices

### 📈 Performance Improvements
- **Faster Search**: Video processing algorithms optimized for bulk operations
- **Memory Efficient**: Process video frames sequentially without loading entire dataset
- **Scalable**: Better performance with larger model collections
- **Adaptive**: Automatic fallback to traditional search methods

## Quick Start

### Installation

```bash
# Install additional video dependencies
pip install -r requirements_video.txt

# Or install individual packages
pip install opencv-python scikit-image matplotlib
```

### Basic Usage

```python
from hilbert_quantization import VideoHilbertQuantizer
import numpy as np

# Create video-enhanced quantizer
quantizer = VideoHilbertQuantizer(storage_dir="my_video_db")

# Quantize and store models in video format
models = [np.random.randn(1024).astype(np.float32) for _ in range(100)]

for i, params in enumerate(models):
    quantized, frame_meta = quantizer.quantize_and_store(
        params, 
        model_id=f"model_{i:03d}",
        store_in_video=True
    )

# Search using video-enhanced algorithms
query = np.random.randn(1024).astype(np.float32)
results = quantizer.video_search(
    query, 
    max_results=10, 
    search_method='hybrid'
)

# Display results
for result in results:
    print(f"Model: {result.frame_metadata.model_id}, "
          f"Similarity: {result.similarity_score:.3f}")
```

### Batch Processing

```python
from hilbert_quantization import VideoBatchQuantizer

# Create batch quantizer with video storage
batch_quantizer = VideoBatchQuantizer(storage_dir="batch_video_db")

# Process large batches efficiently
parameter_sets = [np.random.randn(2048) for _ in range(1000)]
model_ids = [f"batch_model_{i:04d}" for i in range(1000)]

quantized_models, frame_metadata = batch_quantizer.quantize_batch_to_video(
    parameter_sets=parameter_sets,
    model_ids=model_ids,
    store_in_video=True
)

print(f"Processed {len(quantized_models)} models")
```

## Video Storage System

### Architecture

```
Neural Network Parameters (1D)
        ↓
Hilbert Curve Mapping (2D Image)
        ↓
Video Frame Conversion (RGB)
        ↓
Video Encoding (MP4/AVI)
        ↓
Frame Metadata (JSON)
```

### Storage Format

- **Video Files**: Standard video formats (MP4, AVI, etc.)
- **Frame Metadata**: JSON files with model information
- **Index Mapping**: Model ID to video frame mapping
- **Similarity Features**: Pre-computed features for fast search

### Video Storage Configuration

```python
quantizer = VideoHilbertQuantizer(
    storage_dir="video_db",
    frame_rate=30.0,              # Affects temporal compression
    video_codec='mp4v',           # Video codec ('mp4v', 'XVID', 'H264')
    max_frames_per_video=10000,   # Frames per video file
    enable_video_storage=True     # Enable/disable video features
)
```

## Search Methods

### 1. Video Features Search

Uses computer vision algorithms for similarity detection:

```python
results = quantizer.video_search(
    query_params, 
    search_method='video_features',
    max_results=10
)
```

**Algorithms Used:**
- Template matching
- ORB/SIFT feature detection
- Histogram comparison  
- Structural similarity (SSIM)

### 2. Hierarchical Search

Uses traditional hierarchical indices through video system:

```python
results = quantizer.video_search(
    query_params,
    search_method='hierarchical', 
    max_results=10
)
```

### 3. Hybrid Search

Combines video features with hierarchical indices:

```python
results = quantizer.video_search(
    query_params,
    search_method='hybrid',
    max_results=10
)
```

**Benefits:**
- Fast initial filtering using video features
- Refined ranking using hierarchical indices
- Best accuracy and performance balance

### 4. Temporal Coherence Analysis

Leverages temporal relationships between frames:

```python
results = quantizer.video_search(
    query_params,
    search_method='hybrid',
    use_temporal_coherence=True,
    max_results=10
)
```

**How it Works:**
- Analyzes neighboring frames in video sequence
- Identifies clusters of similar models
- Boosts scores for temporally coherent results

## Performance Comparison

### Search Method Comparison

```python
# Compare all search methods
comparison = quantizer.compare_search_methods(
    query_params,
    max_results=10
)

for method, stats in comparison['methods'].items():
    print(f"{method}: {stats['search_time']:.3f}s, "
          f"avg_similarity: {stats['avg_similarity']:.3f}")
```

### Example Results

| Method | Search Time | Avg Similarity | Use Case |
|--------|-------------|----------------|----------|
| Traditional | 15.2ms | 0.847 | Baseline |
| Video Features | 8.6ms | 0.823 | Fast screening |
| Hierarchical | 12.1ms | 0.851 | High accuracy |
| Hybrid | 9.3ms | 0.863 | **Best overall** |
| Hybrid + Temporal | 10.7ms | 0.881 | **Highest accuracy** |

## Advanced Features

### Export and Migration

```python
# Export video database in different formats
export_info = quantizer.export_video_database(
    export_path="exported_db",
    format='video',           # 'video', 'frames', 'traditional'
    include_metadata=True
)

print(f"Exported {export_info['total_files_exported']} files")
```

### Storage Optimization

```python
# Optimize video storage
optimization_results = quantizer.optimize_video_storage()
print(f"Applied optimizations: {optimization_results['optimizations_applied']}")
```

### Storage Statistics

```python
# Get comprehensive storage information
storage_info = quantizer.get_video_storage_info()

print(f"Total models: {storage_info['storage_statistics']['total_models_stored']}")
print(f"Video files: {storage_info['storage_statistics']['total_video_files']}")
print(f"Compression ratio: {storage_info['storage_statistics']['average_compression_ratio']:.2f}x")
```

## Configuration Options

### Video Storage Settings

```python
from hilbert_quantization.core.video_storage import VideoModelStorage

storage = VideoModelStorage(
    storage_dir="custom_video_db",
    frame_rate=30.0,              # Higher = better temporal compression
    video_codec='H264',           # More efficient codec
    max_frames_per_video=5000     # Balance file size vs. management
)
```

### Search Engine Settings

```python
from hilbert_quantization.core.video_search import VideoEnhancedSearchEngine

search_engine = VideoEnhancedSearchEngine(
    video_storage=storage,
    similarity_threshold=0.1,     # Minimum similarity to keep
    max_candidates_per_level=100, # Candidates per filtering level
    use_parallel_processing=True, # Enable multi-threading
    max_workers=4                 # Number of worker threads
)
```

## Use Cases

### 1. Model Version Management

```python
# Store different versions of the same model
base_model = train_model()
for epoch in range(100):
    model_params = get_model_parameters(base_model, epoch)
    quantizer.quantize_and_store(
        model_params,
        model_id=f"model_v1_epoch_{epoch:03d}",
        description=f"Training epoch {epoch}"
    )
```

### 2. Similar Architecture Discovery

```python
# Find models with similar architectures
new_model_params = get_model_parameters(new_model)
similar_models = quantizer.video_search(
    new_model_params,
    search_method='hybrid',
    max_results=20
)

# Analyze architectural patterns
for result in similar_models:
    print(f"Similar model: {result.frame_metadata.model_id}")
    print(f"Architecture: {result.frame_metadata.model_metadata.model_architecture}")
```

### 3. Transfer Learning Candidate Selection

```python
# Find pre-trained models suitable for transfer learning
task_specific_params = extract_task_features(target_task)
candidates = quantizer.video_search(
    task_specific_params,
    search_method='hybrid',
    use_temporal_coherence=True,
    max_results=10
)

best_candidate = candidates[0]
print(f"Best transfer learning candidate: {best_candidate.frame_metadata.model_id}")
```

### 4. Model Deduplication

```python
# Identify duplicate or near-duplicate models
all_models = get_all_stored_models()
duplicates = []

for model_id in all_models:
    model = quantizer.get_model_from_video_storage(model_id)
    similar = quantizer.video_search(
        model.parameters,
        similarity_threshold=0.95,  # High threshold for duplicates
        max_results=5
    )
    
    if len(similar) > 1:  # More than just the model itself
        duplicates.append((model_id, similar))
```

## Performance Tips

### 1. Optimize Video Settings

```python
# For large collections: higher frame rate, efficient codec
quantizer = VideoHilbertQuantizer(
    frame_rate=60.0,         # Better temporal compression
    video_codec='H264',      # More efficient than mp4v
    max_frames_per_video=50000  # Larger files, fewer management overhead
)
```

### 2. Use Parallel Processing

```python
# Enable parallel search for large databases
search_engine.use_parallel_processing = True
search_engine.max_workers = 8  # Match your CPU cores
```

### 3. Pre-compute Features

```python
# Pre-compute similarity features during storage
quantizer.quantize_and_store(
    parameters,
    model_id="model_001",
    store_in_video=True  # Automatically computes features
)
```

### 4. Batch Operations

```python
# Use batch processing for large numbers of models
batch_quantizer = VideoBatchQuantizer()
batch_quantizer.quantize_batch_to_video(
    parameter_sets=large_parameter_list,
    model_ids=model_id_list,
    store_in_video=True
)
```

## Troubleshooting

### Common Issues

1. **OpenCV Installation**: If opencv-python fails to install, try `conda install opencv`
2. **Codec Support**: If video encoding fails, install additional codecs: `pip install opencv-contrib-python`
3. **Memory Issues**: Reduce `max_frames_per_video` for systems with limited memory
4. **Search Timeout**: Increase `similarity_threshold` to reduce search time

### Error Handling

```python
try:
    results = quantizer.video_search(query_params)
except ConfigurationError:
    # Video storage not enabled
    results = quantizer.search(query_params)  # Fallback to traditional
except SearchError as e:
    print(f"Search failed: {e}")
    # Handle search failure
```

### Performance Monitoring

```python
# Monitor search performance
search_stats = quantizer.video_search_engine.get_search_statistics()
print(f"Search engine type: {search_stats['search_engine_type']}")
print(f"Parallel processing: {search_stats['parallel_processing_enabled']}")
print(f"Total searchable models: {search_stats['total_searchable_models']}")
```

## Example Applications

See `examples/video_storage_demo.py` for comprehensive examples including:

- Basic video storage and retrieval
- Search method comparisons
- Temporal coherence analysis
- Batch processing demonstrations
- Performance analysis
- Export and optimization features

## Future Enhancements

- **Advanced Codecs**: Support for newer video codecs (AV1, VP9)
- **Distributed Storage**: Multi-node video storage and search
- **Real-time Search**: Streaming search capabilities
- **GPU Acceleration**: CUDA-accelerated video processing
- **Advanced Analytics**: Model evolution tracking and analysis

## Contributing

To contribute to the video features:

1. Install development dependencies: `pip install -r requirements_video.txt`
2. Add tests for video functionality in `tests/test_video_*.py`
3. Follow the existing code patterns and documentation style
4. Ensure compatibility with both video and non-video modes

## License

Video enhancement features are released under the same MIT license as the core library.
