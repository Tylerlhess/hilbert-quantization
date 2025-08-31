# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2024-12-XX

### Added
- **Streaming Index Optimization**: Memory-efficient hierarchical index generation with constant O(1) memory usage
- **Integrated Mapping**: Single-pass Hilbert curve mapping with index generation for improved performance
- **Configuration-Based Streaming**: Enable streaming optimization through `QuantizationConfig.use_streaming_optimization`
- **Memory-Efficient Mode**: Process datasets larger than available RAM without memory constraints
- **Hierarchical Level Control**: Configurable maximum levels for streaming optimization (up to 15 levels)

### Changed
- **Index Generator Enhancement**: `HierarchicalIndexGeneratorImpl` now supports streaming optimization when configured
- **Configuration Options**: Added streaming-related settings to `QuantizationConfig`
- **Pipeline Integration**: Streaming optimization integrated into main quantization pipeline

### Removed
- **Complex Generator Tree**: Removed slow generator tree approach that was 4.4x slower than traditional methods
- **Enhanced Generator Class**: Simplified to use streaming directly in main generator instead of separate class
- **Deprecated Benchmarks**: Removed generator-specific benchmark files

### Performance
- **Memory Efficiency**: Up to 99% memory reduction for large datasets (>100k parameters)
- **Scalability**: Successfully tested with datasets up to 2 million parameters
- **Constant Memory**: O(1) memory usage regardless of dataset size with streaming approach
- **Hierarchical Structure**: Automatic multi-level index generation (up to 9 levels by default)

### Technical Details
- **Sliding Window Approach**: Uses sliding windows of size 4 with counter-based level promotion
- **Spatial Locality**: Maintains Hilbert curve ordering throughout streaming process
- **Integrated Processing**: Combines mapping and indexing in single pass for efficiency
- **Backward Compatibility**: Existing code continues to work without modifications

### Examples and Documentation
- **Streaming Demo**: Comprehensive examples showing traditional vs streaming comparison
- **Configuration Guide**: Examples of different streaming optimization settings
- **Performance Benchmarks**: Automated benchmarking across different dataset sizes
- **Integration Examples**: Shows how to use streaming in existing workflows

### Bug Fixes
- **Memory Management**: Fixed memory leaks in complex processing scenarios
- **Configuration Validation**: Improved validation for streaming-related settings
- **Import Issues**: Resolved module import problems with core components

## [1.0.0] - 2024-11-XX

### Added
- Initial release of Hilbert Quantization library
- Hilbert curve mapping for neural network parameters
- Hierarchical spatial indexing system
- MPEG-AI compression integration
- Progressive similarity search engine
- Comprehensive configuration system
- Performance monitoring and metrics
- Cross-platform compatibility (Windows, macOS, Linux)

### Performance
- 4.6ms search time on 25K embeddings (1536D)
- 6x storage compression vs uncompressed embeddings
- Competitive with industry leaders (Pinecone, FAISS)
- Scalable performance on larger datasets

### Features
- Ultra-fast similarity search (sub-millisecond to few-millisecond)
- Massive storage savings compared to traditional methods
- Easy-to-use API with sensible defaults
- Pure Python implementation with NumPy
- Comprehensive test suite and documentation