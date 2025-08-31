# Hugging Face Model Video Encoding Examples

This directory contains comprehensive examples demonstrating how to encode Hugging Face models into video format for efficient storage and similarity search using the Hilbert Quantization system.

## Overview

The Hugging Face integration allows you to:
- Extract parameters from any Hugging Face model
- Encode models into compressed video format with hierarchical indices
- Perform fast similarity search across different model architectures
- Compare search method performance and accuracy
- Manage model registries with encoding statistics

## Example Scripts

### 1. `huggingface_video_encoder.py` - Basic Introduction
**Purpose**: Introduction to Hugging Face model encoding with basic examples.

**Features**:
- Model information extraction without downloading full models
- Parameter extraction with stratified sampling
- Video encoding with compression
- Model registry management
- Layer-specific parameter filtering

**Usage**:
```bash
python examples/huggingface_video_encoder.py
```

**Best for**: First-time users wanting to understand the basic workflow.

### 2. `huggingface_model_encoding_examples.py` - Comprehensive Workflow
**Purpose**: Complete demonstration of the entire encoding and search workflow.

**Features**:
- Batch encoding of multiple popular models
- Cross-architecture similarity analysis
- Search method performance comparison
- Advanced model filtering and registry management
- Comprehensive result export and analysis

**Usage**:
```bash
python examples/huggingface_model_encoding_examples.py
```

**Best for**: Users wanting to see the complete system capabilities and workflow.

### 3. `model_similarity_search_demo.py` - Advanced Similarity Search
**Purpose**: Focused demonstration of similarity search capabilities across model architectures.

**Features**:
- Within-family similarity analysis (e.g., BERT variants)
- Cross-architecture similarity detection
- Detailed search method comparison
- Similarity score interpretation guide
- Family cohesion metrics

**Usage**:
```bash
python examples/model_similarity_search_demo.py
```

**Best for**: Users interested in understanding model relationships and similarity patterns.

### 4. `search_performance_comparison.py` - Performance Analysis
**Purpose**: Comprehensive performance benchmarking of different search methods.

**Features**:
- Scalability analysis with different collection sizes
- Memory usage and computational efficiency analysis
- Speed vs accuracy trade-off analysis
- Performance visualization and reporting
- Method recommendation system

**Usage**:
```bash
python examples/search_performance_comparison.py
```

**Best for**: Users optimizing search performance for production systems.

## Search Methods Explained

### Hierarchical Index Search
- **Speed**: Fastest
- **Accuracy**: Good for similar architectures
- **Use Case**: Initial filtering and fast candidate selection
- **How it works**: Uses precomputed spatial averages at multiple granularities

### Video Features Search
- **Speed**: Slower but thorough
- **Accuracy**: Excellent for visual pattern matching
- **Use Case**: Detailed similarity analysis
- **How it works**: Computer vision algorithms (ORB, template matching, histograms)

### Hybrid Search
- **Speed**: Balanced
- **Accuracy**: Best overall
- **Use Case**: Production systems requiring both speed and accuracy
- **How it works**: Weighted combination of hierarchical (40%) and video features (60%)

## Model Families Tested

The examples work with various model architectures:

### BERT Family
- `bert-base-uncased`
- `distilbert-base-uncased`
- `bert-base-cased`
- `bert-large-uncased`

### RoBERTa Family
- `roberta-base`
- `distilroberta-base`
- `roberta-large`

### GPT Family
- `gpt2`
- `gpt2-medium`
- `microsoft/DialoGPT-small`

### Other Architectures
- `albert-base-v2`
- `electra-small-discriminator`
- `google/mobilebert-uncased`

## Configuration Options

### Parameter Limits
- **Small models**: 10K-30K parameters (fast encoding, good for testing)
- **Medium models**: 30K-100K parameters (balanced performance)
- **Large models**: 100K+ parameters (comprehensive analysis, slower)

### Compression Quality
- **0.6**: High compression, lower quality (faster search)
- **0.8**: Balanced compression and quality (recommended)
- **0.9**: Lower compression, higher quality (slower but more accurate)

### Sampling Strategies
- **Stratified sampling**: Maintains layer proportions (recommended)
- **Random sampling**: Faster but may miss important parameters
- **Layer filtering**: Focus on specific layer types (embeddings, attention, MLP)

## Output Files and Results

### Registry Files
- `model_registry.json`: Complete model database with metadata
- `similarity_registry.json`: Specialized registry for similarity analysis
- `performance_registry.json`: Registry with performance benchmarks

### Analysis Results
- `encoding_results.json`: Detailed encoding statistics
- `search_results.json`: Similarity search results
- `performance_results.json`: Performance benchmarking data
- `similarity_analysis.json`: Cross-architecture analysis

### Reports
- `summary_report.md`: Executive summary of encoding results
- `similarity_report.md`: Detailed similarity analysis report
- `performance_report.md`: Performance comparison report

### Visualizations
- `performance_analysis.png`: Performance comparison charts
- Various method comparison plots and scalability graphs

## Performance Expectations

### Encoding Performance
- **Small models** (< 30K params): 2-5 seconds per model
- **Medium models** (30K-100K params): 5-15 seconds per model
- **Large models** (> 100K params): 15-60 seconds per model

### Search Performance
- **Hierarchical search**: < 0.1 seconds for 100 models
- **Video features search**: 0.5-2 seconds for 100 models
- **Hybrid search**: 0.2-1 seconds for 100 models

### Memory Usage
- **Encoding**: 200-500 MB per model during processing
- **Search**: 50-200 MB for registry and indices
- **Storage**: 1-10 MB per encoded model (depending on compression)

## Troubleshooting

### Common Issues

1. **Transformers not available**
   ```bash
   pip install transformers torch
   ```

2. **Out of memory during encoding**
   - Reduce `max_params` parameter
   - Use stratified sampling
   - Process models one at a time

3. **Slow search performance**
   - Use hierarchical search for initial filtering
   - Enable caching for repeated queries
   - Consider parallel processing for large collections

4. **Low similarity scores**
   - Check if models are from similar architectures
   - Verify parameter extraction settings
   - Consider using hybrid search method

### Performance Optimization

1. **For faster encoding**:
   - Reduce parameter limits
   - Use lower compression quality
   - Enable stratified sampling

2. **For better accuracy**:
   - Increase parameter limits
   - Use higher compression quality
   - Include all layer types

3. **For production use**:
   - Use hybrid search method
   - Enable result caching
   - Implement parallel processing

## Integration Examples

### Basic Integration
```python
from hilbert_quantization.huggingface_integration import HuggingFaceVideoEncoder

# Initialize encoder
encoder = HuggingFaceVideoEncoder()

# Encode a model
result = encoder.encode_model_to_video(
    model_name="bert-base-uncased",
    max_params=50000
)

# Search for similar models
similar = encoder.search_similar_models(
    query_model="bert-base-uncased",
    max_results=5,
    search_method="hybrid"
)
```

### Advanced Integration
```python
from hilbert_quantization.huggingface_integration import HuggingFaceVideoEncoder
from hilbert_quantization.model_registry import ModelRegistry

# Initialize with custom settings
encoder = HuggingFaceVideoEncoder(
    registry_path="custom_registry.json",
    video_storage_path="custom_storage"
)

# Batch encode multiple models
models = ["bert-base-uncased", "roberta-base", "gpt2"]
for model in models:
    encoder.encode_model_to_video(
        model_name=model,
        max_params=75000,
        compression_quality=0.8,
        stratified_sampling=True
    )

# Advanced similarity search
results = encoder.search_similar_models(
    query_model="bert-base-uncased",
    max_results=10,
    similarity_threshold=0.3,
    search_method="hybrid",
    architecture_filter="BertModel"
)
```

## Next Steps

1. **Start with the basic example** (`huggingface_video_encoder.py`) to understand the workflow
2. **Run comprehensive examples** (`huggingface_model_encoding_examples.py`) to see full capabilities
3. **Explore similarity search** (`model_similarity_search_demo.py`) for advanced analysis
4. **Benchmark performance** (`search_performance_comparison.py`) for optimization
5. **Integrate into your projects** using the provided code patterns

## Support and Documentation

- See the main README.md for system requirements and installation
- Check the API documentation in `docs/API_GUIDE.md`
- Review the complete usage guide in `docs/guides/HUGGINGFACE_GUIDE.md`
- For issues, check the troubleshooting section above or create an issue

## Contributing

When adding new Hugging Face examples:
1. Follow the existing naming convention
2. Include comprehensive error handling
3. Add progress indicators for long-running operations
4. Document all configuration options
5. Provide clear usage instructions and expected outputs