# 🚀 Quick Start Guide: Hilbert Quantization

## Installation

```bash
# Install required dependencies
pip install numpy pillow

# The hilbert_quantization package is already in your project
```

## Basic Usage

### 1. Quantize a Model

```python
from hilbert_quantization.core.pipeline import QuantizationPipeline
import numpy as np

# Initialize quantizer
quantizer = QuantizationPipeline()

# Load your model parameters (replace with your actual loading code)
# For PyTorch: parameters = torch.cat([p.flatten() for p in model.parameters()]).numpy()
# For TensorFlow: parameters = np.concatenate([w.flatten() for w in model.get_weights()])
parameters = np.random.normal(0, 0.02, 1_000_000).astype(np.float32)  # Example

# Quantize the model
quantized_model = quantizer.quantize_model(
    parameters=parameters,
    model_name="my_model",
    compression_quality=0.85  # 0.0 = max compression, 1.0 = max quality
)

print(f"Compression ratio: {quantized_model.metadata.compression_ratio:.1f}x")
```

### 2. Save and Load Quantized Models

```python
import pickle

# Save quantized model
with open("my_model.hqm", "wb") as f:
    pickle.dump(quantized_model, f)

# Load quantized model
with open("my_model.hqm", "rb") as f:
    loaded_model = pickle.load(f)
```

### 3. Search for Similar Models

```python
from hilbert_quantization.core.search_engine import ProgressiveSimilaritySearchEngine

# Initialize search engine
search_engine = ProgressiveSimilaritySearchEngine()

# Create a database of quantized models
model_database = [quantized_model1, quantized_model2, quantized_model3]

# Search for similar models
query_model = quantizer.quantize_model(query_parameters, "query")
results = search_engine.progressive_search(
    query_model.hierarchical_indices, 
    model_database, 
    max_results=5
)

# Results are SearchResult objects
for search_result in results:
    model = search_result.model
    similarity = search_result.similarity_score
    print(f"Model: {model.metadata.model_name}, Similarity: {similarity:.3f}")
```

### 4. Reconstruct Parameters for Inference

```python
from hilbert_quantization.core.pipeline import ReconstructionPipeline

# Initialize reconstructor
reconstructor = ReconstructionPipeline()

# Reconstruct original parameters
reconstructed_params = reconstructor.reconstruct_with_validation(quantized_model)[0]

# Use reconstructed parameters in your model
# your_model.load_parameters(reconstructed_params)
# output = your_model(input_data)
```

## Complete Example

```python
from hilbert_quantization.core.pipeline import QuantizationPipeline, ReconstructionPipeline
from hilbert_quantization.core.search_engine import ProgressiveSimilaritySearchEngine
import numpy as np
import pickle

# 1. Initialize components
quantizer = QuantizationPipeline()
reconstructor = ReconstructionPipeline()
search_engine = ProgressiveSimilaritySearchEngine()

# 2. Quantize multiple models
models = []
for i in range(3):
    # Load your actual model parameters here
    params = np.random.normal(0, 0.02, 1_000_000).astype(np.float32)
    
    quantized = quantizer.quantize_model(params, f"model_{i}")
    models.append(quantized)
    
    # Save each model
    with open(f"model_{i}.hqm", "wb") as f:
        pickle.dump(quantized, f)

# 3. Search for similar models
query_params = np.random.normal(0, 0.02, 1_000_000).astype(np.float32)
query_model = quantizer.quantize_model(query_params, "query")

results = search_engine.progressive_search(query_model.hierarchical_indices, models, max_results=2)

# 4. Use the best match for inference
best_match = results[0].model  # Most similar model
reconstructed = reconstructor.reconstruct_with_validation(best_match)[0]

print(f"Using model: {best_match.metadata.model_name}")
print(f"Reconstructed {len(reconstructed):,} parameters")
```

## Key Features

### ✅ **Real Model Quantization**
- Compresses actual model parameters using Hilbert curves + MPEG-AI
- Maintains model quality while reducing size significantly
- Works with any model architecture (transformers, CNNs, etc.)

### ✅ **Embedded Search Indices**
- No external database needed - indices stored in the compressed model
- Fast similarity search (milliseconds vs seconds)
- Spatial locality preservation for accurate matching

### ✅ **Quality Preservation**
- Configurable compression quality (0.0 to 1.0)
- Validation metrics to ensure reconstruction accuracy
- Minimal impact on model performance

### ✅ **Easy Integration**
- Simple API that works with existing model formats
- Compatible with PyTorch, TensorFlow, JAX, etc.
- Minimal dependencies (just numpy and PIL)

## Configuration Options

```python
from hilbert_quantization.config import CompressionConfig, QuantizationConfig
from hilbert_quantization.core.dimension_calculator import PowerOf4DimensionCalculator

# For models with non-optimal parameter counts, use relaxed efficiency
quantization_config = QuantizationConfig(min_efficiency_ratio=0.25)
compression_config = CompressionConfig(
    quality=0.85,                    # Compression quality (0.0-1.0)
    preserve_index_row=True,         # Enable fast search
    validate_reconstruction=True,    # Validate reconstruction quality
    max_reconstruction_error=1e-3    # Maximum acceptable error
)

quantizer = QuantizationPipeline(
    dimension_calculator=PowerOf4DimensionCalculator(min_efficiency_ratio=0.25),
    compression_config=compression_config
)
```

## File Formats

- **`.hqm`** - Hilbert Quantized Model (pickled QuantizedModel object)
- Contains compressed parameters + embedded search indices
- Single file contains everything needed for search and reconstruction

## Performance Tips

1. **Quality Setting**: Use 0.8-0.9 for production, 0.5-0.7 for maximum compression
2. **Model Size**: Works best with 1M+ parameters (smaller models have less compression benefit)
3. **Parameter Count**: For best efficiency, use parameter counts close to powers of 4 (4, 16, 64, 256, 1024, 4096, etc.)
4. **Search Database**: Keep 100-1000 models for optimal search performance
5. **Memory**: Reconstruction uses ~2x model size in memory temporarily
6. **Efficiency**: If you get efficiency ratio errors, use `min_efficiency_ratio=0.25` in QuantizationConfig

## Next Steps

- Run `python simple_usage_example.py` for a complete walkthrough
- Run `python real_model_quantization_test.py` to compare with other methods
- Run `python generative_search_test.py` for generative AI applications

## Support

The system supports:
- **Model Types**: Transformers, CNNs, RNNs, Diffusion models, GANs, etc.
- **Frameworks**: PyTorch, TensorFlow, JAX, Flax, etc.
- **Sizes**: 1M to 100B+ parameters
- **Use Cases**: Model compression, similarity search, model recommendation, inference optimization