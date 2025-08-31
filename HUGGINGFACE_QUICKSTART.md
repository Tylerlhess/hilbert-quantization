# 🤗 HuggingFace Quick Start Guide

Get started with Hilbert Quantization and HuggingFace models in minutes!

## 🚀 Installation

```bash
# Install with HuggingFace support
pip install hilbert-quantization[huggingface]

# Or install from source with all features
git clone https://github.com/Tylerlhess/hilbert-quantization.git
cd hilbert-quantization
pip install -e ".[dev,huggingface]"
```

## ⚡ Quick Examples

### 1. Basic Model Encoding

```python
from hilbert_quantization.video_api import VideoHilbertQuantizer
from transformers import AutoModel
import torch

# Initialize quantizer
quantizer = VideoHilbertQuantizer()

# Load and encode a model
model = AutoModel.from_pretrained("bert-base-uncased")
video_path = quantizer.encode_model_to_video(
    model.state_dict(),
    model_name="bert-base-uncased"
)
print(f"Model encoded to: {video_path}")
```

### 2. Streaming Large Models

```python
from hilbert_quantization.video_api import VideoHilbertQuantizer
from transformers import AutoModel
import torch

def stream_model_parameters(model_name: str):
    """Stream model parameters without loading entire model into memory."""
    quantizer = VideoHilbertQuantizer()
    model = AutoModel.from_pretrained(model_name)
    
    encoded_layers = []
    for name, param in model.named_parameters():
        # Process each layer individually
        layer_data = param.detach().numpy()
        quantized = quantizer.quantize(layer_data, model_id=f"{model_name}_{name}")
        encoded_layers.append((name, quantized))
        print(f"✅ Encoded {name}: {param.shape}")
    
    return encoded_layers

# Stream a large model
layers = stream_model_parameters("gpt2")
print(f"Encoded {len(layers)} layers")
```

### 3. Model Similarity Search

```python
from hilbert_quantization.video_api import VideoHilbertQuantizer
from transformers import AutoModel

# Initialize quantizer
quantizer = VideoHilbertQuantizer()

# Encode multiple models
models = ["bert-base-uncased", "distilbert-base-uncased", "roberta-base"]
encoded_models = []

for model_name in models:
    model = AutoModel.from_pretrained(model_name)
    # Extract embeddings layer as representative parameters
    embeddings = model.embeddings.word_embeddings.weight.detach().numpy()
    
    quantized = quantizer.quantize(embeddings, model_id=model_name)
    encoded_models.append(quantized)
    print(f"✅ Encoded {model_name}")

# Search for similar models
query_model = AutoModel.from_pretrained("bert-large-uncased")
query_embeddings = query_model.embeddings.word_embeddings.weight.detach().numpy()

results = quantizer.search(query_embeddings, encoded_models, max_results=3)
print("\n🔍 Most similar models:")
for result in results:
    print(f"  {result.model.model_id}: {result.similarity_score:.3f}")
```

## 🛠️ Command Line Tools

### Encode Models to Video

```bash
# Encode popular models
python examples/huggingface_video_encoder.py --models bert-base-uncased gpt2 distilbert-base-uncased

# Download and encode popular models automatically
python examples/huggingface_video_encoder.py --download-popular

# Search for similar models
python examples/huggingface_video_encoder.py --search-similar "bert-base-uncased"
```

### Streaming Encoder

```bash
# Stream a model with default settings
python examples/streaming_huggingface_encoder.py --model bert-base-uncased --stream

# Stream with custom chunk size
python examples/streaming_huggingface_encoder.py --model gpt2 --stream --chunk-size 2048

# Stream specific layers only
python examples/streaming_huggingface_encoder.py --model t5-large --stream --layers attention

# Benchmark streaming performance
python examples/streaming_huggingface_encoder.py --model bert-base-uncased --benchmark
```

## 📊 Performance Examples

### Memory Usage Comparison

```python
import psutil
import os
from transformers import AutoModel
from hilbert_quantization.video_api import VideoHilbertQuantizer

def measure_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

# Traditional approach (loads entire model)
print("Traditional approach:")
start_memory = measure_memory_usage()
model = AutoModel.from_pretrained("gpt2")
peak_memory = measure_memory_usage()
print(f"Memory usage: {peak_memory - start_memory:.1f} MB")

# Streaming approach
print("\nStreaming approach:")
quantizer = VideoHilbertQuantizer()
start_memory = measure_memory_usage()

# Process model layer by layer
model = AutoModel.from_pretrained("gpt2")
for name, param in model.named_parameters():
    quantized = quantizer.quantize(param.detach().numpy(), model_id=f"gpt2_{name}")
    # Memory stays constant as we process each layer individually

peak_memory = measure_memory_usage()
print(f"Memory usage: {peak_memory - start_memory:.1f} MB")
```

### Compression Ratios

```python
from hilbert_quantization.video_api import VideoHilbertQuantizer
from transformers import AutoModel
import torch

quantizer = VideoHilbertQuantizer()

models = ["distilbert-base-uncased", "bert-base-uncased", "bert-large-uncased"]
for model_name in models:
    model = AutoModel.from_pretrained(model_name)
    
    # Get total parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    # Encode embeddings layer
    embeddings = model.embeddings.word_embeddings.weight.detach().numpy()
    quantized = quantizer.quantize(embeddings, model_id=model_name)
    
    print(f"{model_name}:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Compression ratio: {quantized.metadata.compression_ratio:.2f}x")
    print(f"  Storage savings: {(1 - 1/quantized.metadata.compression_ratio)*100:.1f}%")
```

## 🔧 Advanced Configuration

### Custom Streaming Configuration

```python
from hilbert_quantization import QuantizationConfig, VideoHilbertQuantizer

# Configure for maximum compression
config = QuantizationConfig(
    use_streaming_optimization=True,
    enable_integrated_mapping=True,
    memory_efficient_mode=True,
    max_hierarchical_levels=10  # More levels = better compression
)

quantizer = VideoHilbertQuantizer(config=config)
```

### Batch Processing

```python
from hilbert_quantization.video_api import VideoBatchQuantizer
from transformers import AutoModel

# Process multiple models in batch
batch_quantizer = VideoBatchQuantizer()

model_names = ["bert-base-uncased", "gpt2", "distilbert-base-uncased"]
models_data = []

for name in model_names:
    model = AutoModel.from_pretrained(name)
    embeddings = model.embeddings.word_embeddings.weight.detach().numpy()
    models_data.append(embeddings)

# Batch quantize all models
quantized_models = batch_quantizer.quantize_batch(
    models_data,
    model_ids=model_names,
    descriptions=[f"Embeddings from {name}" for name in model_names]
)

print(f"Batch processed {len(quantized_models)} models")
```

## 🚨 Common Issues & Solutions

### Issue: Out of Memory with Large Models

**Solution**: Use streaming approach
```python
# Instead of loading entire model
model = AutoModel.from_pretrained("gpt2-xl")  # May cause OOM

# Stream layer by layer
for name, param in model.named_parameters():
    quantized = quantizer.quantize(param.detach().numpy(), model_id=f"gpt2-xl_{name}")
```

### Issue: Slow Processing

**Solution**: Enable streaming optimization
```python
from hilbert_quantization import QuantizationConfig

config = QuantizationConfig(use_streaming_optimization=True)
quantizer = VideoHilbertQuantizer(config=config)
```

### Issue: Low Compression Ratios

**Solution**: Adjust quality settings
```python
from hilbert_quantization import CompressionConfig

config = CompressionConfig(
    quality=0.7,  # Lower = more compression
    preserve_index_row=False  # Disable for maximum compression
)
quantizer = VideoHilbertQuantizer(compression_config=config)
```

## 📚 Next Steps

- [Complete Usage Guide](docs/guides/COMPLETE_USAGE_GUIDE.md)
- [Streaming Guide](docs/guides/STREAMING_GUIDE.md)
- [Video Features Guide](docs/guides/VIDEO_FEATURES_README.md)
- [API Reference](docs/API_GUIDE.md)

## 🤝 Community

- [GitHub Issues](https://github.com/Tylerlhess/hilbert-quantization/issues)
- [Discussions](https://github.com/Tylerlhess/hilbert-quantization/discussions)
- Email: tylerlhess@gmail.com

Happy encoding! 🎉
