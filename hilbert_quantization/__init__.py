"""
Hilbert Quantization - Ultra-fast similarity search with compression

A high-performance similarity search library that combines Hilbert curve mapping 
with MPEG-AI compression to deliver both speed and storage efficiency.

Key Features:
- Ultra-fast search (sub-millisecond to few-millisecond)
- 6x compression ratio
- Competitive with industry leaders (Pinecone, FAISS)
- Scalable performance on larger datasets
- Pure Python with NumPy

Example:
    >>> import numpy as np
    >>> from hilbert_quantization import HilbertQuantizer
    >>> 
    >>> # Initialize quantizer
    >>> quantizer = HilbertQuantizer()
    >>> 
    >>> # Quantize embeddings
    >>> embedding = np.random.normal(0, 1, 1024).astype(np.float32)
    >>> quantized = quantizer.quantize(embedding, "doc_1")
    >>> 
    >>> # Search
    >>> query = np.random.normal(0, 1, 1024).astype(np.float32)
    >>> results = quantizer.search(query, [quantized], max_results=5)
"""

__version__ = "1.3.0"
__author__ = "Hilbert Quantization Contributors"
__email__ = "support@example.com"
__license__ = "MIT"

# Core API imports
from .api import (
    HilbertQuantizer,
    BatchQuantizer,
    quantize_model,
    reconstruct_model,
    search_similar_models,
)

# Video-enhanced API imports
from .video_api import (
    VideoHilbertQuantizer,
    VideoBatchQuantizer,
    create_video_quantizer,
    quantize_model_to_video,
    video_search_similar_models,
)

# Streaming index generator for memory-efficient processing
from .core.streaming_index_builder import StreamingHilbertIndexGenerator

# Hugging Face integration (optional import)
try:
    from .huggingface_integration import (
        HuggingFaceParameterExtractor,
        HuggingFaceModelMetadata,
        ParameterExtractionResult,
        extract_huggingface_parameters,
        get_huggingface_model_info,
    )
    _HUGGINGFACE_AVAILABLE = True
except ImportError:
    _HUGGINGFACE_AVAILABLE = False

# Configuration imports
from .config import (
    SystemConfig,
    CompressionConfig,
    QuantizationConfig,
    SearchConfig,
    create_default_config,
)

# Model imports
from .models import (
    QuantizedModel,
    SearchResult,
    CompressionMetrics,
    ModelMetadata,
)

# Exception imports
from .exceptions import (
    HilbertQuantizationError,
    QuantizationError,
    CompressionError,
    SearchError,
    ReconstructionError,
    ConfigurationError,
    ValidationError,
)

# RAG system imports
from . import rag

# Optimized components (optional import)
try:
    from .optimized import (
        CacheOptimizedDatabase,
        CacheOptimizedSearch,
        UltraFastHierarchicalSearch,
    )
    _OPTIMIZED_AVAILABLE = True
except ImportError:
    _OPTIMIZED_AVAILABLE = False

# Version info
VERSION_INFO = tuple(map(int, __version__.split('.')))

# Public API
__all__ = [
    # Version
    "__version__",
    "VERSION_INFO",
    
    # Core API
    "HilbertQuantizer",
    "BatchQuantizer",
    "quantize_model",
    "reconstruct_model", 
    "search_similar_models",
    
    # Video-enhanced API
    "VideoHilbertQuantizer",
    "VideoBatchQuantizer", 
    "create_video_quantizer",
    "quantize_model_to_video",
    "video_search_similar_models",
    
    # Streaming generators
    "StreamingHilbertIndexGenerator",
    
    # RAG system
    "rag",
    
    # Configuration
    "SystemConfig",
    "CompressionConfig",
    "QuantizationConfig", 
    "SearchConfig",
    "create_default_config",
    
    # Models
    "QuantizedModel",
    "SearchResult",
    "CompressionMetrics",
    "ModelMetadata",
    
    # Exceptions
    "HilbertQuantizationError",
    "QuantizationError",
    "CompressionError",
    "SearchError", 
    "ReconstructionError",
    "ConfigurationError",
    "ValidationError",
]

# Add optimized components if available
if _OPTIMIZED_AVAILABLE:
    __all__.extend([
        "CacheOptimizedDatabase",
        "CacheOptimizedSearch", 
        "UltraFastHierarchicalSearch",
    ])

# Add Hugging Face components if available
if _HUGGINGFACE_AVAILABLE:
    __all__.extend([
        "HuggingFaceParameterExtractor",
        "HuggingFaceModelMetadata",
        "ParameterExtractionResult",
        "extract_huggingface_parameters",
        "get_huggingface_model_info",
    ])


def get_version() -> str:
    """Get the current version string."""
    return __version__


def get_version_info() -> tuple:
    """Get the current version as a tuple of integers."""
    return VERSION_INFO


def is_optimized_available() -> bool:
    """Check if optimized components are available."""
    return _OPTIMIZED_AVAILABLE


def is_huggingface_available() -> bool:
    """Check if Hugging Face integration is available."""
    return _HUGGINGFACE_AVAILABLE


# Package-level configuration
import logging

# Set up default logging (can be overridden by users)
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Performance warning for missing optimized components
if not _OPTIMIZED_AVAILABLE:
    import warnings
    warnings.warn(
        "Optimized components not available. "
        "Performance may be reduced. "
        "Consider installing with: pip install hilbert-quantization[gpu]",
        ImportWarning,
        stacklevel=2
    )