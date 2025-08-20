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

__version__ = "1.0.0"
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


def get_version() -> str:
    """Get the current version string."""
    return __version__


def get_version_info() -> tuple:
    """Get the current version as a tuple of integers."""
    return VERSION_INFO


def is_optimized_available() -> bool:
    """Check if optimized components are available."""
    return _OPTIMIZED_AVAILABLE


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