"""
Custom exceptions for the Hilbert quantization system.
"""


class HilbertQuantizationError(Exception):
    """Base exception for Hilbert quantization system."""
    pass


class DimensionCalculationError(HilbertQuantizationError):
    """Raised when dimension calculation fails."""
    pass


class HilbertMappingError(HilbertQuantizationError):
    """Raised when Hilbert curve mapping fails."""
    pass


class IndexGenerationError(HilbertQuantizationError):
    """Raised when hierarchical index generation fails."""
    pass


class CompressionError(HilbertQuantizationError):
    """Raised when MPEG-AI compression/decompression fails."""
    pass


class SearchError(HilbertQuantizationError):
    """Raised when similarity search operations fail."""
    pass


class ValidationError(HilbertQuantizationError):
    """Raised when validation checks fail."""
    pass


class ConfigurationError(HilbertQuantizationError):
    """Raised when configuration is invalid."""
    pass


class QuantizationError(HilbertQuantizationError):
    """Raised when quantization process fails."""
    pass


class ReconstructionError(HilbertQuantizationError):
    """Raised when reconstruction process fails."""
    pass


class GeneratorTreeError(HilbertQuantizationError):
    """Raised when generator tree operations fail."""
    pass


class GeneratorDepthLimitError(GeneratorTreeError):
    """Raised when generator tree exceeds maximum depth limit."""
    pass


class GeneratorMemoryError(GeneratorTreeError):
    """Raised when generator tree operations exceed memory constraints."""
    pass


class GeneratorStateError(GeneratorTreeError):
    """Raised when generator state is inconsistent or invalid."""
    pass


class GeneratorOptimizationError(HilbertQuantizationError):
    """Raised when generator-based optimization fails."""
    pass