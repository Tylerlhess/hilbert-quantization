"""
Core data models for the Hilbert quantization system.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


@dataclass
class ModelMetadata:
    """Metadata for quantized models."""
    model_name: str
    original_size_bytes: int
    compressed_size_bytes: int
    compression_ratio: float
    quantization_timestamp: str
    model_architecture: Optional[str] = None
    additional_info: Optional[Dict[str, Any]] = None


@dataclass
class PaddingConfig:
    """Configuration for parameter padding strategy."""
    target_dimensions: Tuple[int, int]
    padding_value: float
    padding_positions: List[Tuple[int, int]]
    efficiency_ratio: float
    
    def __post_init__(self):
        """Validate padding configuration."""
        if self.efficiency_ratio < 0 or self.efficiency_ratio > 1:
            raise ValueError("Efficiency ratio must be between 0 and 1")
        if len(self.target_dimensions) != 2:
            raise ValueError("Target dimensions must be a 2-tuple")


@dataclass
class SearchResult:
    """Result from similarity search operations."""
    model: 'QuantizedModel'
    similarity_score: float
    matching_indices: Dict[int, float]  # level -> similarity
    reconstruction_error: float
    
    def __post_init__(self):
        """Validate search result."""
        if self.similarity_score < 0 or self.similarity_score > 1:
            raise ValueError("Similarity score must be between 0 and 1")
        if self.reconstruction_error < 0:
            raise ValueError("Reconstruction error must be non-negative")


@dataclass
class QuantizedModel:
    """Complete quantized model representation."""
    compressed_data: bytes
    original_dimensions: Tuple[int, int]
    parameter_count: int
    compression_quality: float
    hierarchical_indices: np.ndarray
    metadata: ModelMetadata
    
    def __post_init__(self):
        """Validate quantized model data."""
        if self.parameter_count <= 0:
            raise ValueError("Parameter count must be positive")
        if self.compression_quality < 0 or self.compression_quality > 1:
            raise ValueError("Compression quality must be between 0 and 1")
        if len(self.original_dimensions) != 2:
            raise ValueError("Original dimensions must be a 2-tuple")
        if self.hierarchical_indices.ndim != 1:
            raise ValueError("Hierarchical indices must be 1-dimensional")


@dataclass
class CompressionMetrics:
    """Metrics for compression performance evaluation."""
    compression_ratio: float
    reconstruction_error: float
    compression_time_seconds: float
    decompression_time_seconds: float
    memory_usage_mb: float
    original_size_bytes: int = 0
    compressed_size_bytes: int = 0
    
    def __post_init__(self):
        """Validate compression metrics."""
        if self.compression_ratio <= 0:
            raise ValueError("Compression ratio must be positive")
        if self.reconstruction_error < 0:
            raise ValueError("Reconstruction error must be non-negative")


@dataclass
class SearchMetrics:
    """Metrics for search performance evaluation."""
    search_time_seconds: float
    candidates_filtered: int
    final_candidates: int
    filtering_efficiency: float
    accuracy_score: float
    
    def __post_init__(self):
        """Validate search metrics."""
        if self.search_time_seconds < 0:
            raise ValueError("Search time must be non-negative")
        if self.candidates_filtered < 0 or self.final_candidates < 0:
            raise ValueError("Candidate counts must be non-negative")
        if self.final_candidates > self.candidates_filtered:
            raise ValueError("Final candidates cannot exceed filtered candidates")