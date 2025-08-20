"""
Core interfaces for the Hilbert quantization system components.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
import numpy as np

from .models import QuantizedModel, PaddingConfig, SearchResult, CompressionMetrics


class DimensionCalculator(ABC):
    """Interface for calculating optimal dimensions for parameter mapping."""
    
    @abstractmethod
    def calculate_optimal_dimensions(self, param_count: int) -> Tuple[int, int]:
        """
        Calculate nearest power-of-4 dimensions that accommodate parameters.
        
        Args:
            param_count: Number of parameters to accommodate
            
        Returns:
            Tuple of (width, height) dimensions
        """
        pass
    
    @abstractmethod
    def calculate_padding_strategy(self, param_count: int, target_dims: Tuple[int, int]) -> PaddingConfig:
        """
        Determine optimal padding to minimize wasted space.
        
        Args:
            param_count: Number of parameters
            target_dims: Target dimensions for mapping
            
        Returns:
            PaddingConfig with padding strategy details
        """
        pass


class HilbertCurveMapper(ABC):
    """Interface for Hilbert curve parameter mapping operations."""
    
    @abstractmethod
    def map_to_2d(self, parameters: np.ndarray, dimensions: Tuple[int, int]) -> np.ndarray:
        """
        Map 1D parameters to 2D using Hilbert curve ordering.
        
        Args:
            parameters: 1D array of parameters
            dimensions: Target 2D dimensions
            
        Returns:
            2D array representation
        """
        pass
    
    @abstractmethod
    def map_from_2d(self, image: np.ndarray) -> np.ndarray:
        """
        Reconstruct 1D parameters from 2D Hilbert curve representation.
        
        Args:
            image: 2D array representation
            
        Returns:
            1D parameter array
        """
        pass
    
    @abstractmethod
    def generate_hilbert_coordinates(self, n: int) -> List[Tuple[int, int]]:
        """
        Generate Hilbert curve coordinate sequence for n×n grid.
        
        Args:
            n: Grid size (must be power of 2)
            
        Returns:
            List of (x, y) coordinates in Hilbert order
        """
        pass


class HierarchicalIndexGenerator(ABC):
    """Interface for generating hierarchical spatial indices."""
    
    @abstractmethod
    def generate_optimized_indices(self, image: np.ndarray, index_space_size: int) -> np.ndarray:
        """
        Generate hierarchical spatial indices using optimized space allocation.
        
        Args:
            image: 2D parameter representation
            index_space_size: Available space for indices
            
        Returns:
            1D array of hierarchical indices
        """
        pass
    
    @abstractmethod
    def calculate_level_allocation(self, total_space: int) -> List[Tuple[int, int]]:
        """
        Calculate optimal space allocation for each granularity level.
        
        Args:
            total_space: Total available index space
            
        Returns:
            List of (level, space_allocated) tuples
        """
        pass
    
    @abstractmethod
    def calculate_spatial_averages(self, image: np.ndarray, grid_size: int) -> List[float]:
        """
        Calculate averages for spatial sections at given grid size.
        
        Args:
            image: 2D parameter representation
            grid_size: Size of grid sections
            
        Returns:
            List of spatial averages
        """
        pass
    
    @abstractmethod
    def embed_indices_in_image(self, image: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """
        Add optimized index row to image representation.
        
        Args:
            image: Original 2D representation
            indices: Hierarchical indices to embed
            
        Returns:
            Enhanced image with index row
        """
        pass


class MPEGAICompressor(ABC):
    """Interface for MPEG-AI compression operations."""
    
    @abstractmethod
    def compress(self, image: np.ndarray, quality: float) -> bytes:
        """
        Apply MPEG-AI compression to image representation.
        
        Args:
            image: 2D image representation
            quality: Compression quality (0.0 to 1.0)
            
        Returns:
            Compressed data as bytes
        """
        pass
    
    @abstractmethod
    def decompress(self, compressed_data: bytes) -> np.ndarray:
        """
        Reconstruct image from compressed representation.
        
        Args:
            compressed_data: Compressed image data
            
        Returns:
            Reconstructed 2D image
        """
        pass
    
    @abstractmethod
    def estimate_compression_ratio(self, original_size: int, compressed_size: int) -> float:
        """
        Calculate compression efficiency metrics.
        
        Args:
            original_size: Size of original data in bytes
            compressed_size: Size of compressed data in bytes
            
        Returns:
            Compression ratio
        """
        pass


class SimilaritySearchEngine(ABC):
    """Interface for progressive similarity search operations."""
    
    @abstractmethod
    def progressive_search(self, query_indices: np.ndarray, 
                          candidate_pool: List[QuantizedModel], 
                          max_results: int) -> List[SearchResult]:
        """
        Perform progressive filtering using hierarchical indices.
        
        Args:
            query_indices: Hierarchical indices of query
            candidate_pool: Pool of candidate models
            max_results: Maximum number of results to return
            
        Returns:
            List of search results ranked by similarity
        """
        pass
    
    @abstractmethod
    def compare_indices_at_level(self, query_indices: np.ndarray, 
                                candidate_indices: np.ndarray, 
                                level: int) -> float:
        """
        Compare spatial indices at specific granularity level.
        
        Args:
            query_indices: Query hierarchical indices
            candidate_indices: Candidate hierarchical indices
            level: Granularity level to compare
            
        Returns:
            Similarity score for the level
        """
        pass