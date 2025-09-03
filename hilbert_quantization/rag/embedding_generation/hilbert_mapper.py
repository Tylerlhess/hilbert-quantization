"""
Hilbert curve mapper implementation for embeddings.
"""

from typing import List, Tuple
import numpy as np


class HilbertCurveMapperImpl:
    """Implementation of Hilbert curve mapping for embeddings."""
    
    def __init__(self, config):
        """Initialize Hilbert curve mapper with configuration."""
        self.config = config
    
    def map_to_2d(self, embeddings: np.ndarray, dimensions: Tuple[int, int]) -> np.ndarray:
        """
        Map 1D embeddings to 2D using Hilbert curve ordering preserving similarity.
        
        This function maps a 1D embedding vector to a 2D representation using
        Hilbert curve coordinates. The Hilbert curve preserves spatial locality,
        meaning similar embedding values will be placed near each other in the
        2D representation.
        
        Args:
            embeddings: 1D array of embedding values
            dimensions: Target 2D dimensions (width, height)
            
        Returns:
            2D array representation of the embedding
            
        Raises:
            ValueError: If dimensions are invalid or embeddings don't fit
        """
        width, height = dimensions
        
        # Validate dimensions
        if width <= 0 or height <= 0:
            raise ValueError(f"Dimensions must be positive, got {width}x{height}")
        
        # For Hilbert curve, we need square dimensions that are powers of 2
        if width != height:
            raise ValueError(f"Hilbert curve requires square dimensions, got {width}x{height}")
        
        if width <= 0 or (width & (width - 1)) != 0:
            raise ValueError(f"Dimension must be a power of 2, got {width}")
        
        total_cells = width * height
        embedding_length = len(embeddings)
        
        if embedding_length > total_cells:
            raise ValueError(
                f"Too many embedding values ({embedding_length}) for dimensions {width}x{height} ({total_cells} cells)"
            )
        
        # Create 2D array filled with zeros (for padding)
        result = np.zeros((height, width), dtype=embeddings.dtype)
        
        # Generate Hilbert curve coordinates
        coordinates = self.generate_hilbert_coordinates(width)
        
        # Map embeddings to 2D using Hilbert curve ordering
        # This preserves spatial locality - similar embedding indices will be
        # mapped to nearby positions in the 2D space
        for i, embedding_value in enumerate(embeddings):
            if i >= len(coordinates):
                break
            x, y = coordinates[i]
            result[y, x] = embedding_value
        
        # Remaining cells are automatically padded with zeros
        # This handles non-square embedding dimensions by padding unused space
        # with a solid color/value (zero) at the end portion of the image
        
        return result
    
    def map_from_2d(self, image: np.ndarray) -> np.ndarray:
        """
        Reconstruct 1D embeddings from 2D Hilbert curve representation.
        
        This function performs the inverse operation of map_to_2d, reconstructing
        the original 1D embedding vector from its 2D Hilbert curve representation.
        The mapping is bijective, ensuring perfect reconstruction when no compression
        artifacts are present.
        
        Args:
            image: 2D array representation of the embedding
            
        Returns:
            1D array of reconstructed embedding values
            
        Raises:
            ValueError: If image dimensions are invalid for Hilbert curve mapping
        """
        if len(image.shape) != 2:
            raise ValueError(f"Input must be 2D array, got {len(image.shape)}D")
        
        height, width = image.shape
        
        # Validate dimensions for Hilbert curve
        if width != height:
            raise ValueError(f"Hilbert curve requires square dimensions, got {width}x{height}")
        
        if width <= 0 or (width & (width - 1)) != 0:
            raise ValueError(f"Dimension must be a power of 2, got {width}")
        
        # Generate Hilbert curve coordinates for reconstruction
        coordinates = self.generate_hilbert_coordinates(width)
        
        # Reconstruct 1D embedding by following Hilbert curve order
        # This ensures bijective mapping with the forward transformation
        reconstructed = []
        for x, y in coordinates:
            value = image[y, x]
            reconstructed.append(value)
        
        # Convert to numpy array with same dtype as input
        result = np.array(reconstructed, dtype=image.dtype)
        
        return result
    
    def generate_hilbert_coordinates(self, n: int) -> List[Tuple[int, int]]:
        """
        Generate Hilbert curve coordinate sequence for n×n grid.
        
        This function implements recursive Hilbert curve coordinate generation
        for embedding mapping. The Hilbert curve preserves spatial locality,
        meaning similar embeddings will be mapped to nearby coordinates.
        
        Args:
            n: Grid size (must be power of 2)
            
        Returns:
            List of (x, y) coordinates in Hilbert order
            
        Raises:
            ValueError: If n is not a power of 2 or is invalid
        """
        if n <= 0 or (n & (n - 1)) != 0:
            raise ValueError(f"Grid size must be a power of 2, got {n}")
        
        coordinates = []
        
        # Generate all coordinates using recursive Hilbert curve algorithm
        for i in range(n * n):
            x, y = self._hilbert_index_to_xy(i, n)
            coordinates.append((x, y))
        
        return coordinates
    
    def _hilbert_index_to_xy(self, index: int, n: int) -> Tuple[int, int]:
        """
        Convert Hilbert curve index to (x, y) coordinates.
        
        This implements the standard Hilbert curve algorithm that recursively
        subdivides the space while maintaining spatial locality properties.
        
        Args:
            index: Position along Hilbert curve
            n: Grid size
            
        Returns:
            (x, y) coordinates
        """
        x = y = 0
        t = index
        s = 1
        
        while s < n:
            rx = 1 & (t // 2)
            ry = 1 & (t ^ rx)
            x, y = self._rotate(s, x, y, rx, ry)
            x += s * rx
            y += s * ry
            t //= 4
            s *= 2
        
        return x, y
    
    def _xy_to_hilbert_index(self, x: int, y: int, n: int) -> int:
        """
        Convert (x, y) coordinates to Hilbert curve index.
        
        This is the inverse operation of _hilbert_index_to_xy.
        
        Args:
            x: X coordinate
            y: Y coordinate
            n: Grid size
            
        Returns:
            Position along Hilbert curve
        """
        index = 0
        s = n // 2
        
        while s > 0:
            rx = 1 if (x & s) > 0 else 0
            ry = 1 if (y & s) > 0 else 0
            index += s * s * ((3 * rx) ^ ry)
            x, y = self._rotate(s, x, y, rx, ry)
            s //= 2
        
        return index
    
    def _rotate(self, n: int, x: int, y: int, rx: int, ry: int) -> Tuple[int, int]:
        """
        Rotate and flip coordinates for Hilbert curve generation.
        
        This function handles the recursive rotation and reflection operations
        that are fundamental to the Hilbert curve construction.
        
        Args:
            n: Current grid size
            x: X coordinate
            y: Y coordinate
            rx: X rotation flag
            ry: Y rotation flag
            
        Returns:
            Rotated (x, y) coordinates
        """
        if ry == 0:
            if rx == 1:
                x = n - 1 - x
                y = n - 1 - y
            # Swap x and y
            x, y = y, x
        
        return x, y