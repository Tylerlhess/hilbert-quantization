"""
Hilbert curve mapping implementation for parameter quantization.
"""

from typing import List, Tuple, Optional, TYPE_CHECKING
import numpy as np
from ..interfaces import HilbertCurveMapper as IHilbertCurveMapper
from ..exceptions import HilbertQuantizationError

if TYPE_CHECKING:
    pass


class HilbertCurveMapper(IHilbertCurveMapper):
    """Implementation of Hilbert curve parameter mapping operations."""
    
    def generate_hilbert_coordinates(self, n: int) -> List[Tuple[int, int]]:
        """
        Generate Hilbert curve coordinate sequence for n×n grid.
        
        Args:
            n: Grid size (must be power of 2)
            
        Returns:
            List of (x, y) coordinates in Hilbert order
            
        Raises:
            HilbertQuantizationError: If n is not a power of 2
        """
        if n <= 0 or (n & (n - 1)) != 0:
            raise HilbertQuantizationError(f"Grid size must be a power of 2, got {n}")
        
        coordinates = []
        
        # Generate all coordinates using recursive Hilbert curve algorithm
        for i in range(n * n):
            x, y = self._hilbert_index_to_xy(i, n)
            coordinates.append((x, y))
        
        return coordinates
    
    def _hilbert_index_to_xy(self, index: int, n: int) -> Tuple[int, int]:
        """
        Convert Hilbert curve index to (x, y) coordinates.
        
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
    
    def map_to_2d(self, parameters: np.ndarray, dimensions: Tuple[int, int], 
                  builder: Optional = None) -> np.ndarray:
        """
        Map 1D parameters to 2D using Hilbert curve ordering.
        
        Args:
            parameters: 1D array of parameters
            dimensions: Target 2D dimensions (width, height)
            builder: Optional builder (GeneratorTreeBuilder or StreamingIndexBuilder) for simultaneous index building
            
        Returns:
            2D array representation
            
        Raises:
            HilbertQuantizationError: If dimensions are invalid or parameters don't fit
        """
        width, height = dimensions
        
        if width != height:
            raise HilbertQuantizationError(f"Hilbert curve requires square dimensions, got {width}x{height}")
        
        if width <= 0 or (width & (width - 1)) != 0:
            raise HilbertQuantizationError(f"Dimension must be a power of 2, got {width}")
        
        total_cells = width * height
        if len(parameters) > total_cells:
            raise HilbertQuantizationError(
                f"Too many parameters ({len(parameters)}) for dimensions {width}x{height} ({total_cells} cells)"
            )
        
        # Create 2D array filled with zeros (for padding)
        result = np.zeros((height, width), dtype=parameters.dtype)
        
        # Generate Hilbert curve coordinates
        coordinates = self.generate_hilbert_coordinates(width)
        
        # Reset builder if provided
        if builder is not None and hasattr(builder, 'reset'):
            builder.reset()
        
        # Map parameters to 2D using Hilbert curve ordering
        # Simultaneously build indices if builder is provided
        for i, param_value in enumerate(parameters):
            if i >= len(coordinates):
                break
            x, y = coordinates[i]
            result[y, x] = param_value
            
            # Add parameter to builder during mapping
            # This preserves spatial locality as adjacent Hilbert positions
            # are processed sequentially
            if builder is not None:
                if hasattr(builder, 'add_parameter_value'):
                    # Generator tree builder
                    builder.add_parameter_value(float(param_value))
                elif hasattr(builder, 'add_value'):
                    # Streaming index builder
                    builder.add_value(float(param_value))
        
        return result
    
    def map_from_2d(self, image: np.ndarray) -> np.ndarray:
        """
        Reconstruct 1D parameters from 2D Hilbert curve representation.
        
        Args:
            image: 2D array representation
            
        Returns:
            1D parameter array
            
        Raises:
            HilbertQuantizationError: If image dimensions are invalid
        """
        height, width = image.shape
        
        if width != height:
            raise HilbertQuantizationError(f"Hilbert curve requires square dimensions, got {width}x{height}")
        
        if width <= 0 or (width & (width - 1)) != 0:
            raise HilbertQuantizationError(f"Dimension must be a power of 2, got {width}")
        
        # Generate Hilbert curve coordinates
        coordinates = self.generate_hilbert_coordinates(width)
        
        # Extract parameters in Hilbert curve order
        parameters = []
        for x, y in coordinates:
            parameters.append(image[y, x])
        
        return np.array(parameters, dtype=image.dtype)
    

