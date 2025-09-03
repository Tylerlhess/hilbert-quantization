"""
Multi-level hierarchical index generator for Hilbert curve embeddings.

This module implements dynamic index space allocation and multi-level hierarchical
indices for efficient progressive filtering during similarity search. It supports
true Hilbert curve order-based spatial averaging for optimal similarity preservation.
"""

import math
from typing import List, Tuple, Dict
import numpy as np


class HierarchicalIndexGenerator:
    """Generator for multi-level hierarchical indices with dynamic space allocation."""
    
    def __init__(self, config=None):
        """Initialize hierarchical index generator with configuration."""
        self.config = config or {}
        self.min_granularity = self.config.get('min_granularity', 2)  # Minimum 2x2 sections
        self.max_index_rows = self.config.get('max_index_rows', 8)  # Maximum index rows to add
    
    def calculate_optimal_granularity(self, image_dimensions: Tuple[int, int]) -> Dict[str, any]:
        """
        Calculate optimal granularity levels based on image dimensions.
        
        Args:
            image_dimensions: Tuple of (width, height) for the embedding image
            
        Returns:
            Dictionary containing granularity information:
            - finest_granularity: Maximum sections per side for finest level
            - granularity_levels: List of granularity levels from finest to coarsest
            - index_rows_needed: Number of index rows needed
            - total_image_height: Original height + index rows
        """
        width, height = image_dimensions
        
        # Calculate maximum section size as sqrt(image_width) for finest level
        # This ensures we don't create more sections than can fit reasonably
        finest_granularity = int(math.sqrt(width))
        
        # Ensure finest granularity is at least the minimum and is a power of 2
        finest_granularity = max(self.min_granularity, finest_granularity)
        finest_granularity = self._nearest_power_of_2(finest_granularity)
        
        # Generate progressive granularity levels (powers of 2)
        granularity_levels = []
        current_granularity = finest_granularity
        
        while current_granularity >= self.min_granularity and len(granularity_levels) < self.max_index_rows:
            granularity_levels.append(current_granularity)
            current_granularity = current_granularity // 2
        
        # Calculate how many index rows we need
        index_rows_needed = len(granularity_levels)
        
        # Calculate total image height including index rows
        total_image_height = height + index_rows_needed
        
        return {
            'finest_granularity': finest_granularity,
            'granularity_levels': granularity_levels,
            'index_rows_needed': index_rows_needed,
            'total_image_height': total_image_height,
            'original_dimensions': image_dimensions,
            'section_sizes': [(width // g, height // g) for g in granularity_levels]
        }
    
    def allocate_index_space(self, image_dimensions: Tuple[int, int]) -> Dict[str, any]:
        """
        Allocate index space dynamically based on image dimensions.
        
        Args:
            image_dimensions: Tuple of (width, height) for the embedding image
            
        Returns:
            Dictionary containing space allocation information:
            - enhanced_dimensions: New dimensions including index rows
            - index_row_positions: List of row indices for each granularity level
            - granularity_info: Detailed granularity information
        """
        granularity_info = self.calculate_optimal_granularity(image_dimensions)
        
        width, height = image_dimensions
        index_rows_needed = granularity_info['index_rows_needed']
        
        # Enhanced dimensions include the original image plus index rows
        enhanced_dimensions = (width, height + index_rows_needed)
        
        # Calculate positions for each index row
        index_row_positions = []
        for i in range(index_rows_needed):
            row_position = height + i  # Index rows start after the original image
            index_row_positions.append(row_position)
        
        return {
            'enhanced_dimensions': enhanced_dimensions,
            'index_row_positions': index_row_positions,
            'granularity_info': granularity_info
        }
    
    def generate_multi_level_indices(self, embedding_image: np.ndarray) -> np.ndarray:
        """
        Generate multiple index rows for different Hilbert curve orders.
        
        This implements requirements 3.1, 3.2, and 3.3 by creating hierarchical indices
        at different Hilbert curve orders with progressive granularity levels.
        
        Args:
            embedding_image: 2D numpy array representing the Hilbert-mapped embedding
            
        Returns:
            Enhanced image with hierarchical index rows appended
        """
        if embedding_image.ndim != 2:
            raise ValueError("Embedding image must be 2D")
        
        height, width = embedding_image.shape
        space_allocation = self.allocate_index_space((width, height))
        
        enhanced_height = space_allocation['enhanced_dimensions'][1]
        granularity_levels = space_allocation['granularity_info']['granularity_levels']
        index_row_positions = space_allocation['index_row_positions']
        
        # Create enhanced image with space for index rows
        enhanced_image = np.zeros((enhanced_height, width), dtype=embedding_image.dtype)
        
        # Copy original embedding image
        enhanced_image[:height, :] = embedding_image
        
        # Generate index rows for each granularity level using Hilbert curve order-based averaging
        for i, granularity in enumerate(granularity_levels):
            row_position = index_row_positions[i]
            
            # Calculate Hilbert curve order-based spatial averages
            index_row = self._calculate_hilbert_order_averages(embedding_image, granularity)
            
            # Ensure index row fits in the image width
            if len(index_row) <= width:
                enhanced_image[row_position, :len(index_row)] = index_row
            else:
                # If index row is too long, truncate it
                enhanced_image[row_position, :] = index_row[:width]
        
        return enhanced_image
    
    def create_progressive_granularity_levels(self, embedding_image: np.ndarray) -> List[np.ndarray]:
        """
        Create multiple index rows for progressive granularity levels.
        
        This implements requirement 3.3 by computing averages for different granularity
        levels starting with finest practical granularity and progressively coarser levels.
        
        Args:
            embedding_image: 2D numpy array representing the Hilbert-mapped embedding
            
        Returns:
            List of 1D arrays, each containing spatial averages for a granularity level
        """
        if embedding_image.ndim != 2:
            raise ValueError("Embedding image must be 2D")
        
        height, width = embedding_image.shape
        space_allocation = self.allocate_index_space((width, height))
        granularity_levels = space_allocation['granularity_info']['granularity_levels']
        
        index_rows = []
        
        # Generate index rows for each granularity level (finest to coarsest)
        for granularity in granularity_levels:
            index_row = self._calculate_hilbert_order_averages(embedding_image, granularity)
            index_rows.append(index_row)
        
        return index_rows
    
    def calculate_averages_for_multiple_granularities(self, embedding_image: np.ndarray, 
                                                    granularity_levels: List[int]) -> Dict[int, np.ndarray]:
        """
        Calculate spatial averages for multiple granularity levels.
        
        This method allows for custom granularity levels and returns a mapping
        from granularity level to the corresponding spatial averages.
        
        Args:
            embedding_image: 2D numpy array representing the Hilbert-mapped embedding
            granularity_levels: List of granularity levels to compute averages for
            
        Returns:
            Dictionary mapping granularity level to spatial averages array
        """
        if embedding_image.ndim != 2:
            raise ValueError("Embedding image must be 2D")
        
        averages_dict = {}
        
        for granularity in granularity_levels:
            if granularity > 0:
                averages = self._calculate_hilbert_order_averages(embedding_image, granularity)
                averages_dict[granularity] = averages
        
        return averages_dict
    
    def _calculate_hilbert_order_averages(self, image: np.ndarray, granularity: int) -> np.ndarray:
        """
        Calculate spatial averages for a specific Hilbert curve order level.
        
        This implements requirement 3.2 by calculating spatial averages at different
        Hilbert curve orders, preserving the spatial locality properties of the curve.
        
        Args:
            image: 2D numpy array to calculate averages for
            granularity: Number of sections per side (e.g., 32 for 32x32 sections)
            
        Returns:
            1D array of spatial averages in Hilbert curve order
        """
        height, width = image.shape
        
        # Calculate section sizes
        section_height = height // granularity
        section_width = width // granularity
        
        if section_height == 0 or section_width == 0:
            # If granularity is too fine, return overall average
            return np.array([np.mean(image)])
        
        # Generate Hilbert curve coordinates for the granularity level
        hilbert_coords = self._generate_hilbert_coordinates(granularity)
        
        averages = []
        
        # Calculate averages for each section following Hilbert curve order
        for row, col in hilbert_coords:
            start_row = row * section_height
            end_row = min((row + 1) * section_height, height)
            start_col = col * section_width
            end_col = min((col + 1) * section_width, width)
            
            section = image[start_row:end_row, start_col:end_col]
            section_average = np.mean(section)
            averages.append(section_average)
        
        return np.array(averages)
    
    def _calculate_spatial_averages(self, image: np.ndarray, granularity: int) -> np.ndarray:
        """
        Calculate spatial averages for a specific granularity level (legacy method).
        
        This method is kept for backward compatibility but _calculate_hilbert_order_averages
        should be preferred for new implementations.
        
        Args:
            image: 2D numpy array to calculate averages for
            granularity: Number of sections per side (e.g., 32 for 32x32 sections)
            
        Returns:
            1D array of spatial averages
        """
        height, width = image.shape
        
        # Calculate section sizes
        section_height = height // granularity
        section_width = width // granularity
        
        if section_height == 0 or section_width == 0:
            # If granularity is too fine, return overall average
            return np.array([np.mean(image)])
        
        averages = []
        
        # Calculate averages for each section in row-major order
        for row in range(granularity):
            for col in range(granularity):
                start_row = row * section_height
                end_row = min((row + 1) * section_height, height)
                start_col = col * section_width
                end_col = min((col + 1) * section_width, width)
                
                section = image[start_row:end_row, start_col:end_col]
                section_average = np.mean(section)
                averages.append(section_average)
        
        return np.array(averages)
    
    def _generate_hilbert_coordinates(self, n: int) -> List[Tuple[int, int]]:
        """
        Generate Hilbert curve coordinates for an n×n grid.
        
        This implements the recursive Hilbert curve generation algorithm to ensure
        spatial locality is preserved in the hierarchical indices.
        
        Args:
            n: Size of the grid (n×n)
            
        Returns:
            List of (row, col) coordinates in Hilbert curve order
        """
        if n == 1:
            return [(0, 0)]
        
        if n == 2:
            return [(0, 0), (0, 1), (1, 1), (1, 0)]
        
        # For larger n, use recursive approach
        # This is a simplified implementation for powers of 2
        if n & (n - 1) != 0:
            # If n is not a power of 2, use the nearest smaller power of 2
            n = self._nearest_power_of_2(n)
        
        coordinates = []
        
        # Generate coordinates using recursive Hilbert curve algorithm
        def hilbert_curve(x, y, xi, xj, yi, yj, n):
            if n <= 0:
                coordinates.append((x + (xi + yi) // 2, y + (xj + yj) // 2))
            else:
                hilbert_curve(x, y, yi // 2, yj // 2, xi // 2, xj // 2, n - 1)
                hilbert_curve(x + xi // 2, y + xj // 2, xi // 2, xj // 2, yi // 2, yj // 2, n - 1)
                hilbert_curve(x + xi // 2 + yi // 2, y + xj // 2 + yj // 2, xi // 2, xj // 2, yi // 2, yj // 2, n - 1)
                hilbert_curve(x + xi // 2 + yi, y + xj // 2 + yj, -yi // 2, -yj // 2, -xi // 2, -xj // 2, n - 1)
        
        # Calculate the order (log2 of n)
        order = int(math.log2(n))
        hilbert_curve(0, 0, n, 0, 0, n, order)
        
        # Filter coordinates to be within bounds and remove duplicates
        valid_coords = []
        seen = set()
        for x, y in coordinates:
            if 0 <= x < n and 0 <= y < n and (x, y) not in seen:
                valid_coords.append((x, y))
                seen.add((x, y))
        
        # If we don't have enough coordinates, fill with row-major order
        if len(valid_coords) < n * n:
            for i in range(n):
                for j in range(n):
                    if (i, j) not in seen:
                        valid_coords.append((i, j))
        
        return valid_coords[:n*n]
    
    def embed_multi_level_indices(self, image: np.ndarray, index_rows: List[np.ndarray]) -> np.ndarray:
        """
        Add multiple index rows to embedding image representation.
        
        This implements requirement 3.4 by embedding multi-level hierarchical indices
        in correct positions within the enhanced image representation.
        
        Args:
            image: Original 2D embedding representation
            index_rows: List of hierarchical index arrays for different granularity levels
            
        Returns:
            Enhanced image with embedded index rows at correct positions
        """
        if image.ndim != 2:
            raise ValueError("Image must be 2D")
        
        if not index_rows:
            return image.copy()
        
        height, width = image.shape
        num_index_rows = len(index_rows)
        
        # Create enhanced image with space for index rows
        enhanced_height = height + num_index_rows
        enhanced_image = np.zeros((enhanced_height, width), dtype=image.dtype)
        
        # Copy original embedding image
        enhanced_image[:height, :] = image
        
        # Embed each index row at correct position
        for i, index_row in enumerate(index_rows):
            row_position = height + i  # Index rows start after original image
            
            # Ensure index row fits in the image width
            if len(index_row) <= width:
                enhanced_image[row_position, :len(index_row)] = index_row
            else:
                # If index row is too long, truncate it
                enhanced_image[row_position, :] = index_row[:width]
        
        return enhanced_image
    
    def extract_indices_from_image(self, enhanced_image: np.ndarray, 
                                 original_height: int = None) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Extract hierarchical indices and original image from enhanced representation.
        
        This implements the inverse operation of embed_multi_level_indices, allowing
        recovery of both the original embedding image and the hierarchical index rows.
        
        Args:
            enhanced_image: Image with embedded index rows
            original_height: Optional hint for the original image height. If not provided,
                           will attempt to detect automatically.
            
        Returns:
            Tuple of (original_image, list_of_index_rows)
        """
        if enhanced_image.ndim != 2:
            raise ValueError("Enhanced image must be 2D")
        
        height, width = enhanced_image.shape
        
        # Use provided original height or try to detect it
        if original_height is None:
            original_height = self._detect_original_image_height(enhanced_image)
        
        # Validate original_height
        if original_height < 0:
            original_height = 0
        elif original_height > height:
            original_height = height
        
        if original_height >= height:
            # No index rows detected
            return enhanced_image, []
        
        # Extract original image
        original_image = enhanced_image[:original_height, :]
        
        # Extract index rows
        index_rows = []
        for row_idx in range(original_height, height):
            index_row = enhanced_image[row_idx, :]
            
            # Remove trailing zeros from index row
            non_zero_indices = np.nonzero(index_row)[0]
            if len(non_zero_indices) > 0:
                last_non_zero = non_zero_indices[-1]
                cleaned_row = index_row[:last_non_zero + 1]
            else:
                # Keep at least one element if row exists
                cleaned_row = index_row[:1] if len(index_row) > 0 else np.array([])
            
            index_rows.append(cleaned_row)
        
        return original_image, index_rows
    
    def _detect_original_image_height(self, enhanced_image: np.ndarray) -> int:
        """
        Detect the height of the original image within an enhanced image.
        
        This uses heuristics to determine where the original embedding ends
        and the index rows begin.
        
        Args:
            enhanced_image: Enhanced image with embedded indices
            
        Returns:
            Height of the original image portion
        """
        height, width = enhanced_image.shape
        
        # Strategy 1: Look for rows that are mostly zeros (typical of index rows)
        # Index rows often have fewer non-zero elements than embedding rows
        zero_threshold = 0.5  # If more than 50% of row is zeros, likely an index row
        
        # Look for the first row from the bottom that has mostly zeros
        first_sparse_row = height
        for row_idx in range(height - 1, -1, -1):  # Search from bottom up
            row = enhanced_image[row_idx, :]
            zero_ratio = np.sum(row == 0) / len(row)
            
            if zero_ratio > zero_threshold:
                # This row is sparse, likely an index row
                first_sparse_row = row_idx
            else:
                # This row has significant data, likely part of original image
                break
        
        # Strategy 2: Look for statistical differences between rows
        # Embedding rows typically have more varied values than index rows
        if height > 2:
            row_variances = []
            for row_idx in range(height):
                row = enhanced_image[row_idx, :]
                non_zero_values = row[row != 0]
                if len(non_zero_values) > 1:
                    variance = np.var(non_zero_values)
                else:
                    variance = 0.0
                row_variances.append(variance)
            
            # Find the point where variance drops significantly
            if len(row_variances) > 1:
                # Calculate mean variance of first 80% of rows (likely original image)
                split_point = int(height * 0.8)
                if split_point > 0:
                    mean_variance = np.mean(row_variances[:split_point])
                    variance_threshold = mean_variance * 0.2  # 20% of mean variance
                    
                    for i in range(height - 1, 0, -1):
                        if (row_variances[i] < variance_threshold and 
                            row_variances[i-1] > variance_threshold):
                            return i
        
        # Use the sparse row detection result
        if first_sparse_row < height:
            return first_sparse_row
        
        # Fallback: Assume no index rows if detection fails
        return height
    
    def validate_embedded_indices(self, enhanced_image: np.ndarray, 
                                expected_index_rows: int, 
                                original_height: int = None) -> bool:
        """
        Validate that indices are properly embedded in the enhanced image.
        
        Args:
            enhanced_image: Image with embedded indices
            expected_index_rows: Expected number of index rows
            original_height: Optional hint for original image height
            
        Returns:
            True if validation passes, False otherwise
        """
        if enhanced_image.ndim != 2:
            return False
        
        try:
            # If original height is provided, use it for more accurate validation
            if original_height is not None:
                original_image, extracted_indices = self.extract_indices_from_image(
                    enhanced_image, original_height
                )
            else:
                original_image, extracted_indices = self.extract_indices_from_image(enhanced_image)
            
            # Check that we extracted the expected number of index rows
            if len(extracted_indices) != expected_index_rows:
                return False
            
            # Check that each index row contains meaningful data
            for index_row in extracted_indices:
                if len(index_row) == 0:
                    return False
                
                # Check that index row has some non-zero values
                if np.all(index_row == 0):
                    return False
            
            # Check that original image dimensions are reasonable
            if original_image.size == 0:
                return False
            
            # Check that enhanced image has the expected structure
            expected_height = (original_height or original_image.shape[0]) + expected_index_rows
            if enhanced_image.shape[0] != expected_height:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _nearest_power_of_2(self, n: int) -> int:
        """Find the nearest power of 2 that is <= n."""
        if n <= 0:
            return 1
        
        power = 1
        while power * 2 <= n:
            power *= 2
        
        return power
    
    def validate_index_allocation(self, image_dimensions: Tuple[int, int]) -> bool:
        """
        Validate that index allocation is feasible for given image dimensions.
        
        Args:
            image_dimensions: Tuple of (width, height) for the embedding image
            
        Returns:
            True if allocation is valid, False otherwise
        """
        try:
            space_allocation = self.allocate_index_space(image_dimensions)
            granularity_levels = space_allocation['granularity_info']['granularity_levels']
            
            # Check that we have at least one granularity level
            if not granularity_levels:
                return False
            
            # Check that finest granularity is reasonable
            finest = granularity_levels[0]
            width, height = image_dimensions
            
            if finest > min(width, height):
                return False
            
            # Check that we don't exceed maximum index rows
            if len(granularity_levels) > self.max_index_rows:
                return False
            
            return True
            
        except Exception:
            return False
    
    def create_enhanced_embedding_with_indices(self, embedding_image: np.ndarray) -> np.ndarray:
        """
        Create enhanced embedding image with embedded multi-level hierarchical indices.
        
        This is a convenience method that combines index generation and embedding
        in a single operation, implementing the complete workflow for requirement 3.4.
        
        Args:
            embedding_image: Original 2D embedding representation
            
        Returns:
            Enhanced image with embedded hierarchical indices
        """
        if embedding_image.ndim != 2:
            raise ValueError("Embedding image must be 2D")
        
        # Generate progressive granularity levels
        index_rows = self.create_progressive_granularity_levels(embedding_image)
        
        # Embed the index rows in the image
        enhanced_image = self.embed_multi_level_indices(embedding_image, index_rows)
        
        return enhanced_image