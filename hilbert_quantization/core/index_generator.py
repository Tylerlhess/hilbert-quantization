"""
Hierarchical index generator implementation for optimized spatial indexing.
"""

from typing import List, Tuple
import numpy as np
import math

from ..interfaces import HierarchicalIndexGenerator


class HierarchicalIndexGeneratorImpl(HierarchicalIndexGenerator):
    """
    Implementation of hierarchical spatial index generation using optimized space allocation.
    
    Uses a 1/2, 1/4, 1/8 strategy to allocate index space across different granularity levels,
    starting with finest granularity for maximum discrimination power.
    """
    
    def calculate_level_allocation(self, total_space: int) -> List[Tuple[int, int]]:
        """
        Calculate optimal space allocation for each granularity level using 1/2, 1/4, 1/8 strategy.
        
        Args:
            total_space: Total available index space
            
        Returns:
            List of (grid_size, space_allocated) tuples ordered by granularity (finest first)
        """
        if total_space <= 0:
            return []
        
        allocations = []
        remaining_space = total_space
        
        # Start with finest granularity that makes sense
        # For typical image sizes, start with 32x32 as finest practical level
        # This gives good discrimination without being too fine-grained
        max_practical_grid = min(32, int(math.sqrt(total_space)))
        
        # Find the largest power of 2 that's reasonable
        current_grid_size = 1
        while current_grid_size <= max_practical_grid:
            current_grid_size *= 2
        current_grid_size //= 2  # Step back to largest valid size
        
        # Ensure we start with at least 2x2 for meaningful spatial division
        current_grid_size = max(current_grid_size, 2)
        
        # Allocate space using 1/2, 1/4, 1/8 strategy
        fraction = 0.5
        
        while remaining_space > 0 and current_grid_size >= 1:
            # Calculate sections needed for this grid size
            sections_needed = current_grid_size * current_grid_size
            
            # Calculate space to allocate (fraction of remaining or what's needed)
            space_to_allocate = min(
                int(remaining_space * fraction),
                sections_needed,
                remaining_space
            )
            
            if space_to_allocate > 0:
                allocations.append((current_grid_size, space_to_allocate))
                remaining_space -= space_to_allocate
            
            # Move to next coarser level
            current_grid_size //= 2
            
            # Reduce fraction for next level (1/2 -> 1/4 -> 1/8 -> 1/16...)
            fraction *= 0.5
            
            # Stop if fraction becomes too small to be useful
            if fraction < 0.01:
                break
        
        # If there's still space, use it for offset sampling at the finest level
        if remaining_space > 0 and allocations:
            finest_grid = allocations[0][0]
            # Add offset sampling allocation (same grid size, different sampling strategy)
            allocations.append((finest_grid, remaining_space))
        
        return allocations
    
    def calculate_spatial_averages(self, image: np.ndarray, grid_size: int) -> List[float]:
        """
        Calculate averages for spatial sections at given grid size.
        
        Args:
            image: 2D parameter representation
            grid_size: Size of grid sections
            
        Returns:
            List of spatial averages
        """
        if image.size == 0 or grid_size <= 0:
            return []
        
        height, width = image.shape
        averages = []
        
        # Calculate section dimensions
        section_height = height // grid_size
        section_width = width // grid_size
        
        # If sections don't divide evenly, we'll handle partial sections
        if section_height == 0 or section_width == 0:
            # Grid is too fine for image size, return overall average
            return [float(np.mean(image))]
        
        # Calculate average for each section
        for row in range(grid_size):
            for col in range(grid_size):
                # Calculate section boundaries
                start_row = row * section_height
                end_row = min((row + 1) * section_height, height)
                start_col = col * section_width
                end_col = min((col + 1) * section_width, width)
                
                # Extract section and calculate average
                section = image[start_row:end_row, start_col:end_col]
                if section.size > 0:
                    avg = float(np.mean(section))
                    averages.append(avg)
                else:
                    # Fallback for empty sections
                    averages.append(0.0)
        
        return averages
    
    def calculate_offset_samples(self, image: np.ndarray, section_size: int, 
                               available_space: int) -> List[float]:
        """
        Calculate offset sampling (corners + centers) for sections.
        
        Args:
            image: 2D parameter representation
            section_size: Size of each section
            available_space: Number of samples to generate
            
        Returns:
            List of offset sample values
        """
        if image.size == 0 or section_size <= 0 or available_space <= 0:
            return []
        
        height, width = image.shape
        samples = []
        
        # Calculate how many sections we can fit
        sections_y = height // section_size
        sections_x = width // section_size
        
        if sections_y == 0 or sections_x == 0:
            # Sections too large, return corner and center samples of whole image
            corner_samples = [
                float(image[0, 0]),  # Top-left
                float(image[0, -1]),  # Top-right
                float(image[-1, 0]),  # Bottom-left
                float(image[-1, -1])  # Bottom-right
            ]
            center_sample = [float(image[height//2, width//2])]  # Center
            
            all_samples = corner_samples + center_sample
            return all_samples[:available_space]
        
        # For each section, calculate corner and center samples
        samples_per_section = 5  # 4 corners + 1 center
        sections_to_sample = min(
            available_space // samples_per_section,
            sections_y * sections_x
        )
        
        section_count = 0
        for row in range(sections_y):
            for col in range(sections_x):
                if section_count >= sections_to_sample:
                    break
                
                # Calculate section boundaries
                start_row = row * section_size
                end_row = min((row + 1) * section_size, height)
                start_col = col * section_size
                end_col = min((col + 1) * section_size, width)
                
                # Sample corners and center of this section
                section_samples = [
                    float(image[start_row, start_col]),  # Top-left
                    float(image[start_row, end_col-1]),  # Top-right
                    float(image[end_row-1, start_col]),  # Bottom-left
                    float(image[end_row-1, end_col-1]),  # Bottom-right
                    float(image[(start_row + end_row)//2, (start_col + end_col)//2])  # Center
                ]
                
                samples.extend(section_samples)
                section_count += 1
                
                if len(samples) >= available_space:
                    break
            
            if len(samples) >= available_space:
                break
        
        return samples[:available_space]
    
    def embed_indices_in_image(self, image: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """
        Add optimized index row to image representation.
        
        Args:
            image: Original 2D representation
            indices: Hierarchical indices to embed
            
        Returns:
            Enhanced image with index row
        """
        if image.size == 0:
            return image
        
        height, width = image.shape
        
        # Create new image with additional row for indices
        enhanced_image = np.zeros((height + 1, width), dtype=image.dtype)
        
        # Copy original image
        enhanced_image[:height, :] = image
        
        # Embed indices in the last row
        # If we have more indices than width, truncate
        # If we have fewer indices than width, pad with zeros
        indices_to_embed = min(len(indices), width)
        enhanced_image[height, :indices_to_embed] = indices[:indices_to_embed]
        
        # Pad remaining positions with zeros if needed
        if indices_to_embed < width:
            enhanced_image[height, indices_to_embed:] = 0.0
        
        return enhanced_image
    
    def extract_indices_from_image(self, enhanced_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract hierarchical indices and original image from enhanced representation.
        
        Args:
            enhanced_image: Image with embedded index row
            
        Returns:
            Tuple of (original_image, indices)
        """
        if enhanced_image.size == 0:
            return enhanced_image, np.array([])
        
        height, width = enhanced_image.shape
        
        if height < 2:
            # No index row to extract
            return enhanced_image, np.array([])
        
        # Extract original image (all rows except last)
        original_image = enhanced_image[:-1, :]
        
        # Extract indices from last row
        indices = enhanced_image[-1, :]
        
        # Remove trailing zeros from indices
        # Find last non-zero index
        non_zero_indices = np.nonzero(indices)[0]
        if len(non_zero_indices) > 0:
            last_non_zero = non_zero_indices[-1]
            indices = indices[:last_non_zero + 1]
        else:
            # All zeros, but keep at least one element if it exists
            indices = indices[:1] if len(indices) > 0 else np.array([])
        
        return original_image, indices
    
    def generate_optimized_indices(self, image: np.ndarray, index_space_size: int) -> np.ndarray:
        """
        Generate hierarchical spatial indices using optimized space allocation.
        
        Args:
            image: 2D parameter representation
            index_space_size: Available space for indices
            
        Returns:
            1D array of hierarchical indices
        """
        if image.size == 0 or index_space_size <= 0:
            return np.array([])
        
        # Calculate optimal space allocation
        allocations = self.calculate_level_allocation(index_space_size)
        
        if not allocations:
            return np.array([])
        
        # Generate indices for each level
        all_indices = []
        
        for grid_size, space_allocated in allocations:
            if space_allocated <= 0:
                continue
            
            # Check if this is offset sampling (duplicate grid size)
            is_offset_sampling = (
                len(all_indices) > 0 and 
                any(prev_grid == grid_size for prev_grid, _ in allocations[:-1])
            )
            
            if is_offset_sampling:
                # Use offset sampling for remaining space
                section_size = max(1, image.shape[0] // grid_size)
                indices = self.calculate_offset_samples(image, section_size, space_allocated)
            else:
                # Use regular spatial averages
                indices = self.calculate_spatial_averages(image, grid_size)
                
                # Truncate to allocated space
                indices = indices[:space_allocated]
            
            all_indices.extend(indices)
        
        # Convert to numpy array and ensure we don't exceed index space
        result = np.array(all_indices[:index_space_size], dtype=np.float32)
        
        # Pad with zeros if we have fewer indices than space
        if len(result) < index_space_size:
            padded_result = np.zeros(index_space_size, dtype=np.float32)
            padded_result[:len(result)] = result
            result = padded_result
        
        return result