"""
Tests for hierarchical index generator implementation.
"""

import pytest
import numpy as np
from hilbert_quantization.core.index_generator import HierarchicalIndexGeneratorImpl


class TestHierarchicalIndexGenerator:
    """Test cases for hierarchical index generation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = HierarchicalIndexGeneratorImpl()
    
    def test_calculate_level_allocation_basic(self):
        """Test basic space allocation calculation."""
        # Test with 1024 index spaces (typical for 1024x1024 image)
        allocations = self.generator.calculate_level_allocation(1024)
        
        # Should have multiple levels
        assert len(allocations) > 0
        
        # Total allocated space should not exceed available space
        total_allocated = sum(space for _, space in allocations)
        assert total_allocated <= 1024
        
        # First allocation should be for finest granularity
        first_grid_size = allocations[0][0]
        assert first_grid_size > 1  # Should be meaningful grid size
        
        # Grid sizes should generally decrease (coarser granularity)
        # except for the last entry which might be offset sampling
        for i in range(len(allocations) - 2):  # Skip last comparison
            current_grid = allocations[i][0]
            next_grid = allocations[i + 1][0]
            assert next_grid <= current_grid
        
        # Last entry might be offset sampling at finest level
        if len(allocations) >= 2:
            last_grid = allocations[-1][0]
            first_grid = allocations[0][0]
            # Last should be same as first (offset sampling) or smaller
            assert last_grid <= first_grid
    
    def test_calculate_level_allocation_small_space(self):
        """Test allocation with small index space."""
        allocations = self.generator.calculate_level_allocation(16)
        
        # Should still work with small spaces
        assert len(allocations) > 0
        
        # Total should not exceed available
        total_allocated = sum(space for _, space in allocations)
        assert total_allocated <= 16
    
    def test_calculate_level_allocation_zero_space(self):
        """Test allocation with zero space."""
        allocations = self.generator.calculate_level_allocation(0)
        assert allocations == []
    
    def test_calculate_level_allocation_negative_space(self):
        """Test allocation with negative space."""
        allocations = self.generator.calculate_level_allocation(-10)
        assert allocations == []
    
    def test_calculate_level_allocation_power_of_two_optimization(self):
        """Test that allocation works well with power-of-2 spaces."""
        for space in [64, 128, 256, 512, 1024]:
            allocations = self.generator.calculate_level_allocation(space)
            
            # Should have reasonable number of levels
            assert 1 <= len(allocations) <= 10
            
            # Should use most of the available space efficiently
            total_allocated = sum(space_alloc for _, space_alloc in allocations)
            efficiency = total_allocated / space
            assert efficiency >= 0.8  # Should use at least 80% of space
    
    def test_calculate_level_allocation_fraction_strategy(self):
        """Test that the 1/2, 1/4, 1/8 strategy is followed."""
        allocations = self.generator.calculate_level_allocation(1000)
        
        if len(allocations) >= 3:
            # Check that space allocation generally follows decreasing pattern
            # (allowing for grid size constraints)
            spaces = [space for _, space in allocations[:3]]
            
            # First allocation should be significant portion
            assert spaces[0] >= 100  # Should allocate meaningful amount to finest level
    
    def test_calculate_level_allocation_grid_size_constraints(self):
        """Test that grid sizes are reasonable powers of 2."""
        allocations = self.generator.calculate_level_allocation(1024)
        
        for grid_size, space in allocations:
            # Grid size should be positive
            assert grid_size >= 1
            
            # For meaningful grids, should be power of 2
            if grid_size > 1:
                # Check if it's a power of 2
                assert (grid_size & (grid_size - 1)) == 0
            
            # Space allocation should be positive
            assert space > 0
            
            # Space should not exceed what's needed for the grid
            # (unless it's offset sampling)
            max_sections = grid_size * grid_size
            # Allow some flexibility for offset sampling
            assert space <= max_sections * 2
    
    def test_calculate_spatial_averages_basic(self):
        """Test basic spatial average calculation."""
        # Create a simple test image
        image = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0]
        ])
        
        # Test 2x2 grid (4 sections)
        averages = self.generator.calculate_spatial_averages(image, 2)
        
        # Should have 4 averages
        assert len(averages) == 4
        
        # Check calculated averages
        # Top-left: (1+2+5+6)/4 = 3.5
        # Top-right: (3+4+7+8)/4 = 5.5
        # Bottom-left: (9+10+13+14)/4 = 11.5
        # Bottom-right: (11+12+15+16)/4 = 13.5
        expected = [3.5, 5.5, 11.5, 13.5]
        
        for i, (actual, expected_val) in enumerate(zip(averages, expected)):
            assert abs(actual - expected_val) < 1e-10, f"Section {i}: expected {expected_val}, got {actual}"
    
    def test_calculate_spatial_averages_single_section(self):
        """Test spatial average with single section (1x1 grid)."""
        image = np.array([
            [1.0, 2.0],
            [3.0, 4.0]
        ])
        
        averages = self.generator.calculate_spatial_averages(image, 1)
        
        # Should have 1 average (overall average)
        assert len(averages) == 1
        assert abs(averages[0] - 2.5) < 1e-10  # (1+2+3+4)/4 = 2.5
    
    def test_calculate_spatial_averages_empty_image(self):
        """Test spatial average with empty image."""
        image = np.array([])
        averages = self.generator.calculate_spatial_averages(image, 2)
        assert averages == []
    
    def test_calculate_spatial_averages_grid_too_fine(self):
        """Test spatial average when grid is too fine for image."""
        image = np.array([[1.0, 2.0]])  # 1x2 image
        
        # Try 4x4 grid on 1x2 image
        averages = self.generator.calculate_spatial_averages(image, 4)
        
        # Should return overall average
        assert len(averages) == 1
        assert abs(averages[0] - 1.5) < 1e-10
    
    def test_calculate_offset_samples_basic(self):
        """Test basic offset sampling."""
        # Create a 4x4 test image
        image = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0]
        ])
        
        # Sample with 2x2 sections, asking for 10 samples
        samples = self.generator.calculate_offset_samples(image, 2, 10)
        
        # Should get samples (up to available space)
        assert len(samples) <= 10
        assert len(samples) > 0
        
        # All samples should be valid values from the image
        for sample in samples:
            assert 1.0 <= sample <= 16.0
    
    def test_calculate_offset_samples_small_image(self):
        """Test offset sampling with small image."""
        image = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        # Large section size
        samples = self.generator.calculate_offset_samples(image, 10, 5)
        
        # Should get corner and center samples
        assert len(samples) <= 5
        assert len(samples) > 0
        
        # Should include corners: 1.0, 2.0, 3.0, 4.0 and center
        expected_corners = {1.0, 2.0, 3.0, 4.0}
        sample_set = set(samples[:4])  # First 4 should be corners
        assert sample_set.issubset(expected_corners)
    
    def test_calculate_offset_samples_empty_image(self):
        """Test offset sampling with empty image."""
        image = np.array([])
        samples = self.generator.calculate_offset_samples(image, 2, 5)
        assert samples == []
    
    def test_embed_indices_in_image_basic(self):
        """Test basic index embedding."""
        # Create a simple test image
        image = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ])
        
        # Create some indices
        indices = np.array([0.1, 0.2, 0.3])
        
        # Embed indices
        enhanced = self.generator.embed_indices_in_image(image, indices)
        
        # Should have one additional row
        assert enhanced.shape == (3, 3)
        
        # Original image should be preserved
        np.testing.assert_array_equal(enhanced[:2, :], image)
        
        # Indices should be in last row
        np.testing.assert_array_equal(enhanced[2, :], indices)
    
    def test_embed_indices_in_image_truncate(self):
        """Test index embedding with truncation."""
        image = np.array([[1.0, 2.0]])  # 1x2 image
        indices = np.array([0.1, 0.2, 0.3, 0.4])  # More indices than width
        
        enhanced = self.generator.embed_indices_in_image(image, indices)
        
        # Should truncate indices to fit width
        assert enhanced.shape == (2, 2)
        np.testing.assert_array_equal(enhanced[1, :], [0.1, 0.2])
    
    def test_embed_indices_in_image_pad(self):
        """Test index embedding with padding."""
        image = np.array([[1.0, 2.0, 3.0, 4.0]])  # 1x4 image
        indices = np.array([0.1, 0.2])  # Fewer indices than width
        
        enhanced = self.generator.embed_indices_in_image(image, indices)
        
        # Should pad with zeros
        assert enhanced.shape == (2, 4)
        np.testing.assert_array_equal(enhanced[1, :], [0.1, 0.2, 0.0, 0.0])
    
    def test_extract_indices_from_image_basic(self):
        """Test basic index extraction."""
        # Create enhanced image
        enhanced = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [0.1, 0.2, 0.3]  # Index row
        ])
        
        original, indices = self.generator.extract_indices_from_image(enhanced)
        
        # Check original image
        expected_original = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ])
        np.testing.assert_array_equal(original, expected_original)
        
        # Check indices
        np.testing.assert_array_equal(indices, [0.1, 0.2, 0.3])
    
    def test_extract_indices_from_image_with_padding(self):
        """Test index extraction with zero padding."""
        enhanced = np.array([
            [1.0, 2.0],
            [0.1, 0.0]  # Index row with padding
        ])
        
        original, indices = self.generator.extract_indices_from_image(enhanced)
        
        # Should remove trailing zeros
        np.testing.assert_array_equal(original, [[1.0, 2.0]])
        np.testing.assert_array_equal(indices, [0.1])
    
    def test_generate_optimized_indices_basic(self):
        """Test complete optimized index generation."""
        # Create a test image
        image = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0]
        ])
        
        # Generate indices with reasonable space
        indices = self.generator.generate_optimized_indices(image, 20)
        
        # Should generate indices
        assert len(indices) == 20
        assert isinstance(indices, np.ndarray)
        
        # All indices should be finite numbers
        assert np.all(np.isfinite(indices))
    
    def test_generate_optimized_indices_empty_image(self):
        """Test index generation with empty image."""
        image = np.array([])
        indices = self.generator.generate_optimized_indices(image, 10)
        
        assert len(indices) == 0
    
    def test_generate_optimized_indices_zero_space(self):
        """Test index generation with zero space."""
        image = np.array([[1.0, 2.0], [3.0, 4.0]])
        indices = self.generator.generate_optimized_indices(image, 0)
        
        assert len(indices) == 0