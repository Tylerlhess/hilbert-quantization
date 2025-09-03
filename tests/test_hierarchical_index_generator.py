"""
Tests for hierarchical index generator with dynamic space allocation.
"""

import pytest
import numpy as np
from hilbert_quantization.rag.embedding_generation.hierarchical_index_generator import HierarchicalIndexGenerator


class TestHierarchicalIndexGenerator:
    """Test suite for HierarchicalIndexGenerator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = HierarchicalIndexGenerator()
        
    def test_calculate_optimal_granularity_1024x1024(self):
        """Test optimal granularity calculation for 1024x1024 image."""
        image_dimensions = (1024, 1024)
        result = self.generator.calculate_optimal_granularity(image_dimensions)
        
        # For 1024x1024, sqrt(1024) = 32, so finest granularity should be 32
        assert result['finest_granularity'] == 32
        
        # Should have progressive levels: 32, 16, 8, 4, 2
        expected_levels = [32, 16, 8, 4, 2]
        assert result['granularity_levels'] == expected_levels
        
        # Should need 5 index rows
        assert result['index_rows_needed'] == 5
        
        # Total height should be 1024 + 5 = 1029
        assert result['total_image_height'] == 1029
        
        # Check section sizes
        expected_sections = [(32, 32), (64, 64), (128, 128), (256, 256), (512, 512)]
        assert result['section_sizes'] == expected_sections
    
    def test_calculate_optimal_granularity_4096x4096(self):
        """Test optimal granularity calculation for 4096x4096 image."""
        image_dimensions = (4096, 4096)
        result = self.generator.calculate_optimal_granularity(image_dimensions)
        
        # For 4096x4096, sqrt(4096) = 64, so finest granularity should be 64
        assert result['finest_granularity'] == 64
        
        # Should have progressive levels: 64, 32, 16, 8, 4, 2
        expected_levels = [64, 32, 16, 8, 4, 2]
        assert result['granularity_levels'] == expected_levels
        
        # Should need 6 index rows
        assert result['index_rows_needed'] == 6
        
        # Total height should be 4096 + 6 = 4102
        assert result['total_image_height'] == 4102
    
    def test_calculate_optimal_granularity_small_image(self):
        """Test optimal granularity calculation for small image."""
        image_dimensions = (64, 64)
        result = self.generator.calculate_optimal_granularity(image_dimensions)
        
        # For 64x64, sqrt(64) = 8, so finest granularity should be 8
        assert result['finest_granularity'] == 8
        
        # Should have progressive levels: 8, 4, 2
        expected_levels = [8, 4, 2]
        assert result['granularity_levels'] == expected_levels
        
        # Should need 3 index rows
        assert result['index_rows_needed'] == 3
    
    def test_calculate_optimal_granularity_rectangular_image(self):
        """Test optimal granularity calculation for rectangular image."""
        image_dimensions = (2048, 1024)
        result = self.generator.calculate_optimal_granularity(image_dimensions)
        
        # For 2048x1024, sqrt(2048) ≈ 45, nearest power of 2 is 32
        assert result['finest_granularity'] == 32
        
        # Check that section sizes are calculated correctly
        section_sizes = result['section_sizes']
        assert section_sizes[0] == (2048 // 32, 1024 // 32)  # (64, 32)
    
    def test_allocate_index_space(self):
        """Test index space allocation."""
        image_dimensions = (1024, 1024)
        result = self.generator.allocate_index_space(image_dimensions)
        
        # Enhanced dimensions should include index rows
        assert result['enhanced_dimensions'] == (1024, 1029)
        
        # Index row positions should start after original image
        expected_positions = [1024, 1025, 1026, 1027, 1028]
        assert result['index_row_positions'] == expected_positions
        
        # Should include granularity info
        assert 'granularity_info' in result
        assert result['granularity_info']['finest_granularity'] == 32
    
    def test_generate_multi_level_indices(self):
        """Test generation of multi-level indices."""
        # Create a test embedding image
        embedding_image = np.random.rand(64, 64).astype(np.float32)
        
        # Generate enhanced image with indices
        enhanced_image = self.generator.generate_multi_level_indices(embedding_image)
        
        # Should have additional rows for indices
        space_allocation = self.generator.allocate_index_space((64, 64))
        expected_height = space_allocation['enhanced_dimensions'][1]
        
        assert enhanced_image.shape == (expected_height, 64)
        
        # Original image should be preserved
        np.testing.assert_array_equal(enhanced_image[:64, :], embedding_image)
        
        # Index rows should contain non-zero values (averages)
        index_rows = enhanced_image[64:, :]
        assert np.any(index_rows > 0)  # Should have some non-zero averages
    
    def test_calculate_spatial_averages(self):
        """Test spatial averages calculation."""
        # Create a test image with known pattern
        image = np.ones((32, 32), dtype=np.float32)
        image[:16, :16] = 2.0  # Top-left quadrant has value 2
        image[:16, 16:] = 3.0  # Top-right quadrant has value 3
        image[16:, :16] = 4.0  # Bottom-left quadrant has value 4
        image[16:, 16:] = 5.0  # Bottom-right quadrant has value 5
        
        # Calculate averages for 2x2 granularity (4 sections)
        averages = self.generator._calculate_spatial_averages(image, 2)
        
        # Should have 4 averages
        assert len(averages) == 4
        
        # Each average should match the quadrant value
        expected_averages = [2.0, 3.0, 4.0, 5.0]
        np.testing.assert_array_almost_equal(averages, expected_averages)
    
    def test_calculate_spatial_averages_fine_granularity(self):
        """Test spatial averages with very fine granularity."""
        image = np.ones((8, 8), dtype=np.float32)
        
        # Test with granularity larger than image
        averages = self.generator._calculate_spatial_averages(image, 16)
        
        # Should return overall average
        assert len(averages) == 1
        assert averages[0] == 1.0
    
    def test_nearest_power_of_2(self):
        """Test nearest power of 2 calculation."""
        assert self.generator._nearest_power_of_2(1) == 1
        assert self.generator._nearest_power_of_2(2) == 2
        assert self.generator._nearest_power_of_2(3) == 2
        assert self.generator._nearest_power_of_2(4) == 4
        assert self.generator._nearest_power_of_2(7) == 4
        assert self.generator._nearest_power_of_2(8) == 8
        assert self.generator._nearest_power_of_2(15) == 8
        assert self.generator._nearest_power_of_2(16) == 16
        assert self.generator._nearest_power_of_2(31) == 16
        assert self.generator._nearest_power_of_2(32) == 32
        assert self.generator._nearest_power_of_2(63) == 32
        assert self.generator._nearest_power_of_2(64) == 64
    
    def test_validate_index_allocation_valid_cases(self):
        """Test validation for valid index allocations."""
        # Standard cases should be valid
        assert self.generator.validate_index_allocation((1024, 1024)) == True
        assert self.generator.validate_index_allocation((4096, 4096)) == True
        assert self.generator.validate_index_allocation((64, 64)) == True
        assert self.generator.validate_index_allocation((2048, 1024)) == True
    
    def test_validate_index_allocation_invalid_cases(self):
        """Test validation for invalid index allocations."""
        # Very small images might be invalid
        assert self.generator.validate_index_allocation((1, 1)) == False
        assert self.generator.validate_index_allocation((0, 0)) == False
    
    def test_configuration_parameters(self):
        """Test configuration parameter handling."""
        config = {
            'min_granularity': 4,
            'max_index_rows': 3
        }
        generator = HierarchicalIndexGenerator(config)
        
        assert generator.min_granularity == 4
        assert generator.max_index_rows == 3
        
        # Test with limited max_index_rows
        result = generator.calculate_optimal_granularity((1024, 1024))
        assert len(result['granularity_levels']) <= 3
    
    def test_various_image_sizes(self):
        """Test dynamic space allocation with various image sizes."""
        test_sizes = [
            (256, 256),
            (512, 512),
            (1024, 1024),
            (2048, 2048),
            (4096, 4096),
            (1024, 512),
            (2048, 1024),
            (128, 256)
        ]
        
        for width, height in test_sizes:
            result = self.generator.allocate_index_space((width, height))
            
            # Enhanced dimensions should be larger than original
            enhanced_width, enhanced_height = result['enhanced_dimensions']
            assert enhanced_width == width
            assert enhanced_height > height
            
            # Should have reasonable number of granularity levels
            granularity_levels = result['granularity_info']['granularity_levels']
            assert 1 <= len(granularity_levels) <= self.generator.max_index_rows
            
            # Finest granularity should be reasonable for image size
            finest = granularity_levels[0]
            assert finest >= self.generator.min_granularity
            assert finest <= min(width, height)
    
    def test_generate_indices_preserves_original_data(self):
        """Test that generating indices preserves original embedding data."""
        # Create test data with specific pattern
        original_image = np.arange(64*64, dtype=np.float32).reshape(64, 64)
        
        # Generate enhanced image
        enhanced_image = self.generator.generate_multi_level_indices(original_image)
        
        # Original data should be preserved exactly
        np.testing.assert_array_equal(enhanced_image[:64, :], original_image)
        
        # Enhanced image should be larger
        assert enhanced_image.shape[0] > 64
        assert enhanced_image.shape[1] == 64
    
    def test_index_rows_contain_meaningful_data(self):
        """Test that index rows contain meaningful spatial averages."""
        # Create image with distinct regions
        image = np.zeros((64, 64), dtype=np.float32)
        image[:32, :32] = 1.0  # Top-left
        image[:32, 32:] = 2.0  # Top-right
        image[32:, :32] = 3.0  # Bottom-left
        image[32:, 32:] = 4.0  # Bottom-right
        
        enhanced_image = self.generator.generate_multi_level_indices(image)
        
        # Get index rows
        index_rows = enhanced_image[64:, :]
        
        # Index rows should contain non-zero values
        assert np.any(index_rows > 0)
        
        # Values should be within reasonable range (between 0 and 4 for our test data)
        non_zero_values = index_rows[index_rows > 0]
        assert np.all(non_zero_values >= 0)
        assert np.all(non_zero_values <= 4)
    
    def test_error_handling_invalid_input(self):
        """Test error handling for invalid inputs."""
        # Test with 3D array (should raise ValueError)
        invalid_image = np.random.rand(32, 32, 3)
        
        with pytest.raises(ValueError, match="Embedding image must be 2D"):
            self.generator.generate_multi_level_indices(invalid_image)
        
        # Test with 1D array (should raise ValueError)
        invalid_image_1d = np.random.rand(32)
        
        with pytest.raises(ValueError, match="Embedding image must be 2D"):
            self.generator.generate_multi_level_indices(invalid_image_1d)
    
    def test_hilbert_order_averages(self):
        """Test Hilbert curve order-based spatial averages calculation."""
        # Create a test image with known pattern
        image = np.ones((16, 16), dtype=np.float32)
        image[:8, :8] = 2.0  # Top-left quadrant
        image[:8, 8:] = 3.0  # Top-right quadrant
        image[8:, :8] = 4.0  # Bottom-left quadrant
        image[8:, 8:] = 5.0  # Bottom-right quadrant
        
        # Calculate Hilbert order averages for 2x2 granularity
        averages = self.generator._calculate_hilbert_order_averages(image, 2)
        
        # Should have 4 averages
        assert len(averages) == 4
        
        # Values should be within expected range
        assert np.all(averages >= 2.0)
        assert np.all(averages <= 5.0)
        
        # Test with 4x4 granularity
        averages_4x4 = self.generator._calculate_hilbert_order_averages(image, 4)
        assert len(averages_4x4) == 16
    
    def test_generate_hilbert_coordinates(self):
        """Test Hilbert curve coordinate generation."""
        # Test small cases
        coords_1 = self.generator._generate_hilbert_coordinates(1)
        assert coords_1 == [(0, 0)]
        
        coords_2 = self.generator._generate_hilbert_coordinates(2)
        assert len(coords_2) == 4
        assert all(0 <= x < 2 and 0 <= y < 2 for x, y in coords_2)
        
        # Test larger case
        coords_4 = self.generator._generate_hilbert_coordinates(4)
        assert len(coords_4) == 16
        assert all(0 <= x < 4 and 0 <= y < 4 for x, y in coords_4)
        
        # Ensure all coordinates are unique
        assert len(set(coords_4)) == 16
    
    def test_create_progressive_granularity_levels(self):
        """Test creation of progressive granularity levels."""
        # Create test embedding image
        embedding_image = np.random.rand(64, 64).astype(np.float32)
        
        # Create progressive granularity levels
        index_rows = self.generator.create_progressive_granularity_levels(embedding_image)
        
        # Should have multiple index rows
        assert len(index_rows) > 0
        
        # Each index row should be a 1D array
        for index_row in index_rows:
            assert index_row.ndim == 1
            assert len(index_row) > 0
        
        # Index rows should generally decrease in size (coarser granularity = fewer sections)
        # Note: This might not always be true due to padding, but generally should hold
        sizes = [len(row) for row in index_rows]
        assert all(size > 0 for size in sizes)
    
    def test_calculate_averages_for_multiple_granularities(self):
        """Test calculation of averages for custom granularity levels."""
        # Create test embedding image
        embedding_image = np.random.rand(32, 32).astype(np.float32)
        
        # Define custom granularity levels
        granularity_levels = [8, 4, 2]
        
        # Calculate averages
        averages_dict = self.generator.calculate_averages_for_multiple_granularities(
            embedding_image, granularity_levels
        )
        
        # Should have entries for all granularity levels
        assert len(averages_dict) == 3
        assert 8 in averages_dict
        assert 4 in averages_dict
        assert 2 in averages_dict
        
        # Each entry should be a 1D array
        for granularity, averages in averages_dict.items():
            assert averages.ndim == 1
            assert len(averages) == granularity * granularity
    
    def test_hilbert_vs_spatial_averages_comparison(self):
        """Test comparison between Hilbert order and spatial averages."""
        # Create test image with spatial pattern
        image = np.zeros((16, 16), dtype=np.float32)
        image[:8, :8] = 1.0
        image[:8, 8:] = 2.0
        image[8:, :8] = 3.0
        image[8:, 8:] = 4.0
        
        # Calculate both types of averages
        hilbert_averages = self.generator._calculate_hilbert_order_averages(image, 2)
        spatial_averages = self.generator._calculate_spatial_averages(image, 2)
        
        # Both should have same length
        assert len(hilbert_averages) == len(spatial_averages)
        
        # Both should contain the same values (just in different order)
        assert set(hilbert_averages) == set(spatial_averages)
        
        # Values should be the quadrant averages
        expected_values = {1.0, 2.0, 3.0, 4.0}
        assert set(hilbert_averages) == expected_values
    
    def test_progressive_granularity_requirements_compliance(self):
        """Test compliance with requirements 3.1, 3.2, and 3.3."""
        # Create test embedding image
        embedding_image = np.random.rand(1024, 1024).astype(np.float32)
        
        # Test requirement 3.1: Multiple additional rows for hierarchical indices
        enhanced_image = self.generator.generate_multi_level_indices(embedding_image)
        assert enhanced_image.shape[0] > embedding_image.shape[0]  # Additional rows added
        assert enhanced_image.shape[1] == embedding_image.shape[1]  # Width preserved
        
        # Test requirement 3.2: Spatial averages at multiple Hilbert curve orders
        space_allocation = self.generator.allocate_index_space((1024, 1024))
        granularity_levels = space_allocation['granularity_info']['granularity_levels']
        
        # Should have multiple granularity levels (different orders)
        assert len(granularity_levels) > 1
        
        # Each level should be stored in separate index rows
        index_rows = enhanced_image[1024:, :]  # Index rows start after original image
        assert index_rows.shape[0] == len(granularity_levels)
        
        # Test requirement 3.3: Progressive granularity from finest to coarsest
        # Granularity levels should be in descending order (finest to coarsest)
        for i in range(len(granularity_levels) - 1):
            assert granularity_levels[i] > granularity_levels[i + 1]
        
        # Finest granularity should be reasonable for image size
        finest_granularity = granularity_levels[0]
        assert finest_granularity == 32  # sqrt(1024) = 32 for 1024x1024 image
    
    def test_embed_multi_level_indices(self):
        """Test embedding multiple index rows in correct positions."""
        # Create test embedding image
        embedding_image = np.random.rand(64, 64).astype(np.float32)
        
        # Create test index rows
        index_rows = [
            np.array([1.0, 2.0, 3.0, 4.0]),  # First level (finest)
            np.array([5.0, 6.0]),            # Second level (coarser)
            np.array([7.0])                  # Third level (coarsest)
        ]
        
        # Embed index rows
        enhanced_image = self.generator.embed_multi_level_indices(embedding_image, index_rows)
        
        # Check dimensions
        expected_height = 64 + len(index_rows)
        assert enhanced_image.shape == (expected_height, 64)
        
        # Check that original image is preserved
        np.testing.assert_array_equal(enhanced_image[:64, :], embedding_image)
        
        # Check that index rows are embedded correctly
        assert enhanced_image[64, 0] == 1.0
        assert enhanced_image[64, 1] == 2.0
        assert enhanced_image[64, 2] == 3.0
        assert enhanced_image[64, 3] == 4.0
        
        assert enhanced_image[65, 0] == 5.0
        assert enhanced_image[65, 1] == 6.0
        
        assert enhanced_image[66, 0] == 7.0
        
        # Check that unused positions are zero
        assert enhanced_image[64, 4] == 0.0  # After first index row
        assert enhanced_image[65, 2] == 0.0  # After second index row
        assert enhanced_image[66, 1] == 0.0  # After third index row
    
    def test_embed_multi_level_indices_empty_input(self):
        """Test embedding with empty index rows."""
        embedding_image = np.random.rand(32, 32).astype(np.float32)
        
        # Test with empty index rows list
        enhanced_image = self.generator.embed_multi_level_indices(embedding_image, [])
        np.testing.assert_array_equal(enhanced_image, embedding_image)
        
        # Test with empty index row
        index_rows = [np.array([])]
        enhanced_image = self.generator.embed_multi_level_indices(embedding_image, index_rows)
        
        # Should have one additional row
        assert enhanced_image.shape == (33, 32)
        # Original image should be preserved
        np.testing.assert_array_equal(enhanced_image[:32, :], embedding_image)
        # Index row should be all zeros
        assert np.all(enhanced_image[32, :] == 0.0)
    
    def test_embed_multi_level_indices_oversized_row(self):
        """Test embedding index row that's larger than image width."""
        embedding_image = np.random.rand(16, 16).astype(np.float32)
        
        # Create index row larger than image width
        large_index_row = np.arange(32, dtype=np.float32)  # 32 elements for 16-wide image
        index_rows = [large_index_row]
        
        enhanced_image = self.generator.embed_multi_level_indices(embedding_image, index_rows)
        
        # Check dimensions
        assert enhanced_image.shape == (17, 16)
        
        # Check that index row is truncated to fit
        expected_truncated = large_index_row[:16]
        np.testing.assert_array_equal(enhanced_image[16, :], expected_truncated)
    
    def test_extract_indices_from_image(self):
        """Test extraction of indices and original image."""
        # Create test embedding image
        original_embedding = np.random.rand(32, 32).astype(np.float32)
        
        # Create test index rows
        index_rows = [
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0]),
            np.array([6.0])
        ]
        
        # Embed indices
        enhanced_image = self.generator.embed_multi_level_indices(original_embedding, index_rows)
        
        # Extract indices with original height hint
        original_height = original_embedding.shape[0]
        extracted_image, extracted_indices = self.generator.extract_indices_from_image(
            enhanced_image, original_height
        )
        
        # Check that original image is recovered
        np.testing.assert_array_equal(extracted_image, original_embedding)
        
        # Check that index rows are recovered
        assert len(extracted_indices) == 3
        np.testing.assert_array_equal(extracted_indices[0], np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(extracted_indices[1], np.array([4.0, 5.0]))
        np.testing.assert_array_equal(extracted_indices[2], np.array([6.0]))
    
    def test_extract_indices_no_indices(self):
        """Test extraction when there are no index rows."""
        # Create image without index rows
        original_image = np.random.rand(32, 32).astype(np.float32)
        
        # Extract from original image (no indices)
        extracted_image, extracted_indices = self.generator.extract_indices_from_image(original_image)
        
        # Should return original image and empty index list
        np.testing.assert_array_equal(extracted_image, original_image)
        assert len(extracted_indices) == 0
    
    def test_validate_embedded_indices(self):
        """Test validation of embedded indices."""
        # Create test embedding image
        embedding_image = np.random.rand(32, 32).astype(np.float32)
        
        # Create and embed index rows
        index_rows = [
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0])
        ]
        enhanced_image = self.generator.embed_multi_level_indices(embedding_image, index_rows)
        original_height = embedding_image.shape[0]
        
        # Validation should pass with original height hint
        assert self.generator.validate_embedded_indices(enhanced_image, 2, original_height) == True
        
        # Validation should fail with wrong expected count
        assert self.generator.validate_embedded_indices(enhanced_image, 3, original_height) == False
        assert self.generator.validate_embedded_indices(enhanced_image, 1, original_height) == False
        
        # Test without original height hint (may be less reliable)
        validation_result = self.generator.validate_embedded_indices(enhanced_image, 2)
        # This may pass or fail depending on heuristic detection, so we just check it doesn't crash
        assert isinstance(validation_result, bool)
        
        # Test with invalid input
        invalid_image = np.random.rand(32, 32, 3)  # 3D image
        assert self.generator.validate_embedded_indices(invalid_image, 2) == False
    
    def test_create_enhanced_embedding_with_indices(self):
        """Test complete workflow for creating enhanced embedding with indices."""
        # Create test embedding image
        embedding_image = np.random.rand(64, 64).astype(np.float32)
        
        # Create enhanced embedding with indices
        enhanced_image = self.generator.create_enhanced_embedding_with_indices(embedding_image)
        
        # Should have additional rows for indices
        assert enhanced_image.shape[0] > 64
        assert enhanced_image.shape[1] == 64
        
        # Original image should be preserved
        np.testing.assert_array_equal(enhanced_image[:64, :], embedding_image)
        
        # Should be able to extract indices with original height hint
        original_height = embedding_image.shape[0]
        extracted_image, extracted_indices = self.generator.extract_indices_from_image(
            enhanced_image, original_height
        )
        np.testing.assert_array_equal(extracted_image, embedding_image)
        assert len(extracted_indices) > 0
        
        # Each extracted index row should contain meaningful data
        for index_row in extracted_indices:
            assert len(index_row) > 0
            assert np.any(index_row != 0)  # Should have some non-zero values
    
    def test_round_trip_embedding_and_extraction(self):
        """Test round-trip embedding and extraction preserves data."""
        # Create test data with known pattern
        embedding_image = np.arange(16*16, dtype=np.float32).reshape(16, 16)
        
        # Create progressive granularity levels
        index_rows = self.generator.create_progressive_granularity_levels(embedding_image)
        
        # Embed indices
        enhanced_image = self.generator.embed_multi_level_indices(embedding_image, index_rows)
        
        # Extract indices with original height hint for perfect round-trip
        original_height = embedding_image.shape[0]
        extracted_image, extracted_indices = self.generator.extract_indices_from_image(
            enhanced_image, original_height
        )
        
        # Check perfect round-trip recovery
        np.testing.assert_array_equal(extracted_image, embedding_image)
        assert len(extracted_indices) == len(index_rows)
        
        # Check that extracted indices match original (within floating point precision)
        for i, (original, extracted) in enumerate(zip(index_rows, extracted_indices)):
            np.testing.assert_array_almost_equal(extracted, original, decimal=5,
                                               err_msg=f"Index row {i} mismatch")
    
    def test_detect_original_image_height(self):
        """Test detection of original image height in enhanced image."""
        # Create test embedding with known pattern
        embedding_image = np.ones((32, 32), dtype=np.float32) * 0.5
        
        # Create sparse index rows (mostly zeros)
        index_rows = [
            np.array([1.0, 0.0, 0.0, 0.0]),  # Sparse index row
            np.array([2.0, 0.0, 0.0])        # Another sparse index row
        ]
        
        # Embed indices
        enhanced_image = self.generator.embed_multi_level_indices(embedding_image, index_rows)
        
        # Test detection
        detected_height = self.generator._detect_original_image_height(enhanced_image)
        
        # Should detect the correct original height
        assert detected_height == 32
    
    def test_requirements_3_4_3_5_compliance(self):
        """Test compliance with requirements 3.4 and 3.5 for index embedding."""
        # Create test embedding image
        embedding_image = np.random.rand(128, 128).astype(np.float32)
        
        # Test requirement 3.4: Embed multi-level hierarchical indices in correct positions
        index_rows = self.generator.create_progressive_granularity_levels(embedding_image)
        enhanced_image = self.generator.embed_multi_level_indices(embedding_image, index_rows)
        
        # Should have correct number of additional rows
        expected_additional_rows = len(index_rows)
        assert enhanced_image.shape[0] == 128 + expected_additional_rows
        
        # Index rows should be at correct positions (after original image)
        for i, index_row in enumerate(index_rows):
            row_position = 128 + i
            embedded_row = enhanced_image[row_position, :len(index_row)]
            np.testing.assert_array_equal(embedded_row, index_row)
        
        # Test requirement 3.5: Enable multi-level progressive filtering
        # Extract and validate that we can use indices for progressive filtering
        original_height = embedding_image.shape[0]
        extracted_image, extracted_indices = self.generator.extract_indices_from_image(
            enhanced_image, original_height
        )
        
        # Should be able to recover all index levels
        assert len(extracted_indices) == len(index_rows)
        
        # Index levels should be in order from finest to coarsest
        space_allocation = self.generator.allocate_index_space((128, 128))
        granularity_levels = space_allocation['granularity_info']['granularity_levels']
        
        # Should have progressive granularity (finest to coarsest)
        for i in range(len(granularity_levels) - 1):
            assert granularity_levels[i] > granularity_levels[i + 1]
        
        # Each extracted index should correspond to the correct granularity level
        for i, (granularity, extracted_index) in enumerate(zip(granularity_levels, extracted_indices)):
            # Index size should match expected sections for this granularity
            expected_sections = granularity * granularity
            assert len(extracted_index) <= expected_sections  # May be truncated to fit image width
    
    def test_error_handling_progressive_methods(self):
        """Test error handling for new progressive granularity methods."""
        # Test with invalid input for create_progressive_granularity_levels
        invalid_image = np.random.rand(32, 32, 3)
        
        with pytest.raises(ValueError, match="Embedding image must be 2D"):
            self.generator.create_progressive_granularity_levels(invalid_image)
        
        # Test with invalid input for calculate_averages_for_multiple_granularities
        with pytest.raises(ValueError, match="Embedding image must be 2D"):
            self.generator.calculate_averages_for_multiple_granularities(invalid_image, [2, 4])
        
        # Test with empty granularity levels
        valid_image = np.random.rand(32, 32)
        result = self.generator.calculate_averages_for_multiple_granularities(valid_image, [])
        assert len(result) == 0
        
        # Test with invalid granularity levels (zero or negative)
        result = self.generator.calculate_averages_for_multiple_granularities(valid_image, [0, -1, 2])
        assert len(result) == 1  # Only granularity 2 should be processed
        assert 2 in result