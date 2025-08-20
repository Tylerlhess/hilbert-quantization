"""
Unit tests for Hilbert curve mapping functionality.
"""

import pytest
import numpy as np
from hilbert_quantization.core.hilbert_mapper import HilbertCurveMapper
from hilbert_quantization.exceptions import HilbertQuantizationError


class TestHilbertCurveMapper:
    """Test cases for HilbertCurveMapper class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mapper = HilbertCurveMapper()
    
    def test_generate_hilbert_coordinates_2x2(self):
        """Test Hilbert curve generation for 2x2 grid."""
        coordinates = self.mapper.generate_hilbert_coordinates(2)
        
        # 2x2 Hilbert curve should have specific order
        expected = [(0, 0), (0, 1), (1, 1), (1, 0)]
        assert coordinates == expected
        assert len(coordinates) == 4
    
    def test_generate_hilbert_coordinates_4x4(self):
        """Test Hilbert curve generation for 4x4 grid."""
        coordinates = self.mapper.generate_hilbert_coordinates(4)
        
        # Should have 16 coordinates
        assert len(coordinates) == 16
        
        # All coordinates should be unique
        assert len(set(coordinates)) == 16
        
        # All coordinates should be within bounds
        for x, y in coordinates:
            assert 0 <= x < 4
            assert 0 <= y < 4
        
        # First few coordinates should follow Hilbert pattern
        assert coordinates[0] == (0, 0)
        assert coordinates[1] == (1, 0)
        assert coordinates[2] == (1, 1)
        assert coordinates[3] == (0, 1)
    
    def test_generate_hilbert_coordinates_8x8(self):
        """Test Hilbert curve generation for 8x8 grid."""
        coordinates = self.mapper.generate_hilbert_coordinates(8)
        
        # Should have 64 coordinates
        assert len(coordinates) == 64
        
        # All coordinates should be unique
        assert len(set(coordinates)) == 64
        
        # All coordinates should be within bounds
        for x, y in coordinates:
            assert 0 <= x < 8
            assert 0 <= y < 8
    
    def test_generate_hilbert_coordinates_invalid_size(self):
        """Test error handling for invalid grid sizes."""
        # Non-power of 2
        with pytest.raises(HilbertQuantizationError, match="must be a power of 2"):
            self.mapper.generate_hilbert_coordinates(3)
        
        with pytest.raises(HilbertQuantizationError, match="must be a power of 2"):
            self.mapper.generate_hilbert_coordinates(5)
        
        with pytest.raises(HilbertQuantizationError, match="must be a power of 2"):
            self.mapper.generate_hilbert_coordinates(6)
        
        # Zero or negative
        with pytest.raises(HilbertQuantizationError, match="must be a power of 2"):
            self.mapper.generate_hilbert_coordinates(0)
        
        with pytest.raises(HilbertQuantizationError, match="must be a power of 2"):
            self.mapper.generate_hilbert_coordinates(-1)
    
    def test_spatial_locality_preservation(self):
        """Test that Hilbert curve preserves spatial locality."""
        coordinates = self.mapper.generate_hilbert_coordinates(4)
        
        # Calculate distances between consecutive points in Hilbert order
        distances = []
        for i in range(len(coordinates) - 1):
            x1, y1 = coordinates[i]
            x2, y2 = coordinates[i + 1]
            distance = abs(x2 - x1) + abs(y2 - y1)  # Manhattan distance
            distances.append(distance)
        
        # Most consecutive points should be adjacent (distance = 1)
        adjacent_count = sum(1 for d in distances if d == 1)
        total_transitions = len(distances)
        
        # At least 75% of transitions should be to adjacent cells
        locality_ratio = adjacent_count / total_transitions
        assert locality_ratio >= 0.75, f"Spatial locality too low: {locality_ratio:.2f}"
    
    def test_hilbert_index_conversion_consistency(self):
        """Test that index to coordinate conversion is consistent."""
        n = 4
        coordinates = self.mapper.generate_hilbert_coordinates(n)
        
        # Test that _hilbert_index_to_xy produces the same coordinates
        for i, expected_coord in enumerate(coordinates):
            actual_coord = self.mapper._hilbert_index_to_xy(i, n)
            assert actual_coord == expected_coord, f"Index {i}: expected {expected_coord}, got {actual_coord}"
    
    def test_coordinate_to_index_conversion(self):
        """Test coordinate to index conversion."""
        n = 4
        coordinates = self.mapper.generate_hilbert_coordinates(n)
        
        # Test that _xy_to_hilbert_index is inverse of _hilbert_index_to_xy
        for i, (x, y) in enumerate(coordinates):
            actual_index = self.mapper._xy_to_hilbert_index(x, y, n)
            assert actual_index == i, f"Coordinate ({x}, {y}): expected index {i}, got {actual_index}"
    
    def test_map_to_2d_basic(self):
        """Test basic 1D to 2D parameter mapping."""
        parameters = np.array([1.0, 2.0, 3.0, 4.0])
        result = self.mapper.map_to_2d(parameters, (2, 2))
        
        # Should create 2x2 array
        assert result.shape == (2, 2)
        
        # Values should be placed according to Hilbert curve order
        coordinates = self.mapper.generate_hilbert_coordinates(2)
        for i, param_value in enumerate(parameters):
            x, y = coordinates[i]
            assert result[y, x] == param_value
    
    def test_map_to_2d_with_padding(self):
        """Test 1D to 2D mapping with padding."""
        parameters = np.array([1.0, 2.0, 3.0])  # Only 3 parameters for 2x2 grid
        result = self.mapper.map_to_2d(parameters, (2, 2))
        
        # Should create 2x2 array
        assert result.shape == (2, 2)
        
        # First 3 positions should have parameter values
        coordinates = self.mapper.generate_hilbert_coordinates(2)
        for i, param_value in enumerate(parameters):
            x, y = coordinates[i]
            assert result[y, x] == param_value
        
        # Last position should be padded with zero
        x, y = coordinates[3]
        assert result[y, x] == 0.0
    
    def test_map_to_2d_invalid_dimensions(self):
        """Test error handling for invalid dimensions."""
        parameters = np.array([1.0, 2.0, 3.0, 4.0])
        
        # Non-square dimensions
        with pytest.raises(HilbertQuantizationError, match="requires square dimensions"):
            self.mapper.map_to_2d(parameters, (2, 3))
        
        # Non-power of 2 dimensions
        with pytest.raises(HilbertQuantizationError, match="must be a power of 2"):
            self.mapper.map_to_2d(parameters, (3, 3))
        
        # Too many parameters
        with pytest.raises(HilbertQuantizationError, match="Too many parameters"):
            self.mapper.map_to_2d(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), (2, 2))
    
    def test_map_from_2d_basic(self):
        """Test basic 2D to 1D parameter reconstruction."""
        # Create a 2x2 image with known values
        image = np.array([[1.0, 4.0],
                         [2.0, 3.0]])
        
        result = self.mapper.map_from_2d(image)
        
        # Should extract parameters in Hilbert curve order
        # Hilbert order for 2x2: (0,0), (0,1), (1,1), (1,0)
        # Which corresponds to: image[0,0], image[1,0], image[1,1], image[0,1]
        expected = np.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_equal(result, expected)
    
    def test_map_from_2d_invalid_dimensions(self):
        """Test error handling for invalid image dimensions."""
        # Non-square image
        image = np.array([[1.0, 2.0, 3.0],
                         [4.0, 5.0, 6.0]])
        
        with pytest.raises(HilbertQuantizationError, match="requires square dimensions"):
            self.mapper.map_from_2d(image)
        
        # Non-power of 2 dimensions
        image = np.array([[1.0, 2.0, 3.0],
                         [4.0, 5.0, 6.0],
                         [7.0, 8.0, 9.0]])
        
        with pytest.raises(HilbertQuantizationError, match="must be a power of 2"):
            self.mapper.map_from_2d(image)
    
    def test_round_trip_mapping_accuracy(self):
        """Test that mapping to 2D and back preserves parameter values."""
        # Test with various parameter arrays
        test_cases = [
            np.array([1.0, 2.0, 3.0, 4.0]),  # Exact fit for 2x2
            np.array([1.5, -2.3, 0.0, 4.7, 8.1, 9.9]),  # Partial fill for 4x4
            np.random.randn(16),  # Full 4x4 with random values
            np.random.randn(10),  # Partial 4x4 with random values
        ]
        
        for parameters in test_cases:
            # Determine appropriate dimensions
            param_count = len(parameters)
            if param_count <= 4:
                dimensions = (2, 2)
            elif param_count <= 16:
                dimensions = (4, 4)
            else:
                dimensions = (8, 8)
            
            # Round trip: parameters -> 2D -> parameters
            image = self.mapper.map_to_2d(parameters, dimensions)
            reconstructed = self.mapper.map_from_2d(image)
            
            # Original parameters should be preserved (first N elements)
            np.testing.assert_array_equal(
                reconstructed[:len(parameters)], 
                parameters,
                err_msg=f"Round trip failed for parameters: {parameters}"
            )
    
    def test_spatial_locality_in_mapping(self):
        """Test that spatial locality is preserved in parameter mapping."""
        # Create parameters with spatial structure
        parameters = np.array([1.0, 1.1, 1.2, 1.3,  # Similar values
                              2.0, 2.1, 2.2, 2.3,  # Another group
                              3.0, 3.1, 3.2, 3.3,  # Third group
                              4.0, 4.1, 4.2, 4.3]) # Fourth group
        
        image = self.mapper.map_to_2d(parameters, (4, 4))
        
        # Check that similar parameter values are placed near each other
        # This is a heuristic test - we expect some spatial clustering
        coordinates = self.mapper.generate_hilbert_coordinates(4)
        
        # Group parameters by their integer part
        groups = {}
        for i, param in enumerate(parameters):
            group = int(param)
            if group not in groups:
                groups[group] = []
            groups[group].append(coordinates[i])
        
        # For each group, calculate average distance between coordinates
        for group, coords in groups.items():
            if len(coords) > 1:
                total_distance = 0
                pair_count = 0
                for i in range(len(coords)):
                    for j in range(i + 1, len(coords)):
                        x1, y1 = coords[i]
                        x2, y2 = coords[j]
                        distance = abs(x2 - x1) + abs(y2 - y1)
                        total_distance += distance
                        pair_count += 1
                
                avg_distance = total_distance / pair_count
                # Similar parameters should be relatively close (average distance < 4)
                assert avg_distance < 4.0, f"Group {group} has poor spatial locality: {avg_distance:.2f}" 
   
    def test_parameter_to_2d_mapping_with_various_sizes(self):
        """Test parameter to 2D mapping with various parameter array sizes."""
        # Test different parameter counts and their mapping
        test_cases = [
            (np.array([1.0]), (2, 2)),  # Single parameter
            (np.array([1.0, 2.0]), (2, 2)),  # Two parameters
            (np.array([1.0, 2.0, 3.0]), (2, 2)),  # Three parameters
            (np.array(range(1, 5)), (2, 2)),  # Exact fit
            (np.array(range(1, 10)), (4, 4)),  # Partial fill for 4x4
            (np.array(range(1, 17)), (4, 4)),  # Exact fit for 4x4
        ]
        
        for parameters, dimensions in test_cases:
            result = self.mapper.map_to_2d(parameters, dimensions)
            
            # Check dimensions
            assert result.shape == dimensions
            
            # Check that parameters are placed correctly
            coordinates = self.mapper.generate_hilbert_coordinates(dimensions[0])
            for i, param_value in enumerate(parameters):
                x, y = coordinates[i]
                assert result[y, x] == param_value
            
            # Check that unused positions are zero (padding)
            for i in range(len(parameters), len(coordinates)):
                x, y = coordinates[i]
                assert result[y, x] == 0.0
    
    def test_padding_for_non_square_parameter_counts(self):
        """Test that padding is handled correctly for non-square parameter counts."""
        # Test with parameter count that doesn't fill the grid completely
        parameters = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # 5 parameters for 4x4 grid
        result = self.mapper.map_to_2d(parameters, (4, 4))
        
        # Should be 4x4
        assert result.shape == (4, 4)
        
        # First 5 positions should have parameter values
        coordinates = self.mapper.generate_hilbert_coordinates(4)
        for i, param_value in enumerate(parameters):
            x, y = coordinates[i]
            assert result[y, x] == param_value
        
        # Remaining positions should be padded with zeros
        for i in range(len(parameters), len(coordinates)):
            x, y = coordinates[i]
            assert result[y, x] == 0.0
        
        # Count non-zero elements
        non_zero_count = np.count_nonzero(result)
        assert non_zero_count == len(parameters)
    
    def test_spatial_locality_preservation_detailed(self):
        """Detailed test for spatial locality preservation in parameter mapping."""
        # Create parameters with clear spatial structure
        # Group 1: values 1.0-1.9 (should be clustered)
        # Group 2: values 2.0-2.9 (should be clustered)
        # etc.
        parameters = []
        for group in range(1, 5):  # 4 groups
            for i in range(4):  # 4 parameters per group
                parameters.append(group + i * 0.1)
        
        parameters = np.array(parameters)  # 16 parameters total
        result = self.mapper.map_to_2d(parameters, (4, 4))
        
        # Analyze spatial clustering
        coordinates = self.mapper.generate_hilbert_coordinates(4)
        
        # For each group, find the coordinates of its parameters
        group_coords = {1: [], 2: [], 3: [], 4: []}
        for i, param in enumerate(parameters):
            group = int(param)
            group_coords[group].append(coordinates[i])
        
        # Calculate compactness for each group
        for group, coords in group_coords.items():
            # Calculate the bounding box area for each group
            if len(coords) > 1:
                xs = [x for x, y in coords]
                ys = [y for x, y in coords]
                bbox_area = (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1)
                
                # The bounding box should be reasonably compact
                # For 4 points, ideal bbox area would be 4, acceptable up to 8
                assert bbox_area <= 8, f"Group {group} is too spread out: bbox area {bbox_area}"
    
    def test_different_data_types(self):
        """Test parameter mapping with different numpy data types."""
        # Test with different data types
        test_cases = [
            np.array([1, 2, 3, 4], dtype=np.int32),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        ]
        
        for parameters in test_cases:
            result = self.mapper.map_to_2d(parameters, (2, 2))
            
            # Result should maintain the same dtype
            assert result.dtype == parameters.dtype
            
            # Values should be preserved
            coordinates = self.mapper.generate_hilbert_coordinates(2)
            for i, param_value in enumerate(parameters):
                x, y = coordinates[i]
                assert result[y, x] == param_value    

    def test_inverse_2d_to_parameter_mapping_bijective(self):
        """Test that 2D to parameter mapping is bijective with forward mapping."""
        # Test various parameter arrays
        test_cases = [
            np.array([1.0, 2.0, 3.0, 4.0]),  # 2x2 exact fit
            np.array([1.5, -2.3, 0.0, 4.7]),  # 2x2 with negative and zero
            np.array(range(1, 17), dtype=float),  # 4x4 exact fit
            np.array([i * 0.1 for i in range(10)]),  # 4x4 partial with decimals
        ]
        
        for original_params in test_cases:
            # Determine appropriate dimensions
            param_count = len(original_params)
            if param_count <= 4:
                dimensions = (2, 2)
            else:
                dimensions = (4, 4)
            
            # Forward mapping: parameters -> 2D
            image = self.mapper.map_to_2d(original_params, dimensions)
            
            # Inverse mapping: 2D -> parameters
            reconstructed_params = self.mapper.map_from_2d(image)
            
            # The first N elements should exactly match original parameters
            np.testing.assert_array_equal(
                reconstructed_params[:len(original_params)],
                original_params,
                err_msg=f"Bijective mapping failed for: {original_params}"
            )
            
            # Remaining elements should be zeros (padding)
            padding_elements = reconstructed_params[len(original_params):]
            np.testing.assert_array_equal(
                padding_elements,
                np.zeros_like(padding_elements),
                err_msg="Padding elements should be zero"
            )
    
    def test_round_trip_mapping_accuracy_comprehensive(self):
        """Comprehensive test for round-trip mapping accuracy."""
        # Test with edge cases and various data patterns
        test_cases = [
            # Edge values
            np.array([0.0, 1.0, -1.0, 1e-10]),
            np.array([1e6, -1e6, np.pi, np.e]),
            
            # Random patterns
            np.random.randn(4),
            np.random.randn(16),
            np.random.uniform(-100, 100, 9),
            
            # Structured patterns
            np.array([i**2 for i in range(8)], dtype=float),
            np.array([np.sin(i) for i in range(12)]),
        ]
        
        for original_params in test_cases:
            # Determine appropriate dimensions
            param_count = len(original_params)
            if param_count <= 4:
                dimensions = (2, 2)
            elif param_count <= 16:
                dimensions = (4, 4)
            else:
                dimensions = (8, 8)
            
            # Perform round trip
            image = self.mapper.map_to_2d(original_params, dimensions)
            reconstructed = self.mapper.map_from_2d(image)
            
            # Check accuracy with appropriate tolerance for floating point
            np.testing.assert_allclose(
                reconstructed[:len(original_params)],
                original_params,
                rtol=1e-15,  # Very tight tolerance for exact reconstruction
                atol=1e-15,
                err_msg=f"Round trip accuracy failed for: {original_params}"
            )
    
    def test_inverse_mapping_preserves_structure(self):
        """Test that inverse mapping preserves the structure of the original parameters."""
        # Create a structured parameter array
        original_params = np.array([
            1.0, 1.1, 1.2, 1.3,  # Group 1
            2.0, 2.1, 2.2, 2.3,  # Group 2
            3.0, 3.1, 3.2, 3.3,  # Group 3
            4.0, 4.1, 4.2, 4.3   # Group 4
        ])
        
        # Map to 2D and back
        image = self.mapper.map_to_2d(original_params, (4, 4))
        reconstructed = self.mapper.map_from_2d(image)
        
        # Check that the structure is preserved
        np.testing.assert_array_equal(
            reconstructed[:len(original_params)],
            original_params
        )
        
        # Verify that the grouping structure is maintained
        for i in range(4):  # 4 groups
            group_start = i * 4
            group_end = group_start + 4
            original_group = original_params[group_start:group_end]
            reconstructed_group = reconstructed[group_start:group_end]
            
            np.testing.assert_array_equal(
                reconstructed_group,
                original_group,
                err_msg=f"Group {i} structure not preserved"
            )
    
    def test_inverse_mapping_with_different_dtypes(self):
        """Test inverse mapping with different numpy data types."""
        test_cases = [
            (np.array([1, 2, 3, 4], dtype=np.int32), (2, 2)),
            (np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32), (2, 2)),
            (np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64), (2, 2)),
            (np.array(range(16), dtype=np.int64), (4, 4)),
        ]
        
        for original_params, dimensions in test_cases:
            # Forward and inverse mapping
            image = self.mapper.map_to_2d(original_params, dimensions)
            reconstructed = self.mapper.map_from_2d(image)
            
            # Check dtype preservation
            assert reconstructed.dtype == image.dtype
            
            # Check value preservation
            np.testing.assert_array_equal(
                reconstructed[:len(original_params)],
                original_params.astype(image.dtype)
            )
    
    def test_bijective_property_mathematical(self):
        """Mathematical test for bijective property of the mapping."""
        # For a bijective function f: A -> B, we need:
        # 1. f is injective (one-to-one): if f(a1) = f(a2), then a1 = a2
        # 2. f is surjective (onto): for every b in B, there exists a in A such that f(a) = b
        # 3. f has an inverse: f^(-1)(f(a)) = a for all a in A
        
        # Test the inverse property: f^(-1)(f(a)) = a
        dimensions = (4, 4)
        total_positions = dimensions[0] * dimensions[1]
        
        # Create test parameter arrays of different sizes
        for param_count in [1, 4, 8, 12, 16]:
            if param_count > total_positions:
                continue
                
            # Create unique parameter values to test injectivity
            original_params = np.array([i + 0.1 * i for i in range(param_count)])
            
            # Apply forward and inverse mapping
            image = self.mapper.map_to_2d(original_params, dimensions)
            reconstructed = self.mapper.map_from_2d(image)
            
            # Test inverse property: f^(-1)(f(a)) = a
            np.testing.assert_array_equal(
                reconstructed[:param_count],
                original_params,
                err_msg=f"Inverse property failed for {param_count} parameters"
            )
            
            # Test that unused positions are consistently zero
            if param_count < total_positions:
                unused_positions = reconstructed[param_count:]
                np.testing.assert_array_equal(
                    unused_positions,
                    np.zeros_like(unused_positions),
                    err_msg="Unused positions should be zero"
                )