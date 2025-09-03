"""
Unit tests for RAG Hilbert curve coordinate generation functionality.
"""

import pytest
import numpy as np
from hilbert_quantization.rag.embedding_generation.hilbert_mapper import HilbertCurveMapperImpl
from hilbert_quantization.rag.config import RAGConfig


class TestRAGHilbertCurveMapper:
    """Test cases for RAG HilbertCurveMapperImpl coordinate generation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = RAGConfig()
        self.mapper = HilbertCurveMapperImpl(self.config)
    
    def test_generate_hilbert_coordinates_2x2(self):
        """Test Hilbert curve coordinate generation for 2x2 grid."""
        coordinates = self.mapper.generate_hilbert_coordinates(2)
        
        # 2x2 Hilbert curve should have specific order
        expected = [(0, 0), (0, 1), (1, 1), (1, 0)]
        assert coordinates == expected
        assert len(coordinates) == 4
    
    def test_generate_hilbert_coordinates_4x4(self):
        """Test Hilbert curve coordinate generation for 4x4 grid."""
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
        """Test Hilbert curve coordinate generation for 8x8 grid."""
        coordinates = self.mapper.generate_hilbert_coordinates(8)
        
        # Should have 64 coordinates
        assert len(coordinates) == 64
        
        # All coordinates should be unique
        assert len(set(coordinates)) == 64
        
        # All coordinates should be within bounds
        for x, y in coordinates:
            assert 0 <= x < 8
            assert 0 <= y < 8
    
    def test_generate_hilbert_coordinates_16x16(self):
        """Test Hilbert curve coordinate generation for 16x16 grid for embeddings."""
        coordinates = self.mapper.generate_hilbert_coordinates(16)
        
        # Should have 256 coordinates
        assert len(coordinates) == 256
        
        # All coordinates should be unique
        assert len(set(coordinates)) == 256
        
        # All coordinates should be within bounds
        for x, y in coordinates:
            assert 0 <= x < 16
            assert 0 <= y < 16
    
    def test_generate_hilbert_coordinates_32x32(self):
        """Test Hilbert curve coordinate generation for 32x32 grid for large embeddings."""
        coordinates = self.mapper.generate_hilbert_coordinates(32)
        
        # Should have 1024 coordinates
        assert len(coordinates) == 1024
        
        # All coordinates should be unique
        assert len(set(coordinates)) == 1024
        
        # All coordinates should be within bounds
        for x, y in coordinates:
            assert 0 <= x < 32
            assert 0 <= y < 32
    
    def test_generate_hilbert_coordinates_invalid_size(self):
        """Test error handling for invalid grid sizes."""
        # Non-power of 2
        with pytest.raises(ValueError, match="must be a power of 2"):
            self.mapper.generate_hilbert_coordinates(3)
        
        with pytest.raises(ValueError, match="must be a power of 2"):
            self.mapper.generate_hilbert_coordinates(5)
        
        with pytest.raises(ValueError, match="must be a power of 2"):
            self.mapper.generate_hilbert_coordinates(6)
        
        with pytest.raises(ValueError, match="must be a power of 2"):
            self.mapper.generate_hilbert_coordinates(7)
        
        with pytest.raises(ValueError, match="must be a power of 2"):
            self.mapper.generate_hilbert_coordinates(9)
        
        # Zero or negative
        with pytest.raises(ValueError, match="must be a power of 2"):
            self.mapper.generate_hilbert_coordinates(0)
        
        with pytest.raises(ValueError, match="must be a power of 2"):
            self.mapper.generate_hilbert_coordinates(-1)
        
        with pytest.raises(ValueError, match="must be a power of 2"):
            self.mapper.generate_hilbert_coordinates(-4)
    
    def test_spatial_locality_preservation_2x2(self):
        """Test spatial locality preservation for 2x2 grid."""
        coordinates = self.mapper.generate_hilbert_coordinates(2)
        
        # Calculate distances between consecutive points in Hilbert order
        distances = []
        for i in range(len(coordinates) - 1):
            x1, y1 = coordinates[i]
            x2, y2 = coordinates[i + 1]
            distance = abs(x2 - x1) + abs(y2 - y1)  # Manhattan distance
            distances.append(distance)
        
        # For 2x2, all transitions should be to adjacent cells (distance = 1)
        assert all(d == 1 for d in distances), f"Non-adjacent transitions found: {distances}"
    
    def test_spatial_locality_preservation_4x4(self):
        """Test spatial locality preservation for 4x4 grid."""
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
    
    def test_spatial_locality_preservation_8x8(self):
        """Test spatial locality preservation for 8x8 grid."""
        coordinates = self.mapper.generate_hilbert_coordinates(8)
        
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
        
        # At least 70% of transitions should be to adjacent cells for larger grids
        locality_ratio = adjacent_count / total_transitions
        assert locality_ratio >= 0.70, f"Spatial locality too low: {locality_ratio:.2f}"
    
    def test_spatial_locality_preservation_16x16(self):
        """Test spatial locality preservation for 16x16 grid (embedding size)."""
        coordinates = self.mapper.generate_hilbert_coordinates(16)
        
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
        
        # At least 65% of transitions should be to adjacent cells for larger grids
        locality_ratio = adjacent_count / total_transitions
        assert locality_ratio >= 0.65, f"Spatial locality too low: {locality_ratio:.2f}"
        
        # No transition should be too far (max distance should be reasonable)
        max_distance = max(distances)
        assert max_distance <= 4, f"Maximum transition distance too large: {max_distance}"
    
    def test_hilbert_index_conversion_consistency_2x2(self):
        """Test that index to coordinate conversion is consistent for 2x2."""
        n = 2
        coordinates = self.mapper.generate_hilbert_coordinates(n)
        
        # Test that _hilbert_index_to_xy produces the same coordinates
        for i, expected_coord in enumerate(coordinates):
            actual_coord = self.mapper._hilbert_index_to_xy(i, n)
            assert actual_coord == expected_coord, f"Index {i}: expected {expected_coord}, got {actual_coord}"
    
    def test_hilbert_index_conversion_consistency_4x4(self):
        """Test that index to coordinate conversion is consistent for 4x4."""
        n = 4
        coordinates = self.mapper.generate_hilbert_coordinates(n)
        
        # Test that _hilbert_index_to_xy produces the same coordinates
        for i, expected_coord in enumerate(coordinates):
            actual_coord = self.mapper._hilbert_index_to_xy(i, n)
            assert actual_coord == expected_coord, f"Index {i}: expected {expected_coord}, got {actual_coord}"
    
    def test_hilbert_index_conversion_consistency_8x8(self):
        """Test that index to coordinate conversion is consistent for 8x8."""
        n = 8
        coordinates = self.mapper.generate_hilbert_coordinates(n)
        
        # Test that _hilbert_index_to_xy produces the same coordinates
        for i, expected_coord in enumerate(coordinates):
            actual_coord = self.mapper._hilbert_index_to_xy(i, n)
            assert actual_coord == expected_coord, f"Index {i}: expected {expected_coord}, got {actual_coord}"
    
    def test_coordinate_to_index_conversion_2x2(self):
        """Test coordinate to index conversion for 2x2."""
        n = 2
        coordinates = self.mapper.generate_hilbert_coordinates(n)
        
        # Test that _xy_to_hilbert_index is inverse of _hilbert_index_to_xy
        for i, (x, y) in enumerate(coordinates):
            actual_index = self.mapper._xy_to_hilbert_index(x, y, n)
            assert actual_index == i, f"Coordinate ({x}, {y}): expected index {i}, got {actual_index}"
    
    def test_coordinate_to_index_conversion_4x4(self):
        """Test coordinate to index conversion for 4x4."""
        n = 4
        coordinates = self.mapper.generate_hilbert_coordinates(n)
        
        # Test that _xy_to_hilbert_index is inverse of _hilbert_index_to_xy
        for i, (x, y) in enumerate(coordinates):
            actual_index = self.mapper._xy_to_hilbert_index(x, y, n)
            assert actual_index == i, f"Coordinate ({x}, {y}): expected index {i}, got {actual_index}"
    
    def test_coordinate_to_index_conversion_8x8(self):
        """Test coordinate to index conversion for 8x8."""
        n = 8
        coordinates = self.mapper.generate_hilbert_coordinates(n)
        
        # Test that _xy_to_hilbert_index is inverse of _hilbert_index_to_xy
        for i, (x, y) in enumerate(coordinates):
            actual_index = self.mapper._xy_to_hilbert_index(x, y, n)
            assert actual_index == i, f"Coordinate ({x}, {y}): expected index {i}, got {actual_index}"
    
    def test_bijective_property_2x2(self):
        """Test bijective property of coordinate mapping for 2x2."""
        n = 2
        
        # Test forward and inverse mapping for all positions
        for i in range(n * n):
            # Forward: index -> coordinates
            x, y = self.mapper._hilbert_index_to_xy(i, n)
            
            # Inverse: coordinates -> index
            recovered_index = self.mapper._xy_to_hilbert_index(x, y, n)
            
            assert recovered_index == i, f"Bijective property failed for index {i}"
    
    def test_bijective_property_4x4(self):
        """Test bijective property of coordinate mapping for 4x4."""
        n = 4
        
        # Test forward and inverse mapping for all positions
        for i in range(n * n):
            # Forward: index -> coordinates
            x, y = self.mapper._hilbert_index_to_xy(i, n)
            
            # Inverse: coordinates -> index
            recovered_index = self.mapper._xy_to_hilbert_index(x, y, n)
            
            assert recovered_index == i, f"Bijective property failed for index {i}"
    
    def test_bijective_property_8x8(self):
        """Test bijective property of coordinate mapping for 8x8."""
        n = 8
        
        # Test forward and inverse mapping for all positions
        for i in range(n * n):
            # Forward: index -> coordinates
            x, y = self.mapper._hilbert_index_to_xy(i, n)
            
            # Inverse: coordinates -> index
            recovered_index = self.mapper._xy_to_hilbert_index(x, y, n)
            
            assert recovered_index == i, f"Bijective property failed for index {i}"
    
    def test_coordinate_sequence_correctness_2x2(self):
        """Test correctness of coordinate sequence for 2x2 Hilbert curve."""
        coordinates = self.mapper.generate_hilbert_coordinates(2)
        
        # Known correct sequence for 2x2 Hilbert curve
        expected_sequence = [(0, 0), (0, 1), (1, 1), (1, 0)]
        
        assert coordinates == expected_sequence, f"Incorrect sequence: {coordinates}"
    
    def test_coordinate_sequence_correctness_4x4(self):
        """Test correctness of coordinate sequence for 4x4 Hilbert curve."""
        coordinates = self.mapper.generate_hilbert_coordinates(4)
        
        # Known correct first 8 coordinates for 4x4 Hilbert curve
        expected_start = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 2), (0, 3), (1, 3), (1, 2)]
        
        assert coordinates[:8] == expected_start, f"Incorrect start sequence: {coordinates[:8]}"
        
        # Check that we visit all positions exactly once
        assert len(coordinates) == 16
        assert len(set(coordinates)) == 16
    
    def test_embedding_dimension_compatibility(self):
        """Test coordinate generation for typical embedding dimensions."""
        # Test common embedding dimensions with appropriate grid sizes
        embedding_sizes = [
            (64, 8),     # 64-dim embedding -> 8x8 grid (64 used exactly)
            (128, 16),   # 128-dim embedding -> 16x16 grid (128 used)
            (256, 16),   # 256-dim embedding -> 16x16 grid (256 used exactly)
            (512, 32),   # 512-dim embedding -> 32x32 grid (512 used)
            (768, 32),   # 768-dim embedding -> 32x32 grid (768 used)
            (1024, 32),  # 1024-dim embedding -> 32x32 grid (1024 used exactly)
        ]
        
        for embedding_dim, grid_size in embedding_sizes:
            coordinates = self.mapper.generate_hilbert_coordinates(grid_size)
            
            # Should generate enough coordinates for the embedding
            assert len(coordinates) >= embedding_dim, f"Not enough coordinates for {embedding_dim}-dim embedding"
            
            # All coordinates should be valid
            for x, y in coordinates:
                assert 0 <= x < grid_size
                assert 0 <= y < grid_size
    
    def test_spatial_clustering_for_embeddings(self):
        """Test that similar embedding indices get clustered spatially."""
        # Test with 8x8 grid (64 positions)
        coordinates = self.mapper.generate_hilbert_coordinates(8)
        
        # Group consecutive indices and check their spatial clustering
        group_size = 4
        for start_idx in range(0, len(coordinates) - group_size, group_size):
            group_coords = coordinates[start_idx:start_idx + group_size]
            
            # Calculate bounding box for this group
            xs = [x for x, y in group_coords]
            ys = [y for x, y in group_coords]
            
            bbox_width = max(xs) - min(xs) + 1
            bbox_height = max(ys) - min(ys) + 1
            bbox_area = bbox_width * bbox_height
            
            # Consecutive indices should form compact clusters
            # For 4 consecutive points, bbox area should be reasonable
            assert bbox_area <= 8, f"Group {start_idx//group_size} too spread: area {bbox_area}"
    
    def test_coordinate_generation_performance(self):
        """Test that coordinate generation performs well for large grids."""
        import time
        
        # Test performance for embedding-relevant sizes
        test_sizes = [16, 32, 64]
        
        for n in test_sizes:
            start_time = time.time()
            coordinates = self.mapper.generate_hilbert_coordinates(n)
            end_time = time.time()
            
            generation_time = end_time - start_time
            
            # Should generate coordinates quickly (less than 1 second for reasonable sizes)
            assert generation_time < 1.0, f"Coordinate generation too slow for {n}x{n}: {generation_time:.3f}s"
            
            # Verify correctness
            assert len(coordinates) == n * n
            assert len(set(coordinates)) == n * n
    
    def test_coordinate_determinism(self):
        """Test that coordinate generation is deterministic."""
        # Generate coordinates multiple times and ensure they're identical
        n = 8
        
        coords1 = self.mapper.generate_hilbert_coordinates(n)
        coords2 = self.mapper.generate_hilbert_coordinates(n)
        coords3 = self.mapper.generate_hilbert_coordinates(n)
        
        assert coords1 == coords2 == coords3, "Coordinate generation is not deterministic"
    
    def test_coordinate_bounds_checking(self):
        """Test that all generated coordinates are within valid bounds."""
        test_sizes = [2, 4, 8, 16, 32]
        
        for n in test_sizes:
            coordinates = self.mapper.generate_hilbert_coordinates(n)
            
            for i, (x, y) in enumerate(coordinates):
                assert 0 <= x < n, f"X coordinate {x} out of bounds [0, {n}) at index {i}"
                assert 0 <= y < n, f"Y coordinate {y} out of bounds [0, {n}) at index {i}"
    
    def test_coordinate_completeness(self):
        """Test that coordinate generation covers all grid positions exactly once."""
        test_sizes = [2, 4, 8, 16]
        
        for n in test_sizes:
            coordinates = self.mapper.generate_hilbert_coordinates(n)
            
            # Should have exactly n*n coordinates
            assert len(coordinates) == n * n, f"Wrong number of coordinates for {n}x{n}"
            
            # Should cover all positions exactly once
            expected_positions = {(x, y) for x in range(n) for y in range(n)}
            actual_positions = set(coordinates)
            
            assert actual_positions == expected_positions, f"Missing or duplicate positions for {n}x{n}"
    
    def test_embedding_spatial_locality_simulation(self):
        """Test spatial locality preservation using simulated embedding similarity."""
        # Create a simulated scenario where similar embeddings should be close
        n = 8
        coordinates = self.mapper.generate_hilbert_coordinates(n)
        
        # Simulate embedding similarity: embeddings with close indices are similar
        similarity_threshold = 5  # Embeddings within 5 indices are considered similar
        
        for i in range(len(coordinates) - similarity_threshold):
            # Get coordinates for similar embeddings
            similar_coords = coordinates[i:i + similarity_threshold]
            
            # Calculate spatial spread
            xs = [x for x, y in similar_coords]
            ys = [y for x, y in similar_coords]
            
            spatial_spread = (max(xs) - min(xs)) + (max(ys) - min(ys))
            
            # Similar embeddings should have low spatial spread
            max_allowed_spread = n // 2  # Allow reasonable spread
            assert spatial_spread <= max_allowed_spread, f"Similar embeddings too spread out: {spatial_spread}"
    
    def test_map_to_2d_basic_functionality(self):
        """Test basic functionality of map_to_2d method."""
        # Test with simple 2x2 embedding
        embeddings = np.array([1.0, 2.0, 3.0, 4.0])
        dimensions = (2, 2)
        
        result = self.mapper.map_to_2d(embeddings, dimensions)
        
        # Should be 2x2 array
        assert result.shape == (2, 2)
        
        # Values should be mapped according to Hilbert curve order
        coordinates = self.mapper.generate_hilbert_coordinates(2)
        for i, value in enumerate(embeddings):
            x, y = coordinates[i]
            assert result[y, x] == value, f"Value {value} not at expected position ({x}, {y})"
    
    def test_map_to_2d_with_padding(self):
        """Test map_to_2d with fewer embeddings than grid cells (padding)."""
        # Test with 3 embeddings in 2x2 grid (1 cell should be padded)
        embeddings = np.array([1.5, 2.5, 3.5])
        dimensions = (2, 2)
        
        result = self.mapper.map_to_2d(embeddings, dimensions)
        
        # Should be 2x2 array
        assert result.shape == (2, 2)
        
        # First 3 values should be mapped according to Hilbert curve
        coordinates = self.mapper.generate_hilbert_coordinates(2)
        for i, value in enumerate(embeddings):
            x, y = coordinates[i]
            assert result[y, x] == value
        
        # Last position should be padded with zero
        last_x, last_y = coordinates[3]
        assert result[last_y, last_x] == 0.0, "Padding should be zero"
    
    def test_map_to_2d_4x4_grid(self):
        """Test map_to_2d with 4x4 grid for typical embedding size."""
        # Test with 16 embedding values
        embeddings = np.array([i * 0.1 for i in range(16)])
        dimensions = (4, 4)
        
        result = self.mapper.map_to_2d(embeddings, dimensions)
        
        # Should be 4x4 array
        assert result.shape == (4, 4)
        
        # All values should be mapped according to Hilbert curve order
        coordinates = self.mapper.generate_hilbert_coordinates(4)
        for i, value in enumerate(embeddings):
            x, y = coordinates[i]
            assert result[y, x] == value, f"Value {value} not at expected position ({x}, {y})"
    
    def test_map_to_2d_8x8_grid_with_padding(self):
        """Test map_to_2d with 8x8 grid and partial embeddings."""
        # Test with 50 embedding values in 8x8 grid (14 cells padded)
        embeddings = np.array([i * 0.01 for i in range(50)])
        dimensions = (8, 8)
        
        result = self.mapper.map_to_2d(embeddings, dimensions)
        
        # Should be 8x8 array
        assert result.shape == (8, 8)
        
        # First 50 values should be mapped
        coordinates = self.mapper.generate_hilbert_coordinates(8)
        for i, value in enumerate(embeddings):
            x, y = coordinates[i]
            assert result[y, x] == value
        
        # Remaining positions should be padded with zeros
        for i in range(50, 64):
            x, y = coordinates[i]
            assert result[y, x] == 0.0, f"Position ({x}, {y}) should be padded with zero"
    
    def test_map_to_2d_spatial_locality_preservation(self):
        """Test that spatial locality is preserved in 2D mapping."""
        # Create embeddings with gradual changes (simulating similar embeddings)
        embeddings = np.array([i * 0.1 for i in range(16)])
        dimensions = (4, 4)
        
        result = self.mapper.map_to_2d(embeddings, dimensions)
        coordinates = self.mapper.generate_hilbert_coordinates(4)
        
        # Check that consecutive embedding values are mapped to nearby positions
        for i in range(len(embeddings) - 1):
            x1, y1 = coordinates[i]
            x2, y2 = coordinates[i + 1]
            
            # Calculate Manhattan distance between consecutive positions
            distance = abs(x2 - x1) + abs(y2 - y1)
            
            # Most consecutive positions should be adjacent (distance = 1)
            # Allow some longer jumps for Hilbert curve transitions
            assert distance <= 3, f"Consecutive embeddings too far apart: distance {distance}"
    
    def test_map_to_2d_embedding_similarity_clustering(self):
        """Test that similar embedding values cluster spatially."""
        # Create embeddings with distinct groups of similar values
        embeddings = np.array([
            1.0, 1.1, 1.2, 1.3,  # Group 1: similar values
            5.0, 5.1, 5.2, 5.3,  # Group 2: similar values
            9.0, 9.1, 9.2, 9.3,  # Group 3: similar values
            2.0, 2.1, 2.2, 2.3   # Group 4: similar values
        ])
        dimensions = (4, 4)
        
        result = self.mapper.map_to_2d(embeddings, dimensions)
        coordinates = self.mapper.generate_hilbert_coordinates(4)
        
        # Check that the first 4 values (similar group) are spatially clustered
        group1_positions = []
        for i in range(4):
            x, y = coordinates[i]
            group1_positions.append((x, y))
        
        # Calculate bounding box for first group
        xs = [x for x, y in group1_positions]
        ys = [y for x, y in group1_positions]
        bbox_width = max(xs) - min(xs) + 1
        bbox_height = max(ys) - min(ys) + 1
        bbox_area = bbox_width * bbox_height
        
        # Similar values should form a compact cluster
        assert bbox_area <= 8, f"Similar embeddings not clustered: bbox area {bbox_area}"
    
    def test_map_to_2d_invalid_dimensions(self):
        """Test error handling for invalid dimensions."""
        embeddings = np.array([1.0, 2.0, 3.0, 4.0])
        
        # Non-square dimensions
        with pytest.raises(ValueError, match="requires square dimensions"):
            self.mapper.map_to_2d(embeddings, (2, 3))
        
        with pytest.raises(ValueError, match="requires square dimensions"):
            self.mapper.map_to_2d(embeddings, (4, 2))
        
        # Non-power of 2 dimensions
        with pytest.raises(ValueError, match="must be a power of 2"):
            self.mapper.map_to_2d(embeddings, (3, 3))
        
        with pytest.raises(ValueError, match="must be a power of 2"):
            self.mapper.map_to_2d(embeddings, (5, 5))
        
        # Zero or negative dimensions
        with pytest.raises(ValueError, match="must be positive"):
            self.mapper.map_to_2d(embeddings, (0, 0))
        
        with pytest.raises(ValueError, match="must be positive"):
            self.mapper.map_to_2d(embeddings, (-1, -1))
    
    def test_map_to_2d_too_many_embeddings(self):
        """Test error handling when embeddings exceed grid capacity."""
        # 5 embeddings for 2x2 grid (only 4 cells available)
        embeddings = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dimensions = (2, 2)
        
        with pytest.raises(ValueError, match="Too many embedding values"):
            self.mapper.map_to_2d(embeddings, dimensions)
    
    def test_map_to_2d_empty_embeddings(self):
        """Test map_to_2d with empty embeddings array."""
        embeddings = np.array([])
        dimensions = (2, 2)
        
        result = self.mapper.map_to_2d(embeddings, dimensions)
        
        # Should be 2x2 array filled with zeros (padding)
        assert result.shape == (2, 2)
        assert np.all(result == 0.0), "Empty embeddings should result in zero-filled array"
    
    def test_map_to_2d_data_type_preservation(self):
        """Test that data types are preserved in mapping."""
        # Test with different data types
        test_cases = [
            (np.float32, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)),
            (np.float64, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)),
            (np.int32, np.array([1, 2, 3, 4], dtype=np.int32)),
        ]
        
        dimensions = (2, 2)
        
        for expected_dtype, embeddings in test_cases:
            result = self.mapper.map_to_2d(embeddings, dimensions)
            
            assert result.dtype == expected_dtype, f"Data type not preserved: expected {expected_dtype}, got {result.dtype}"
            assert result.shape == (2, 2)
    
    def test_map_to_2d_large_embedding_dimensions(self):
        """Test map_to_2d with larger embedding dimensions typical for real models."""
        # Test with 256-dimensional embedding (16x16 grid)
        embeddings = np.random.rand(256).astype(np.float32)
        dimensions = (16, 16)
        
        result = self.mapper.map_to_2d(embeddings, dimensions)
        
        # Should be 16x16 array
        assert result.shape == (16, 16)
        assert result.dtype == np.float32
        
        # All embedding values should be present in the result
        coordinates = self.mapper.generate_hilbert_coordinates(16)
        for i, value in enumerate(embeddings):
            x, y = coordinates[i]
            assert result[y, x] == value
    
    def test_map_to_2d_partial_large_embedding(self):
        """Test map_to_2d with partial large embedding (common in practice)."""
        # Test with 200 values in 16x16 grid (56 cells padded)
        embeddings = np.random.rand(200).astype(np.float32)
        dimensions = (16, 16)
        
        result = self.mapper.map_to_2d(embeddings, dimensions)
        
        # Should be 16x16 array
        assert result.shape == (16, 16)
        
        # First 200 values should be mapped
        coordinates = self.mapper.generate_hilbert_coordinates(16)
        for i, value in enumerate(embeddings):
            x, y = coordinates[i]
            assert result[y, x] == value
        
        # Remaining 56 positions should be padded with zeros
        for i in range(200, 256):
            x, y = coordinates[i]
            assert result[y, x] == 0.0, f"Position ({x}, {y}) should be padded with zero"


class TestRAGHilbertCurveMapperInverse:
    """Test cases for inverse 2D to embedding mapping functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = RAGConfig()
        self.mapper = HilbertCurveMapperImpl(self.config)
    
    def test_map_from_2d_basic_functionality(self):
        """Test basic functionality of map_from_2d method."""
        # Create a simple 2x2 image
        image = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        result = self.mapper.map_from_2d(image)
        
        # Should reconstruct 1D array following Hilbert curve order
        # For 2x2: (0,0), (0,1), (1,1), (1,0) -> [1.0, 3.0, 4.0, 2.0]
        expected = np.array([1.0, 3.0, 4.0, 2.0])
        np.testing.assert_array_equal(result, expected)
    
    def test_map_from_2d_4x4_grid(self):
        """Test map_from_2d with 4x4 grid."""
        # Create a 4x4 image with sequential values
        image = np.arange(16, dtype=np.float32).reshape(4, 4)
        
        result = self.mapper.map_from_2d(image)
        
        # Should be 1D array with 16 values
        assert result.shape == (16,)
        assert result.dtype == np.float32
        
        # Values should follow Hilbert curve order
        coordinates = self.mapper.generate_hilbert_coordinates(4)
        expected = []
        for x, y in coordinates:
            expected.append(image[y, x])
        
        np.testing.assert_array_equal(result, np.array(expected, dtype=np.float32))
    
    def test_map_from_2d_8x8_grid(self):
        """Test map_from_2d with 8x8 grid for larger embeddings."""
        # Create an 8x8 image with random values
        np.random.seed(42)  # For reproducible tests
        image = np.random.rand(8, 8).astype(np.float32)
        
        result = self.mapper.map_from_2d(image)
        
        # Should be 1D array with 64 values
        assert result.shape == (64,)
        assert result.dtype == np.float32
        
        # Values should follow Hilbert curve order
        coordinates = self.mapper.generate_hilbert_coordinates(8)
        for i, (x, y) in enumerate(coordinates):
            assert result[i] == image[y, x], f"Mismatch at index {i}: expected {image[y, x]}, got {result[i]}"
    
    def test_round_trip_mapping_accuracy_2x2(self):
        """Test round-trip mapping accuracy for 2x2 embeddings."""
        # Original embedding
        original = np.array([1.5, 2.5, 3.5, 4.5])
        dimensions = (2, 2)
        
        # Forward mapping: 1D -> 2D
        image = self.mapper.map_to_2d(original, dimensions)
        
        # Inverse mapping: 2D -> 1D
        reconstructed = self.mapper.map_from_2d(image)
        
        # Should perfectly reconstruct original embedding
        np.testing.assert_array_equal(reconstructed, original)
    
    def test_round_trip_mapping_accuracy_4x4(self):
        """Test round-trip mapping accuracy for 4x4 embeddings."""
        # Original embedding with 16 values
        original = np.array([i * 0.1 for i in range(16)], dtype=np.float32)
        dimensions = (4, 4)
        
        # Forward mapping: 1D -> 2D
        image = self.mapper.map_to_2d(original, dimensions)
        
        # Inverse mapping: 2D -> 1D
        reconstructed = self.mapper.map_from_2d(image)
        
        # Should perfectly reconstruct original embedding
        np.testing.assert_array_equal(reconstructed, original)
    
    def test_round_trip_mapping_accuracy_8x8(self):
        """Test round-trip mapping accuracy for 8x8 embeddings."""
        # Original embedding with 64 values
        np.random.seed(123)
        original = np.random.rand(64).astype(np.float64)
        dimensions = (8, 8)
        
        # Forward mapping: 1D -> 2D
        image = self.mapper.map_to_2d(original, dimensions)
        
        # Inverse mapping: 2D -> 1D
        reconstructed = self.mapper.map_from_2d(image)
        
        # Should perfectly reconstruct original embedding
        np.testing.assert_array_equal(reconstructed, original)
    
    def test_round_trip_mapping_with_padding(self):
        """Test round-trip mapping accuracy with padded embeddings."""
        # Original embedding with fewer values than grid capacity
        original = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])  # 6 values
        dimensions = (4, 4)  # 16 cells available
        
        # Forward mapping: 1D -> 2D (with padding)
        image = self.mapper.map_to_2d(original, dimensions)
        
        # Inverse mapping: 2D -> 1D
        reconstructed = self.mapper.map_from_2d(image)
        
        # First 6 values should match original
        np.testing.assert_array_equal(reconstructed[:6], original)
        
        # Remaining values should be zeros (padding)
        np.testing.assert_array_equal(reconstructed[6:], np.zeros(10))
    
    def test_bijective_mapping_property_2x2(self):
        """Test bijective property of mapping for 2x2 grid."""
        # Test multiple different embeddings
        test_embeddings = [
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([0.1, 0.2, 0.3, 0.4]),
            np.array([-1.0, 0.0, 1.0, 2.0]),
            np.array([100.0, 200.0, 300.0, 400.0])
        ]
        
        dimensions = (2, 2)
        
        for original in test_embeddings:
            # Forward then inverse mapping
            image = self.mapper.map_to_2d(original, dimensions)
            reconstructed = self.mapper.map_from_2d(image)
            
            # Should be bijective (perfect reconstruction)
            np.testing.assert_array_equal(reconstructed, original)
    
    def test_bijective_mapping_property_4x4(self):
        """Test bijective property of mapping for 4x4 grid."""
        # Test with various embedding patterns
        test_cases = [
            np.arange(16, dtype=np.float32),  # Sequential
            np.random.rand(16).astype(np.float32),  # Random
            np.ones(16, dtype=np.float32),  # Uniform
            np.array([i % 3 for i in range(16)], dtype=np.float32)  # Repeating pattern
        ]
        
        dimensions = (4, 4)
        
        for original in test_cases:
            # Forward then inverse mapping
            image = self.mapper.map_to_2d(original, dimensions)
            reconstructed = self.mapper.map_from_2d(image)
            
            # Should be bijective (perfect reconstruction)
            np.testing.assert_array_equal(reconstructed, original)
    
    def test_bijective_mapping_property_8x8(self):
        """Test bijective property of mapping for 8x8 grid."""
        # Test with large embedding
        np.random.seed(456)
        original = np.random.rand(64).astype(np.float64)
        dimensions = (8, 8)
        
        # Forward then inverse mapping
        image = self.mapper.map_to_2d(original, dimensions)
        reconstructed = self.mapper.map_from_2d(image)
        
        # Should be bijective (perfect reconstruction)
        np.testing.assert_array_equal(reconstructed, original)
    
    def test_data_type_preservation_in_inverse_mapping(self):
        """Test that data types are preserved in inverse mapping."""
        test_cases = [
            (np.float32, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)),
            (np.float64, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)),
            (np.int32, np.array([[1, 2], [3, 4]], dtype=np.int32)),
        ]
        
        for expected_dtype, image in test_cases:
            result = self.mapper.map_from_2d(image)
            
            assert result.dtype == expected_dtype, f"Data type not preserved: expected {expected_dtype}, got {result.dtype}"
            assert result.shape == (4,)
    
    def test_map_from_2d_invalid_dimensions(self):
        """Test error handling for invalid image dimensions."""
        # Non-square image
        with pytest.raises(ValueError, match="requires square dimensions"):
            self.mapper.map_from_2d(np.array([[1, 2, 3], [4, 5, 6]]))
        
        # Non-power of 2 dimensions
        with pytest.raises(ValueError, match="must be a power of 2"):
            self.mapper.map_from_2d(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        
        # 1D array instead of 2D
        with pytest.raises(ValueError, match="must be 2D array"):
            self.mapper.map_from_2d(np.array([1, 2, 3, 4]))
        
        # 3D array instead of 2D
        with pytest.raises(ValueError, match="must be 2D array"):
            self.mapper.map_from_2d(np.array([[[1, 2], [3, 4]]]))
    
    def test_map_from_2d_zero_dimension(self):
        """Test error handling for zero dimension."""
        with pytest.raises(ValueError, match="must be a power of 2"):
            # This should fail during dimension validation
            empty_image = np.array([]).reshape(0, 0)
            self.mapper.map_from_2d(empty_image)
    
    def test_embedding_reconstruction_with_compression_artifacts(self):
        """Test reconstruction behavior with simulated compression artifacts."""
        # Original embedding
        original = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        dimensions = (2, 2)
        
        # Forward mapping
        image = self.mapper.map_to_2d(original, dimensions)
        
        # Simulate compression artifacts (small numerical errors)
        compressed_image = image + np.random.normal(0, 0.001, image.shape).astype(np.float32)
        
        # Inverse mapping
        reconstructed = self.mapper.map_from_2d(compressed_image)
        
        # Should be close to original (within compression tolerance)
        np.testing.assert_allclose(reconstructed, original, atol=0.01)
    
    def test_large_embedding_reconstruction_16x16(self):
        """Test reconstruction for large 16x16 embeddings."""
        # Create 256-dimensional embedding
        np.random.seed(789)
        original = np.random.rand(256).astype(np.float32)
        dimensions = (16, 16)
        
        # Forward mapping
        image = self.mapper.map_to_2d(original, dimensions)
        
        # Inverse mapping
        reconstructed = self.mapper.map_from_2d(image)
        
        # Should perfectly reconstruct
        np.testing.assert_array_equal(reconstructed, original)
        assert reconstructed.shape == (256,)
        assert reconstructed.dtype == np.float32
    
    def test_partial_embedding_reconstruction_16x16(self):
        """Test reconstruction for partial embeddings in 16x16 grid."""
        # Create 200-dimensional embedding (partial fill of 16x16 grid)
        original = np.array([i * 0.01 for i in range(200)], dtype=np.float32)
        dimensions = (16, 16)
        
        # Forward mapping (with padding)
        image = self.mapper.map_to_2d(original, dimensions)
        
        # Inverse mapping
        reconstructed = self.mapper.map_from_2d(image)
        
        # Should reconstruct full 256 values (200 original + 56 padding zeros)
        assert reconstructed.shape == (256,)
        
        # First 200 should match original
        np.testing.assert_array_equal(reconstructed[:200], original)
        
        # Remaining should be zeros
        np.testing.assert_array_equal(reconstructed[200:], np.zeros(56))
    
    def test_reconstruction_spatial_locality_preservation(self):
        """Test that spatial locality is preserved through reconstruction."""
        # Create embedding with spatial patterns
        original = np.array([
            1.0, 1.1, 1.2, 1.3,  # Similar values (should be spatially close)
            5.0, 5.1, 5.2, 5.3,  # Different group
            9.0, 9.1, 9.2, 9.3,  # Another group
            2.0, 2.1, 2.2, 2.3   # Final group
        ], dtype=np.float32)
        dimensions = (4, 4)
        
        # Forward mapping
        image = self.mapper.map_to_2d(original, dimensions)
        
        # Inverse mapping
        reconstructed = self.mapper.map_from_2d(image)
        
        # Should perfectly reconstruct original ordering
        np.testing.assert_array_equal(reconstructed, original)
        
        # Verify that similar values in original are still grouped
        # (This tests that Hilbert curve ordering is preserved)
        for i in range(0, len(original), 4):
            group = reconstructed[i:i+4]
            # Values in each group should be similar (within 0.5)
            assert np.max(group) - np.min(group) <= 0.5, f"Group {i//4} not preserved: {group}"
    
    def test_reconstruction_performance_large_embeddings(self):
        """Test reconstruction performance for large embeddings."""
        import time
        
        # Test with 32x32 grid (1024 values)
        original = np.random.rand(1024).astype(np.float32)
        dimensions = (32, 32)
        
        # Forward mapping
        image = self.mapper.map_to_2d(original, dimensions)
        
        # Time the inverse mapping
        start_time = time.time()
        reconstructed = self.mapper.map_from_2d(image)
        end_time = time.time()
        
        reconstruction_time = end_time - start_time
        
        # Should reconstruct quickly (less than 0.1 seconds)
        assert reconstruction_time < 0.1, f"Reconstruction too slow: {reconstruction_time:.3f}s"
        
        # Verify correctness
        np.testing.assert_array_equal(reconstructed, original)
    
    def test_reconstruction_determinism(self):
        """Test that reconstruction is deterministic."""
        # Create test image
        image = np.random.rand(8, 8).astype(np.float32)
        
        # Reconstruct multiple times
        result1 = self.mapper.map_from_2d(image)
        result2 = self.mapper.map_from_2d(image)
        result3 = self.mapper.map_from_2d(image)
        
        # Should be identical
        np.testing.assert_array_equal(result1, result2)
        np.testing.assert_array_equal(result2, result3)
    
    def test_reconstruction_with_edge_values(self):
        """Test reconstruction with edge case values."""
        # Test with extreme values
        edge_cases = [
            np.array([[0.0, 0.0], [0.0, 0.0]]),  # All zeros
            np.array([[1.0, 1.0], [1.0, 1.0]]),  # All ones
            np.array([[-1.0, 1.0], [1.0, -1.0]]),  # Negative and positive
            np.array([[1e-10, 1e10], [1e-5, 1e5]]),  # Very small and large
        ]
        
        for image in edge_cases:
            result = self.mapper.map_from_2d(image)
            
            # Should handle edge cases without errors
            assert result.shape == (4,)
            assert not np.any(np.isnan(result)), "NaN values in reconstruction"
            assert not np.any(np.isinf(result)), "Infinite values in reconstruction"
    
    def test_full_pipeline_integration_with_real_embedding_dimensions(self):
        """Test full pipeline integration with realistic embedding dimensions."""
        # Test common embedding dimensions used in practice
        test_cases = [
            (64, 8),    # 64-dim -> 8x8
            (128, 16),  # 128-dim -> 16x16 (partial)
            (256, 16),  # 256-dim -> 16x16 (exact)
            (384, 32),  # 384-dim -> 32x32 (partial)
            (512, 32),  # 512-dim -> 32x32 (partial)
            (768, 32),  # 768-dim -> 32x32 (partial)
        ]
        
        for embedding_dim, grid_size in test_cases:
            # Create realistic embedding
            np.random.seed(embedding_dim)  # Reproducible per dimension
            original = np.random.normal(0, 1, embedding_dim).astype(np.float32)
            dimensions = (grid_size, grid_size)
            
            # Full round-trip
            image = self.mapper.map_to_2d(original, dimensions)
            reconstructed = self.mapper.map_from_2d(image)
            
            # Verify reconstruction
            total_cells = grid_size * grid_size
            
            # Original embedding should be perfectly reconstructed
            np.testing.assert_array_equal(reconstructed[:embedding_dim], original)
            
            # Padding should be zeros
            if embedding_dim < total_cells:
                padding_size = total_cells - embedding_dim
                np.testing.assert_array_equal(reconstructed[embedding_dim:], np.zeros(padding_size))
            
            # Verify shape and type
            assert reconstructed.shape == (total_cells,)
            assert reconstructed.dtype == np.float32
        # This test ensures the mapping is suitable for inverse operations
        embeddings = np.array([1.5, 2.7, 3.1, 4.9, 5.2, 6.8, 7.3, 8.6])
        dimensions = (4, 4)
        
        result = self.mapper.map_to_2d(embeddings, dimensions)
        
        # Extract values back in Hilbert order to simulate reconstruction
        coordinates = self.mapper.generate_hilbert_coordinates(4)
        reconstructed = []
        
        for i in range(len(embeddings)):
            x, y = coordinates[i]
            reconstructed.append(result[y, x])
        
        reconstructed = np.array(reconstructed)
        
        # Should perfectly match original embeddings
        np.testing.assert_array_equal(reconstructed, embeddings)
    
    def test_map_to_2d_spatial_locality_quantitative(self):
        """Test spatial locality preservation with quantitative metrics."""
        # Create embeddings with known similarity structure
        embeddings = np.array([i * 0.1 for i in range(64)])  # Linear progression
        dimensions = (8, 8)
        
        result = self.mapper.map_to_2d(embeddings, dimensions)
        coordinates = self.mapper.generate_hilbert_coordinates(8)
        
        # Calculate average distance between consecutive embeddings
        total_distance = 0
        for i in range(len(embeddings) - 1):
            x1, y1 = coordinates[i]
            x2, y2 = coordinates[i + 1]
            distance = abs(x2 - x1) + abs(y2 - y1)
            total_distance += distance
        
        average_distance = total_distance / (len(embeddings) - 1)
        
        # Average distance should be close to 1 (mostly adjacent moves)
        assert average_distance <= 1.5, f"Poor spatial locality: average distance {average_distance:.2f}"
        
        # Count adjacent moves (distance = 1)
        adjacent_moves = 0
        for i in range(len(embeddings) - 1):
            x1, y1 = coordinates[i]
            x2, y2 = coordinates[i + 1]
            if abs(x2 - x1) + abs(y2 - y1) == 1:
                adjacent_moves += 1
        
        locality_ratio = adjacent_moves / (len(embeddings) - 1)
        
        # At least 60% of moves should be to adjacent cells
        assert locality_ratio >= 0.6, f"Insufficient spatial locality: {locality_ratio:.2f}"