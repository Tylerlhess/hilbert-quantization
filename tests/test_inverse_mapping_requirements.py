"""
Test verification that inverse mapping implementation meets specific requirements 5.2, 5.3, 5.4.
"""

import pytest
import numpy as np
from hilbert_quantization.rag.embedding_generation.hilbert_mapper import HilbertCurveMapperImpl
from hilbert_quantization.rag.config import RAGConfig


class TestInverseMappingRequirements:
    """Test cases verifying requirements 5.2, 5.3, 5.4 are met."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = RAGConfig()
        self.mapper = HilbertCurveMapperImpl(self.config)
    
    def test_requirement_5_2_inverse_decompression_support(self):
        """
        Requirement 5.2: WHEN reconstruction is performed THEN the system SHALL 
        apply inverse MPEG decompression to recover the embedding frames.
        
        This test verifies that the system supports reconstruction from 2D representation
        (the core component after MPEG decompression would occur).
        """
        # Simulate a 2D representation that would come from MPEG decompression
        compressed_representation = np.array([
            [1.5, 2.5],
            [3.5, 4.5]
        ], dtype=np.float32)
        
        # The system should support reconstruction from this 2D representation
        reconstructed = self.mapper.map_from_2d(compressed_representation)
        
        # Verify reconstruction is supported and produces valid output
        assert isinstance(reconstructed, np.ndarray)
        assert reconstructed.shape == (4,)
        assert reconstructed.dtype == np.float32
        
        # Verify the values follow Hilbert curve order
        expected_hilbert_order = [1.5, 3.5, 4.5, 2.5]  # Based on 2x2 Hilbert curve
        np.testing.assert_array_equal(reconstructed, expected_hilbert_order)
    
    def test_requirement_5_3_inverse_hilbert_mapping(self):
        """
        Requirement 5.3: WHEN decompression is complete THEN the system SHALL 
        map the 2D representation back to the original embedding space using 
        inverse Hilbert curve mapping.
        """
        # Test various embedding dimensions to ensure inverse mapping works
        test_cases = [
            (4, (2, 2)),    # 4-dimensional embedding -> 2x2 grid
            (16, (4, 4)),   # 16-dimensional embedding -> 4x4 grid
            (64, (8, 8)),   # 64-dimensional embedding -> 8x8 grid
            (256, (16, 16)) # 256-dimensional embedding -> 16x16 grid
        ]
        
        for embedding_dim, grid_dims in test_cases:
            # Create original embedding in original embedding space
            original_embedding = np.random.rand(embedding_dim).astype(np.float32)
            
            # Forward map to 2D representation (simulating what would be compressed)
            two_d_representation = self.mapper.map_to_2d(original_embedding, grid_dims)
            
            # Apply inverse Hilbert curve mapping back to original embedding space
            reconstructed_embedding = self.mapper.map_from_2d(two_d_representation)
            
            # Verify mapping back to original embedding space
            assert reconstructed_embedding.shape[0] >= embedding_dim
            np.testing.assert_array_equal(
                reconstructed_embedding[:embedding_dim], 
                original_embedding
            )
            
            # Verify it's truly in the original embedding space (1D)
            assert len(reconstructed_embedding.shape) == 1
    
    def test_requirement_5_4_dimension_validation(self):
        """
        Requirement 5.4: WHEN embeddings are reconstructed THEN the system SHALL 
        validate that embedding dimensions match the original embedding model structure.
        """
        # Test dimension validation during reconstruction
        
        # Valid cases - should work without errors
        valid_cases = [
            np.array([[1.0, 2.0], [3.0, 4.0]]),  # 2x2 (power of 2)
            np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]),  # 4x4
            np.random.rand(8, 8)  # 8x8
        ]
        
        for valid_input in valid_cases:
            # Should reconstruct without validation errors
            result = self.mapper.map_from_2d(valid_input)
            
            # Verify dimensions are validated and preserved correctly
            expected_length = valid_input.shape[0] * valid_input.shape[1]
            assert result.shape == (expected_length,)
            assert result.dtype == valid_input.dtype
        
        # Invalid cases - should raise validation errors
        invalid_cases = [
            # Non-square dimensions
            (np.array([[1, 2, 3], [4, 5, 6]]), "requires square dimensions"),
            # Non-power of 2 dimensions  
            (np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), "must be a power of 2"),
            # Wrong number of dimensions
            (np.array([1, 2, 3, 4]), "must be 2D array"),
            (np.array([[[1, 2], [3, 4]]]), "must be 2D array")
        ]
        
        for invalid_input, expected_error in invalid_cases:
            with pytest.raises(ValueError, match=expected_error):
                self.mapper.map_from_2d(invalid_input)
    
    def test_requirement_5_3_bijective_property_verification(self):
        """
        Additional verification that inverse Hilbert mapping is truly bijective
        (requirement 5.3 implies this for accurate reconstruction).
        """
        # Test bijective property across different embedding sizes
        test_embeddings = [
            np.array([1.0, 2.0, 3.0, 4.0]),  # Simple case
            np.random.rand(16).astype(np.float32),  # Random 4x4
            np.random.rand(64).astype(np.float64),  # Random 8x8
            np.arange(256, dtype=np.int32)  # Sequential 16x16
        ]
        
        dimensions_map = {
            4: (2, 2),
            16: (4, 4), 
            64: (8, 8),
            256: (16, 16)
        }
        
        for original in test_embeddings:
            dims = dimensions_map[len(original)]
            
            # Forward mapping: embedding -> 2D
            two_d = self.mapper.map_to_2d(original, dims)
            
            # Inverse mapping: 2D -> embedding  
            reconstructed = self.mapper.map_from_2d(two_d)
            
            # Verify bijective property (perfect reconstruction)
            np.testing.assert_array_equal(reconstructed[:len(original)], original)
            
            # Verify data type preservation (part of dimension validation)
            assert reconstructed.dtype == original.dtype
    
    def test_requirement_5_4_embedding_model_structure_preservation(self):
        """
        Verify that reconstructed embeddings maintain the structure expected
        by embedding models (requirement 5.4).
        """
        # Test common embedding model dimensions
        common_embedding_dims = [
            64,    # Small transformer models
            128,   # Medium models
            256,   # BERT-base hidden size / 3
            384,   # BERT-base hidden size / 2  
            512,   # Common embedding size
            768,   # BERT-base hidden size
        ]
        
        for embedding_dim in common_embedding_dims:
            # Create embedding similar to what a model would produce
            # (normally distributed values, typical of neural network outputs)
            original_embedding = np.random.normal(0, 1, embedding_dim).astype(np.float32)
            
            # Determine appropriate grid size (next power of 4)
            grid_size = 1
            while grid_size * grid_size < embedding_dim:
                grid_size *= 2
            
            # Forward and inverse mapping
            two_d_repr = self.mapper.map_to_2d(original_embedding, (grid_size, grid_size))
            reconstructed = self.mapper.map_from_2d(two_d_repr)
            
            # Verify embedding model structure is preserved
            assert len(reconstructed) >= embedding_dim  # Has space for full embedding
            assert reconstructed.dtype == np.float32    # Preserves model data type
            assert reconstructed.shape == (grid_size * grid_size,)  # Correct total shape
            
            # Verify original embedding values are perfectly preserved
            np.testing.assert_array_equal(
                reconstructed[:embedding_dim], 
                original_embedding
            )
            
            # Verify padding is clean (zeros) for unused space
            if embedding_dim < len(reconstructed):
                padding = reconstructed[embedding_dim:]
                assert np.all(padding == 0.0), f"Padding not clean for {embedding_dim}D embedding"
    
    def test_requirement_integration_full_pipeline(self):
        """
        Integration test verifying all requirements 5.2, 5.3, 5.4 work together
        in a complete reconstruction pipeline.
        """
        # Simulate a complete pipeline for a realistic embedding
        original_embedding = np.random.normal(0, 1, 384).astype(np.float32)  # BERT-like
        
        # Step 1: Forward mapping (would be followed by MPEG compression)
        two_d_representation = self.mapper.map_to_2d(original_embedding, (32, 32))
        
        # Step 2: Simulate compression artifacts (small numerical errors)
        # This simulates what might happen after MPEG compression/decompression
        compressed_2d = two_d_representation + np.random.normal(0, 1e-6, two_d_representation.shape)
        compressed_2d = compressed_2d.astype(np.float32)
        
        # Step 3: Inverse Hilbert mapping (requirement 5.3)
        reconstructed_embedding = self.mapper.map_from_2d(compressed_2d)
        
        # Step 4: Validation (requirement 5.4)
        # Verify dimensions match original embedding model structure
        assert reconstructed_embedding.shape == (1024,)  # 32x32 total
        assert reconstructed_embedding.dtype == np.float32
        
        # Verify original embedding is reconstructed (within compression tolerance)
        original_part = reconstructed_embedding[:384]
        np.testing.assert_allclose(original_part, original_embedding, atol=1e-5)
        
        # Verify padding is preserved
        padding_part = reconstructed_embedding[384:]
        assert np.allclose(padding_part, 0.0, atol=1e-5)
        
        # This demonstrates that requirements 5.2, 5.3, 5.4 are all satisfied:
        # - 5.2: System supports reconstruction from 2D representation
        # - 5.3: Inverse Hilbert mapping correctly maps back to embedding space  
        # - 5.4: Dimensions are validated and embedding structure is preserved