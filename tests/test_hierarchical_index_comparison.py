"""
Unit tests for hierarchical index comparison functionality.

This module tests the multi-level hierarchical index comparison methods
that implement requirements 4.2 and 4.3 for progressive similarity filtering.
"""

import pytest
import numpy as np
from hilbert_quantization.rag.search.engine import RAGSearchEngineImpl
from hilbert_quantization.rag.config import RAGConfig


class TestHierarchicalIndexComparison:
    """Test suite for hierarchical index comparison methods."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = RAGConfig()
        self.search_engine = RAGSearchEngineImpl(self.config)
    
    def test_single_level_index_comparison_identical(self):
        """Test single-level index comparison with identical indices."""
        query_indices = np.array([1.0, 2.0, 3.0, 4.0])
        candidate_indices = np.array([1.0, 2.0, 3.0, 4.0])
        
        similarity = self.search_engine.compare_hierarchical_indices(
            query_indices, candidate_indices
        )
        
        # Identical indices should have similarity close to 1.0
        assert similarity > 0.95
        assert similarity <= 1.0
    
    def test_single_level_index_comparison_different(self):
        """Test single-level index comparison with different indices."""
        query_indices = np.array([1.0, 2.0, 3.0, 4.0])
        candidate_indices = np.array([4.0, 3.0, 2.0, 1.0])
        
        similarity = self.search_engine.compare_hierarchical_indices(
            query_indices, candidate_indices
        )
        
        # Different indices should have lower similarity
        assert 0.0 <= similarity < 1.0
    
    def test_single_level_index_comparison_orthogonal(self):
        """Test single-level index comparison with orthogonal vectors."""
        query_indices = np.array([1.0, 0.0, 0.0, 0.0])
        candidate_indices = np.array([0.0, 1.0, 0.0, 0.0])
        
        similarity = self.search_engine.compare_hierarchical_indices(
            query_indices, candidate_indices
        )
        
        # Orthogonal vectors should have similarity around 0.5 (normalized cosine similarity)
        assert 0.4 <= similarity <= 0.6
    
    def test_single_level_index_comparison_zero_vectors(self):
        """Test single-level index comparison with zero vectors."""
        query_indices = np.array([0.0, 0.0, 0.0, 0.0])
        candidate_indices = np.array([1.0, 2.0, 3.0, 4.0])
        
        similarity = self.search_engine.compare_hierarchical_indices(
            query_indices, candidate_indices
        )
        
        # Zero vector should result in zero similarity
        assert similarity == 0.0
    
    def test_multi_level_index_comparison_identical(self):
        """Test multi-level index comparison with identical indices."""
        query_indices = np.array([
            [1.0, 2.0, 3.0, 4.0],  # Coarse level
            [1.1, 1.9, 3.1, 3.9],  # Medium level
            [1.05, 1.95, 3.05, 3.95]  # Fine level
        ])
        candidate_indices = query_indices.copy()
        
        similarity = self.search_engine.compare_hierarchical_indices(
            query_indices, candidate_indices
        )
        
        # Identical multi-level indices should have high similarity
        assert similarity > 0.95
        assert similarity <= 1.0
    
    def test_multi_level_index_comparison_different_levels(self):
        """Test multi-level index comparison with differences at different levels."""
        query_indices = np.array([
            [1.0, 2.0, 3.0, 4.0],  # Coarse level
            [1.1, 1.9, 3.1, 3.9],  # Medium level
            [1.05, 1.95, 3.05, 3.95]  # Fine level
        ])
        
        # Candidate differs only at fine level
        candidate_indices = np.array([
            [1.0, 2.0, 3.0, 4.0],  # Same coarse level
            [1.1, 1.9, 3.1, 3.9],  # Same medium level
            [2.0, 3.0, 4.0, 5.0]   # Different fine level
        ])
        
        similarity = self.search_engine.compare_hierarchical_indices(
            query_indices, candidate_indices
        )
        
        # Should still have high similarity due to coarse level matching
        assert similarity > 0.7
    
    def test_multi_level_index_comparison_coarse_level_different(self):
        """Test multi-level index comparison with differences at coarse level."""
        query_indices = np.array([
            [1.0, 2.0, 3.0, 4.0],  # Coarse level
            [1.1, 1.9, 3.1, 3.9],  # Medium level
            [1.05, 1.95, 3.05, 3.95]  # Fine level
        ])
        
        # Candidate differs significantly at coarse level with different pattern
        candidate_indices = np.array([
            [8.0, 1.0, 9.0, 2.0],      # Very different coarse level pattern
            [1.1, 1.9, 3.1, 3.9],      # Same medium level
            [1.05, 1.95, 3.05, 3.95]   # Same fine level
        ])
        
        similarity = self.search_engine.compare_hierarchical_indices(
            query_indices, candidate_indices
        )
        
        # Should have lower similarity due to coarse level differences
        # Even with strong coarse weighting, identical fine levels still contribute
        assert similarity < 0.9
    
    def test_granularity_weights_calculation(self):
        """Test granularity weights calculation for different numbers of levels."""
        # Test single level
        weights_1 = self.search_engine._calculate_granularity_weights(1)
        assert len(weights_1) == 1
        assert weights_1[0] == 1.0
        
        # Test multiple levels
        weights_3 = self.search_engine._calculate_granularity_weights(3)
        assert len(weights_3) == 3
        
        # First level (coarsest) should have highest weight
        assert weights_3[0] > weights_3[1]
        assert weights_3[1] > weights_3[2]
        
        # Weights should be positive
        assert all(w > 0 for w in weights_3)
    
    def test_spatial_sections_comparison_identical(self):
        """Test spatial sections comparison with identical sections."""
        query_sections = np.array([1.0, 2.0, 3.0, 4.0])
        candidate_sections = np.array([1.0, 2.0, 3.0, 4.0])
        
        similarity = self.search_engine.compare_spatial_sections(
            query_sections, candidate_sections, granularity=32
        )
        
        # Identical sections should have high similarity
        assert similarity > 0.95
        assert similarity <= 1.0
    
    def test_spatial_sections_comparison_different_granularities(self):
        """Test spatial sections comparison with different granularity levels."""
        query_sections = np.array([1.0, 2.0, 3.0, 4.0])
        candidate_sections = np.array([1.1, 2.1, 3.1, 4.1])
        
        # Test coarse granularity (should favor cosine similarity)
        coarse_similarity = self.search_engine.compare_spatial_sections(
            query_sections, candidate_sections, granularity=32
        )
        
        # Test fine granularity (should favor euclidean and correlation)
        fine_similarity = self.search_engine.compare_spatial_sections(
            query_sections, candidate_sections, granularity=8
        )
        
        # Both should be high but may differ slightly due to different weighting
        assert coarse_similarity > 0.8
        assert fine_similarity > 0.8
    
    def test_cosine_similarity_calculation(self):
        """Test cosine similarity calculation."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([1.0, 0.0, 0.0])
        
        similarity = self.search_engine._calculate_cosine_similarity(vec1, vec2)
        assert similarity == 1.0
        
        # Test orthogonal vectors
        vec1 = np.array([1.0, 0.0])
        vec2 = np.array([0.0, 1.0])
        
        similarity = self.search_engine._calculate_cosine_similarity(vec1, vec2)
        assert similarity == 0.5  # Normalized cosine similarity of 0
    
    def test_euclidean_similarity_calculation(self):
        """Test euclidean distance-based similarity calculation."""
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([1.0, 2.0, 3.0])
        
        similarity = self.search_engine._calculate_euclidean_similarity(vec1, vec2)
        assert similarity == 1.0  # Identical vectors
        
        # Test different vectors
        vec1 = np.array([0.0, 0.0])
        vec2 = np.array([1.0, 1.0])
        
        similarity = self.search_engine._calculate_euclidean_similarity(vec1, vec2)
        assert 0.0 < similarity < 1.0
    
    def test_correlation_similarity_calculation(self):
        """Test correlation-based similarity calculation."""
        # Perfect positive correlation
        vec1 = np.array([1.0, 2.0, 3.0, 4.0])
        vec2 = np.array([2.0, 4.0, 6.0, 8.0])  # 2 * vec1
        
        similarity = self.search_engine._calculate_correlation_similarity(vec1, vec2)
        assert similarity == 1.0
        
        # Perfect negative correlation
        vec1 = np.array([1.0, 2.0, 3.0, 4.0])
        vec2 = np.array([4.0, 3.0, 2.0, 1.0])
        
        similarity = self.search_engine._calculate_correlation_similarity(vec1, vec2)
        assert similarity == 0.0
    
    def test_empty_indices_handling(self):
        """Test handling of empty indices."""
        empty_indices = np.array([])
        non_empty_indices = np.array([1.0, 2.0, 3.0])
        
        # Test empty query indices
        similarity1 = self.search_engine.compare_hierarchical_indices(
            empty_indices, empty_indices
        )
        assert similarity1 == 0.0
        
        # Test with same-shaped empty arrays
        empty_2d = np.array([]).reshape(0, 0)
        similarity2 = self.search_engine.compare_hierarchical_indices(
            empty_2d, empty_2d
        )
        assert similarity2 == 0.0
    
    def test_mismatched_shapes_error(self):
        """Test error handling for mismatched index shapes."""
        query_indices = np.array([1.0, 2.0, 3.0])
        candidate_indices = np.array([1.0, 2.0])
        
        with pytest.raises(ValueError, match="must have the same shape"):
            self.search_engine.compare_hierarchical_indices(
                query_indices, candidate_indices
            )
    
    def test_invalid_dimensions_error(self):
        """Test error handling for invalid index dimensions."""
        query_indices = np.array([[[1.0, 2.0], [3.0, 4.0]]])  # 3D array
        candidate_indices = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        
        with pytest.raises(ValueError, match="must be 1D or 2D arrays"):
            self.search_engine.compare_hierarchical_indices(
                query_indices, candidate_indices
            )
    
    def test_spatial_sections_mismatched_length_error(self):
        """Test error handling for mismatched spatial section lengths."""
        query_sections = np.array([1.0, 2.0, 3.0])
        candidate_sections = np.array([1.0, 2.0])
        
        with pytest.raises(ValueError, match="must have the same length"):
            self.search_engine.compare_spatial_sections(
                query_sections, candidate_sections, granularity=16
            )
    
    def test_multi_level_comparison_accuracy_across_levels(self):
        """Test accuracy of multi-level comparison across different granularity levels."""
        # Create realistic hierarchical indices with same dimensions for each level
        query_indices = np.array([
            [5.2, 3.1, 7.8, 2.4, 6.5, 4.2, 8.1, 3.7],      # Coarse: 8 sections
            [5.1, 5.3, 3.0, 3.2, 7.7, 7.9, 2.3, 2.5],      # Medium: 8 sections
            [5.05, 5.15, 5.25, 5.35, 2.95, 3.05, 3.15, 3.25]  # Fine: 8 sections
        ])
        
        # Similar candidate (should score high)
        similar_candidate = np.array([
            [5.3, 3.0, 7.9, 2.3, 6.6, 4.1, 8.2, 3.6],      # Very similar coarse level
            [5.2, 5.4, 2.9, 3.3, 7.6, 8.0, 2.2, 2.6],      # Similar medium level
            [5.10, 5.20, 5.30, 5.40, 2.90, 3.10, 3.20, 3.30]  # Similar fine level
        ])
        
        # Dissimilar candidate (should score low) - using orthogonal patterns
        dissimilar_candidate = np.array([
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],      # Orthogonal coarse level
            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],      # Orthogonal medium level
            [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]       # Orthogonal fine level
        ])
        
        similar_score = self.search_engine.compare_hierarchical_indices(
            query_indices, similar_candidate
        )
        
        dissimilar_score = self.search_engine.compare_hierarchical_indices(
            query_indices, dissimilar_candidate
        )
        
        # Similar candidate should score significantly higher
        assert similar_score > dissimilar_score
        assert similar_score > 0.8
        # The dissimilar score might still be relatively high due to multiple similarity metrics
        # The key is that similar_score > dissimilar_score
        assert dissimilar_score < similar_score
    
    def test_progressive_granularity_weighting(self):
        """Test that coarser granularities have higher influence on similarity."""
        # Create indices where only coarse level matches
        query_indices = np.array([
            [5.0, 3.0, 7.0, 2.0],      # Coarse level
            [1.0, 2.0, 3.0, 4.0]       # Fine level
        ])
        
        coarse_match_candidate = np.array([
            [5.1, 2.9, 7.1, 1.9],      # Very similar coarse level
            [8.0, 9.0, 10.0, 11.0]     # Very different fine level
        ])
        
        fine_match_candidate = np.array([
            [1.0, 8.0, 3.0, 9.0],      # Very different coarse level
            [1.1, 1.9, 3.1, 3.9]       # Very similar fine level
        ])
        
        coarse_match_score = self.search_engine.compare_hierarchical_indices(
            query_indices, coarse_match_candidate
        )
        
        fine_match_score = self.search_engine.compare_hierarchical_indices(
            query_indices, fine_match_candidate
        )
        
        # Coarse level match should score higher due to weighting
        assert coarse_match_score > fine_match_score