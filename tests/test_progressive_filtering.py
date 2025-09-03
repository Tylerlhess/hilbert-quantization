"""
Unit tests for progressive filtering algorithm functionality.

This module tests the progressive hierarchical filtering methods that implement
requirements 4.2, 4.3, and 4.4 for efficient candidate elimination.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from hilbert_quantization.rag.search.engine import RAGSearchEngineImpl
from hilbert_quantization.rag.config import RAGConfig


class TestProgressiveFiltering:
    """Test suite for progressive filtering algorithm methods."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = RAGConfig()
        self.search_engine = RAGSearchEngineImpl(self.config)
    
    def test_extract_hierarchical_indices_2d_embedding(self):
        """Test extraction of hierarchical indices from 2D embedding."""
        # Create enhanced embedding with index rows (more zeros to be clearly sparse)
        enhanced_embedding = np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],  # Original embedding row 1
            [7.0, 8.0, 9.0, 1.0, 2.0, 3.0],  # Original embedding row 2
            [1.5, 2.5, 0.0, 0.0, 0.0, 0.0],  # Index row 1 (coarse) - 67% zeros
            [1.2, 2.2, 3.2, 0.0, 0.0, 0.0]   # Index row 2 (fine) - 50% zeros
        ])
        
        indices = self.search_engine._extract_hierarchical_indices(enhanced_embedding)
        
        assert len(indices) == 2
        # Check that trailing zeros are removed
        assert len(indices[0]) == 2  # [1.5, 2.5]
        assert len(indices[1]) == 3  # [1.2, 2.2, 3.2]
        np.testing.assert_array_equal(indices[0], [1.5, 2.5])
        np.testing.assert_array_equal(indices[1], [1.2, 2.2, 3.2])
    
    def test_extract_hierarchical_indices_1d_embedding(self):
        """Test extraction from 1D embedding (no indices)."""
        embedding_1d = np.array([1.0, 2.0, 3.0, 4.0])
        
        indices = self.search_engine._extract_hierarchical_indices(embedding_1d)
        
        assert len(indices) == 0
    
    def test_detect_original_embedding_height(self):
        """Test detection of original embedding height."""
        # Create embedding where last 2 rows are mostly zeros (index rows)
        enhanced_embedding = np.array([
            [1.0, 2.0, 3.0, 4.0],  # Original embedding
            [5.0, 6.0, 7.0, 8.0],  # Original embedding
            [1.5, 0.0, 0.0, 0.0],  # Index row (75% zeros)
            [1.2, 2.2, 0.0, 0.0]   # Index row (50% zeros)
        ])
        
        original_height = self.search_engine._detect_original_embedding_height(enhanced_embedding)
        
        # Should detect that first 2 rows are original embedding
        assert original_height == 2
    
    def test_filter_candidates_at_level_basic(self):
        """Test basic candidate filtering at a specific level."""
        query_level_indices = np.array([1.0, 2.0, 3.0])
        
        # Mock candidate embeddings with hierarchical indices
        candidate_embeddings = [
            # Candidate 0: similar to query
            np.array([
                [1.0, 2.0, 3.0, 4.0],  # Original
                [1.1, 2.1, 3.1, 0.0]   # Index row (similar)
            ]),
            # Candidate 1: different from query
            np.array([
                [5.0, 6.0, 7.0, 8.0],  # Original
                [5.0, 6.0, 7.0, 0.0]   # Index row (different)
            ])
        ]
        
        current_candidates = [0, 1]
        
        # Mock the hierarchical index extraction
        with patch.object(self.search_engine, '_extract_hierarchical_indices') as mock_extract:
            mock_extract.side_effect = [
                [np.array([1.1, 2.1, 3.1])],  # Candidate 0 indices
                [np.array([5.0, 6.0, 7.0])]   # Candidate 1 indices
            ]
            
            filtered_candidates = self.search_engine._filter_candidates_at_level(
                query_level_indices, candidate_embeddings, current_candidates, level=0
            )
        
        # Should return candidates (exact filtering depends on threshold)
        assert isinstance(filtered_candidates, list)
        assert len(filtered_candidates) <= len(current_candidates)
    
    def test_apply_progressive_threshold_coarse_level(self):
        """Test progressive threshold application at coarse level."""
        # Create candidate scores with varying similarities
        candidate_scores = [
            (0, 0.9),  # High similarity
            (1, 0.7),  # Medium similarity
            (2, 0.5),  # Low similarity
            (3, 0.2)   # Very low similarity
        ]
        
        # Test coarse level (should be more aggressive)
        filtered_coarse = self.search_engine._apply_progressive_threshold(
            candidate_scores, level=0
        )
        
        # Test fine level (should be more lenient)
        filtered_fine = self.search_engine._apply_progressive_threshold(
            candidate_scores, level=2
        )
        
        # Coarse level should filter more aggressively
        assert len(filtered_coarse) <= len(filtered_fine)
        
        # High similarity candidate should always be included
        assert 0 in filtered_coarse
        assert 0 in filtered_fine
    
    def test_progressive_hierarchical_search_empty_query(self):
        """Test progressive search with empty query."""
        empty_query = np.array([])
        
        result = self.search_engine.progressive_hierarchical_search(empty_query)
        
        assert result == []
    
    def test_progressive_hierarchical_search_no_candidates(self):
        """Test progressive search with no candidates."""
        query_embedding = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [1.5, 2.5, 0.0, 0.0]
        ])
        
        # Mock empty candidate list
        with patch.object(self.search_engine, '_get_all_candidate_embeddings', return_value=[]):
            result = self.search_engine.progressive_hierarchical_search(query_embedding)
        
        assert result == []
    
    def test_progressive_hierarchical_search_with_candidates(self):
        """Test progressive search with actual candidates."""
        query_embedding = np.array([
            [1.0, 2.0, 3.0, 4.0],  # Original embedding
            [1.5, 2.5, 0.0, 0.0]   # Index row
        ])
        
        # Mock candidate embeddings
        candidate_embeddings = [
            np.array([
                [1.1, 2.1, 3.1, 4.1],  # Similar to query
                [1.6, 2.6, 0.0, 0.0]   # Similar index
            ]),
            np.array([
                [5.0, 6.0, 7.0, 8.0],  # Different from query
                [5.0, 6.0, 0.0, 0.0]   # Different index
            ])
        ]
        
        with patch.object(self.search_engine, '_get_all_candidate_embeddings', 
                         return_value=candidate_embeddings):
            result = self.search_engine.progressive_hierarchical_search(query_embedding)
        
        # Should return some candidates
        assert isinstance(result, list)
    
    def test_calculate_adaptive_threshold_empty_scores(self):
        """Test adaptive threshold calculation with empty scores."""
        threshold = self.search_engine._calculate_adaptive_threshold([], level=0)
        
        assert threshold == 0.0
    
    def test_calculate_adaptive_threshold_coarse_vs_fine(self):
        """Test adaptive threshold calculation for different levels."""
        scores = [
            (0, 0.9), (1, 0.8), (2, 0.7), (3, 0.6), (4, 0.5)
        ]
        
        # Coarse level should have higher threshold
        coarse_threshold = self.search_engine._calculate_adaptive_threshold(scores, level=0)
        
        # Fine level should have lower threshold
        fine_threshold = self.search_engine._calculate_adaptive_threshold(scores, level=2)
        
        assert coarse_threshold >= fine_threshold
        
        # Both should be within reasonable bounds
        assert 0.1 <= coarse_threshold <= 0.9
        assert 0.1 <= fine_threshold <= 0.9
    
    def test_progressive_filter_with_adaptive_thresholds_empty_query(self):
        """Test adaptive progressive filtering with empty query."""
        empty_query = np.array([])
        
        result = self.search_engine.progressive_filter_with_adaptive_thresholds(empty_query)
        
        assert result == []
    
    def test_progressive_filter_with_adaptive_thresholds_with_initial_candidates(self):
        """Test adaptive progressive filtering with initial candidate list."""
        query_embedding = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [1.5, 2.5, 0.0, 0.0]
        ])
        
        initial_candidates = [0, 2, 4]  # Subset of candidates
        
        # Mock candidate embeddings
        candidate_embeddings = [
            np.array([[1.1, 2.1, 3.1, 4.1], [1.6, 2.6, 0.0, 0.0]]),
            np.array([[2.0, 3.0, 4.0, 5.0], [2.0, 3.0, 0.0, 0.0]]),
            np.array([[1.2, 2.2, 3.2, 4.2], [1.7, 2.7, 0.0, 0.0]]),
            np.array([[3.0, 4.0, 5.0, 6.0], [3.0, 4.0, 0.0, 0.0]]),
            np.array([[1.0, 2.0, 3.0, 4.0], [1.5, 2.5, 0.0, 0.0]])  # Identical to query
        ]
        
        with patch.object(self.search_engine, '_get_all_candidate_embeddings', 
                         return_value=candidate_embeddings):
            result = self.search_engine.progressive_filter_with_adaptive_thresholds(
                query_embedding, initial_candidates
            )
        
        # Should return subset of initial candidates
        assert isinstance(result, list)
        assert all(candidate in initial_candidates for candidate in result)
    
    def test_filter_candidates_length_mismatch_handling(self):
        """Test handling of length mismatches in candidate filtering."""
        query_level_indices = np.array([1.0, 2.0, 3.0])
        
        candidate_embeddings = [
            # Candidate with shorter indices
            np.array([
                [1.0, 2.0, 3.0, 4.0],
                [1.1, 2.1, 0.0, 0.0]  # Only 2 non-zero elements
            ]),
            # Candidate with longer indices
            np.array([
                [1.0, 2.0, 3.0, 4.0],
                [1.1, 2.1, 3.1, 4.1]  # 4 elements
            ])
        ]
        
        current_candidates = [0, 1]
        
        # Mock the hierarchical index extraction
        with patch.object(self.search_engine, '_extract_hierarchical_indices') as mock_extract:
            mock_extract.side_effect = [
                [np.array([1.1, 2.1])],        # Shorter indices
                [np.array([1.1, 2.1, 3.1, 4.1])]  # Longer indices
            ]
            
            filtered_candidates = self.search_engine._filter_candidates_at_level(
                query_level_indices, candidate_embeddings, current_candidates, level=0
            )
        
        # Should handle length mismatches gracefully
        assert isinstance(filtered_candidates, list)
    
    def test_filter_candidates_missing_level(self):
        """Test filtering when candidates don't have the requested level."""
        query_level_indices = np.array([1.0, 2.0, 3.0])
        
        candidate_embeddings = [
            np.array([[1.0, 2.0, 3.0, 4.0]])  # No index rows
        ]
        
        current_candidates = [0]
        
        # Mock extraction to return empty indices
        with patch.object(self.search_engine, '_extract_hierarchical_indices', return_value=[]):
            filtered_candidates = self.search_engine._filter_candidates_at_level(
                query_level_indices, candidate_embeddings, current_candidates, level=0
            )
        
        # Should handle missing levels gracefully
        assert isinstance(filtered_candidates, list)
    
    def test_progressive_threshold_bounds_checking(self):
        """Test that progressive thresholds stay within reasonable bounds."""
        # Test with extreme scores
        extreme_scores = [
            (0, 1.0), (1, 0.0)  # Maximum spread
        ]
        
        for level in range(4):  # Test multiple levels
            threshold = self.search_engine._calculate_adaptive_threshold(extreme_scores, level)
            
            # Should be within bounds
            assert 0.1 <= threshold <= 0.9
    
    def test_progressive_filtering_efficiency_across_levels(self):
        """Test that progressive filtering becomes more efficient at coarser levels."""
        # Create scores with clear hierarchy
        candidate_scores = [(i, 0.9 - i * 0.1) for i in range(10)]  # Scores from 0.9 to 0.0
        
        # Test filtering at different levels
        coarse_filtered = self.search_engine._apply_progressive_threshold(candidate_scores, level=0)
        medium_filtered = self.search_engine._apply_progressive_threshold(candidate_scores, level=1)
        fine_filtered = self.search_engine._apply_progressive_threshold(candidate_scores, level=2)
        
        # Coarser levels should filter more aggressively (keep fewer candidates)
        assert len(coarse_filtered) <= len(medium_filtered)
        assert len(medium_filtered) <= len(fine_filtered)
        
        # All should keep the best candidate
        assert 0 in coarse_filtered
        assert 0 in medium_filtered
        assert 0 in fine_filtered
    
    def test_adaptive_threshold_statistical_properties(self):
        """Test that adaptive thresholds have correct statistical properties."""
        # Create scores with known statistical properties
        scores = [(i, score) for i, score in enumerate([0.1, 0.3, 0.5, 0.7, 0.9])]
        # Mean = 0.5, Median = 0.5, Std ≈ 0.283
        
        coarse_threshold = self.search_engine._calculate_adaptive_threshold(scores, level=0)
        fine_threshold = self.search_engine._calculate_adaptive_threshold(scores, level=2)
        
        # Coarse threshold should be above mean
        assert coarse_threshold > 0.5
        
        # Fine threshold should be below or at mean
        assert fine_threshold <= 0.5
        
        # Both should be reasonable
        assert 0.1 <= coarse_threshold <= 0.9
        assert 0.1 <= fine_threshold <= 0.9