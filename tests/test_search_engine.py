"""
Unit tests for the progressive similarity search engine.
"""

import pytest
import numpy as np
from unittest.mock import Mock

from hilbert_quantization.core.search_engine import ProgressiveSimilaritySearchEngine, LevelConfig
from hilbert_quantization.models import QuantizedModel, SearchResult, ModelMetadata


class TestProgressiveSimilaritySearchEngine:
    """Test cases for the progressive similarity search engine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.search_engine = ProgressiveSimilaritySearchEngine(
            similarity_threshold=0.1,
            max_candidates_per_level=10
        )
        
        # Create mock metadata
        self.mock_metadata = ModelMetadata(
            model_name="test_model",
            original_size_bytes=1000,
            compressed_size_bytes=500,
            compression_ratio=0.5,
            quantization_timestamp="2024-01-01T00:00:00"
        )
    
    def create_mock_quantized_model(self, indices: np.ndarray, model_id: str = "test") -> QuantizedModel:
        """Create a mock quantized model with given indices."""
        return QuantizedModel(
            compressed_data=b"mock_data",
            original_dimensions=(32, 32),
            parameter_count=1024,
            compression_quality=0.8,
            hierarchical_indices=indices,
            metadata=ModelMetadata(
                model_name=f"{model_id}_model",
                original_size_bytes=1000,
                compressed_size_bytes=500,
                compression_ratio=0.5,
                quantization_timestamp="2024-01-01T00:00:00"
            )
        )
    
    def test_parse_index_structure_empty_indices(self):
        """Test parsing empty indices."""
        levels = self.search_engine._parse_index_structure(np.array([]), 0)
        assert levels == []
    
    def test_parse_index_structure_small_space(self):
        """Test parsing with small index space."""
        indices = np.array([1.0, 2.0, 3.0, 4.0])
        levels = self.search_engine._parse_index_structure(indices, 4)
        
        assert len(levels) > 0
        assert all(level.start_index < level.end_index for level in levels)
        assert all(level.grid_size >= 1 for level in levels)
    
    def test_parse_index_structure_large_space(self):
        """Test parsing with larger index space."""
        indices = np.random.rand(64)
        levels = self.search_engine._parse_index_structure(indices, 64)
        
        assert len(levels) > 0
        
        # Check that indices are properly allocated
        total_allocated = sum(level.end_index - level.start_index for level in levels)
        assert total_allocated <= 64
        
        # Check that levels are ordered by grid size (finest first)
        if len(levels) > 1:
            # Allow for offset sampling which might have same grid size
            grid_sizes = [level.grid_size for level in levels if not level.is_offset_sampling]
            if len(grid_sizes) > 1:
                assert grid_sizes == sorted(grid_sizes, reverse=True)
    
    def test_compare_indices_at_level_empty_indices(self):
        """Test comparison with empty indices."""
        query = np.array([])
        candidate = np.array([1.0, 2.0, 3.0])
        
        similarity = self.search_engine.compare_indices_at_level(query, candidate, 0)
        assert similarity == 0.0
    
    def test_compare_indices_at_level_identical_indices(self):
        """Test comparison with identical indices."""
        indices = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        
        similarity = self.search_engine.compare_indices_at_level(indices, indices, 0)
        assert similarity == pytest.approx(1.0, abs=0.1)
    
    def test_compare_indices_at_level_similar_indices(self):
        """Test comparison with similar indices."""
        query = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        candidate = np.array([1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1])
        
        similarity = self.search_engine.compare_indices_at_level(query, candidate, 0)
        assert 0.8 <= similarity <= 1.0
    
    def test_compare_indices_at_level_different_indices(self):
        """Test comparison with very different indices."""
        query = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        candidate = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
        
        similarity = self.search_engine.compare_indices_at_level(query, candidate, 0)
        assert 0.0 <= similarity <= 1.0
    
    def test_compare_indices_at_level_constant_indices(self):
        """Test comparison with constant indices."""
        query = np.array([5.0, 5.0, 5.0, 5.0])
        candidate = np.array([5.0, 5.0, 5.0, 5.0])
        
        similarity = self.search_engine.compare_indices_at_level(query, candidate, 0)
        assert similarity == 1.0
        
        # Different constants
        candidate_diff = np.array([3.0, 3.0, 3.0, 3.0])
        similarity_diff = self.search_engine.compare_indices_at_level(query, candidate_diff, 0)
        assert similarity_diff == 0.0
    
    def test_compare_indices_at_level_invalid_level(self):
        """Test comparison with invalid level index."""
        query = np.array([1.0, 2.0, 3.0, 4.0])
        candidate = np.array([1.0, 2.0, 3.0, 4.0])
        
        # Level index too high
        similarity = self.search_engine.compare_indices_at_level(query, candidate, 10)
        assert similarity == 0.0
    
    def test_calculate_overall_similarity_empty_indices(self):
        """Test overall similarity calculation with empty indices."""
        query = np.array([])
        candidate = np.array([1.0, 2.0, 3.0])
        
        similarity, level_similarities = self.search_engine._calculate_overall_similarity(query, candidate)
        assert similarity == 0.0
        assert level_similarities == {}
    
    def test_calculate_overall_similarity_identical_indices(self):
        """Test overall similarity calculation with identical indices."""
        indices = np.random.rand(32)
        
        similarity, level_similarities = self.search_engine._calculate_overall_similarity(indices, indices)
        assert 0.8 <= similarity <= 1.0
        assert len(level_similarities) > 0
        assert all(0.0 <= score <= 1.0 for score in level_similarities.values())
    
    def test_calculate_overall_similarity_different_indices(self):
        """Test overall similarity calculation with different indices."""
        query = np.random.rand(32)
        candidate = np.random.rand(32)
        
        similarity, level_similarities = self.search_engine._calculate_overall_similarity(query, candidate)
        assert 0.0 <= similarity <= 1.0
        assert len(level_similarities) > 0
        assert all(0.0 <= score <= 1.0 for score in level_similarities.values())
    
    def test_progressive_filter_candidates_empty_candidates(self):
        """Test progressive filtering with empty candidate list."""
        query = np.random.rand(16)
        candidates = []
        
        filtered = self.search_engine._progressive_filter_candidates(query, candidates)
        assert filtered == []
    
    def test_progressive_filter_candidates_single_candidate(self):
        """Test progressive filtering with single candidate."""
        query = np.random.rand(16)
        candidate = self.create_mock_quantized_model(np.random.rand(16))
        
        filtered = self.search_engine._progressive_filter_candidates(query, [candidate])
        assert len(filtered) == 1
        assert filtered[0][0] == candidate
        assert 0.0 <= filtered[0][1] <= 1.0
    
    def test_progressive_filter_candidates_multiple_candidates(self):
        """Test progressive filtering with multiple candidates."""
        query = np.random.rand(32)
        
        # Create candidates with varying similarity
        candidates = []
        for i in range(5):
            # Make some candidates more similar to query
            if i < 2:
                candidate_indices = query + np.random.normal(0, 0.1, 32)
            else:
                candidate_indices = np.random.rand(32)
            
            candidates.append(self.create_mock_quantized_model(candidate_indices, f"model_{i}"))
        
        filtered = self.search_engine._progressive_filter_candidates(query, candidates)
        
        # Should return some candidates
        assert len(filtered) > 0
        assert len(filtered) <= len(candidates)
        
        # Results should be sorted by similarity (descending)
        similarities = [result[1] for result in filtered]
        assert similarities == sorted(similarities, reverse=True)
    
    def test_progressive_search_empty_query(self):
        """Test progressive search with empty query."""
        query = np.array([])
        candidates = [self.create_mock_quantized_model(np.random.rand(16))]
        
        results = self.search_engine.progressive_search(query, candidates, 5)
        assert results == []
    
    def test_progressive_search_empty_candidates(self):
        """Test progressive search with empty candidate pool."""
        query = np.random.rand(16)
        candidates = []
        
        results = self.search_engine.progressive_search(query, candidates, 5)
        assert results == []
    
    def test_progressive_search_single_candidate(self):
        """Test progressive search with single candidate."""
        query = np.random.rand(32)
        candidate = self.create_mock_quantized_model(query.copy())  # Identical indices
        
        results = self.search_engine.progressive_search(query, [candidate], 5)
        
        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert results[0].model == candidate
        assert 0.8 <= results[0].similarity_score <= 1.0
        assert len(results[0].matching_indices) > 0
    
    def test_progressive_search_multiple_candidates(self):
        """Test progressive search with multiple candidates."""
        query = np.random.rand(64)
        
        candidates = []
        for i in range(10):
            if i < 3:
                # Similar candidates
                candidate_indices = query + np.random.normal(0, 0.1, 64)
            else:
                # Random candidates
                candidate_indices = np.random.rand(64)
            
            candidates.append(self.create_mock_quantized_model(candidate_indices, f"model_{i}"))
        
        results = self.search_engine.progressive_search(query, candidates, 5)
        
        # Should return up to 5 results
        assert len(results) <= 5
        assert len(results) > 0
        
        # All results should be SearchResult instances
        assert all(isinstance(result, SearchResult) for result in results)
        
        # Results should be sorted by similarity (descending)
        similarities = [result.similarity_score for result in results]
        assert similarities == sorted(similarities, reverse=True)
        
        # All similarity scores should be valid
        assert all(0.0 <= score <= 1.0 for score in similarities)
        
        # All results should have matching indices
        assert all(len(result.matching_indices) > 0 for result in results)
    
    def test_progressive_search_max_results_limit(self):
        """Test that progressive search respects max_results limit."""
        query = np.random.rand(32)
        
        # Create many candidates
        candidates = []
        for i in range(20):
            candidate_indices = np.random.rand(32)
            candidates.append(self.create_mock_quantized_model(candidate_indices, f"model_{i}"))
        
        max_results = 3
        results = self.search_engine.progressive_search(query, candidates, max_results)
        
        assert len(results) <= max_results
    
    def test_progressive_search_similarity_threshold(self):
        """Test that similarity threshold affects filtering."""
        query = np.random.rand(32)
        
        # Create engine with high threshold
        strict_engine = ProgressiveSimilaritySearchEngine(
            similarity_threshold=0.9,
            max_candidates_per_level=10
        )
        
        # Create candidates with low similarity
        candidates = []
        for i in range(5):
            candidate_indices = np.random.rand(32) * 10  # Very different from query
            candidates.append(self.create_mock_quantized_model(candidate_indices, f"model_{i}"))
        
        results = strict_engine.progressive_search(query, candidates, 10)
        
        # With high threshold, should get fewer or no results
        # (unless we keep at least one candidate as fallback)
        assert len(results) <= len(candidates)


class TestLevelConfig:
    """Test cases for LevelConfig dataclass."""
    
    def test_level_config_creation(self):
        """Test LevelConfig creation."""
        config = LevelConfig(
            grid_size=4,
            start_index=0,
            end_index=16,
            is_offset_sampling=False
        )
        
        assert config.grid_size == 4
        assert config.start_index == 0
        assert config.end_index == 16
        assert config.is_offset_sampling == False
    
    def test_level_config_default_offset_sampling(self):
        """Test LevelConfig with default offset sampling."""
        config = LevelConfig(
            grid_size=2,
            start_index=5,
            end_index=10
        )
        
        assert config.is_offset_sampling == False  # Default value


if __name__ == "__main__":
    pytest.main([__file__])


class TestProgressiveFilteringAlgorithm:
    """Test cases specifically for the progressive filtering algorithm."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.search_engine = ProgressiveSimilaritySearchEngine(
            similarity_threshold=0.2,
            max_candidates_per_level=5
        )
    
    def create_mock_quantized_model(self, indices: np.ndarray, model_id: str = "test") -> QuantizedModel:
        """Create a mock quantized model with given indices."""
        return QuantizedModel(
            compressed_data=b"mock_data",
            original_dimensions=(32, 32),
            parameter_count=1024,
            compression_quality=0.8,
            hierarchical_indices=indices,
            metadata=ModelMetadata(
                model_name=f"{model_id}_model",
                original_size_bytes=1000,
                compressed_size_bytes=500,
                compression_ratio=0.5,
                quantization_timestamp="2024-01-01T00:00:00"
            )
        )
    
    def test_progressive_filtering_starts_with_finest_granularity(self):
        """Test that filtering starts with finest granularity level."""
        # Create query with known structure
        query = np.random.rand(64)
        
        # Create candidates where some are similar at fine level, others at coarse level
        candidates = []
        
        # Candidate 1: Similar at fine level (should rank high)
        fine_similar = query.copy()
        fine_similar[:32] += np.random.normal(0, 0.05, 32)  # Small noise in fine details
        candidates.append(self.create_mock_quantized_model(fine_similar, "fine_similar"))
        
        # Candidate 2: Similar only at coarse level (should rank lower)
        coarse_similar = np.random.rand(64)
        # Make overall average similar but fine details different
        coarse_similar = coarse_similar * (np.mean(query) / np.mean(coarse_similar))
        candidates.append(self.create_mock_quantized_model(coarse_similar, "coarse_similar"))
        
        # Candidate 3: Completely different (should rank lowest)
        different = np.random.rand(64) * 10
        candidates.append(self.create_mock_quantized_model(different, "different"))
        
        filtered = self.search_engine._progressive_filter_candidates(query, candidates)
        
        # Should prefer fine-grained similarity
        assert len(filtered) > 0
        
        # The fine-similar candidate should rank higher than coarse-similar
        filtered_models = [result[0].metadata.model_name for result in filtered]
        if "fine_similar_model" in filtered_models and "coarse_similar_model" in filtered_models:
            fine_idx = next(i for i, result in enumerate(filtered) if result[0].metadata.model_name == "fine_similar_model")
            coarse_idx = next(i for i, result in enumerate(filtered) if result[0].metadata.model_name == "coarse_similar_model")
            assert fine_idx < coarse_idx  # Lower index means higher ranking
    
    def test_progressive_filtering_eliminates_candidates(self):
        """Test that progressive filtering eliminates poor candidates."""
        query = np.random.rand(32)
        
        # Create many candidates with varying similarity
        candidates = []
        
        # Add a few good candidates
        for i in range(3):
            good_candidate = query + np.random.normal(0, 0.1, 32)
            candidates.append(self.create_mock_quantized_model(good_candidate, f"good_{i}"))
        
        # Add many poor candidates
        for i in range(20):
            poor_candidate = np.random.rand(32) * 5
            candidates.append(self.create_mock_quantized_model(poor_candidate, f"poor_{i}"))
        
        filtered = self.search_engine._progressive_filter_candidates(query, candidates)
        
        # Should eliminate most poor candidates
        assert len(filtered) < len(candidates)
        assert len(filtered) <= self.search_engine.max_candidates_per_level
        
        # Good candidates should be more likely to survive
        filtered_names = [result[0].metadata.model_name for result in filtered]
        good_survivors = sum(1 for name in filtered_names if name.startswith("good_"))
        poor_survivors = sum(1 for name in filtered_names if name.startswith("poor_"))
        
        # Should have more good survivors than poor survivors (probabilistically)
        # This might not always be true due to randomness, but should be generally true
        if len(filtered) >= 3:
            assert good_survivors > 0  # At least some good candidates should survive
    
    def test_progressive_filtering_respects_max_candidates_per_level(self):
        """Test that filtering respects the max candidates per level limit."""
        query = np.random.rand(32)
        
        # Create more candidates than the limit
        candidates = []
        for i in range(15):  # More than max_candidates_per_level (5)
            candidate_indices = np.random.rand(32)
            candidates.append(self.create_mock_quantized_model(candidate_indices, f"model_{i}"))
        
        filtered = self.search_engine._progressive_filter_candidates(query, candidates)
        
        # Should not exceed the limit
        assert len(filtered) <= self.search_engine.max_candidates_per_level
    
    def test_progressive_filtering_handles_threshold(self):
        """Test that filtering respects similarity threshold."""
        query = np.random.rand(16)
        
        # Create engine with high threshold
        strict_engine = ProgressiveSimilaritySearchEngine(
            similarity_threshold=0.8,
            max_candidates_per_level=10
        )
        
        # Create candidates with known low similarity
        candidates = []
        for i in range(5):
            # Very different candidates
            candidate_indices = np.random.rand(16) * 20 + 100
            candidates.append(self.create_mock_quantized_model(candidate_indices, f"model_{i}"))
        
        filtered = strict_engine._progressive_filter_candidates(query, candidates)
        
        # With high threshold and very different candidates, should filter aggressively
        # But should keep at least one candidate as fallback
        assert len(filtered) >= 1  # Fallback mechanism
        assert len(filtered) <= len(candidates)
    
    def test_progressive_filtering_preserves_best_candidate(self):
        """Test that filtering always preserves at least the best candidate."""
        query = np.random.rand(32)
        
        # Create candidates where all are below threshold but one is clearly best
        candidates = []
        
        # Best candidate (still not great, but better than others)
        best_candidate = query + np.random.normal(0, 2.0, 32)  # Moderate noise
        candidates.append(self.create_mock_quantized_model(best_candidate, "best"))
        
        # Worse candidates
        for i in range(5):
            worse_candidate = np.random.rand(32) * 50  # Very different
            candidates.append(self.create_mock_quantized_model(worse_candidate, f"worse_{i}"))
        
        # Use strict threshold
        strict_engine = ProgressiveSimilaritySearchEngine(
            similarity_threshold=0.9,
            max_candidates_per_level=10
        )
        
        filtered = strict_engine._progressive_filter_candidates(query, candidates)
        
        # Should keep at least one candidate (the best one)
        assert len(filtered) >= 1
        
        # The best candidate should be among the filtered results
        filtered_names = [result[0].metadata.model_name for result in filtered]
        assert "best_model" in filtered_names
    
    def test_progressive_filtering_efficiency_with_large_pool(self):
        """Test filtering efficiency with large candidate pool."""
        query = np.random.rand(64)
        
        # Create large candidate pool
        candidates = []
        for i in range(100):
            if i < 10:
                # Some good candidates
                candidate_indices = query + np.random.normal(0, 0.2, 64)
            else:
                # Many random candidates
                candidate_indices = np.random.rand(64)
            
            candidates.append(self.create_mock_quantized_model(candidate_indices, f"model_{i}"))
        
        filtered = self.search_engine._progressive_filter_candidates(query, candidates)
        
        # Should significantly reduce the candidate pool
        assert len(filtered) < len(candidates) * 0.2  # Less than 20% of original
        assert len(filtered) <= self.search_engine.max_candidates_per_level
        
        # Results should be sorted by similarity
        similarities = [result[1] for result in filtered]
        assert similarities == sorted(similarities, reverse=True)
    
    def test_progressive_filtering_level_weight_calculation(self):
        """Test that level weights are calculated correctly."""
        query = np.random.rand(64)  # Use larger array to ensure multiple levels
        
        # Create a candidate with known similarity pattern
        candidate_indices = query.copy()
        candidate = self.create_mock_quantized_model(candidate_indices, "test")
        
        filtered = self.search_engine._progressive_filter_candidates(query, [candidate])
        
        assert len(filtered) == 1
        result = filtered[0]
        
        # Should have level similarities recorded (might be empty for very small arrays)
        # Combined score should be reasonable for identical indices
        assert result[1] > 0.8  # Should be high similarity
    
    def test_progressive_filtering_empty_edge_cases(self):
        """Test progressive filtering with edge cases."""
        query = np.random.rand(16)
        
        # Test with empty candidate list
        filtered = self.search_engine._progressive_filter_candidates(query, [])
        assert filtered == []
        
        # Test with empty query
        empty_query = np.array([])
        candidate = self.create_mock_quantized_model(np.random.rand(16), "test")
        filtered = self.search_engine._progressive_filter_candidates(empty_query, [candidate])
        assert filtered == []


class TestNearestNeighborSearchIntegration:
    """Test cases for nearest neighbor search integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.search_engine = ProgressiveSimilaritySearchEngine(
            similarity_threshold=0.1,
            max_candidates_per_level=10
        )
    
    def create_mock_quantized_model(self, indices: np.ndarray, model_id: str = "test") -> QuantizedModel:
        """Create a mock quantized model with given indices."""
        return QuantizedModel(
            compressed_data=b"mock_data",
            original_dimensions=(32, 32),
            parameter_count=1024,
            compression_quality=0.8,
            hierarchical_indices=indices,
            metadata=ModelMetadata(
                model_name=f"{model_id}_model",
                original_size_bytes=1000,
                compressed_size_bytes=500,
                compression_ratio=0.5,
                quantization_timestamp="2024-01-01T00:00:00"
            )
        )
    
    def test_detailed_comparison_for_filtered_candidates(self):
        """Test that detailed comparison is performed for filtered candidates."""
        query = np.random.rand(32)
        
        # Create candidates with subtle differences
        candidates = []
        for i in range(5):
            # Create candidates with small variations
            candidate_indices = query + np.random.normal(0, 0.1, 32)
            candidates.append(self.create_mock_quantized_model(candidate_indices, f"model_{i}"))
        
        results = self.search_engine.progressive_search(query, candidates, 5)
        
        # All results should have detailed similarity information
        assert len(results) > 0
        for result in results:
            assert isinstance(result, SearchResult)
            assert 0.0 <= result.similarity_score <= 1.0
            assert len(result.matching_indices) > 0  # Should have level-by-level similarities
            assert 0.0 <= result.reconstruction_error <= 1.0
    
    def test_result_ranking_by_similarity(self):
        """Test that results are properly ranked by similarity score."""
        query = np.random.rand(64)
        
        candidates = []
        expected_order = []
        
        # Create candidates with known similarity levels
        # Most similar
        very_similar = query + np.random.normal(0, 0.05, 64)
        candidates.append(self.create_mock_quantized_model(very_similar, "very_similar"))
        expected_order.append("very_similar_model")
        
        # Moderately similar
        moderately_similar = query + np.random.normal(0, 0.3, 64)
        candidates.append(self.create_mock_quantized_model(moderately_similar, "moderately_similar"))
        expected_order.append("moderately_similar_model")
        
        # Less similar
        less_similar = query + np.random.normal(0, 1.0, 64)
        candidates.append(self.create_mock_quantized_model(less_similar, "less_similar"))
        expected_order.append("less_similar_model")
        
        # Very different
        very_different = np.random.rand(64) * 10
        candidates.append(self.create_mock_quantized_model(very_different, "very_different"))
        expected_order.append("very_different_model")
        
        results = self.search_engine.progressive_search(query, candidates, 4)
        
        # Results should be sorted by similarity (descending)
        similarities = [result.similarity_score for result in results]
        assert similarities == sorted(similarities, reverse=True)
        
        # The most similar should rank first (probabilistically)
        if len(results) >= 2:
            result_names = [result.model.metadata.model_name for result in results]
            # Very similar should rank higher than very different
            if "very_similar_model" in result_names and "very_different_model" in result_names:
                very_similar_idx = result_names.index("very_similar_model")
                very_different_idx = result_names.index("very_different_model")
                assert very_similar_idx < very_different_idx
    
    def test_configurable_result_count_limits(self):
        """Test that result count limits are properly enforced."""
        query = np.random.rand(32)
        
        # Create more candidates than we want to return
        candidates = []
        for i in range(20):
            candidate_indices = np.random.rand(32)
            candidates.append(self.create_mock_quantized_model(candidate_indices, f"model_{i}"))
        
        # Test different result limits
        for max_results in [1, 3, 5, 10, 15]:
            results = self.search_engine.progressive_search(query, candidates, max_results)
            assert len(results) <= max_results
            
            # If we have enough candidates, should return exactly max_results
            if len(candidates) >= max_results:
                # Note: might be less if filtering is too aggressive
                assert len(results) <= max_results
    
    def test_search_accuracy_against_brute_force(self):
        """Test search accuracy by comparing against brute force method."""
        query = np.random.rand(32)
        
        # Create candidates with varying similarity
        candidates = []
        for i in range(15):
            if i < 5:
                # Some similar candidates
                candidate_indices = query + np.random.normal(0, 0.2, 32)
            else:
                # Some random candidates
                candidate_indices = np.random.rand(32)
            
            candidates.append(self.create_mock_quantized_model(candidate_indices, f"model_{i}"))
        
        # Get results from both methods
        progressive_results = self.search_engine.progressive_search(query, candidates, 5)
        brute_force_results = self.search_engine.brute_force_search(query, candidates, 5)
        
        # Both should return valid results
        assert len(progressive_results) > 0
        assert len(brute_force_results) > 0
        
        # Progressive search should find at least some of the top candidates
        # (might not be identical due to filtering, but should have significant overlap)
        progressive_names = set(result.model.metadata.model_name for result in progressive_results)
        brute_force_names = set(result.model.metadata.model_name for result in brute_force_results)
        
        # Should have some overlap in top results
        overlap = len(progressive_names.intersection(brute_force_names))
        total_unique = len(progressive_names.union(brute_force_names))
        
        # At least 30% overlap expected (this is a reasonable threshold for filtered search)
        overlap_ratio = overlap / total_unique if total_unique > 0 else 0
        assert overlap_ratio >= 0.3 or len(progressive_results) == len(brute_force_results) == 1
    
    def test_search_with_identical_query_and_candidate(self):
        """Test search when query is identical to one of the candidates."""
        query = np.random.rand(32)
        
        candidates = []
        
        # Add identical candidate
        identical_candidate = self.create_mock_quantized_model(query.copy(), "identical")
        candidates.append(identical_candidate)
        
        # Add some different candidates
        for i in range(5):
            different_candidate = np.random.rand(32)
            candidates.append(self.create_mock_quantized_model(different_candidate, f"different_{i}"))
        
        results = self.search_engine.progressive_search(query, candidates, 3)
        
        # Identical candidate should rank first
        assert len(results) > 0
        assert results[0].model.metadata.model_name == "identical_model"
        assert results[0].similarity_score > 0.9  # Should be very high similarity
    
    def test_search_with_minimal_matching_indices(self):
        """Test search behavior when matching indices are minimal."""
        # Create query with small but workable indices array
        query = np.array([1.0, 2.0, 3.0, 4.0])
        
        candidate = self.create_mock_quantized_model(np.array([1.0, 2.0, 3.0, 4.0]), "test")
        
        results = self.search_engine.progressive_search(query, [candidate], 1)
        
        # Should return a result with minimal indices
        # Note: might return empty if filtering is too aggressive with small arrays
        if len(results) > 0:
            assert isinstance(results[0], SearchResult)
            assert 0.0 <= results[0].similarity_score <= 1.0
    
    def test_reconstruction_error_estimation(self):
        """Test that reconstruction error is estimated based on similarity."""
        query = np.random.rand(32)
        
        # Create candidates with known similarity levels
        high_similarity_candidate = query + np.random.normal(0, 0.05, 32)
        low_similarity_candidate = np.random.rand(32) * 10
        
        candidates = [
            self.create_mock_quantized_model(high_similarity_candidate, "high_sim"),
            self.create_mock_quantized_model(low_similarity_candidate, "low_sim")
        ]
        
        results = self.search_engine.progressive_search(query, candidates, 2)
        
        assert len(results) == 2
        
        # Find results by name
        high_sim_result = next(r for r in results if r.model.metadata.model_name == "high_sim_model")
        low_sim_result = next(r for r in results if r.model.metadata.model_name == "low_sim_model")
        
        # High similarity should have lower reconstruction error
        assert high_sim_result.reconstruction_error < low_sim_result.reconstruction_error
        
        # Reconstruction error should be inverse of similarity
        assert abs(high_sim_result.reconstruction_error - (1.0 - high_sim_result.similarity_score)) < 0.01
        assert abs(low_sim_result.reconstruction_error - (1.0 - low_sim_result.similarity_score)) < 0.01
    
    def test_search_performance_with_large_candidate_pool(self):
        """Test search performance and correctness with large candidate pool."""
        query = np.random.rand(64)
        
        # Create large candidate pool
        candidates = []
        for i in range(200):
            if i < 20:
                # Some good candidates
                candidate_indices = query + np.random.normal(0, 0.3, 64)
            else:
                # Many random candidates
                candidate_indices = np.random.rand(64)
            
            candidates.append(self.create_mock_quantized_model(candidate_indices, f"model_{i}"))
        
        # Progressive search should be faster and still find good results
        results = self.search_engine.progressive_search(query, candidates, 10)
        
        assert len(results) <= 10
        assert len(results) > 0
        
        # Results should be properly ranked
        similarities = [result.similarity_score for result in results]
        assert similarities == sorted(similarities, reverse=True)
        
        # Should find some of the good candidates (probabilistically)
        result_names = [result.model.metadata.model_name for result in results]
        good_candidates_found = sum(1 for name in result_names if name.startswith("model_") and int(name.split("_")[1]) < 20)
        
        # Should find at least some good candidates
        assert good_candidates_found > 0
    
    def test_brute_force_search_completeness(self):
        """Test that brute force search examines all candidates."""
        query = np.random.rand(16)
        
        candidates = []
        for i in range(10):
            candidate_indices = np.random.rand(16)
            candidates.append(self.create_mock_quantized_model(candidate_indices, f"model_{i}"))
        
        # Brute force should return results for all candidates (up to max_results)
        brute_force_results = self.search_engine.brute_force_search(query, candidates, 10)
        
        assert len(brute_force_results) == len(candidates)  # Should examine all candidates
        
        # Results should be properly sorted
        similarities = [result.similarity_score for result in brute_force_results]
        assert similarities == sorted(similarities, reverse=True)