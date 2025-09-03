"""
Unit tests for similarity calculation with cached frames functionality.

This module tests the detailed embedding similarity calculation methods that implement
requirements 4.4, 4.7, and 4.8 for comprehensive similarity scoring and result ranking.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from hilbert_quantization.rag.search.engine import RAGSearchEngineImpl
from hilbert_quantization.rag.config import RAGConfig
from hilbert_quantization.rag.models import DocumentSearchResult, DocumentChunk


class TestSimilarityCalculation:
    """Test suite for similarity calculation with cached frames methods."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = RAGConfig()
        self.search_engine = RAGSearchEngineImpl(self.config)
    
    def test_calculate_embedding_similarity_empty_inputs(self):
        """Test similarity calculation with empty inputs."""
        # Empty query embedding
        empty_query = np.array([])
        cached_frames = {1: np.random.rand(64, 64)}
        
        result = self.search_engine.calculate_embedding_similarity(empty_query, cached_frames)
        assert result == []
        
        # Empty cached frames
        query_embedding = np.random.rand(64, 64)
        empty_frames = {}
        
        result = self.search_engine.calculate_embedding_similarity(query_embedding, empty_frames)
        assert result == []
    
    def test_calculate_embedding_similarity_basic(self):
        """Test basic embedding similarity calculation."""
        # Create query embedding with hierarchical indices
        query_embedding = np.zeros((68, 64), dtype=np.float32)
        query_embedding[:64, :] = np.random.rand(64, 64)
        query_embedding[64, :8] = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        
        # Create similar cached frame
        similar_frame = np.zeros((68, 64), dtype=np.float32)
        similar_frame[:64, :] = query_embedding[:64, :] * 1.1  # Slightly different
        similar_frame[64, :8] = np.array([1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1])
        
        # Create different cached frame
        different_frame = np.zeros((68, 64), dtype=np.float32)
        different_frame[:64, :] = np.random.rand(64, 64)
        different_frame[64, :8] = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
        
        cached_frames = {
            1: similar_frame,
            2: different_frame
        }
        
        similarities = self.search_engine.calculate_embedding_similarity(
            query_embedding, cached_frames
        )
        
        # Should return sorted similarities
        assert len(similarities) == 2
        assert all(isinstance(item, tuple) and len(item) == 2 for item in similarities)
        assert all(0.0 <= score <= 1.0 for _, score in similarities)
        
        # Similar frame should have higher similarity
        frame_scores = {frame_num: score for frame_num, score in similarities}
        assert frame_scores[1] > frame_scores[2]
    
    def test_comprehensive_similarity_calculation(self):
        """Test comprehensive similarity calculation with multiple metrics."""
        # Create test embeddings
        query_embedding = np.random.rand(64, 64).astype(np.float32)
        query_indices = [
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([1.1, 2.1, 3.1, 4.1])
        ]
        
        candidate_frame = np.random.rand(64, 64).astype(np.float32)
        
        # Mock hierarchical index extraction
        with patch.object(self.search_engine, '_extract_hierarchical_indices') as mock_extract:
            mock_extract.return_value = [
                np.array([1.2, 2.2, 3.2, 4.2]),
                np.array([1.3, 2.3, 3.3, 4.3])
            ]
            
            similarity = self.search_engine._calculate_comprehensive_similarity(
                query_embedding, query_indices, candidate_frame, frame_number=1
            )
        
        # Should return valid similarity score
        assert 0.0 <= similarity <= 1.0
    
    def test_pad_indices_to_length(self):
        """Test padding of hierarchical indices to target length."""
        indices = [
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0])
        ]
        
        padded = self.search_engine._pad_indices_to_length(indices, target_length=4)
        
        # Should have correct shape
        assert padded.shape == (4, 3)  # 4 levels, max width 3
        
        # First two rows should contain original data
        np.testing.assert_array_equal(padded[0, :3], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(padded[1, :2], [4.0, 5.0])
        
        # Remaining should be zeros
        assert np.all(padded[2:, :] == 0)
        assert np.all(padded[1, 2:] == 0)
    
    def test_pad_indices_empty_list(self):
        """Test padding with empty indices list."""
        padded = self.search_engine._pad_indices_to_length([], target_length=3)
        
        assert padded.shape == (3, 1)
        assert np.all(padded == 0)
    
    def test_extract_original_embedding_2d(self):
        """Test extraction of original embedding from 2D enhanced representation."""
        # Create enhanced embedding with index rows
        enhanced = np.random.rand(68, 64).astype(np.float32)
        
        # Mock detection to return first 64 rows as original
        with patch.object(self.search_engine, '_detect_original_embedding_height', return_value=64):
            original = self.search_engine._extract_original_embedding(enhanced)
        
        assert original.shape == (64, 64)
        np.testing.assert_array_equal(original, enhanced[:64, :])
    
    def test_extract_original_embedding_1d(self):
        """Test extraction from 1D embedding (no enhancement)."""
        embedding_1d = np.random.rand(256).astype(np.float32)
        
        original = self.search_engine._extract_original_embedding(embedding_1d)
        
        np.testing.assert_array_equal(original, embedding_1d)
    
    def test_embedding_cosine_similarity_identical(self):
        """Test cosine similarity with identical embeddings."""
        embedding = np.random.rand(32, 32).astype(np.float32)
        
        similarity = self.search_engine._calculate_embedding_cosine_similarity(
            embedding, embedding
        )
        
        assert similarity == 1.0
    
    def test_embedding_cosine_similarity_orthogonal(self):
        """Test cosine similarity with orthogonal embeddings."""
        embedding1 = np.zeros((4, 4))
        embedding1[0, 0] = 1.0
        
        embedding2 = np.zeros((4, 4))
        embedding2[1, 1] = 1.0
        
        similarity = self.search_engine._calculate_embedding_cosine_similarity(
            embedding1, embedding2
        )
        
        # Orthogonal vectors should have similarity around 0.5 (normalized)
        assert 0.4 <= similarity <= 0.6
    
    def test_embedding_cosine_similarity_empty(self):
        """Test cosine similarity with empty embeddings."""
        empty = np.array([])
        non_empty = np.random.rand(16, 16)
        
        similarity = self.search_engine._calculate_embedding_cosine_similarity(
            empty, non_empty
        )
        
        assert similarity == 0.0
    
    def test_embedding_cosine_similarity_different_sizes(self):
        """Test cosine similarity with different sized embeddings."""
        embedding1 = np.random.rand(32, 32)
        embedding2 = np.random.rand(16, 16)
        
        similarity = self.search_engine._calculate_embedding_cosine_similarity(
            embedding1, embedding2
        )
        
        # Should handle size mismatch gracefully
        assert 0.0 <= similarity <= 1.0
    
    def test_spatial_locality_similarity_same_shape(self):
        """Test spatial locality similarity with same shaped embeddings."""
        # Create embeddings with spatial patterns
        embedding1 = np.zeros((32, 32))
        embedding1[:16, :16] = 1.0  # Top-left quadrant
        
        embedding2 = np.zeros((32, 32))
        embedding2[:16, :16] = 0.9  # Similar pattern, slightly different values
        
        similarity = self.search_engine._calculate_spatial_locality_similarity(
            embedding1, embedding2
        )
        
        assert 0.0 <= similarity <= 1.0
        assert similarity > 0.2  # Should be reasonably similar due to spatial pattern
    
    def test_spatial_locality_similarity_different_shapes(self):
        """Test spatial locality similarity with different shaped embeddings."""
        embedding1 = np.random.rand(32, 32)
        embedding2 = np.random.rand(16, 16)
        
        similarity = self.search_engine._calculate_spatial_locality_similarity(
            embedding1, embedding2
        )
        
        assert similarity == 0.0  # Different shapes should return 0
    
    def test_spatial_locality_similarity_1d(self):
        """Test spatial locality similarity with 1D embeddings."""
        embedding1 = np.random.rand(256)
        embedding2 = np.random.rand(256)
        
        similarity = self.search_engine._calculate_spatial_locality_similarity(
            embedding1, embedding2
        )
        
        assert similarity == 0.0  # 1D embeddings should return 0
    
    def test_similarity_weights(self):
        """Test similarity component weights."""
        weights = self.search_engine._get_similarity_weights()
        
        # Should have all required components
        assert 'hierarchical' in weights
        assert 'embedding' in weights
        assert 'spatial' in weights
        
        # Weights should be positive and sum to 1.0
        assert all(w > 0 for w in weights.values())
        assert abs(sum(weights.values()) - 1.0) < 0.01
        
        # Hierarchical should have highest weight
        assert weights['hierarchical'] >= weights['embedding']
        assert weights['hierarchical'] >= weights['spatial']
    
    def test_search_similar_documents_with_caching_basic(self):
        """Test complete search workflow with caching."""
        query_text = "test query"
        max_results = 5
        
        # Mock the progressive filtering and caching
        with patch.object(self.search_engine, 'progressive_hierarchical_search') as mock_filter, \
             patch.object(self.search_engine, 'cache_consecutive_frames') as mock_cache, \
             patch.object(self.search_engine, '_generate_query_embedding') as mock_embed:
            
            # Setup mocks
            mock_filter.return_value = [1, 2, 3, 4, 5]
            mock_cache.return_value = {
                i: np.random.rand(64, 64) for i in range(1, 6)
            }
            mock_embed.return_value = np.random.rand(68, 64)
            
            results = self.search_engine.search_similar_documents_with_caching(
                query_text, max_results=max_results
            )
        
        # Should return results
        assert len(results) <= max_results
        assert all(isinstance(result, DocumentSearchResult) for result in results)
    
    def test_search_similar_documents_with_threshold(self):
        """Test search with similarity threshold filtering."""
        query_text = "test query"
        similarity_threshold = 0.8
        
        # Mock to return low similarity scores
        with patch.object(self.search_engine, 'calculate_embedding_similarity') as mock_calc, \
             patch.object(self.search_engine, '_generate_query_embedding') as mock_embed, \
             patch.object(self.search_engine, 'progressive_hierarchical_search') as mock_filter, \
             patch.object(self.search_engine, 'cache_consecutive_frames') as mock_cache:
            
            mock_embed.return_value = np.random.rand(68, 64)
            mock_filter.return_value = [1, 2, 3]
            mock_cache.return_value = {i: np.random.rand(64, 64) for i in [1, 2, 3]}
            mock_calc.return_value = [(1, 0.9), (2, 0.7), (3, 0.5)]  # Only first passes threshold
            
            results = self.search_engine.search_similar_documents_with_caching(
                query_text, similarity_threshold=similarity_threshold
            )
        
        # Should only return results above threshold
        assert len(results) == 1
        assert results[0].similarity_score >= similarity_threshold
    
    def test_search_without_progressive_filtering(self):
        """Test search without progressive filtering enabled."""
        query_text = "test query"
        
        with patch.object(self.search_engine, 'cache_consecutive_frames') as mock_cache, \
             patch.object(self.search_engine, '_generate_query_embedding') as mock_embed:
            
            mock_embed.return_value = np.random.rand(68, 64)
            mock_cache.return_value = {i: np.random.rand(64, 64) for i in range(5)}
            
            results = self.search_engine.search_similar_documents_with_caching(
                query_text, use_progressive_filtering=False
            )
        
        # Should still return results
        assert isinstance(results, list)
    
    def test_benchmark_search_accuracy_basic(self):
        """Test basic search accuracy benchmarking."""
        test_queries = ["query1", "query2"]
        ground_truth = {
            "query1": [1, 2, 3],
            "query2": [4, 5, 6]
        }
        
        # Mock search results
        with patch.object(self.search_engine, 'search_similar_documents_with_caching') as mock_search:
            def mock_search_func(query, max_results=10):
                if query == "query1":
                    # Perfect match for query1
                    return [
                        self._create_mock_result(1, 0.9),
                        self._create_mock_result(2, 0.8),
                        self._create_mock_result(3, 0.7)
                    ]
                elif query == "query2":
                    # Partial match for query2
                    return [
                        self._create_mock_result(4, 0.9),
                        self._create_mock_result(7, 0.8),  # Not in ground truth
                        self._create_mock_result(5, 0.7)
                    ]
                return []
            
            mock_search.side_effect = mock_search_func
            
            metrics = self.search_engine.benchmark_search_accuracy(
                test_queries, ground_truth, max_results=3
            )
        
        # Should return accuracy metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'queries_tested' in metrics
        
        # All metrics should be between 0 and 1
        assert 0.0 <= metrics['precision'] <= 1.0
        assert 0.0 <= metrics['recall'] <= 1.0
        assert 0.0 <= metrics['f1_score'] <= 1.0
        
        # Should have tested both queries
        assert metrics['queries_tested'] == 2
    
    def test_benchmark_search_accuracy_empty_inputs(self):
        """Test benchmarking with empty inputs."""
        # Empty queries
        metrics = self.search_engine.benchmark_search_accuracy([], {})
        assert metrics == {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
        
        # Queries not in ground truth
        metrics = self.search_engine.benchmark_search_accuracy(
            ["unknown_query"], {"known_query": [1, 2, 3]}
        )
        assert metrics == {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
    
    def test_benchmark_search_accuracy_perfect_results(self):
        """Test benchmarking with perfect search results."""
        test_queries = ["perfect_query"]
        ground_truth = {"perfect_query": [1, 2, 3]}
        
        with patch.object(self.search_engine, 'search_similar_documents_with_caching') as mock_search:
            mock_search.return_value = [
                self._create_mock_result(1, 1.0),
                self._create_mock_result(2, 0.9),
                self._create_mock_result(3, 0.8)
            ]
            
            metrics = self.search_engine.benchmark_search_accuracy(
                test_queries, ground_truth
            )
        
        # Perfect results should have precision and recall of 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1_score'] == 1.0
    
    def test_create_document_search_result(self):
        """Test creation of DocumentSearchResult objects."""
        frame_number = 42
        similarity_score = 0.85
        query_text = "test query"
        
        result = self.search_engine._create_document_search_result(
            frame_number, similarity_score, query_text
        )
        
        # Should create valid DocumentSearchResult
        assert isinstance(result, DocumentSearchResult)
        assert result.frame_number == frame_number
        assert result.similarity_score == similarity_score
        assert result.search_method == "cached_similarity"
        assert isinstance(result.document_chunk, DocumentChunk)
        assert result.cached_neighbors is not None
    
    def test_generate_query_embedding(self):
        """Test query embedding generation."""
        query_text = "test query"
        
        embedding = self.search_engine._generate_query_embedding(query_text)
        
        # Should generate valid embedding with hierarchical indices
        assert embedding.ndim == 2
        assert embedding.shape[0] > 64  # Should have index rows
        assert embedding.shape[1] == 64
        assert embedding.dtype == np.float32
    
    def test_main_search_interface(self):
        """Test main search_similar_documents interface."""
        query_text = "test query"
        max_results = 5
        
        # Mock the underlying search method
        with patch.object(self.search_engine, 'search_similar_documents_with_caching') as mock_search:
            mock_search.return_value = [self._create_mock_result(i, 0.8) for i in range(max_results)]
            
            results = self.search_engine.search_similar_documents(query_text, max_results)
        
        # Should delegate to the caching method
        mock_search.assert_called_once_with(
            query_text,
            max_results=max_results,
            similarity_threshold=0.1,
            use_progressive_filtering=True
        )
        
        assert len(results) == max_results
    
    def _create_mock_result(self, frame_number: int, similarity_score: float) -> DocumentSearchResult:
        """Helper to create mock DocumentSearchResult."""
        chunk = DocumentChunk(
            content=f"Content {frame_number}",
            ipfs_hash=f"hash_{frame_number}",
            source_path=f"doc_{frame_number}.txt",
            start_position=0,
            end_position=100,
            chunk_sequence=frame_number,
            creation_timestamp="2024-01-01T00:00:00Z",
            chunk_size=100
        )
        
        return DocumentSearchResult(
            document_chunk=chunk,
            similarity_score=similarity_score,
            embedding_similarity_score=similarity_score * 0.8,
            hierarchical_similarity_score=similarity_score * 0.9,
            frame_number=frame_number,
            search_method="test",
            cached_neighbors=None
        )
    
    def test_comprehensive_similarity_edge_cases(self):
        """Test comprehensive similarity calculation edge cases."""
        query_embedding = np.random.rand(32, 32)
        query_indices = []  # Empty indices
        candidate_frame = np.random.rand(32, 32)
        
        # Mock to return empty indices for candidate too
        with patch.object(self.search_engine, '_extract_hierarchical_indices', return_value=[]):
            similarity = self.search_engine._calculate_comprehensive_similarity(
                query_embedding, query_indices, candidate_frame, frame_number=1
            )
        
        # Should handle empty indices gracefully
        assert 0.0 <= similarity <= 1.0
    
    def test_spatial_locality_small_embeddings(self):
        """Test spatial locality with very small embeddings."""
        # Small embeddings where window size would be too large
        embedding1 = np.random.rand(2, 2)
        embedding2 = np.random.rand(2, 2)
        
        similarity = self.search_engine._calculate_spatial_locality_similarity(
            embedding1, embedding2
        )
        
        # Should fall back to cosine similarity
        assert 0.0 <= similarity <= 1.0
    
    def test_similarity_calculation_performance(self):
        """Test similarity calculation with larger datasets."""
        # Create larger test data
        query_embedding = np.random.rand(128, 128).astype(np.float32)
        cached_frames = {
            i: np.random.rand(128, 128).astype(np.float32) 
            for i in range(50)  # 50 frames
        }
        
        # Should complete in reasonable time
        similarities = self.search_engine.calculate_embedding_similarity(
            query_embedding, cached_frames
        )
        
        assert len(similarities) == 50
        assert all(0.0 <= score <= 1.0 for _, score in similarities)
        
        # Results should be sorted by similarity (descending)
        scores = [score for _, score in similarities]
        assert scores == sorted(scores, reverse=True)