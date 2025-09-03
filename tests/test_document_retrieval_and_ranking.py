"""
Tests for document retrieval and result ranking implementation.
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch

from hilbert_quantization.rag.models import DocumentChunk, DocumentSearchResult, VideoFrameMetadata
from hilbert_quantization.rag.search.document_retrieval import DocumentRetrievalImpl
from hilbert_quantization.rag.search.result_ranking import ResultRankingSystem
from hilbert_quantization.rag.search.engine import RAGSearchEngineImpl
from hilbert_quantization.rag.video_storage.dual_storage import DualVideoStorageImpl


class TestDocumentRetrieval:
    """Test document retrieval functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = Mock()
        self.config.max_search_results = 10
        self.config.similarity_threshold = 0.1
        self.config.enable_frame_validation = True
        self.config.storage_root = tempfile.mkdtemp()
        self.config.max_frames_per_file = 1000
        self.config.frame_rate = 30.0
        self.config.video_codec = 'mp4v'
        self.config.compression_quality = 0.8
        
        # Create mock dual storage
        self.dual_storage = Mock(spec=DualVideoStorageImpl)
        self.document_retrieval = DocumentRetrievalImpl(self.dual_storage, self.config)
        
        # Create test document chunks
        self.test_chunks = [
            DocumentChunk(
                content="This is test content for chunk 1",
                ipfs_hash="QmTest1234567890abcdef1234567890abcdef123456",
                source_path="/test/document1.txt",
                start_position=0,
                end_position=35,
                chunk_sequence=0,
                creation_timestamp="2024-01-01T00:00:00Z",
                chunk_size=35
            ),
            DocumentChunk(
                content="This is test content for chunk 2",
                ipfs_hash="QmTest1234567890abcdef1234567890abcdef123456",
                source_path="/test/document1.txt",
                start_position=35,
                end_position=70,
                chunk_sequence=1,
                creation_timestamp="2024-01-01T00:00:00Z",
                chunk_size=35
            )
        ]
    
    def test_retrieve_documents_by_frame_numbers(self):
        """Test retrieving documents by frame numbers."""
        # Mock dual storage response
        self.dual_storage.get_document_chunks_by_frame_numbers.return_value = [
            (0, self.test_chunks[0]),
            (1, self.test_chunks[1])
        ]
        self.dual_storage.validate_frame_synchronization.return_value = {
            'validation_passed': True,
            'synchronization_errors': []
        }
        
        # Test retrieval
        frame_numbers = [0, 1]
        results = self.document_retrieval.retrieve_documents_by_frame_numbers(frame_numbers)
        
        assert len(results) == 2
        assert results[0][0] == 0
        assert results[0][1].content == "This is test content for chunk 1"
        assert results[1][0] == 1
        assert results[1][1].content == "This is test content for chunk 2"
    
    def test_retrieve_single_document(self):
        """Test retrieving a single document by frame number."""
        # Mock dual storage response
        self.dual_storage.get_document_chunk.return_value = self.test_chunks[0]
        
        # Test retrieval
        result = self.document_retrieval.retrieve_single_document(0)
        
        assert result is not None
        assert result.content == "This is test content for chunk 1"
        assert result.ipfs_hash == "QmTest1234567890abcdef1234567890abcdef123456"
    
    def test_retrieve_single_document_not_found(self):
        """Test retrieving a document that doesn't exist."""
        # Mock dual storage to raise ValueError
        self.dual_storage.get_document_chunk.side_effect = ValueError("Frame not found")
        
        # Test retrieval
        result = self.document_retrieval.retrieve_single_document(999)
        
        assert result is None
    
    def test_validate_retrieval_synchronization(self):
        """Test frame synchronization validation."""
        # Mock validation response
        validation_result = {
            'total_frames_checked': 2,
            'synchronized_frames': 2,
            'missing_frames': [],
            'synchronization_errors': [],
            'validation_passed': True
        }
        self.dual_storage.validate_frame_synchronization.return_value = validation_result
        
        # Test validation
        result = self.document_retrieval.validate_retrieval_synchronization([0, 1])
        
        assert result['validation_passed'] is True
        assert result['synchronized_frames'] == 2
        assert len(result['synchronization_errors']) == 0
    
    def test_get_retrieval_statistics(self):
        """Test retrieval statistics calculation."""
        # Mock metadata
        metadata = VideoFrameMetadata(
            frame_index=0,
            chunk_id="test_chunk_1",
            ipfs_hash="QmTest1234567890abcdef1234567890abcdef123456",
            source_document="/test/document1.txt",
            compression_quality=0.8,
            hierarchical_indices=[],
            embedding_model="test_model",
            frame_timestamp=1640995200.0,
            chunk_metadata=self.test_chunks[0]
        )
        
        # Mock dual storage response
        self.dual_storage._get_frame_metadata_by_number.return_value = metadata
        
        retrieved_docs = [(0, self.test_chunks[0], metadata)]
        self.document_retrieval.retrieve_documents_with_metadata = Mock(return_value=retrieved_docs)
        
        # Test statistics
        stats = self.document_retrieval.get_retrieval_statistics([0])
        
        assert stats['total_requested'] == 1
        assert stats['total_retrieved'] == 1
        assert stats['retrieval_success_rate'] == 1.0
        assert stats['unique_documents'] == 1
        assert stats['average_chunk_size'] == 35


class TestResultRanking:
    """Test result ranking functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = Mock()
        self.config.similarity_weights = {
            'embedding': 0.4,
            'hierarchical': 0.4,
            'spatial': 0.2
        }
        self.config.metadata_boost_factors = {
            'recent_documents': 1.1,
            'high_quality_embeddings': 1.05,
            'complete_document_chunks': 1.02
        }
        self.config.enable_metadata_integration = True
        
        # Create mock document retrieval
        self.document_retrieval = Mock(spec=DocumentRetrievalImpl)
        self.result_ranking = ResultRankingSystem(self.document_retrieval, self.config)
        
        # Create test data
        self.test_chunk = DocumentChunk(
            content="This is test content for ranking",
            ipfs_hash="QmTest1234567890abcdef1234567890abcdef123456",
            source_path="/test/document1.txt",
            start_position=0,
            end_position=33,
            chunk_sequence=0,
            creation_timestamp="2024-01-01T00:00:00Z",
            chunk_size=33
        )
        
        self.test_metadata = VideoFrameMetadata(
            frame_index=0,
            chunk_id="test_chunk_1",
            ipfs_hash="QmTest1234567890abcdef1234567890abcdef123456",
            source_document="/test/document1.txt",
            compression_quality=0.8,
            hierarchical_indices=[],
            embedding_model="test_model",
            frame_timestamp=1640995200.0,
            chunk_metadata=self.test_chunk
        )
    
    def test_rank_search_results(self):
        """Test basic search result ranking."""
        # Mock document retrieval
        self.document_retrieval.retrieve_documents_with_metadata.return_value = [
            (0, self.test_chunk, self.test_metadata)
        ]
        
        # Test ranking
        similarity_results = [(0, 0.8)]
        embedding_similarities = [(0, 0.75)]
        hierarchical_similarities = [(0, 0.7)]
        cached_neighbors = {0: [1, 2]}
        
        results = self.result_ranking.rank_search_results(
            similarity_results,
            embedding_similarities,
            hierarchical_similarities,
            cached_neighbors
        )
        
        assert len(results) == 1
        assert isinstance(results[0], DocumentSearchResult)
        assert results[0].similarity_score >= 0.8  # May be boosted
        assert results[0].embedding_similarity_score == 0.75
        assert results[0].hierarchical_similarity_score == 0.7
        assert results[0].cached_neighbors == [1, 2]
    
    def test_apply_metadata_boosts(self):
        """Test metadata-based ranking boosts."""
        # Test with high quality embedding
        boosted_score = self.result_ranking._apply_metadata_boosts(
            0.8, self.test_chunk, self.test_metadata
        )
        
        # Should be boosted due to high quality embedding (0.8 >= 0.8)
        assert boosted_score > 0.8
        assert boosted_score <= 1.0
    
    def test_integrate_ipfs_metadata(self):
        """Test IPFS metadata integration."""
        # Create test search result
        search_result = DocumentSearchResult(
            document_chunk=self.test_chunk,
            similarity_score=0.8,
            embedding_similarity_score=0.75,
            hierarchical_similarity_score=0.7,
            frame_number=0,
            search_method="test_method",
            cached_neighbors=[1, 2]
        )
        
        # Mock document retrieval for IPFS integration
        self.document_retrieval.get_document_by_ipfs_hash.return_value = [
            (0, self.test_chunk)
        ]
        
        # Test integration
        enhanced_results = self.result_ranking.integrate_ipfs_metadata([search_result])
        
        assert len(enhanced_results) == 1
        assert "ipfs_integration" in enhanced_results[0].search_method
    
    def test_create_result_with_cached_neighbors(self):
        """Test creating result with cached neighbors."""
        # Mock document retrieval
        self.document_retrieval.retrieve_single_document.return_value = self.test_chunk
        
        # Test result creation
        result = self.result_ranking.create_result_with_cached_neighbors(
            frame_number=0,
            similarity_score=0.8,
            embedding_similarity=0.75,
            hierarchical_similarity=0.7,
            cached_neighbors=[1, 2, 3]
        )
        
        assert result is not None
        assert result.frame_number == 0
        assert result.similarity_score == 0.8
        assert result.embedding_similarity_score == 0.75
        assert result.hierarchical_similarity_score == 0.7
        assert result.cached_neighbors == [1, 2, 3]
    
    def test_get_ranking_statistics(self):
        """Test ranking statistics calculation."""
        # Create test search results
        search_results = [
            DocumentSearchResult(
                document_chunk=self.test_chunk,
                similarity_score=0.8,
                embedding_similarity_score=0.75,
                hierarchical_similarity_score=0.7,
                frame_number=0,
                search_method="test_method",
                cached_neighbors=[1, 2]
            )
        ]
        
        # Test statistics
        stats = self.result_ranking.get_ranking_statistics(search_results)
        
        assert stats['total_results'] == 1
        assert stats['average_similarity'] == 0.8
        assert stats['unique_documents'] == 1
        assert stats['total_cached_neighbors'] == 2
    
    def test_filter_and_deduplicate_results(self):
        """Test result filtering and deduplication."""
        # Create test results with duplicates
        search_results = [
            DocumentSearchResult(
                document_chunk=self.test_chunk,
                similarity_score=0.8,
                embedding_similarity_score=0.75,
                hierarchical_similarity_score=0.7,
                frame_number=0,
                search_method="test_method",
                cached_neighbors=[]
            ),
            DocumentSearchResult(
                document_chunk=self.test_chunk,  # Same IPFS hash
                similarity_score=0.6,
                embedding_similarity_score=0.55,
                hierarchical_similarity_score=0.5,
                frame_number=1,
                search_method="test_method",
                cached_neighbors=[]
            )
        ]
        
        # Test filtering and deduplication
        filtered_results = self.result_ranking.filter_and_deduplicate_results(
            search_results,
            max_results=10,
            similarity_threshold=0.5,
            deduplicate_by_ipfs=True
        )
        
        # Should keep only the first (higher scoring) result due to deduplication
        assert len(filtered_results) == 1
        assert filtered_results[0].similarity_score == 0.8


class TestIntegratedSearchEngine:
    """Test integrated search engine with document retrieval and ranking."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = Mock()
        self.config.max_search_results = 10
        self.config.similarity_threshold = 0.1
        
        # Create mock dual storage
        self.dual_storage = Mock(spec=DualVideoStorageImpl)
        
        # Create search engine with dual storage
        self.search_engine = RAGSearchEngineImpl(self.config, self.dual_storage)
    
    def test_search_engine_initialization_with_dual_storage(self):
        """Test search engine initialization with dual storage."""
        assert self.search_engine.document_retrieval is not None
        assert self.search_engine.result_ranking is not None
    
    def test_search_engine_initialization_without_dual_storage(self):
        """Test search engine initialization without dual storage."""
        search_engine = RAGSearchEngineImpl(self.config)
        assert search_engine.document_retrieval is None
        assert search_engine.result_ranking is None
    
    @patch('hilbert_quantization.rag.search.engine.RAGSearchEngineImpl._generate_query_embedding')
    def test_search_with_comprehensive_ranking(self, mock_generate_embedding):
        """Test comprehensive search with ranking."""
        # Mock query embedding
        mock_generate_embedding.return_value = np.random.rand(64, 64)
        
        # Mock search engine methods
        self.search_engine.progressive_hierarchical_search = Mock(return_value=[0, 1])
        self.search_engine.cache_consecutive_frames = Mock(return_value={0: np.random.rand(64, 64)})
        self.search_engine.calculate_embedding_similarity = Mock(return_value=[(0, 0.8)])
        
        # Mock result ranking
        mock_result = DocumentSearchResult(
            document_chunk=Mock(),
            similarity_score=0.8,
            embedding_similarity_score=0.75,
            hierarchical_similarity_score=0.7,
            frame_number=0,
            search_method="comprehensive",
            cached_neighbors=[]
        )
        self.search_engine.result_ranking.rank_with_advanced_scoring = Mock(return_value=[mock_result])
        self.search_engine.result_ranking.integrate_ipfs_metadata = Mock(return_value=[mock_result])
        self.search_engine.result_ranking.filter_and_deduplicate_results = Mock(return_value=[mock_result])
        
        # Test comprehensive search
        results = self.search_engine.search_with_comprehensive_ranking("test query")
        
        assert len(results) == 1
        assert results[0].similarity_score == 0.8


if __name__ == "__main__":
    pytest.main([__file__])