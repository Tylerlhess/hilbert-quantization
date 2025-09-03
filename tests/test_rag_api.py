"""
Tests for the high-level RAG API interface.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from hilbert_quantization.rag.api import (
    RAGSystem, RAGSystemError, DocumentProcessingError, 
    EmbeddingGenerationError, SearchError, StorageError,
    create_rag_system, process_document_collection, search_documents
)
from hilbert_quantization.rag.config import RAGConfig, create_default_rag_config
from hilbert_quantization.rag.models import (
    DocumentChunk, DocumentSearchResult, ProcessingProgress
)


class TestRAGSystem:
    """Test RAGSystem main functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = create_default_rag_config()
        self.config.storage.base_storage_path = self.temp_dir
    
    def test_initialization_default_config(self):
        """Test RAG system initialization with default configuration."""
        with patch.multiple(
            'hilbert_quantization.rag.api',
            DocumentChunkerImpl=Mock(),
            DocumentMetadataManager=Mock(),
            BatchDocumentProcessor=Mock(),
            EmbeddingGeneratorImpl=Mock(),
            HierarchicalIndexGenerator=Mock(),
            EmbeddingCompressorImpl=Mock(),
            EmbeddingReconstructorImpl=Mock(),
            DualVideoStorageImpl=Mock(),
            RAGSearchEngineImpl=Mock(),
            DocumentRetrievalImpl=Mock(),
            ResultRankingSystem=Mock(),
            RAGValidator=Mock()
        ):
            rag_system = RAGSystem()
            assert rag_system.config is not None
            assert rag_system.config_manager is not None
            assert hasattr(rag_system, 'chunker')
            assert hasattr(rag_system, 'embedding_generator')
            assert hasattr(rag_system, 'storage')
            assert hasattr(rag_system, 'search_engine')
    
    def test_initialization_custom_config(self):
        """Test RAG system initialization with custom configuration."""
        custom_config = create_default_rag_config()
        custom_config.embedding.batch_size = 64
        custom_config.video.quality = 0.9
        
        with patch.multiple(
            'hilbert_quantization.rag.api',
            DocumentChunkerImpl=Mock(),
            DocumentMetadataManager=Mock(),
            BatchDocumentProcessor=Mock(),
            EmbeddingGeneratorImpl=Mock(),
            HierarchicalIndexGenerator=Mock(),
            EmbeddingCompressorImpl=Mock(),
            EmbeddingReconstructorImpl=Mock(),
            DualVideoStorageImpl=Mock(),
            RAGSearchEngineImpl=Mock(),
            DocumentRetrievalImpl=Mock(),
            ResultRankingSystem=Mock(),
            RAGValidator=Mock()
        ):
            rag_system = RAGSystem(custom_config)
            assert rag_system.config.embedding.batch_size == 64
            assert rag_system.config.video.quality == 0.9
    
    def test_initialization_with_storage_path(self):
        """Test RAG system initialization with custom storage path."""
        with patch.multiple(
            'hilbert_quantization.rag.api',
            DocumentChunkerImpl=Mock(),
            DocumentMetadataManager=Mock(),
            BatchDocumentProcessor=Mock(),
            EmbeddingGeneratorImpl=Mock(),
            HierarchicalIndexGenerator=Mock(),
            EmbeddingCompressorImpl=Mock(),
            EmbeddingReconstructorImpl=Mock(),
            DualVideoStorageImpl=Mock(),
            RAGSearchEngineImpl=Mock(),
            DocumentRetrievalImpl=Mock(),
            ResultRankingSystem=Mock(),
            RAGValidator=Mock()
        ):
            custom_path = "/custom/storage/path"
            rag_system = RAGSystem(storage_path=custom_path)
            assert rag_system.config.storage.base_storage_path == custom_path
    
    @patch('hilbert_quantization.rag.api.RAGSystem._initialize_components')
    def test_initialization_component_failure(self, mock_init):
        """Test handling of component initialization failure."""
        mock_init.side_effect = Exception("Component initialization failed")
        
        with pytest.raises(RAGSystemError, match="Component initialization failed"):
            RAGSystem()
    
    def test_process_documents_string_list(self):
        """Test processing documents from string list."""
        with patch.multiple(
            'hilbert_quantization.rag.api',
            DocumentChunkerImpl=Mock(),
            DocumentMetadataManager=Mock(),
            BatchDocumentProcessor=Mock(),
            EmbeddingGeneratorImpl=Mock(),
            HierarchicalIndexGenerator=Mock(),
            EmbeddingCompressorImpl=Mock(),
            EmbeddingReconstructorImpl=Mock(),
            DualVideoStorageImpl=Mock(),
            RAGSearchEngineImpl=Mock(),
            DocumentRetrievalImpl=Mock(),
            ResultRankingSystem=Mock(),
            RAGValidator=Mock()
        ):
            rag_system = RAGSystem(self.config)
            
            # Mock components
            mock_chunk = DocumentChunk(
                content="test content",
                ipfs_hash="test_hash",
                source_path="test_doc",
                start_position=0,
                end_position=12,
                chunk_sequence=0,
                creation_timestamp="2024-01-01",
                chunk_size=12
            )
            rag_system.chunker.chunk_document.return_value = [mock_chunk]
            rag_system.embedding_generator.generate_embeddings.return_value = [np.array([1, 2, 3, 4])]
            rag_system.embedding_generator.calculate_optimal_dimensions.return_value = (2, 2)
            
            with patch('hilbert_quantization.rag.api.RAGSystem._map_embedding_to_2d') as mock_map:
                mock_map.return_value = np.array([[1, 2], [3, 4]])
                rag_system.index_generator.generate_multi_level_indices.return_value = np.array([[1, 2], [3, 4]])
                rag_system.storage.add_document_chunk.return_value = Mock()
                rag_system.storage.get_video_metadata.return_value = {'compression_ratio': 0.5}
                
                documents = ["Test document content", "Another test document"]
                results = rag_system.process_documents(documents)
                
                assert results['total_documents'] == 2
                assert results['processed_documents'] == 2
                assert results['total_chunks'] == 2
                assert 'processing_time' in results
    
    def test_process_documents_file_paths(self):
        """Test processing documents from file paths."""
        with patch.multiple(
            'hilbert_quantization.rag.api',
            DocumentChunkerImpl=Mock(),
            DocumentMetadataManager=Mock(),
            BatchDocumentProcessor=Mock(),
            EmbeddingGeneratorImpl=Mock(),
            HierarchicalIndexGenerator=Mock(),
            EmbeddingCompressorImpl=Mock(),
            EmbeddingReconstructorImpl=Mock(),
            DualVideoStorageImpl=Mock(),
            RAGSearchEngineImpl=Mock(),
            DocumentRetrievalImpl=Mock(),
            ResultRankingSystem=Mock(),
            RAGValidator=Mock()
        ):
            rag_system = RAGSystem(self.config)
            
            # Create temporary test files
            test_file1 = Path(self.temp_dir) / "test1.txt"
            test_file2 = Path(self.temp_dir) / "test2.txt"
            test_file1.write_text("Test document 1 content")
            test_file2.write_text("Test document 2 content")
            
            # Mock components
            mock_chunk = DocumentChunk(
                content="test content",
                ipfs_hash="test_hash",
                source_path="test_doc",
                start_position=0,
                end_position=12,
                chunk_sequence=0,
                creation_timestamp="2024-01-01",
                chunk_size=12
            )
            rag_system.chunker.chunk_document.return_value = [mock_chunk]
            rag_system.embedding_generator.generate_embeddings.return_value = [np.array([1, 2, 3, 4])]
            
            with patch('hilbert_quantization.rag.api.RAGSystem._store_embeddings_and_documents') as mock_store:
                mock_store.return_value = {'frames_stored': 2, 'compression_ratio': 0.5}
                
                results = rag_system.process_documents([test_file1, test_file2])
                
                assert results['total_documents'] == 2
                assert results['processed_documents'] == 2
    
    def test_process_documents_with_progress_callback(self):
        """Test document processing with progress callback."""
        with patch.multiple(
            'hilbert_quantization.rag.api',
            DocumentChunkerImpl=Mock(),
            DocumentMetadataManager=Mock(),
            BatchDocumentProcessor=Mock(),
            EmbeddingGeneratorImpl=Mock(),
            HierarchicalIndexGenerator=Mock(),
            EmbeddingCompressorImpl=Mock(),
            EmbeddingReconstructorImpl=Mock(),
            DualVideoStorageImpl=Mock(),
            RAGSearchEngineImpl=Mock(),
            DocumentRetrievalImpl=Mock(),
            ResultRankingSystem=Mock(),
            RAGValidator=Mock()
        ):
            rag_system = RAGSystem(self.config)
            
            # Mock components
            mock_chunk = DocumentChunk(
                content="test content",
                ipfs_hash="test_hash",
                source_path="test_doc",
                start_position=0,
                end_position=12,
                chunk_sequence=0,
                creation_timestamp="2024-01-01",
                chunk_size=12
            )
            rag_system.chunker.chunk_document.return_value = [mock_chunk]
            rag_system.embedding_generator.generate_embeddings.return_value = [np.array([1, 2, 3, 4])]
            
            with patch('hilbert_quantization.rag.api.RAGSystem._store_embeddings_and_documents') as mock_store:
                mock_store.return_value = {'frames_stored': 1, 'compression_ratio': 0.5}
                
                progress_calls = []
                def progress_callback(progress):
                    progress_calls.append(progress)
                
                documents = ["Test document"]
                rag_system.process_documents(documents, progress_callback=progress_callback)
                
                assert len(progress_calls) == 1
                assert isinstance(progress_calls[0], ProcessingProgress)
                assert progress_calls[0].total_documents == 1
    
    def test_process_documents_error_handling(self):
        """Test error handling during document processing."""
        with patch.multiple(
            'hilbert_quantization.rag.api',
            DocumentChunkerImpl=Mock(),
            DocumentMetadataManager=Mock(),
            BatchDocumentProcessor=Mock(),
            EmbeddingGeneratorImpl=Mock(),
            HierarchicalIndexGenerator=Mock(),
            EmbeddingCompressorImpl=Mock(),
            EmbeddingReconstructorImpl=Mock(),
            DualVideoStorageImpl=Mock(),
            RAGSearchEngineImpl=Mock(),
            DocumentRetrievalImpl=Mock(),
            ResultRankingSystem=Mock(),
            RAGValidator=Mock()
        ):
            rag_system = RAGSystem(self.config)
            rag_system.chunker.chunk_document.side_effect = Exception("Chunking failed")
            
            documents = ["Test document"]
            results = rag_system.process_documents(documents)
            
            assert results['processed_documents'] == 0
            assert len(results['failed_documents']) == 1
    
    def test_search_similar_documents(self):
        """Test similarity search functionality."""
        with patch.multiple(
            'hilbert_quantization.rag.api',
            DocumentChunkerImpl=Mock(),
            DocumentMetadataManager=Mock(),
            BatchDocumentProcessor=Mock(),
            EmbeddingGeneratorImpl=Mock(),
            HierarchicalIndexGenerator=Mock(),
            EmbeddingCompressorImpl=Mock(),
            EmbeddingReconstructorImpl=Mock(),
            DualVideoStorageImpl=Mock(),
            RAGSearchEngineImpl=Mock(),
            DocumentRetrievalImpl=Mock(),
            ResultRankingSystem=Mock(),
            RAGValidator=Mock()
        ):
            rag_system = RAGSystem(self.config)
            
            # Mock search results
            mock_result = DocumentSearchResult(
                document_chunk=DocumentChunk(
                    content="test content",
                    ipfs_hash="test_hash",
                    source_path="test_doc",
                    start_position=0,
                    end_position=12,
                    chunk_sequence=0,
                    creation_timestamp="2024-01-01",
                    chunk_size=12
                ),
                similarity_score=0.8,
                embedding_similarity_score=0.8,
                hierarchical_similarity_score=0.8,
                frame_number=1,
                search_method="progressive",
                cached_neighbors=None
            )
            
            rag_system.search_engine.search_similar_documents.return_value = [mock_result]
            rag_system.result_ranking.rank_search_results.return_value = [mock_result]
            
            query = "test query"
            results = rag_system.search_similar_documents(query, max_results=5)
            
            assert len(results) == 1
            assert results[0].similarity_score == 0.8
            assert rag_system._stats['searches_performed'] == 1
    
    def test_search_with_similarity_threshold_filtering(self):
        """Test search with similarity threshold filtering."""
        with patch.multiple(
            'hilbert_quantization.rag.api',
            DocumentChunkerImpl=Mock(),
            DocumentMetadataManager=Mock(),
            BatchDocumentProcessor=Mock(),
            EmbeddingGeneratorImpl=Mock(),
            HierarchicalIndexGenerator=Mock(),
            EmbeddingCompressorImpl=Mock(),
            EmbeddingReconstructorImpl=Mock(),
            DualVideoStorageImpl=Mock(),
            RAGSearchEngineImpl=Mock(),
            DocumentRetrievalImpl=Mock(),
            ResultRankingSystem=Mock(),
            RAGValidator=Mock()
        ):
            rag_system = RAGSystem(self.config)
            
            # Mock search results with different similarity scores
            high_score_result = DocumentSearchResult(
                document_chunk=Mock(),
                similarity_score=0.9,
                embedding_similarity_score=0.9,
                hierarchical_similarity_score=0.9,
                frame_number=1,
                search_method="progressive",
                cached_neighbors=None
            )
            
            low_score_result = DocumentSearchResult(
                document_chunk=Mock(),
                similarity_score=0.3,
                embedding_similarity_score=0.3,
                hierarchical_similarity_score=0.3,
                frame_number=2,
                search_method="progressive",
                cached_neighbors=None
            )
            
            rag_system.search_engine.search_similar_documents.return_value = [high_score_result, low_score_result]
            rag_system.result_ranking.rank_search_results.return_value = [high_score_result]
            
            query = "test query"
            results = rag_system.search_similar_documents(query, similarity_threshold=0.8)
            
            # Only high score result should be returned
            assert len(results) == 1
            assert results[0].similarity_score == 0.9
    
    def test_search_error_handling(self):
        """Test error handling during search operations."""
        with patch.multiple(
            'hilbert_quantization.rag.api',
            DocumentChunkerImpl=Mock(),
            DocumentMetadataManager=Mock(),
            BatchDocumentProcessor=Mock(),
            EmbeddingGeneratorImpl=Mock(),
            HierarchicalIndexGenerator=Mock(),
            EmbeddingCompressorImpl=Mock(),
            EmbeddingReconstructorImpl=Mock(),
            DualVideoStorageImpl=Mock(),
            RAGSearchEngineImpl=Mock(),
            DocumentRetrievalImpl=Mock(),
            ResultRankingSystem=Mock(),
            RAGValidator=Mock()
        ):
            rag_system = RAGSystem(self.config)
            rag_system.search_engine.search_similar_documents.side_effect = Exception("Search failed")
            
            with pytest.raises(SearchError, match="Failed to search similar documents"):
                rag_system.search_similar_documents("test query")
    
    def test_add_documents(self):
        """Test adding new documents to existing system."""
        with patch.multiple(
            'hilbert_quantization.rag.api',
            DocumentChunkerImpl=Mock(),
            DocumentMetadataManager=Mock(),
            BatchDocumentProcessor=Mock(),
            EmbeddingGeneratorImpl=Mock(),
            HierarchicalIndexGenerator=Mock(),
            EmbeddingCompressorImpl=Mock(),
            EmbeddingReconstructorImpl=Mock(),
            DualVideoStorageImpl=Mock(),
            RAGSearchEngineImpl=Mock(),
            DocumentRetrievalImpl=Mock(),
            ResultRankingSystem=Mock(),
            RAGValidator=Mock()
        ):
            rag_system = RAGSystem(self.config)
            
            with patch.object(rag_system, 'process_documents') as mock_process:
                mock_process.return_value = {'processed_documents': 2}
                rag_system.storage.optimize_frame_ordering = Mock()
                
                documents = ["New document 1", "New document 2"]
                results = rag_system.add_documents(documents, optimize_insertion=True)
                
                mock_process.assert_called_once_with(documents)
                rag_system.storage.optimize_frame_ordering.assert_called_once()
                assert results['processed_documents'] == 2
    
    def test_validate_system_integrity(self):
        """Test system integrity validation."""
        with patch.multiple(
            'hilbert_quantization.rag.api',
            DocumentChunkerImpl=Mock(),
            DocumentMetadataManager=Mock(),
            BatchDocumentProcessor=Mock(),
            EmbeddingGeneratorImpl=Mock(),
            HierarchicalIndexGenerator=Mock(),
            EmbeddingCompressorImpl=Mock(),
            EmbeddingReconstructorImpl=Mock(),
            DualVideoStorageImpl=Mock(),
            RAGSearchEngineImpl=Mock(),
            DocumentRetrievalImpl=Mock(),
            ResultRankingSystem=Mock(),
            RAGValidator=Mock()
        ):
            rag_system = RAGSystem(self.config)
            
            mock_validation_results = {
                'overall_status': 'passed',
                'compression_accuracy': 0.95,
                'retrieval_accuracy': 0.98,
                'synchronization_check': True
            }
            rag_system.validator.validate_system_integrity.return_value = mock_validation_results
            
            results = rag_system.validate_system_integrity()
            
            assert results['overall_status'] == 'passed'
            assert results['compression_accuracy'] == 0.95
    
    def test_get_system_statistics(self):
        """Test getting system statistics."""
        with patch.multiple(
            'hilbert_quantization.rag.api',
            DocumentChunkerImpl=Mock(),
            DocumentMetadataManager=Mock(),
            BatchDocumentProcessor=Mock(),
            EmbeddingGeneratorImpl=Mock(),
            HierarchicalIndexGenerator=Mock(),
            EmbeddingCompressorImpl=Mock(),
            EmbeddingReconstructorImpl=Mock(),
            DualVideoStorageImpl=Mock(),
            RAGSearchEngineImpl=Mock(),
            DocumentRetrievalImpl=Mock(),
            ResultRankingSystem=Mock(),
            RAGValidator=Mock()
        ):
            rag_system = RAGSystem(self.config)
            
            # Set some statistics
            rag_system._stats['documents_processed'] = 10
            rag_system._stats['embeddings_generated'] = 50
            
            # Mock storage metadata
            rag_system.storage.get_video_metadata.return_value = {
                'total_size_mb': 100.5,
                'total_frames': 50,
                'video_files': 2
            }
            
            stats = rag_system.get_system_statistics()
            
            assert stats['documents_processed'] == 10
            assert stats['embeddings_generated'] == 50
            assert stats['storage_size_mb'] == 100.5
            assert stats['total_frames'] == 50
            assert 'configuration' in stats
            assert stats['configuration']['embedding_model'] == self.config.embedding.model_name
    
    def test_optimize_configuration(self):
        """Test configuration optimization."""
        with patch.multiple(
            'hilbert_quantization.rag.api',
            DocumentChunkerImpl=Mock(),
            DocumentMetadataManager=Mock(),
            BatchDocumentProcessor=Mock(),
            EmbeddingGeneratorImpl=Mock(),
            HierarchicalIndexGenerator=Mock(),
            EmbeddingCompressorImpl=Mock(),
            EmbeddingReconstructorImpl=Mock(),
            DualVideoStorageImpl=Mock(),
            RAGSearchEngineImpl=Mock(),
            DocumentRetrievalImpl=Mock(),
            ResultRankingSystem=Mock(),
            RAGValidator=Mock()
        ):
            rag_system = RAGSystem(self.config)
            
            with patch.object(rag_system, '_initialize_components'):
                results = rag_system.optimize_configuration('performance', dataset_size=50000)
                
                assert results['target_metric'] == 'performance'
                assert results['dataset_size'] == 50000
                assert results['changes_applied'] is True
                assert 'warnings' in results
    
    def test_configuration_file_operations(self):
        """Test configuration export and import."""
        with patch.multiple(
            'hilbert_quantization.rag.api',
            DocumentChunkerImpl=Mock(),
            DocumentMetadataManager=Mock(),
            BatchDocumentProcessor=Mock(),
            EmbeddingGeneratorImpl=Mock(),
            HierarchicalIndexGenerator=Mock(),
            EmbeddingCompressorImpl=Mock(),
            EmbeddingReconstructorImpl=Mock(),
            DualVideoStorageImpl=Mock(),
            RAGSearchEngineImpl=Mock(),
            DocumentRetrievalImpl=Mock(),
            ResultRankingSystem=Mock(),
            RAGValidator=Mock()
        ):
            rag_system = RAGSystem(self.config)
            
            config_file = Path(self.temp_dir) / "test_config.json"
            
            # Test export
            with patch.object(rag_system.config_manager, 'save_config') as mock_save:
                rag_system.export_configuration(config_file)
                mock_save.assert_called_once_with(config_file)
            
            # Test import
            with patch.object(rag_system.config_manager, 'load_config') as mock_load:
                with patch.object(rag_system, '_initialize_components'):
                    rag_system.import_configuration(config_file)
                    mock_load.assert_called_once_with(config_file)
    
    def test_context_manager(self):
        """Test RAG system as context manager."""
        with patch.multiple(
            'hilbert_quantization.rag.api',
            DocumentChunkerImpl=Mock(),
            DocumentMetadataManager=Mock(),
            BatchDocumentProcessor=Mock(),
            EmbeddingGeneratorImpl=Mock(),
            HierarchicalIndexGenerator=Mock(),
            EmbeddingCompressorImpl=Mock(),
            EmbeddingReconstructorImpl=Mock(),
            DualVideoStorageImpl=Mock(),
            RAGSearchEngineImpl=Mock(),
            DocumentRetrievalImpl=Mock(),
            ResultRankingSystem=Mock(),
            RAGValidator=Mock()
        ):
            with patch.object(RAGSystem, 'close') as mock_close:
                with RAGSystem(self.config) as rag_system:
                    assert isinstance(rag_system, RAGSystem)
                
                mock_close.assert_called_once()


class TestConvenienceFunctions:
    """Test convenience functions for common use cases."""
    
    def test_create_rag_system_default(self):
        """Test creating RAG system with default settings."""
        with patch('hilbert_quantization.rag.api.RAGSystem') as mock_rag:
            create_rag_system()
            
            # Verify RAGSystem was called with appropriate config
            mock_rag.assert_called_once()
            call_args = mock_rag.call_args[0]
            config = call_args[0]
            assert config.embedding.model_name == "sentence-transformers/all-MiniLM-L6-v2"
            assert config.storage.base_storage_path == "./rag_storage"
    
    def test_create_rag_system_custom_parameters(self):
        """Test creating RAG system with custom parameters."""
        with patch('hilbert_quantization.rag.api.RAGSystem') as mock_rag:
            create_rag_system(
                storage_path="/custom/path",
                embedding_model="custom/model",
                quality="high"
            )
            
            mock_rag.assert_called_once()
            call_args = mock_rag.call_args[0]
            config = call_args[0]
            assert config.embedding.model_name == "custom/model"
            assert config.storage.base_storage_path == "/custom/path"
    
    def test_create_rag_system_quality_presets(self):
        """Test different quality presets."""
        with patch('hilbert_quantization.rag.api.RAGSystem') as mock_rag:
            # Test high quality
            create_rag_system(quality="high")
            call_args = mock_rag.call_args[0]
            config = call_args[0]
            assert config.video.quality == 0.95  # High quality setting
            
            mock_rag.reset_mock()
            
            # Test performance quality
            create_rag_system(quality="performance")
            call_args = mock_rag.call_args[0]
            config = call_args[0]
            assert config.video.adaptive_quality is True  # Performance setting
    
    def test_process_document_collection(self):
        """Test processing document collection convenience function."""
        with patch('hilbert_quantization.rag.api.create_rag_system') as mock_create:
            mock_rag_system = Mock()
            mock_create.return_value = mock_rag_system
            
            documents = ["doc1", "doc2"]
            result = process_document_collection(
                documents,
                storage_path="/test/path",
                embedding_model="test/model"
            )
            
            mock_create.assert_called_once_with("/test/path", "test/model")
            mock_rag_system.process_documents.assert_called_once_with(documents)
            assert result == mock_rag_system
    
    def test_search_documents(self):
        """Test search documents convenience function."""
        with patch('hilbert_quantization.rag.api.RAGSystem') as mock_rag_class:
            mock_rag_system = Mock()
            mock_rag_class.return_value = mock_rag_system
            mock_results = [Mock()]
            mock_rag_system.search_similar_documents.return_value = mock_results
            
            query = "test query"
            results = search_documents(
                query,
                storage_path="/test/path",
                max_results=5
            )
            
            mock_rag_class.assert_called_once_with(storage_path="/test/path")
            mock_rag_system.search_similar_documents.assert_called_once_with(query, 5)
            assert results == mock_results


class TestErrorHandling:
    """Test error handling and exception scenarios."""
    
    def test_document_processing_error(self):
        """Test DocumentProcessingError handling."""
        with pytest.raises(DocumentProcessingError):
            raise DocumentProcessingError("Test error")
    
    def test_embedding_generation_error(self):
        """Test EmbeddingGenerationError handling."""
        with pytest.raises(EmbeddingGenerationError):
            raise EmbeddingGenerationError("Test error")
    
    def test_search_error(self):
        """Test SearchError handling."""
        with pytest.raises(SearchError):
            raise SearchError("Test error")
    
    def test_storage_error(self):
        """Test StorageError handling."""
        with pytest.raises(StorageError):
            raise StorageError("Test error")
    
    def test_rag_system_error_inheritance(self):
        """Test that all specific errors inherit from RAGSystemError."""
        assert issubclass(DocumentProcessingError, RAGSystemError)
        assert issubclass(EmbeddingGenerationError, RAGSystemError)
        assert issubclass(SearchError, RAGSystemError)
        assert issubclass(StorageError, RAGSystemError)


class TestIntegrationScenarios:
    """Test integration scenarios and workflows."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
    
    def test_complete_workflow_simulation(self):
        """Test complete RAG workflow simulation."""
        with patch.multiple(
            'hilbert_quantization.rag.api',
            DocumentChunkerImpl=Mock(),
            DocumentMetadataManager=Mock(),
            BatchDocumentProcessor=Mock(),
            EmbeddingGeneratorImpl=Mock(),
            HierarchicalIndexGenerator=Mock(),
            EmbeddingCompressorImpl=Mock(),
            EmbeddingReconstructorImpl=Mock(),
            DualVideoStorageImpl=Mock(),
            RAGSearchEngineImpl=Mock(),
            DocumentRetrievalImpl=Mock(),
            ResultRankingSystem=Mock(),
            RAGValidator=Mock()
        ):
            # Create RAG system
            config = create_default_rag_config()
            config.storage.base_storage_path = self.temp_dir
            rag_system = RAGSystem(config)
            
            # Mock all necessary components for workflow
            mock_chunk = DocumentChunk(
                content="test content",
                ipfs_hash="test_hash",
                source_path="test_doc",
                start_position=0,
                end_position=12,
                chunk_sequence=0,
                creation_timestamp="2024-01-01",
                chunk_size=12
            )
            
            mock_result = DocumentSearchResult(
                document_chunk=mock_chunk,
                similarity_score=0.8,
                embedding_similarity_score=0.8,
                hierarchical_similarity_score=0.8,
                frame_number=1,
                search_method="progressive",
                cached_neighbors=None
            )
            
            # Setup mocks
            rag_system.chunker.chunk_document.return_value = [mock_chunk]
            rag_system.embedding_generator.generate_embeddings.return_value = [np.array([1, 2, 3, 4])]
            rag_system.search_engine.search_similar_documents.return_value = [mock_result]
            rag_system.result_ranking.rank_search_results.return_value = [mock_result]
            
            with patch('hilbert_quantization.rag.api.RAGSystem._store_embeddings_and_documents') as mock_store:
                mock_store.return_value = {'frames_stored': 1, 'compression_ratio': 0.5}
                
                # 1. Process documents
                documents = ["Test document content"]
                process_results = rag_system.process_documents(documents)
                assert process_results['processed_documents'] == 1
                
                # 2. Search for similar documents
                search_results = rag_system.search_similar_documents("test query")
                assert len(search_results) == 1
                assert search_results[0].similarity_score == 0.8
                
                # 3. Get system statistics
                stats = rag_system.get_system_statistics()
                assert stats['documents_processed'] == 1
                assert stats['searches_performed'] == 1
    
    def test_error_recovery_workflow(self):
        """Test error recovery in workflow scenarios."""
        with patch.multiple(
            'hilbert_quantization.rag.api',
            DocumentChunkerImpl=Mock(),
            DocumentMetadataManager=Mock(),
            BatchDocumentProcessor=Mock(),
            EmbeddingGeneratorImpl=Mock(),
            HierarchicalIndexGenerator=Mock(),
            EmbeddingCompressorImpl=Mock(),
            EmbeddingReconstructorImpl=Mock(),
            DualVideoStorageImpl=Mock(),
            RAGSearchEngineImpl=Mock(),
            DocumentRetrievalImpl=Mock(),
            ResultRankingSystem=Mock(),
            RAGValidator=Mock()
        ):
            config = create_default_rag_config()
            config.storage.base_storage_path = self.temp_dir
            rag_system = RAGSystem(config)
            
            # Simulate partial failure in document processing
            rag_system.chunker.chunk_document.side_effect = [
                Exception("First document failed"),
                [Mock()]  # Second document succeeds
            ]
            rag_system.embedding_generator.generate_embeddings.return_value = [np.array([1, 2, 3, 4])]
            
            with patch('hilbert_quantization.rag.api.RAGSystem._store_embeddings_and_documents') as mock_store:
                mock_store.return_value = {'frames_stored': 1, 'compression_ratio': 0.5}
                
                documents = ["Failed document", "Successful document"]
                results = rag_system.process_documents(documents)
                
                # Should process successfully despite one failure
                assert results['total_documents'] == 2
                assert results['processed_documents'] == 1
                assert len(results['failed_documents']) == 1