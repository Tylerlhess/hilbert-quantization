"""
Tests for batch document processing functionality.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import time

from hilbert_quantization.rag.document_processing.batch_processor import (
    BatchDocumentProcessor,
    BatchConfig,
    BatchProcessingStats,
    MemoryMonitor
)
from hilbert_quantization.rag.document_processing.document_validator import (
    DocumentFilterConfig,
    DocumentType
)
from hilbert_quantization.rag.models import DocumentChunk, ProcessingProgress, RAGMetrics


class TestBatchConfig:
    """Test batch configuration validation."""
    
    def test_valid_config(self):
        """Test valid batch configuration."""
        config = BatchConfig(
            initial_batch_size=10,
            max_batch_size=50,
            min_batch_size=1,
            memory_threshold_mb=512.0,
            max_workers=2
        )
        
        assert config.initial_batch_size == 10
        assert config.max_batch_size == 50
        assert config.min_batch_size == 1
        assert config.memory_threshold_mb == 512.0
        assert config.max_workers == 2
    
    def test_invalid_batch_sizes(self):
        """Test invalid batch size configurations."""
        with pytest.raises(ValueError, match="Initial batch size must be positive"):
            BatchConfig(initial_batch_size=0)
        
        with pytest.raises(ValueError, match="Max batch size must be >= min batch size"):
            BatchConfig(max_batch_size=5, min_batch_size=10)
    
    def test_invalid_memory_threshold(self):
        """Test invalid memory threshold."""
        with pytest.raises(ValueError, match="Memory threshold must be positive"):
            BatchConfig(memory_threshold_mb=0)
    
    def test_invalid_workers(self):
        """Test invalid worker count."""
        with pytest.raises(ValueError, match="Max workers must be positive"):
            BatchConfig(max_workers=0)
    
    def test_invalid_memory_usage_percent(self):
        """Test invalid memory usage percentage."""
        with pytest.raises(ValueError, match="Target memory usage percent must be between 0 and 100"):
            BatchConfig(target_memory_usage_percent=0)
        
        with pytest.raises(ValueError, match="Target memory usage percent must be between 0 and 100"):
            BatchConfig(target_memory_usage_percent=150)


class TestBatchProcessingStats:
    """Test batch processing statistics."""
    
    def test_stats_initialization(self):
        """Test statistics initialization."""
        stats = BatchProcessingStats()
        
        assert stats.total_batches == 0
        assert stats.successful_batches == 0
        assert stats.failed_batches == 0
        assert stats.total_documents == 0
        assert stats.processed_documents == 0
        assert stats.total_chunks == 0
        assert stats.total_embeddings == 0
        assert stats.current_batch_size == 10
        assert stats.peak_memory_usage_mb == 0.0
        assert stats.average_batch_time == 0.0
        assert len(stats.batch_times) == 0
    
    def test_progress_percent_calculation(self):
        """Test progress percentage calculation."""
        stats = BatchProcessingStats()
        
        # No documents
        assert stats.progress_percent == 0.0
        
        # Some progress
        stats.total_documents = 100
        stats.processed_documents = 25
        assert stats.progress_percent == 25.0
        
        # Complete
        stats.processed_documents = 100
        assert stats.progress_percent == 100.0
    
    def test_documents_per_second_calculation(self):
        """Test processing rate calculation."""
        stats = BatchProcessingStats()
        stats.processing_start_time = time.time() - 10  # 10 seconds ago
        stats.processed_documents = 50
        
        rate = stats.documents_per_second
        assert 4.5 <= rate <= 5.5  # Should be around 5 docs/sec
    
    def test_success_rate_calculation(self):
        """Test batch success rate calculation."""
        stats = BatchProcessingStats()
        
        # No batches
        assert stats.success_rate == 0.0
        
        # Some batches
        stats.total_batches = 10
        stats.successful_batches = 8
        assert stats.success_rate == 0.8


class TestMemoryMonitor:
    """Test memory monitoring functionality."""
    
    def test_memory_monitor_initialization(self):
        """Test memory monitor initialization."""
        monitor = MemoryMonitor(target_usage_percent=75.0)
        assert monitor.target_usage_percent == 75.0
    
    @patch('psutil.Process')
    def test_get_memory_usage_mb(self, mock_process_class):
        """Test memory usage retrieval."""
        mock_process = Mock()
        mock_process.memory_info.return_value.rss = 1024 * 1024 * 100  # 100 MB
        mock_process_class.return_value = mock_process
        
        monitor = MemoryMonitor()
        usage = monitor.get_memory_usage_mb()
        assert usage == 100.0
    
    @patch('psutil.virtual_memory')
    def test_memory_pressure_detection(self, mock_virtual_memory):
        """Test memory pressure detection."""
        mock_virtual_memory.return_value.percent = 85.0
        
        monitor = MemoryMonitor(target_usage_percent=80.0)
        
        assert monitor.should_reduce_batch_size() is True
        assert monitor.should_increase_batch_size() is False
    
    @patch('psutil.virtual_memory')
    def test_memory_availability_detection(self, mock_virtual_memory):
        """Test memory availability detection."""
        mock_virtual_memory.return_value.percent = 50.0
        
        monitor = MemoryMonitor(target_usage_percent=80.0)
        
        assert monitor.should_reduce_batch_size() is False
        assert monitor.should_increase_batch_size() is True
    
    @patch('psutil.virtual_memory')
    def test_batch_size_recommendation(self, mock_virtual_memory):
        """Test batch size recommendations."""
        monitor = MemoryMonitor(target_usage_percent=80.0)
        
        # High memory usage - should reduce
        mock_virtual_memory.return_value.percent = 90.0
        new_size = monitor.get_recommended_batch_size(10, 1, 50)
        assert new_size == 8  # 80% of 10
        
        # Low memory usage - should increase
        mock_virtual_memory.return_value.percent = 40.0
        new_size = monitor.get_recommended_batch_size(10, 1, 50)
        assert new_size == 12  # 120% of 10
        
        # Normal memory usage - should keep same
        mock_virtual_memory.return_value.percent = 75.0
        new_size = monitor.get_recommended_batch_size(10, 1, 50)
        assert new_size == 10


class TestBatchDocumentProcessor:
    """Test batch document processor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_chunker = Mock()
        self.mock_embedding_generator = Mock()
        self.mock_video_storage = Mock()
        
        self.processor = BatchDocumentProcessor(
            chunker=self.mock_chunker,
            embedding_generator=self.mock_embedding_generator,
            video_storage=self.mock_video_storage,
            config=BatchConfig(initial_batch_size=2, max_workers=1)
        )
    
    def create_test_documents(self, count: int = 3) -> list:
        """Create temporary test documents."""
        temp_files = []
        
        for i in range(count):
            temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
            temp_file.write(f"This is test document {i + 1} content.")
            temp_file.close()
            temp_files.append(temp_file.name)
        
        return temp_files
    
    def cleanup_test_documents(self, file_paths: list):
        """Clean up temporary test documents."""
        for file_path in file_paths:
            try:
                os.unlink(file_path)
            except FileNotFoundError:
                pass
    
    def test_processor_initialization(self):
        """Test processor initialization."""
        processor = BatchDocumentProcessor()
        
        assert processor.chunker is not None
        assert processor.embedding_generator is not None
        assert processor.video_storage is not None
        assert processor.config is not None
        assert processor.memory_monitor is not None
        assert processor.stats is not None
    
    def test_create_batches(self):
        """Test batch creation from document paths."""
        document_paths = ['doc1.txt', 'doc2.txt', 'doc3.txt', 'doc4.txt', 'doc5.txt']
        
        batches = list(self.processor._create_batches(document_paths))
        
        assert len(batches) == 3  # 2 + 2 + 1
        assert batches[0] == ['doc1.txt', 'doc2.txt']
        assert batches[1] == ['doc3.txt', 'doc4.txt']
        assert batches[2] == ['doc5.txt']
    
    def test_process_single_document(self):
        """Test processing a single document."""
        # Create test document
        test_files = self.create_test_documents(1)
        
        try:
            # Mock chunker and embedding generator
            mock_chunk = DocumentChunk(
                content="test content",
                ipfs_hash="test_hash",
                source_path=test_files[0],
                start_position=0,
                end_position=12,
                chunk_sequence=0,
                creation_timestamp="2023-01-01T00:00:00",
                chunk_size=12
            )
            
            self.mock_chunker.chunk_document.return_value = [mock_chunk]
            self.mock_embedding_generator.generate_embeddings.return_value = [[0.1, 0.2, 0.3]]
            
            chunks, embeddings = self.processor._process_single_document(
                test_files[0], "test-model"
            )
            
            assert len(chunks) == 1
            assert len(embeddings) == 1
            assert chunks[0] == mock_chunk
            assert embeddings[0] == [0.1, 0.2, 0.3]
            
        finally:
            self.cleanup_test_documents(test_files)
    
    def test_process_batch_sequential(self):
        """Test sequential batch processing."""
        # Create test documents
        test_files = self.create_test_documents(2)
        
        try:
            # Mock chunker and embedding generator
            mock_chunks = [
                DocumentChunk(
                    content=f"test content {i}",
                    ipfs_hash=f"test_hash_{i}",
                    source_path=test_files[i],
                    start_position=0,
                    end_position=15,
                    chunk_sequence=i,
                    creation_timestamp="2023-01-01T00:00:00",
                    chunk_size=15
                )
                for i in range(2)
            ]
            
            self.mock_chunker.chunk_document.side_effect = [[chunk] for chunk in mock_chunks]
            self.mock_embedding_generator.generate_embeddings.side_effect = [
                [[0.1, 0.2, 0.3]], [[0.4, 0.5, 0.6]]
            ]
            
            chunks, embeddings = self.processor._process_batch_sequential(
                test_files, "test-model"
            )
            
            assert len(chunks) == 2
            assert len(embeddings) == 2
            
        finally:
            self.cleanup_test_documents(test_files)
    
    def test_store_batch_results(self):
        """Test storing batch results."""
        mock_chunk = DocumentChunk(
            content="test content",
            ipfs_hash="test_hash",
            source_path="test.txt",
            start_position=0,
            end_position=12,
            chunk_sequence=0,
            creation_timestamp="2023-01-01T00:00:00",
            chunk_size=12
        )
        
        mock_embedding = [0.1, 0.2, 0.3]
        
        self.processor._store_batch_results([mock_chunk], [mock_embedding])
        
        self.mock_video_storage.add_document_chunk.assert_called_once_with(
            mock_chunk, mock_embedding
        )
    
    def test_update_batch_stats_success(self):
        """Test updating batch statistics for successful batch."""
        batch_paths = ['doc1.txt', 'doc2.txt']
        batch_chunks = [Mock(), Mock()]
        batch_embeddings = [Mock(), Mock()]
        batch_time = 1.5
        
        initial_batches = self.processor.stats.total_batches
        initial_successful = self.processor.stats.successful_batches
        initial_processed = self.processor.stats.processed_documents
        
        self.processor._update_batch_stats(
            batch_paths, batch_chunks, batch_embeddings, batch_time, success=True
        )
        
        assert self.processor.stats.total_batches == initial_batches + 1
        assert self.processor.stats.successful_batches == initial_successful + 1
        assert self.processor.stats.processed_documents == initial_processed + 2
        assert self.processor.stats.total_chunks == 2
        assert self.processor.stats.total_embeddings == 2
        assert batch_time in self.processor.stats.batch_times
    
    def test_update_batch_stats_failure(self):
        """Test updating batch statistics for failed batch."""
        batch_paths = ['doc1.txt', 'doc2.txt']
        
        initial_batches = self.processor.stats.total_batches
        initial_failed = self.processor.stats.failed_batches
        initial_processed = self.processor.stats.processed_documents
        
        self.processor._update_batch_stats(
            batch_paths, [], [], 0.0, success=False
        )
        
        assert self.processor.stats.total_batches == initial_batches + 1
        assert self.processor.stats.failed_batches == initial_failed + 1
        assert self.processor.stats.processed_documents == initial_processed  # No change
    
    def test_create_progress_report(self):
        """Test creating progress report."""
        self.processor.stats.total_documents = 100
        self.processor.stats.processed_documents = 25
        self.processor.stats.total_chunks = 50
        self.processor.stats.total_embeddings = 50
        
        progress = self.processor._create_progress_report()
        
        assert isinstance(progress, ProcessingProgress)
        assert progress.total_documents == 100
        assert progress.processed_documents == 25
        assert progress.chunks_created == 50
        assert progress.embeddings_generated == 50
        assert progress.processing_time >= 0
    
    def test_generate_final_metrics(self):
        """Test generating final metrics."""
        self.processor.stats.processed_documents = 10
        self.processor.stats.total_chunks = 20
        self.processor.stats.peak_memory_usage_mb = 256.0
        
        metrics = self.processor._generate_final_metrics()
        
        assert isinstance(metrics, RAGMetrics)
        assert metrics.total_documents_processed == 10
        assert metrics.total_chunks_created == 20
        assert metrics.memory_usage_mb == 256.0
        assert metrics.average_chunk_size == 2.0  # 20 chunks / 10 docs
    
    @patch('psutil.virtual_memory')
    def test_memory_management_and_batch_size_adjustment(self, mock_virtual_memory):
        """Test memory management and dynamic batch size adjustment."""
        # Enable dynamic batching
        self.processor.config.enable_dynamic_batching = True
        self.processor.config.memory_check_interval = 1
        self.processor.stats.processed_documents = 1
        
        # High memory usage should reduce batch size
        mock_virtual_memory.return_value.percent = 90.0
        initial_batch_size = self.processor.stats.current_batch_size
        
        self.processor._manage_memory_and_batch_size()
        
        assert self.processor.stats.current_batch_size < initial_batch_size
    
    def test_process_document_collection_integration(self):
        """Test complete document collection processing."""
        # Create test documents
        test_files = self.create_test_documents(2)
        
        try:
            # Mock all components
            mock_chunks = [
                DocumentChunk(
                    content=f"test content {i}",
                    ipfs_hash=f"test_hash_{i}",
                    source_path=test_files[i],
                    start_position=0,
                    end_position=15,
                    chunk_sequence=i,
                    creation_timestamp="2023-01-01T00:00:00",
                    chunk_size=15
                )
                for i in range(2)
            ]
            
            self.mock_chunker.chunk_document.side_effect = [[chunk] for chunk in mock_chunks]
            self.mock_embedding_generator.generate_embeddings.side_effect = [
                [[0.1, 0.2, 0.3]], [[0.4, 0.5, 0.6]]
            ]
            
            # Process collection
            metrics = self.processor.process_document_collection(test_files, "test-model")
            
            # Verify results
            assert isinstance(metrics, RAGMetrics)
            assert metrics.total_documents_processed == 2
            assert metrics.total_chunks_created == 2
            
            # Verify video storage was called
            assert self.mock_video_storage.add_document_chunk.call_count == 2
            
        finally:
            self.cleanup_test_documents(test_files)
    
    def test_document_validation_integration(self):
        """Test document validation integration in batch processing."""
        # Create test documents with different types
        test_files = self.create_test_documents(3)
        
        # Create a config with document validation enabled
        filter_config = DocumentFilterConfig(
            allowed_types={DocumentType.TEXT},
            max_file_size_mb=1.0,
            min_file_size_bytes=5
        )
        
        config = BatchConfig(
            initial_batch_size=2,
            enable_document_validation=True,
            document_filter_config=filter_config
        )
        
        processor = BatchDocumentProcessor(
            chunker=self.mock_chunker,
            embedding_generator=self.mock_embedding_generator,
            video_storage=self.mock_video_storage,
            config=config
        )
        
        try:
            # Mock chunker and embedding generator
            mock_chunk = DocumentChunk(
                content="test content",
                ipfs_hash="test_hash",
                source_path="test.txt",
                start_position=0,
                end_position=12,
                chunk_sequence=0,
                creation_timestamp="2023-01-01T00:00:00",
                chunk_size=12
            )
            
            self.mock_chunker.chunk_document.return_value = [mock_chunk]
            self.mock_embedding_generator.generate_embeddings.return_value = [[0.1, 0.2, 0.3]]
            
            # Process collection with validation
            metrics = processor.process_document_collection(test_files, "test-model")
            
            # Verify that validation occurred
            assert len(processor.stats.validation_results) == 3
            
            # Some documents should be valid (text files)
            valid_count = sum(1 for r in processor.stats.validation_results if r.is_valid)
            assert valid_count > 0
            
        finally:
            self.cleanup_test_documents(test_files)
    
    def test_batch_encoding_as_video_frames(self):
        """Test encoding batches as synchronized video frame sequences."""
        mock_chunks = [
            DocumentChunk(
                content="test content 1",
                ipfs_hash="hash1",
                source_path="test1.txt",
                start_position=0,
                end_position=14,
                chunk_sequence=0,
                creation_timestamp="2023-01-01T00:00:00",
                chunk_size=14
            ),
            DocumentChunk(
                content="test content 2",
                ipfs_hash="hash2",
                source_path="test2.txt",
                start_position=0,
                end_position=14,
                chunk_sequence=1,
                creation_timestamp="2023-01-01T00:00:00",
                chunk_size=14
            )
        ]
        
        mock_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        
        # Mock video storage to return frame metadata
        from unittest.mock import Mock
        mock_metadata = Mock()
        self.mock_video_storage.add_document_chunk.return_value = mock_metadata
        
        # Test the encoding method
        self.processor._encode_batch_as_video_frames(mock_chunks, mock_embeddings)
        
        # Verify that video storage was called for each chunk-embedding pair
        assert self.mock_video_storage.add_document_chunk.call_count == 2
        
        # Verify the calls were made with correct parameters
        calls = self.mock_video_storage.add_document_chunk.call_args_list
        assert calls[0][0][0] == mock_chunks[0]  # First chunk
        assert calls[0][0][1] == mock_embeddings[0]  # First embedding
        assert calls[1][0][0] == mock_chunks[1]  # Second chunk
        assert calls[1][0][1] == mock_embeddings[1]  # Second embedding


if __name__ == "__main__":
    pytest.main([__file__])