"""
Tests for dual-video storage system implementation.
"""

import os
import tempfile
import shutil
import numpy as np
import pytest
from unittest.mock import Mock

from hilbert_quantization.rag.models import DocumentChunk
from hilbert_quantization.rag.video_storage.dual_storage import DualVideoStorageImpl
from hilbert_quantization.rag.video_storage.video_manager import VideoFileManager


class TestVideoFileManager:
    """Test cases for VideoFileManager."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = Mock()
        self.config.frame_rate = 30.0
        self.config.video_codec = 'mp4v'
        self.manager = VideoFileManager(self.config)
    
    def teardown_method(self):
        """Clean up test environment."""
        self.manager.close_all_writers()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_video_file(self):
        """Test video file creation."""
        video_path = os.path.join(self.temp_dir, 'test_video.mp4')
        frame_dimensions = (480, 640)
        
        self.manager.create_video_file(video_path, frame_dimensions)
        
        assert video_path in self.manager._video_writers
        assert video_path in self.manager._video_metadata
        assert self.manager._video_metadata[video_path]['frame_dimensions'] == (480, 640)
        assert self.manager._video_metadata[video_path]['frame_rate'] == 30.0
    
    def test_add_frame_to_video(self):
        """Test adding frames to video."""
        video_path = os.path.join(self.temp_dir, 'test_video.mp4')
        frame_dimensions = (480, 640)
        
        self.manager.create_video_file(video_path, frame_dimensions)
        
        # Create test frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        self.manager.add_frame(video_path, frame, 0)
        
        assert self.manager._video_metadata[video_path]['frame_count'] == 1
    
    def test_prepare_frame_for_video_grayscale(self):
        """Test frame preparation for grayscale input."""
        video_path = os.path.join(self.temp_dir, 'test_video.mp4')
        frame_dimensions = (480, 640)
        
        self.manager.create_video_file(video_path, frame_dimensions)
        
        # Test grayscale frame
        gray_frame = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        prepared_frame = self.manager._prepare_frame_for_video(gray_frame, video_path)
        
        assert prepared_frame.shape == (480, 640, 3)
        assert prepared_frame.dtype == np.uint8
    
    def test_prepare_frame_for_video_resize(self):
        """Test frame preparation with resizing."""
        video_path = os.path.join(self.temp_dir, 'test_video.mp4')
        frame_dimensions = (480, 640)
        
        self.manager.create_video_file(video_path, frame_dimensions)
        
        # Test frame with different dimensions
        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        prepared_frame = self.manager._prepare_frame_for_video(frame, video_path)
        
        assert prepared_frame.shape == (480, 640, 3)
        assert prepared_frame.dtype == np.uint8


class TestDualVideoStorageImpl:
    """Test cases for DualVideoStorageImpl."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = Mock()
        self.config.max_frames_per_file = 100
        self.config.frame_rate = 30.0
        self.config.video_codec = 'mp4v'
        self.config.compression_quality = 0.8
        self.config.storage_root = self.temp_dir
        self.config.embedding_model = 'test_model'
        
        self.storage = DualVideoStorageImpl(self.config)
    
    def teardown_method(self):
        """Clean up test environment."""
        self.storage.video_manager.close_all_writers()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test dual video storage initialization."""
        assert self.storage.max_frames_per_file == 100
        assert self.storage.frame_rate == 30.0
        assert self.storage.video_codec == 'mp4v'
        assert self.storage.compression_quality == 0.8
        
        # Check directories are created
        assert os.path.exists(self.storage.embedding_video_dir)
        assert os.path.exists(self.storage.document_video_dir)
        assert os.path.exists(self.storage.metadata_dir)
    
    def test_get_current_video_paths(self):
        """Test video path generation."""
        embedding_path, document_path = self.storage._get_current_video_paths()
        
        assert 'embeddings_000000.mp4' in embedding_path
        assert 'documents_000000.mp4' in document_path
        assert self.storage.embedding_video_dir in embedding_path
        assert self.storage.document_video_dir in document_path
    
    def test_check_rollover_needed(self):
        """Test rollover detection."""
        # Initially no rollover needed
        assert not self.storage._check_rollover_needed()
        
        # Set frame count to limit
        self.storage.current_frame_count = self.storage.max_frames_per_file
        assert self.storage._check_rollover_needed()
    
    def test_rollover_to_new_videos(self):
        """Test video rollover functionality."""
        initial_index = self.storage.current_video_index
        initial_count = self.storage.current_frame_count
        
        self.storage._rollover_to_new_videos()
        
        assert self.storage.current_video_index == initial_index + 1
        assert self.storage.current_frame_count == 0
    
    def test_convert_chunk_to_frame(self):
        """Test document chunk to frame conversion."""
        chunk = DocumentChunk(
            content="This is a test document chunk with some content.",
            ipfs_hash="QmTest123",
            source_path="/test/document.txt",
            start_position=0,
            end_position=48,
            chunk_sequence=1,
            creation_timestamp="2024-01-01T00:00:00Z",
            chunk_size=48
        )
        
        frame = self.storage._convert_chunk_to_frame(chunk)
        
        assert frame.shape == (480, 640, 3)
        assert frame.dtype == np.uint8
    
    def test_add_document_chunk(self):
        """Test adding synchronized document chunk and embedding frame."""
        # Create test chunk
        chunk = DocumentChunk(
            content="This is a test document chunk for video storage.",
            ipfs_hash="QmTest123456",
            source_path="/test/document.txt",
            start_position=0,
            end_position=47,
            chunk_sequence=1,
            creation_timestamp="2024-01-01T00:00:00Z",
            chunk_size=47
        )
        
        # Create test embedding frame
        embedding_frame = np.random.rand(64, 64, 3).astype(np.float32)
        
        # Add chunk and frame
        metadata = self.storage.add_document_chunk(chunk, embedding_frame)
        
        # Verify metadata
        assert metadata.frame_index == 0
        assert metadata.chunk_id == "QmTest123456_1"
        assert metadata.ipfs_hash == "QmTest123456"
        assert metadata.source_document == "/test/document.txt"
        assert metadata.compression_quality == 0.8
        assert metadata.embedding_model == "test_model"
        assert metadata.chunk_metadata == chunk
        
        # Verify frame count updated
        assert self.storage.current_frame_count == 1
        assert len(self.storage.frame_metadata) == 1
    
    def test_add_multiple_chunks_with_rollover(self):
        """Test adding multiple chunks with automatic rollover."""
        # Set low frame limit for testing
        self.storage.max_frames_per_file = 2
        
        chunks = []
        for i in range(3):
            chunk = DocumentChunk(
                content=f"Test document chunk {i}",
                ipfs_hash=f"QmTest{i:06d}",
                source_path=f"/test/document_{i}.txt",
                start_position=i * 20,
                end_position=(i + 1) * 20,
                chunk_sequence=i,
                creation_timestamp="2024-01-01T00:00:00Z",
                chunk_size=20
            )
            chunks.append(chunk)
        
        # Add chunks
        for i, chunk in enumerate(chunks):
            embedding_frame = np.random.rand(64, 64, 3).astype(np.float32)
            metadata = self.storage.add_document_chunk(chunk, embedding_frame)
            
            if i < 2:
                # First two chunks in first video
                assert metadata.frame_index == i
                assert self.storage.current_video_index == 0
            else:
                # Third chunk triggers rollover
                assert metadata.frame_index == 2  # Global frame number
                assert self.storage.current_video_index == 1
                assert self.storage.current_frame_count == 1
    
    def test_save_and_load_metadata(self):
        """Test metadata persistence."""
        # Add a chunk
        chunk = DocumentChunk(
            content="Test chunk for metadata persistence",
            ipfs_hash="QmTestPersist",
            source_path="/test/persist.txt",
            start_position=0,
            end_position=34,
            chunk_sequence=1,
            creation_timestamp="2024-01-01T00:00:00Z",
            chunk_size=34
        )
        
        embedding_frame = np.random.rand(64, 64, 3).astype(np.float32)
        self.storage.add_document_chunk(chunk, embedding_frame)
        
        # Create new storage instance to test loading
        new_storage = DualVideoStorageImpl(self.config)
        
        # Verify metadata was loaded
        assert new_storage.current_frame_count == 1
        assert len(new_storage.frame_metadata) == 1
        assert new_storage.frame_metadata[0].chunk_id == "QmTestPersist_1"
    
    def test_extract_hierarchical_indices(self):
        """Test hierarchical indices extraction from embedding frames."""
        # Create frame with hierarchical indices (taller than wide)
        embedding_frame = np.random.rand(68, 64, 3).astype(np.float32)  # 64x64 main + 4 index rows
        
        indices = self.storage._extract_hierarchical_indices(embedding_frame)
        
        assert indices is not None
        assert len(indices) == 4  # 4 additional rows
        assert all(len(idx) == 64 * 3 for idx in indices)  # Each row has width * channels
    
    def test_extract_hierarchical_indices_square_frame(self):
        """Test hierarchical indices extraction from square frame."""
        # Create square frame (no hierarchical indices)
        embedding_frame = np.random.rand(64, 64, 3).astype(np.float32)
        
        indices = self.storage._extract_hierarchical_indices(embedding_frame)
        
        assert indices is None
    
    def test_calculate_hierarchical_similarity(self):
        """Test hierarchical similarity calculation."""
        # Create test indices
        indices1 = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
        indices2 = [np.array([1.1, 2.1, 3.1]), np.array([4.1, 5.1, 6.1])]
        
        similarity = self.storage._calculate_hierarchical_similarity(indices1, indices2)
        
        assert 0.0 <= similarity <= 1.0
        assert similarity > 0.9  # Should be high similarity for similar vectors
    
    def test_calculate_hierarchical_similarity_empty(self):
        """Test hierarchical similarity with empty indices."""
        indices1 = []
        indices2 = [np.array([1.0, 2.0, 3.0])]
        
        similarity = self.storage._calculate_hierarchical_similarity(indices1, indices2)
        
        assert similarity == 0.0
    
    def test_find_optimal_insertion_point_empty(self):
        """Test optimal insertion point with no existing frames."""
        embedding_frame = np.random.rand(64, 64, 3).astype(np.float32)
        
        insertion_point = self.storage.find_optimal_insertion_point(embedding_frame)
        
        assert insertion_point == 0
    
    def test_find_optimal_insertion_point_with_indices(self):
        """Test optimal insertion point with hierarchical indices."""
        # Add some frames first
        for i in range(3):
            chunk = DocumentChunk(
                content=f"Test chunk {i}",
                ipfs_hash=f"QmTest{i}",
                source_path=f"/test/doc_{i}.txt",
                start_position=i * 10,
                end_position=(i + 1) * 10,
                chunk_sequence=i,
                creation_timestamp="2024-01-01T00:00:00Z",
                chunk_size=10
            )
            
            # Create frame with hierarchical indices
            embedding_frame = np.random.rand(68, 64, 3).astype(np.float32)
            metadata = self.storage.add_document_chunk(chunk, embedding_frame)
            
            # Add mock hierarchical indices
            metadata.hierarchical_indices = [
                np.random.rand(64),
                np.random.rand(64)
            ]
        
        # Test insertion point for new frame
        new_embedding_frame = np.random.rand(68, 64, 3).astype(np.float32)
        insertion_point = self.storage.find_optimal_insertion_point(new_embedding_frame)
        
        assert 0 <= insertion_point <= len(self.storage.frame_metadata)
    
    def test_insert_synchronized_frames(self):
        """Test synchronized frame insertion."""
        # Add initial frame
        chunk1 = DocumentChunk(
            content="First chunk",
            ipfs_hash="QmFirst",
            source_path="/test/first.txt",
            start_position=0,
            end_position=11,
            chunk_sequence=1,
            creation_timestamp="2024-01-01T00:00:00Z",
            chunk_size=11
        )
        
        embedding_frame1 = np.random.rand(64, 64, 3).astype(np.float32)
        self.storage.add_document_chunk(chunk1, embedding_frame1)
        
        # Insert new frame
        chunk2 = DocumentChunk(
            content="Second chunk",
            ipfs_hash="QmSecond",
            source_path="/test/second.txt",
            start_position=0,
            end_position=12,
            chunk_sequence=2,
            creation_timestamp="2024-01-01T00:00:00Z",
            chunk_size=12
        )
        
        embedding_frame2 = np.random.rand(64, 64, 3).astype(np.float32)
        metadata = self.storage.insert_synchronized_frames(chunk2, embedding_frame2)
        
        assert metadata is not None
        assert metadata.chunk_id == "QmSecond_2"
        assert len(self.storage.frame_metadata) == 2
    
    def test_reindex_frames_after_insertion(self):
        """Test frame reindexing after insertion."""
        # Add multiple frames
        for i in range(3):
            chunk = DocumentChunk(
                content=f"Chunk {i}",
                ipfs_hash=f"QmChunk{i}",
                source_path=f"/test/chunk_{i}.txt",
                start_position=i * 10,
                end_position=(i + 1) * 10,
                chunk_sequence=i,
                creation_timestamp="2024-01-01T00:00:00Z",
                chunk_size=10
            )
            
            embedding_frame = np.random.rand(64, 64, 3).astype(np.float32)
            self.storage.add_document_chunk(chunk, embedding_frame)
        
        # Test reindexing
        self.storage.reindex_frames_after_insertion(1)
        
        # Verify frames are still sorted by frame index
        frame_indices = [metadata.frame_index for metadata in self.storage.frame_metadata]
        assert frame_indices == sorted(frame_indices)
    
    def test_get_video_metadata(self):
        """Test comprehensive video metadata retrieval."""
        # Add some frames first
        for i in range(2):
            chunk = DocumentChunk(
                content=f"Test chunk {i} for metadata",
                ipfs_hash=f"QmMeta{i}",
                source_path=f"/test/meta_{i}.txt",
                start_position=i * 20,
                end_position=(i + 1) * 20,
                chunk_sequence=i,
                creation_timestamp="2024-01-01T00:00:00Z",
                chunk_size=20
            )
            
            embedding_frame = np.random.rand(64, 64, 3).astype(np.float32)
            self.storage.add_document_chunk(chunk, embedding_frame)
        
        metadata = self.storage.get_video_metadata()
        
        # Check structure
        assert 'storage_info' in metadata
        assert 'video_settings' in metadata
        assert 'storage_paths' in metadata
        assert 'video_files' in metadata
        assert 'compression_stats' in metadata
        assert 'frame_metadata_summary' in metadata
        
        # Check storage info
        storage_info = metadata['storage_info']
        assert storage_info['total_frames'] == 2
        assert storage_info['total_documents_stored'] == 2
        
        # Check video settings
        video_settings = metadata['video_settings']
        assert video_settings['frame_rate'] == 30.0
        assert video_settings['video_codec'] == 'mp4v'
        assert video_settings['compression_quality'] == 0.8
        
        # Check frame metadata summary
        frame_summary = metadata['frame_metadata_summary']
        assert frame_summary['total_frames'] == 2
        assert frame_summary['unique_documents'] == 2
        assert frame_summary['average_chunk_size'] == 20
    
    def test_get_frame_metadata_by_range(self):
        """Test frame metadata retrieval by range."""
        # Add frames
        for i in range(5):
            chunk = DocumentChunk(
                content=f"Range test chunk {i}",
                ipfs_hash=f"QmRange{i}",
                source_path=f"/test/range_{i}.txt",
                start_position=i * 10,
                end_position=(i + 1) * 10,
                chunk_sequence=i,
                creation_timestamp="2024-01-01T00:00:00Z",
                chunk_size=10
            )
            
            embedding_frame = np.random.rand(64, 64, 3).astype(np.float32)
            self.storage.add_document_chunk(chunk, embedding_frame)
        
        # Test range retrieval
        range_metadata = self.storage.get_frame_metadata_by_range(1, 4)
        
        assert len(range_metadata) == 3
        assert all(1 <= meta.frame_index < 4 for meta in range_metadata)
    
    def test_get_frame_metadata_by_document(self):
        """Test frame metadata retrieval by document."""
        # Add frames from same document
        for i in range(3):
            chunk = DocumentChunk(
                content=f"Document chunk {i}",
                ipfs_hash="QmSameDoc",  # Same document
                source_path="/test/same_doc.txt",
                start_position=i * 15,
                end_position=(i + 1) * 15,
                chunk_sequence=i,
                creation_timestamp="2024-01-01T00:00:00Z",
                chunk_size=15
            )
            
            embedding_frame = np.random.rand(64, 64, 3).astype(np.float32)
            self.storage.add_document_chunk(chunk, embedding_frame)
        
        # Test document-based retrieval
        doc_metadata = self.storage.get_frame_metadata_by_document("QmSameDoc")
        
        assert len(doc_metadata) == 3
        assert all(meta.ipfs_hash == "QmSameDoc" for meta in doc_metadata)
    
    def test_optimize_video_compression(self):
        """Test video compression optimization."""
        initial_quality = self.storage.compression_quality
        
        # Optimize compression
        result = self.storage.optimize_video_compression(0.9)
        
        assert result['old_quality'] == initial_quality
        assert result['new_quality'] == 0.9
        assert result['optimization_applied'] is True
        assert self.storage.compression_quality == 0.9
    
    def test_optimize_video_compression_invalid_quality(self):
        """Test video compression optimization with invalid quality."""
        with pytest.raises(ValueError):
            self.storage.optimize_video_compression(1.5)
        
        with pytest.raises(ValueError):
            self.storage.optimize_video_compression(-0.1)


class TestVideoFileManagerCompression:
    """Test cases for VideoFileManager compression features."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = Mock()
        self.config.frame_rate = 30.0
        self.config.video_codec = 'mp4v'
        self.manager = VideoFileManager(self.config)
    
    def teardown_method(self):
        """Clean up test environment."""
        self.manager.close_all_writers()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_get_video_metadata_existing_file(self):
        """Test metadata retrieval for existing video file."""
        video_path = os.path.join(self.temp_dir, 'test_video.mp4')
        frame_dimensions = (480, 640)
        
        # Create video and add frame
        self.manager.create_video_file(video_path, frame_dimensions)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.manager.add_frame(video_path, frame, 0)
        self.manager.close_video_writer(video_path)
        
        # Get metadata
        metadata = self.manager.get_video_metadata(video_path)
        
        assert 'frame_dimensions' in metadata
        assert 'frame_rate' in metadata
        assert 'codec' in metadata
        assert 'frame_count' in metadata
        assert metadata['frame_dimensions'] == (480, 640)
    
    def test_get_compression_statistics(self):
        """Test compression statistics calculation."""
        video_path = os.path.join(self.temp_dir, 'test_compression.mp4')
        frame_dimensions = (240, 320)
        
        # Create video and add frames
        self.manager.create_video_file(video_path, frame_dimensions)
        for i in range(5):
            frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            self.manager.add_frame(video_path, frame, i)
        self.manager.close_video_writer(video_path)
        
        # Get compression stats
        stats = self.manager.get_compression_statistics(video_path)
        
        assert 'file_size_mb' in stats
        assert 'estimated_uncompressed_mb' in stats
        assert 'compression_ratio' in stats
        assert 'frame_count' in stats
        assert 'dimensions' in stats
        assert stats['frame_count'] == 5
        assert stats['dimensions'] == (320, 240)  # Note: OpenCV returns (width, height)
    
    def test_update_compression_settings(self):
        """Test compression settings update."""
        video_path = os.path.join(self.temp_dir, 'test_settings.mp4')
        frame_dimensions = (480, 640)
        
        self.manager.create_video_file(video_path, frame_dimensions)
        self.manager.update_compression_settings(video_path, 0.9)
        
        metadata = self.manager.get_video_metadata(video_path)
        assert metadata.get('compression_quality') == 0.9


if __name__ == "__main__":
    pytest.main([__file__])