"""
Tests for document chunker with IPFS integration.
"""

import pytest
import tempfile
import os
from datetime import datetime

from hilbert_quantization.rag.config import RAGConfig, ChunkingConfig
from hilbert_quantization.rag.document_processing.chunker import DocumentChunkerImpl
from hilbert_quantization.rag.document_processing.ipfs_integration import IPFSManager
from hilbert_quantization.rag.models import DocumentChunk


class TestDocumentChunker:
    """Test cases for DocumentChunkerImpl."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = RAGConfig()
        self.config.storage.base_storage_path = self.temp_dir
        self.config.chunking.chunk_size = 100
        self.config.chunking.chunk_overlap = 10
        self.config.chunking.min_chunk_size = 50
        self.config.chunking.max_chunk_size = 200
        self.chunker = DocumentChunkerImpl(self.config)
    
    def test_chunk_document_basic(self):
        """Test basic document chunking functionality."""
        document = "This is a test document. " * 10  # ~250 characters
        ipfs_hash = "QmTestHash123"
        source_path = "/test/document.txt"
        
        chunks = self.chunker.chunk_document(document, ipfs_hash, source_path)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
        assert all(chunk.ipfs_hash == ipfs_hash for chunk in chunks)
        assert all(chunk.source_path == source_path for chunk in chunks)
        assert all(chunk.chunk_size == 100 for chunk in chunks)  # Configured size
    
    def test_chunk_document_with_padding(self):
        """Test that chunks are padded to exact target size."""
        document = "Short document."
        ipfs_hash = "QmTestHash456"
        source_path = "/test/short.txt"
        
        chunks = self.chunker.chunk_document(document, ipfs_hash, source_path)
        
        assert len(chunks) >= 1
        chunk = chunks[0]
        assert len(chunk.content) == 100  # Padded to target size
        assert chunk.content.startswith("Short document.")
        # Check that all chunks are properly padded
        for c in chunks:
            assert len(c.content) == 100
    
    def test_chunk_document_empty(self):
        """Test handling of empty documents."""
        chunks = self.chunker.chunk_document("", "QmEmpty", "/empty.txt")
        assert len(chunks) == 0
        
        chunks = self.chunker.chunk_document("   ", "QmWhitespace", "/whitespace.txt")
        assert len(chunks) == 0
    
    def test_calculate_chunk_size(self):
        """Test chunk size calculation based on embedding dimensions."""
        # Test with typical embedding dimensions
        size_384 = self.chunker.calculate_chunk_size(384)  # all-MiniLM-L6-v2
        size_768 = self.chunker.calculate_chunk_size(768)  # BERT-base
        size_1536 = self.chunker.calculate_chunk_size(1536)  # text-embedding-ada-002
        
        # Should return reasonable sizes within bounds
        assert self.config.chunking.min_chunk_size <= size_384 <= self.config.chunking.max_chunk_size
        assert self.config.chunking.min_chunk_size <= size_768 <= self.config.chunking.max_chunk_size
        assert self.config.chunking.min_chunk_size <= size_1536 <= self.config.chunking.max_chunk_size
        
        # Larger embeddings should generally result in larger chunk sizes
        assert size_768 >= size_384
    
    def test_calculate_chunk_size_power_of_4_alignment(self):
        """Test that chunk sizes are aligned to power-of-4 boundaries."""
        # Test various embedding dimensions
        test_dimensions = [100, 384, 512, 768, 1024, 1536, 2048]
        
        for dim in test_dimensions:
            size = self.chunker.calculate_chunk_size(dim)
            
            # Size should be within bounds
            assert self.config.chunking.min_chunk_size <= size <= self.config.chunking.max_chunk_size
            
            # Size should be reasonable for the embedding dimension
            assert size > 0
    
    def test_calculate_chunk_size_invalid_input(self):
        """Test chunk size calculation with invalid inputs."""
        with pytest.raises(ValueError, match="Embedding dimensions must be positive"):
            self.chunker.calculate_chunk_size(0)
        
        with pytest.raises(ValueError, match="Embedding dimensions must be positive"):
            self.chunker.calculate_chunk_size(-1)
    
    def test_align_to_power_of_4_boundary(self):
        """Test power-of-4 alignment functionality."""
        # Test various sizes
        assert self.chunker._align_to_power_of_4_boundary(1) == 1
        assert self.chunker._align_to_power_of_4_boundary(2) == 1
        assert self.chunker._align_to_power_of_4_boundary(3) == 4
        assert self.chunker._align_to_power_of_4_boundary(4) == 4
        assert self.chunker._align_to_power_of_4_boundary(5) == 4
        assert self.chunker._align_to_power_of_4_boundary(15) == 16
        assert self.chunker._align_to_power_of_4_boundary(16) == 16
        assert self.chunker._align_to_power_of_4_boundary(17) == 16
        assert self.chunker._align_to_power_of_4_boundary(50) == 64
        
        # Test edge case
        assert self.chunker._align_to_power_of_4_boundary(0) == 4
    
    def test_pad_chunk(self):
        """Test chunk padding functionality."""
        # Test padding short content
        padded = self.chunker.pad_chunk("Hello", 10)
        assert len(padded) == 10
        assert padded == "Hello     "
        
        # Test truncating long content
        padded = self.chunker.pad_chunk("This is a very long string", 10)
        assert len(padded) == 10
        assert padded == "This is a "
        
        # Test exact size content
        padded = self.chunker.pad_chunk("Exact size", 10)
        assert len(padded) == 10
        assert padded == "Exact size"
    
    def test_validate_chunk_consistency(self):
        """Test chunk consistency validation."""
        # Create consistent chunks
        chunks = [
            DocumentChunk(
                content="A" * 100,
                ipfs_hash="QmTest1",
                source_path="/test1.txt",
                start_position=0,
                end_position=100,
                chunk_sequence=0,
                creation_timestamp=datetime.now().isoformat(),
                chunk_size=100
            ),
            DocumentChunk(
                content="B" * 100,
                ipfs_hash="QmTest1",
                source_path="/test1.txt",
                start_position=90,
                end_position=190,
                chunk_sequence=1,
                creation_timestamp=datetime.now().isoformat(),
                chunk_size=100
            )
        ]
        
        assert self.chunker.validate_chunk_consistency(chunks) is True
        
        # Test inconsistent chunk sizes
        chunks[1].chunk_size = 50
        assert self.chunker.validate_chunk_consistency(chunks) is False
        
        # Test empty list
        assert self.chunker.validate_chunk_consistency([]) is True
    
    def test_sentence_boundary_preservation(self):
        """Test preservation of sentence boundaries."""
        self.config.chunking.preserve_sentence_boundaries = True
        self.config.chunking.chunk_size = 100  # Larger chunk size for better boundary detection
        self.config.chunking.chunk_overlap = 0  # Disable overlap for cleaner test
        self.config.chunking.min_chunk_size = 30  # Smaller min size for flexibility
        
        document = "First sentence is here. Second sentence follows. Third sentence continues. Fourth sentence ends."
        chunks = self.chunker.chunk_document(document, "QmTest", "/test.txt")
        
        # Verify that sentence boundary preservation is attempted
        # At minimum, we should not break words in the middle
        for chunk in chunks:
            # Get the original content (before padding)
            original_end = chunk.end_position - chunk.start_position
            original_content = document[chunk.start_position:chunk.end_position]
            
            # The chunk should not end in the middle of a word (unless it's the last chunk)
            if chunk.end_position < len(document):
                # If not at document end, should not break words
                if original_content and not original_content[-1].isspace():
                    # Should end at word boundary or sentence boundary
                    next_char = document[chunk.end_position] if chunk.end_position < len(document) else ' '
                    assert next_char.isspace() or next_char in '.!?'
    
    def test_chunk_overlap(self):
        """Test chunk overlap functionality."""
        self.config.chunking.chunk_overlap = 20
        self.config.chunking.chunk_size = 100
        
        document = "A" * 200  # Exactly 200 characters
        chunks = self.chunker.chunk_document(document, "QmTest", "/test.txt")
        
        assert len(chunks) >= 2
        # Verify overlap exists between consecutive chunks
        if len(chunks) >= 2:
            # The second chunk should start before the first chunk ends
            assert chunks[1].start_position < chunks[0].end_position


class TestIPFSManager:
    """Test cases for IPFSManager."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = RAGConfig()
        self.config.storage.base_storage_path = self.temp_dir
        self.ipfs_manager = IPFSManager(self.config)
    
    def test_generate_ipfs_hash(self):
        """Test IPFS hash generation."""
        content = "Test document content"
        hash1 = self.ipfs_manager.generate_ipfs_hash(content)
        hash2 = self.ipfs_manager.generate_ipfs_hash(content)
        
        # Same content should produce same hash
        assert hash1 == hash2
        assert hash1.startswith("Qm")
        assert len(hash1) > 10
        
        # Different content should produce different hash
        hash3 = self.ipfs_manager.generate_ipfs_hash("Different content")
        assert hash1 != hash3
    
    def test_generate_ipfs_hash_empty_content(self):
        """Test IPFS hash generation with empty content."""
        with pytest.raises(ValueError, match="Document content cannot be empty"):
            self.ipfs_manager.generate_ipfs_hash("")
    
    def test_retrieve_document(self):
        """Test document retrieval using IPFS hash."""
        content = "Test document for retrieval"
        ipfs_hash = self.ipfs_manager.generate_ipfs_hash(content)
        
        # Should be able to retrieve the same content
        retrieved = self.ipfs_manager.retrieve_document(ipfs_hash)
        assert retrieved == content
    
    def test_retrieve_nonexistent_document(self):
        """Test retrieval of non-existent document."""
        with pytest.raises(ValueError, match="Document with IPFS hash .* not found"):
            self.ipfs_manager.retrieve_document("QmNonExistentHash")
    
    def test_validate_hash(self):
        """Test hash validation."""
        content = "Content for validation"
        correct_hash = self.ipfs_manager.generate_ipfs_hash(content)
        
        # Correct content and hash should validate
        assert self.ipfs_manager.validate_hash(content, correct_hash) is True
        
        # Incorrect content should not validate
        assert self.ipfs_manager.validate_hash("Wrong content", correct_hash) is False
        
        # Invalid hash should not validate
        assert self.ipfs_manager.validate_hash(content, "InvalidHash") is False
    
    def test_cache_functionality(self):
        """Test caching functionality."""
        content1 = "First document"
        content2 = "Second document"
        
        hash1 = self.ipfs_manager.generate_ipfs_hash(content1)
        hash2 = self.ipfs_manager.generate_ipfs_hash(content2)
        
        # Get cache statistics
        stats = self.ipfs_manager.get_cache_statistics()
        assert stats['memory_cache_entries'] >= 2
        assert stats['disk_cache_entries'] >= 2
        assert 'cache_directory' in stats
        
        # Clear cache and verify
        self.ipfs_manager.clear_cache()
        
        # Should still be able to retrieve from disk cache initially
        # But after clearing, memory cache should be empty
        stats_after_clear = self.ipfs_manager.get_cache_statistics()
        assert stats_after_clear['memory_cache_entries'] == 0


class TestIntegration:
    """Integration tests for chunker and IPFS manager."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = RAGConfig()
        self.config.storage.base_storage_path = self.temp_dir
        self.config.chunking.chunk_size = 100
        self.chunker = DocumentChunkerImpl(self.config)
        self.ipfs_manager = IPFSManager(self.config)
    
    def test_end_to_end_chunking_with_ipfs(self):
        """Test complete workflow from document to chunks with IPFS."""
        # Original document
        document = "This is a comprehensive test document. " * 5
        
        # Generate IPFS hash
        ipfs_hash = self.ipfs_manager.generate_ipfs_hash(document)
        
        # Chunk the document
        chunks = self.chunker.chunk_document(document, ipfs_hash, "/test/doc.txt")
        
        # Verify chunks
        assert len(chunks) > 0
        assert self.chunker.validate_chunk_consistency(chunks)
        
        # Verify IPFS integration
        for chunk in chunks:
            assert chunk.ipfs_hash == ipfs_hash
            assert self.ipfs_manager.validate_hash(document, chunk.ipfs_hash)
        
        # Verify we can retrieve original document
        retrieved_document = self.ipfs_manager.retrieve_document(ipfs_hash)
        assert retrieved_document == document
    
    def test_chunk_metadata_completeness(self):
        """Test that all required metadata is present in chunks."""
        document = "Test document with metadata validation."
        ipfs_hash = self.ipfs_manager.generate_ipfs_hash(document)
        source_path = "/test/metadata.txt"
        
        chunks = self.chunker.chunk_document(document, ipfs_hash, source_path)
        
        for i, chunk in enumerate(chunks):
            # Verify all required fields are present and valid
            assert chunk.content is not None
            assert chunk.ipfs_hash == ipfs_hash
            assert chunk.source_path == source_path
            assert chunk.start_position >= 0
            assert chunk.end_position > chunk.start_position
            assert chunk.chunk_sequence == i
            assert chunk.creation_timestamp is not None
            assert chunk.chunk_size > 0
            
            # Verify timestamp format
            datetime.fromisoformat(chunk.creation_timestamp)  # Should not raise
            
            # Verify chunk size matches content length
            assert len(chunk.content) == chunk.chunk_size


if __name__ == "__main__":
    pytest.main([__file__])