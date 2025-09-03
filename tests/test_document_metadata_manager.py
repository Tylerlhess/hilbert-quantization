"""
Tests for document metadata management functionality.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch
from hilbert_quantization.rag.document_processing.metadata_manager import DocumentMetadataManager
from hilbert_quantization.rag.models import DocumentChunk
from hilbert_quantization.rag.config import RAGConfig


class TestDocumentMetadataManager:
    """Test cases for DocumentMetadataManager."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = Mock(spec=RAGConfig)
        config.chunking = Mock()
        config.chunking.chunk_size = 1000
        config.chunking.min_chunk_size = 100
        config.chunking.max_chunk_size = 2000
        config.chunking.chunk_overlap = 50
        config.chunking.padding_char = " "
        config.chunking.preserve_sentence_boundaries = True
        
        config.storage = Mock()
        config.storage.base_storage_path = "/tmp/test_rag"
        
        return config
    
    @pytest.fixture
    def metadata_manager(self, config):
        """Create test metadata manager."""
        return DocumentMetadataManager(config)
    
    @pytest.fixture
    def sample_chunk(self):
        """Create sample document chunk."""
        return DocumentChunk(
            content="This is a test document chunk with some content that needs to be processed.",
            ipfs_hash="QmTestHash123456789012345678901234",
            source_path="/path/to/test/document.txt",
            start_position=0,
            end_position=75,
            chunk_sequence=0,
            creation_timestamp="2024-01-01T12:00:00",
            chunk_size=75
        )
    
    def test_create_chunk_metadata_basic(self, metadata_manager, sample_chunk):
        """Test basic metadata creation for document chunk."""
        with patch.object(metadata_manager.ipfs_manager, 'retrieve_document', return_value="Test document content"):
            metadata = metadata_manager.create_chunk_metadata(sample_chunk)
            
            # Check basic fields
            assert 'chunk_id' in metadata
            assert metadata['content_length'] == 75
            assert metadata['chunk_size'] == 75
            assert metadata['chunk_sequence'] == 0
    
    def test_create_chunk_metadata_section_position(self, metadata_manager, sample_chunk):
        """Test section position tracking (Requirement 11.4)."""
        with patch.object(metadata_manager.ipfs_manager, 'retrieve_document', return_value="Test document content"):
            metadata = metadata_manager.create_chunk_metadata(sample_chunk)
            
            section_pos = metadata['section_position']
            assert section_pos['start_position'] == 0
            assert section_pos['end_position'] == 75
            assert section_pos['character_count'] == 75
            assert section_pos['position_type'] == 'character_positions'
    
    def test_create_chunk_metadata_document_metadata(self, metadata_manager, sample_chunk):
        """Test document metadata tracking (Requirement 11.5)."""
        with patch.object(metadata_manager.ipfs_manager, 'retrieve_document', return_value="Test document content"):
            metadata = metadata_manager.create_chunk_metadata(sample_chunk)
            
            doc_metadata = metadata['document_metadata']
            assert doc_metadata['source_path'] == "/path/to/test/document.txt"
            assert doc_metadata['creation_timestamp'] == "2024-01-01T12:00:00"
            assert doc_metadata['chunk_sequence_number'] == 0
            assert 'processing_timestamp' in doc_metadata
            
            # Validate timestamp format
            datetime.fromisoformat(doc_metadata['processing_timestamp'])
    
    def test_create_chunk_metadata_ipfs_metadata(self, metadata_manager, sample_chunk):
        """Test IPFS metadata and retrieval capabilities (Requirement 11.6)."""
        with patch.object(metadata_manager.ipfs_manager, 'retrieve_document', return_value="Test document content"):
            metadata = metadata_manager.create_chunk_metadata(sample_chunk)
            
            ipfs_metadata = metadata['ipfs_metadata']
            assert ipfs_metadata['ipfs_hash'] == "QmTestHash123456789012345678901234"
            assert isinstance(ipfs_metadata['hash_validated'], bool)
            assert isinstance(ipfs_metadata['retrieval_available'], bool)
            assert ipfs_metadata['hash_generation_method'] == 'sha256_base64_truncated'
    
    def test_create_chunk_metadata_validation_status(self, metadata_manager, sample_chunk):
        """Test validation status in metadata."""
        with patch.object(metadata_manager.ipfs_manager, 'retrieve_document', return_value="Test document content"):
            metadata = metadata_manager.create_chunk_metadata(sample_chunk)
            
            validation = metadata['validation_status']
            assert isinstance(validation['size_consistent'], bool)
            assert isinstance(validation['positions_valid'], bool)
            assert isinstance(validation['metadata_complete'], bool)
            assert isinstance(validation['ipfs_hash_valid'], bool)
    
    def test_create_chunk_metadata_invalid_input(self, metadata_manager):
        """Test metadata creation with invalid input."""
        with pytest.raises(ValueError, match="Input must be a DocumentChunk instance"):
            metadata_manager.create_chunk_metadata("not a chunk")
    
    def test_validate_metadata_complete(self, metadata_manager, sample_chunk):
        """Test validation of complete metadata."""
        with patch.object(metadata_manager.ipfs_manager, 'retrieve_document', return_value="Test document content"):
            metadata = metadata_manager.create_chunk_metadata(sample_chunk)
            assert metadata_manager.validate_metadata(metadata) is True
    
    def test_validate_metadata_missing_fields(self, metadata_manager):
        """Test validation with missing required fields."""
        incomplete_metadata = {
            'chunk_id': 'test123',
            'content_length': 100
            # Missing other required fields
        }
        
        assert metadata_manager.validate_metadata(incomplete_metadata) is False
    
    def test_validate_metadata_invalid_section_position(self, metadata_manager, sample_chunk):
        """Test validation with invalid section position."""
        with patch.object(metadata_manager.ipfs_manager, 'retrieve_document', return_value="Test document content"):
            metadata = metadata_manager.create_chunk_metadata(sample_chunk)
            
            # Corrupt section position
            metadata['section_position']['start_position'] = -1
            
            assert metadata_manager.validate_metadata(metadata) is False
    
    def test_validate_metadata_invalid_document_metadata(self, metadata_manager, sample_chunk):
        """Test validation with invalid document metadata."""
        with patch.object(metadata_manager.ipfs_manager, 'retrieve_document', return_value="Test document content"):
            metadata = metadata_manager.create_chunk_metadata(sample_chunk)
            
            # Corrupt document metadata
            metadata['document_metadata']['creation_timestamp'] = "invalid-timestamp"
            
            assert metadata_manager.validate_metadata(metadata) is False
    
    def test_validate_metadata_invalid_ipfs_metadata(self, metadata_manager, sample_chunk):
        """Test validation with invalid IPFS metadata."""
        with patch.object(metadata_manager.ipfs_manager, 'retrieve_document', return_value="Test document content"):
            metadata = metadata_manager.create_chunk_metadata(sample_chunk)
            
            # Remove required IPFS field
            del metadata['ipfs_metadata']['ipfs_hash']
            
            assert metadata_manager.validate_metadata(metadata) is False
    
    def test_validate_metadata_inconsistent_data(self, metadata_manager, sample_chunk):
        """Test validation with inconsistent metadata."""
        with patch.object(metadata_manager.ipfs_manager, 'retrieve_document', return_value="Test document content"):
            metadata = metadata_manager.create_chunk_metadata(sample_chunk)
            
            # Make chunk_size inconsistent with content_length
            metadata['chunk_size'] = 999
            
            assert metadata_manager.validate_metadata(metadata) is False
    
    def test_retrieve_original_document_success(self, metadata_manager):
        """Test successful document retrieval using IPFS hash."""
        test_content = "This is the original document content."
        
        with patch.object(metadata_manager.ipfs_manager, 'retrieve_document', return_value=test_content):
            result = metadata_manager.retrieve_original_document("QmTestHash123")
            assert result == test_content
    
    def test_retrieve_original_document_failure(self, metadata_manager):
        """Test document retrieval failure."""
        with patch.object(metadata_manager.ipfs_manager, 'retrieve_document', side_effect=Exception("Not found")):
            with pytest.raises(ValueError, match="Failed to retrieve document"):
                metadata_manager.retrieve_original_document("QmInvalidHash")
    
    def test_retrieve_original_document_empty_hash(self, metadata_manager):
        """Test document retrieval with empty hash."""
        with pytest.raises(ValueError, match="IPFS hash cannot be empty"):
            metadata_manager.retrieve_original_document("")
    
    def test_validate_chunk_against_original_success(self, metadata_manager, sample_chunk):
        """Test successful chunk validation against original document."""
        original_content = "This is a test document chunk with some content that needs to be processed."
        
        with patch.object(metadata_manager.ipfs_manager, 'retrieve_document', return_value=original_content):
            result = metadata_manager.validate_chunk_against_original(sample_chunk)
            
            assert result['chunk_valid'] is True
            assert result['original_retrieved'] is True
            assert result['position_valid'] is True
            assert result['content_matches'] is True
            assert result['error_message'] is None
    
    def test_validate_chunk_against_original_position_invalid(self, metadata_manager, sample_chunk):
        """Test chunk validation with invalid positions."""
        original_content = "Short content"
        
        # Chunk positions exceed document length
        sample_chunk.end_position = 1000
        
        with patch.object(metadata_manager.ipfs_manager, 'retrieve_document', return_value=original_content):
            result = metadata_manager.validate_chunk_against_original(sample_chunk)
            
            assert result['chunk_valid'] is False
            assert result['original_retrieved'] is True
            assert result['position_valid'] is False
            assert "positions are invalid" in result['error_message']
    
    def test_validate_chunk_against_original_content_mismatch(self, metadata_manager, sample_chunk):
        """Test chunk validation with content mismatch."""
        # Create content that's long enough to contain the chunk positions but has different content
        original_content = "X" * 100  # 100 characters of 'X', different from chunk content
        
        with patch.object(metadata_manager.ipfs_manager, 'retrieve_document', return_value=original_content):
            result = metadata_manager.validate_chunk_against_original(sample_chunk)
            
            assert result['chunk_valid'] is False
            assert result['original_retrieved'] is True
            assert result['position_valid'] is True
            assert result['content_matches'] is False
            assert "content does not match" in result['error_message']
    
    def test_validate_chunk_against_original_retrieval_failure(self, metadata_manager, sample_chunk):
        """Test chunk validation when document retrieval fails."""
        with patch.object(metadata_manager.ipfs_manager, 'retrieve_document', side_effect=Exception("Not found")):
            result = metadata_manager.validate_chunk_against_original(sample_chunk)
            
            assert result['chunk_valid'] is False
            assert result['original_retrieved'] is False
            assert "Validation failed" in result['error_message']
    
    def test_get_chunk_context_success(self, metadata_manager, sample_chunk):
        """Test getting chunk context from original document."""
        # Create content where the chunk starts at position 16 (after "Before content. ")
        original_content = "Before content. This is a test document chunk with some content that needs to be processed. After content."
        
        # Update chunk positions to match the content
        sample_chunk.start_position = 16
        sample_chunk.end_position = 91  # 16 + 75
        
        with patch.object(metadata_manager.ipfs_manager, 'retrieve_document', return_value=original_content):
            context = metadata_manager.get_chunk_context(sample_chunk, context_chars=16)  # Use 16 to get full "Before content. "
            
            assert context['before_context'] == "Before content. "
            assert context['chunk_content'] == "This is a test document chunk with some content that needs to be processed."
            assert context['after_context'] == " After content."
            assert 'full_context' in context
    
    def test_get_chunk_context_boundary_handling(self, metadata_manager, sample_chunk):
        """Test context retrieval with boundary conditions."""
        original_content = "This is a test document chunk with some content that needs to be processed."
        
        # Chunk at document start
        sample_chunk.start_position = 0
        sample_chunk.end_position = 20
        
        with patch.object(metadata_manager.ipfs_manager, 'retrieve_document', return_value=original_content):
            context = metadata_manager.get_chunk_context(sample_chunk, context_chars=10)
            
            assert context['before_context'] == ""
            assert len(context['chunk_content']) == 20
            assert len(context['after_context']) <= 10
    
    def test_get_chunk_context_retrieval_failure(self, metadata_manager, sample_chunk):
        """Test context retrieval when document retrieval fails."""
        with patch.object(metadata_manager.ipfs_manager, 'retrieve_document', side_effect=Exception("Not found")):
            context = metadata_manager.get_chunk_context(sample_chunk)
            
            assert 'error' in context
            assert context['before_context'] == ''
            assert context['chunk_content'] == ''
            assert context['after_context'] == ''
            assert context['full_context'] == ''
    
    def test_generate_chunk_id_consistency(self, metadata_manager, sample_chunk):
        """Test that chunk ID generation is consistent."""
        id1 = metadata_manager._generate_chunk_id(sample_chunk)
        id2 = metadata_manager._generate_chunk_id(sample_chunk)
        
        assert id1 == id2
        assert len(id1) == 16  # MD5 hash truncated to 16 characters
    
    def test_generate_chunk_id_uniqueness(self, metadata_manager, sample_chunk):
        """Test that different chunks generate different IDs."""
        id1 = metadata_manager._generate_chunk_id(sample_chunk)
        
        # Modify chunk
        sample_chunk.chunk_sequence = 1
        id2 = metadata_manager._generate_chunk_id(sample_chunk)
        
        assert id1 != id2
    
    def test_validate_positions_valid(self, metadata_manager, sample_chunk):
        """Test position validation with valid positions."""
        assert metadata_manager._validate_positions(sample_chunk) is True
    
    def test_validate_positions_invalid(self, metadata_manager, sample_chunk):
        """Test position validation with invalid positions."""
        sample_chunk.start_position = -1
        assert metadata_manager._validate_positions(sample_chunk) is False
        
        sample_chunk.start_position = 100
        sample_chunk.end_position = 50
        assert metadata_manager._validate_positions(sample_chunk) is False
        
        sample_chunk.start_position = 0
        sample_chunk.end_position = 100
        sample_chunk.chunk_sequence = -1
        assert metadata_manager._validate_positions(sample_chunk) is False
    
    def test_check_metadata_completeness_complete(self, metadata_manager, sample_chunk):
        """Test metadata completeness check with complete metadata."""
        assert metadata_manager._check_metadata_completeness(sample_chunk) is True
    
    def test_check_metadata_completeness_incomplete(self, metadata_manager, sample_chunk):
        """Test metadata completeness check with incomplete metadata."""
        sample_chunk.ipfs_hash = ""
        assert metadata_manager._check_metadata_completeness(sample_chunk) is False
        
        sample_chunk.ipfs_hash = "QmTestHash"
        sample_chunk.source_path = ""
        assert metadata_manager._check_metadata_completeness(sample_chunk) is False
        
        sample_chunk.source_path = "/path/to/doc"
        sample_chunk.creation_timestamp = ""
        assert metadata_manager._check_metadata_completeness(sample_chunk) is False
        
        sample_chunk.creation_timestamp = "2024-01-01T12:00:00"
        sample_chunk.chunk_size = 0
        assert metadata_manager._check_metadata_completeness(sample_chunk) is False
    
    def test_validate_section_position_valid(self, metadata_manager):
        """Test section position validation with valid data."""
        section_position = {
            'start_position': 0,
            'end_position': 100,
            'character_count': 100,
            'position_type': 'character_positions'
        }
        
        assert metadata_manager._validate_section_position(section_position) is True
    
    def test_validate_section_position_invalid(self, metadata_manager):
        """Test section position validation with invalid data."""
        # Missing fields
        section_position = {'start_position': 0}
        assert metadata_manager._validate_section_position(section_position) is False
        
        # Invalid positions
        section_position = {
            'start_position': -1,
            'end_position': 100,
            'character_count': 101,
            'position_type': 'character_positions'
        }
        assert metadata_manager._validate_section_position(section_position) is False
        
        # Inconsistent character count
        section_position = {
            'start_position': 0,
            'end_position': 100,
            'character_count': 50,
            'position_type': 'character_positions'
        }
        assert metadata_manager._validate_section_position(section_position) is False
    
    def test_validate_document_metadata_valid(self, metadata_manager):
        """Test document metadata validation with valid data."""
        document_metadata = {
            'source_path': '/path/to/document.txt',
            'creation_timestamp': '2024-01-01T12:00:00',
            'chunk_sequence_number': 0,
            'processing_timestamp': '2024-01-01T12:01:00'
        }
        
        assert metadata_manager._validate_document_metadata(document_metadata) is True
    
    def test_validate_document_metadata_invalid(self, metadata_manager):
        """Test document metadata validation with invalid data."""
        # Missing fields
        document_metadata = {'source_path': '/path/to/document.txt'}
        assert metadata_manager._validate_document_metadata(document_metadata) is False
        
        # Invalid timestamp
        document_metadata = {
            'source_path': '/path/to/document.txt',
            'creation_timestamp': 'invalid-timestamp',
            'chunk_sequence_number': 0,
            'processing_timestamp': '2024-01-01T12:01:00'
        }
        assert metadata_manager._validate_document_metadata(document_metadata) is False
        
        # Invalid sequence number
        document_metadata = {
            'source_path': '/path/to/document.txt',
            'creation_timestamp': '2024-01-01T12:00:00',
            'chunk_sequence_number': -1,
            'processing_timestamp': '2024-01-01T12:01:00'
        }
        assert metadata_manager._validate_document_metadata(document_metadata) is False
    
    def test_validate_ipfs_metadata_valid(self, metadata_manager):
        """Test IPFS metadata validation with valid data."""
        ipfs_metadata = {
            'ipfs_hash': 'QmTestHash123456789012345678901234',
            'hash_validated': True,
            'retrieval_available': True,
            'hash_generation_method': 'sha256_base64_truncated'
        }
        
        assert metadata_manager._validate_ipfs_metadata(ipfs_metadata) is True
    
    def test_validate_ipfs_metadata_invalid(self, metadata_manager):
        """Test IPFS metadata validation with invalid data."""
        # Missing fields
        ipfs_metadata = {'ipfs_hash': 'QmTestHash'}
        assert metadata_manager._validate_ipfs_metadata(ipfs_metadata) is False
        
        # Empty hash
        ipfs_metadata = {
            'ipfs_hash': '',
            'hash_validated': True,
            'retrieval_available': True,
            'hash_generation_method': 'sha256_base64_truncated'
        }
        assert metadata_manager._validate_ipfs_metadata(ipfs_metadata) is False
        
        # Invalid boolean values
        ipfs_metadata = {
            'ipfs_hash': 'QmTestHash',
            'hash_validated': 'not_boolean',
            'retrieval_available': True,
            'hash_generation_method': 'sha256_base64_truncated'
        }
        assert metadata_manager._validate_ipfs_metadata(ipfs_metadata) is False
    
    def test_validate_metadata_consistency_valid(self, metadata_manager, sample_chunk):
        """Test metadata consistency validation with consistent data."""
        with patch.object(metadata_manager.ipfs_manager, 'retrieve_document', return_value="Test document content"):
            metadata = metadata_manager.create_chunk_metadata(sample_chunk)
            assert metadata_manager._validate_metadata_consistency(metadata) is True
    
    def test_validate_metadata_consistency_invalid(self, metadata_manager, sample_chunk):
        """Test metadata consistency validation with inconsistent data."""
        with patch.object(metadata_manager.ipfs_manager, 'retrieve_document', return_value="Test document content"):
            metadata = metadata_manager.create_chunk_metadata(sample_chunk)
            
            # Make chunk_size inconsistent with content_length
            metadata['chunk_size'] = 999
            assert metadata_manager._validate_metadata_consistency(metadata) is False
            
            # Reset and test section position inconsistency
            metadata['chunk_size'] = metadata['content_length']
            metadata['section_position']['character_count'] = 999
            assert metadata_manager._validate_metadata_consistency(metadata) is False
            
            # Reset and test sequence number inconsistency
            metadata['section_position']['character_count'] = 75
            metadata['document_metadata']['chunk_sequence_number'] = 999
            assert metadata_manager._validate_metadata_consistency(metadata) is False


if __name__ == "__main__":
    pytest.main([__file__])