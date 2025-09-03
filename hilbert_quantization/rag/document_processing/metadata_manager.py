"""
Document metadata management for comprehensive tracking.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from ..models import DocumentChunk
from .ipfs_integration import IPFSManager


class DocumentMetadataManager:
    """Manager for document metadata tracking and validation."""
    
    def __init__(self, config):
        """Initialize metadata manager with configuration."""
        self.config = config
        self.ipfs_manager = IPFSManager(config)
        self._metadata_cache: Dict[str, Dict[str, Any]] = {}
    
    def create_chunk_metadata(self, chunk: DocumentChunk) -> Dict[str, Any]:
        """
        Create comprehensive metadata for document chunk.
        
        Implements requirements 11.4, 11.5, and 11.6 by creating complete
        metadata including section positions, timestamps, and IPFS hash validation.
        
        Args:
            chunk: DocumentChunk to create metadata for
            
        Returns:
            Dictionary containing comprehensive chunk metadata
        """
        if not isinstance(chunk, DocumentChunk):
            raise ValueError("Input must be a DocumentChunk instance")
        
        # Create comprehensive metadata dictionary
        metadata = {
            # Basic chunk information
            'chunk_id': self._generate_chunk_id(chunk),
            'content_length': len(chunk.content),
            'chunk_size': chunk.chunk_size,
            'chunk_sequence': chunk.chunk_sequence,
            
            # Section position tracking (Requirement 11.4)
            'section_position': {
                'start_position': chunk.start_position,
                'end_position': chunk.end_position,
                'character_count': chunk.end_position - chunk.start_position,
                'position_type': 'character_positions'  # Could be extended to page numbers
            },
            
            # Document source and timestamp metadata (Requirement 11.5)
            'document_metadata': {
                'source_path': chunk.source_path,
                'creation_timestamp': chunk.creation_timestamp,
                'chunk_sequence_number': chunk.chunk_sequence,
                'processing_timestamp': datetime.now().isoformat()
            },
            
            # IPFS hash and retrieval information (Requirement 11.6)
            'ipfs_metadata': {
                'ipfs_hash': chunk.ipfs_hash,
                'hash_validated': self._validate_ipfs_hash(chunk),
                'retrieval_available': self._check_retrieval_availability(chunk.ipfs_hash),
                'hash_generation_method': 'sha256_base64_truncated'
            },
            
            # Additional tracking metadata
            'validation_status': {
                'size_consistent': chunk.validate_size(chunk.chunk_size),
                'positions_valid': self._validate_positions(chunk),
                'metadata_complete': self._check_metadata_completeness(chunk),
                'ipfs_hash_valid': self._validate_ipfs_hash(chunk)
            },
            
            # Processing context
            'processing_context': {
                'chunker_version': '1.0.0',
                'config_hash': self._get_config_hash(),
                'validation_timestamp': datetime.now().isoformat()
            }
        }
        
        # Cache metadata for future reference
        chunk_id = metadata['chunk_id']
        self._metadata_cache[chunk_id] = metadata
        
        return metadata
    
    def validate_metadata(self, metadata: Dict[str, Any]) -> bool:
        """
        Validate document metadata completeness and consistency.
        
        Ensures all required metadata fields are present and valid according
        to requirements 11.4, 11.5, and 11.6.
        
        Args:
            metadata: Metadata dictionary to validate
            
        Returns:
            True if metadata is complete and valid, False otherwise
        """
        if not isinstance(metadata, dict):
            return False
        
        # Check required top-level fields
        required_fields = [
            'chunk_id', 'content_length', 'chunk_size', 'chunk_sequence',
            'section_position', 'document_metadata', 'ipfs_metadata',
            'validation_status', 'processing_context'
        ]
        
        for field in required_fields:
            if field not in metadata:
                return False
        
        # Validate section position metadata (Requirement 11.4)
        if not self._validate_section_position(metadata.get('section_position', {})):
            return False
        
        # Validate document metadata (Requirement 11.5)
        if not self._validate_document_metadata(metadata.get('document_metadata', {})):
            return False
        
        # Validate IPFS metadata (Requirement 11.6)
        if not self._validate_ipfs_metadata(metadata.get('ipfs_metadata', {})):
            return False
        
        # Validate consistency across fields
        if not self._validate_metadata_consistency(metadata):
            return False
        
        return True
    
    def retrieve_original_document(self, ipfs_hash: str) -> str:
        """
        Retrieve the full original document using IPFS hash.
        
        Implements requirement 11.6 by enabling retrieval of the full original
        document for context or verification purposes.
        
        Args:
            ipfs_hash: IPFS hash of the original document
            
        Returns:
            Full original document content
            
        Raises:
            ValueError: If hash is invalid or document not found
        """
        if not ipfs_hash:
            raise ValueError("IPFS hash cannot be empty")
        
        try:
            return self.ipfs_manager.retrieve_document(ipfs_hash)
        except Exception as e:
            raise ValueError(f"Failed to retrieve document with hash {ipfs_hash}: {str(e)}")
    
    def validate_chunk_against_original(self, chunk: DocumentChunk) -> Dict[str, Any]:
        """
        Validate chunk content against original document using IPFS hash.
        
        Args:
            chunk: DocumentChunk to validate
            
        Returns:
            Dictionary containing validation results
        """
        validation_result = {
            'chunk_valid': False,
            'original_retrieved': False,
            'position_valid': False,
            'content_matches': False,
            'error_message': None
        }
        
        try:
            # Retrieve original document
            original_content = self.retrieve_original_document(chunk.ipfs_hash)
            validation_result['original_retrieved'] = True
            
            # Validate positions are within document bounds
            if (chunk.start_position >= 0 and 
                chunk.end_position <= len(original_content) and
                chunk.start_position < chunk.end_position):
                validation_result['position_valid'] = True
                
                # Extract expected content from original document
                expected_content = original_content[chunk.start_position:chunk.end_position]
                
                # Compare with chunk content (removing padding)
                chunk_content_unpadded = chunk.content.rstrip()
                if expected_content == chunk_content_unpadded:
                    validation_result['content_matches'] = True
                    validation_result['chunk_valid'] = True
                else:
                    validation_result['error_message'] = "Chunk content does not match original document section"
            else:
                validation_result['error_message'] = "Chunk positions are invalid for original document"
                
        except Exception as e:
            validation_result['error_message'] = f"Validation failed: {str(e)}"
        
        return validation_result
    
    def get_chunk_context(self, chunk: DocumentChunk, context_chars: int = 200) -> Dict[str, str]:
        """
        Get surrounding context from original document for a chunk.
        
        Args:
            chunk: DocumentChunk to get context for
            context_chars: Number of characters to include before and after
            
        Returns:
            Dictionary containing context information
        """
        try:
            original_content = self.retrieve_original_document(chunk.ipfs_hash)
            
            # Calculate context boundaries
            context_start = max(0, chunk.start_position - context_chars)
            context_end = min(len(original_content), chunk.end_position + context_chars)
            
            return {
                'before_context': original_content[context_start:chunk.start_position],
                'chunk_content': original_content[chunk.start_position:chunk.end_position],
                'after_context': original_content[chunk.end_position:context_end],
                'full_context': original_content[context_start:context_end]
            }
        except Exception as e:
            return {
                'error': f"Failed to retrieve context: {str(e)}",
                'before_context': '',
                'chunk_content': '',
                'after_context': '',
                'full_context': ''
            }
    
    def track_processing_progress(self, document_path: str, chunks_created: int) -> None:
        """Track document processing progress."""
        # Implementation will be added in task 9.1
        raise NotImplementedError("Will be implemented in task 9.1")
    
    def _generate_chunk_id(self, chunk: DocumentChunk) -> str:
        """Generate unique identifier for chunk."""
        import hashlib
        
        # Create unique ID based on source path, positions, and sequence
        id_string = f"{chunk.source_path}:{chunk.start_position}:{chunk.end_position}:{chunk.chunk_sequence}"
        return hashlib.md5(id_string.encode()).hexdigest()[:16]
    
    def _validate_ipfs_hash(self, chunk: DocumentChunk) -> bool:
        """Validate IPFS hash for chunk."""
        try:
            # Try to retrieve document to validate hash
            self.ipfs_manager.retrieve_document(chunk.ipfs_hash)
            return True
        except Exception:
            return False
    
    def _check_retrieval_availability(self, ipfs_hash: str) -> bool:
        """Check if document can be retrieved using IPFS hash."""
        try:
            self.ipfs_manager.retrieve_document(ipfs_hash)
            return True
        except Exception:
            return False
    
    def _validate_positions(self, chunk: DocumentChunk) -> bool:
        """Validate chunk position metadata."""
        return (chunk.start_position >= 0 and 
                chunk.end_position > chunk.start_position and
                chunk.chunk_sequence >= 0)
    
    def _check_metadata_completeness(self, chunk: DocumentChunk) -> bool:
        """Check if chunk has complete metadata."""
        return (bool(chunk.ipfs_hash) and 
                bool(chunk.source_path) and
                bool(chunk.creation_timestamp) and
                chunk.chunk_size > 0)
    
    def _get_config_hash(self) -> str:
        """Generate hash of current configuration for tracking."""
        import hashlib
        
        try:
            # Try to extract basic config values safely
            config_values = []
            
            # Extract chunking config if available
            if hasattr(self.config, 'chunking'):
                chunking = self.config.chunking
                if hasattr(chunking, 'chunk_size'):
                    config_values.append(f"chunk_size:{getattr(chunking, 'chunk_size', 'unknown')}")
                if hasattr(chunking, 'min_chunk_size'):
                    config_values.append(f"min_chunk_size:{getattr(chunking, 'min_chunk_size', 'unknown')}")
                if hasattr(chunking, 'max_chunk_size'):
                    config_values.append(f"max_chunk_size:{getattr(chunking, 'max_chunk_size', 'unknown')}")
            
            # Extract storage config if available
            if hasattr(self.config, 'storage'):
                storage = self.config.storage
                if hasattr(storage, 'base_storage_path'):
                    config_values.append(f"storage_path:{getattr(storage, 'base_storage_path', 'unknown')}")
            
            # Create a simple string representation
            config_string = "|".join(sorted(config_values)) if config_values else "default_config"
            
        except Exception:
            # Fallback to a simple default if config extraction fails
            config_string = "default_config"
        
        return hashlib.md5(config_string.encode()).hexdigest()[:8]
    
    def _validate_section_position(self, section_position: Dict[str, Any]) -> bool:
        """Validate section position metadata (Requirement 11.4)."""
        required_fields = ['start_position', 'end_position', 'character_count', 'position_type']
        
        for field in required_fields:
            if field not in section_position:
                return False
        
        # Validate position values
        start_pos = section_position.get('start_position', -1)
        end_pos = section_position.get('end_position', -1)
        char_count = section_position.get('character_count', -1)
        
        return (start_pos >= 0 and 
                end_pos > start_pos and 
                char_count == (end_pos - start_pos) and
                char_count > 0)
    
    def _validate_document_metadata(self, document_metadata: Dict[str, Any]) -> bool:
        """Validate document metadata (Requirement 11.5)."""
        required_fields = ['source_path', 'creation_timestamp', 'chunk_sequence_number', 'processing_timestamp']
        
        for field in required_fields:
            if field not in document_metadata:
                return False
        
        # Validate timestamp formats
        try:
            datetime.fromisoformat(document_metadata['creation_timestamp'])
            datetime.fromisoformat(document_metadata['processing_timestamp'])
        except (ValueError, TypeError):
            return False
        
        # Validate other fields
        return (bool(document_metadata['source_path']) and
                document_metadata['chunk_sequence_number'] >= 0)
    
    def _validate_ipfs_metadata(self, ipfs_metadata: Dict[str, Any]) -> bool:
        """Validate IPFS metadata (Requirement 11.6)."""
        required_fields = ['ipfs_hash', 'hash_validated', 'retrieval_available', 'hash_generation_method']
        
        for field in required_fields:
            if field not in ipfs_metadata:
                return False
        
        return (bool(ipfs_metadata['ipfs_hash']) and
                isinstance(ipfs_metadata['hash_validated'], bool) and
                isinstance(ipfs_metadata['retrieval_available'], bool) and
                bool(ipfs_metadata['hash_generation_method']))
    
    def _validate_metadata_consistency(self, metadata: Dict[str, Any]) -> bool:
        """Validate consistency across metadata fields."""
        # Check that chunk_size matches content_length
        if metadata['chunk_size'] != metadata['content_length']:
            return False
        
        # Check that section position character count is consistent
        section_pos = metadata.get('section_position', {})
        expected_char_count = section_pos.get('end_position', 0) - section_pos.get('start_position', 0)
        if section_pos.get('character_count', -1) != expected_char_count:
            return False
        
        # Check that chunk sequence numbers match
        doc_metadata = metadata.get('document_metadata', {})
        if doc_metadata.get('chunk_sequence_number', -1) != metadata.get('chunk_sequence', -2):
            return False
        
        return True