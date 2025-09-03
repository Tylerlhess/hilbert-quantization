"""
Document chunker implementation with IPFS integration.
"""

import math
import re
from datetime import datetime
from typing import List
from ..interfaces import DocumentChunker
from ..models import DocumentChunk
from .ipfs_integration import IPFSManager


class DocumentChunkerImpl(DocumentChunker):
    """Implementation of document chunking with standardized sizes and IPFS metadata."""
    
    def __init__(self, config):
        """Initialize document chunker with configuration."""
        self.config = config
        self.ipfs_manager = IPFSManager(config)
    
    def chunk_document(self, document: str, ipfs_hash: str, source_path: str) -> List[DocumentChunk]:
        """
        Create standardized fixed-size chunks with comprehensive metadata.
        
        Args:
            document: Document content to chunk
            ipfs_hash: IPFS hash of the source document
            source_path: Path to the source document
            
        Returns:
            List of DocumentChunk objects with metadata
        """
        if not document.strip():
            return []
        
        # Calculate chunk size based on embedding dimensions if not configured
        if self.config.chunking.chunk_size is None:
            # Use default embedding dimensions for calculation
            embedding_dims = 384  # Default for all-MiniLM-L6-v2
            chunk_size = self.calculate_chunk_size(embedding_dims)
        else:
            chunk_size = self.config.chunking.chunk_size
        
        # Ensure chunk size is within configured bounds
        chunk_size = max(self.config.chunking.min_chunk_size, 
                        min(chunk_size, self.config.chunking.max_chunk_size))
        
        chunks = []
        overlap = self.config.chunking.chunk_overlap
        current_position = 0
        chunk_sequence = 0
        creation_timestamp = datetime.now().isoformat()
        
        while current_position < len(document):
            # Calculate end position for this chunk (before padding)
            raw_end_position = min(current_position + chunk_size, len(document))
            
            # Extract raw chunk content
            raw_chunk_content = document[current_position:raw_end_position]
            
            # Preserve sentence boundaries if configured
            actual_end_position = raw_end_position
            if (self.config.chunking.preserve_sentence_boundaries and 
                raw_end_position < len(document) and 
                len(raw_chunk_content) >= self.config.chunking.min_chunk_size):
                
                # Find the last sentence boundary within the chunk
                sentence_endings = ['.', '!', '?']
                last_boundary = -1
                
                # Look for sentence endings, but ensure we have enough content
                for i in range(len(raw_chunk_content) - 1, self.config.chunking.min_chunk_size - 1, -1):
                    if i < len(raw_chunk_content) and raw_chunk_content[i] in sentence_endings:
                        # Check if there's a space or end after the punctuation
                        if i == len(raw_chunk_content) - 1 or raw_chunk_content[i + 1] == ' ':
                            last_boundary = i + 1
                            break
                
                if last_boundary > 0 and last_boundary >= self.config.chunking.min_chunk_size:
                    raw_chunk_content = raw_chunk_content[:last_boundary]
                    actual_end_position = current_position + last_boundary
            
            # Pad chunk to exact target size for consistency
            padded_content = self.pad_chunk(raw_chunk_content, chunk_size)
            
            # Create DocumentChunk with comprehensive metadata
            chunk = DocumentChunk(
                content=padded_content,
                ipfs_hash=ipfs_hash,
                source_path=source_path,
                start_position=current_position,
                end_position=actual_end_position,
                chunk_sequence=chunk_sequence,
                creation_timestamp=creation_timestamp,
                chunk_size=len(padded_content)
            )
            
            chunks.append(chunk)
            
            # Move to next chunk position with overlap
            next_position = actual_end_position - overlap
            
            # Ensure we make progress
            if next_position <= current_position:
                next_position = current_position + 1
            
            current_position = next_position
            chunk_sequence += 1
            
            # Break if we've processed the entire document
            if actual_end_position >= len(document):
                break
        
        return chunks
    
    def calculate_chunk_size(self, embedding_dimensions: int) -> int:
        """
        Calculate optimal chunk size based on Hilbert curve dimensions (powers of 4).
        
        This method ensures that document chunks correspond to Hilbert curve dimensions
        to form neat square representations for efficient spatial mapping.
        
        Args:
            embedding_dimensions: Dimensions of the embedding model
            
        Returns:
            Optimal chunk size for consistent processing
        """
        if embedding_dimensions <= 0:
            raise ValueError("Embedding dimensions must be positive")
        
        # Find the nearest power of 4 that accommodates the embedding dimensions
        # This ensures efficient Hilbert curve mapping with square representations
        power_of_2 = 1
        while power_of_2 * power_of_2 < embedding_dimensions:
            power_of_2 *= 2
        
        # The Hilbert curve dimensions will be power_of_2 x power_of_2
        hilbert_area = power_of_2 * power_of_2
        
        # Calculate optimal chunk size based on Hilbert curve area
        # Use empirically determined scaling factors for different embedding sizes
        if embedding_dimensions <= 384:  # Small embeddings (e.g., all-MiniLM-L6-v2)
            chars_per_dimension = 4
        elif embedding_dimensions <= 768:  # Medium embeddings (e.g., BERT-base)
            chars_per_dimension = 5
        elif embedding_dimensions <= 1536:  # Large embeddings (e.g., text-embedding-ada-002)
            chars_per_dimension = 6
        else:  # Very large embeddings
            chars_per_dimension = 7
        
        # Calculate base chunk size from Hilbert curve area
        base_chunk_size = hilbert_area * chars_per_dimension
        
        # Apply power-of-4 alignment for optimal Hilbert curve processing
        # Round to nearest power of 4 boundary for consistency
        aligned_size = self._align_to_power_of_4_boundary(base_chunk_size)
        
        # Ensure the size is within configured bounds
        min_size = self.config.chunking.min_chunk_size
        max_size = self.config.chunking.max_chunk_size
        
        optimal_size = max(min_size, min(aligned_size, max_size))
        
        # Log the calculation for debugging
        if hasattr(self.config, 'debug') and self.config.debug:
            print(f"Embedding dims: {embedding_dimensions}, Hilbert dims: {power_of_2}x{power_of_2}, "
                  f"Base size: {base_chunk_size}, Aligned: {aligned_size}, Final: {optimal_size}")
        
        return optimal_size
    
    def _align_to_power_of_4_boundary(self, size: int) -> int:
        """
        Align chunk size to power-of-4 boundary for optimal Hilbert curve processing.
        
        Args:
            size: Base chunk size to align
            
        Returns:
            Size aligned to nearest power-of-4 boundary
        """
        if size <= 0:
            return 4  # Minimum power of 4
        
        # Find the nearest power of 4
        power = 1
        while power < size:
            power *= 4
        
        # Choose between current power and previous power based on proximity
        prev_power = power // 4
        if prev_power > 0 and (size - prev_power) < (power - size):
            return prev_power
        else:
            return power
    
    def pad_chunk(self, chunk: str, target_size: int) -> str:
        """
        Pad chunk to exact target size for consistency.
        
        Ensures all document chunks are exactly the same size through padding,
        which is essential for consistent Hilbert curve mapping and video frame generation.
        
        Args:
            chunk: Original chunk content
            target_size: Target size for padding
            
        Returns:
            Padded chunk content of exactly target_size length
            
        Raises:
            ValueError: If target_size is not positive
        """
        if target_size <= 0:
            raise ValueError("Target size must be positive")
        
        if len(chunk) >= target_size:
            # Truncate to exact target size
            return chunk[:target_size]
        
        # Calculate padding needed
        padding_needed = target_size - len(chunk)
        padding_char = self.config.chunking.padding_char
        
        # Validate padding character
        if not padding_char or len(padding_char) != 1:
            padding_char = " "  # Default to space
        
        # Add padding at the end to reach exact target size
        padded_chunk = chunk + (padding_char * padding_needed)
        
        # Verify exact size (safety check)
        if len(padded_chunk) != target_size:
            raise RuntimeError(f"Padding failed: expected {target_size}, got {len(padded_chunk)}")
        
        return padded_chunk
    
    def validate_chunk_consistency(self, chunks: List[DocumentChunk]) -> bool:
        """
        Validate that all chunks have consistent sizes across the document collection.
        
        This validation ensures that all document chunks meet the requirements for
        consistent Hilbert curve mapping and video frame generation.
        
        Args:
            chunks: List of document chunks to validate
            
        Returns:
            True if all chunks are consistent, False otherwise
        """
        if not chunks:
            return True
        
        # Get expected size from first chunk
        expected_size = chunks[0].chunk_size
        
        # Validate that expected size is reasonable
        if expected_size <= 0:
            return False
        
        # Check consistency across all chunks
        for i, chunk in enumerate(chunks):
            # Validate chunk size consistency
            if chunk.chunk_size != expected_size:
                if hasattr(self.config, 'debug') and self.config.debug:
                    print(f"Chunk {i}: size mismatch - expected {expected_size}, got {chunk.chunk_size}")
                return False
            
            # Validate chunk content length matches reported size (critical for exact sizing)
            if len(chunk.content) != chunk.chunk_size:
                if hasattr(self.config, 'debug') and self.config.debug:
                    print(f"Chunk {i}: content length mismatch - reported {chunk.chunk_size}, actual {len(chunk.content)}")
                return False
            
            # Validate metadata completeness
            if not chunk.ipfs_hash or not chunk.source_path:
                if hasattr(self.config, 'debug') and self.config.debug:
                    print(f"Chunk {i}: missing metadata - ipfs_hash: {bool(chunk.ipfs_hash)}, source_path: {bool(chunk.source_path)}")
                return False
            
            # Validate position consistency
            if chunk.start_position < 0 or chunk.end_position <= chunk.start_position:
                if hasattr(self.config, 'debug') and self.config.debug:
                    print(f"Chunk {i}: invalid positions - start: {chunk.start_position}, end: {chunk.end_position}")
                return False
            
            # Validate chunk sequence ordering
            if chunk.chunk_sequence != i:
                if hasattr(self.config, 'debug') and self.config.debug:
                    print(f"Chunk {i}: sequence mismatch - expected {i}, got {chunk.chunk_sequence}")
                return False
            
            # Validate timestamp format
            try:
                from datetime import datetime
                datetime.fromisoformat(chunk.creation_timestamp)
            except (ValueError, TypeError):
                if hasattr(self.config, 'debug') and self.config.debug:
                    print(f"Chunk {i}: invalid timestamp format - {chunk.creation_timestamp}")
                return False
        
        return True
    
    def validate_chunk_size_across_collection(self, chunk_collections: List[List[DocumentChunk]]) -> bool:
        """
        Validate chunk size consistency across multiple document collections.
        
        This method ensures that chunks from different documents all have the same size,
        which is essential for consistent processing in the RAG system.
        
        Args:
            chunk_collections: List of chunk lists from different documents
            
        Returns:
            True if all chunks across all collections have consistent sizes
        """
        if not chunk_collections:
            return True
        
        # Find the first non-empty collection to get expected size
        expected_size = None
        for collection in chunk_collections:
            if collection:
                expected_size = collection[0].chunk_size
                break
        
        if expected_size is None:
            return True  # All collections are empty
        
        # Validate each collection
        for i, collection in enumerate(chunk_collections):
            if not self.validate_chunk_consistency(collection):
                if hasattr(self.config, 'debug') and self.config.debug:
                    print(f"Collection {i}: internal consistency failed")
                return False
            
            # Check size consistency across collections
            for chunk in collection:
                if chunk.chunk_size != expected_size:
                    if hasattr(self.config, 'debug') and self.config.debug:
                        print(f"Collection {i}: size mismatch - expected {expected_size}, got {chunk.chunk_size}")
                    return False
        
        return True
    
    def get_chunk_size_statistics(self, chunks: List[DocumentChunk]) -> dict:
        """
        Get statistics about chunk sizes for analysis and debugging.
        
        Args:
            chunks: List of document chunks to analyze
            
        Returns:
            Dictionary containing chunk size statistics
        """
        if not chunks:
            return {
                'total_chunks': 0,
                'unique_sizes': 0,
                'size_distribution': {},
                'is_consistent': True,
                'expected_size': None
            }
        
        sizes = [chunk.chunk_size for chunk in chunks]
        content_lengths = [len(chunk.content) for chunk in chunks]
        
        from collections import Counter
        size_distribution = Counter(sizes)
        content_length_distribution = Counter(content_lengths)
        
        return {
            'total_chunks': len(chunks),
            'unique_sizes': len(size_distribution),
            'size_distribution': dict(size_distribution),
            'content_length_distribution': dict(content_length_distribution),
            'is_consistent': len(size_distribution) == 1 and len(content_length_distribution) == 1,
            'expected_size': chunks[0].chunk_size,
            'min_size': min(sizes),
            'max_size': max(sizes),
            'size_matches_content': all(chunk.chunk_size == len(chunk.content) for chunk in chunks)
        }