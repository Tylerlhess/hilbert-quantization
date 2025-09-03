"""
Document retrieval system for RAG with frame-based access.
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from ..interfaces import DocumentRetrieval
from ..models import DocumentChunk, DocumentSearchResult, VideoFrameMetadata
from ..video_storage.dual_storage import DualVideoStorageImpl


class DocumentRetrievalImpl(DocumentRetrieval):
    """Implementation of frame-based document retrieval system."""
    
    def __init__(self, dual_storage: DualVideoStorageImpl, config):
        """Initialize document retrieval with dual storage system."""
        self.dual_storage = dual_storage
        self.config = config
        
        # Configuration parameters
        self.max_results = getattr(config, 'max_search_results', 10)
        self.similarity_threshold = getattr(config, 'similarity_threshold', 0.1)
        self.enable_validation = getattr(config, 'enable_frame_validation', True)
    
    def retrieve_documents_by_frame_numbers(self, frame_numbers: List[int]) -> List[Tuple[int, DocumentChunk]]:
        """
        Retrieve document chunks using frame numbers from similarity search.
        
        This implements requirement 4.7 by using frame numbers to retrieve
        associated document chunks from the document video.
        
        Args:
            frame_numbers: List of frame numbers from embedding similarity search
            
        Returns:
            List of (frame_number, document_chunk) tuples
        """
        if not frame_numbers:
            return []
        
        # Validate frame synchronization if enabled
        if self.enable_validation:
            validation_result = self.dual_storage.validate_frame_synchronization(frame_numbers)
            if not validation_result['validation_passed']:
                print(f"Warning: Frame synchronization validation failed for {len(validation_result['synchronization_errors'])} frames")
        
        # Retrieve document chunks
        return self.dual_storage.get_document_chunks_by_frame_numbers(frame_numbers)
    
    def retrieve_single_document(self, frame_number: int) -> Optional[DocumentChunk]:
        """
        Retrieve a single document chunk by frame number.
        
        Args:
            frame_number: Frame number from embedding similarity search
            
        Returns:
            Document chunk if found, None otherwise
        """
        try:
            return self.dual_storage.get_document_chunk(frame_number)
        except ValueError:
            return None
    
    def retrieve_documents_with_metadata(self, frame_numbers: List[int]) -> List[Tuple[int, DocumentChunk, VideoFrameMetadata]]:
        """
        Retrieve document chunks with their complete metadata.
        
        Args:
            frame_numbers: List of frame numbers from similarity search
            
        Returns:
            List of (frame_number, document_chunk, metadata) tuples
        """
        results = []
        
        for frame_number in frame_numbers:
            # Get frame metadata
            frame_metadata = self.dual_storage._get_frame_metadata_by_number(frame_number)
            if frame_metadata is None:
                continue
            
            # Extract document chunk
            document_chunk = frame_metadata.chunk_metadata
            
            results.append((frame_number, document_chunk, frame_metadata))
        
        return results
    
    def validate_retrieval_synchronization(self, frame_numbers: List[int]) -> Dict[str, Any]:
        """
        Validate embedding-document frame synchronization for retrieval.
        
        This implements requirement 7.5 by validating synchronized access
        to document video using embedding search results.
        
        Args:
            frame_numbers: Frame numbers to validate
            
        Returns:
            Validation results dictionary
        """
        return self.dual_storage.validate_frame_synchronization(frame_numbers)
    
    def get_retrieval_statistics(self, frame_numbers: List[int]) -> Dict[str, Any]:
        """
        Get statistics about document retrieval operation.
        
        Args:
            frame_numbers: Frame numbers used for retrieval
            
        Returns:
            Statistics dictionary
        """
        # Retrieve documents with metadata
        retrieved_docs = self.retrieve_documents_with_metadata(frame_numbers)
        
        if not retrieved_docs:
            return {
                'total_requested': len(frame_numbers),
                'total_retrieved': 0,
                'retrieval_success_rate': 0.0,
                'unique_documents': 0,
                'unique_sources': 0,
                'average_chunk_size': 0,
                'chunk_size_range': (0, 0),
                'embedding_models': [],
                'ipfs_hashes': []
            }
        
        # Extract statistics
        document_chunks = [doc for _, doc, _ in retrieved_docs]
        metadata_list = [meta for _, _, meta in retrieved_docs]
        
        chunk_sizes = [chunk.chunk_size for chunk in document_chunks]
        unique_documents = set(chunk.ipfs_hash for chunk in document_chunks)
        unique_sources = set(chunk.source_path for chunk in document_chunks)
        embedding_models = list(set(meta.embedding_model for meta in metadata_list))
        ipfs_hashes = list(set(chunk.ipfs_hash for chunk in document_chunks))
        
        return {
            'total_requested': len(frame_numbers),
            'total_retrieved': len(retrieved_docs),
            'retrieval_success_rate': len(retrieved_docs) / len(frame_numbers) if frame_numbers else 0.0,
            'unique_documents': len(unique_documents),
            'unique_sources': len(unique_sources),
            'average_chunk_size': sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0,
            'chunk_size_range': (min(chunk_sizes), max(chunk_sizes)) if chunk_sizes else (0, 0),
            'embedding_models': embedding_models,
            'ipfs_hashes': ipfs_hashes
        }
    
    def retrieve_documents_by_similarity_results(self, similarity_results: List[Tuple[int, float]]) -> List[Tuple[int, DocumentChunk, float]]:
        """
        Retrieve documents using similarity search results.
        
        Args:
            similarity_results: List of (frame_number, similarity_score) tuples
            
        Returns:
            List of (frame_number, document_chunk, similarity_score) tuples
        """
        results = []
        
        for frame_number, similarity_score in similarity_results:
            document_chunk = self.retrieve_single_document(frame_number)
            if document_chunk is not None:
                results.append((frame_number, document_chunk, similarity_score))
        
        return results
    
    def retrieve_documents_with_context(self, frame_numbers: List[int], context_window: int = 2) -> List[Dict[str, Any]]:
        """
        Retrieve documents with surrounding context chunks.
        
        Args:
            frame_numbers: Primary frame numbers to retrieve
            context_window: Number of adjacent frames to include as context
            
        Returns:
            List of dictionaries containing primary document and context
        """
        results = []
        
        for frame_number in frame_numbers:
            # Get primary document
            primary_doc = self.retrieve_single_document(frame_number)
            if primary_doc is None:
                continue
            
            # Get context documents
            context_frames = []
            for offset in range(-context_window, context_window + 1):
                if offset == 0:
                    continue  # Skip the primary frame
                
                context_frame_number = frame_number + offset
                if context_frame_number >= 0:
                    context_doc = self.retrieve_single_document(context_frame_number)
                    if context_doc is not None:
                        context_frames.append({
                            'frame_number': context_frame_number,
                            'document_chunk': context_doc,
                            'offset': offset
                        })
            
            result = {
                'primary_frame_number': frame_number,
                'primary_document': primary_doc,
                'context_documents': context_frames,
                'total_context_frames': len(context_frames)
            }
            
            results.append(result)
        
        return results
    
    def get_document_by_ipfs_hash(self, ipfs_hash: str) -> List[Tuple[int, DocumentChunk]]:
        """
        Retrieve all chunks for a document by IPFS hash.
        
        Args:
            ipfs_hash: IPFS hash of the document
            
        Returns:
            List of (frame_number, document_chunk) tuples for all chunks of the document
        """
        results = []
        
        # Search through all frame metadata for matching IPFS hash
        for metadata in self.dual_storage.frame_metadata:
            if metadata.ipfs_hash == ipfs_hash:
                results.append((metadata.frame_index, metadata.chunk_metadata))
        
        # Sort by chunk sequence to maintain document order
        results.sort(key=lambda x: x[1].chunk_sequence)
        
        return results
    
    def reconstruct_full_document(self, ipfs_hash: str) -> Optional[str]:
        """
        Reconstruct full document content from chunks.
        
        Args:
            ipfs_hash: IPFS hash of the document to reconstruct
            
        Returns:
            Full document content if all chunks are available, None otherwise
        """
        # Get all chunks for the document
        document_chunks = self.get_document_by_ipfs_hash(ipfs_hash)
        
        if not document_chunks:
            return None
        
        # Sort by chunk sequence
        document_chunks.sort(key=lambda x: x[1].chunk_sequence)
        
        # Reconstruct document content
        full_content = ""
        expected_sequence = 0
        
        for frame_number, chunk in document_chunks:
            if chunk.chunk_sequence != expected_sequence:
                # Missing chunk - cannot reconstruct complete document
                return None
            
            full_content += chunk.content
            expected_sequence += 1
        
        return full_content
    
    def get_retrieval_performance_metrics(self, frame_numbers: List[int]) -> Dict[str, Any]:
        """
        Get performance metrics for document retrieval operations.
        
        Args:
            frame_numbers: Frame numbers used for retrieval
            
        Returns:
            Performance metrics dictionary
        """
        import time
        
        start_time = time.time()
        
        # Perform retrieval
        retrieved_docs = self.retrieve_documents_by_frame_numbers(frame_numbers)
        
        retrieval_time = time.time() - start_time
        
        # Calculate metrics
        success_rate = len(retrieved_docs) / len(frame_numbers) if frame_numbers else 0.0
        avg_retrieval_time_per_frame = retrieval_time / len(frame_numbers) if frame_numbers else 0.0
        
        # Get validation metrics if enabled
        validation_metrics = {}
        if self.enable_validation and frame_numbers:
            validation_start = time.time()
            validation_result = self.validate_retrieval_synchronization(frame_numbers)
            validation_time = time.time() - validation_start
            
            validation_metrics = {
                'validation_time': validation_time,
                'validation_passed': validation_result['validation_passed'],
                'synchronized_frames': validation_result['synchronized_frames'],
                'synchronization_errors': len(validation_result['synchronization_errors'])
            }
        
        return {
            'total_frames_requested': len(frame_numbers),
            'total_frames_retrieved': len(retrieved_docs),
            'retrieval_success_rate': success_rate,
            'total_retrieval_time': retrieval_time,
            'average_time_per_frame': avg_retrieval_time_per_frame,
            'frames_per_second': len(frame_numbers) / retrieval_time if retrieval_time > 0 else 0,
            **validation_metrics
        }