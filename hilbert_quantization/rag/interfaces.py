"""
Core interfaces for the RAG system components.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional, Any
import numpy as np

from .models import (
    DocumentChunk, 
    EmbeddingFrame, 
    VideoFrameMetadata, 
    DocumentSearchResult,
    ProcessingProgress
)


class DocumentChunker(ABC):
    """Interface for document chunking with IPFS integration."""
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def calculate_chunk_size(self, embedding_dimensions: int) -> int:
        """
        Calculate chunk size based on Hilbert curve dimensions (powers of 4).
        
        Args:
            embedding_dimensions: Dimensions of the embedding model
            
        Returns:
            Optimal chunk size for consistent processing
        """
        pass
    
    @abstractmethod
    def pad_chunk(self, chunk: str, target_size: int) -> str:
        """
        Pad chunk to exact target size for consistency.
        
        Args:
            chunk: Original chunk content
            target_size: Target size for padding
            
        Returns:
            Padded chunk content
        """
        pass
    
    @abstractmethod
    def validate_chunk_consistency(self, chunks: List[DocumentChunk]) -> bool:
        """
        Validate that all chunks have consistent sizes.
        
        Args:
            chunks: List of document chunks to validate
            
        Returns:
            True if all chunks are consistent, False otherwise
        """
        pass
    
    @abstractmethod
    def validate_chunk_size_across_collection(self, chunk_collections: List[List[DocumentChunk]]) -> bool:
        """
        Validate chunk size consistency across multiple document collections.
        
        Args:
            chunk_collections: List of chunk lists from different documents
            
        Returns:
            True if all chunks across all collections have consistent sizes
        """
        pass
    
    @abstractmethod
    def get_chunk_size_statistics(self, chunks: List[DocumentChunk]) -> dict:
        """
        Get statistics about chunk sizes for analysis and debugging.
        
        Args:
            chunks: List of document chunks to analyze
            
        Returns:
            Dictionary containing chunk size statistics
        """
        pass


class EmbeddingGenerator(ABC):
    """Interface for embedding generation with configurable models."""
    
    @abstractmethod
    def generate_embeddings(self, chunks: List[DocumentChunk], model_name: str) -> List[np.ndarray]:
        """
        Generate embeddings for document chunks using specified embedding model.
        
        Args:
            chunks: List of document chunks
            model_name: Name of the embedding model to use
            
        Returns:
            List of embedding arrays
        """
        pass
    
    @abstractmethod
    def calculate_optimal_dimensions(self, embedding_size: int) -> Tuple[int, int]:
        """
        Calculate nearest power-of-4 dimensions that accommodate embeddings.
        
        Args:
            embedding_size: Size of the embedding vectors
            
        Returns:
            Tuple of optimal (width, height) dimensions
        """
        pass
    
    @abstractmethod
    def validate_embedding_consistency(self, embeddings: List[np.ndarray]) -> bool:
        """
        Ensure all embeddings have consistent dimensions.
        
        Args:
            embeddings: List of embedding arrays to validate
            
        Returns:
            True if embeddings are consistent, False otherwise
        """
        pass
    
    @abstractmethod
    def get_supported_models(self) -> List[str]:
        """
        Get list of supported embedding models.
        
        Returns:
            List of available embedding model names
        """
        pass


class MultiLevelHierarchicalIndexGenerator(ABC):
    """Interface for generating multi-level hierarchical indices."""
    
    @abstractmethod
    def generate_multi_level_indices(self, embedding_image: np.ndarray) -> np.ndarray:
        """
        Generate multiple index rows for different Hilbert curve orders.
        
        Args:
            embedding_image: 2D embedding representation
            
        Returns:
            Enhanced image with multiple hierarchical index rows
        """
        pass
    
    @abstractmethod
    def calculate_hilbert_order_averages(self, image: np.ndarray, order: int) -> np.ndarray:
        """
        Calculate spatial averages for specific Hilbert curve order.
        
        Args:
            image: 2D embedding representation
            order: Hilbert curve order level
            
        Returns:
            Array of spatial averages for the order
        """
        pass
    
    @abstractmethod
    def create_progressive_granularity_levels(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Create index rows for progressive granularity levels (32x32, 16x16, 8x8, etc.).
        
        Args:
            image: 2D embedding representation
            
        Returns:
            List of index arrays for different granularity levels
        """
        pass
    
    @abstractmethod
    def embed_multi_level_indices(self, image: np.ndarray, index_rows: List[np.ndarray]) -> np.ndarray:
        """
        Add multiple index rows to embedding image representation.
        
        Args:
            image: Original 2D embedding representation
            index_rows: List of hierarchical index arrays
            
        Returns:
            Enhanced image with embedded index rows
        """
        pass
    
    @abstractmethod
    def calculate_optimal_granularity(self, image_dimensions: Tuple[int, int]) -> List[int]:
        """
        Calculate optimal granularity levels based on image dimensions.
        
        Args:
            image_dimensions: Dimensions of the embedding image
            
        Returns:
            List of optimal granularity levels
        """
        pass


class DualVideoStorage(ABC):
    """Interface for synchronized dual-video storage system."""
    
    @abstractmethod
    def add_document_chunk(self, chunk: DocumentChunk, embedding_frame: np.ndarray) -> VideoFrameMetadata:
        """
        Add synchronized document chunk and embedding frame to respective videos.
        
        Args:
            chunk: Document chunk to store
            embedding_frame: Corresponding embedding frame
            
        Returns:
            Metadata for the stored video frames
        """
        pass
    
    @abstractmethod
    def get_document_chunk(self, frame_number: int) -> DocumentChunk:
        """
        Retrieve document chunk using frame number from embedding similarity search.
        
        Args:
            frame_number: Frame number from similarity search
            
        Returns:
            Corresponding document chunk
        """
        pass
    
    @abstractmethod
    def insert_synchronized_frames(self, chunk: DocumentChunk, embedding_frame: np.ndarray) -> VideoFrameMetadata:
        """
        Insert new synchronized frames maintaining hierarchical index ordering.
        
        Args:
            chunk: Document chunk to insert
            embedding_frame: Corresponding embedding frame
            
        Returns:
            Metadata for the inserted frames
        """
        pass
    
    @abstractmethod
    def find_optimal_insertion_point(self, embedding_frame: np.ndarray) -> int:
        """
        Find the optimal frame position for insertion based on hierarchical index similarity.
        
        Args:
            embedding_frame: Embedding frame to insert
            
        Returns:
            Optimal frame position for insertion
        """
        pass
    
    @abstractmethod
    def reindex_frames_after_insertion(self, insertion_point: int) -> None:
        """
        Update frame indices and metadata after inserting new frames in the middle of videos.
        
        Args:
            insertion_point: Frame position where insertion occurred
        """
        pass
    
    @abstractmethod
    def get_video_metadata(self) -> Dict[str, Any]:
        """
        Get comprehensive metadata for both video files.
        
        Returns:
            Dictionary containing video metadata
        """
        pass


class RAGSearchEngine(ABC):
    """Interface for RAG similarity search with progressive filtering."""
    
    @abstractmethod
    def search_similar_documents(self, query_text: str, max_results: int) -> List[DocumentSearchResult]:
        """
        Search for similar documents using progressive hierarchical filtering.
        
        Args:
            query_text: Query text to search for
            max_results: Maximum number of results to return
            
        Returns:
            List of similar document search results
        """
        pass
    
    @abstractmethod
    def progressive_hierarchical_search(self, query_embedding: np.ndarray) -> List[int]:
        """
        Use multi-level hierarchical indices for progressive filtering.
        
        Args:
            query_embedding: Query embedding for similarity search
            
        Returns:
            List of candidate frame numbers after filtering
        """
        pass
    
    @abstractmethod
    def calculate_embedding_similarity(self, query_embedding: np.ndarray, 
                                     cached_frames: Dict[int, np.ndarray]) -> List[Tuple[int, float]]:
        """
        Calculate similarity scores using cached consecutive frames.
        
        Args:
            query_embedding: Query embedding
            cached_frames: Dictionary of cached frame embeddings
            
        Returns:
            List of (frame_number, similarity_score) tuples
        """
        pass
    
    @abstractmethod
    def compare_hierarchical_indices(self, query_indices: np.ndarray, 
                                   candidate_indices: np.ndarray) -> float:
        """
        Compare hierarchical indices for similarity scoring.
        
        Args:
            query_indices: Query hierarchical indices
            candidate_indices: Candidate hierarchical indices
            
        Returns:
            Similarity score between indices
        """
        pass


class FrameCacheManager(ABC):
    """Interface for intelligent frame caching system."""
    
    @abstractmethod
    def cache_consecutive_frames(self, target_frame: int, video_path: str, cache_size: int) -> Dict[int, np.ndarray]:
        """
        Cache consecutive frames leveraging hierarchical similarity ordering.
        
        Args:
            target_frame: Target frame number for caching
            video_path: Path to the video file
            cache_size: Number of frames to cache
            
        Returns:
            Dictionary of cached frames {frame_number: frame_data}
        """
        pass
    
    @abstractmethod
    def calculate_optimal_cache_size(self, similarity_threshold: float) -> int:
        """
        Calculate optimal cache size based on hierarchical similarity patterns.
        
        Args:
            similarity_threshold: Threshold for similarity-based caching
            
        Returns:
            Optimal cache size
        """
        pass
    
    @abstractmethod
    def invalidate_cache(self, frame_range: Tuple[int, int]) -> None:
        """
        Invalidate cache entries when frames are updated or reordered.
        
        Args:
            frame_range: Range of frame numbers to invalidate
        """
        pass
    
    @abstractmethod
    def get_cached_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """
        Retrieve frame from cache if available.
        
        Args:
            frame_number: Frame number to retrieve
            
        Returns:
            Cached frame data or None if not cached
        """
        pass
    
    @abstractmethod
    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.
        
        Returns:
            Dictionary containing cache statistics
        """
        pass


class DocumentRetrieval(ABC):
    """Interface for frame-based document retrieval system."""
    
    @abstractmethod
    def retrieve_documents_by_frame_numbers(self, frame_numbers: List[int]) -> List[Tuple[int, DocumentChunk]]:
        """
        Retrieve document chunks using frame numbers from similarity search.
        
        Args:
            frame_numbers: List of frame numbers from embedding similarity search
            
        Returns:
            List of (frame_number, document_chunk) tuples
        """
        pass
    
    @abstractmethod
    def retrieve_single_document(self, frame_number: int) -> Optional[DocumentChunk]:
        """
        Retrieve a single document chunk by frame number.
        
        Args:
            frame_number: Frame number from embedding similarity search
            
        Returns:
            Document chunk if found, None otherwise
        """
        pass
    
    @abstractmethod
    def validate_retrieval_synchronization(self, frame_numbers: List[int]) -> Dict[str, Any]:
        """
        Validate embedding-document frame synchronization for retrieval.
        
        Args:
            frame_numbers: Frame numbers to validate
            
        Returns:
            Validation results dictionary
        """
        pass
    
    @abstractmethod
    def get_retrieval_statistics(self, frame_numbers: List[int]) -> Dict[str, Any]:
        """
        Get statistics about document retrieval operation.
        
        Args:
            frame_numbers: Frame numbers used for retrieval
            
        Returns:
            Statistics dictionary
        """
        pass


class EmbeddingCompressor(ABC):
    """Interface for embedding compression with hierarchical index preservation."""
    
    @abstractmethod
    def compress_embedding_frame(self, embedding_frame: EmbeddingFrame, quality: float) -> bytes:
        """
        Compress embedding frame while preserving hierarchical index integrity.
        
        Args:
            embedding_frame: Embedding frame with hierarchical indices
            quality: Compression quality (0.0 to 1.0)
            
        Returns:
            Compressed embedding data as bytes
        """
        pass
    
    @abstractmethod
    def decompress_embedding_frame(self, compressed_data: bytes) -> EmbeddingFrame:
        """
        Decompress embedding frame and validate hierarchical index integrity.
        
        Args:
            compressed_data: Compressed embedding frame data
            
        Returns:
            Reconstructed embedding frame with validated indices
        """
        pass
    
    @abstractmethod
    def validate_index_preservation(self, original_frame: EmbeddingFrame, 
                                  reconstructed_frame: EmbeddingFrame, 
                                  tolerance: float = 1e-3) -> bool:
        """
        Validate that hierarchical indices are preserved during compression.
        
        Args:
            original_frame: Original embedding frame
            reconstructed_frame: Reconstructed embedding frame
            tolerance: Acceptable difference tolerance
            
        Returns:
            True if indices are preserved within tolerance
        """
        pass
    
    @abstractmethod
    def get_compression_metrics(self, original_frame: EmbeddingFrame, 
                              reconstructed_frame: EmbeddingFrame,
                              compressed_size: int) -> Dict[str, Any]:
        """
        Calculate comprehensive compression metrics for embedding frames.
        
        Args:
            original_frame: Original embedding frame
            reconstructed_frame: Reconstructed embedding frame
            compressed_size: Size of compressed data in bytes
            
        Returns:
            Dictionary containing compression metrics
        """
        pass
    
    @abstractmethod
    def configure_quality_settings(self, embedding_quality: float, index_quality: float) -> None:
        """
        Configure separate quality settings for embeddings and hierarchical indices.
        
        Args:
            embedding_quality: Quality setting for embedding data (0.0 to 1.0)
            index_quality: Quality setting for hierarchical indices (0.0 to 1.0)
        """
        pass


class EmbeddingReconstructor(ABC):
    """Interface for complete embedding reconstruction pipeline."""
    
    @abstractmethod
    def reconstruct_from_compressed_frame(self, compressed_data: bytes) -> np.ndarray:
        """
        Complete reconstruction workflow from compressed embedding frame.
        
        Args:
            compressed_data: Compressed embedding frame data
            
        Returns:
            Reconstructed 1D embedding array
        """
        pass
    
    @abstractmethod
    def extract_hierarchical_indices(self, embedding_frame: EmbeddingFrame) -> List[np.ndarray]:
        """
        Extract hierarchical indices from embedding frame.
        
        Args:
            embedding_frame: Embedding frame with indices
            
        Returns:
            List of hierarchical index arrays
        """
        pass
    
    @abstractmethod
    def apply_inverse_hilbert_mapping(self, embedding_image: np.ndarray, 
                                    original_dimensions: int) -> np.ndarray:
        """
        Apply inverse Hilbert mapping to reconstruct 1D embedding.
        
        Args:
            embedding_image: 2D embedding representation
            original_dimensions: Original embedding dimensions
            
        Returns:
            Reconstructed 1D embedding array
        """
        pass
    
    @abstractmethod
    def validate_reconstruction_accuracy(self, original_embedding: np.ndarray,
                                       reconstructed_embedding: np.ndarray,
                                       tolerance: float = 1e-3) -> Dict[str, Any]:
        """
        Validate reconstruction accuracy and embedding dimension consistency.
        
        Args:
            original_embedding: Original 1D embedding
            reconstructed_embedding: Reconstructed 1D embedding
            tolerance: Acceptable difference tolerance
            
        Returns:
            Validation results dictionary
        """
        pass
    
    @abstractmethod
    def get_reconstruction_metrics(self, original_embedding: np.ndarray,
                                 reconstructed_embedding: np.ndarray) -> Dict[str, Any]:
        """
        Calculate comprehensive reconstruction metrics.
        
        Args:
            original_embedding: Original 1D embedding
            reconstructed_embedding: Reconstructed 1D embedding
            
        Returns:
            Dictionary containing reconstruction metrics
        """
        pass