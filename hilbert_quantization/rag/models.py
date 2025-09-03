"""
Core data models for the RAG system with Hilbert curve embedding storage.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import numpy as np


@dataclass
class DocumentChunk:
    """Standardized document chunk with comprehensive metadata."""
    content: str
    ipfs_hash: str
    source_path: str
    start_position: int
    end_position: int
    chunk_sequence: int
    creation_timestamp: str
    chunk_size: int
    
    def validate_size(self, target_size: int) -> bool:
        """Validate chunk meets exact size requirements."""
        return len(self.content) == target_size
    
    def __post_init__(self):
        """Validate document chunk data."""
        if self.chunk_size <= 0:
            raise ValueError("Chunk size must be positive")
        if self.start_position < 0 or self.end_position < 0:
            raise ValueError("Positions must be non-negative")
        if self.start_position >= self.end_position:
            raise ValueError("Start position must be less than end position")
        if self.chunk_sequence < 0:
            raise ValueError("Chunk sequence must be non-negative")


@dataclass
class EmbeddingFrame:
    """Embedding frame with hierarchical indices for video storage."""
    embedding_data: np.ndarray
    hierarchical_indices: List[np.ndarray]
    original_embedding_dimensions: int
    hilbert_dimensions: Tuple[int, int]
    compression_quality: float
    frame_number: int
    
    def __post_init__(self):
        """Validate embedding frame data."""
        if self.original_embedding_dimensions <= 0:
            raise ValueError("Original embedding dimensions must be positive")
        if len(self.hilbert_dimensions) != 2:
            raise ValueError("Hilbert dimensions must be a 2-tuple")
        if self.compression_quality < 0 or self.compression_quality > 1:
            raise ValueError("Compression quality must be between 0 and 1")
        if self.frame_number < 0:
            raise ValueError("Frame number must be non-negative")
        if self.embedding_data.ndim != 2:
            raise ValueError("Embedding data must be 2-dimensional")


@dataclass
class VideoFrameMetadata:
    """Metadata for synchronized video frames."""
    frame_index: int
    chunk_id: str
    ipfs_hash: str
    source_document: str
    compression_quality: float
    hierarchical_indices: List[np.ndarray]
    embedding_model: str
    frame_timestamp: float
    chunk_metadata: DocumentChunk
    
    def __post_init__(self):
        """Validate video frame metadata."""
        if self.frame_index < 0:
            raise ValueError("Frame index must be non-negative")
        if self.compression_quality < 0 or self.compression_quality > 1:
            raise ValueError("Compression quality must be between 0 and 1")
        if self.frame_timestamp < 0:
            raise ValueError("Frame timestamp must be non-negative")


@dataclass
class DualVideoStorageMetadata:
    """Metadata for dual-video storage system."""
    embedding_video_path: str
    document_video_path: str
    total_frames: int
    frame_rate: float
    video_codec: str
    frame_dimensions: Tuple[int, int]
    creation_timestamp: str
    total_documents_stored: int
    average_compression_ratio: float
    frame_metadata: List[VideoFrameMetadata]
    
    def __post_init__(self):
        """Validate dual-video storage metadata."""
        if self.total_frames < 0:
            raise ValueError("Total frames must be non-negative")
        if self.frame_rate <= 0:
            raise ValueError("Frame rate must be positive")
        if len(self.frame_dimensions) != 2:
            raise ValueError("Frame dimensions must be a 2-tuple")
        if self.total_documents_stored < 0:
            raise ValueError("Total documents stored must be non-negative")
        if self.average_compression_ratio <= 0:
            raise ValueError("Average compression ratio must be positive")


@dataclass
class DocumentSearchResult:
    """Result from RAG document similarity search."""
    document_chunk: DocumentChunk
    similarity_score: float
    embedding_similarity_score: float
    hierarchical_similarity_score: float
    frame_number: int
    search_method: str
    cached_neighbors: Optional[List[int]] = None
    
    def __post_init__(self):
        """Validate document search result."""
        if not (0 <= self.similarity_score <= 1):
            raise ValueError("Similarity score must be between 0 and 1")
        if not (0 <= self.embedding_similarity_score <= 1):
            raise ValueError("Embedding similarity score must be between 0 and 1")
        if not (0 <= self.hierarchical_similarity_score <= 1):
            raise ValueError("Hierarchical similarity score must be between 0 and 1")
        if self.frame_number < 0:
            raise ValueError("Frame number must be non-negative")


@dataclass
class ProcessingProgress:
    """Progress tracking for document processing operations."""
    total_documents: int
    processed_documents: int
    current_document: str
    chunks_created: int
    embeddings_generated: int
    processing_time: float
    
    @property
    def progress_percent(self) -> float:
        """Calculate progress percentage."""
        if self.total_documents == 0:
            return 0.0
        return (self.processed_documents / self.total_documents) * 100
    
    def __post_init__(self):
        """Validate processing progress data."""
        if self.total_documents < 0:
            raise ValueError("Total documents must be non-negative")
        if self.processed_documents < 0:
            raise ValueError("Processed documents must be non-negative")
        if self.processed_documents > self.total_documents:
            raise ValueError("Processed documents cannot exceed total documents")
        if self.chunks_created < 0:
            raise ValueError("Chunks created must be non-negative")
        if self.embeddings_generated < 0:
            raise ValueError("Embeddings generated must be non-negative")
        if self.processing_time < 0:
            raise ValueError("Processing time must be non-negative")


@dataclass
class RAGMetrics:
    """Performance metrics for RAG system operations."""
    document_processing_time: float
    embedding_generation_time: float
    video_compression_time: float
    search_time: float
    total_documents_processed: int
    total_chunks_created: int
    average_chunk_size: float
    compression_ratio: float
    search_accuracy: float
    memory_usage_mb: float
    
    def __post_init__(self):
        """Validate RAG metrics."""
        if self.document_processing_time < 0:
            raise ValueError("Document processing time must be non-negative")
        if self.embedding_generation_time < 0:
            raise ValueError("Embedding generation time must be non-negative")
        if self.video_compression_time < 0:
            raise ValueError("Video compression time must be non-negative")
        if self.search_time < 0:
            raise ValueError("Search time must be non-negative")
        if self.total_documents_processed < 0:
            raise ValueError("Total documents processed must be non-negative")
        if self.total_chunks_created < 0:
            raise ValueError("Total chunks created must be non-negative")
        if self.average_chunk_size < 0:
            raise ValueError("Average chunk size must be non-negative")
        if self.compression_ratio <= 0:
            raise ValueError("Compression ratio must be positive")
        if not (0 <= self.search_accuracy <= 1):
            raise ValueError("Search accuracy must be between 0 and 1")
        if self.memory_usage_mb < 0:
            raise ValueError("Memory usage must be non-negative")