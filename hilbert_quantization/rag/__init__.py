"""
RAG (Retrieval-Augmented Generation) system using Hilbert curve spatial mapping.

This module implements a novel approach to document storage and retrieval using
Hilbert curve spatial mapping for embeddings with dual-video architecture.
"""

from .interfaces import (
    DocumentChunker,
    EmbeddingGenerator,
    MultiLevelHierarchicalIndexGenerator,
    DualVideoStorage,
    RAGSearchEngine,
    FrameCacheManager
)

from .models import (
    DocumentChunk,
    EmbeddingFrame,
    VideoFrameMetadata,
    DualVideoStorageMetadata,
    DocumentSearchResult,
    ProcessingProgress
)

from .config import RAGConfig

__all__ = [
    'DocumentChunker',
    'EmbeddingGenerator', 
    'MultiLevelHierarchicalIndexGenerator',
    'DualVideoStorage',
    'RAGSearchEngine',
    'FrameCacheManager',
    'DocumentChunk',
    'EmbeddingFrame',
    'VideoFrameMetadata',
    'DualVideoStorageMetadata',
    'DocumentSearchResult',
    'ProcessingProgress',
    'RAGConfig'
]