"""
Test basic RAG system structure and imports.
"""

import pytest
import numpy as np
from hilbert_quantization.rag.config import RAGConfig
from hilbert_quantization.rag.models import DocumentChunk, EmbeddingFrame, ProcessingProgress


def test_rag_config_creation():
    """Test RAG configuration creation and validation."""
    config = RAGConfig()
    
    # Test default values
    assert config.embedding.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert config.video.codec == "libx264"
    assert config.chunking.chunk_overlap == 50
    assert config.search.max_results == 10
    
    # Test configuration validation
    warnings = config.validate_compatibility()
    assert isinstance(warnings, list)


def test_rag_config_from_dict():
    """Test RAG configuration creation from dictionary."""
    config_dict = {
        'embedding': {'model_name': 'custom-model', 'batch_size': 64},
        'video': {'quality': 0.9, 'frame_rate': 60.0},
        'search': {'max_results': 20}
    }
    
    config = RAGConfig.from_dict(config_dict)
    
    assert config.embedding.model_name == 'custom-model'
    assert config.embedding.batch_size == 64
    assert config.video.quality == 0.9
    assert config.video.frame_rate == 60.0
    assert config.search.max_results == 20


def test_document_chunk_model():
    """Test DocumentChunk data model."""
    chunk = DocumentChunk(
        content="This is a test document chunk.",
        ipfs_hash="QmTest123",
        source_path="/path/to/document.txt",
        start_position=0,
        end_position=30,
        chunk_sequence=1,
        creation_timestamp="2024-01-01T00:00:00Z",
        chunk_size=30
    )
    
    assert chunk.content == "This is a test document chunk."
    assert chunk.validate_size(30) == True
    assert chunk.validate_size(25) == False


def test_embedding_frame_model():
    """Test EmbeddingFrame data model."""
    embedding_data = np.random.rand(32, 32).astype(np.float32)
    hierarchical_indices = [np.random.rand(10), np.random.rand(5)]
    
    frame = EmbeddingFrame(
        embedding_data=embedding_data,
        hierarchical_indices=hierarchical_indices,
        original_embedding_dimensions=1024,
        hilbert_dimensions=(32, 32),
        compression_quality=0.8,
        frame_number=1
    )
    
    assert frame.embedding_data.shape == (32, 32)
    assert len(frame.hierarchical_indices) == 2
    assert frame.original_embedding_dimensions == 1024
    assert frame.hilbert_dimensions == (32, 32)


def test_processing_progress_model():
    """Test ProcessingProgress data model."""
    progress = ProcessingProgress(
        total_documents=100,
        processed_documents=25,
        current_document="document_25.txt",
        chunks_created=250,
        embeddings_generated=250,
        processing_time=120.5
    )
    
    assert progress.progress_percent == 25.0
    assert progress.total_documents == 100
    assert progress.processed_documents == 25


def test_rag_imports():
    """Test that RAG module imports work correctly."""
    from hilbert_quantization import rag
    from hilbert_quantization.rag import RAGConfig, DocumentChunk, EmbeddingFrame
    from hilbert_quantization.rag.interfaces import DocumentChunker, EmbeddingGenerator
    
    # Test that classes are available
    assert RAGConfig is not None
    assert DocumentChunk is not None
    assert EmbeddingFrame is not None
    assert DocumentChunker is not None
    assert EmbeddingGenerator is not None


def test_config_validation_errors():
    """Test configuration validation with invalid values."""
    # Test invalid embedding config
    with pytest.raises(ValueError):
        from hilbert_quantization.rag.config import EmbeddingConfig
        EmbeddingConfig(batch_size=-1)
    
    # Test invalid video config
    with pytest.raises(ValueError):
        from hilbert_quantization.rag.config import VideoConfig
        VideoConfig(quality=1.5)
    
    # Test invalid search config
    with pytest.raises(ValueError):
        from hilbert_quantization.rag.config import SearchConfig
        SearchConfig(embedding_weight=0.8, hierarchical_weight=0.3)  # Sum > 1.0


def test_model_validation_errors():
    """Test model validation with invalid values."""
    # Test invalid DocumentChunk
    with pytest.raises(ValueError):
        DocumentChunk(
            content="test",
            ipfs_hash="hash",
            source_path="path",
            start_position=10,
            end_position=5,  # Invalid: end < start
            chunk_sequence=1,
            creation_timestamp="2024-01-01T00:00:00Z",
            chunk_size=4
        )
    
    # Test invalid EmbeddingFrame
    with pytest.raises(ValueError):
        EmbeddingFrame(
            embedding_data=np.random.rand(32, 32),
            hierarchical_indices=[],
            original_embedding_dimensions=-1,  # Invalid: negative
            hilbert_dimensions=(32, 32),
            compression_quality=0.8,
            frame_number=1
        )