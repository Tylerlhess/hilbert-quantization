"""
Tests for the embedding generator implementation.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from hilbert_quantization.rag.embedding_generation.generator import EmbeddingGeneratorImpl
from hilbert_quantization.rag.config import RAGConfig, EmbeddingConfig
from hilbert_quantization.rag.models import DocumentChunk


class TestEmbeddingGeneratorImpl:
    """Test cases for EmbeddingGeneratorImpl."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = RAGConfig()
        self.generator = EmbeddingGeneratorImpl(self.config)
        
        # Create sample document chunks
        self.sample_chunks = [
            DocumentChunk(
                content="This is the first test document chunk.",
                ipfs_hash="QmTest1",
                source_path="/test/doc1.txt",
                start_position=0,
                end_position=40,
                chunk_sequence=0,
                creation_timestamp="2024-01-01T00:00:00Z",
                chunk_size=40
            ),
            DocumentChunk(
                content="This is the second test document chunk.",
                ipfs_hash="QmTest2", 
                source_path="/test/doc2.txt",
                start_position=0,
                end_position=41,
                chunk_sequence=1,
                creation_timestamp="2024-01-01T00:00:01Z",
                chunk_size=41
            )
        ]
    
    def test_initialization(self):
        """Test generator initialization."""
        assert self.generator.config == self.config
        assert isinstance(self.generator._models_cache, dict)
        assert isinstance(self.generator._embedding_stats, dict)
    
    def test_get_supported_models(self):
        """Test getting supported models list."""
        models = self.generator.get_supported_models()
        assert isinstance(models, list)
        assert len(models) > 0
        assert "sentence-transformers/all-MiniLM-L6-v2" in models
        assert "tfidf" in models
    
    def test_get_model_info(self):
        """Test getting model information."""
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        info = self.generator.get_model_info(model_name)
        
        assert isinstance(info, dict)
        assert "type" in info
        assert "dimensions" in info
        assert "description" in info
        assert info["type"] == "sentence_transformer"
        assert info["dimensions"] == 384
    
    def test_get_model_info_unsupported(self):
        """Test getting info for unsupported model."""
        with pytest.raises(ValueError, match="Unsupported model"):
            self.generator.get_model_info("unsupported-model")
    
    def test_get_embedding_dimensions(self):
        """Test getting embedding dimensions."""
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        dimensions = self.generator.get_embedding_dimensions(model_name)
        assert dimensions == 384
    
    def test_calculate_optimal_dimensions(self):
        """Test calculating optimal dimensions for Hilbert mapping."""
        # Test various embedding sizes
        test_cases = [
            (100, (16, 16)),   # 16*16 = 256 > 100
            (384, (32, 32)),   # 32*32 = 1024 > 384
            (768, (32, 32)),   # 32*32 = 1024 > 768
            (1024, (32, 32)),  # 32*32 = 1024 = 1024
            (2000, (64, 64))   # 64*64 = 4096 > 2000
        ]
        
        for embedding_size, expected in test_cases:
            result = self.generator.calculate_optimal_dimensions(embedding_size)
            assert result == expected, f"For size {embedding_size}, expected {expected}, got {result}"
    
    def test_validate_embedding_consistency_empty(self):
        """Test validation with empty embeddings list."""
        assert self.generator.validate_embedding_consistency([]) is True
    
    def test_validate_embedding_consistency_valid(self):
        """Test validation with consistent embeddings."""
        embeddings = [
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0, 6.0]),
            np.array([7.0, 8.0, 9.0])
        ]
        assert self.generator.validate_embedding_consistency(embeddings) is True
    
    def test_validate_embedding_consistency_inconsistent_shape(self):
        """Test validation with inconsistent embedding shapes."""
        embeddings = [
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0]),  # Different shape
            np.array([7.0, 8.0, 9.0])
        ]
        assert self.generator.validate_embedding_consistency(embeddings) is False
    
    def test_validate_embedding_consistency_nan_values(self):
        """Test validation with NaN values."""
        embeddings = [
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, np.nan, 6.0]),  # Contains NaN
            np.array([7.0, 8.0, 9.0])
        ]
        assert self.generator.validate_embedding_consistency(embeddings) is False
    
    def test_validate_embedding_consistency_infinite_values(self):
        """Test validation with infinite values."""
        embeddings = [
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, np.inf, 6.0]),  # Contains infinity
            np.array([7.0, 8.0, 9.0])
        ]
        assert self.generator.validate_embedding_consistency(embeddings) is False
    
    @patch('sklearn.feature_extraction.text.TfidfVectorizer')
    def test_generate_sklearn_embeddings(self, mock_vectorizer_class):
        """Test generating embeddings with sklearn TF-IDF."""
        # Mock the TfidfVectorizer
        mock_vectorizer = Mock()
        mock_vectorizer.fit.return_value = None
        mock_vectorizer.transform.return_value = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        mock_vectorizer_class.return_value = mock_vectorizer
        
        # Load the model (which creates the vectorizer)
        model = self.generator._load_model("tfidf")
        
        # Generate embeddings
        texts = ["test text 1", "test text 2"]
        embeddings = self.generator._generate_sklearn_embeddings(model, texts, "tfidf")
        
        assert len(embeddings) == 2
        assert np.array_equal(embeddings[0], np.array([1.0, 2.0, 3.0]))
        assert np.array_equal(embeddings[1], np.array([4.0, 5.0, 6.0]))
    
    def test_generate_embeddings_unsupported_model(self):
        """Test generating embeddings with unsupported model."""
        with pytest.raises(ValueError, match="Unsupported model"):
            self.generator.generate_embeddings(self.sample_chunks, "unsupported-model")
    
    def test_generate_embeddings_empty_chunks(self):
        """Test generating embeddings with empty chunks list."""
        result = self.generator.generate_embeddings([], "tfidf")
        assert result == []
    
    @patch('sklearn.feature_extraction.text.TfidfVectorizer')
    def test_generate_embeddings_sklearn_success(self, mock_vectorizer_class):
        """Test successful embedding generation with sklearn."""
        # Mock the TfidfVectorizer
        mock_vectorizer = Mock()
        mock_vectorizer.fit.return_value = None
        mock_vectorizer.transform.return_value = np.array([[1.0, 2.0], [3.0, 4.0]])
        mock_vectorizer_class.return_value = mock_vectorizer
        
        embeddings = self.generator.generate_embeddings(self.sample_chunks, "tfidf")
        
        assert len(embeddings) == 2
        assert isinstance(embeddings[0], np.ndarray)
        assert isinstance(embeddings[1], np.ndarray)
    
    def test_get_device_auto_cpu(self):
        """Test device selection when CUDA is not available."""
        device = self.generator._get_device()
        # Should return 'cpu' when CUDA is not available or transformers not installed
        assert device in ['cpu', 'cuda']
    
    def test_get_device_explicit(self):
        """Test explicit device configuration."""
        self.config.embedding.device = "cpu"
        device = self.generator._get_device()
        assert device == "cpu"
    
    def test_clear_model_cache(self):
        """Test clearing model cache."""
        # Add something to cache
        self.generator._models_cache["test"] = "test_model"
        assert len(self.generator._models_cache) > 0
        
        # Clear cache
        self.generator.clear_model_cache()
        assert len(self.generator._models_cache) == 0
    
    def test_get_embedding_stats_empty(self):
        """Test getting embedding stats when no embeddings generated."""
        stats = self.generator.get_embedding_stats()
        assert isinstance(stats, dict)
        assert len(stats) == 0
    
    def test_update_embedding_stats(self):
        """Test updating embedding statistics."""
        model_name = "test_model"
        self.generator._update_embedding_stats(model_name, 10, 2.5, 384)
        
        stats = self.generator.get_embedding_stats()
        assert model_name in stats
        assert stats[model_name]["total_chunks"] == 10
        assert stats[model_name]["total_time"] == 2.5
        assert stats[model_name]["embedding_dimension"] == 384
        assert stats[model_name]["generation_count"] == 1
        assert stats[model_name]["avg_time_per_chunk"] == 0.25
        assert stats[model_name]["avg_chunks_per_second"] == 4.0
    
    def test_validate_model_availability_sklearn(self):
        """Test model availability validation for sklearn models."""
        # TF-IDF should be available if sklearn is installed
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            expected = True
        except ImportError:
            expected = False
        
        result = self.generator.validate_model_availability("tfidf")
        assert result == expected
    
    def test_validate_model_availability_unsupported(self):
        """Test model availability for unsupported model."""
        result = self.generator.validate_model_availability("unsupported-model")
        assert result is False
    
    def test_validate_config_invalid(self):
        """Test validation with invalid config."""
        with pytest.raises(ValueError, match="Config must be a RAGConfig instance"):
            EmbeddingGeneratorImpl("invalid_config")


class TestEmbeddingGeneratorIntegration:
    """Integration tests for embedding generator."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        self.config = RAGConfig()
        self.generator = EmbeddingGeneratorImpl(self.config)
        
        # Create more realistic document chunks
        self.document_chunks = [
            DocumentChunk(
                content="Machine learning is a subset of artificial intelligence that focuses on algorithms.",
                ipfs_hash="QmML1",
                source_path="/docs/ml_intro.txt",
                start_position=0,
                end_position=89,
                chunk_sequence=0,
                creation_timestamp="2024-01-01T00:00:00Z",
                chunk_size=89
            ),
            DocumentChunk(
                content="Deep learning uses neural networks with multiple layers to learn complex patterns.",
                ipfs_hash="QmDL1",
                source_path="/docs/dl_intro.txt", 
                start_position=0,
                end_position=84,
                chunk_sequence=1,
                creation_timestamp="2024-01-01T00:00:01Z",
                chunk_size=84
            )
        ]
    
    @patch('sklearn.feature_extraction.text.TfidfVectorizer')
    def test_end_to_end_sklearn_embeddings(self, mock_vectorizer_class):
        """Test end-to-end embedding generation with sklearn."""
        # Mock TfidfVectorizer to return predictable results
        mock_vectorizer = Mock()
        mock_vectorizer.fit.return_value = None
        # Return embeddings with expected dimensions
        mock_embeddings = np.random.rand(2, 1000)  # 2 documents, 1000 features
        mock_vectorizer.transform.return_value = mock_embeddings
        mock_vectorizer_class.return_value = mock_vectorizer
        
        # Generate embeddings
        embeddings = self.generator.generate_embeddings(self.document_chunks, "tfidf")
        
        # Validate results
        assert len(embeddings) == 2
        assert all(isinstance(emb, np.ndarray) for emb in embeddings)
        assert self.generator.validate_embedding_consistency(embeddings)
        
        # Check that statistics were updated
        stats = self.generator.get_embedding_stats()
        assert "tfidf" in stats
        assert stats["tfidf"]["total_chunks"] == 2
    
    def test_model_info_completeness(self):
        """Test that all supported models have complete information."""
        for model_name in self.generator.get_supported_models():
            info = self.generator.get_model_info(model_name)
            
            # Check required fields
            required_fields = ["type", "dimensions", "description"]
            for field in required_fields:
                assert field in info, f"Model {model_name} missing field: {field}"
            
            # Check dimensions are positive
            assert info["dimensions"] > 0, f"Model {model_name} has invalid dimensions"
            
            # Check type is valid
            valid_types = ["sentence_transformer", "huggingface_transformer", "sklearn"]
            assert info["type"] in valid_types, f"Model {model_name} has invalid type: {info['type']}"