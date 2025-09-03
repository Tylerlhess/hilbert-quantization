"""
Tests for RAG system configuration management.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from hilbert_quantization.rag.config import (
    EmbeddingConfig, VideoConfig, ChunkingConfig, HilbertConfig,
    IndexConfig, SearchConfig, StorageConfig, ProcessingConfig,
    RAGConfig, RAGConfigurationManager,
    create_default_rag_config, create_high_performance_rag_config,
    create_high_quality_rag_config, validate_embedding_model_compatibility
)


class TestEmbeddingConfig:
    """Test EmbeddingConfig validation and functionality."""
    
    def test_default_config_creation(self):
        """Test creating default embedding configuration."""
        config = EmbeddingConfig()
        assert config.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.batch_size == 32
        assert config.max_sequence_length == 512
        assert config.normalize_embeddings is True
        assert config.device == "auto"
        assert config.trust_remote_code is False
        assert len(config.supported_models) > 0
    
    def test_valid_parameters(self):
        """Test valid parameter validation."""
        config = EmbeddingConfig(
            batch_size=64,
            max_sequence_length=256,
            device="cuda"
        )
        assert config.batch_size == 64
        assert config.max_sequence_length == 256
        assert config.device == "cuda"
    
    def test_invalid_batch_size(self):
        """Test invalid batch size validation."""
        with pytest.raises(ValueError, match="Batch size must be positive"):
            EmbeddingConfig(batch_size=0)
        
        with pytest.raises(ValueError, match="Batch size must be positive"):
            EmbeddingConfig(batch_size=-5)
    
    def test_invalid_sequence_length(self):
        """Test invalid sequence length validation."""
        with pytest.raises(ValueError, match="Max sequence length must be positive"):
            EmbeddingConfig(max_sequence_length=0)
        
        with pytest.raises(ValueError, match="Max sequence length must be positive"):
            EmbeddingConfig(max_sequence_length=-100)
    
    def test_invalid_device(self):
        """Test invalid device validation."""
        with pytest.raises(ValueError, match="Device must be"):
            EmbeddingConfig(device="invalid")
    
    def test_model_compatibility_validation(self):
        """Test model compatibility validation."""
        config = EmbeddingConfig()
        
        # Test supported model
        assert config.validate_model_compatibility("sentence-transformers/all-MiniLM-L6-v2") is True
        
        # Test unsupported model (should still return True but log warning)
        with patch('logging.warning') as mock_warning:
            result = config.validate_model_compatibility("custom/unknown-model")
            assert result is True
            mock_warning.assert_called_once()
    
    def test_model_dimensions_retrieval(self):
        """Test model dimensions retrieval."""
        config = EmbeddingConfig()
        
        # Test known models
        assert config.get_model_dimensions("sentence-transformers/all-MiniLM-L6-v2") == 384
        assert config.get_model_dimensions("sentence-transformers/all-mpnet-base-v2") == 768
        assert config.get_model_dimensions("BAAI/bge-large-en-v1.5") == 1024
        
        # Test unknown model
        assert config.get_model_dimensions("unknown/model") is None


class TestVideoConfig:
    """Test VideoConfig validation and functionality."""
    
    def test_default_config_creation(self):
        """Test creating default video configuration."""
        config = VideoConfig()
        assert config.codec == "libx264"
        assert config.quality == 0.8
        assert config.frame_rate == 30.0
        assert config.max_frames_per_file == 10000
        assert config.compression_preset == "medium"
        assert config.adaptive_quality is False
        assert config.preserve_index_rows is True
    
    def test_quality_validation(self):
        """Test quality parameter validation."""
        # Valid quality
        config = VideoConfig(quality=0.5)
        assert config.quality == 0.5
        
        # Invalid quality
        with pytest.raises(ValueError, match="Quality must be between 0 and 1"):
            VideoConfig(quality=0.0)
        
        with pytest.raises(ValueError, match="Quality must be between 0 and 1"):
            VideoConfig(quality=1.5)
    
    def test_codec_validation(self):
        """Test codec validation."""
        # Valid codec
        config = VideoConfig(codec="libx265")
        assert config.codec == "libx265"
        
        # Invalid codec
        with pytest.raises(ValueError, match="Codec must be one of"):
            VideoConfig(codec="invalid_codec")
    
    def test_preset_validation(self):
        """Test compression preset validation."""
        # Valid preset
        config = VideoConfig(compression_preset="fast")
        assert config.compression_preset == "fast"
        
        # Invalid preset
        with pytest.raises(ValueError, match="Preset must be one of"):
            VideoConfig(compression_preset="invalid_preset")
    
    def test_quality_range_validation(self):
        """Test quality range validation."""
        # Valid range
        config = VideoConfig(quality=0.7, quality_range=(0.5, 0.9))
        assert config.quality_range == (0.5, 0.9)
        
        # Invalid range (min >= max)
        with pytest.raises(ValueError, match="Quality range must be valid"):
            VideoConfig(quality_range=(0.8, 0.5))
        
        # Quality outside range
        with pytest.raises(ValueError, match="Quality must be within quality range"):
            VideoConfig(quality=0.3, quality_range=(0.5, 0.9))
    
    def test_adaptive_quality_calculation(self):
        """Test adaptive quality calculation."""
        config = VideoConfig(
            quality=0.8,
            quality_range=(0.4, 0.9),
            adaptive_quality=True
        )
        
        # Test different compression ratios
        assert config.get_quality_for_size(100, 100) == 0.9  # No compression needed
        assert config.get_quality_for_size(50, 100) == 0.65  # 50% compression
        assert config.get_quality_for_size(10, 100) == 0.45  # Heavy compression
        
        # Test with adaptive quality disabled
        config.adaptive_quality = False
        assert config.get_quality_for_size(50, 100) == 0.8  # Returns fixed quality


class TestIndexConfig:
    """Test IndexConfig validation and functionality."""
    
    def test_default_config_creation(self):
        """Test creating default index configuration."""
        config = IndexConfig()
        assert config.max_index_levels == 5
        assert config.min_granularity == 2
        assert config.progressive_filtering is True
        assert config.auto_calculate_levels is True
        assert config.spatial_locality_weight == 0.7
        assert config.magnitude_weight == 0.3
    
    def test_granularity_validation(self):
        """Test granularity parameter validation."""
        # Valid granularity
        config = IndexConfig(min_granularity=4, max_granularity=32)
        assert config.min_granularity == 4
        assert config.max_granularity == 32
        
        # Invalid min granularity
        with pytest.raises(ValueError, match="Min granularity must be positive"):
            IndexConfig(min_granularity=0)
        
        # Invalid max granularity
        with pytest.raises(ValueError, match="Max granularity must be greater than min granularity"):
            IndexConfig(min_granularity=8, max_granularity=4)
    
    def test_weight_validation(self):
        """Test weight parameter validation."""
        # Valid weights
        config = IndexConfig(spatial_locality_weight=0.6, magnitude_weight=0.4)
        assert config.spatial_locality_weight == 0.6
        assert config.magnitude_weight == 0.4
        
        # Invalid weights (don't sum to 1.0)
        with pytest.raises(ValueError, match="weights must sum to 1.0"):
            IndexConfig(spatial_locality_weight=0.5, magnitude_weight=0.3)
        
        # Invalid weight values
        with pytest.raises(ValueError, match="weight must be between 0 and 1"):
            IndexConfig(spatial_locality_weight=1.5, magnitude_weight=-0.5)
    
    def test_custom_granularity_levels(self):
        """Test custom granularity levels validation."""
        # Valid levels (powers of 2)
        config = IndexConfig(granularity_levels=[32, 16, 8, 4])
        assert config.granularity_levels == [32, 16, 8, 4]  # Should be sorted
        
        # Invalid levels (not powers of 2)
        with pytest.raises(ValueError, match="must be powers of 2"):
            IndexConfig(granularity_levels=[32, 15, 8])
        
        # Invalid levels (not positive)
        with pytest.raises(ValueError, match="must be positive integers"):
            IndexConfig(granularity_levels=[32, 0, 8])
    
    def test_granularity_level_calculation(self):
        """Test automatic granularity level calculation."""
        config = IndexConfig(max_index_levels=4, min_granularity=2)
        
        # Test with different image sizes
        levels_1024 = config.calculate_granularity_levels(1024)
        assert len(levels_1024) <= 4
        assert all(level >= 2 for level in levels_1024)
        assert levels_1024 == sorted(levels_1024, reverse=True)
        
        levels_4096 = config.calculate_granularity_levels(4096)
        assert len(levels_4096) <= 4
        assert levels_4096[0] <= 64  # Capped at 64


class TestRAGConfig:
    """Test RAGConfig integration and functionality."""
    
    def test_default_config_creation(self):
        """Test creating default RAG configuration."""
        config = RAGConfig()
        assert isinstance(config.embedding, EmbeddingConfig)
        assert isinstance(config.video, VideoConfig)
        assert isinstance(config.chunking, ChunkingConfig)
        assert isinstance(config.hilbert, HilbertConfig)
        assert isinstance(config.index, IndexConfig)
        assert isinstance(config.search, SearchConfig)
        assert isinstance(config.storage, StorageConfig)
        assert isinstance(config.processing, ProcessingConfig)
    
    def test_config_serialization(self):
        """Test configuration serialization to/from dictionary."""
        config = RAGConfig()
        config.embedding.batch_size = 64
        config.video.quality = 0.9
        
        # Test to_dict
        config_dict = config.to_dict()
        assert config_dict["embedding"]["batch_size"] == 64
        assert config_dict["video"]["quality"] == 0.9
        
        # Test from_dict
        restored_config = RAGConfig.from_dict(config_dict)
        assert restored_config.embedding.batch_size == 64
        assert restored_config.video.quality == 0.9
    
    def test_compatibility_validation(self):
        """Test configuration compatibility validation."""
        config = RAGConfig()
        
        # Set up configuration that should generate warnings
        config.chunking.chunk_size = 3000  # Exceeds max
        config.video.max_frames_per_file = 60000  # Very large
        config.search.cache_size = 2000  # Large cache
        
        warnings = config.validate_compatibility()
        assert len(warnings) >= 2
        assert any("chunk size" in warning.lower() for warning in warnings)
        assert any("memory" in warning.lower() for warning in warnings)


class TestRAGConfigurationManager:
    """Test RAGConfigurationManager functionality."""
    
    def test_manager_initialization(self):
        """Test configuration manager initialization."""
        # Default initialization
        manager = RAGConfigurationManager()
        assert isinstance(manager.config, RAGConfig)
        
        # Custom config initialization
        custom_config = RAGConfig()
        custom_config.embedding.batch_size = 128
        manager = RAGConfigurationManager(custom_config)
        assert manager.config.embedding.batch_size == 128
    
    def test_embedding_config_updates(self):
        """Test embedding configuration updates."""
        manager = RAGConfigurationManager()
        
        # Valid update
        manager.update_embedding_config(batch_size=64, device="cuda")
        assert manager.config.embedding.batch_size == 64
        assert manager.config.embedding.device == "cuda"
        
        # Invalid parameter
        with pytest.raises(ValueError, match="Unknown embedding parameter"):
            manager.update_embedding_config(invalid_param=123)
        
        # Invalid value
        with pytest.raises(ValueError):
            manager.update_embedding_config(batch_size=-5)
    
    def test_video_config_updates(self):
        """Test video configuration updates."""
        manager = RAGConfigurationManager()
        
        # Valid update
        manager.update_video_config(quality=0.9, codec="libx265")
        assert manager.config.video.quality == 0.9
        assert manager.config.video.codec == "libx265"
        
        # Invalid parameter
        with pytest.raises(ValueError, match="Unknown video parameter"):
            manager.update_video_config(invalid_param=123)
    
    def test_model_optimization(self):
        """Test configuration optimization for specific models."""
        manager = RAGConfigurationManager()
        
        # Optimize for small model
        manager.optimize_for_model("sentence-transformers/all-MiniLM-L6-v2")
        assert manager.config.embedding.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert manager.config.embedding.batch_size == 64  # Should be optimized for small model
        
        # Optimize for large model
        manager.optimize_for_model("BAAI/bge-large-en-v1.5")
        assert manager.config.embedding.model_name == "BAAI/bge-large-en-v1.5"
        assert manager.config.embedding.batch_size == 16  # Should be optimized for large model
    
    def test_dataset_size_optimization(self):
        """Test configuration optimization for different dataset sizes."""
        manager = RAGConfigurationManager()
        
        # Small dataset
        small_config = manager.get_optimal_config_for_dataset_size(500)
        assert small_config.video.quality == 0.95
        assert small_config.video.adaptive_quality is False
        assert small_config.index.max_index_levels == 3
        
        # Medium dataset
        medium_config = manager.get_optimal_config_for_dataset_size(5000)
        assert medium_config.video.quality == 0.8
        assert medium_config.video.adaptive_quality is True
        assert medium_config.index.max_index_levels == 5
        
        # Large dataset
        large_config = manager.get_optimal_config_for_dataset_size(50000)
        assert large_config.video.quality == 0.7
        assert large_config.video.adaptive_quality is True
        assert large_config.index.max_index_levels == 7
    
    def test_configuration_validation(self):
        """Test comprehensive configuration validation."""
        manager = RAGConfigurationManager()
        
        # Set up problematic configuration
        manager.config.video.quality = 0.95
        manager.config.video.adaptive_quality = True
        manager.config.video.memory_limit_mb = 3000
        manager.config.processing.memory_limit_mb = 2000
        manager.config.index.max_index_levels = 10
        
        warnings = manager.validate_configuration()
        assert len(warnings) >= 3
        assert any("adaptive quality" in warning.lower() for warning in warnings)
        assert any("memory" in warning.lower() for warning in warnings)
        assert any("index levels" in warning.lower() for warning in warnings)
    
    def test_config_file_operations(self):
        """Test configuration file save/load operations."""
        manager = RAGConfigurationManager()
        manager.config.embedding.batch_size = 128
        manager.config.video.quality = 0.85
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Test save
            manager.save_config(temp_path)
            assert Path(temp_path).exists()
            
            # Test load
            new_manager = RAGConfigurationManager()
            new_manager.load_config(temp_path)
            assert new_manager.config.embedding.batch_size == 128
            assert new_manager.config.video.quality == 0.85
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_config_template_export(self):
        """Test configuration template export."""
        manager = RAGConfigurationManager()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            manager.export_config_template(temp_path)
            assert Path(temp_path).exists()
            
            # Verify template content
            with open(temp_path, 'r') as f:
                template = json.load(f)
            
            assert "_description" in template
            assert "embedding" in template
            assert "video" in template
            assert "index" in template
            assert "search" in template
            assert "_supported_models" in template["embedding"]
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_config_backup_restore(self):
        """Test configuration backup and restore."""
        manager = RAGConfigurationManager()
        
        # Modify configuration
        original_batch_size = manager.config.embedding.batch_size
        manager.update_embedding_config(batch_size=128)
        
        # Modify again
        manager.update_embedding_config(batch_size=256)
        assert manager.config.embedding.batch_size == 256
        
        # Restore previous
        restored = manager.restore_previous_config()
        assert restored is True
        assert manager.config.embedding.batch_size == 128
        
        # Restore again
        restored = manager.restore_previous_config()
        assert restored is True
        assert manager.config.embedding.batch_size == original_batch_size
        
        # Try to restore when no more backups
        restored = manager.restore_previous_config()
        assert restored is False


class TestConfigurationUtilities:
    """Test configuration utility functions."""
    
    def test_create_default_rag_config(self):
        """Test creating default RAG configuration."""
        config = create_default_rag_config()
        assert isinstance(config, RAGConfig)
        assert config.embedding.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.video.quality == 0.8
    
    def test_create_high_performance_rag_config(self):
        """Test creating high-performance RAG configuration."""
        config = create_high_performance_rag_config()
        assert config.video.adaptive_quality is True
        assert config.video.compression_threads == 8
        assert config.processing.max_workers == 8
        assert config.processing.batch_size == 200
        assert config.search.cache_size == 200
    
    def test_create_high_quality_rag_config(self):
        """Test creating high-quality RAG configuration."""
        config = create_high_quality_rag_config()
        assert config.video.quality == 0.95
        assert config.video.preserve_index_rows is True
        assert config.embedding.normalize_embeddings is True
        assert config.search.similarity_threshold == 0.8
        assert config.index.spatial_locality_weight == 0.8
    
    def test_validate_embedding_model_compatibility(self):
        """Test embedding model compatibility validation."""
        config = RAGConfig()
        config.hilbert.target_dimensions = (32, 32)  # 1024 total
        
        # Test compatible model
        issues = validate_embedding_model_compatibility(
            "sentence-transformers/all-MiniLM-L6-v2", config
        )
        assert len(issues) == 0
        
        # Test model with dimensions too large for Hilbert space
        config.hilbert.target_dimensions = (16, 16)  # 256 total
        issues = validate_embedding_model_compatibility(
            "sentence-transformers/all-mpnet-base-v2", config  # 768 dimensions
        )
        assert len(issues) > 0
        assert any("too small" in issue for issue in issues)
        
        # Test large model with large batch size
        config.embedding.batch_size = 64
        issues = validate_embedding_model_compatibility(
            "BAAI/bge-large-en-v1.5", config  # 1024 dimensions
        )
        assert any("batch size" in issue for issue in issues)


class TestConfigurationEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_invalid_config_file_loading(self):
        """Test loading invalid configuration files."""
        manager = RAGConfigurationManager()
        
        # Test non-existent file
        with pytest.raises(FileNotFoundError):
            manager.load_config("non_existent_file.json")
        
        # Test invalid JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            temp_path = f.name
        
        try:
            with pytest.raises(json.JSONDecodeError):
                manager.load_config(temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_extreme_configuration_values(self):
        """Test handling of extreme configuration values."""
        # Test very large batch sizes
        config = EmbeddingConfig(batch_size=10000)
        assert config.batch_size == 10000
        
        # Test very small quality values (within valid range)
        config = VideoConfig(quality=0.15, quality_range=(0.1, 1.0))
        assert config.quality == 0.15
        
        # Test many index levels
        config = IndexConfig(max_index_levels=20)
        assert config.max_index_levels == 20
    
    def test_configuration_with_none_values(self):
        """Test configuration handling of None values."""
        config = RAGConfig()
        config.chunking.chunk_size = None
        config.hilbert.target_dimensions = None
        config.index.max_granularity = None
        
        # Should not raise errors
        warnings = config.validate_compatibility()
        assert isinstance(warnings, list)
    
    def test_memory_limit_consistency(self):
        """Test memory limit consistency validation."""
        manager = RAGConfigurationManager()
        manager.config.video.memory_limit_mb = 2000
        manager.config.processing.memory_limit_mb = 3000
        
        warnings = manager.validate_configuration()
        assert any("memory" in warning.lower() for warning in warnings)